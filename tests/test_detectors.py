"""
tests/test_detectors.py
-----------------------
One synthetic failure injection test per detector.

Run with:  python -m unittest tests.test_detectors -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression

from core.ingestion import load_model_input
from core.report import Severity
import detectors.overfitting        as overfitting_det
import detectors.class_imbalance    as imbalance_det
import detectors.data_leakage       as leakage_det
import detectors.data_quality       as quality_det
import detectors.distribution_shift as drift_det
import detectors.metric_mismatch    as metric_det
from detectors import run_all


# ── Shared helpers ────────────────────────────────────────────────────────────

def _make_base_clf(n_samples=400, n_features=6, random_state=42):
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features,
        n_informative=4, random_state=random_state
    )
    cols = [f"feat_{i}" for i in range(n_features)]
    X = pd.DataFrame(X, columns=cols)
    y = pd.Series(y, name="target")
    return X, y


def _split_and_fit(X, y, model=None, test_size=0.25, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    if model is None:
        model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


def _load(model, X_train, X_test, y_train, y_test,
          task="classification", domain="general"):
    return load_model_input(
        model, X_train, X_test, y_train, y_test,
        task_type=task, domain=domain,
    )


# ── Test 1: Overfitting detector ─────────────────────────────────────────────

class TestOverfittingDetector(unittest.TestCase):

    def test_detects_overfitting(self):
        """
        Inject overfitting: train a deep, unlimited decision tree on noisy labels.
        The tree memorises training data perfectly → large train-test gap.
        """
        X, y = _make_base_clf(n_samples=300)
        # Add label noise only to a copy we'll use as training labels
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=0
        )
        # Overfit by adding noise to TEST labels after training
        rng = np.random.default_rng(0)
        noisy_y_train = y_train.copy()
        flip_idx = rng.choice(len(noisy_y_train), size=len(noisy_y_train)//4, replace=False)
        noisy_y_train.iloc[flip_idx] = 1 - noisy_y_train.iloc[flip_idx]

        # Unlimited depth tree on noisy labels → will memorise noise
        model = DecisionTreeClassifier(max_depth=None, random_state=0)
        model.fit(X_train, noisy_y_train)

        mi = _load(model, X_train, X_test, noisy_y_train, y_test)
        finding = overfitting_det.detect(mi)

        self.assertIsNotNone(finding, "Overfitting should be detected")
        self.assertEqual(finding.id, "L1.1")
        self.assertIn(finding.severity,
                      [Severity.HIGH, Severity.CRITICAL, Severity.MEDIUM])
        self.assertIn("gap", finding.evidence)
        self.assertGreater(finding.evidence["gap"], 0.05)

    def test_no_false_positive_on_clean_model(self):
        """A well-generalising model should NOT trigger overfitting."""
        X, y = _make_base_clf(n_samples=500)
        model, X_train, X_test, y_train, y_test = _split_and_fit(
            X, y, LogisticRegression(C=1.0, max_iter=500)
        )
        mi = _load(model, X_train, X_test, y_train, y_test)
        finding = overfitting_det.detect(mi)
        # Gap should be small for logistic regression on clean data
        if finding is not None:
            self.assertLessEqual(finding.severity, Severity.MEDIUM)


# ── Test 2: Class imbalance detector ─────────────────────────────────────────

class TestClassImbalanceDetector(unittest.TestCase):

    def test_detects_severe_imbalance(self):
        """
        Inject 95:5 class imbalance.
        Minority ratio ≈ 0.05 → should trigger CRITICAL or HIGH.
        """
        n = 400
        y_vals  = np.array([0] * 380 + [1] * 20)   # 5% minority
        X_vals  = np.random.default_rng(1).standard_normal((n, 5))
        X = pd.DataFrame(X_vals, columns=[f"feat_{i}" for i in range(5)])
        y = pd.Series(y_vals, name="target")

        model, X_train, X_test, y_train, y_test = _split_and_fit(
            X, y, LogisticRegression(max_iter=500, class_weight="balanced")
        )
        mi = _load(model, X_train, X_test, y_train, y_test)
        finding = imbalance_det.detect(mi)

        self.assertIsNotNone(finding, "Imbalance should be detected")
        self.assertEqual(finding.id, "L0.1")
        self.assertIn(finding.severity, [Severity.CRITICAL, Severity.HIGH])
        self.assertIn("minority_ratio", finding.evidence)
        self.assertLess(finding.evidence["minority_ratio"], 0.10)

    def test_no_detection_on_balanced_data(self):
        """50:50 balanced classes should not trigger imbalance detector."""
        X, y = _make_base_clf(n_samples=400)
        model, X_train, X_test, y_train, y_test = _split_and_fit(X, y)
        mi = _load(model, X_train, X_test, y_train, y_test)
        finding = imbalance_det.detect(mi)
        self.assertIsNone(finding, "Balanced data should not trigger detector")

    def test_skipped_for_regression(self):
        """Imbalance detector must return None for regression tasks."""
        X, y = make_regression(n_samples=200, n_features=4, random_state=0)
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
        y = pd.Series(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = LinearRegression().fit(X_train, y_train)
        mi = _load(model, X_train, X_test, y_train, y_test, task="regression")
        self.assertIsNone(imbalance_det.detect(mi))


# ── Test 3: Data leakage detector ────────────────────────────────────────────

class TestDataLeakageDetector(unittest.TestCase):

    def test_detects_feature_target_leakage(self):
        """
        Inject leakage: add a feature that IS the target (perfect correlation).
        """
        X, y = _make_base_clf(n_samples=300)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=0
        )
        # Inject: add target as a feature → |correlation| = 1.0
        X_train_leaked = X_train.copy()
        X_test_leaked  = X_test.copy()
        X_train_leaked["leaked_target"] = y_train.values
        X_test_leaked["leaked_target"]  = y_test.values

        model = LogisticRegression(max_iter=500)
        model.fit(X_train_leaked, y_train)

        mi = _load(model, X_train_leaked, X_test_leaked, y_train, y_test)
        finding = leakage_det.detect(mi)

        self.assertIsNotNone(finding, "Leakage should be detected")
        self.assertEqual(finding.id, "L0.2")
        self.assertEqual(finding.severity, Severity.CRITICAL)
        self.assertIn("suspicious_features", finding.evidence)
        self.assertIn("leaked_target", finding.evidence["suspicious_features"])

    def test_no_false_positive_on_clean_features(self):
        """Clean features with moderate correlations should not flag leakage."""
        X, y = _make_base_clf(n_samples=400)
        model, X_train, X_test, y_train, y_test = _split_and_fit(X, y)
        mi = _load(model, X_train, X_test, y_train, y_test)
        finding = leakage_det.detect(mi)
        self.assertIsNone(finding)


# ── Test 4: Data quality detector ────────────────────────────────────────────

class TestDataQualityDetector(unittest.TestCase):

    def test_detects_null_values(self):
        """Inject 40% nulls into one column → CRITICAL null finding."""
        X, y = _make_base_clf(n_samples=300)
        model, X_train, X_test, y_train, y_test = _split_and_fit(X, y)

        X_train_dirty = X_train.copy()
        null_idx = X_train_dirty.sample(frac=0.40, random_state=0).index
        X_train_dirty.loc[null_idx, "feat_0"] = np.nan

        mi = _load(model, X_train_dirty, X_test, y_train, y_test)
        finding = quality_det.detect(mi)

        self.assertIsNotNone(finding)
        self.assertEqual(finding.id, "L0.3")
        self.assertIn("null_columns", finding.evidence)
        self.assertIn("feat_0", finding.evidence["null_columns"])
        self.assertEqual(finding.severity, Severity.CRITICAL)

    def test_detects_duplicate_rows(self):
        """Inject 20% duplicate rows → duplicate finding triggered."""
        X, y = _make_base_clf(n_samples=200)
        model, X_train, X_test, y_train, y_test = _split_and_fit(X, y)

        # Duplicate the first 30 rows (>10% of 150 train rows)
        n_dups = 30
        X_dup = pd.concat([X_train, X_train.iloc[:n_dups]], ignore_index=True)
        y_dup = pd.concat([y_train, y_train.iloc[:n_dups]], ignore_index=True)

        mi = _load(model, X_dup, X_test, y_dup, y_test)
        finding = quality_det.detect(mi)

        self.assertIsNotNone(finding)
        self.assertIn("duplicate_rows", finding.evidence)
        self.assertGreater(finding.evidence["duplicate_rows"], 0)

    def test_detects_constant_columns(self):
        """Inject a zero-variance column."""
        X, y = _make_base_clf(n_samples=200)
        model, X_train, X_test, y_train, y_test = _split_and_fit(X, y)

        X_train_const = X_train.copy()
        X_train_const["constant_col"] = 42.0
        X_test_const  = X_test.copy()
        X_test_const["constant_col"]  = 42.0

        mi = _load(model, X_train_const, X_test_const, y_train, y_test)
        finding = quality_det.detect(mi)

        self.assertIsNotNone(finding)
        self.assertIn("constant_columns", finding.evidence)
        self.assertIn("constant_col", finding.evidence["constant_columns"])

    def test_clean_data_returns_none(self):
        """Clean data with no nulls, no duplicates, no constants → None."""
        X, y = _make_base_clf(n_samples=300)
        model, X_train, X_test, y_train, y_test = _split_and_fit(X, y)
        mi = _load(model, X_train, X_test, y_train, y_test)
        self.assertIsNone(quality_det.detect(mi))


# ── Test 5: Distribution shift detector ──────────────────────────────────────

class TestDistributionShiftDetector(unittest.TestCase):

    def test_detects_covariate_shift(self):
        """
        Inject shift by scaling test features by a large factor.
        Feature PSI will be high; label distribution unchanged → covariate shift.
        """
        X, y = _make_base_clf(n_samples=500)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=0
        )
        # Inject: multiply test features by 10 → massive distribution shift
        X_test_shifted = X_test.copy()
        X_test_shifted = X_test_shifted * 10 + 50

        model = LogisticRegression(max_iter=500)
        model.fit(X_train, y_train)

        mi = _load(model, X_train, X_test_shifted, y_train, y_test)
        finding = drift_det.detect(mi)

        self.assertIsNotNone(finding, "Shift should be detected")
        self.assertIn(finding.id, ["L2.1", "L2.2", "L2.3"])
        self.assertIn("min_feature_ks_pvalue", finding.evidence)
        self.assertLess(finding.evidence["min_feature_ks_pvalue"], 0.05)

    def test_no_shift_on_same_distribution(self):
        """
        Train and test from same distribution should not flag shift.
        (Uses random split — same underlying distribution by construction.)
        """
        X, y = _make_base_clf(n_samples=600)
        model, X_train, X_test, y_train, y_test = _split_and_fit(X, y)
        mi = _load(model, X_train, X_test, y_train, y_test)
        finding = drift_det.detect(mi)
        # Should be None or at most LOW/MEDIUM from sampling noise
        if finding is not None:
            self.assertIn(finding.severity, [Severity.LOW, Severity.MEDIUM])


# ── Test 6: Metric mismatch detector ─────────────────────────────────────────

class TestMetricMismatchDetector(unittest.TestCase):

    def test_detects_accuracy_f1_gap_on_imbalanced_data(self):
        """
        Inject 95:5 imbalance. A model defaulting to majority class will
        have high accuracy but very low F1 → metric mismatch detected.
        """
        n = 500
        rng = np.random.default_rng(7)
        y_vals = np.array([0] * 475 + [1] * 25)
        X_vals = rng.standard_normal((n, 5))
        X = pd.DataFrame(X_vals, columns=[f"feat_{i}" for i in range(5)])
        y = pd.Series(y_vals, name="target")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=0, stratify=y
        )
        # Train WITHOUT class weight so it ignores minority
        model = LogisticRegression(max_iter=500, class_weight=None)
        model.fit(X_train, y_train)

        mi = _load(model, X_train, X_test, y_train, y_test)
        finding = metric_det.detect(mi)

        self.assertIsNotNone(finding, "Metric mismatch should be detected")
        self.assertEqual(finding.id, "L3.1")
        self.assertIn("accuracy", finding.evidence)
        self.assertIn("weighted_f1", finding.evidence)

    def test_skipped_for_regression(self):
        """Metric mismatch detector only applies to classification."""
        X, y = make_regression(n_samples=200, n_features=4, random_state=0)
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
        y = pd.Series(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = LinearRegression().fit(X_train, y_train)
        mi = _load(model, X_train, X_test, y_train, y_test, task="regression")
        self.assertIsNone(metric_det.detect(mi))


# ── Test 7: run_all integration ───────────────────────────────────────────────

class TestRunAll(unittest.TestCase):

    def test_run_all_returns_list(self):
        """run_all() should always return a list (possibly empty)."""
        X, y = _make_base_clf(n_samples=400)
        model, X_train, X_test, y_train, y_test = _split_and_fit(X, y)
        mi = _load(model, X_train, X_test, y_train, y_test)
        results = run_all(mi)
        self.assertIsInstance(results, list)

    def test_run_all_multiple_failures_detected(self):
        """
        Inject BOTH imbalance AND data quality issues.
        run_all() should surface at least 2 findings.
        """
        n = 400
        rng = np.random.default_rng(3)
        y_vals = np.array([0] * 360 + [1] * 40)   # 10% minority
        X_vals = rng.standard_normal((n, 5))
        X = pd.DataFrame(X_vals, columns=[f"feat_{i}" for i in range(5)])
        y = pd.Series(y_vals, name="target")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=0
        )
        # Inject nulls
        X_train_dirty = X_train.copy()
        null_idx = X_train_dirty.sample(frac=0.25, random_state=0).index
        X_train_dirty.loc[null_idx, "feat_0"] = np.nan

        model = LogisticRegression(max_iter=500, class_weight="balanced")
        model.fit(X_train_dirty.fillna(0), y_train)

        mi = load_model_input(
            model, X_train_dirty, X_test, y_train, y_test,
            task_type="classification",
        )
        results = run_all(mi)
        self.assertGreaterEqual(len(results), 2)

    def test_run_all_finding_ids_are_valid_taxonomy_codes(self):
        """All finding IDs returned by run_all must be valid taxonomy codes."""
        from core.registry import FailureTaxonomy
        X, y = _make_base_clf(n_samples=300)
        model, X_train, X_test, y_train, y_test = _split_and_fit(X, y)
        mi = _load(model, X_train, X_test, y_train, y_test)
        for finding in run_all(mi):
            self.assertIn(finding.id, FailureTaxonomy.all_codes())


if __name__ == "__main__":
    unittest.main(verbosity=2)