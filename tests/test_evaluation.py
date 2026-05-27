"""
tests/test_evaluation.py
-------------------------
Tests for Phase 7: FailureInjector and BenchmarkRunner.

Run with:  python -m unittest tests.test_evaluation -v
"""

import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
import pandas as pd
import numpy as np

from core.ingestion import load_model_input
from core.correlation_engine import run_diagnosis
from evaluation.injector import FailureInjector, InjectedBundle
from evaluation.benchmark import BenchmarkRunner, BenchmarkReport, CaseResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def _engine_finds(bundle: InjectedBundle, domain="general") -> list:
    """Run the engine on a bundle and return list of finding IDs."""
    mi = load_model_input(
        model=bundle.model,
        X_train=bundle.X_train, X_test=bundle.X_test,
        y_train=bundle.y_train, y_test=bundle.y_test,
        task_type=bundle.task_type, domain=domain,
    )
    report = run_diagnosis(mi)
    return [f.id for f in report.findings]


# ── Test 1: FailureInjector bundles ──────────────────────────────────────────

class TestFailureInjector(unittest.TestCase):

    def setUp(self):
        self.inj = FailureInjector(random_state=42)

    def _check_bundle(self, bundle: InjectedBundle):
        """Shared assertions for all injected bundles."""
        self.assertIsInstance(bundle, InjectedBundle)
        self.assertIsNotNone(bundle.model)
        self.assertIsInstance(bundle.X_train, pd.DataFrame)
        self.assertIsInstance(bundle.X_test,  pd.DataFrame)
        self.assertIsInstance(bundle.y_train, pd.Series)
        self.assertIsInstance(bundle.y_test,  pd.Series)
        self.assertGreater(len(bundle.X_train), 0)
        self.assertGreater(len(bundle.X_test),  0)
        self.assertFalse(bundle.injected_failure == "")
        self.assertFalse(bundle.description == "")

    def test_overfitting_bundle_valid(self):
        b = self.inj.inject_overfitting()
        self._check_bundle(b)
        self.assertEqual(b.injected_failure, "L1.1")
        self.assertEqual(b.task_type, "classification")

    def test_class_imbalance_bundle_valid(self):
        b = self.inj.inject_class_imbalance(minority_ratio=0.05)
        self._check_bundle(b)
        self.assertEqual(b.injected_failure, "L0.1")

    def test_class_imbalance_minority_ratio_correct(self):
        ratio = 0.05
        b = self.inj.inject_class_imbalance(minority_ratio=ratio)
        actual_ratio = b.y_train.value_counts().min() / len(b.y_train)
        # Allow ±3% tolerance from train/test split
        self.assertAlmostEqual(actual_ratio, ratio, delta=0.04)

    def test_data_leakage_bundle_valid(self):
        b = self.inj.inject_data_leakage()
        self._check_bundle(b)
        self.assertEqual(b.injected_failure, "L0.2")
        self.assertIn("leaked_target", b.X_train.columns)

    def test_data_quality_bundle_valid(self):
        b = self.inj.inject_data_quality()
        self._check_bundle(b)
        self.assertEqual(b.injected_failure, "L0.3")
        # Should have nulls
        self.assertTrue(b.X_train.isnull().any().any())

    def test_data_quality_has_constant_column(self):
        b = self.inj.inject_data_quality(add_constant_col=True)
        constant_cols = [c for c in b.X_train.columns if b.X_train[c].nunique() <= 1]
        self.assertGreater(len(constant_cols), 0)

    def test_distribution_shift_bundle_valid(self):
        b = self.inj.inject_distribution_shift(shift_magnitude=10.0)
        self._check_bundle(b)
        self.assertEqual(b.injected_failure, "L2.1")
        # Test features should be shifted far from train features
        train_mean = b.X_train.mean().mean()
        test_mean  = b.X_test.mean().mean()
        self.assertGreater(abs(test_mean - train_mean), 5.0)

    def test_label_shift_bundle_valid(self):
        b = self.inj.inject_label_shift(shift_ratio=0.45)
        self._check_bundle(b)
        self.assertEqual(b.injected_failure, "L2.2")
        # Label distribution should actually have shifted
        train_pos = b.y_train.mean()
        test_pos  = b.y_test.mean()
        self.assertGreater(abs(train_pos - test_pos), 0.10)

    def test_metric_mismatch_bundle_valid(self):
        b = self.inj.inject_metric_mismatch(minority_ratio=0.04)
        self._check_bundle(b)
        self.assertEqual(b.injected_failure, "L3.1")

    def test_clean_baseline_bundle_valid(self):
        b = self.inj.make_clean_baseline()
        self._check_bundle(b)
        self.assertEqual(b.injected_failure, "NONE")

    def test_different_random_states_produce_different_data(self):
        b1 = FailureInjector(random_state=1).inject_overfitting()
        b2 = FailureInjector(random_state=2).inject_overfitting()
        # Different seeds should produce different training data
        self.assertFalse(b1.X_train.equals(b2.X_train))


# ── Test 2: Engine detects each injected failure ──────────────────────────────

class TestEngineDetectsInjections(unittest.TestCase):
    """
    These are the core validation tests: for each injected failure,
    verify the engine actually detects it.
    """

    def setUp(self):
        self.inj = FailureInjector(random_state=42)

    def test_detects_overfitting(self):
        b = self.inj.inject_overfitting(noise_ratio=0.35)
        ids = _engine_finds(b)
        self.assertIn("L1.1", ids, f"L1.1 not in {ids}")

    def test_detects_class_imbalance(self):
        b = self.inj.inject_class_imbalance(minority_ratio=0.04)
        ids = _engine_finds(b)
        self.assertIn("L0.1", ids, f"L0.1 not in {ids}")

    def test_detects_data_leakage(self):
        b = self.inj.inject_data_leakage()
        ids = _engine_finds(b)
        self.assertIn("L0.2", ids, f"L0.2 not in {ids}")

    def test_detects_data_quality(self):
        b = self.inj.inject_data_quality()
        ids = _engine_finds(b)
        self.assertIn("L0.3", ids, f"L0.3 not in {ids}")

    def test_detects_distribution_shift(self):
        b = self.inj.inject_distribution_shift(shift_magnitude=10.0)
        ids = _engine_finds(b)
        self.assertIn("L2.1", ids, f"L2.1 not in {ids}")

    def test_detects_label_shift(self):
        b = self.inj.inject_label_shift(shift_ratio=0.45)
        ids = _engine_finds(b)
        # L2.2 (Label Shift) OR L2.3 (Concept Drift) are both correct:
        # asymmetric label flip changes P(Y) and can also affect P(Y|X).
        detected = "L2.2" in ids or "L2.3" in ids
        self.assertTrue(detected, f"Expected L2.2 or L2.3 in {ids}")

    def test_detects_metric_mismatch(self):
        b = self.inj.inject_metric_mismatch(minority_ratio=0.04)
        ids = _engine_finds(b)
        self.assertIn("L3.1", ids, f"L3.1 not in {ids}")

    def test_leakage_finding_is_critical(self):
        """Data leakage must always surface as CRITICAL."""
        from core.report import Severity
        b = self.inj.inject_data_leakage()
        mi = load_model_input(
            b.model, b.X_train, b.X_test, b.y_train, b.y_test,
            task_type=b.task_type, domain="general",
        )
        report = run_diagnosis(mi)
        leakage_findings = [f for f in report.findings if f.id == "L0.2"]
        self.assertTrue(len(leakage_findings) > 0)
        self.assertEqual(leakage_findings[0].severity, Severity.CRITICAL)

    def test_each_finding_has_non_empty_fix(self):
        """Every finding the engine produces must include a fix."""
        bundles = [
            self.inj.inject_overfitting(),
            self.inj.inject_class_imbalance(),
            self.inj.inject_data_leakage(),
            self.inj.inject_data_quality(),
        ]
        for b in bundles:
            mi = load_model_input(
                b.model, b.X_train, b.X_test, b.y_train, b.y_test,
                task_type=b.task_type, domain="general",
            )
            report = run_diagnosis(mi)
            for f in report.findings:
                self.assertTrue(
                    f.fix and f.fix.strip(),
                    f"Finding {f.id} has empty fix"
                )


# ── Test 3: BenchmarkRunner ───────────────────────────────────────────────────

class TestBenchmarkRunner(unittest.TestCase):

    def setUp(self):
        # Use fewer clean cases for speed
        self.runner = BenchmarkRunner(
            domain="general",
            random_state=42,
            n_clean_cases=2,
        )

    def test_run_returns_benchmark_report(self):
        bm = self.runner.run()
        self.assertIsInstance(bm, BenchmarkReport)

    def test_case_count_correct(self):
        bm = self.runner.run()
        # 8 failure types + 2 clean = 10
        self.assertEqual(bm.n_cases, 10)
        self.assertEqual(bm.n_failure_cases, 8)
        self.assertEqual(bm.n_clean_cases, 2)

    def test_detection_rate_keys_are_taxonomy_codes(self):
        from core.registry import FailureTaxonomy
        bm = self.runner.run()
        for code in bm.detection_rate.keys():
            self.assertIn(code, FailureTaxonomy.all_codes(),
                          f"Unknown code in detection_rate: {code}")

    def test_detection_rate_in_valid_range(self):
        bm = self.runner.run()
        self.assertGreaterEqual(bm.overall_detection_rate, 0.0)
        self.assertLessEqual(bm.overall_detection_rate, 1.0)

    def test_far_in_valid_range(self):
        bm = self.runner.run()
        self.assertGreaterEqual(bm.false_alarm_rate, 0.0)
        self.assertLessEqual(bm.false_alarm_rate, 1.0)

    def test_severity_accuracy_in_valid_range(self):
        bm = self.runner.run()
        self.assertGreaterEqual(bm.severity_accuracy, 0.0)
        self.assertLessEqual(bm.severity_accuracy, 1.0)

    def test_fix_recommendation_rate_is_1(self):
        """Every detected finding should have a fix — target is 100%."""
        bm = self.runner.run()
        self.assertEqual(bm.fix_recommendation_rate, 1.0)

    def test_overall_detection_meets_target(self):
        """Core requirement: overall detection rate > 85%."""
        bm = self.runner.run()
        self.assertGreaterEqual(
            bm.overall_detection_rate, bm.TARGET_DR,
            f"Detection rate {bm.overall_detection_rate:.1%} < target {bm.TARGET_DR:.1%}"
        )

    def test_all_failure_types_detected(self):
        """Each individual failure type should have DR >= 85%."""
        bm = self.runner.run()
        for code, dr in bm.detection_rate.items():
            self.assertGreaterEqual(
                dr, bm.TARGET_DR,
                f"Detection rate for {code} = {dr:.1%} < target {bm.TARGET_DR:.1%}"
            )

    def test_avg_elapsed_is_fast(self):
        """Engine should run in well under 1 second per case."""
        bm = self.runner.run()
        self.assertLess(bm.avg_elapsed_seconds, 1.0)

    def test_case_results_length_matches_n_cases(self):
        bm = self.runner.run()
        self.assertEqual(len(bm.case_results), bm.n_cases)

    def test_case_result_fields_populated(self):
        bm = self.runner.run()
        for case in bm.case_results:
            self.assertIsInstance(case, CaseResult)
            self.assertIsInstance(case.detected, bool)
            self.assertIsInstance(case.finding_ids, list)
            self.assertIsInstance(case.elapsed_seconds, float)
            self.assertGreater(case.elapsed_seconds, 0)


# ── Test 4: BenchmarkRunner.save ─────────────────────────────────────────────

class TestBenchmarkSave(unittest.TestCase):

    def setUp(self):
        self.runner  = BenchmarkRunner(domain="general", random_state=42,
                                       n_clean_cases=1)
        self.tmp     = tempfile.mkdtemp()
        self.bm      = self.runner.run()

    def test_save_creates_json_file(self):
        path = os.path.join(self.tmp, "bench.json")
        self.runner.save(self.bm, path)
        self.assertTrue(os.path.exists(path))

    def test_saved_json_is_valid(self):
        path = os.path.join(self.tmp, "bench.json")
        self.runner.save(self.bm, path)
        with open(path) as f:
            data = json.load(f)
        self.assertIsInstance(data, dict)

    def test_saved_json_has_required_keys(self):
        path = os.path.join(self.tmp, "bench.json")
        self.runner.save(self.bm, path)
        with open(path) as f:
            data = json.load(f)
        for key in ["domain", "n_cases", "overall_detection_rate",
                    "false_alarm_rate", "severity_accuracy",
                    "fix_recommendation_rate", "passed",
                    "detection_rate_by_type", "cases", "targets"]:
            self.assertIn(key, data)

    def test_saved_cases_count_matches(self):
        path = os.path.join(self.tmp, "bench.json")
        self.runner.save(self.bm, path)
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(len(data["cases"]), self.bm.n_cases)


if __name__ == "__main__":
    unittest.main(verbosity=2)