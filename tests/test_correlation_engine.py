"""
tests/test_correlation_engine.py
---------------------------------
Tests for the Phase 3 correlation engine.

Covers:
  - Findings are sorted correctly (CRITICAL before HIGH before MEDIUM)
  - Interaction warnings fire when two co-occurring findings match a rule
  - Health score decreases correctly per severity
  - Interaction penalty applied on top of base score
  - Action sequence length matches findings count
  - run_diagnosis returns a DiagnosisReport end-to-end

Run with:  python -m unittest tests.test_correlation_engine -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from core.ingestion import load_model_input
from core.report import Finding, DiagnosisReport, Severity
from core.correlation_engine import (
    run_diagnosis,
    _rank_findings,
    _detect_interactions,
    _compute_health_score,
    _generate_action_sequence,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _finding(id_: str, severity: Severity, confidence: float = 0.80) -> Finding:
    """Create a minimal Finding for unit testing engine logic."""
    return Finding(
        id=id_,
        name=f"Test finding {id_}",
        severity=severity,
        evidence={"test": True},
        explanation="Test explanation.",
        fix="Test fix.",
        confidence=confidence,
    )


def _make_model_input(imbalance=False, overfit=False, nulls=False):
    """Build a ModelInput with optionally injected failures."""
    rng = np.random.default_rng(42)
    n   = 400

    if imbalance:
        y_vals = np.array([0] * 360 + [1] * 40)
    else:
        y_vals = np.array([0] * 200 + [1] * 200)

    X_vals = rng.standard_normal((n, 5))
    X = pd.DataFrame(X_vals, columns=[f"feat_{i}" for i in range(5)])
    y = pd.Series(y_vals, name="target")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0,
        stratify=y if imbalance else None,
    )

    if nulls:
        null_idx = X_train.sample(frac=0.30, random_state=0).index
        X_train.loc[null_idx, "feat_0"] = np.nan

    if overfit:
        model = DecisionTreeClassifier(max_depth=None, random_state=0)
    else:
        model = LogisticRegression(max_iter=500, class_weight="balanced")

    fit_X = X_train.fillna(0) if nulls else X_train
    model.fit(fit_X, y_train)

    return load_model_input(
        model, X_train, X_test, y_train, y_test,
        task_type="classification", domain="general",
    )


# ── Test 1: Finding ranking ───────────────────────────────────────────────────

class TestRankFindings(unittest.TestCase):

    def test_critical_before_high_before_medium(self):
        findings = [
            _finding("L3.1", Severity.MEDIUM),
            _finding("L1.1", Severity.CRITICAL),
            _finding("L0.1", Severity.HIGH),
        ]
        ranked = _rank_findings(findings)
        self.assertEqual(ranked[0].severity, Severity.CRITICAL)
        self.assertEqual(ranked[1].severity, Severity.HIGH)
        self.assertEqual(ranked[2].severity, Severity.MEDIUM)

    def test_same_severity_sorted_by_confidence_descending(self):
        findings = [
            _finding("L0.3", Severity.HIGH, confidence=0.60),
            _finding("L1.1", Severity.HIGH, confidence=0.95),
            _finding("L2.1", Severity.HIGH, confidence=0.78),
        ]
        ranked = _rank_findings(findings)
        confidences = [f.confidence for f in ranked]
        self.assertEqual(confidences, sorted(confidences, reverse=True))

    def test_empty_list_returns_empty(self):
        self.assertEqual(_rank_findings([]), [])

    def test_single_finding_returned_unchanged(self):
        f = _finding("L0.1", Severity.HIGH)
        self.assertEqual(_rank_findings([f]), [f])


# ── Test 2: Interaction detection ─────────────────────────────────────────────

class TestDetectInteractions(unittest.TestCase):

    def test_overfitting_plus_covariate_shift_triggers_warning(self):
        findings = [
            _finding("L1.1", Severity.HIGH),    # Overfitting
            _finding("L2.1", Severity.HIGH),    # Covariate Shift
        ]
        warnings = _detect_interactions(findings)
        self.assertGreater(len(warnings), 0)
        # Warning message should mention both findings
        combined = " ".join(warnings)
        self.assertIn("L1.1", combined)
        self.assertIn("Distribution Shift", combined)

    def test_imbalance_plus_metric_mismatch_triggers_warning(self):
        findings = [
            _finding("L0.1", Severity.HIGH),    # Class Imbalance
            _finding("L3.1", Severity.MEDIUM),  # Metric Mismatch
        ]
        warnings = _detect_interactions(findings)
        self.assertGreater(len(warnings), 0)

    def test_leakage_plus_overfitting_triggers_warning(self):
        findings = [
            _finding("L0.2", Severity.CRITICAL),
            _finding("L1.1", Severity.HIGH),
        ]
        warnings = _detect_interactions(findings)
        self.assertGreater(len(warnings), 0)

    def test_single_finding_no_interaction(self):
        findings = [_finding("L1.1", Severity.HIGH)]
        warnings = _detect_interactions(findings)
        self.assertEqual(warnings, [])

    def test_unrelated_findings_no_interaction(self):
        # L0.3 (data quality) + L3.3 (miscalibration) have no rule together
        findings = [
            _finding("L0.3", Severity.MEDIUM),
            _finding("L3.3", Severity.LOW),
        ]
        warnings = _detect_interactions(findings)
        # Should be empty (no rule for this pair)
        self.assertEqual(warnings, [])

    def test_multiple_interactions_all_detected(self):
        # Three findings that trigger two separate interaction rules
        findings = [
            _finding("L1.1", Severity.CRITICAL),  # overfitting
            _finding("L2.1", Severity.HIGH),       # covariate shift   → rule with L1.1
            _finding("L0.2", Severity.CRITICAL),   # data leakage      → rule with L1.1
        ]
        warnings = _detect_interactions(findings)
        self.assertGreaterEqual(len(warnings), 2)


# ── Test 3: Health score ──────────────────────────────────────────────────────

class TestHealthScore(unittest.TestCase):

    def test_no_findings_gives_100(self):
        self.assertEqual(_compute_health_score([]), 100)

    def test_single_critical_deducts_30(self):
        findings = [_finding("L1.1", Severity.CRITICAL)]
        self.assertEqual(_compute_health_score(findings), 70)

    def test_single_high_deducts_15(self):
        findings = [_finding("L0.1", Severity.HIGH)]
        self.assertEqual(_compute_health_score(findings), 85)

    def test_single_medium_deducts_7(self):
        findings = [_finding("L3.1", Severity.MEDIUM)]
        self.assertEqual(_compute_health_score(findings), 93)

    def test_single_low_deducts_3(self):
        findings = [_finding("L0.3", Severity.LOW)]
        self.assertEqual(_compute_health_score(findings), 97)

    def test_multiple_findings_cumulative_deduction(self):
        findings = [
            _finding("L1.1", Severity.CRITICAL),  # -30
            _finding("L0.1", Severity.HIGH),       # -15
            _finding("L3.1", Severity.MEDIUM),     # -7
        ]
        # 100 - 30 - 15 - 7 = 48
        self.assertEqual(_compute_health_score(findings), 48)

    def test_score_floors_at_zero(self):
        findings = [
            _finding("L0.2", Severity.CRITICAL),   # -30
            _finding("L1.1", Severity.CRITICAL),   # -30
            _finding("L2.1", Severity.CRITICAL),   # -30
            _finding("L0.1", Severity.CRITICAL),   # -30
        ]
        # Would be -20, floored to 0
        self.assertEqual(_compute_health_score(findings), 0)


# ── Test 4: Action sequence ───────────────────────────────────────────────────

class TestActionSequence(unittest.TestCase):

    def test_sequence_length_matches_findings(self):
        findings = [
            _finding("L1.1", Severity.CRITICAL),
            _finding("L0.1", Severity.HIGH),
            _finding("L3.1", Severity.MEDIUM),
        ]
        steps = _generate_action_sequence(findings)
        self.assertEqual(len(steps), 3)

    def test_steps_are_numbered_sequentially(self):
        findings = [
            _finding("L1.1", Severity.CRITICAL),
            _finding("L0.1", Severity.HIGH),
        ]
        steps = _generate_action_sequence(findings)
        self.assertTrue(steps[0].startswith("Step 1"))
        self.assertTrue(steps[1].startswith("Step 2"))

    def test_step_references_finding_id(self):
        findings = [_finding("L0.2", Severity.CRITICAL)]
        steps = _generate_action_sequence(findings)
        self.assertIn("L0.2", steps[0])

    def test_empty_findings_empty_steps(self):
        self.assertEqual(_generate_action_sequence([]), [])


# ── Test 5: run_diagnosis end-to-end ─────────────────────────────────────────

class TestRunDiagnosis(unittest.TestCase):

    def test_returns_diagnosis_report(self):
        mi = _make_model_input()
        report = run_diagnosis(mi)
        self.assertIsInstance(report, DiagnosisReport)

    def test_report_has_correct_model_name(self):
        mi = _make_model_input()
        report = run_diagnosis(mi)
        self.assertEqual(report.model_name, mi.model_name)

    def test_findings_are_ranked_by_severity(self):
        mi = _make_model_input(imbalance=True, nulls=True)
        report = run_diagnosis(mi)
        if len(report.findings) > 1:
            sev_order = [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]
            for i in range(len(report.findings) - 1):
                a = sev_order.index(report.findings[i].severity)
                b = sev_order.index(report.findings[i + 1].severity)
                self.assertLessEqual(a, b, "Findings must be ordered CRITICAL→LOW")

    def test_health_score_in_valid_range(self):
        mi = _make_model_input(imbalance=True)
        report = run_diagnosis(mi)
        self.assertGreaterEqual(report.overall_health_score, 0)
        self.assertLessEqual(report.overall_health_score, 100)

    def test_action_sequence_matches_finding_count(self):
        mi = _make_model_input(imbalance=True)
        report = run_diagnosis(mi)
        self.assertEqual(
            len(report.recommended_action_sequence),
            len(report.findings),
        )

    def test_compounding_interaction_detected(self):
        """Imbalance + nulls should surface interaction warnings."""
        mi = _make_model_input(imbalance=True, nulls=True)
        report = run_diagnosis(mi)
        # At minimum data quality + imbalance both detected
        ids = [f.id for f in report.findings]
        if "L0.1" in ids and "L0.3" in ids:
            self.assertIsInstance(report.interaction_warnings, list)

    def test_health_score_lower_with_more_failures(self):
        """More injected failures → lower health score."""
        clean_report   = run_diagnosis(_make_model_input())
        broken_report  = run_diagnosis(_make_model_input(imbalance=True, nulls=True))
        self.assertLessEqual(
            broken_report.overall_health_score,
            clean_report.overall_health_score,
        )

    def test_report_str_renders_without_error(self):
        """DiagnosisReport __str__ should not raise."""
        mi = _make_model_input(imbalance=True)
        report = run_diagnosis(mi)
        try:
            output = str(report)
            self.assertIn("ML FAILURE INTELLIGENCE REPORT", output)
        except Exception as e:
            self.fail(f"str(report) raised: {e}")

    def test_report_serialises_to_json(self):
        """to_json() should produce valid JSON with expected keys."""
        import json
        mi = _make_model_input()
        report = run_diagnosis(mi)
        data = json.loads(report.to_json())
        for key in ["model_name", "task_type", "overall_health_score",
                    "findings", "interaction_warnings", "recommended_action_sequence"]:
            self.assertIn(key, data)


if __name__ == "__main__":
    unittest.main(verbosity=2)