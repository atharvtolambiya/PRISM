"""
tests/test_hitl.py
-------------------
Tests for Phase 6: Human-in-the-Loop validation.

Because HITL is interactive (reads from stdin), all tests mock
the built-in input() function to simulate user responses.

Run with:  python -m unittest tests.test_hitl -v
"""

import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from core.ingestion import load_model_input
from core.report import Finding, DiagnosisReport, Severity
from core.correlation_engine import run_diagnosis
from core.hitl import (
    run_interactive_review,
    _review_finding,
    _append_to_feedback_log,
    _build_session_record,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _f(id_="L1.1", severity=Severity.HIGH, name="Test Finding"):
    return Finding(
        id=id_, name=name, severity=severity,
        evidence={"train_accuracy": 0.97, "test_accuracy": 0.74, "gap": 0.23},
        explanation="Model has memorised training data.",
        fix="Reduce model complexity.",
        confidence=0.88,
    )


def _make_report_with_findings(*findings) -> DiagnosisReport:
    return DiagnosisReport(
        model_name="TestModel",
        task_type="classification",
        domain="general",
        overall_health_score=55,
        findings=list(findings),
        interaction_warnings=[],
        recommended_action_sequence=["Step 1 → Fix overfitting"],
    )


def _make_real_report() -> DiagnosisReport:
    """Build a real report with injected failures."""
    rng = np.random.default_rng(42)
    n   = 400
    y   = pd.Series([0] * 360 + [1] * 40)
    X   = pd.DataFrame(rng.standard_normal((n, 5)),
                       columns=[f"f{i}" for i in range(5)])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )
    model = DecisionTreeClassifier(max_depth=None, random_state=0)
    model.fit(X_train, y_train)
    mi = load_model_input(model, X_train, X_test, y_train, y_test,
                          task_type="classification", domain="general")
    return run_diagnosis(mi)


# ── Test 1: _review_finding unit tests ───────────────────────────────────────

class TestReviewFinding(unittest.TestCase):

    def _run(self, finding, inputs):
        """Helper: run _review_finding with mocked stdin inputs."""
        input_iter = iter(inputs)
        with patch("builtins.input", side_effect=lambda _="": next(input_iter)):
            return _review_finding(finding, index=1, total=3)

    def test_confirm_returns_original_finding(self):
        f = _f()
        result, fb = self._run(f, ["1"])  # choice=1 (confirm)
        self.assertIsNotNone(result)
        self.assertEqual(result.id, f.id)
        self.assertEqual(fb["action"], "confirmed")

    def test_confirm_preserves_severity(self):
        f = _f(severity=Severity.CRITICAL)
        result, fb = self._run(f, ["1"])
        self.assertEqual(result.severity, Severity.CRITICAL)

    def test_reject_returns_none(self):
        f = _f()
        # choice=2 (reject), then empty reason
        result, fb = self._run(f, ["2", ""])
        self.assertIsNone(result)
        self.assertEqual(fb["action"], "rejected")

    def test_reject_with_reason_logs_note(self):
        f = _f()
        result, fb = self._run(f, ["2", "False positive — this is normal"])
        self.assertIsNone(result)
        self.assertEqual(fb["user_note"], "False positive — this is normal")

    def test_reject_no_reason_logs_none(self):
        f = _f()
        result, fb = self._run(f, ["2", ""])
        self.assertIsNone(fb["user_note"])

    def test_change_severity_to_critical(self):
        f = _f(severity=Severity.MEDIUM)
        # choice=3 (change), new_severity=1 (CRITICAL), empty reason
        result, fb = self._run(f, ["3", "1", ""])
        self.assertIsNotNone(result)
        self.assertEqual(result.severity, Severity.CRITICAL)
        self.assertEqual(fb["action"], "severity_changed")
        self.assertEqual(fb["new_severity"], "CRITICAL")
        self.assertEqual(fb["original_severity"], "MEDIUM")

    def test_change_severity_to_low(self):
        f = _f(severity=Severity.HIGH)
        result, fb = self._run(f, ["3", "4", "not important"])
        self.assertEqual(result.severity, Severity.LOW)
        self.assertEqual(fb["user_note"], "not important")

    def test_change_severity_adds_override_note(self):
        f = _f(severity=Severity.HIGH)
        result, fb = self._run(f, ["3", "1", ""])
        self.assertIn("HUMAN OVERRIDE", result.notes)

    def test_add_note_keeps_severity(self):
        f = _f(severity=Severity.HIGH)
        result, fb = self._run(f, ["4", "Reviewed by team lead"])
        self.assertIsNotNone(result)
        self.assertEqual(result.severity, Severity.HIGH)
        self.assertEqual(fb["action"], "note_added")
        self.assertIn("HUMAN NOTE", result.notes)
        self.assertIn("Reviewed by team lead", result.notes)

    def test_skip_returns_original_unchanged(self):
        f = _f(severity=Severity.CRITICAL)
        result, fb = self._run(f, ["5"])
        self.assertIsNotNone(result)
        self.assertEqual(result.severity, Severity.CRITICAL)
        self.assertEqual(fb["action"], "skipped")

    def test_feedback_always_contains_required_keys(self):
        f = _f()
        for choice, extras in [("1", []), ("2", [""]), ("5", [])]:
            _, fb = self._run(f, [choice] + extras)
            for key in ["finding_id", "finding_name", "action",
                        "original_severity", "new_severity", "user_note"]:
                self.assertIn(key, fb)


# ── Test 2: Feedback log ──────────────────────────────────────────────────────

class TestFeedbackLog(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.log_path = os.path.join(self.tmp, "feedback_log.json")

    def _make_session(self, score_orig=55, score_valid=70):
        report = _make_report_with_findings(_f())
        validated = _make_report_with_findings()
        validated.overall_health_score = score_valid
        items = [{
            "finding_id": "L1.1", "finding_name": "Overfitting",
            "action": "confirmed", "original_severity": "HIGH",
            "new_severity": None, "user_note": None,
        }]
        return _build_session_record(report, validated, items, "abc123")

    def test_log_file_created(self):
        session = self._make_session()
        _append_to_feedback_log(session, self.log_path)
        self.assertTrue(os.path.exists(self.log_path))

    def test_log_contains_session_record(self):
        session = self._make_session()
        _append_to_feedback_log(session, self.log_path)
        with open(self.log_path) as f:
            data = json.load(f)
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 1)

    def test_multiple_sessions_appended(self):
        for _ in range(3):
            _append_to_feedback_log(self._make_session(), self.log_path)
        with open(self.log_path) as f:
            data = json.load(f)
        self.assertEqual(len(data), 3)

    def test_session_has_required_keys(self):
        session = self._make_session()
        _append_to_feedback_log(session, self.log_path)
        with open(self.log_path) as f:
            record = json.load(f)[0]
        for key in ["session_id", "timestamp", "model_hash", "model_name",
                    "domain", "feedback", "original_health_score",
                    "validated_health_score", "findings_removed",
                    "findings_escalated"]:
            self.assertIn(key, record)

    def test_session_id_is_unique(self):
        for _ in range(5):
            _append_to_feedback_log(self._make_session(), self.log_path)
        with open(self.log_path) as f:
            data = json.load(f)
        ids = [r["session_id"] for r in data]
        self.assertEqual(len(ids), len(set(ids)))

    def test_session_counts_removals_correctly(self):
        report    = _make_report_with_findings(_f("L1.1"), _f("L0.1"))
        validated = _make_report_with_findings(_f("L1.1"))  # one removed
        items = [
            {"finding_id": "L1.1", "action": "confirmed",
             "original_severity": "HIGH", "new_severity": None, "user_note": None,
             "finding_name": "Overfitting"},
            {"finding_id": "L0.1", "action": "rejected",
             "original_severity": "HIGH", "new_severity": None, "user_note": None,
             "finding_name": "Class Imbalance"},
        ]
        session = _build_session_record(report, validated, items, "xyz")
        self.assertEqual(session["findings_removed"], 1)


# ── Test 3: run_interactive_review end-to-end ─────────────────────────────────

class TestRunInteractiveReview(unittest.TestCase):

    def setUp(self):
        self.tmp      = tempfile.mkdtemp()
        self.log_path = os.path.join(self.tmp, "feedback.json")

    def _run_review(self, report, choices):
        """
        Simulate a full review session.
        `choices` is a flat list of all stdin inputs in order.
        """
        input_iter = iter(choices)
        with patch("builtins.input", side_effect=lambda _="": next(input_iter)):
            return run_interactive_review(
                report,
                feedback_log_path=self.log_path,
                model_hash="testhash",
            )

    def test_confirm_all_keeps_all_findings(self):
        report = _make_report_with_findings(_f("L1.1"), _f("L0.1"))
        # Two findings, both confirmed (choice=1 each)
        result = self._run_review(report, ["1", "1"])
        self.assertEqual(len(result.findings), 2)

    def test_reject_one_removes_it(self):
        report = _make_report_with_findings(_f("L1.1"), _f("L0.1"))
        # First: reject (choice=2, empty reason), Second: confirm (choice=1)
        result = self._run_review(report, ["2", "", "1"])
        self.assertEqual(len(result.findings), 1)
        self.assertEqual(result.findings[0].id, "L0.1")

    def test_reject_all_gives_empty_findings(self):
        report = _make_report_with_findings(_f("L1.1"), _f("L0.1"))
        result = self._run_review(report, ["2", "", "2", ""])
        self.assertEqual(len(result.findings), 0)

    def test_severity_change_is_applied(self):
        report = _make_report_with_findings(_f("L1.1", Severity.MEDIUM))
        # choice=3 (change), new=1 (CRITICAL), empty reason
        result = self._run_review(report, ["3", "1", ""])
        self.assertEqual(result.findings[0].severity, Severity.CRITICAL)

    def test_health_score_increases_when_finding_rejected(self):
        report = _make_real_report()
        # reject all findings
        inputs = ["2", ""] * len(report.findings)
        result = self._run_review(report, inputs)
        self.assertGreaterEqual(result.overall_health_score,
                                report.overall_health_score)

    def test_health_score_is_100_when_all_rejected(self):
        report = _make_report_with_findings(_f("L1.1", Severity.HIGH))
        result = self._run_review(report, ["2", ""])
        self.assertEqual(result.overall_health_score, 100)

    def test_feedback_log_written(self):
        report = _make_report_with_findings(_f())
        self._run_review(report, ["1"])
        self.assertTrue(os.path.exists(self.log_path))

    def test_feedback_log_contains_one_record_per_session(self):
        report = _make_report_with_findings(_f())
        self._run_review(report, ["1"])
        with open(self.log_path) as f:
            data = json.load(f)
        self.assertEqual(len(data), 1)

    def test_empty_report_returns_unchanged(self):
        report = _make_report_with_findings()
        # no input needed — no findings to review
        with patch("builtins.input", side_effect=lambda _="": ""):
            result = run_interactive_review(
                report, feedback_log_path=self.log_path
            )
        self.assertEqual(result.findings, [])

    def test_action_sequence_updated_after_review(self):
        report = _make_report_with_findings(_f("L1.1"), _f("L0.1"))
        # Keep both
        result = self._run_review(report, ["1", "1"])
        self.assertEqual(len(result.recommended_action_sequence),
                         len(result.findings))

    def test_returns_diagnosis_report_instance(self):
        report = _make_report_with_findings(_f())
        result = self._run_review(report, ["1"])
        self.assertIsInstance(result, DiagnosisReport)


if __name__ == "__main__":
    unittest.main(verbosity=2)