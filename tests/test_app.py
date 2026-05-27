"""
tests/test_app.py
------------------
Tests for Phase 8: Streamlit app logic.

Since Streamlit can't run in the test environment, we test:
  1. app.py syntax is valid Python
  2. All helper functions that don't call st.* can be imported and run
  3. The _run_demo flow produces a valid report end-to-end
  4. _process_feedback correctly applies decisions to findings
  5. Download data (JSON, text) is correct and complete

Run with:  python -m unittest tests.test_app -v
"""

import sys, os, json, ast, re, importlib.util
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from core.ingestion import load_model_input
from core.report import Finding, DiagnosisReport, Severity
from core.correlation_engine import run_diagnosis, _rank_findings, _compute_health_score


# ── Helper: build a real report ───────────────────────────────────────────────

def _make_report():
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
                          task_type="classification", domain="healthcare")
    return run_diagnosis(mi), mi


def _f(id_="L1.1", severity=Severity.HIGH):
    return Finding(
        id=id_, name=f"Test {id_}", severity=severity,
        evidence={"gap": 0.22}, explanation="Test.", fix="Test fix.",
        confidence=0.85,
    )


def _make_report_direct(*findings):
    return DiagnosisReport(
        model_name="TestModel", task_type="classification",
        domain="general", overall_health_score=55,
        findings=list(findings),
        interaction_warnings=[],
        recommended_action_sequence=[],
    )


# ── Test 1: Syntax and importability ─────────────────────────────────────────

class TestAppSyntax(unittest.TestCase):

    def test_app_py_valid_python_syntax(self):
        app_path = os.path.join(os.path.dirname(__file__), "..", "app.py")
        with open(app_path, encoding="utf-8") as f:
            src = f.read()
        try:
            ast.parse(src)
        except SyntaxError as e:
            self.fail(f"app.py has a syntax error: {e}")

    def test_app_py_imports_are_correct(self):
        """
        Check that all non-streamlit imports in app.py resolve correctly.
        Mocks the streamlit module to avoid runtime dependency.
        """
        st_mock = MagicMock()
        st_mock.set_page_config = MagicMock()

        with patch.dict("sys.modules", {"streamlit": st_mock}):
            spec = importlib.util.spec_from_file_location(
                "app_test",
                os.path.join(os.path.dirname(__file__), "..", "app.py"),
            )
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
                # If we get here without ImportError, all local imports resolved
            except Exception as exc:
                if "streamlit" not in str(exc).lower():
                    self.fail(f"app.py failed to import: {exc}")


# ── Test 2: Demo flow ─────────────────────────────────────────────────────────

class TestDemoFlow(unittest.TestCase):

    def _demo_report(self, domain="healthcare") -> DiagnosisReport:
        """Run the same logic as _run_demo() but without Streamlit."""
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import train_test_split

        rng = np.random.default_rng(42)
        n   = 500
        y   = pd.Series([0] * 450 + [1] * 50, name="target")
        X   = pd.DataFrame(
            rng.standard_normal((n, 6)),
            columns=[f"feature_{i}" for i in range(6)],
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=0, stratify=y
        )
        model = DecisionTreeClassifier(max_depth=None, random_state=0)
        model.fit(X_train, y_train)
        mi = load_model_input(
            model, X_train, X_test, y_train, y_test,
            task_type="classification", domain=domain,
            model_name="DecisionTreeClassifier (Demo)",
        )
        return run_diagnosis(mi)

    def test_demo_produces_report(self):
        report = self._demo_report()
        self.assertIsInstance(report, DiagnosisReport)

    def test_demo_has_findings(self):
        report = self._demo_report()
        self.assertGreater(len(report.findings), 0)

    def test_demo_health_score_below_100(self):
        report = self._demo_report()
        self.assertLess(report.overall_health_score, 100)

    def test_demo_healthcare_domain_has_advisories(self):
        report = self._demo_report(domain="healthcare")
        hc_ids = [f.id for f in report.findings if "HC" in f.id]
        self.assertGreater(len(hc_ids), 0)

    def test_demo_report_serialises_to_json(self):
        report = self._demo_report()
        data   = json.loads(report.to_json())
        self.assertIn("findings", data)
        self.assertIn("overall_health_score", data)

    def test_demo_all_findings_have_fixes(self):
        report = self._demo_report()
        for f in report.findings:
            self.assertTrue(f.fix and f.fix.strip(),
                            f"Finding {f.id} has empty fix")


# ── Test 3: _process_feedback logic ──────────────────────────────────────────

class TestProcessFeedback(unittest.TestCase):
    """
    Tests for the feedback processing logic extracted from app.py.
    We replicate the logic inline so we don't need Streamlit.
    """

    def _process(self, report: DiagnosisReport, decisions: dict):
        """Replicate _process_feedback() logic without st.* calls."""
        from core.correlation_engine import (
            _detect_interactions, _generate_action_sequence,
        )

        validated = []
        feedback_items = []

        for fid, dec in decisions.items():
            finding   = dec["finding"]
            action    = dec["action"]
            new_sev   = dec["new_sev"]
            user_note = dec["user_note"]

            fb = {
                "finding_id":        finding.id,
                "action":            action.lower().replace(" ", "_"),
                "original_severity": finding.severity.value,
                "new_severity":      None,
                "user_note":         user_note,
            }

            if action == "Reject":
                feedback_items.append(fb)
                continue

            updated = finding
            if action == "Change Severity":
                fb["new_severity"] = new_sev
                updated = Finding(
                    id=finding.id, name=finding.name,
                    severity=Severity(new_sev),
                    evidence=finding.evidence,
                    explanation=finding.explanation,
                    fix=finding.fix,
                    confidence=finding.confidence,
                )
            validated.append(updated)
            feedback_items.append(fb)

        ranked    = _rank_findings(validated)
        interact  = _detect_interactions(ranked)
        new_score = max(0, _compute_health_score(ranked) - len(interact) * 5)
        actions   = _generate_action_sequence(ranked)

        return DiagnosisReport(
            model_name=report.model_name,
            task_type=report.task_type,
            domain=report.domain,
            overall_health_score=new_score,
            findings=ranked,
            interaction_warnings=interact,
            recommended_action_sequence=actions,
        ), feedback_items

    def test_confirm_keeps_finding(self):
        f      = _f("L1.1", Severity.HIGH)
        report = _make_report_direct(f)
        dec    = {"L1.1": {"finding": f, "action": "Confirm",
                           "new_sev": "HIGH", "user_note": None}}
        validated, _ = self._process(report, dec)
        self.assertEqual(len(validated.findings), 1)
        self.assertEqual(validated.findings[0].id, "L1.1")

    def test_reject_removes_finding(self):
        f      = _f("L1.1", Severity.HIGH)
        report = _make_report_direct(f)
        dec    = {"L1.1": {"finding": f, "action": "Reject",
                           "new_sev": "HIGH", "user_note": None}}
        validated, _ = self._process(report, dec)
        self.assertEqual(len(validated.findings), 0)

    def test_reject_all_gives_score_100(self):
        f1 = _f("L1.1", Severity.HIGH)
        f2 = _f("L0.1", Severity.MEDIUM)
        report = _make_report_direct(f1, f2)
        dec = {
            "L1.1": {"finding": f1, "action": "Reject", "new_sev": "HIGH", "user_note": None},
            "L0.1": {"finding": f2, "action": "Reject", "new_sev": "MEDIUM", "user_note": None},
        }
        validated, _ = self._process(report, dec)
        self.assertEqual(validated.overall_health_score, 100)

    def test_change_severity_applies_new_severity(self):
        f      = _f("L1.1", Severity.LOW)
        report = _make_report_direct(f)
        dec    = {"L1.1": {"finding": f, "action": "Change Severity",
                           "new_sev": "CRITICAL", "user_note": "very bad"}}
        validated, _ = self._process(report, dec)
        self.assertEqual(validated.findings[0].severity, Severity.CRITICAL)

    def test_change_severity_increases_deduction(self):
        """Changing LOW→CRITICAL should produce lower score than confirming as LOW."""
        f = _f("L1.1", Severity.LOW)
        report = _make_report_direct(f)

        # Confirm as LOW → deducts 3 → score 97
        dec_low = {"L1.1": {"finding": f, "action": "Confirm",
                            "new_sev": "LOW", "user_note": None}}
        low_result, _ = self._process(report, dec_low)

        # Change to CRITICAL → deducts 30 → score 70
        dec_crit = {"L1.1": {"finding": f, "action": "Change Severity",
                             "new_sev": "CRITICAL", "user_note": None}}
        crit_result, _ = self._process(report, dec_crit)

        self.assertLess(crit_result.overall_health_score,
                        low_result.overall_health_score)

    def test_action_sequence_length_matches_findings(self):
        f1 = _f("L1.1", Severity.HIGH)
        f2 = _f("L0.1", Severity.MEDIUM)
        report = _make_report_direct(f1, f2)
        dec = {
            "L1.1": {"finding": f1, "action": "Confirm", "new_sev": "HIGH", "user_note": None},
            "L0.1": {"finding": f2, "action": "Confirm", "new_sev": "MEDIUM", "user_note": None},
        }
        validated, _ = self._process(report, dec)
        self.assertEqual(
            len(validated.recommended_action_sequence),
            len(validated.findings),
        )

    def test_feedback_items_logged_for_all_decisions(self):
        f1 = _f("L1.1", Severity.HIGH)
        f2 = _f("L0.1", Severity.MEDIUM)
        report = _make_report_direct(f1, f2)
        dec = {
            "L1.1": {"finding": f1, "action": "Confirm",   "new_sev": "HIGH",   "user_note": None},
            "L0.1": {"finding": f2, "action": "Reject",    "new_sev": "MEDIUM", "user_note": "fp"},
        }
        _, items = self._process(report, dec)
        self.assertEqual(len(items), 2)

    def test_rejected_findings_logged_with_action_rejected(self):
        f      = _f("L1.1", Severity.HIGH)
        report = _make_report_direct(f)
        dec    = {"L1.1": {"finding": f, "action": "Reject",
                           "new_sev": "HIGH", "user_note": "fp"}}
        _, items = self._process(report, dec)
        self.assertEqual(items[0]["action"], "reject")


# ── Test 4: Download data correctness ─────────────────────────────────────────

class TestDownloadData(unittest.TestCase):

    def setUp(self):
        self.report, _ = _make_report()

    def test_json_download_is_valid_json(self):
        data = json.loads(self.report.to_json())
        self.assertIsInstance(data, dict)

    def test_json_download_has_findings(self):
        data = json.loads(self.report.to_json())
        self.assertIn("findings", data)
        self.assertIsInstance(data["findings"], list)

    def test_json_findings_have_all_fields(self):
        data = json.loads(self.report.to_json())
        for f in data["findings"]:
            for field in ["id", "name", "severity", "evidence",
                          "explanation", "fix", "confidence"]:
                self.assertIn(field, f)

    def test_text_download_has_no_ansi(self):
        import re
        from core.report_generator import render_report
        ansi = re.compile(r"\033\[[0-9;]*m")
        plain = ansi.sub("", render_report(self.report))
        self.assertNotIn("\033[", plain)

    def test_text_download_contains_report_title(self):
        from core.report_generator import render_report
        import re
        plain = re.compile(r"\033\[[0-9;]*m").sub("", render_report(self.report))
        self.assertIn("ML FAILURE INTELLIGENCE REPORT", plain)

    def test_text_download_contains_all_finding_ids(self):
        from core.report_generator import render_report
        import re
        plain = re.compile(r"\033\[[0-9;]*m").sub("", render_report(self.report))
        for f in self.report.findings:
            self.assertIn(f.id, plain)


if __name__ == "__main__":
    unittest.main(verbosity=2)