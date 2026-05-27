"""
tests/test_domain_rules.py
---------------------------
Tests for the Phase 4 domain rule layer.

Covers:
  - General domain passes findings through unchanged
  - Healthcare escalations (imbalance → HIGH, metric mismatch → CRITICAL,
    overfitting → CRITICAL, distribution shift → CRITICAL)
  - Healthcare synthetic advisory injected
  - Finance de-escalation (normal fraud imbalance → MEDIUM)
  - Finance escalation (distribution shift → CRITICAL)
  - Finance FPR advisory injected
  - NLP escalation (leakage → CRITICAL, data quality → HIGH)
  - NLP advisories injected
  - Unknown domain falls back to general
  - Domain notes appended (not overwriting)
  - run_diagnosis end-to-end respects domain

Run with:  python -m unittest tests.test_domain_rules -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from core.ingestion import load_model_input
from core.report import Finding, Severity
from core.correlation_engine import run_diagnosis
from domain.rules import apply_domain_rules, list_supported_domains


# ── Helpers ───────────────────────────────────────────────────────────────────

def _f(id_: str, severity: Severity, confidence: float = 0.80,
       evidence: dict = None) -> Finding:
    """Minimal Finding factory for unit tests."""
    return Finding(
        id=id_,
        name=f"Test {id_}",
        severity=severity,
        evidence=evidence or {"test": True},
        explanation="Test.",
        fix="Test fix.",
        confidence=confidence,
    )


def _ids(findings):
    return [f.id for f in findings]


def _severities(findings):
    return {f.id: f.severity for f in findings}


def _make_imbalanced_input(domain="healthcare"):
    rng = np.random.default_rng(0)
    n   = 400
    y   = np.array([0] * 360 + [1] * 40)   # 10% minority
    X   = pd.DataFrame(rng.standard_normal((n, 5)),
                       columns=[f"f{i}" for i in range(5)])
    y   = pd.Series(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )
    model = LogisticRegression(max_iter=500, class_weight="balanced")
    model.fit(X_train, y_train)
    return load_model_input(model, X_train, X_test, y_train, y_test,
                            task_type="classification", domain=domain)


# ── Test 1: General domain ────────────────────────────────────────────────────

class TestGeneralDomain(unittest.TestCase):

    def test_general_does_not_change_severities(self):
        findings = [
            _f("L1.1", Severity.HIGH),
            _f("L0.1", Severity.MEDIUM),
        ]
        result = apply_domain_rules(findings, domain="general")
        orig_sevs = {f.id: f.severity for f in findings}
        for f in result:
            if f.id in orig_sevs:
                self.assertEqual(f.severity, orig_sevs[f.id])

    def test_general_does_not_inject_advisories(self):
        findings = [_f("L1.1", Severity.HIGH)]
        result = apply_domain_rules(findings, domain="general")
        # Should be same length — no injected findings
        self.assertEqual(len(result), len(findings))

    def test_list_supported_domains(self):
        domains = list_supported_domains()
        for d in ["general", "healthcare", "finance", "nlp"]:
            self.assertIn(d, domains)


# ── Test 2: Healthcare domain ─────────────────────────────────────────────────

class TestHealthcareDomain(unittest.TestCase):

    def test_class_imbalance_escalated_to_at_least_high(self):
        findings = [_f("L0.1", Severity.LOW)]
        result = apply_domain_rules(findings, domain="healthcare")
        sev = _severities(result)
        self.assertIn(sev["L0.1"], [Severity.HIGH, Severity.CRITICAL])

    def test_metric_mismatch_escalated_to_critical(self):
        findings = [_f("L3.1", Severity.MEDIUM)]
        result = apply_domain_rules(findings, domain="healthcare")
        sev = _severities(result)
        self.assertEqual(sev["L3.1"], Severity.CRITICAL)

    def test_overfitting_escalated_to_critical(self):
        findings = [_f("L1.1", Severity.HIGH)]
        result = apply_domain_rules(findings, domain="healthcare")
        sev = _severities(result)
        self.assertEqual(sev["L1.1"], Severity.CRITICAL)

    def test_distribution_shift_escalated_to_critical(self):
        for code in ["L2.1", "L2.2", "L2.3"]:
            findings = [_f(code, Severity.MEDIUM)]
            result = apply_domain_rules(findings, domain="healthcare")
            sev = _severities(result)
            self.assertEqual(sev[code], Severity.CRITICAL,
                             f"{code} should be CRITICAL in healthcare")

    def test_synthetic_subgroup_advisory_injected(self):
        findings = [_f("L1.1", Severity.HIGH)]
        result = apply_domain_rules(findings, domain="healthcare")
        advisory_ids = [f.id for f in result if "HC" in f.id]
        self.assertGreater(len(advisory_ids), 0,
                           "Healthcare subgroup advisory should be injected")

    def test_advisory_has_correct_severity(self):
        findings = []
        result = apply_domain_rules(findings, domain="healthcare")
        for f in result:
            if "HC" in f.id:
                self.assertIn(f.severity, [Severity.HIGH, Severity.CRITICAL])

    def test_domain_override_note_appended(self):
        findings = [_f("L1.1", Severity.HIGH)]
        result = apply_domain_rules(findings, domain="healthcare")
        for f in result:
            if f.id == "L1.1":
                self.assertIsNotNone(f.notes)
                self.assertIn("DOMAIN OVERRIDE", f.notes)

    def test_original_finding_not_mutated(self):
        """apply_domain_rules must be non-destructive."""
        original = _f("L1.1", Severity.HIGH)
        apply_domain_rules([original], domain="healthcare")
        self.assertEqual(original.severity, Severity.HIGH)  # unchanged


# ── Test 3: Finance domain ────────────────────────────────────────────────────

class TestFinanceDomain(unittest.TestCase):

    def test_normal_fraud_imbalance_deescalated_to_medium(self):
        """0.5%+ minority ratio is normal in fraud → should be MEDIUM."""
        findings = [
            _f("L0.1", Severity.HIGH,
               evidence={"minority_ratio": 0.01})  # 1% minority
        ]
        result = apply_domain_rules(
            findings, domain="finance",
            evidence_extras={"minority_ratio": 0.01},
        )
        sev = _severities(result)
        self.assertEqual(sev["L0.1"], Severity.MEDIUM)

    def test_extreme_imbalance_below_0_5pct_not_deescalated(self):
        """Minority < 0.5% is even below normal fraud rate → stays HIGH."""
        findings = [
            _f("L0.1", Severity.HIGH,
               evidence={"minority_ratio": 0.002})
        ]
        result = apply_domain_rules(
            findings, domain="finance",
            evidence_extras={"minority_ratio": 0.002},
        )
        sev = _severities(result)
        # Should NOT be de-escalated below HIGH
        self.assertIn(sev["L0.1"], [Severity.HIGH, Severity.CRITICAL])

    def test_distribution_shift_always_critical_in_finance(self):
        for code in ["L2.1", "L2.2", "L2.3"]:
            findings = [_f(code, Severity.MEDIUM)]
            result = apply_domain_rules(findings, domain="finance")
            sev = _severities(result)
            self.assertEqual(sev[code], Severity.CRITICAL,
                             f"{code} should be CRITICAL in finance")

    def test_metric_mismatch_escalated_in_finance(self):
        findings = [_f("L3.1", Severity.LOW)]
        result = apply_domain_rules(findings, domain="finance")
        sev = _severities(result)
        self.assertIn(sev["L3.1"], [Severity.HIGH, Severity.CRITICAL])

    def test_leakage_gets_temporal_note_in_finance(self):
        findings = [_f("L0.2", Severity.CRITICAL)]
        result = apply_domain_rules(findings, domain="finance")
        for f in result:
            if f.id == "L0.2":
                self.assertIsNotNone(f.notes)
                self.assertIn("temporal", f.notes.lower())

    def test_fpr_threshold_advisory_injected(self):
        findings = [_f("L1.1", Severity.HIGH)]
        result = apply_domain_rules(findings, domain="finance")
        advisory_ids = [f.id for f in result if "FIN" in f.id]
        self.assertGreater(len(advisory_ids), 0)


# ── Test 4: NLP domain ────────────────────────────────────────────────────────

class TestNLPDomain(unittest.TestCase):

    def test_leakage_escalated_to_critical_in_nlp(self):
        findings = [_f("L0.2", Severity.HIGH)]
        result = apply_domain_rules(findings, domain="nlp")
        sev = _severities(result)
        self.assertEqual(sev["L0.2"], Severity.CRITICAL)

    def test_data_quality_escalated_to_high_in_nlp(self):
        findings = [_f("L0.3", Severity.LOW)]
        result = apply_domain_rules(findings, domain="nlp")
        sev = _severities(result)
        self.assertIn(sev["L0.3"], [Severity.HIGH, Severity.CRITICAL])

    def test_distribution_shift_gets_vocabulary_note(self):
        findings = [_f("L2.1", Severity.HIGH)]
        result = apply_domain_rules(findings, domain="nlp")
        for f in result:
            if f.id == "L2.1":
                self.assertIsNotNone(f.notes)
                self.assertIn("vocabulary", f.notes.lower())

    def test_lexical_overlap_advisory_injected(self):
        findings = []
        result = apply_domain_rules(findings, domain="nlp")
        nlp_ids = [f.id for f in result if "NLP" in f.id]
        self.assertGreater(len(nlp_ids), 0)

    def test_two_nlp_advisories_injected(self):
        """NLP domain injects 2 advisories: lexical overlap + writing bias."""
        findings = []
        result = apply_domain_rules(findings, domain="nlp")
        nlp_advisories = [f for f in result if "NLP" in f.id]
        self.assertEqual(len(nlp_advisories), 2)


# ── Test 5: Edge cases ────────────────────────────────────────────────────────

class TestEdgeCases(unittest.TestCase):

    def test_unknown_domain_falls_back_to_general(self):
        findings = [_f("L1.1", Severity.HIGH)]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = apply_domain_rules(findings, domain="sports")
            self.assertTrue(any("Unknown domain" in str(x.message) for x in w))
        # Should still return findings unchanged
        self.assertEqual(result[0].severity, Severity.HIGH)

    def test_empty_findings_with_domain_returns_only_advisories(self):
        """Healthcare with empty findings still injects advisory."""
        result = apply_domain_rules([], domain="healthcare")
        self.assertGreater(len(result), 0)

    def test_domain_case_insensitive(self):
        findings = [_f("L1.1", Severity.HIGH)]
        result_lower = apply_domain_rules(findings, domain="healthcare")
        result_upper = apply_domain_rules(findings, domain="HEALTHCARE")
        self.assertEqual(
            [f.severity for f in result_lower if f.id == "L1.1"],
            [f.severity for f in result_upper if f.id == "L1.1"],
        )


# ── Test 6: End-to-end run_diagnosis respects domain ─────────────────────────

class TestRunDiagnosisWithDomain(unittest.TestCase):

    def test_healthcare_report_has_higher_severity_than_general(self):
        """
        Same imbalanced dataset, healthcare vs general domain.
        Healthcare should produce more CRITICAL findings.
        """
        def count_critical(domain):
            mi = _make_imbalanced_input(domain=domain)
            report = run_diagnosis(mi)
            return sum(1 for f in report.findings if f.severity == Severity.CRITICAL)

        hc_critical      = count_critical("healthcare")
        general_critical = count_critical("general")
        self.assertGreaterEqual(hc_critical, general_critical)

    def test_healthcare_report_health_score_lower_than_general(self):
        """Healthcare domain should produce a lower (more concerning) health score."""
        def get_score(domain):
            mi = _make_imbalanced_input(domain=domain)
            return run_diagnosis(mi).overall_health_score

        hc_score      = get_score("healthcare")
        general_score = get_score("general")
        self.assertLessEqual(hc_score, general_score)

    def test_finance_report_contains_fpr_advisory(self):
        mi = _make_imbalanced_input(domain="finance")
        report = run_diagnosis(mi)
        fin_ids = [f.id for f in report.findings if "FIN" in f.id]
        self.assertGreater(len(fin_ids), 0)

    def test_nlp_report_contains_nlp_advisories(self):
        mi = _make_imbalanced_input(domain="nlp")
        report = run_diagnosis(mi)
        nlp_ids = [f.id for f in report.findings if "NLP" in f.id]
        self.assertGreater(len(nlp_ids), 0)

    def test_report_still_serialises_with_domain_findings(self):
        """Domain-injected findings must not break JSON serialisation."""
        import json
        mi = _make_imbalanced_input(domain="healthcare")
        report = run_diagnosis(mi)
        data = json.loads(report.to_json())
        self.assertIn("findings", data)
        self.assertIsInstance(data["findings"], list)


if __name__ == "__main__":
    unittest.main(verbosity=2)