"""
After the correlation engine produces ranked findings, this layer
applies domain-specific overrides based on the user-declared context.

Supported domains
-----------------
  general    → no overrides (default)
  healthcare → patient-safety-driven rules (recall, rare disease)
  finance    → fraud-detection and regulatory rules
  nlp        → text-pipeline specific rules

How overrides work
------------------
Each domain rule can:
  1. Escalate a finding's severity (e.g. LOW → CRITICAL)
  2. De-escalate a finding's severity (e.g. HIGH → MEDIUM)
  3. Append a domain-specific note explaining the override
  4. Add a new domain-specific advisory Finding with no detector backing

All overrides are non-destructive — original evidence and fix text
are preserved; only severity and notes are modified.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import List, Callable, Dict, Optional

from core.report import Finding, Severity
from core.registry import FailureTaxonomy



def _override(
    finding: Finding,
    new_severity: Severity,
    reason: str,
) -> Finding:
    """
    Return a copy of `finding` with updated severity and an appended note.
    Original finding is never mutated.
    """
    existing_note = finding.notes or ""
    separator     = "  " if existing_note else ""
    new_note      = f"{existing_note}{separator}[DOMAIN OVERRIDE] {reason}"

    return Finding(
        id=finding.id,
        name=finding.name,
        severity=new_severity,
        evidence=finding.evidence,
        explanation=finding.explanation,
        fix=finding.fix,
        confidence=finding.confidence,
        notes=new_note,
    )


def _add_note(finding: Finding, note: str) -> Finding:
    """Append a domain advisory note without changing severity."""
    existing = finding.notes or ""
    separator = "  " if existing else ""
    return Finding(
        id=finding.id,
        name=finding.name,
        severity=finding.severity,
        evidence=finding.evidence,
        explanation=finding.explanation,
        fix=finding.fix,
        confidence=finding.confidence,
        notes=f"{existing}{separator}[DOMAIN NOTE] {note}",
    )


def _synthetic_finding(
    id_: str,
    name: str,
    severity: Severity,
    explanation: str,
    fix: str,
    confidence: float = 0.90,
    notes: Optional[str] = None,
) -> Finding:
    """
    Create a domain-injected advisory Finding that has no detector backing.
    These surface domain-specific concerns that statistical detectors
    cannot infer without context.
    """
    return Finding(
        id=id_,
        name=name,
        severity=severity,
        evidence={"source": "domain_rule_layer", "domain_injected": True},
        explanation=explanation,
        fix=fix,
        confidence=confidence,
        notes=notes,
    )



_SEVERITY_RANK = {
    Severity.CRITICAL: 0,
    Severity.HIGH:     1,
    Severity.MEDIUM:   2,
    Severity.LOW:      3,
}


def _is_more_severe(a: Severity, b: Severity) -> bool:
    return _SEVERITY_RANK[a] < _SEVERITY_RANK[b]


def _escalate_to(
    finding: Finding,
    target: Severity,
    reason: str,
) -> Finding:
    """Escalate severity only if target is more severe than current."""
    if _is_more_severe(target, finding.severity):
        return _override(finding, target, reason)
    return finding


def _deescalate_to(
    finding: Finding,
    target: Severity,
    reason: str,
) -> Finding:
    """De-escalate severity only if target is less severe than current."""
    if not _is_more_severe(target, finding.severity):
        return _override(finding, target, reason)
    return finding


def _apply_general(
    findings: List[Finding],
    task_type: str,
    evidence_extras: dict,
) -> List[Finding]:
    """General domain — no overrides. Pass through unchanged."""
    return findings


def _apply_healthcare(
    findings: List[Finding],
    task_type: str,
    evidence_extras: dict,
) -> List[Finding]:
    """
    Healthcare domain rules.

    Context: Medical diagnosis, clinical decision support, drug discovery.
    Stakes:  False Negatives (missing disease) >> False Positives (false alarm).

    Rules
    -----
    HC-1  Recall below 0.90 → any related finding escalates to CRITICAL
    HC-2  Class Imbalance (L0.1) → always at least HIGH
          (rare disease detection context — imbalance is clinically meaningful)
    HC-3  Metric Mismatch (L3.1) → always CRITICAL
          (accuracy is never acceptable as primary metric in clinical context)
    HC-4  Inject advisory: subgroup performance check reminder
    HC-5  Overfitting (L1.1) in healthcare → always CRITICAL
          (a model that memorises training patients fails on new patients)
    """
    adjusted = []
    injected_subgroup_advisory = False

    for f in findings:

        # HC-2: Class Imbalance → minimum HIGH
        if f.id == "L0.1":
            f = _escalate_to(
                f, Severity.HIGH,
                "Healthcare: Class imbalance in clinical data is never trivial. "
                "Rare disease detection requires the minority class to be learned. "
                "Minimum severity is HIGH regardless of ratio.",
            )

        # HC-3: Metric Mismatch → CRITICAL
        elif f.id == "L3.1":
            f = _escalate_to(
                f, Severity.CRITICAL,
                "Healthcare: Using accuracy as a primary metric is clinically "
                "unacceptable. A model that misses 10% of cancer cases kills patients. "
                "Sensitivity (recall) must be the primary metric. "
                "Regulatory bodies (FDA, CE) require per-class performance reporting.",
            )
            f = _add_note(
                f,
                "Required clinical metrics: Sensitivity ≥ 0.90, Specificity ≥ 0.85, "
                "AUC-ROC ≥ 0.90 as minimum thresholds for deployment consideration.",
            )

        # HC-5: Overfitting → CRITICAL
        elif f.id == "L1.1":
            f = _escalate_to(
                f, Severity.CRITICAL,
                "Healthcare: An overfit model has memorised patient records "
                "from training data and will fail silently on new patients — "
                "potentially causing misdiagnosis. This is a patient safety issue.",
            )

        # HC-1: Distribution Shift → CRITICAL (patient population changes are dangerous)
        elif f.id in ("L2.1", "L2.2", "L2.3"):
            f = _escalate_to(
                f, Severity.CRITICAL,
                "Healthcare: Distribution shift between training and deployment "
                "populations (e.g. different hospital, different demographic) "
                "invalidates clinical validation. FDA requires re-validation "
                "when input distribution changes significantly.",
            )

        adjusted.append(f)

    # HC-4: Inject subgroup performance advisory (always, for healthcare)
    adjusted.append(_synthetic_finding(
        id_="L3.1-HC",
        name="Subgroup Performance Audit Required (Healthcare)",
        severity=Severity.HIGH,
        explanation=(
            "Clinical AI systems must demonstrate equitable performance across "
            "demographic subgroups (age, sex, ethnicity, socioeconomic status). "
            "Aggregate metrics can hide dangerous disparities in subgroup performance."
        ),
        fix=(
            "1. Evaluate model performance separately for each demographic subgroup.\n"
            "2. Use fairness metrics: Equalised Odds, Demographic Parity.\n"
            "3. from fairlearn.metrics import MetricFrame — compute per-group metrics.\n"
            "4. Document subgroup results in your model card before deployment."
        ),
        confidence=0.95,
        notes="Injected by healthcare domain rules — applies to all clinical models.",
    ))

    return adjusted


def _apply_finance(
    findings: List[Finding],
    task_type: str,
    evidence_extras: dict,
) -> List[Finding]:
    """
    Finance domain rules.

    Context: Fraud detection, credit scoring, risk modelling, trading.
    Stakes:  Regulatory compliance, financial loss, model auditability.

    Rules
    -----
    FIN-1  Class Imbalance (L0.1) with minority > 0.5% → de-escalate to MEDIUM
           (fraud datasets are expected to be 0.1–2% positive — not a bug)
    FIN-2  Distribution Shift (L2.*) → always CRITICAL
           (fraud patterns change monthly; stale models miss new attack vectors)
    FIN-3  Data Leakage (L0.2) → add note about temporal leakage risk
    FIN-4  Inject advisory: threshold evaluation reminder
           (AUC alone is insufficient — evaluate at specific FPR thresholds)
    FIN-5  Metric Mismatch (L3.1) → escalate to HIGH + note on cost-sensitivity
    """
    adjusted = []
    minority_ratio = evidence_extras.get("minority_ratio", 1.0)

    for f in findings:

        # FIN-1: Class Imbalance — expected in fraud, de-escalate if ratio is "normal"
        if f.id == "L0.1":
            if minority_ratio >= 0.005:   # >= 0.5% is normal for fraud
                f = _deescalate_to(
                    f, Severity.MEDIUM,
                    f"Finance: Minority class ratio of {minority_ratio:.2%} is within "
                    "the normal range for fraud detection (0.1%–2% positive rate). "
                    "This is an expected characteristic of the domain, not a defect. "
                    "Ensure scale_pos_weight is set appropriately.",
                )
            else:
                f = _add_note(
                    f,
                    f"Finance: Minority ratio of {minority_ratio:.2%} is even below "
                    "typical fraud rates. Verify sampling strategy and data collection.",
                )

        # FIN-2: Distribution Shift → always CRITICAL in finance
        elif f.id in ("L2.1", "L2.2", "L2.3"):
            f = _escalate_to(
                f, Severity.CRITICAL,
                "Finance: Fraud patterns and market conditions change rapidly "
                "(monthly or faster). Distribution shift means your model is "
                "missing new attack vectors or market regimes. "
                "Regulatory frameworks (SR 11-7, Basel III) require documented "
                "model monitoring and timely recalibration.",
            )

        # FIN-3: Data Leakage → add temporal leakage note
        elif f.id == "L0.2":
            f = _add_note(
                f,
                "Finance: In transaction data, ensure no future information "
                "(post-transaction features, settlement status, chargeback flags) "
                "leaked into features. Temporal leakage is the most common "
                "finance-specific leakage pattern.",
            )

        # FIN-5: Metric Mismatch → HIGH with cost note
        elif f.id == "L3.1":
            f = _escalate_to(
                f, Severity.HIGH,
                "Finance: Model evaluation must account for the asymmetric cost "
                "of errors. Missing a $10M fraud ≠ missing a $100 fraud. "
                "Use cost-sensitive evaluation with a business-defined cost matrix.",
            )
            f = _add_note(
                f,
                "Evaluate at specific False Positive Rate thresholds "
                "(e.g. FPR ≤ 1%) rather than optimising overall AUC. "
                "Report Precision-Recall AUC for imbalanced finance tasks.",
            )

        adjusted.append(f)

    # FIN-4: Inject threshold evaluation advisory (always for finance)
    adjusted.append(_synthetic_finding(
        id_="L3.1-FIN",
        name="FPR-Threshold Evaluation Required (Finance)",
        severity=Severity.MEDIUM,
        explanation=(
            "AUC-ROC alone is insufficient for fraud detection models. "
            "Operational systems work at a fixed decision threshold, so "
            "performance at that specific FPR matters more than global AUC. "
            "A model with AUC=0.97 can still be operationally poor "
            "if Precision at your operating FPR is unacceptable."
        ),
        fix=(
            "1. Define your operational FPR target (e.g. 1% false alert rate).\n"
            "2. Evaluate Precision, Recall, and F1 at that specific threshold.\n"
            "3. from sklearn.metrics import precision_recall_curve — "
            "plot the full PR curve.\n"
            "4. Report: Precision@FPR=1%, Recall@FPR=1% in your model documentation."
        ),
        confidence=0.90,
        notes="Injected by finance domain rules — applies to all fraud/risk models.",
    ))

    return adjusted


def _apply_nlp(
    findings: List[Finding],
    task_type: str,
    evidence_extras: dict,
) -> List[Finding]:
    """
    NLP domain rules.

    Context: Text classification, sentiment analysis, NER, summarisation.
    Stakes:  Lexical overlap leakage, demographic bias, OOV degradation.

    Rules
    -----
    NLP-1  Data Leakage (L0.2) involving text → escalate to CRITICAL
           (train/test text overlap is the most common NLP leakage type)
    NLP-2  Data Quality (L0.3) on text columns → escalate to HIGH
           (nulls in text data are structural, not random)
    NLP-3  Distribution Shift (L2.*) → add note on vocabulary shift
    NLP-4  Inject advisory: lexical overlap check reminder
    NLP-5  Inject advisory: demographic writing style bias reminder
    """
    adjusted = []

    for f in findings:

        # NLP-1: Leakage → CRITICAL (text overlap is pervasive and easy to miss)
        if f.id == "L0.2":
            f = _escalate_to(
                f, Severity.CRITICAL,
                "NLP: Data leakage in text tasks often means train/test documents "
                "share overlapping sentences, paraphrases, or source articles. "
                "This inflates benchmark scores dramatically and produces models "
                "that fail on genuinely unseen text.",
            )
            f = _add_note(
                f,
                "Check for: near-duplicate documents (MinHash/LSH), "
                "shared source articles, data contamination from web crawls.",
            )

        # NLP-2: Data Quality on text → HIGH minimum
        elif f.id == "L0.3":
            f = _escalate_to(
                f, Severity.HIGH,
                "NLP: Null or empty text fields are structural failures — "
                "they indicate broken data ingestion, encoding errors, or "
                "scraping failures. Unlike numeric nulls, text nulls cannot "
                "be imputed meaningfully.",
            )

        # NLP-3: Distribution Shift → add vocabulary note
        elif f.id in ("L2.1", "L2.2", "L2.3"):
            f = _add_note(
                f,
                "NLP: Distribution shift in text tasks is often caused by "
                "domain shift (e.g. trained on news, tested on social media), "
                "temporal vocabulary shift (new slang, terminology), or "
                "out-of-vocabulary token explosion. "
                "Check OOV rate: tokens in test not seen during training.",
            )

        adjusted.append(f)

    # NLP-4: Inject lexical overlap advisory
    adjusted.append(_synthetic_finding(
        id_="L0.2-NLP",
        name="Lexical Overlap Audit Required (NLP)",
        severity=Severity.HIGH,
        explanation=(
            "Train/test lexical overlap is the most common form of NLP data "
            "leakage and is missed by standard correlation-based detectors. "
            "Even a 5% document overlap can inflate accuracy by 10–15 points "
            "on text classification benchmarks."
        ),
        fix=(
            "1. Compute character n-gram overlap between train and test documents.\n"
            "2. Use MinHash LSH for near-duplicate detection at scale:\n"
            "   from datasketch import MinHash, MinHashLSH\n"
            "3. Deduplicate at SOURCE level, not just document level.\n"
            "4. Report overlap statistics in your paper/report."
        ),
        confidence=0.88,
        notes="Injected by NLP domain rules — applies to all text classification tasks.",
    ))

    # NLP-5: Inject writing style bias advisory
    adjusted.append(_synthetic_finding(
        id_="L3.3-NLP",
        name="Demographic Writing Style Bias Check (NLP)",
        severity=Severity.MEDIUM,
        explanation=(
            "NLP models often learn demographic correlates of writing style "
            "(dialect, vocabulary, syntax) rather than task-relevant features. "
            "This causes disparate performance across user groups and "
            "introduces downstream fairness risks."
        ),
        fix=(
            "1. Evaluate model performance stratified by author demographics.\n"
            "2. Test on dialectally diverse text (AAE, non-native English, etc.).\n"
            "3. Use counterfactual data augmentation to reduce style dependence.\n"
            "4. Report performance variance across writing styles."
        ),
        confidence=0.80,
        notes="Injected by NLP domain rules.",
    ))

    return adjusted



_DOMAIN_HANDLERS: Dict[str, Callable] = {
    "general":    _apply_general,
    "healthcare": _apply_healthcare,
    "finance":    _apply_finance,
    "nlp":        _apply_nlp,
}



def apply_domain_rules(
    findings: List[Finding],
    domain: str,
    task_type: str = "classification",
    evidence_extras: Optional[dict] = None,
) -> List[Finding]:
    """
    Apply domain-specific severity overrides and inject domain advisories.

    This is Layer 3 of the MLFIE pipeline. It runs AFTER the correlation
    engine has ranked findings, and BEFORE the final report is assembled.

    Parameters
    ----------
    findings        : Ranked list of Finding objects from correlation engine
    domain          : One of 'general' | 'healthcare' | 'finance' | 'nlp'
    task_type       : 'classification' or 'regression'
    evidence_extras : Optional dict of extra context (e.g. minority_ratio)
                      used by domain rules to make finer-grained decisions

    Returns
    -------
    List[Finding] — adjusted findings (may include domain-injected advisories)

    Notes
    -----
    - Original Finding objects are never mutated (copies are returned)
    - Unknown domain falls back to 'general' with a warning
    - Domain-injected synthetic findings have id suffixes (e.g. 'L3.1-HC')
    """
    domain = domain.lower().strip()
    extras = evidence_extras or {}

    if domain not in _DOMAIN_HANDLERS:
        import warnings
        warnings.warn(
            f"Unknown domain '{domain}' — falling back to 'general'. "
            f"Valid domains: {list(_DOMAIN_HANDLERS.keys())}",
            UserWarning,
            stacklevel=2,
        )
        domain = "general"

    handler = _DOMAIN_HANDLERS[domain]
    adjusted = handler(findings, task_type, extras)

    return adjusted


def list_supported_domains() -> List[str]:
    """Return all supported domain names."""
    return list(_DOMAIN_HANDLERS.keys())