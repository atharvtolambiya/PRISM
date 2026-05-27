"""
Layer 2 — Cross-Signal Correlation Engine.
Responsibilities
----------------
1. Run all detectors and collect raw findings
2. Detect compound interactions between findings
3. Rank findings by severity + confidence
4. Compute overall health score (0-100)
5. Generate ordered recommended action sequence
6. Return a fully assembled DiagnosisReport

This is the orchestrator that ties all detectors together.
"""

from __future__ import annotations
from typing import List, Tuple, Optional

from core.ingestion import ModelInput
from core.report import Finding, DiagnosisReport, Severity
from detectors import run_all


# Interaction rule definitions

# Each rule is: (finding_id_A, finding_id_B, warning_message)
# Triggered when BOTH findings are present in the same diagnosis.

_INTERACTION_RULES: List[Tuple[str, str, str]] = [
    (
        "L1.1",  # Overfitting
        "L2.1",  # Covariate Shift
        "⚠  COMPOUNDING: Overfitting + Distribution Shift detected together. "
        "An overfit model is far more sensitive to distribution shift — "
        "small input changes cause large prediction swings. "
        "Fix overfitting (L1.1) FIRST, then retrain on updated data.",
    ),
    (
        "L1.1",  # Overfitting
        "L2.3",  # Concept Drift
        "⚠  COMPOUNDING: Overfitting + Concept Drift detected together. "
        "The model has memorised patterns that no longer exist in production. "
        "Full retraining on recent data is required — regularisation alone is insufficient.",
    ),
    (
        "L0.1",  # Class Imbalance
        "L3.1",  # Metric Mismatch
        "⚠  COMPOUNDING: Class Imbalance + Metric Mismatch detected together. "
        "Accuracy is hiding the model's failure on minority classes. "
        "Switch your primary metric to F1/AUC-ROC BEFORE retraining — "
        "otherwise you cannot measure whether your fix actually worked.",
    ),
    (
        "L0.2",  # Data Leakage
        "L1.1",  # Overfitting
        "⚠  COMPOUNDING: Data Leakage + Overfitting detected together. "
        "All performance metrics are invalid. "
        "Fix leakage (L0.2) first — overfitting diagnosis cannot be trusted "
        "until the data pipeline is clean.",
    ),
    (
        "L0.2",  # Data Leakage
        "L2.1",  # Covariate Shift
        "⚠  COMPOUNDING: Data Leakage + Distribution Shift detected together. "
        "Leakage may be artificially masking the true severity of distribution shift. "
        "Fix leakage (L0.2) first, then re-run diagnosis.",
    ),
    (
        "L0.3",  # Data Quality
        "L1.1",  # Overfitting
        "⚠  COMPOUNDING: Data Quality Issues + Overfitting detected together. "
        "Nulls and duplicates in training data amplify memorisation. "
        "Clean the data (L0.3) before adjusting model complexity.",
    ),
    (
        "L0.3",  # Data Quality
        "L2.1",  # Covariate Shift
        "⚠  COMPOUNDING: Data Quality Issues + Distribution Shift detected together. "
        "Missing values in train data that are absent in test (or vice versa) "
        "can appear as false distribution shift. "
        "Resolve data quality (L0.3) first to get a clean drift estimate.",
    ),
    (
        "L1.2",  # Underfitting (future detector — handled gracefully if absent)
        "L0.1",  # Class Imbalance
        "⚠  COMPOUNDING: Underfitting + Class Imbalance detected together. "
        "The model is too simple to distinguish minority class patterns. "
        "Increase model capacity AND apply class balancing simultaneously.",
    ),
]


# Severity ordering for sort key
_SEVERITY_RANK = {
    Severity.CRITICAL: 0,
    Severity.HIGH:     1,
    Severity.MEDIUM:   2,
    Severity.LOW:      3,
}

# Health score
def _compute_health_score(findings: List[Finding]) -> int:
    """
    Start at 100 and deduct points per finding severity.
    Each interaction warning adds an additional -5 penalty.
    Floor at 0.
    """
    score = 100
    for f in findings:
        score -= f.severity.score()
    return max(0, score)


# Interaction detection
def _detect_interactions(findings: List[Finding]) -> List[str]:
    """
    Check every interaction rule against the current finding set.
    Returns list of warning strings for pairs that both appear.
    """
    # Strip any suffix from compound finding names (e.g. "L2.1" from "L2.1 (Covariate Shift)")
    present_ids = set()
    for f in findings:
        # id field is always the clean taxonomy code e.g. "L2.1"
        present_ids.add(f.id)

    warnings = []
    for id_a, id_b, message in _INTERACTION_RULES:
        if id_a in present_ids and id_b in present_ids:
            warnings.append(message)
    return warnings


# Finding ranking
def _rank_findings(findings: List[Finding]) -> List[Finding]:
    """
    Sort findings:
      Primary   → severity (CRITICAL first)
      Secondary → confidence descending (more certain findings surface first)
    """
    return sorted(
        findings,
        key=lambda f: (_SEVERITY_RANK[f.severity], -f.confidence),
    )


# Action sequence generator
# Effort estimates per finding type (rough dev-hours)
_EFFORT_ESTIMATES = {
    "L0.1": "2–4 hrs",
    "L0.2": "4–8 hrs (audit full pipeline)",
    "L0.3": "1–2 hrs",
    "L1.1": "1–3 hrs",
    "L1.2": "2–4 hrs",
    "L1.3": "2–6 hrs",
    "L2.1": "1–2 days (data collection)",
    "L2.2": "4–6 hrs",
    "L2.3": "1–3 days (full retrain)",
    "L3.1": "30 mins",
    "L3.2": "1–2 hrs",
    "L3.3": "2–4 hrs",
}


def _generate_action_sequence(ranked_findings: List[Finding]) -> List[str]:
    """
    Convert ranked findings into a numbered, actionable fix sequence.
    Each step includes estimated effort and references the finding ID.
    """
    steps = []
    for i, finding in enumerate(ranked_findings, start=1):
        effort = _EFFORT_ESTIMATES.get(finding.id, "~2 hrs")
        # Extract first line of fix as the headline action
        fix_headline = finding.fix.split("\n")[0].strip()
        steps.append(
            f"Step {i} [{finding.id}] → {fix_headline}  (~{effort})"
        )
    return steps


# Public API
def run_diagnosis(model_input: ModelInput) -> DiagnosisReport:
    """
    Full MLFIE Layer 1 + Layer 2 + Layer 3 pipeline.

    Steps
    -----
    1. Run all detectors          (Layer 1 - statistical signals)
    2. Apply domain rule layer    (Layer 3 - severity overrides + advisories)
    3. Detect interactions        (Layer 2 - cross-signal correlation)
    4. Rank findings by severity + confidence
    5. Compute health score
    6. Generate action sequence
    7. Return DiagnosisReport
    """
    from domain.rules import apply_domain_rules

    # Step 1: Run all detectors
    raw_findings: List[Finding] = run_all(model_input)

    # Step 2: Apply domain rules (Layer 3)
    minority_ratio = 1.0
    for f in raw_findings:
        if f.id == "L0.1" and "minority_ratio" in f.evidence:
            minority_ratio = float(f.evidence["minority_ratio"])

    domain_findings = apply_domain_rules(
        findings=raw_findings,
        domain=model_input.domain,
        task_type=model_input.task_type,
        evidence_extras={"minority_ratio": minority_ratio},
    )

    # Step 3: Detect interactions
    interaction_warnings = _detect_interactions(domain_findings)

    # Step 4: Rank
    ranked = _rank_findings(domain_findings)

    # Step 5: Health score
    health_score = _compute_health_score(ranked)
    health_score = max(0, health_score - len(interaction_warnings) * 5)

    # Step 6: Action sequence
    action_sequence = _generate_action_sequence(ranked)

    # Step 7: Assemble report
    return DiagnosisReport(
        model_name=model_input.model_name,
        task_type=model_input.task_type,
        domain=model_input.domain,
        overall_health_score=health_score,
        findings=ranked,
        interaction_warnings=interaction_warnings,
        recommended_action_sequence=action_sequence,
    )