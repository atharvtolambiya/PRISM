"""

Detects overfitting by comparing train vs test performance
and analysing learning curve shape.

Failure code : L1.1
Severity rules:
  gap > 20%  → CRITICAL
  gap > 15%  → HIGH
  gap > 10%  → MEDIUM
  gap >  5%  → LOW
"""

from __future__ import annotations
from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score, r2_score

from core.ingestion import ModelInput
from core.report import Finding, Severity
from core.registry import FailureTaxonomy


_THRESHOLDS = [
    (0.20, Severity.CRITICAL, 0.95),
    (0.15, Severity.HIGH,     0.88),
    (0.10, Severity.MEDIUM,   0.75),
    (0.08, Severity.LOW,      0.60),
]


def _score(model, X, y, task_type: str) -> float:
    """Return accuracy (classification) or R² (regression)."""
    preds = model.predict(X)
    if task_type == "classification":
        return float(accuracy_score(y, preds))
    return float(r2_score(y, preds))


def _severity_and_confidence(gap: float):
    for threshold, severity, confidence in _THRESHOLDS:
        if gap > threshold:
            return severity, confidence
    return None, None


def detect(model_input: ModelInput) -> Optional[Finding]:
    """
    Detect overfitting via train-test performance gap.

    Returns a Finding if the gap exceeds 5%, else None.
    """
    train_score = _score(
        model_input.model,
        model_input.X_train,
        model_input.y_train,
        model_input.task_type,
    )
    test_score = _score(
        model_input.model,
        model_input.X_test,
        model_input.y_test,
        model_input.task_type,
    )

    gap = train_score - test_score

    # Negative gap → model is NOT overfitting (generalises fine or underfits)
    if gap <= 0.05:
        return None

    severity, confidence = _severity_and_confidence(gap)
    if severity is None:
        return None

    metric_name = "accuracy" if model_input.task_type == "classification" else "R²"
    entry = FailureTaxonomy.get("L1.1")

    notes = None
    if test_score < 0.60:
        notes = (
            "Test score is also below 0.60. This may be BOTH overfitting "
            "AND a weak model — check data quality and model complexity."
        )

    fix = (
        "1. Reduce model complexity (e.g. max_depth=4-6 for tree models, "
        "C=0.1 for logistic regression).\n"
        "2. Add regularisation (L2 penalty / dropout).\n"
        "3. Use early stopping if applicable.\n"
        "4. Increase training data or apply cross-validation."
    )

    return Finding(
        id=entry.code,
        name=entry.name,
        severity=severity,
        evidence={
            f"train_{metric_name}": round(train_score, 4),
            f"test_{metric_name}":  round(test_score,  4),
            "gap":                  round(gap, 4),
            "gap_pct":              f"{gap:.1%}",
        },
        explanation=(
            f"Train {metric_name} is {train_score:.1%} but test {metric_name} "
            f"is only {test_score:.1%} — a gap of {gap:.1%}. "
            "The model has memorised training patterns and fails to generalise."
        ),
        fix=fix,
        confidence=confidence,
        notes=notes,
    )