"""
Detects when the primary evaluation metric (accuracy) gives a
misleading picture of model quality.

Two checks:
  1. Accuracy-F1 gap on imbalanced data
     Accuracy >> F1 means the model is ignoring minority classes.
  2. Zero-rate baseline comparison
     If model barely beats "always predict majority class",
     it has learned almost nothing meaningful.

Failure code : L3.1
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from core.ingestion import ModelInput
from core.report import Finding, Severity
from core.registry import FailureTaxonomy


_IMBALANCE_RATIO      = 0.30   # minority class below this → imbalanced task
_GAP_CRITICAL         = 0.25   # accuracy - F1 gap above this → CRITICAL
_GAP_HIGH             = 0.15   # gap above this → HIGH
_GAP_MEDIUM           = 0.08   # gap above this → MEDIUM
_BASELINE_MARGIN      = 0.03   # model must beat baseline by at least 3%


def _minority_ratio(y: pd.Series) -> float:
    counts = y.value_counts()
    return float(counts.min() / len(y))


def _zero_rate_baseline(y: pd.Series) -> float:
    """Accuracy of a classifier that always predicts the majority class."""
    return float(y.value_counts().max() / len(y))


def _gap_severity(gap: float) -> Optional[Severity]:
    if gap > _GAP_CRITICAL:
        return Severity.CRITICAL
    if gap > _GAP_HIGH:
        return Severity.HIGH
    if gap > _GAP_MEDIUM:
        return Severity.MEDIUM
    return None


def detect(model_input: ModelInput) -> Optional[Finding]:
    """
    Detect metric mismatch: accuracy being used on imbalanced data,
    or model barely beating the trivial baseline.

    Only runs for classification tasks.
    """
    if model_input.task_type != "classification":
        return None

    y_test  = model_input.y_test
    y_pred  = model_input.model.predict(model_input.X_test)

    accuracy    = float(accuracy_score(y_test, y_pred))
    f1          = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    gap         = accuracy - f1
    min_ratio   = _minority_ratio(model_input.y_train)
    baseline    = _zero_rate_baseline(model_input.y_test)
    beats_baseline_by = accuracy - baseline

    gap_severity = None
    if min_ratio < _IMBALANCE_RATIO and gap > _GAP_MEDIUM:
        gap_severity = _gap_severity(gap)

    weak_model = beats_baseline_by < _BASELINE_MARGIN

    if gap_severity is None and not weak_model:
        return None

    severity = gap_severity or Severity.MEDIUM

    entry = FailureTaxonomy.get("L3.1")

    evidence: dict = {
        "accuracy":              round(accuracy, 4),
        "weighted_f1":           round(f1, 4),
        "accuracy_f1_gap":       round(gap, 4),
        "minority_class_ratio":  round(min_ratio, 4),
        "zero_rate_baseline":    round(baseline, 4),
        "beats_baseline_by":     round(beats_baseline_by, 4),
    }
    if weak_model:
        evidence["warning"] = "Model barely beats trivial zero-rate baseline"

    parts = []
    if gap_severity is not None:
        parts.append(
            f"Accuracy ({accuracy:.1%}) is significantly higher than "
            f"weighted F1 ({f1:.1%}) — a gap of {gap:.1%}. "
            "On imbalanced data this means the model is mostly predicting "
            "the majority class and accuracy is hiding this."
        )
    if weak_model:
        parts.append(
            f"Model accuracy ({accuracy:.1%}) barely exceeds the trivial "
            f"zero-rate baseline ({baseline:.1%}) by only {beats_baseline_by:.1%}. "
            "A classifier that always predicts the majority class would perform similarly."
        )
    explanation = "  ".join(parts)

    fix = (
        "1. Replace accuracy with a task-appropriate metric:\n"
        "   → Binary classification: F1-score, AUC-ROC, PR-AUC\n"
        "   → Multi-class imbalanced: weighted F1 or macro F1\n"
        "2. Evaluate per-class performance using a confusion matrix.\n"
        "3. Set class_weight='balanced' to force the model to learn minority classes.\n"
        "4. Report both accuracy AND F1 to avoid misleading stakeholders."
    )

    return Finding(
        id=entry.code,
        name=entry.name,
        severity=severity,
        evidence=evidence,
        explanation=explanation,
        fix=fix,
        confidence=0.85 if gap_severity else 0.70,
        notes=(
            "Accuracy is not a reliable metric when classes are imbalanced. "
            "Always cross-check with F1, precision, recall."
        ),
    )