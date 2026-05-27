"""

Detects class imbalance in classification tasks by examining
the minority class frequency in the training labels.

Failure code : L0.1
Severity rules:
  minority ratio < 5%  → CRITICAL
  minority ratio < 10% → HIGH
  minority ratio < 20% → MEDIUM
  minority ratio < 30% → LOW
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd

from core.ingestion import ModelInput
from core.report import Finding, Severity
from core.registry import FailureTaxonomy


_THRESHOLDS = [
    (0.05, Severity.CRITICAL, 0.95),
    (0.10, Severity.HIGH,     0.90),
    (0.20, Severity.MEDIUM,   0.80),
    (0.30, Severity.LOW,      0.65),
]


def _class_distribution(y: pd.Series) -> dict:
    counts = y.value_counts()
    total  = len(y)
    return {str(cls): {"count": int(cnt), "ratio": round(cnt / total, 4)}
            for cls, cnt in counts.items()}


def _minority_ratio(y: pd.Series) -> float:
    counts = y.value_counts()
    return float(counts.min() / len(y))


def _severity_and_confidence(ratio: float):
    for threshold, severity, confidence in _THRESHOLDS:
        if ratio < threshold:
            return severity, confidence
    return None, None


def detect(model_input: ModelInput) -> Optional[Finding]:
    """
    Detect class imbalance in training labels.

    Only runs for classification tasks.
    Returns a Finding if minority class ratio is below 30%, else None.
    """
    if model_input.task_type != "classification":
        return None

    y = model_input.y_train
    n_classes = y.nunique()

    # Binary or multi-class check
    if n_classes < 2:
        return None   # single class is a data error — caught by data_quality

    minority_ratio = _minority_ratio(y)
    severity, confidence = _severity_and_confidence(minority_ratio)

    if severity is None:
        return None

    distribution = _class_distribution(y)
    entry = FailureTaxonomy.get("L0.1")

    minority_class = str(y.value_counts().idxmin())
    majority_class = str(y.value_counts().idxmax())
    majority_count = int(y.value_counts().max())
    minority_count = int(y.value_counts().min())
    imbalance_ratio = round(majority_count / minority_count, 1)

    fix = (
        f"1. Set class_weight='balanced' in your model "
        f"(sklearn models support this natively).\n"
        f"2. Use SMOTE oversampling: "
        f"from imblearn.over_sampling import SMOTE.\n"
        f"3. For tree models (XGBoost): set "
        f"scale_pos_weight={imbalance_ratio}.\n"
        f"4. Switch primary metric from accuracy to F1-score or AUC-ROC.\n"
        f"5. Consider undersampling the majority class if data is large."
    )

    return Finding(
        id=entry.code,
        name=entry.name,
        severity=severity,
        evidence={
            "minority_class":         minority_class,
            "minority_count":         minority_count,
            "minority_ratio":         round(minority_ratio, 4),
            "majority_class":         majority_class,
            "majority_count":         majority_count,
            "imbalance_ratio":        f"{imbalance_ratio}:1",
            "n_classes":              n_classes,
            "class_distribution":     distribution,
        },
        explanation=(
            f"The minority class '{minority_class}' represents only "
            f"{minority_ratio:.1%} of training data "
            f"({minority_count} vs {majority_count} samples — "
            f"{imbalance_ratio}:1 ratio). "
            "The model will likely ignore the minority class, producing "
            "misleadingly high accuracy while failing on the rare class."
        ),
        fix=fix,
        confidence=confidence,
        notes=(
            "In some domains (e.g. fraud detection) imbalance is expected. "
            "Review domain context before treating this as a defect."
        ) if minority_ratio >= 0.05 else None,
    )