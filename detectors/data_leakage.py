"""
Detects two forms of data leakage:

  1. Feature-target correlation leakage
     Any feature with |correlation| > 0.95 with the target
     is suspicious — it likely contains target information.

  2. Suspiciously perfect test score
     If test accuracy / R² > 0.99 on a real-world dataset,
     the model has almost certainly seen the answers.

Failure code : L0.2
Severity      : Always CRITICAL — leakage invalidates all results.
"""

from __future__ import annotations
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score

from core.ingestion import ModelInput
from core.report import Finding, Severity
from core.registry import FailureTaxonomy


_CORRELATION_THRESHOLD   = 0.95
_PERFECT_SCORE_THRESHOLD = 0.99


def _numeric_features(X: pd.DataFrame) -> pd.DataFrame:
    return X.select_dtypes(include=[np.number])


def _feature_target_correlations(
    X: pd.DataFrame,
    y: pd.Series,
) -> List[Tuple[str, float]]:
    """
    Return list of (feature_name, correlation) pairs where
    |correlation| exceeds the leakage threshold.
    """
    X_num = _numeric_features(X)
    suspicious = []
    for col in X_num.columns:
        try:
            corr = float(X_num[col].corr(y))
            if not np.isnan(corr) and abs(corr) >= _CORRELATION_THRESHOLD:
                suspicious.append((col, round(corr, 4)))
        except Exception:
            pass
    return suspicious


def _test_score(model, X_test, y_test, task_type: str) -> float:
    preds = model.predict(X_test)
    if task_type == "classification":
        return float(accuracy_score(y_test, preds))
    return float(r2_score(y_test, preds))


def detect(model_input: ModelInput) -> Optional[Finding]:
    """
    Detect data leakage via feature-target correlation
    and suspiciously perfect performance.

    Returns a CRITICAL Finding if leakage is suspected, else None.
    """
    entry = FailureTaxonomy.get("L0.2")

    suspicious_features = _feature_target_correlations(
        model_input.X_train,
        model_input.y_train,
    )

    score = _test_score(
        model_input.model,
        model_input.X_test,
        model_input.y_test,
        model_input.task_type,
    )
    perfect_score = score >= _PERFECT_SCORE_THRESHOLD

    if not suspicious_features and not perfect_score:
        return None

    evidence: dict = {}

    if suspicious_features:
        evidence["suspicious_features"] = {
            feat: corr for feat, corr in suspicious_features
        }
        evidence["correlation_threshold"] = _CORRELATION_THRESHOLD

    if perfect_score:
        metric = "accuracy" if model_input.task_type == "classification" else "R²"
        evidence[f"test_{metric}"] = round(score, 4)
        evidence["perfect_score_threshold"] = _PERFECT_SCORE_THRESHOLD

    confidence = 0.97 if (suspicious_features and perfect_score) else (
        0.85 if suspicious_features else 0.75
    )

    parts = []
    if suspicious_features:
        feat_str = ", ".join(
            f"'{f}' (r={c})" for f, c in suspicious_features
        )
        parts.append(
            f"Features with near-perfect target correlation detected: {feat_str}. "
            "These features almost certainly contain or are derived from the target."
        )
    if perfect_score:
        parts.append(
            f"Test score of {score:.1%} is suspiciously high for a real-world dataset. "
            "This strongly suggests target information has leaked into features or the split."
        )
    explanation = "  ".join(parts)

    fix = (
        "1. Inspect suspicious features — remove any derived from the target variable.\n"
        "2. Audit your preprocessing pipeline: ensure all transformers "
        "(scalers, encoders, imputers) are fitted on TRAINING data only.\n"
        "3. Check your train/test split — ensure no test rows appear in training.\n"
        "4. For time-series data: always use temporal splits, never random splits.\n"
        "5. Re-examine feature engineering steps for any look-ahead bias."
    )

    return Finding(
        id=entry.code,
        name=entry.name,
        severity=Severity.CRITICAL,
        evidence=evidence,
        explanation=explanation,
        fix=fix,
        confidence=confidence,
        notes=(
            "Data leakage invalidates ALL performance metrics. "
            "Fix this before interpreting any results from this model."
        ),
    )