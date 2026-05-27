"""
Detects distributional differences between train and test sets
and classifies the shift type.

Detection methods
-----------------
Primary   : Kolmogorov-Smirnov test per feature (cross-platform stable)
Secondary : Population Stability Index (PSI) for severity grading
Label     : Proportion difference for binary classification label shift

Shift classification
--------------------
  Covariate Shift (L2.1) : features shifted, label distribution stable
  Label Shift     (L2.2) : label distribution shifted, features stable
  Concept Drift   (L2.3) : both features AND labels shifted

KS p-value thresholds (primary trigger)
  p < 0.001 -> CRITICAL
  p < 0.01  -> HIGH
  p < 0.05  -> MEDIUM

PSI thresholds (secondary, for evidence reporting)
  PSI > 0.20 -> significant
  PSI > 0.10 -> moderate
"""

from __future__ import annotations
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from core.ingestion import ModelInput
from core.report import Finding, Severity
from core.registry import FailureTaxonomy


_KS_CRITICAL = 0.001   # very strong evidence of shift
_KS_HIGH     = 0.01
_KS_MEDIUM   = 0.05
_PSI_SIGNIFICANT = 0.20
_PSI_MODERATE    = 0.10
_N_BINS          = 10
_EPSILON         = 1e-8

_LABEL_PROP_THRESHOLD = 0.15   # |P(Y)_train - P(Y)_test| above this




def _psi_single(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """
    Population Stability Index between two 1-D arrays.
    Returns 0.0 on any numerical failure (safe fallback).
    """
    try:
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.unique(np.percentile(expected, quantiles))
        if len(bin_edges) < 2:
            return 0.0

        exp_counts = np.histogram(expected, bins=bin_edges)[0].astype(float)
        act_counts = np.histogram(actual,   bins=bin_edges)[0].astype(float)

        # Guard: if actual has NO values in the bin range, PSI is very high
        # Cap at 5.0 to avoid inf
        if act_counts.sum() == 0:
            return 5.0

        exp_pct = (exp_counts + _EPSILON) / (exp_counts.sum() + _EPSILON)
        act_pct = (act_counts + _EPSILON) / (act_counts.sum() + _EPSILON)

        psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
        return round(psi, 4) if np.isfinite(psi) else 5.0
    except Exception:
        return 0.0



def _ks_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    """
    Two-sample KS test p-value.
    Small p-value = strong evidence the two samples come from different distributions.
    Returns 1.0 on failure (conservative — no false alarm).
    """
    try:
        if len(a) < 5 or len(b) < 5:
            return 1.0
        result = stats.ks_2samp(a, b)
        return float(result.pvalue)
    except Exception:
        return 1.0


def _compute_ks_map(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Dict[str, float]:
    """KS p-value for every numeric feature. Returns {col: p_value}."""
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    return {
        col: round(_ks_pvalue(
            X_train[col].dropna().values,
            X_test[col].dropna().values,
        ), 6)
        for col in numeric_cols
    }


def _compute_psi_map(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Dict[str, float]:
    """PSI for every numeric feature (evidence only)."""
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    return {
        col: _psi_single(
            X_train[col].dropna().values,
            X_test[col].dropna().values,
        )
        for col in numeric_cols
    }


def _label_proportion_diff(y_train: pd.Series, y_test: pd.Series) -> float:
    """Absolute difference in positive class rate between train and test."""
    try:
        return abs(float(y_train.mean()) - float(y_test.mean()))
    except Exception:
        return 0.0


def _label_ks_pvalue(y_train: pd.Series, y_test: pd.Series) -> float:
    """KS p-value on label distributions."""
    return _ks_pvalue(
        y_train.values.astype(float),
        y_test.values.astype(float),
    )



def _ks_severity(min_pvalue: float) -> Optional[Severity]:
    """Convert minimum KS p-value across features to severity."""
    if min_pvalue < _KS_CRITICAL:
        return Severity.CRITICAL
    if min_pvalue < _KS_HIGH:
        return Severity.HIGH
    if min_pvalue < _KS_MEDIUM:
        return Severity.MEDIUM
    return None



def _classify_shift_type(
    ks_map: Dict[str, float],
    label_prop_diff: float,
    label_pvalue: float,
) -> Tuple[str, str]:
    """
    Classify shift type using KS test results and label proportion difference.

    Decision logic:
      feature KS significant + label stable  -> Covariate Shift  L2.1
      feature KS stable      + label shifted -> Label Shift       L2.2
      both significant                       -> Concept Drift     L2.3
    """
    # Feature shift: any feature has KS p-value < threshold
    feature_shifted = any(p < _KS_MEDIUM for p in ks_map.values())

    # Label shift: proportion diff OR label KS significant
    label_shifted = (
        label_prop_diff >= _LABEL_PROP_THRESHOLD
        or label_pvalue < _KS_MEDIUM
    )

    if feature_shifted and not label_shifted:
        return "L2.1", "Covariate Shift"
    if label_shifted and not feature_shifted:
        return "L2.2", "Label Shift"
    if feature_shifted and label_shifted:
        return "L2.3", "Concept Drift"
    # Fallback (shouldn't reach — only called when something drifted)
    return "L2.1", "Covariate Shift"



def detect(model_input: ModelInput) -> Optional[Finding]:
    """
    Detect distribution shift between train and test sets.

    Uses KS test as the primary trigger (numerically stable across platforms).
    PSI is computed for evidence reporting only.

    Returns a Finding with shift type classified, or None if no significant
    shift is detected.
    """
    X_train = model_input.X_train
    X_test  = model_input.X_test
    y_train = model_input.y_train
    y_test  = model_input.y_test

    ks_map         = _compute_ks_map(X_train, X_test)
    label_pvalue   = _label_ks_pvalue(y_train, y_test)
    label_prop_diff = _label_proportion_diff(y_train, y_test)

    if not ks_map:
        return None


    min_feature_pvalue = min(ks_map.values()) if ks_map else 1.0

    label_triggers = (
        label_prop_diff >= _LABEL_PROP_THRESHOLD
        or label_pvalue < _KS_MEDIUM
    )

    effective_min_p = min_feature_pvalue
    if label_triggers:
        # Force at least HIGH severity when label shift is detected
        effective_min_p = min(effective_min_p, _KS_HIGH * 0.5)

    severity = _ks_severity(effective_min_p)
    if severity is None:
        return None

    code, shift_label = _classify_shift_type(ks_map, label_prop_diff, label_pvalue)
    entry = FailureTaxonomy.get(code)

    psi_map = _compute_psi_map(X_train, X_test)
    max_psi = max(psi_map.values()) if psi_map else 0.0

    significant_features = {
        col: round(p, 6)
        for col, p in ks_map.items()
        if p < _KS_MEDIUM
    }
    top_shifted = dict(
        sorted(significant_features.items(), key=lambda x: x[1])[:5]
    )

    evidence = {
        "shift_type":              shift_label,
        "taxonomy_code":           code,
        "min_feature_ks_pvalue":   round(min_feature_pvalue, 6),
        "label_ks_pvalue":         round(label_pvalue, 6),
        "label_proportion_diff":   round(label_prop_diff, 4),
        "max_feature_psi":         round(max_psi, 4),
        "features_shifted":        len(significant_features),
        "top_shifted_features_ks": top_shifted,
        "ks_significance_threshold": _KS_MEDIUM,
    }

    if code == "L2.1":
        fix = (
            "Covariate Shift — input feature distributions changed:\n"
            "1. Retrain model on more recent data that matches test distribution.\n"
            "2. Apply importance weighting to reweight training samples.\n"
            "3. Use domain adaptation or transfer learning techniques.\n"
            "4. Engineer features that are more invariant to distribution changes."
        )
    elif code == "L2.2":
        fix = (
            "Label Shift — class frequencies changed between train and test:\n"
            "1. Estimate new label priors from recent production data.\n"
            "2. Apply prior probability correction to model predictions.\n"
            "3. Recalibrate model using Platt scaling or isotonic regression.\n"
            "4. Collect labelled data from the new distribution and retrain."
        )
    else:
        fix = (
            "Concept Drift — the relationship P(Y|X) has fundamentally changed:\n"
            "1. Full model retraining on recent data is required.\n"
            "2. Implement a sliding window or time-weighted training strategy.\n"
            "3. Consider online learning for continuously changing environments.\n"
            "4. Add temporal features to help the model adapt over time."
        )

    confidence = 0.92 if severity == Severity.CRITICAL else 0.80

    return Finding(
        id=entry.code,
        name=f"{entry.name} ({shift_label})",
        severity=severity,
        evidence=evidence,
        explanation=(
            f"{shift_label} detected between train and test distributions. "
            f"Minimum KS p-value across features: {min_feature_pvalue:.4f} "
            f"(significance threshold: {_KS_MEDIUM}). "
            f"Label proportion difference: {label_prop_diff:.1%}. "
            f"{len(significant_features)} feature(s) show statistically "
            f"significant distributional change. "
            "The model was trained on data from a different distribution than "
            "it is being evaluated on — predictions cannot be trusted."
        ),
        fix=fix,
        confidence=confidence,
    )