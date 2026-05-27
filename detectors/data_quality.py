"""

Detects three classes of data quality issues:

  1. Null values         — columns with missing data
  2. Duplicate rows      — exact repeated rows in training set
  3. Constant columns    — zero-variance features (carry no information)

Failure code : L0.3
Severity depends on worst issue found across all checks.
"""

from __future__ import annotations
from typing import Optional, Dict, List

import numpy as np
import pandas as pd

from core.ingestion import ModelInput
from core.report import Finding, Severity
from core.registry import FailureTaxonomy


_NULL_CRITICAL = 0.30   # > 30% nulls in any column → CRITICAL
_NULL_HIGH     = 0.10   # > 10% nulls in any column → HIGH
_NULL_MEDIUM   = 0.01   # >  1% nulls in any column → MEDIUM

_DUP_HIGH   = 0.10   # > 10% duplicates → HIGH
_DUP_MEDIUM = 0.01   # >  1% duplicates → MEDIUM


def _null_report(X: pd.DataFrame) -> Dict[str, float]:
    """Return columns with any nulls as {col: null_ratio}."""
    null_ratios = X.isnull().mean()
    return {col: round(float(r), 4)
            for col, r in null_ratios.items() if r > 0}


def _null_severity(null_map: Dict[str, float]) -> Optional[Severity]:
    if not null_map:
        return None
    max_ratio = max(null_map.values())
    if max_ratio > _NULL_CRITICAL:
        return Severity.CRITICAL
    if max_ratio > _NULL_HIGH:
        return Severity.HIGH
    if max_ratio > _NULL_MEDIUM:
        return Severity.MEDIUM
    return None


def _duplicate_report(X: pd.DataFrame) -> Dict[str, object]:
    n_total = len(X)
    n_dups  = int(X.duplicated().sum())
    return {
        "duplicate_rows": n_dups,
        "total_rows":     n_total,
        "duplicate_ratio": round(n_dups / n_total, 4) if n_total else 0.0,
    }


def _duplicate_severity(dup_ratio: float) -> Optional[Severity]:
    if dup_ratio > _DUP_HIGH:
        return Severity.HIGH
    if dup_ratio > _DUP_MEDIUM:
        return Severity.MEDIUM
    return None


def _constant_columns(X: pd.DataFrame) -> List[str]:
    """Return columns with zero variance (all values identical)."""
    return [col for col in X.columns if X[col].nunique(dropna=False) <= 1]


def _worst_severity(*severities) -> Optional[Severity]:
    """Return the highest-priority severity from a list, ignoring Nones."""
    order = [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]
    for s in order:
        if s in severities:
            return s
    return None


def detect(model_input: ModelInput) -> Optional[Finding]:
    """
    Detect data quality issues: nulls, duplicates, constant columns.

    Returns a Finding summarising all issues found, or None if data is clean.
    """
    X = model_input.X_train

    null_map       = _null_report(X)
    dup_report     = _duplicate_report(X)
    constant_cols  = _constant_columns(X)

    null_sev   = _null_severity(null_map)
    dup_sev    = _duplicate_severity(dup_report["duplicate_ratio"])
    const_sev  = Severity.MEDIUM if constant_cols else None

    overall = _worst_severity(null_sev, dup_sev, const_sev)


    if overall is None:
        return None

    evidence: dict = {"issues_found": []}

    if null_map:
        evidence["issues_found"].append("null_values")
        evidence["null_columns"]      = null_map
        evidence["worst_null_ratio"]  = max(null_map.values())
        evidence["columns_with_nulls"] = len(null_map)

    if dup_report["duplicate_rows"] > 0:
        evidence["issues_found"].append("duplicate_rows")
        evidence.update(dup_report)

    if constant_cols:
        evidence["issues_found"].append("constant_columns")
        evidence["constant_columns"] = constant_cols

    parts = []
    if null_map:
        worst_col   = max(null_map, key=lambda c: null_map[c])
        worst_ratio = null_map[worst_col]
        parts.append(
            f"{len(null_map)} column(s) contain null values "
            f"(worst: '{worst_col}' at {worst_ratio:.1%})."
        )
    if dup_report["duplicate_rows"] > 0:
        parts.append(
            f"{dup_report['duplicate_rows']} duplicate rows found "
            f"({dup_report['duplicate_ratio']:.1%} of training data) — "
            "these cause inflated training scores and misleading CV results."
        )
    if constant_cols:
        parts.append(
            f"{len(constant_cols)} constant column(s) detected "
            f"({constant_cols}) — zero-variance features add noise and slow training."
        )
    explanation = "  ".join(parts)

    n_issues   = len(evidence["issues_found"])
    confidence = min(0.70 + (n_issues - 1) * 0.10, 0.95)

    fix = (
        "Null values:\n"
        "  → Impute with median/mode: SimpleImputer(strategy='median')\n"
        "  → Or drop columns with > 50% nulls.\n"
        "Duplicate rows:\n"
        "  → df.drop_duplicates(inplace=True) before train/test split.\n"
        "Constant columns:\n"
        "  → from sklearn.feature_selection import VarianceThreshold\n"
        "  → VarianceThreshold(threshold=0).fit_transform(X)"
    )

    return Finding(
        id=FailureTaxonomy.get("L0.3").code,
        name=FailureTaxonomy.get("L0.3").name,
        severity=overall,
        evidence=evidence,
        explanation=explanation,
        fix=fix,
        confidence=round(confidence, 2),
    )