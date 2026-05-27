"""
ModelInput dataclass and load_model_input() entry point.

Responsibilities
----------------
- Accept a trained model + train/test DataFrames from the user
- Validate shapes, column alignment, task type, and domain
- Return a clean ModelInput ready for all detectors to consume
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import numpy as np
import pandas as pd


# Constants

VALID_TASK_TYPES = ("classification", "regression")
VALID_DOMAINS    = ("general", "finance", "healthcare", "nlp")


# ModelInput


@dataclass
class ModelInput:
    """
    Canonical input bundle consumed by every MLFIE detector.

    Parameters
    ----------
    model        : Any trained sklearn-compatible or XGBoost model
    X_train      : Training features as a DataFrame
    X_test       : Test features as a DataFrame
    y_train      : Training labels as a Series
    y_test       : Test labels as a Series
    task_type    : 'classification' or 'regression'
    domain       : 'general' | 'finance' | 'healthcare' | 'nlp'
    model_name   : Display name (auto-inferred if not provided)
    feature_names: Column names (auto-inferred from X_train if not provided)
    """
    model:         object
    X_train:       pd.DataFrame
    X_test:        pd.DataFrame
    y_train:       pd.Series
    y_test:        pd.Series
    task_type:     str
    domain:        str                   = "general"
    model_name:    str                   = ""
    feature_names: List[str]             = field(default_factory=list)

    #  Post-init: normalise + auto-fill optional fields
    def __post_init__(self):
        # Normalise strings
        self.task_type = self.task_type.lower().strip()
        self.domain    = self.domain.lower().strip()

        # Auto-fill model_name
        if not self.model_name:
            self.model_name = type(self.model).__name__

        # Auto-fill feature_names from DataFrame columns
        if not self.feature_names:
            self.feature_names = list(self.X_train.columns)

    #  Convenience properties
    @property
    def n_train(self) -> int:
        return len(self.X_train)

    @property
    def n_test(self) -> int:
        return len(self.X_test)

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    @property
    def classes(self) -> Optional[np.ndarray]:
        """Unique class labels (classification only)."""
        if self.task_type == "classification":
            return np.unique(self.y_train)
        return None

    @property
    def model_hash(self) -> str:
        """SHA-256 fingerprint of the serialised model (for feedback logs)."""
        return hashlib.sha256(pickle.dumps(self.model)).hexdigest()

    def __repr__(self) -> str:
        return (
            f"ModelInput("
            f"model={self.model_name}, "
            f"task={self.task_type}, "
            f"domain={self.domain}, "
            f"train={self.n_train} rows, "
            f"test={self.n_test} rows, "
            f"features={self.n_features})"
        )



# Validation helpers


class IngestionError(ValueError):
    """Raised when ModelInput validation fails."""


def _validate_task_type(task_type: str) -> None:
    if task_type not in VALID_TASK_TYPES:
        raise IngestionError(
            f"task_type must be one of {VALID_TASK_TYPES}, got '{task_type}'."
        )


def _validate_domain(domain: str) -> None:
    if domain not in VALID_DOMAINS:
        raise IngestionError(
            f"domain must be one of {VALID_DOMAINS}, got '{domain}'."
        )


def _validate_dataframe(df: object, name: str) -> pd.DataFrame:
    """Ensure input is a non-empty DataFrame."""
    if isinstance(df, np.ndarray):
        df = pd.DataFrame(df)
    if not isinstance(df, pd.DataFrame):
        raise IngestionError(
            f"'{name}' must be a pandas DataFrame (or numpy array), "
            f"got {type(df).__name__}."
        )
    if df.empty:
        raise IngestionError(f"'{name}' is empty.")
    return df


def _validate_series(series: object, name: str, df: pd.DataFrame) -> pd.Series:
    """Ensure input is a Series and length matches its paired DataFrame."""
    if isinstance(series, np.ndarray):
        series = pd.Series(series)
    if not isinstance(series, pd.Series):
        raise IngestionError(
            f"'{name}' must be a pandas Series (or 1-D numpy array), "
            f"got {type(series).__name__}."
        )
    if len(series) != len(df):
        raise IngestionError(
            f"Length mismatch: '{name}' has {len(series)} rows "
            f"but its paired DataFrame has {len(df)} rows."
        )
    return series


def _validate_column_alignment(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> None:
    """Train and test must share the same columns in the same order."""
    if list(X_train.columns) != list(X_test.columns):
        train_cols = set(X_train.columns)
        test_cols  = set(X_test.columns)
        only_train = train_cols - test_cols
        only_test  = test_cols - train_cols
        msg_parts  = ["X_train and X_test have different columns."]
        if only_train:
            msg_parts.append(f"Only in X_train: {sorted(only_train)}")
        if only_test:
            msg_parts.append(f"Only in X_test:  {sorted(only_test)}")
        raise IngestionError("  ".join(msg_parts))


def _validate_model_has_predict(model: object) -> None:
    if not hasattr(model, "predict"):
        raise IngestionError(
            f"Model '{type(model).__name__}' must implement a predict() method. "
            "Ensure it is a fitted sklearn-compatible or XGBoost model."
        )


def _validate_minimum_samples(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    min_train: int = 10,
    min_test:  int = 5,
) -> None:
    if len(X_train) < min_train:
        raise IngestionError(
            f"X_train has only {len(X_train)} rows; "
            f"minimum required is {min_train}."
        )
    if len(X_test) < min_test:
        raise IngestionError(
            f"X_test has only {len(X_test)} rows; "
            f"minimum required is {min_test}."
        )


# Public API

def load_model_input(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    task_type: str,
    domain:    str  = "general",
    model_name: str = "",
    feature_names: Optional[List[str]] = None,
) -> ModelInput:
    """
    Validate and package a trained model + datasets into a ModelInput.

    Parameters
    ----------
    model        : Fitted sklearn-compatible or XGBoost model
    X_train      : Training features (DataFrame or 2-D numpy array)
    X_test       : Test features (DataFrame or 2-D numpy array)
    y_train      : Training labels (Series or 1-D numpy array)
    y_test       : Test labels (Series or 1-D numpy array)
    task_type    : 'classification' or 'regression'
    domain       : 'general' | 'finance' | 'healthcare' | 'nlp'
    model_name   : Optional display name for the model
    feature_names: Optional list of feature names

    Returns
    -------
    ModelInput

    Raises
    ------
    IngestionError : If any validation check fails (informative message)
    """
    # 1. Normalise strings
    task_type = task_type.lower().strip()
    domain    = domain.lower().strip()

    # 2. Type & value checks
    _validate_task_type(task_type)
    _validate_domain(domain)
    _validate_model_has_predict(model)

    #  3. Coerce to DataFrame / Series
    X_train = _validate_dataframe(X_train, "X_train")
    X_test  = _validate_dataframe(X_test,  "X_test")
    y_train = _validate_series(y_train, "y_train", X_train)
    y_test  = _validate_series(y_test,  "y_test",  X_test)

    #  4. Column alignment
    _validate_column_alignment(X_train, X_test)

    #  5. Minimum sample check
    _validate_minimum_samples(X_train, X_test)

    # 6. Resolve feature names
    resolved_features = feature_names if feature_names else list(X_train.columns)

    #  7. Build and return
    return ModelInput(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        task_type=task_type,
        domain=domain,
        model_name=model_name,
        feature_names=resolved_features,
    )