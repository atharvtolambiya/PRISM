"""
FailureInjector — creates datasets with precisely controlled,
known failures so the engine's detection accuracy can be measured.

Each inject_* method:
  1. Takes a clean dataset
  2. Applies a specific, measurable failure
  3. Returns (modified_dataset, ground_truth_label)

Ground truth label uses the taxonomy codes defined in core/registry.py
so results can be matched against engine findings directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split



@dataclass
class InjectedBundle:
    """
    Everything needed to run the engine on an injected dataset.

    Attributes
    ----------
    model            : Fitted model (may be intentionally bad)
    X_train          : Training features (may contain injected flaws)
    X_test           : Test features   (may contain injected flaws)
    y_train          : Training labels  (may contain injected flaws)
    y_test           : Test labels
    task_type        : 'classification' or 'regression'
    injected_failure : Taxonomy code(s) of what was injected, e.g. "L1.1"
                       Multiple failures separated by comma: "L1.1,L0.1"
    description      : Human-readable description of the injection
    severity_hint    : Expected severity of the primary injected failure
    """
    model:             object
    X_train:           pd.DataFrame
    X_test:            pd.DataFrame
    y_train:           pd.Series
    y_test:            pd.Series
    task_type:         str
    injected_failure:  str
    description:       str
    severity_hint:     str = "HIGH"


def _make_clean_clf(
    n_samples: int = 400,
    n_features: int = 6,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=4,
        n_redundant=1,
        random_state=random_state,
    )
    cols = [f"feat_{i}" for i in range(n_features)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="target")


def _make_clean_reg(
    n_samples: int = 400,
    n_features: int = 5,
    random_state: int = 0,
) -> Tuple[pd.DataFrame, pd.Series]:
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=10,
        random_state=random_state,
    )
    cols = [f"feat_{i}" for i in range(n_features)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="target")


class FailureInjector:
    """
    Injects specific, controlled ML failures into clean datasets.

    Usage
    -----
        injector = FailureInjector(random_state=42)
        bundle   = injector.inject_overfitting()
        # bundle.injected_failure == "L1.1"
        # Run engine on bundle → check if L1.1 is detected
    """

    def __init__(self, random_state: int = 42):
        self.rng = np.random.default_rng(random_state)
        self.random_state = random_state


    def inject_overfitting(
        self,
        noise_ratio: float = 0.30,
        n_samples: int = 400,
    ) -> InjectedBundle:
        """
        Inject overfitting by adding label noise to training data
        and using an unlimited depth decision tree that memorises it.

        The tree achieves ~100% train accuracy on noisy labels but
        generalises poorly → large train-test gap.
        """
        X, y = _make_clean_clf(n_samples=n_samples,
                               random_state=self.random_state)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=self.random_state
        )
        # Flip `noise_ratio` fraction of training labels
        n_flip   = int(len(y_train) * noise_ratio)
        flip_idx = self.rng.choice(len(y_train), size=n_flip, replace=False)
        y_noisy  = y_train.copy()
        y_noisy.iloc[flip_idx] = 1 - y_noisy.iloc[flip_idx]

        model = DecisionTreeClassifier(max_depth=None,
                                       random_state=self.random_state)
        model.fit(X_train, y_noisy)

        return InjectedBundle(
            model=model,
            X_train=X_train, X_test=X_test,
            y_train=y_noisy, y_test=y_test,
            task_type="classification",
            injected_failure="L1.1",
            description=f"Overfitting: unlimited tree + {noise_ratio:.0%} label noise",
            severity_hint="CRITICAL",
        )


    def inject_class_imbalance(
        self,
        minority_ratio: float = 0.05,
        n_samples: int = 500,
    ) -> InjectedBundle:
        """
        Inject class imbalance by constructing a dataset where the
        minority class represents `minority_ratio` of the total.
        """
        n_minority = max(5, int(n_samples * minority_ratio))
        n_majority = n_samples - n_minority

        X_maj = self.rng.standard_normal((n_majority, 6))
        X_min = self.rng.standard_normal((n_minority, 6)) + 1.5  # separable
        X     = np.vstack([X_maj, X_min])
        y     = np.array([0] * n_majority + [1] * n_minority)

        cols  = [f"feat_{i}" for i in range(6)]
        X_df  = pd.DataFrame(X, columns=cols)
        y_s   = pd.Series(y, name="target")

        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y_s, test_size=0.25,
            random_state=self.random_state, stratify=y_s
        )
        model = LogisticRegression(max_iter=500)
        model.fit(X_train, y_train)

        sev = "CRITICAL" if minority_ratio < 0.05 else "HIGH"
        return InjectedBundle(
            model=model,
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
            task_type="classification",
            injected_failure="L0.1",
            description=f"Class imbalance: {minority_ratio:.0%} minority ratio",
            severity_hint=sev,
        )


    def inject_data_leakage(
        self,
        feature_idx: int = 0,
        n_samples: int = 400,
    ) -> InjectedBundle:
        """
        Inject leakage by adding the target variable as a feature.
        Correlation of that feature with the target = 1.0.
        """
        X, y = _make_clean_clf(n_samples=n_samples,
                               random_state=self.random_state)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=self.random_state
        )
        # Add target as a direct feature → perfect leakage
        X_train_leaked = X_train.copy()
        X_test_leaked  = X_test.copy()
        X_train_leaked["leaked_target"] = y_train.values
        X_test_leaked["leaked_target"]  = y_test.values

        model = LogisticRegression(max_iter=500)
        model.fit(X_train_leaked, y_train)

        return InjectedBundle(
            model=model,
            X_train=X_train_leaked, X_test=X_test_leaked,
            y_train=y_train, y_test=y_test,
            task_type="classification",
            injected_failure="L0.2",
            description="Data leakage: target variable added as a feature",
            severity_hint="CRITICAL",
        )


    def inject_data_quality(
        self,
        null_ratio: float = 0.35,
        add_duplicates: bool = True,
        add_constant_col: bool = True,
        n_samples: int = 400,
    ) -> InjectedBundle:
        """
        Inject data quality issues: nulls, duplicate rows, constant column.
        """
        X, y = _make_clean_clf(n_samples=n_samples,
                               random_state=self.random_state)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=self.random_state
        )
        X_dirty = X_train.copy()

        # Nulls in first column
        null_idx = X_dirty.sample(
            frac=null_ratio, random_state=self.random_state
        ).index
        X_dirty.loc[null_idx, "feat_0"] = np.nan

        # Duplicate rows
        if add_duplicates:
            n_dups  = max(1, int(len(X_dirty) * 0.15))
            X_dirty = pd.concat(
                [X_dirty, X_dirty.iloc[:n_dups]], ignore_index=True
            )
            y_train = pd.concat(
                [y_train, y_train.iloc[:n_dups]], ignore_index=True
            )

        # Constant column
        if add_constant_col:
            X_dirty["constant_col"] = 99.0
            X_test = X_test.copy()
            X_test["constant_col"] = 99.0

        # Train on filled data (model needs to work despite quality issues)
        model = LogisticRegression(max_iter=500)
        model.fit(X_dirty.fillna(X_dirty.median(numeric_only=True)), y_train)

        return InjectedBundle(
            model=model,
            X_train=X_dirty, X_test=X_test,
            y_train=y_train, y_test=y_test,
            task_type="classification",
            injected_failure="L0.3",
            description=(
                f"Data quality: {null_ratio:.0%} nulls, duplicates, constant column"
            ),
            severity_hint="CRITICAL",
        )


    def inject_distribution_shift(
        self,
        shift_magnitude: float = 8.0,
        n_samples: int = 500,
    ) -> InjectedBundle:
        """
        Inject covariate shift by scaling test features by a large factor,
        moving them far outside the training distribution.
        PSI will be very high; label distribution stays the same.
        """
        X, y = _make_clean_clf(n_samples=n_samples,
                               random_state=self.random_state)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=self.random_state
        )
        # Massively shift test feature distributions
        X_test_shifted = X_test.copy() * shift_magnitude + (shift_magnitude * 3)

        model = LogisticRegression(max_iter=500)
        model.fit(X_train, y_train)

        return InjectedBundle(
            model=model,
            X_train=X_train, X_test=X_test_shifted,
            y_train=y_train, y_test=y_test,
            task_type="classification",
            injected_failure="L2.1",
            description=(
                f"Covariate shift: test features scaled by {shift_magnitude}×"
            ),
            severity_hint="CRITICAL",
        )


    def inject_label_shift(
        self,
        shift_ratio: float = 0.40,
        n_samples: int = 500,
    ) -> InjectedBundle:
        """
        Inject label shift by flipping a large fraction of test labels,
        while keeping feature distributions the same.
        Label PSI will be high; feature PSI stays low.
        """
        X, y = _make_clean_clf(n_samples=n_samples,
                               random_state=self.random_state)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=self.random_state
        )
        # Asymmetric flip: flip class-0 samples to class-1 in test
        # This genuinely shifts P(Y) distribution (not symmetric cancel-out)
        y_shifted = y_test.copy()
        class0_idx = y_shifted[y_shifted == 0].index[:int(len(y_shifted) * shift_ratio)]
        y_shifted.loc[class0_idx] = 1

        model = LogisticRegression(max_iter=500)
        model.fit(X_train, y_train)

        return InjectedBundle(
            model=model,
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_shifted,
            task_type="classification",
            injected_failure="L2.2",
            description=(
                f"Label shift: {shift_ratio:.0%} of test labels flipped"
            ),
            severity_hint="HIGH",
        )


    def inject_metric_mismatch(
        self,
        minority_ratio: float = 0.04,
        n_samples: int = 500,
    ) -> InjectedBundle:
        """
        Inject metric mismatch: severe imbalance + model trained without
        class weighting → high accuracy, near-zero minority recall.
        """
        n_minority = max(5, int(n_samples * minority_ratio))
        n_majority = n_samples - n_minority

        X_maj = self.rng.standard_normal((n_majority, 5))
        X_min = self.rng.standard_normal((n_minority, 5)) + 0.3  # hard to separate
        X     = np.vstack([X_maj, X_min])
        y     = np.array([0] * n_majority + [1] * n_minority)

        cols  = [f"feat_{i}" for i in range(5)]
        X_df  = pd.DataFrame(X, columns=cols)
        y_s   = pd.Series(y, name="target")

        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y_s, test_size=0.25,
            random_state=self.random_state, stratify=y_s
        )
        # Deliberately no class_weight → model ignores minority
        model = LogisticRegression(max_iter=500, class_weight=None)
        model.fit(X_train, y_train)

        return InjectedBundle(
            model=model,
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
            task_type="classification",
            injected_failure="L3.1",
            description=(
                f"Metric mismatch: {minority_ratio:.0%} imbalance, no class weighting"
            ),
            severity_hint="HIGH",
        )


    def make_clean_baseline(self, n_samples: int = 400) -> InjectedBundle:
        """
        A clean, well-behaved model with no injected failures.
        Used to measure false alarm rate.
        """
        X, y = _make_clean_clf(n_samples=n_samples,
                               random_state=self.random_state)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=self.random_state
        )
        model = LogisticRegression(max_iter=500, C=1.0,
                                   class_weight="balanced")
        model.fit(X_train, y_train)

        return InjectedBundle(
            model=model,
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
            task_type="classification",
            injected_failure="NONE",
            description="Clean baseline: no injected failures",
            severity_hint="NONE",
        )