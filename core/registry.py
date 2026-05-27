"""
Central registry of all failure taxonomy codes used by MLFIE.
Detectors reference these codes when creating Finding objects.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class TaxonomyEntry:
    code:        str
    name:        str
    layer:       str
    description: str


class FailureTaxonomy:
    """
    Taxonomy of failure types supported in MLFIE v1.

    Layers
    ------
    L0  Data Failures
    L1  Model-Data Fit Failures
    L2  Distribution Failures
    L3  Evaluation Failures
    """

    _entries: Dict[str, TaxonomyEntry] = {
        # ── L0 Data Failures
        "L0.1": TaxonomyEntry(
            code="L0.1",
            name="Class Imbalance",
            layer="L0",
            description="Minority class is severely underrepresented, "
                        "causing the model to ignore it.",
        ),
        "L0.2": TaxonomyEntry(
            code="L0.2",
            name="Data Leakage",
            layer="L0",
            description="Future or target-correlated information has leaked "
                        "into features, inflating apparent performance.",
        ),
        "L0.3": TaxonomyEntry(
            code="L0.3",
            name="Data Quality Issues",
            layer="L0",
            description="Null values, duplicates, or zero-variance columns "
                        "degrading model signal.",
        ),

        # ── L1 Model-Data Fit Failures
        "L1.1": TaxonomyEntry(
            code="L1.1",
            name="Overfitting",
            layer="L1",
            description="Model has memorised training data and fails to "
                        "generalise to unseen examples.",
        ),
        "L1.2": TaxonomyEntry(
            code="L1.2",
            name="Underfitting",
            layer="L1",
            description="Model is too simple to capture underlying patterns; "
                        "both train and test performance are poor.",
        ),
        "L1.3": TaxonomyEntry(
            code="L1.3",
            name="Model-Complexity Mismatch",
            layer="L1",
            description="Model capacity is misaligned with dataset size or "
                        "feature complexity.",
        ),

        # ── L2 Distribution Failures
        "L2.1": TaxonomyEntry(
            code="L2.1",
            name="Covariate Shift",
            layer="L2",
            description="Input feature distributions differ between train "
                        "and test while the label relationship is stable.",
        ),
        "L2.2": TaxonomyEntry(
            code="L2.2",
            name="Label Shift",
            layer="L2",
            description="Label distribution has changed while the conditional "
                        "P(X|Y) relationship remains the same.",
        ),
        "L2.3": TaxonomyEntry(
            code="L2.3",
            name="Concept Drift",
            layer="L2",
            description="The underlying relationship P(Y|X) has changed — "
                        "same inputs now map to different outputs.",
        ),

        # ── L3 Evaluation Failures
        "L3.1": TaxonomyEntry(
            code="L3.1",
            name="Metric Mismatch",
            layer="L3",
            description="The chosen evaluation metric does not reflect "
                        "true model quality for this task (e.g., accuracy "
                        "on imbalanced data).",
        ),
        "L3.2": TaxonomyEntry(
            code="L3.2",
            name="Validation Design Failure",
            layer="L3",
            description="Train/test split or cross-validation strategy "
                        "allows information leakage or unrepresentative splits.",
        ),
        "L3.3": TaxonomyEntry(
            code="L3.3",
            name="Model Miscalibration",
            layer="L3",
            description="Predicted probabilities are poorly calibrated — "
                        "confidence scores do not match actual accuracy.",
        ),
    }

    @classmethod
    def get(cls, code: str) -> TaxonomyEntry:
        if code not in cls._entries:
            raise KeyError(f"Unknown taxonomy code: '{code}'. "
                           f"Valid codes: {list(cls._entries.keys())}")
        return cls._entries[code]

    @classmethod
    def all_codes(cls) -> list:
        return list(cls._entries.keys())

    @classmethod
    def by_layer(cls, layer: str) -> list:
        return [e for e in cls._entries.values() if e.layer == layer]