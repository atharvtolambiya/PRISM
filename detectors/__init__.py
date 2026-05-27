"""

Registry of all Phase 2 detectors.
run_all() executes every detector and returns the list of findings.
"""

from __future__ import annotations
from typing import List, Optional

from core.ingestion import ModelInput
from core.report import Finding

from detectors import (
    overfitting,
    class_imbalance,
    data_leakage,
    data_quality,
    distribution_shift,
    metric_mismatch,
)

# Ordered list — data checks before model checks
_DETECTORS = [
    data_quality.detect,
    data_leakage.detect,
    class_imbalance.detect,
    distribution_shift.detect,
    overfitting.detect,
    metric_mismatch.detect,
]


def run_all(model_input: ModelInput) -> List[Finding]:
    """
    Run every registered detector against the given ModelInput.

    Returns a flat list of all Finding objects (None results dropped).
    Detectors are independent — one failing does not block others.
    """
    findings: List[Finding] = []
    for detector in _DETECTORS:
        try:
            result: Optional[Finding] = detector(model_input)
            if result is not None:
                findings.append(result)
        except Exception as exc:
            import warnings
            warnings.warn(
                f"Detector '{detector.__module__}' raised an unexpected error "
                f"and was skipped: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    return findings