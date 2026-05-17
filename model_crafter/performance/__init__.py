"""Performance analysis — the second first-class output of fitting.

DESIGN.md §3.3 elevates the :class:`PerformanceReport` to the same
structural status as :class:`~model_crafter.assumptions.AssumptionReport`:
a named operation (``mc.performance``) returns a value with a rich
``__repr__``, and the individual metric primitives stay accessible
underneath. Assumptions answer "is the model valid?"; performance
answers "is the model good?".
"""

from __future__ import annotations

from model_crafter.performance.report import (
    CalibrationReport,
    DiscriminationReport,
    DistributionReport,
    PerformanceReport,
    StabilityReport,
    performance,
)

__all__ = [
    "CalibrationReport",
    "DiscriminationReport",
    "DistributionReport",
    "PerformanceReport",
    "StabilityReport",
    "performance",
]
