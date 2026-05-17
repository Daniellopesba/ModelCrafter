"""Performance analysis — the second first-class output of fitting.

DESIGN.md §3.3 elevates :class:`PerformanceReport` to the same structural
status as :class:`~model_crafter.assumptions.AssumptionReport`: assumptions
answer "is the model valid?"; performance answers "is the model good?".
Both come back as values from named operations with rich ``__repr__``s.
"""

from __future__ import annotations

from model_crafter.performance.by_segment import (
    SegmentedPerformanceReport,
    performance_by_segment,
)
from model_crafter.performance.compare import (
    Comparison,
    DeLongResult,
    compare,
    delong_test,
)
from model_crafter.performance.over_time import (
    TemporalPerformanceReport,
    performance_over_time,
)
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
    "Comparison",
    "DeLongResult",
    "DiscriminationReport",
    "DistributionReport",
    "PerformanceReport",
    "SegmentedPerformanceReport",
    "StabilityReport",
    "TemporalPerformanceReport",
    "compare",
    "delong_test",
    "performance",
    "performance_by_segment",
    "performance_over_time",
]
