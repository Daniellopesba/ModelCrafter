"""Metric primitives for ``model_crafter`` (DESIGN.md §3.3).

Every primitive accepts ``(sol, data, weights=...)`` (with the exception
of :func:`psi`, which takes two raw distributions instead of a fitted
solution — DESIGN.md §3.3) and returns a frozen-dataclass result type
with a rich ``__repr__`` and ``__float__`` for use in numeric contexts.

Sub-modules

* :mod:`.classification` — AUC, Gini, KS, Cohen's d, DeLong test
* :mod:`.calibration` — Brier, ECE, calibration curve, log-loss,
  slope/intercept
* :mod:`.stability` — PSI
* :mod:`.rank` — lift table, cumulative gains
"""

from __future__ import annotations

from model_crafter.metrics.calibration import (
    BrierResult,
    CalibrationCurve,
    CalibrationFit,
    ECEResult,
    LogLossResult,
    brier_score,
    calibration_curve,
    calibration_slope_intercept,
    ece,
    log_loss,
)
from model_crafter.metrics.classification import (
    AUCResult,
    CohensDResult,
    DeLongResult,
    GiniResult,
    KSResult,
    auc,
    cohens_d,
    delong_test,
    gini,
    ks,
)
from model_crafter.metrics.rank import (
    GainsCurve,
    LiftTable,
    cumulative_gains,
    lift_table,
)
from model_crafter.metrics.stability import PSIResult, psi

__all__ = [
    "AUCResult",
    "BrierResult",
    "CalibrationCurve",
    "CalibrationFit",
    "CohensDResult",
    "DeLongResult",
    "ECEResult",
    "GainsCurve",
    "GiniResult",
    "KSResult",
    "LiftTable",
    "LogLossResult",
    "PSIResult",
    "auc",
    "brier_score",
    "calibration_curve",
    "calibration_slope_intercept",
    "cohens_d",
    "cumulative_gains",
    "delong_test",
    "ece",
    "gini",
    "ks",
    "lift_table",
    "log_loss",
    "psi",
]
