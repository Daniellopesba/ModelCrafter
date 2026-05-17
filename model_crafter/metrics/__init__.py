"""Metric primitives (DESIGN.md §3.3).

Every primitive accepts ``(sol, data, weights=...)`` (with the exception
of :func:`psi`, which takes two distributions directly) and returns a
frozen-dataclass result with a rich ``__repr__`` and ``__float__``.

Sub-modules

* :mod:`.discrimination` — AUC, Gini, KS
* :mod:`.effect_size` — Cohen's d
* :mod:`.calibration` — Brier, ECE, calibration curve, log-loss,
  slope/intercept
* :mod:`.stability` — PSI
* :mod:`.rank` — lift table, cumulative gains

The DeLong (1988) paired AUC test lives in
:mod:`model_crafter.performance.compare` next to ``mc.compare`` — see
that module for why.
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
from model_crafter.metrics.discrimination import (
    AUCResult,
    GiniResult,
    KSResult,
    auc,
    gini,
    ks,
)
from model_crafter.metrics.effect_size import CohensDResult, cohens_d
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
    "ece",
    "gini",
    "ks",
    "lift_table",
    "log_loss",
    "psi",
]
