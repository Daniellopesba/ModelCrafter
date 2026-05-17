"""Validation utilities.

Phase 2 shipped the lambda-path helpers (:func:`lambda_path`,
:func:`log_grid`); Phase 3 (AGENTS.md Task P3.B) adds the temporal
splitters, the assessment-only :func:`cross_validate` runner, the
tuning runners :func:`tune` and :func:`nested_cv`, and the
time-indexed metric runner :func:`over_time`. The selection rules
:func:`best_mean` and :func:`one_se_rule` are exported alongside.

The bootstrap (:func:`mc.bootstrap`) is a parallel P3.C workspace and
is wired in at P3.INTEG.
"""

from __future__ import annotations

from model_crafter.validation.cross_validate import CVResult, cross_validate
from model_crafter.validation.lambda_path import lambda_path, log_grid
from model_crafter.validation.over_time import over_time
from model_crafter.validation.splitters import (
    Splitter,
    expanding_window,
    purged_kfold,
    rolling_window,
    time_split,
)
from model_crafter.validation.tune import (
    NestedCVResult,
    TuneResult,
    best_mean,
    nested_cv,
    one_se_rule,
    tune,
)

__all__ = [
    "CVResult",
    "NestedCVResult",
    "Splitter",
    "TuneResult",
    "best_mean",
    "cross_validate",
    "expanding_window",
    "lambda_path",
    "log_grid",
    "nested_cv",
    "one_se_rule",
    "over_time",
    "purged_kfold",
    "rolling_window",
    "time_split",
    "tune",
]
