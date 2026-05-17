"""Validation utilities.

Phase 2 shipped the lambda-path helpers (:func:`lambda_path`,
:func:`log_grid`). Phase 3 adds the temporal splitters, the
assessment-only :func:`cross_validate` runner, the tuning runners
:func:`tune` and :func:`nested_cv`, the time-indexed metric runner
:func:`over_time`, and the bootstrap (:func:`bootstrap`). Selection
rules :func:`best_mean` and :func:`one_se_rule` are also exported.
"""

from __future__ import annotations

from model_crafter.validation.bootstrap import bootstrap
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
    "bootstrap",
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
