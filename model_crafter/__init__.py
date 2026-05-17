"""model_crafter — a Python package for credit risk modeling that feels like pen and paper.

See DESIGN.md for the architectural contract and AGENTS.md for the build plan.

Phase 1–3 public surface — re-exports follow the AGENTS.md per-phase scopes.
The reference end-to-end example in DESIGN.md §10 is the north star for what
should be importable from ``model_crafter``.
"""

# Phase 1: spec, solve, predict, OLS, assumption framework
from model_crafter.assumptions import check_assumptions

# Phase 4: basis terms, WoE / binned, interactions, binning_table
from model_crafter.inspect import binning_table
from model_crafter.loss import logistic, squared_error
from model_crafter.metrics import (
    auc,
    brier_score,
    calibration_curve,
    calibration_slope_intercept,
    cohens_d,
    cumulative_gains,
    delong_test,
    ece,
    gini,
    ks,
    lift_table,
    log_loss,
    psi,
)

# Phase 2: penalties + lambda path
from model_crafter.penalty import NoPenalty, l1, l2

# Phase 3: performance bundle
from model_crafter.performance import performance
from model_crafter.solve import predict, solve
from model_crafter.spec import linear
from model_crafter.terms import (
    binned,
    bs,
    categorical,
    cross,
    hinge,
    interact,
    manual,
    monotonic,
    ns,
    poly,
    smooth,
    step,
    tree_bins,
    woe,
)

# Phase 3: temporal CV, tune, bootstrap
from model_crafter.validation import (
    bootstrap,
    cross_validate,
    expanding_window,
    lambda_path,
    log_grid,
    nested_cv,
    one_se_rule,
    over_time,
    purged_kfold,
    rolling_window,
    time_split,
    tune,
)

__version__ = "0.0.0"

__all__ = [
    "NoPenalty",
    "__version__",
    "auc",
    "binned",
    "binning_table",
    "bootstrap",
    "brier_score",
    "bs",
    "calibration_curve",
    "calibration_slope_intercept",
    "categorical",
    "check_assumptions",
    "cohens_d",
    "cross",
    "cross_validate",
    "cumulative_gains",
    "delong_test",
    "ece",
    "expanding_window",
    "gini",
    "hinge",
    "interact",
    "ks",
    "l1",
    "l2",
    "lambda_path",
    "lift_table",
    "linear",
    "log_grid",
    "log_loss",
    "logistic",
    "manual",
    "monotonic",
    "nested_cv",
    "ns",
    "one_se_rule",
    "over_time",
    "performance",
    "poly",
    "predict",
    "psi",
    "purged_kfold",
    "rolling_window",
    "smooth",
    "solve",
    "squared_error",
    "step",
    "time_split",
    "tree_bins",
    "tune",
    "woe",
]
