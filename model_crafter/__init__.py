"""model_crafter — a Python package for credit risk modeling that feels like pen and paper.

See DESIGN.md for the architectural contract and AGENTS.md for the build plan.

Phase 1–3 public surface — re-exports follow the AGENTS.md per-phase scopes.
The reference end-to-end example in DESIGN.md §10 is the north star for what
should be importable from ``model_crafter``.
"""

# Phase 1: spec, solve, predict, OLS, assumption framework
from model_crafter.assumptions import check_assumptions

# Phase 4: basis terms, WoE / binned, interactions, binning_table
# Phase 6: inspection helpers (coefficients, diagnostics, hat_matrix, influence)
from model_crafter.inspect import (
    Diagnostics,
    Influence,
    binning_table,
    coefficients,
    diagnostics,
    hat_matrix,
    influence,
)
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

# Phase 3 + 5: performance bundle, temporal / segmented / comparison
from model_crafter.performance import (
    compare,
    performance,
    performance_by_segment,
    performance_over_time,
)
from model_crafter.solve import predict, solve

# Phase 6: segmented spec / solution
from model_crafter.spec import SegmentedSpec, linear, segmented
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

from model_crafter.solution import SegmentedSolution

__all__ = [
    "Diagnostics",
    "Influence",
    "NoPenalty",
    "SegmentedSolution",
    "SegmentedSpec",
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
    "coefficients",
    "cohens_d",
    "cross",
    "cross_validate",
    "cumulative_gains",
    "delong_test",
    "diagnostics",
    "ece",
    "expanding_window",
    "gini",
    "hat_matrix",
    "hinge",
    "influence",
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
    "compare",
    "performance",
    "performance_by_segment",
    "performance_over_time",
    "poly",
    "predict",
    "psi",
    "purged_kfold",
    "rolling_window",
    "segmented",
    "smooth",
    "solve",
    "squared_error",
    "step",
    "time_split",
    "tree_bins",
    "tune",
    "woe",
]
