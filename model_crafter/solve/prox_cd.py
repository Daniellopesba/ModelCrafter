r"""Proximal-Newton / coordinate-descent solver for L1 and elastic-net logistic.

FHT 2010 §2.6: wrap an IRLS outer loop around a weighted-elastic-net
inner solve. The outer loop computes :math:`(W, z)` from the current
:math:`\beta^{(t)}`; the inner step solves

.. math::

    Q(\beta) = \tfrac{1}{2n} \sum_i w'_i (z_i - x_i^\top \beta)^2
             + \lambda_1 \|\beta\|_1
             + \tfrac{\lambda_2}{2} \|\beta\|_2^2

with :math:`w'_i = w_i^{\text{user}} \cdot p_i (1 - p_i)` the working
weights, via the squared-error coordinate-descent path in
:mod:`.coordinate`. We iterate until
:math:`\|\beta^{(t+1)} - \beta^{(t)}\| / \max(\|\beta^{(t)}\|, 1)` is
below ``tol = 1e-8`` or ``max_iter = 100``.

CD weight rescaling
-------------------

The CD path normalises weights to :math:`\sum w = n` (the ``glmnet``
convention), which is equivalent to multiplying :math:`Q` by
:math:`n / \sum w'_i`. We absorb the same factor into the penalty
parameters before handing them in: :math:`\lambda_k^{\text{CD}} =
\lambda_k \cdot n / \sum w'_i`. Without this rescaling the inner
subproblem optimises a different objective than the outer Taylor
expansion suggests and the outer loop converges to the wrong optimum.

Self-registers two solver entries::

   (LogisticLoss, L1Penalty)  -> solve_logistic_prox_cd
   (LogisticLoss, PenaltySum) -> solve_logistic_prox_cd

The pure-ridge case is routed by dispatch to :mod:`.irls` instead;
:class:`PenaltySum` here may be pure L1 or elastic net.
"""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pandas as pd

from model_crafter.loss import LogisticLoss
from model_crafter.penalty import L1Penalty, L2Penalty, NoPenalty, PenaltySum
from model_crafter.solve._irls_core import (
    _MAX_ITER_DEFAULT,
    _TOL_DEFAULT,
    _build_working_response,
    _initialise_eta,
    _intercept_index,
    _normalize_weights,
    _relative_beta_change,
)
from model_crafter.solve._registry import (
    SolverInputs,
    SolverOutputs,
    register,
)

__all__ = ["solve_logistic_prox_cd"]


def _split_l1_l2(penalty: object) -> tuple[float, float]:
    """Read ``(lam_l1, lam_l2)`` from an L1 / L2 / L1+L2 ``PenaltySum``."""
    if isinstance(penalty, L1Penalty):
        return float(penalty.lam), 0.0
    if isinstance(penalty, L2Penalty):
        return 0.0, float(penalty.lam)
    if isinstance(penalty, PenaltySum):
        lam_l1 = 0.0
        lam_l2 = 0.0
        for part in penalty.parts:
            if isinstance(part, L1Penalty):
                lam_l1 += float(part.lam)
            elif isinstance(part, L2Penalty):
                lam_l2 += float(part.lam)
            elif isinstance(part, NoPenalty):
                continue
            else:
                raise TypeError(
                    "logistic proximal-Newton solver only supports L1 / L2 / "
                    f"L1+L2 penalties; got {type(part).__name__} in PenaltySum"
                )
        return lam_l1, lam_l2
    raise TypeError(
        "logistic proximal-Newton solver requires L1Penalty, L2Penalty, or "
        f"PenaltySum; got {type(penalty).__name__}"
    )


def _solve_weighted_enet_subproblem(
    X_no_int: np.ndarray,
    z: np.ndarray,
    working_w: np.ndarray,
    *,
    has_intercept: bool,
    lam_l1: float,
    lam_l2: float,
    init_beta: np.ndarray | None,
    init_intercept: float,
) -> tuple[float, np.ndarray]:
    """Solve one weighted elastic-net subproblem via the CD path.

    See the module docstring for the math and for why the penalties are
    rescaled by ``n / sum(working_w)`` before being handed to the CD
    solver.

    ``init_beta`` / ``init_intercept`` are reserved for a future
    warm-start hook on :func:`coordinate_descent_path`; for now the
    inner CD always starts from zero, which is fine because each
    subproblem is strongly convex (when ``lam_l2 > 0``) or piecewise
    quadratic with a unique argmin (lasso).
    """
    from model_crafter.solve.coordinate import coordinate_descent_path

    _ = init_beta  # warm start is implicit via the working response.
    _ = init_intercept

    sw = float(np.sum(working_w))
    n = working_w.shape[0]
    scale = (n / sw) if sw > 0 else 1.0
    lam_l1_cd = lam_l1 * scale
    lam_l2_cd = lam_l2 * scale

    betas, intercepts, _infos = coordinate_descent_path(
        X_no_int,
        z,
        lambdas_l1=[lam_l1_cd],
        lam_l2=lam_l2_cd,
        weights=working_w,
        intercept=has_intercept,
        tol=1e-9,
        max_iter=2000,
        warm_start=False,
    )
    return float(intercepts[0]), betas[0]


def solve_logistic_prox_cd(inputs: SolverInputs) -> SolverOutputs:
    r"""Solver entry for ``(LogisticLoss, L1Penalty)`` and the elastic-net
    :class:`PenaltySum`.

    A pure-ridge ``PenaltySum`` would be cheaper through
    :func:`solve_logistic_ridge_irls`, but the dispatch routes by penalty
    *type* (``PenaltySum`` rather than :class:`L2Penalty`) so we
    re-detect the structure here. Pure-L1 ``PenaltySum`` (only L1 parts)
    and ``L1Penalty`` route through the same path with ``lam_l2 = 0``.
    """
    penalty = inputs.spec.penalty
    lam_l1, lam_l2 = _split_l1_l2(penalty)

    X = inputs.design.values
    y = inputs.y
    columns = inputs.design.columns
    w = _normalize_weights(inputs.weights, n=X.shape[0])
    intercept_idx = _intercept_index(columns)
    has_intercept = intercept_idx is not None

    # The inner CD solver expects the design *without* an intercept column
    # (it handles the intercept by centring). Split and remember the
    # mapping back into the original column order.
    if has_intercept:
        keep = [j for j in range(X.shape[1]) if j != intercept_idx]
        X_no_int = X[:, keep]
        slope_names = [columns[j] for j in keep]
    else:
        X_no_int = X
        slope_names = list(columns)

    beta_full = np.zeros(X.shape[1], dtype=float)
    if has_intercept:
        beta_full[intercept_idx] = _initialise_eta(y, w, has_intercept=True)
    eta = X @ beta_full

    converged = False
    n_iter = 0
    last_change = float("inf")
    for it in range(1, _MAX_ITER_DEFAULT + 1):
        n_iter = it
        z, working_w = _build_working_response(eta, y, w)
        intercept_new, slopes_new = _solve_weighted_enet_subproblem(
            X_no_int,
            z,
            working_w,
            has_intercept=has_intercept,
            lam_l1=lam_l1,
            lam_l2=lam_l2,
            init_beta=None,
            init_intercept=0.0,
        )
        beta_new = np.zeros_like(beta_full)
        if has_intercept:
            beta_new[intercept_idx] = intercept_new
        slope_pos = 0
        for j in range(X.shape[1]):
            if has_intercept and j == intercept_idx:
                continue
            beta_new[j] = slopes_new[slope_pos]
            slope_pos += 1
        change = _relative_beta_change(beta_new, beta_full)
        last_change = change
        eta = X @ beta_new
        if change < _TOL_DEFAULT:
            beta_full = beta_new
            converged = True
            break
        beta_full = beta_new

    coefficients = pd.Series(beta_full, index=list(columns), name="coefficient")
    loss_value = inputs.spec.loss.value(y, eta, weights=inputs.weights)
    penalty_value = float(inputs.spec.penalty.value(beta_full))

    solver_info = MappingProxyType(
        {
            "solver": "irls_prox_cd",
            "lam_l1": float(lam_l1),
            "lam_l2": float(lam_l2),
            "n_iter": int(n_iter),
            "max_iter": int(_MAX_ITER_DEFAULT),
            "tol": float(_TOL_DEFAULT),
            "converged": bool(converged),
            "max_coef_change": float(last_change),
            "intercept_unpenalised": has_intercept,
            "slope_names": tuple(slope_names),
        }
    )
    return SolverOutputs(
        coefficients=coefficients,
        coefficient_se=None,
        fit_state=MappingProxyType({}),
        loss_value=float(loss_value),
        penalty_value=penalty_value,
        n_obs=int(X.shape[0]),
        converged=bool(converged),
        solver_info=solver_info,
    )


register((LogisticLoss, L1Penalty), solve_logistic_prox_cd)
register((LogisticLoss, PenaltySum), solve_logistic_prox_cd)
