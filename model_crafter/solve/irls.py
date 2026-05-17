r"""IRLS solver for logistic regression and its penalised variants (Task P3.A).

References
----------
* Hastie, Tibshirani & Friedman, *Elements of Statistical Learning* (2nd ed.),
  §4.4 (logistic regression), §4.4.2 (perfect separation), §3.4.1 (ridge),
  §3.4.2 (lasso).
* Friedman, Hastie & Tibshirani 2010, *Regularization Paths for Generalized
  Linear Models via Coordinate Descent*, JStatSoft 33(1) — referred to
  below as **FHT 2010**. §2.6 spells out the proximal-Newton / coordinate
  descent recipe we use for the L1 and elastic-net cases.

Algorithms
----------

**IRLS (unpenalised and ridge).** Newton-Raphson on the logistic
likelihood, rewritten as iteratively-reweighted least squares (ESL §4.4.1).
At iteration :math:`t` with current :math:`\beta^{(t)}`:

.. math::

   \eta &= X\beta^{(t)} \\
   p     &= \sigma(\eta) \\
   W     &= \mathrm{diag}\!\bigl(w \odot p(1 - p)\bigr) \quad \text{(working weights)} \\
   z     &= \eta + (y - p) / \bigl(p(1 - p)\bigr) \quad \text{(working response)} \\
   \beta^{(t+1)} &= (X^\top W X + n\lambda\,I')^{-1}\,X^\top W z

where :math:`I'` is the identity matrix with zero on the intercept row (the
intercept is unpenalised, ESL §3.4.1 / glmnet convention).
Sample weights :math:`w` enter ``W`` multiplicatively per row.
:math:`\lambda = 0` reduces to plain IRLS.

**Proximal-Newton CD (L1 and elastic net).** FHT 2010 §2.6:
the *outer* loop is IRLS, computing :math:`(W, z)`; the *inner* loop is a
single pass of weighted-lasso (or weighted-elastic-net) coordinate descent
on the working response, solved via
:func:`model_crafter.solve.coordinate.coordinate_descent_path`. Outer
convergence is checked on :math:`\|\beta^{(t+1)} - \beta^{(t)}\|` divided by
:math:`\max(\|\beta^{(t)}\|, 1)`.

Convergence
-----------
The default tolerance is ``tol=1e-8`` on the relative change in
:math:`\beta`, and ``max_iter=100``. On non-convergence the solver returns
``converged=False`` on :class:`SolverOutputs`; the post-fit
:class:`~model_crafter.assumptions.NoPerfectSeparation` check examines the
fitted coefficient magnitudes and fires a HARD violation with the ESL
§4.4.2 L2-remedy message when the magnitudes are large.

Standard errors
---------------
For the unpenalised IRLS the closed form
:math:`\mathrm{SE}(\hat\beta) = \sqrt{\mathrm{diag}((X^\top W X)^{-1})}`
evaluated at convergence matches ``statsmodels.GLM(family=Binomial())``'s
``.bse`` to machine precision. For the ridge variant the closed-form SE
does not exist (the shrinkage is a function of the same data); we return
``coefficient_se=None`` and rely on :func:`mc.bootstrap` (Phase 3.C) for
honest CIs (DESIGN.md §3.2.3).

Registration
------------
Self-registers four solver entries::

   (LogisticLoss, NoPenalty)    -> solve_logistic_irls
   (LogisticLoss, L2Penalty)    -> solve_logistic_ridge_irls
   (LogisticLoss, L1Penalty)    -> solve_logistic_prox_cd
   (LogisticLoss, PenaltySum)   -> solve_logistic_prox_cd
"""

from __future__ import annotations

from types import MappingProxyType
from typing import cast

import numpy as np
import pandas as pd
import scipy.linalg as sla
from scipy.special import expit

from model_crafter._internal.design import INTERCEPT_NAME
from model_crafter.loss import LogisticLoss
from model_crafter.penalty import L1Penalty, L2Penalty, NoPenalty, PenaltySum
from model_crafter.solve._registry import (
    SolverInputs,
    SolverOutputs,
    register,
)

__all__ = [
    "solve_logistic_irls",
    "solve_logistic_prox_cd",
    "solve_logistic_ridge_irls",
]


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------


_TOL_DEFAULT = 1e-8
_MAX_ITER_DEFAULT = 100
_P_CLIP = 1e-12  # avoid 0 / 0 in the working response when p saturates


def _normalize_weights(weights: np.ndarray | None, n: int) -> np.ndarray:
    """Coerce ``weights`` to an ``(n,)`` float vector, broadcasting ``None``
    to ones. Same convention as the squared-error solvers (no normalisation
    to sum-to-n — IRLS uses the absolute weights).
    """
    if weights is None:
        return np.ones(n, dtype=float)
    w = np.asarray(weights, dtype=float)
    if w.shape != (n,):
        raise ValueError(f"weights shape {w.shape} != ({n},)")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    if not np.any(w > 0):
        raise ValueError("weights must contain at least one positive value")
    return w


def _intercept_index(columns: tuple[str, ...]) -> int | None:
    """Position of ``(Intercept)`` in ``columns``, or ``None`` if absent."""
    try:
        return columns.index(INTERCEPT_NAME)
    except ValueError:
        return None


def _initialise_eta(y: np.ndarray, w: np.ndarray, has_intercept: bool) -> float:
    r"""Initial value for the (unpenalised) intercept when slopes are zero.

    For a logistic GLM with intercept-only spec, the MLE of the intercept
    is :math:`\mathrm{logit}(\bar y_w)` where :math:`\bar y_w` is the
    weighted mean of :math:`y`. We use this to seed :math:`\eta` so that
    IRLS converges in a single iteration for the intercept-only case and
    spends fewer iterations on richer specs.
    """
    if not has_intercept:
        return 0.0
    p = float(np.sum(w * y) / np.sum(w))
    # Clip to keep logit finite for degenerate y (all 0 / all 1).
    p_clipped = float(min(max(p, _P_CLIP), 1.0 - _P_CLIP))
    return float(np.log(p_clipped / (1.0 - p_clipped)))


def _build_working_response(
    eta: np.ndarray, y: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return ``(z, working_weights)`` for one IRLS iteration.

    ``z = eta + (y - p) / (p (1-p))`` is the working response; the
    working weights are ``w * p (1 - p)`` (FHT 2010 eq. 18). We clip ``p``
    away from 0 / 1 by ``_P_CLIP`` so the division stays finite when the
    iterate has begun to saturate (a separation symptom).
    """
    p = expit(eta)
    p = np.clip(p, _P_CLIP, 1.0 - _P_CLIP)
    var = p * (1.0 - p)
    z = eta + (y - p) / var
    return z, w * var


def _solve_weighted_least_squares(
    X: np.ndarray,
    z: np.ndarray,
    working_w: np.ndarray,
) -> np.ndarray:
    r"""Solve :math:`\beta = \mathrm{argmin}\;\sum_i w_i (z_i - x_i^\top\beta)^2`
    via QR factorisation of :math:`W^{1/2} X`.

    Returns the coefficient vector. Used by the unpenalised IRLS update.
    Raises ``np.linalg.LinAlgError`` if :math:`X^\top W X` is singular —
    the caller is expected to surface this as a HARD ``FullRankDesign``
    violation (in practice the dispatch in
    :mod:`model_crafter.solve.__init__` runs the rank check *before*
    handing the design to a solver, so this is defensive).
    """
    sqrtw = np.sqrt(working_w)
    Xw = X * sqrtw[:, None]
    zw = z * sqrtw
    qr_result = cast(tuple[np.ndarray, np.ndarray], sla.qr(Xw, mode="economic"))
    Q, R = qr_result
    return sla.solve_triangular(R, Q.T @ zw, lower=False)


def _solve_ridge_weighted_least_squares(
    X: np.ndarray,
    z: np.ndarray,
    working_w: np.ndarray,
    lam: float,
    intercept_idx: int | None,
) -> np.ndarray:
    r"""Solve the ridge-WLS subproblem
    :math:`(X^\top W X + n\lambda\,I')\,\beta = X^\top W z`
    via the augmented design QR.

    :math:`I'` is the identity with a zero on the intercept row, so the
    intercept is unpenalised (ESL §3.4.1 / glmnet). The augmented system

    .. math::

        \begin{bmatrix}\sqrt{W}\,X \\ \sqrt{n\lambda}\,I'\end{bmatrix}\beta
            \;=\; \begin{bmatrix}\sqrt{W}\,z \\ 0\end{bmatrix}

    has the same solution and a condition number that's the square root of
    the one we'd get from inverting :math:`X^\top W X + n\lambda I'`
    directly (ESL §3.2 argues this for OLS; the IRLS case is identical).
    """
    n, p = X.shape
    sqrtw = np.sqrt(working_w)
    Xw = X * sqrtw[:, None]
    zw = z * sqrtw

    if lam == 0.0:
        return _solve_weighted_least_squares(X, z, working_w)

    # Augmented identity rows penalising every slope (intercept row zeroed).
    I_pen = np.eye(p)
    if intercept_idx is not None:
        I_pen[intercept_idx, intercept_idx] = 0.0
    scale = float(np.sqrt(n * lam))
    Xa = np.vstack([Xw, scale * I_pen])
    za = np.concatenate([zw, np.zeros(p)])
    qr_result = cast(tuple[np.ndarray, np.ndarray], sla.qr(Xa, mode="economic"))
    Q, R = qr_result
    return sla.solve_triangular(R, Q.T @ za, lower=False)


def _relative_beta_change(beta_new: np.ndarray, beta_old: np.ndarray) -> float:
    """``||β_new − β_old|| / max(||β_old||, 1)`` — the convergence statistic."""
    denom = max(float(np.linalg.norm(beta_old)), 1.0)
    return float(np.linalg.norm(beta_new - beta_old) / denom)


# ---------------------------------------------------------------------------
# Unpenalised IRLS
# ---------------------------------------------------------------------------


def _irls_loop(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    intercept_idx: int | None,
    *,
    lam: float = 0.0,
    tol: float = _TOL_DEFAULT,
    max_iter: int = _MAX_ITER_DEFAULT,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    r"""Run IRLS to convergence and return ``(beta, working_weights, n_iter, converged)``.

    ``lam = 0`` is unpenalised; ``lam > 0`` uses the ridge update with the
    intercept row unpenalised.
    """
    n, p = X.shape
    beta = np.zeros(p, dtype=float)
    if intercept_idx is not None:
        beta[intercept_idx] = _initialise_eta(y, w, has_intercept=True)
    eta = X @ beta

    last_working_w = np.zeros(n, dtype=float)
    for it in range(1, max_iter + 1):
        z, working_w = _build_working_response(eta, y, w)
        last_working_w = working_w
        beta_new = _solve_ridge_weighted_least_squares(
            X, z, working_w, lam=lam, intercept_idx=intercept_idx
        )
        change = _relative_beta_change(beta_new, beta)
        eta = X @ beta_new
        if change < tol:
            return beta_new, working_w, it, True
        beta = beta_new
    return beta, last_working_w, max_iter, False


def _compute_irls_se(
    X: np.ndarray, working_w: np.ndarray
) -> np.ndarray | None:
    r"""Closed-form SE = sqrt(diag((X^T W X)^{-1})) at convergence.

    Used only for the **unpenalised** logistic IRLS — for ridge there is no
    closed-form CI (DESIGN.md §3.2.3 -> bootstrap). Returns ``None`` if the
    inverse fails (e.g., near-singular working weights).
    """
    try:
        sqrtw = np.sqrt(working_w)
        Xw = X * sqrtw[:, None]
        qr_result = cast(
            tuple[np.ndarray, np.ndarray], sla.qr(Xw, mode="economic")
        )
        _, R = qr_result
        Rinv = sla.solve_triangular(R, np.eye(R.shape[0]), lower=False)
        cov = Rinv @ Rinv.T
        diag = np.diag(cov)
        return np.sqrt(np.maximum(diag, 0.0))
    except (np.linalg.LinAlgError, ValueError):
        return None


def solve_logistic_irls(inputs: SolverInputs) -> SolverOutputs:
    r"""Solver entry for ``(LogisticLoss, NoPenalty)`` — plain IRLS."""
    X = inputs.design.values
    y = inputs.y
    columns = inputs.design.columns
    w = _normalize_weights(inputs.weights, n=X.shape[0])
    intercept_idx = _intercept_index(columns)

    beta, working_w, n_iter, converged = _irls_loop(
        X, y, w, intercept_idx, lam=0.0
    )
    eta = X @ beta
    se = _compute_irls_se(X, working_w)

    coefficients = pd.Series(beta, index=list(columns), name="coefficient")
    coefficient_se = (
        pd.Series(se, index=list(columns), name="se") if se is not None else None
    )

    loss_value = inputs.spec.loss.value(y, eta, weights=inputs.weights)
    penalty_value = float(inputs.spec.penalty.value(beta))

    solver_info = MappingProxyType(
        {
            "solver": "irls",
            "n_iter": int(n_iter),
            "max_iter": int(_MAX_ITER_DEFAULT),
            "tol": float(_TOL_DEFAULT),
            "converged": bool(converged),
            "intercept_unpenalised": intercept_idx is not None,
        }
    )

    return SolverOutputs(
        coefficients=coefficients,
        coefficient_se=coefficient_se,
        fit_state=MappingProxyType({}),
        loss_value=float(loss_value),
        penalty_value=penalty_value,
        n_obs=int(X.shape[0]),
        converged=bool(converged),
        solver_info=solver_info,
    )


# ---------------------------------------------------------------------------
# Ridge-IRLS (L2Penalty)
# ---------------------------------------------------------------------------


def solve_logistic_ridge_irls(inputs: SolverInputs) -> SolverOutputs:
    r"""Solver entry for ``(LogisticLoss, L2Penalty)``.

    Same IRLS loop as :func:`solve_logistic_irls` but with the ridge
    augmented system. Coefficient SEs are intentionally ``None``;
    bootstrap (Phase 3.C) is the recommended uncertainty tool.
    """
    penalty = inputs.spec.penalty
    lam = float(getattr(penalty, "lam", 0.0))
    if lam < 0.0:
        raise ValueError(f"L2 strength must be non-negative; got lam={lam}")

    X = inputs.design.values
    y = inputs.y
    columns = inputs.design.columns
    w = _normalize_weights(inputs.weights, n=X.shape[0])
    intercept_idx = _intercept_index(columns)

    beta, _working_w, n_iter, converged = _irls_loop(
        X, y, w, intercept_idx, lam=lam
    )
    eta = X @ beta

    coefficients = pd.Series(beta, index=list(columns), name="coefficient")

    loss_value = inputs.spec.loss.value(y, eta, weights=inputs.weights)
    penalty_value = float(inputs.spec.penalty.value(beta))

    solver_info = MappingProxyType(
        {
            "solver": "irls_ridge",
            "lambda": lam,
            "n_iter": int(n_iter),
            "max_iter": int(_MAX_ITER_DEFAULT),
            "tol": float(_TOL_DEFAULT),
            "converged": bool(converged),
            "intercept_unpenalised": intercept_idx is not None,
        }
    )

    return SolverOutputs(
        coefficients=coefficients,
        coefficient_se=None,  # ridge-logistic: bootstrap is the right tool
        fit_state=MappingProxyType({}),
        loss_value=float(loss_value),
        penalty_value=penalty_value,
        n_obs=int(X.shape[0]),
        converged=bool(converged),
        solver_info=solver_info,
    )


# ---------------------------------------------------------------------------
# Proximal-Newton CD (L1Penalty, PenaltySum) — FHT 2010 §2.6
# ---------------------------------------------------------------------------


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
    r"""Solve one weighted elastic-net subproblem of the proximal-Newton outer loop.

    We delegate to :func:`coordinate_descent_path` (squared-error
    coordinate descent on the working response) for the actual coordinate
    sweeps. The outer-loop quadratic Taylor expansion of the *averaged*
    logistic loss at :math:`\beta^{(t)}` is

    .. math::

        Q(\beta) \;=\; \frac{1}{2n}\sum_i w'_i (z_i - x_i^\top\beta)^2
                  \;+\; \lambda_1\|\beta\|_1
                  \;+\; \tfrac{\lambda_2}{2}\|\beta\|_2^2

    with :math:`w'_i = w^{\text{user}}_i \cdot p_i(1 - p_i)` the **unscaled**
    working weights. The CD solver internally rescales weights to sum to
    :math:`n` (the ``glmnet`` convention), which is equivalent to
    multiplying :math:`Q` by :math:`n / \sum_i w'_i`. To preserve the
    argmin we apply the same multiplicative scaling to the penalty
    parameters before handing them to CD: :math:`\lambda_1^{\text{CD}} =
    \lambda_1 \cdot n / \sum_i w'_i` (and analogously for
    :math:`\lambda_2`). Without this scaling, the inner subproblem solves
    a different objective than the outer Taylor expansion suggests and the
    overall fit converges to the wrong optimum.

    ``init_beta`` and ``init_intercept`` are reserved for a future
    warm-start hook on :func:`coordinate_descent_path`; for now the inner
    CD always starts from zero, which is fine because each subproblem is
    strongly convex (when :math:`\lambda_2 > 0`) or piecewise quadratic
    with a unique argmin (lasso).
    """
    from model_crafter.solve.coordinate import coordinate_descent_path

    _ = init_beta  # warm start is implicit via the working response
    _ = init_intercept

    # Rescale penalties to compensate for the CD solver's internal weight
    # normalisation (sum_w -> n).  See docstring above.
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
    r"""Solver entry for ``(LogisticLoss, L1Penalty)`` and
    ``(LogisticLoss, PenaltySum)`` (elastic net).

    FHT 2010 §2.6 proximal-Newton / coordinate-descent recipe:

    1. Outer loop: IRLS to compute the working weights and working response.
    2. Inner step: solve the weighted-(elastic-net) subproblem on
       :math:`(X, z)` with the working weights via
       :func:`coordinate_descent_path`.
    3. Repeat until :math:`\|\beta^{(t+1)} - \beta^{(t)}\|` is below
       tolerance.

    A pure-ridge ``PenaltySum`` would be cheaper through
    :func:`solve_logistic_ridge_irls`, but the dispatch routes by penalty
    *type* (``PenaltySum`` not :class:`L2Penalty`) so we re-detect the
    structure here. Pure-L1 :class:`PenaltySum` (only L1 parts) and pure-L1
    :class:`L1Penalty` route through the same path with ``lam_l2 = 0``.
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
    # (it handles the intercept by centring). Split the design and remember
    # the mapping back into the original column order.
    if has_intercept:
        keep = [j for j in range(X.shape[1]) if j != intercept_idx]
        X_no_int = X[:, keep]
        slope_names = [columns[j] for j in keep]
    else:
        X_no_int = X
        slope_names = list(columns)

    # Initialise.
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


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------


register((LogisticLoss, NoPenalty), solve_logistic_irls)
register((LogisticLoss, L2Penalty), solve_logistic_ridge_irls)
register((LogisticLoss, L1Penalty), solve_logistic_prox_cd)
register((LogisticLoss, PenaltySum), solve_logistic_prox_cd)
