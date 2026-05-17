r"""IRLS solver for logistic regression and its ridge variant.

ESL §4.4.1 in equations: at iteration :math:`t` with current
:math:`\beta^{(t)}`,

.. math::

    \eta &= X \beta^{(t)} \\
    p     &= \sigma(\eta) \\
    W     &= \mathrm{diag}\bigl(w \odot p (1 - p)\bigr) \\
    z     &= \eta + (y - p) / \bigl(p(1 - p)\bigr) \\
    \beta^{(t+1)} &= (X^\top W X + n \lambda I')^{-1} X^\top W z

with :math:`I'` the identity with a zero on the intercept row
(unpenalised intercept; ESL §3.4.1 / glmnet convention) and sample
weights :math:`w` entering :math:`W` multiplicatively per row.
:math:`\lambda = 0` reduces to plain IRLS.

We solve the WLS step via QR of the augmented design

.. math::

    \begin{bmatrix} \sqrt{W}\,X \\ \sqrt{n\lambda}\,I' \end{bmatrix} \beta
        = \begin{bmatrix} \sqrt{W}\,z \\ 0 \end{bmatrix}.

The condition number is the square root of what direct inversion of
:math:`X^\top W X + n\lambda I'` would give (ESL §3.2's OLS argument
transfers unchanged to the IRLS subproblem).

For the unpenalised IRLS the closed form
:math:`\mathrm{SE}(\hat\beta) = \sqrt{\mathrm{diag}((X^\top W X)^{-1})}`
at convergence matches ``statsmodels.GLM(family=Binomial()).bse`` to
machine precision. For ridge there is no closed-form CI (the shrinkage
is a function of the same data); :func:`mc.bootstrap` is the v0 path
(DESIGN.md §3.2.3).

Self-registers two solver entries::

   (LogisticLoss, NoPenalty) -> solve_logistic_irls
   (LogisticLoss, L2Penalty) -> solve_logistic_ridge_irls

The L1 / elastic-net cases route through :mod:`.prox_cd` instead.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import cast

import numpy as np
import pandas as pd
import scipy.linalg as sla

from model_crafter.loss import LogisticLoss
from model_crafter.penalty import L2Penalty, NoPenalty
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

__all__ = ["solve_logistic_irls", "solve_logistic_ridge_irls"]


# Weighted least squares (one IRLS step).


def _solve_weighted_least_squares(
    X: np.ndarray,
    z: np.ndarray,
    working_w: np.ndarray,
) -> np.ndarray:
    r"""Argmin of :math:`\sum_i w_i (z_i - x_i^\top \beta)^2` via QR of
    :math:`W^{1/2} X`.

    Raises ``np.linalg.LinAlgError`` if :math:`X^\top W X` is singular —
    expected to be pre-empted by the HARD :class:`FullRankDesign` check
    in :mod:`model_crafter.solve.__init__`.
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
    r"""Ridge-WLS subproblem
    :math:`(X^\top W X + n \lambda I') \beta = X^\top W z` via the
    augmented-design QR (intercept unpenalised when present).
    """
    n, p = X.shape
    sqrtw = np.sqrt(working_w)
    Xw = X * sqrtw[:, None]
    zw = z * sqrtw

    if lam == 0.0:
        return _solve_weighted_least_squares(X, z, working_w)

    I_pen = np.eye(p)
    if intercept_idx is not None:
        I_pen[intercept_idx, intercept_idx] = 0.0
    scale = float(np.sqrt(n * lam))
    Xa = np.vstack([Xw, scale * I_pen])
    za = np.concatenate([zw, np.zeros(p)])
    qr_result = cast(tuple[np.ndarray, np.ndarray], sla.qr(Xa, mode="economic"))
    Q, R = qr_result
    return sla.solve_triangular(R, Q.T @ za, lower=False)


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
    """``(beta, working_weights, n_iter, converged)`` from running IRLS to convergence.

    ``lam = 0`` is unpenalised; ``lam > 0`` runs ridge with the intercept
    row unpenalised.
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
    r"""Closed-form SE :math:`= \sqrt{\mathrm{diag}((X^\top W X)^{-1})}`.

    Unpenalised IRLS only — for ridge there is no closed-form CI
    (DESIGN.md §3.2.3 → bootstrap). Returns ``None`` if the inverse
    fails (near-singular working weights).
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


# Solver entries.


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


def solve_logistic_ridge_irls(inputs: SolverInputs) -> SolverOutputs:
    r"""Solver entry for ``(LogisticLoss, L2Penalty)``.

    Same IRLS loop as :func:`solve_logistic_irls` but with the ridge
    augmented system. Coefficient SEs are intentionally ``None``;
    :func:`mc.bootstrap` is the recommended uncertainty tool.
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
        coefficient_se=None,  # ridge-logistic: bootstrap is the right tool.
        fit_state=MappingProxyType({}),
        loss_value=float(loss_value),
        penalty_value=penalty_value,
        n_obs=int(X.shape[0]),
        converged=bool(converged),
        solver_info=solver_info,
    )


register((LogisticLoss, NoPenalty), solve_logistic_irls)
register((LogisticLoss, L2Penalty), solve_logistic_ridge_irls)
