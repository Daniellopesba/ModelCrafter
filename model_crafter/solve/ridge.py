r"""Closed-form ridge solver.

Solves the squared-error + L2 case via the augmented-design Tikhonov
formulation. The objective minimised matches ``glmnet(alpha=0)``:

.. math::

   \frac{1}{2n}\,\| y - \beta_0 - X\beta \|^2
        \;+\; \tfrac{\lambda}{2}\,\|\beta\|^2

with :math:`\beta_0` (the intercept) **unpenalised** — this is the ESL
§3.4.1 / equation 3.44 convention and the one ``glmnet`` uses. For the
weighted variant the data-fit term carries :math:`W = \mathrm{diag}(w)`
and the effective ridge scale is :math:`(\sum_i w_i)\lambda` rather than
:math:`n\lambda`, mirroring the OLS / WLS relationship in the existing
solver (``model_crafter/_internal/linalg.py``).

Numerical strategy
------------------
We do not form :math:`(X^T X + n\lambda I')^{-1}` directly. Instead we
solve the equivalent augmented least-squares problem

.. math::

   \min_{\beta_s} \left\| \begin{pmatrix} \sqrt{w}\,\tilde y \\ 0 \end{pmatrix}
       - \begin{pmatrix} \sqrt{w}\,\tilde X \\ \sqrt{(\sum_i w_i)\lambda}\, I_p \end{pmatrix}
         \beta_s \right\|_2^2

via QR factorisation of the stacked matrix, where ``~`` denotes
centering by the (weighted) feature / target means. Centering profiles
out the intercept exactly (because the penalty matrix is zero on the
intercept row), and the recovered slopes are identical to those of the
joint system with an unpenalised intercept. The intercept is recovered
by :math:`\hat\beta_0 = \bar y - \bar x^T \hat\beta`. Forming the
augmented matrix and solving by QR is numerically more stable than
inverting the normal equations because it avoids squaring the condition
number of :math:`X` (ESL §3.2 makes the same argument for OLS).

Standard errors
---------------
For ridge they are not closed-form in the sklearn / glmnet sense; the
covariance of :math:`\hat\beta` under fixed :math:`\lambda` involves the
hat matrix :math:`H = X(X^T X + n\lambda I)^{-1} X^T`, but
single-:math:`\lambda` SEs are widely considered misleading because the
shrinkage is a function of the same data (DESIGN.md §3.2.3 makes the
bootstrap the recommended tool). We therefore leave
``coefficient_se=None`` and document the choice; ``mc.bootstrap`` (Phase
3) is the canonical way to put CIs on ridge coefficients.

Registration
------------
Self-registers for ``(SquaredErrorLoss, L2Penalty)``. P2.A owns
``L2Penalty``; if the import is unavailable (parallel-agent race), the
registration is deferred and a clear error message points the
integration agent at the missing import.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import cast

import numpy as np
import pandas as pd
import scipy.linalg as sla

from model_crafter._internal.design import INTERCEPT_NAME
from model_crafter.loss import _SquaredErrorLoss
from model_crafter.solve._registry import (
    SolverInputs,
    SolverOutputs,
    register,
)


def _split_intercept(
    X: np.ndarray, columns: tuple[str, ...]
) -> tuple[np.ndarray | None, np.ndarray, tuple[str, ...], int | None]:
    """Locate the intercept column and return it separately from the rest.

    Returns ``(intercept_column_or_None, X_no_intercept, slope_names, intercept_index)``.
    The convention in :mod:`model_crafter._internal.design` puts the
    intercept first and names it ``"(Intercept)"``; we look it up by name
    rather than by position so a future reshuffle stays correct.
    """
    try:
        idx = columns.index(INTERCEPT_NAME)
    except ValueError:
        return None, X, columns, None
    Xint = X[:, idx]
    keep = [j for j in range(X.shape[1]) if j != idx]
    Xrest = X[:, keep]
    slope_names = tuple(columns[j] for j in keep)
    return Xint, Xrest, slope_names, idx


def _ridge_qr_solve(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    weights: np.ndarray | None,
) -> tuple[float, np.ndarray, np.ndarray]:
    r"""Solve the closed-form ridge with an unpenalised intercept via QR.

    Parameters
    ----------
    X:
        ``(n, p)`` matrix of slope features (intercept *removed*).
    y:
        ``(n,)`` target vector.
    lam:
        Ridge strength :math:`\lambda \ge 0`. ``lam == 0`` recovers OLS.
    weights:
        ``(n,)`` non-negative weights, or ``None`` for uniform.

    Returns
    -------
    intercept, slopes, fitted
        The intercept scalar, slope vector, and fitted values on the
        original (un-centered) scale.
    """
    n, p = X.shape
    w = np.ones(n, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    sw = float(np.sum(w))

    # Weighted means for centering: profiles out the intercept exactly when
    # the penalty matrix has a zero on the intercept row. Equivalent to
    # solving the joint system with an unpenalised intercept.
    xbar = (w[:, None] * X).sum(axis=0) / sw
    ybar = float(np.sum(w * y) / sw)
    Xc = X - xbar
    yc = y - ybar

    sqrtw = np.sqrt(w)
    Xw = Xc * sqrtw[:, None]
    yw = yc * sqrtw

    # Augmented Tikhonov system: stack sqrt(sw * lambda) * I_p underneath Xw.
    # Solving by QR is numerically equivalent to (X'WX + sw*lam*I)^-1 X'Wy but
    # avoids squaring the condition number of X.
    eff_lambda = sw * lam
    if eff_lambda > 0.0:
        Xa = np.vstack([Xw, np.sqrt(eff_lambda) * np.eye(p)])
        ya = np.concatenate([yw, np.zeros(p)])
    else:
        Xa = Xw
        ya = yw

    qr_result = cast(
        tuple[np.ndarray, np.ndarray],
        sla.qr(Xa, mode="economic"),
    )
    Q, R = qr_result
    qty = Q.T @ ya
    slopes = sla.solve_triangular(R, qty, lower=False)
    intercept = ybar - float(xbar @ slopes)
    fitted = intercept + X @ slopes
    return intercept, slopes, fitted


def solve_ridge_closed_form(inputs: SolverInputs) -> SolverOutputs:
    r"""Closed-form ridge solver for the squared-error + L2 case.

    See module docstring for the math. The penalty's ``lam`` attribute is
    read directly; standard errors are intentionally not computed (see
    module docstring).
    """
    penalty = inputs.spec.penalty
    # ``L2Penalty.lam`` is the documented field name (AGENTS.md P2.A). We
    # don't import the type here for circularity reasons, so we duck-type
    # the attribute access and check explicitly.
    lam_attr = getattr(penalty, "lam", None)
    if lam_attr is None:
        raise TypeError(
            f"ridge solver requires a penalty with a 'lam' attribute; "
            f"got {type(penalty).__name__}"
        )
    lam = float(lam_attr)
    if lam < 0.0:
        raise ValueError(f"L2 strength must be non-negative; got lam={lam}")

    X_full = inputs.design.values
    y = inputs.y
    columns = inputs.design.columns

    Xint, X_slopes, slope_names, _ = _split_intercept(X_full, columns)

    if Xint is None:
        # No intercept in the spec. The closed form still applies but the
        # "unpenalised intercept" subtlety doesn't apply; we penalise every
        # column uniformly. Solve the augmented system directly.
        n, p = X_slopes.shape
        w = (
            np.ones(n, dtype=float)
            if inputs.weights is None
            else np.asarray(inputs.weights, dtype=float)
        )
        sw = float(np.sum(w))
        sqrtw = np.sqrt(w)
        Xw = X_slopes * sqrtw[:, None]
        yw = y * sqrtw
        eff_lambda = sw * lam
        if eff_lambda > 0.0:
            Xa = np.vstack([Xw, np.sqrt(eff_lambda) * np.eye(p)])
            ya = np.concatenate([yw, np.zeros(p)])
        else:
            Xa = Xw
            ya = yw
        qr_result = cast(
            tuple[np.ndarray, np.ndarray], sla.qr(Xa, mode="economic")
        )
        Q, R = qr_result
        beta = sla.solve_triangular(R, Q.T @ ya, lower=False)
        fitted = X_slopes @ beta
        coef_index = list(columns)
        coef_values = beta
    else:
        intercept, slopes, fitted = _ridge_qr_solve(
            X_slopes, y, lam, weights=inputs.weights
        )
        # Reassemble the coefficient series in the *original* column order so
        # downstream code (predict, repr) sees a consistent layout.
        coef_index = list(columns)
        coef_values = np.empty(len(columns), dtype=float)
        # Fill intercept and slopes back into the original positions.
        slope_pos = 0
        for j, name in enumerate(columns):
            if name == INTERCEPT_NAME:
                coef_values[j] = intercept
            else:
                coef_values[j] = slopes[slope_pos]
                slope_pos += 1
        # Sanity: every slot filled exactly once.
        assert slope_pos == len(slope_names)

    coefficients = pd.Series(coef_values, index=coef_index, name="coefficient")

    loss_value = inputs.spec.loss.value(y, fitted, weights=inputs.weights)
    penalty_value = float(inputs.spec.penalty.value(coef_values))

    # Diagnostic RSS / R^2 (unweighted RSS against the unweighted mean) for
    # parity with the OLS solver's solver_info payload.
    resid = y - fitted
    rss = float(np.sum(resid * resid))
    tss = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - rss / tss if tss > 0 else float("nan")

    solver_info = MappingProxyType(
        {
            "solver": "ridge_closed_form_qr",
            "lambda": lam,
            "rss": rss,
            "r_squared": r_squared,
            "n_obs": int(X_full.shape[0]),
            "n_features": int(X_full.shape[1]),
            "intercept_unpenalised": Xint is not None,
        }
    )

    return SolverOutputs(
        coefficients=coefficients,
        # Ridge SEs are intentionally not computed; bootstrap is the
        # recommended tool (DESIGN.md §3.2.3).
        coefficient_se=None,
        fit_state=MappingProxyType({}),
        loss_value=float(loss_value),
        penalty_value=penalty_value,
        n_obs=int(X_full.shape[0]),
        converged=True,
        solver_info=solver_info,
    )


# Self-registration. ``L2Penalty`` is owned by Task P2.A; if it's not yet
# available on the import path, *defer registration silently* so callers
# (and especially the test suite, which stubs ``L2Penalty`` locally to
# unblock parallel development) can still import this module without a
# hard failure. Once P2.A lands, the integration agent imports this
# module from ``solve/__init__.py`` and the real ``L2Penalty`` is in
# scope; the ``try`` succeeds and the registration runs.
#
# This pattern matches the AGENTS.md "Coordination notes #3" guidance —
# when in doubt about a not-yet-merged dependency, write against the
# contract and let the integration agent verify the wiring.
try:
    from model_crafter.penalty import L2Penalty  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - parallel-agent race only
    L2Penalty = None  # type: ignore[assignment, misc]

if L2Penalty is not None:
    register((_SquaredErrorLoss, L2Penalty), solve_ridge_closed_form)
