r"""OLS (ordinary / weighted least squares) solver.

Closed-form normal-equations path for ``(squared_error, NoPenalty)``. The
math is ESL §3.2:

.. math::

   \hat\beta = (X^\top W X)^{-1} X^\top W y

with :math:`W = I` for ordinary least squares and :math:`W = \mathrm{diag}(w)`
for the weighted variant. The actual numerical machinery (QR factorisation)
lives in :mod:`model_crafter._internal.linalg`; this module is the thin
adapter that hooks it into the dispatch registry.

Standard errors are :math:`\hat\sigma^2 (X^\top W X)^{-1}` with
:math:`\hat\sigma^2 = \mathrm{RSS}/(n - p)` — matching R's ``lm()`` and
statsmodels' ``OLS().fit().bse`` to machine precision.
"""

from __future__ import annotations

from types import MappingProxyType

import pandas as pd

from model_crafter._internal.linalg import solve_ols as _solve_ols_kernel
from model_crafter.loss import _SquaredErrorLoss
from model_crafter.penalty import NoPenalty
from model_crafter.solve._registry import (
    SolverInputs,
    SolverOutputs,
    register,
)

__all__ = ["solve_ols"]


def solve_ols(inputs: SolverInputs) -> SolverOutputs:
    """Closed-form OLS / WLS solver for the squared-error + NoPenalty case."""
    X = inputs.design.values
    y = inputs.y
    fit = _solve_ols_kernel(X, y, weights=inputs.weights)

    columns = inputs.design.columns
    coefficients = pd.Series(fit.beta, index=list(columns), name="coefficient")
    coefficient_se = pd.Series(fit.se, index=list(columns), name="se")

    # Loss value, on the convention chosen in loss.py.
    loss_value = inputs.spec.loss.value(y, fit.fitted, weights=inputs.weights)
    # NoPenalty contributes 0; honour the protocol anyway.
    penalty_value = float(inputs.spec.penalty.value(fit.beta))

    solver_info = MappingProxyType(
        {
            "solver": "ols_qr",
            "residual_std_error": float(fit.sigma2 ** 0.5),
            "sigma2": float(fit.sigma2),
            "r_squared": float(fit.r_squared),
            "rss": float(fit.rss),
            "df_resid": int(fit.df_resid),
        }
    )

    return SolverOutputs(
        coefficients=coefficients,
        coefficient_se=coefficient_se,
        fit_state=MappingProxyType({}),
        loss_value=float(loss_value),
        penalty_value=penalty_value,
        n_obs=int(X.shape[0]),
        converged=True,
        solver_info=solver_info,
    )


# Self-registration. The registry forbids duplicate keys, so this runs once.
register((_SquaredErrorLoss, NoPenalty), solve_ols)
