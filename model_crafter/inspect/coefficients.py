r"""Coefficient tables with estimates, standard errors, and Wald statistics.

For OLS the SE is the closed-form

.. math::

    \widehat{\mathrm{Var}}(\hat\beta) = \hat\sigma^2 (X^\top W X)^{-1},
    \qquad \hat\sigma^2 = \mathrm{RSS} / (n - p),

the ``z`` column is the t-statistic
:math:`\hat\beta_j / \mathrm{SE}_j`, and the ``p_value`` is the
two-sided tail probability of :math:`t_{n-p}`. ESL §3.2.

For logistic IRLS (when ``coefficient_se`` is available) the ``z`` is
the Wald statistic and the p-value is two-sided standard normal — the
convention R's ``glm`` reports. ESL §4.4.4.

For ridge / lasso / elastic-net fits ``coefficient_se`` is ``None`` by
deliberate design (DESIGN.md §3.2 — the closed-form penalised SE is
biased and uninformative); ``std_error`` becomes ``None`` and ``z`` /
``p_value`` become NaN. The alternative uncertainty path is
:func:`mc.bootstrap` (ESL §7.11, §3.4).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from model_crafter.inspect._common import (
    _coef_se_from_solution,
    _solution_is_ols,
)

__all__ = ["coefficients"]


def coefficients(sol: Any) -> pd.DataFrame:
    """Coefficient table indexed by design column.

    Columns: ``estimate``, ``std_error``, ``z``, ``p_value``. See the
    module docstring for the math and ``z`` interpretation per
    solver family.
    """
    from scipy.stats import norm, t  # noqa: PLC0415

    coef = sol.coefficients
    if not isinstance(coef, pd.Series):
        raise TypeError("sol.coefficients must be a pandas Series")

    se = _coef_se_from_solution(sol)
    n_cols = len(coef)

    if se is None:
        std_arr = np.full(n_cols, np.nan, dtype=float)
        z_arr = np.full(n_cols, np.nan, dtype=float)
        p_arr = np.full(n_cols, np.nan, dtype=float)
    else:
        std_arr = se.reindex(coef.index).to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            z_arr = coef.to_numpy(dtype=float) / std_arr

        if _solution_is_ols(sol):
            df_resid = int(
                sol.solver_info.get("df_resid", max(sol.n_obs - n_cols, 1))
            )
            p_arr = 2.0 * (1.0 - t.cdf(np.abs(z_arr), df=df_resid))
        else:
            # Wald: standard-normal tail probability.
            p_arr = 2.0 * (1.0 - norm.cdf(np.abs(z_arr)))

    return pd.DataFrame(
        {
            "estimate": coef.to_numpy(dtype=float),
            "std_error": std_arr,
            "z": z_arr,
            "p_value": p_arr,
        },
        index=coef.index,
    )
