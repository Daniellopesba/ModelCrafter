r"""Closed-form linear-model diagnostics (DESIGN.md §5; ESL §3.3).

For OLS (and closed-form ridge), the hat matrix and its diagonal carry
the entire residual / leverage / Cook's-distance / DFBETAS apparatus:

.. math::

    H &= X (X^\top X)^{-1} X^\top
        \qquad \text{(OLS)} \\
    H_\lambda &= X (X^\top X + n \lambda I')^{-1} X^\top
        \qquad \text{(ridge; }I'\text{ zeroes the intercept row/col)}

    e_i &= y_i - \hat y_i \\
    h_{ii} &= H_{ii} \\
    \hat\sigma^2 &= \mathrm{RSS} / (n - p) \\
    D_i &= \frac{e_i^2}{p \hat\sigma^2} \cdot
             \frac{h_{ii}}{(1 - h_{ii})^2} \\
    r_i &= \frac{e_i}{\hat\sigma \sqrt{1 - h_{ii}}} \\
    \mathrm{DFBETAS}_{ij} &=
        \frac{\hat\beta_j - \hat\beta_j^{(-i)}}{\mathrm{SE}^{(-i)}(\hat\beta_j)}

We use the leave-one-out closed form

.. math::

    \hat\beta - \hat\beta^{(-i)} =
        \frac{(X^\top X)^{-1} x_i e_i}{1 - h_{ii}}

and Belsley-Kuh-Welsch's (1980) externally-studentized sigma to
standardise DFBETAS. Lasso, elastic net, and logistic raise
:class:`NotImplementedError` and point at :func:`mc.bootstrap`
(ESL §7.11) — that is the package's uncertainty / influence path for
non-closed-form fits.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from model_crafter.inspect._common import (
    _bootstrap_pointer,
    _solution_is_logistic,
    _solution_is_ols,
    _solution_supports_closed_form_hat,
)


@dataclass(frozen=True, slots=True)
class Diagnostics:
    """Residual / leverage / Cook's-distance bundle."""

    residuals: pd.Series
    leverage: pd.Series
    cooks_distance: pd.Series
    studentized_residuals: pd.Series
    sigma2: float

    def __repr__(self) -> str:
        n = len(self.residuals)
        lev = self.leverage
        high_lev = int((lev > 2.0 * lev.mean()).sum()) if n > 0 else 0
        cook = self.cooks_distance
        # Bollen-Jackman rule of thumb: "large" Cook's D > 4/n.
        cook_threshold = 4.0 / max(n, 1)
        high_cook = int((cook > cook_threshold).sum()) if n > 0 else 0
        lines = [
            "Diagnostics",
            f"  n={n}  sigma^2={self.sigma2:.4g}",
            f"  high leverage (h_ii > 2*mean): {high_lev}",
            f"  influential rows (D > 4/n={cook_threshold:.3g}): {high_cook}",
        ]
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class Influence:
    """DFBETAS + Cook's distance + leverage (ESL §3.3; Belsley-Kuh-Welsch 1980)."""

    dfbetas: pd.DataFrame
    cooks_distance: pd.Series
    leverage: pd.Series

    def __repr__(self) -> str:
        n_rows, n_cols = self.dfbetas.shape
        # |DFBETAS| > 2/sqrt(n) is the standard "large" threshold.
        n = max(n_rows, 1)
        thresh = 2.0 / np.sqrt(n)
        flagged = int((self.dfbetas.abs() > thresh).any(axis=1).sum())
        return (
            f"Influence(n_rows={n_rows}, n_coef={n_cols}, "
            f"rows_with_|dfbetas|>2/sqrt(n)={flagged})"
        )


def _rebuild_design_and_y(sol: Any, data: pd.DataFrame) -> tuple[
    pd.DataFrame, np.ndarray
]:
    """Build the design matrix and y vector from a fitted solution + the
    training frame.

    Solution does not persist X or y (DESIGN.md §6.2 keeps the value
    compact), so the caller must pass the same frame the solution was
    fit on. We verify by comparing reconstructed column names.
    """
    from model_crafter._internal.design import build_design  # noqa: PLC0415

    design = build_design(sol.spec, data, fit_state=sol.fit_state)
    if list(design.columns) != list(sol.design_columns):
        raise ValueError(
            "data does not reproduce the training design columns; "
            "diagnostics(sol, data) expects the same frame the solution "
            "was fit on. "
            f"Training columns: {list(sol.design_columns)}; "
            f"got: {list(design.columns)}"
        )
    target = sol.spec.target
    if target not in data.columns:
        raise KeyError(
            f"target column {target!r} not in data; "
            f"diagnostics(sol, data) needs the original training frame"
        )
    y_series = pd.to_numeric(data[target], errors="raise")
    y = np.asarray(y_series, dtype=float)
    return pd.DataFrame(design.values, columns=pd.Index(design.columns)), y


def hat_matrix(sol: Any, data: pd.DataFrame | None = None) -> np.ndarray:
    r"""Return the :math:`n \times n` hat matrix for a closed-form fit.

    O(n^2) memory — for large n prefer
    :func:`diagnostics`\\ ``(sol, data).leverage``, which materialises
    only the diagonal.
    """
    if not _solution_supports_closed_form_hat(sol):
        if _solution_is_logistic(sol):
            raise NotImplementedError(_bootstrap_pointer("hat_matrix"))
        raise NotImplementedError(_bootstrap_pointer("hat_matrix"))
    if data is None:
        raise TypeError(
            "hat_matrix(sol, data) requires the training data frame "
            "(Solution does not persist X)"
        )

    X, _y = _rebuild_design_and_y(sol, data)
    X_arr = X.to_numpy(dtype=float)
    n, _p = X_arr.shape

    if n > 5000:
        warnings.warn(
            f"hat_matrix returning an (n={n}) x (n={n}) dense array — "
            f"this is O(n^2) memory. For large n, prefer "
            f"`diagnostics(sol, data).leverage` which materialises only "
            f"the diagonal.",
            stacklevel=2,
        )

    if _solution_is_ols(sol):
        # H = X (X^T X)^-1 X^T.
        XtX = X_arr.T @ X_arr
        A = np.linalg.solve(XtX, X_arr.T)
        H = X_arr @ A
    else:
        # Ridge: H_lambda = X (X^T X + n*lambda*I')^-1 X^T,
        # with I' zeroing the intercept row/column.
        from model_crafter._internal.design import INTERCEPT_NAME  # noqa: PLC0415

        lam = float(getattr(sol.spec.penalty, "lam", 0.0))
        XtX = X_arr.T @ X_arr
        Iprime = np.eye(X_arr.shape[1], dtype=float)
        cols = list(sol.design_columns)
        if INTERCEPT_NAME in cols:
            idx = cols.index(INTERCEPT_NAME)
            Iprime[idx, idx] = 0.0
        A = np.linalg.solve(XtX + n * lam * Iprime, X_arr.T)
        H = X_arr @ A
    return np.asarray(H, dtype=float)


def diagnostics(sol: Any, data: pd.DataFrame | None = None) -> Diagnostics:
    r"""Residual / leverage / Cook's-distance diagnostics (closed-form only)."""
    if not _solution_supports_closed_form_hat(sol):
        raise NotImplementedError(_bootstrap_pointer("diagnostics"))
    if data is None:
        raise TypeError(
            "diagnostics(sol, data) requires the training data frame "
            "(Solution does not persist X)"
        )

    X, y = _rebuild_design_and_y(sol, data)
    X_arr = X.to_numpy(dtype=float)
    n, p = X_arr.shape
    cols = list(sol.design_columns)

    beta = sol.coefficients.reindex(cols).to_numpy(dtype=float)
    yhat = X_arr @ beta
    resid = y - yhat
    rss = float(np.sum(resid * resid))
    df_resid = max(n - p, 1)
    sigma2 = rss / df_resid

    H = hat_matrix(sol, data)
    leverage = np.diag(H)
    # 1 - h_ii blows up when h_ii = 1; surface as NaN.
    one_minus_h = np.where(leverage >= 1.0, np.nan, 1.0 - leverage)

    with np.errstate(divide="ignore", invalid="ignore"):
        cooks = (resid * resid / (p * sigma2)) * (
            leverage / (one_minus_h * one_minus_h)
        )
        studentized = resid / (np.sqrt(sigma2) * np.sqrt(one_minus_h))

    index = data.index
    return Diagnostics(
        residuals=pd.Series(resid, index=index, name="residual"),
        leverage=pd.Series(leverage, index=index, name="leverage"),
        cooks_distance=pd.Series(cooks, index=index, name="cooks_distance"),
        studentized_residuals=pd.Series(
            studentized, index=index, name="studentized_residual"
        ),
        sigma2=float(sigma2),
    )


def influence(sol: Any, data: pd.DataFrame | None = None) -> Influence:
    r"""Per-row DFBETAS + Cook's distance + leverage (closed-form only)."""
    if not _solution_supports_closed_form_hat(sol):
        raise NotImplementedError(_bootstrap_pointer("influence"))
    if data is None:
        raise TypeError(
            "influence(sol, data) requires the training data frame "
            "(Solution does not persist X)"
        )

    X, y = _rebuild_design_and_y(sol, data)
    X_arr = X.to_numpy(dtype=float)
    n, p = X_arr.shape
    cols = list(sol.design_columns)

    beta = sol.coefficients.reindex(cols).to_numpy(dtype=float)
    yhat = X_arr @ beta
    resid = y - yhat
    rss = float(np.sum(resid * resid))
    df_resid = max(n - p, 1)
    sigma2 = rss / df_resid

    XtX = X_arr.T @ X_arr
    XtX_inv = np.linalg.inv(XtX)
    H = X_arr @ XtX_inv @ X_arr.T
    leverage = np.diag(H)
    one_minus_h = np.where(leverage >= 1.0, np.nan, 1.0 - leverage)

    # Vectorised LOO coefficient change: A = (X^T X)^-1 X^T (shape (p, n))
    # scaled per-column by e_i / (1 - h_ii).
    A = XtX_inv @ X_arr.T
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = resid / one_minus_h
    delta_beta = A * scale[np.newaxis, :]

    # Internally-studentised residual r_i feeds the LOO sigma estimate.
    with np.errstate(divide="ignore", invalid="ignore"):
        studentized = resid / (np.sqrt(sigma2) * np.sqrt(one_minus_h))
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = df_resid - 1
        # sigma2_(-i) = sigma2 * (n - p - r_i^2) / (n - p - 1).
        sigma2_minus_i = sigma2 * (df_resid - studentized * studentized) / max(denom, 1)
        sigma2_minus_i = np.where(sigma2_minus_i <= 0, np.nan, sigma2_minus_i)

    # SE^{(-i)}(beta_j) = sqrt( sigma2_(-i) * [(X^T X)^-1]_{jj} ).
    diag_inv = np.diag(XtX_inv)
    se_minus_i = np.sqrt(np.outer(sigma2_minus_i, diag_inv))

    with np.errstate(divide="ignore", invalid="ignore"):
        dfbetas = delta_beta.T / se_minus_i

    with np.errstate(divide="ignore", invalid="ignore"):
        cooks = (resid * resid / (p * sigma2)) * (
            leverage / (one_minus_h * one_minus_h)
        )

    index = data.index
    return Influence(
        dfbetas=pd.DataFrame(dfbetas, index=index, columns=pd.Index(cols)),
        cooks_distance=pd.Series(cooks, index=index, name="cooks_distance"),
        leverage=pd.Series(leverage, index=index, name="leverage"),
    )
