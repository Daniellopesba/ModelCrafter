r"""Numerical helpers for OLS / WLS.

Centralises rank-deficiency detection (so the assumption framework can name
offending columns) and the actual least-squares solve. The implementation
follows ESL §3.2 and uses a QR factorisation for numerical stability —
:math:`X = QR`, :math:`\hat\beta = R^{-1} Q^\top y` — rather than forming
:math:`(X^\top X)^{-1}` directly, which squares the condition number of
:math:`X`.

For weighted least squares, ESL §3.2.3 derives the same equations from
:math:`X^\top W X \beta = X^\top W y`, equivalent to OLS on the scaled
problem :math:`\tilde X = W^{1/2} X`, :math:`\tilde y = W^{1/2} y`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import scipy.linalg as sla

__all__ = [
    "OLSFit",
    "find_rank_deficient_columns",
    "solve_ols",
]


@dataclass(frozen=True, slots=True)
class OLSFit:
    """Outputs of the OLS / WLS normal-equations solve.

    Attributes
    ----------
    beta:
        Estimated coefficient vector, shape ``(p,)``.
    se:
        Standard errors of the coefficients, shape ``(p,)``.
    fitted:
        Fitted values :math:`X\\hat\\beta`, shape ``(n,)``.
    residuals:
        :math:`y - X\\hat\\beta`, shape ``(n,)``. For WLS these are the
        *un*-weighted residuals on the original scale.
    rss:
        Weighted residual sum of squares :math:`\\sum_i w_i (y_i - \\hat y_i)^2`.
    sigma2:
        Unbiased estimate of :math:`\\sigma^2`,
        :math:`\\mathrm{RSS}/(n - p)` for OLS,
        :math:`\\sum w_i r_i^2 / (n - p)` for WLS — matches R's
        ``summary(lm)$sigma^2`` and statsmodels' ``OLS().fit().scale``.
    r_squared:
        Coefficient of determination relative to the (weighted) mean model.
    df_resid:
        Residual degrees of freedom ``n - p``.
    """

    beta: np.ndarray
    se: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray
    rss: float
    sigma2: float
    r_squared: float
    df_resid: int


def find_rank_deficient_columns(
    X: np.ndarray,
    column_names: tuple[str, ...],
    *,
    tol: float | None = None,
) -> tuple[str, ...]:
    """Identify columns of ``X`` that are linearly dependent on earlier ones.

    Uses pivoted QR (``scipy.linalg.qr(pivoting=True)``). Returns the names
    of columns whose absolute pivot-diagonal entry falls below ``tol``,
    flagged in the *original* column order. An empty tuple means full
    column rank.

    Tolerance defaults to ``max(n, p) * eps * |R[0, 0]|`` (NumPy's
    ``matrix_rank`` convention).
    """
    n, p = X.shape
    if p == 0:
        return ()
    # Pivoted QR: gives a permutation that orders columns by "informativeness".
    # The diagonal of R, in pivoted order, drops to ~0 at the first redundant
    # column. The columns flagged are those whose pivot-diagonal entry is
    # below tolerance.
    qr_result = cast(
        tuple[np.ndarray, np.ndarray, np.ndarray],
        sla.qr(X, mode="economic", pivoting=True),
    )
    _Q, R, piv = qr_result
    diag_R = np.abs(np.diag(R))
    if tol is None:
        if diag_R.size == 0:
            return ()
        tol = max(n, p) * np.finfo(float).eps * diag_R[0]
    # Indices (in pivoted order) flagged as redundant.
    redundant_pivot = np.where(diag_R <= tol)[0]
    # Map back to original column names.
    offending = [column_names[piv[i]] for i in redundant_pivot]
    return tuple(offending)


def solve_ols(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None = None,
) -> OLSFit:
    r"""Closed-form OLS / WLS solve via QR.

    Parameters
    ----------
    X:
        ``(n, p)`` design matrix. Must be full column rank — callers check
        rank with :func:`find_rank_deficient_columns` first and raise the
        appropriate ``AssumptionError`` upstream.
    y:
        ``(n,)`` target vector.
    weights:
        ``(n,)`` non-negative weight vector or ``None`` for OLS.

    Returns
    -------
    OLSFit
        See :class:`OLSFit`.

    Notes
    -----
    For OLS the standard errors come from the unbiased
    :math:`\hat\sigma^2 (X^\top X)^{-1}`, with :math:`\hat\sigma^2 =
    \mathrm{RSS}/(n - p)`. This matches R's ``lm()`` (and ``summary(lm)``'s
    Std. Error column) and statsmodels' ``OLS.fit().bse``.

    For WLS the covariance is :math:`\hat\sigma^2 (X^\top W X)^{-1}` with
    :math:`\hat\sigma^2 = \sum w_i r_i^2 / (n - p)` — matching ``statsmodels``'
    ``WLS.fit().bse`` and R's ``lm(..., weights=)``. Note that this is the
    *unscaled* WLS — it does not normalise weights to sum-to-n.
    """
    X = np.ascontiguousarray(X, dtype=float)
    y = np.ascontiguousarray(y, dtype=float)
    n, p = X.shape
    if y.shape != (n,):
        raise ValueError(f"y shape {y.shape} does not match X first dim {n}")

    if weights is None:
        Xw = X
        yw = y
        sqrtw = None
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (n,):
            raise ValueError(f"weights shape {w.shape} does not match X first dim {n}")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        if not np.any(w > 0):
            raise ValueError("weights must contain at least one positive value")
        sqrtw = np.sqrt(w)
        Xw = X * sqrtw[:, None]
        yw = y * sqrtw

    # QR solve. scipy's lstsq with gelsd is robust; for full-rank designs the
    # QR-direct path is simpler and matches R/statsmodels bit-for-bit.
    qr_result = cast(
        tuple[np.ndarray, np.ndarray],
        sla.qr(Xw, mode="economic"),
    )
    Q, R = qr_result
    qty = Q.T @ yw
    beta = sla.solve_triangular(R, qty, lower=False)

    fitted = X @ beta
    residuals = y - fitted
    if weights is None:
        rss = float(np.sum(residuals * residuals))
        df_resid = n - p
        sigma2 = float("nan") if df_resid <= 0 else rss / df_resid
        # (X^T X)^{-1} = R^{-1} R^{-T}
        Rinv = sla.solve_triangular(R, np.eye(p), lower=False)
        cov_unscaled = Rinv @ Rinv.T
        se = np.sqrt(np.maximum(0.0, sigma2 * np.diag(cov_unscaled)))
        # R^2 against the (unweighted) mean.
        tss = float(np.sum((y - y.mean()) ** 2))
        r_squared = 1.0 - rss / tss if tss > 0 else float("nan")
    else:
        # Weighted RSS on the scaled problem equals sum w * (y - Xb)^2.
        w = np.asarray(weights, dtype=float)
        wresid = sqrtw * residuals  # = sqrt(w) * (y - Xb)
        rss = float(np.sum(wresid * wresid))
        df_resid = n - p
        sigma2 = rss / df_resid if df_resid > 0 else float("nan")
        # cov = sigma^2 * (X^T W X)^{-1}.  Xw^T Xw = X^T W X, so reuse R.
        Rinv = sla.solve_triangular(R, np.eye(p), lower=False)
        cov_unscaled = Rinv @ Rinv.T
        se = np.sqrt(np.maximum(0.0, sigma2 * np.diag(cov_unscaled)))
        # R^2 weighted, against weighted mean of y. This matches statsmodels
        # WLS.rsquared (centered, uncentered=False).
        ybar_w = float(np.sum(w * y) / np.sum(w))
        tss = float(np.sum(w * (y - ybar_w) ** 2))
        r_squared = 1.0 - rss / tss if tss > 0 else float("nan")

    return OLSFit(
        beta=beta,
        se=se,
        fitted=fitted,
        residuals=residuals,
        rss=rss,
        sigma2=sigma2,
        r_squared=r_squared,
        df_resid=df_resid,
    )
