r"""Lambda-path utilities for regularised linear models.

A lambda path is a *descending* sequence of regularisation strengths
:math:`\lambda_{\max} \geq \lambda_1 \geq \cdots \geq \lambda_n` along which
a coordinate-descent (or proximal) solver is fit with warm starts. The
convention follows ``glmnet`` (Friedman, Hastie & Tibshirani 2010,
*Regularization Paths for Generalized Linear Models via Coordinate
Descent*) and ESL §3.4.

The largest :math:`\lambda` is chosen so that the all-zero coefficient
vector is optimal — for a lasso fit on standardised :math:`X` and centred
:math:`y` (with no intercept penalised), the Karush–Kuhn–Tucker
conditions give

.. math::

   \lambda_{\max} \;=\; \frac{1}{n\,\alpha} \max_j \lvert x_j^\top y\rvert

with :math:`\alpha` the elastic-net mixing parameter (``alpha=1`` for lasso,
``alpha=0`` for ridge — although ``alpha=0`` would yield infinite
:math:`\lambda_{\max}` and is handled separately by the ridge solver).

The minimum :math:`\lambda` is a small fraction of :math:`\lambda_{\max}`
(``ratio * lambda_max``); the default ``ratio=1e-3`` follows ``glmnet``'s
``lambda.min.ratio`` for ``n > p``.

The returned grid is log-spaced and **descending** so callers can warm-start
from one fit to the next (large :math:`\lambda` → small :math:`\lambda`,
each previous coefficient vector seeds the next).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model_crafter._internal.design import build_design
from model_crafter.spec import LinearSpec


def log_grid(low: float, high: float, n: int) -> np.ndarray:
    r"""Return ``n`` log-spaced points from ``high`` *down to* ``low`` inclusive.

    The grid is descending: ``out[0] == high`` and ``out[-1] == low``. This
    is the order coordinate-descent path-fitters consume — warm starts go
    from large :math:`\lambda` (heavy regularisation, β ≈ 0) down to small
    :math:`\lambda` (light regularisation).

    Parameters
    ----------
    low, high:
        Strictly positive endpoints. ``low < high`` is required.
    n:
        Number of points. Must be ``>= 2``.

    Returns
    -------
    np.ndarray
        Descending 1-D array of length ``n``.
    """
    if not (np.isfinite(low) and np.isfinite(high)):
        raise ValueError(f"low / high must be finite; got low={low}, high={high}")
    if low <= 0 or high <= 0:
        raise ValueError(
            f"log_grid requires strictly positive endpoints; got low={low}, high={high}"
        )
    if low >= high:
        raise ValueError(
            f"log_grid requires low < high; got low={low}, high={high}"
        )
    if n < 2:
        raise ValueError(f"log_grid requires n >= 2; got n={n}")
    # geomspace returns ascending; flip to descending for warm-start order.
    asc = np.geomspace(low, high, num=n, dtype=float)
    return asc[::-1].copy()


def _extract_alpha(spec: LinearSpec) -> float:
    r"""Pull the elastic-net mixing parameter :math:`\alpha` from a spec's penalty.

    The convention (matching ``glmnet``) is

    .. math::

       \mathcal{R}(\beta) \;=\; \lambda \Bigl[\alpha \lVert\beta\rVert_1
                                 + \tfrac{1-\alpha}{2} \lVert\beta\rVert_2^2\Bigr]

    The :func:`lambda_path` helper *doesn't* re-parameterise: it accepts a
    spec whose penalty is L1, L1+L2 (elastic net), or L2, and returns the
    path-of-λ at which the configuration would be evaluated. To do that it
    only needs the *L1 weight* to compute :math:`\lambda_{\max}` — when L1
    is absent (pure ridge) :math:`\lambda_{\max}` is infinite and the path
    isn't well-defined; we raise in that case and direct the caller at the
    ridge solver.

    Returns ``alpha = lam_l1 / (lam_l1 + lam_l2)`` for an L1+L2 sum (so
    pure L1 yields ``alpha=1``), or ``1.0`` for plain L1.
    """
    penalty = spec.penalty
    # Plain L1
    if _is_l1(penalty):
        return 1.0
    # PenaltySum: extract parts
    if hasattr(penalty, "parts"):
        lam_l1 = 0.0
        lam_l2 = 0.0
        for p in penalty.parts:  # type: ignore[attr-defined]
            if _is_l1(p):
                lam_l1 += float(p.lam)
            elif _is_l2(p):
                lam_l2 += float(p.lam)
            else:
                raise TypeError(
                    f"lambda_path: unsupported penalty part {type(p).__name__}; "
                    "expected L1 / L2 / L1+L2 (elastic net)"
                )
        total = lam_l1 + lam_l2
        if total <= 0:
            raise ValueError(
                "lambda_path: penalty has zero total weight; nothing to vary"
            )
        return lam_l1 / total
    # Plain L2 — no L1 component, λ_max is undefined for the lasso path.
    if _is_l2(penalty):
        raise ValueError(
            "lambda_path: spec has only an L2 (ridge) penalty; the lasso "
            "lambda_max is undefined for pure ridge. Use a manual log_grid "
            "with the closed-form ridge solver instead."
        )
    raise TypeError(
        f"lambda_path: unsupported penalty type {type(penalty).__name__}; "
        "expected L1Penalty, L2Penalty, or PenaltySum"
    )


def _is_l1(p: object) -> bool:
    return type(p).__name__ == "L1Penalty" and hasattr(p, "lam")


def _is_l2(p: object) -> bool:
    return type(p).__name__ == "L2Penalty" and hasattr(p, "lam")


def compute_lambda_max(
    X: np.ndarray,
    y: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    alpha: float = 1.0,
    intercept: bool = True,
    standardise: bool = True,
) -> float:
    r"""Compute :math:`\lambda_{\max}` for an elastic-net path.

    For the standardised-X / centred-y convention used by ``glmnet`` and by
    this package's coordinate-descent solver,

    .. math::

       \lambda_{\max} \;=\; \frac{1}{n \alpha} \max_j \lvert x_j^\top y \rvert

    where the columns of :math:`X` have weighted mean 0 and weighted
    standard deviation 1, and :math:`y` is weighted-centred when an
    intercept is included. At :math:`\lambda \geq \lambda_{\max}` the
    all-zero coefficient vector satisfies the lasso KKT conditions, so the
    path can be warm-started from β=0 at the top of the grid.

    Parameters
    ----------
    X:
        ``(n, p)`` design matrix **excluding** any intercept column.
    y:
        ``(n,)`` target vector.
    weights:
        Optional ``(n,)`` non-negative sample weights.
    alpha:
        Elastic-net mixing parameter; must be in ``(0, 1]``. For ``alpha=0``
        (pure ridge) :math:`\lambda_{\max}` is infinite — caller should
        handle that case separately.
    intercept:
        When True the inner products are computed against weighted-centred
        ``y`` (and ``X`` is centred too if not standardised).
    standardise:
        When True the inner products use standardised ``X`` columns
        (mean 0, weighted SD 1). This is the ``glmnet`` convention.
    """
    if alpha <= 0:
        raise ValueError(
            f"compute_lambda_max requires alpha > 0; got alpha={alpha}. "
            "For pure ridge, lambda_max is undefined."
        )
    n = X.shape[0]
    if y.shape != (n,):
        raise ValueError(f"y shape {y.shape} does not match X first dim {n}")
    w = np.ones(n, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (n,):
        raise ValueError(f"weights shape {w.shape} does not match X first dim {n}")
    wsum = float(np.sum(w))
    if wsum <= 0:
        raise ValueError("weights sum to zero")

    # Centring
    if intercept:
        y_bar = float(np.sum(w * y) / wsum)
        y_c = y - y_bar
    else:
        y_c = y

    if standardise:
        # Weighted column means and SDs (population SD via mean of squared deviations).
        col_mean = (w @ X) / wsum
        Xc = X - col_mean
        col_var = (w @ (Xc * Xc)) / wsum
        col_sd = np.sqrt(col_var)
        # Avoid divide-by-zero on constant columns: they cannot enter the model
        # via the L1 penalty (their gradient is identically zero), but treat
        # their contribution as zero rather than NaN.
        safe_sd = np.where(col_sd > 0, col_sd, 1.0)
        Xn = Xc / safe_sd
        # Zero out columns that were constant — they contribute nothing.
        if np.any(col_sd == 0):
            Xn[:, col_sd == 0] = 0.0
    elif intercept:
        col_mean = (w @ X) / wsum
        Xn = X - col_mean
    else:
        Xn = X

    # Weighted inner products xj^T (W y) / n.  We use 1/n (not 1/wsum) to
    # match glmnet's convention (mean-1 normalised loss when w sum to n;
    # equivalent up to a rescaling of lambda otherwise).
    inner = Xn.T @ (w * y_c)
    return float(np.max(np.abs(inner)) / (n * alpha))


def lambda_path(
    spec: LinearSpec,
    data: pd.DataFrame,
    n: int = 100,
    ratio: float = 1e-3,
    *,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    r"""Return a descending log-spaced lambda grid for ``spec`` on ``data``.

    The grid spans :math:`[\lambda_{\max} \cdot \text{ratio},\; \lambda_{\max}]`
    in :math:`n` log-spaced points, **descending** so the first element is
    :math:`\lambda_{\max}` and the last is the smallest. This is the order
    a warm-started coordinate-descent path consumes (β=0 is the optimum at
    :math:`\lambda_{\max}`; each subsequent fit warm-starts from the prior).

    The :math:`\alpha` (L1 weight) of an L1+L2 elastic-net sum is read from
    the spec's penalty; for plain L1 :math:`\alpha=1`.

    Parameters
    ----------
    spec:
        A :class:`LinearSpec` whose penalty is an L1 / L1+L2 (elastic net)
        / L2 penalty. Pure-L2 is rejected with a pointer to the ridge
        solver, since :math:`\lambda_{\max}` is undefined.
    data:
        Training data frame. Materialised through the spec's terms.
    n:
        Number of grid points; default 100, matching ``glmnet``.
    ratio:
        Minimum-λ to maximum-λ ratio; default ``1e-3`` (``glmnet`` default
        when ``n > p``).
    weights:
        Optional sample weights, matching the ``solve(..., weights=...)``
        contract. Strings are NOT resolved — pass a numpy array.
    """
    alpha = _extract_alpha(spec)
    design = build_design(spec, data)
    if spec.target not in data.columns:
        raise KeyError(f"target column '{spec.target}' not in data")
    y = np.asarray(data[spec.target], dtype=float)

    # Drop the intercept column before computing lambda_max — the intercept
    # is unpenalised so it doesn't contribute to KKT.
    cols = list(design.columns)
    if spec.intercept and cols and cols[0] == "(Intercept)":
        X = design.values[:, 1:]
    else:
        X = design.values

    lam_max = compute_lambda_max(
        X, y, weights=weights, alpha=alpha, intercept=spec.intercept, standardise=True
    )
    if lam_max <= 0:
        # X^T y is identically zero — every coefficient is already optimal at 0
        # for any positive λ. Return a flat grid at a tiny positive value so
        # downstream callers can still iterate; pick eps so the grid is
        # well-defined.
        lam_max = float(np.finfo(float).tiny) * 10
    lam_min = max(lam_max * float(ratio), float(np.finfo(float).tiny) * 10)
    return log_grid(low=lam_min, high=lam_max, n=int(n))
