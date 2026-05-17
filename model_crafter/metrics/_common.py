"""Internal helpers shared by all metric primitives.

All public metric primitives accept ``(sol, data, weights=...)``. Internally
they need to:

1. Resolve ``weights=`` to a numpy array or ``None`` (uniform).
2. Pull the target column out of ``data`` via ``sol.spec.target``.
3. Pull predictions out via :func:`model_crafter.solve.predict`.

The from-arrays variant (``_*_from_arrays``) is exposed so unit tests can
hit the numerical core directly without constructing a real ``Solution``.

The package contract (DESIGN.md §3.3) is that ``mc.predict`` returns
probabilities for classification. The squared-error solver used for testing
returns raw linear predictions :math:`X\\beta`, which may fall slightly
outside :math:`[0, 1]`. Metrics that require probabilities clip to
``[eps, 1 - eps]`` before consumption; metrics that are scale-/range-
invariant (AUC, KS, Cohen's d, gains, lift) leave predictions untouched.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DEFAULT_EPS = 1e-12
"""Default clipping bound for probabilities entering ``log`` (log_loss, ECE)."""


def coerce_weights(
    weights: str | np.ndarray | pd.Series | None,
    data: pd.DataFrame,
    *,
    n: int | None = None,
) -> np.ndarray | None:
    """Resolve ``weights=`` to a 1-D float array (or ``None``).

    Mirrors :func:`model_crafter.solve._coerce_weights` to keep the
    metric API consistent with the solver API. We re-implement rather
    than import because :func:`solve._coerce_weights` is private.
    """
    if weights is None:
        return None
    if isinstance(weights, str):
        if weights not in data.columns:
            raise KeyError(
                f"weights column '{weights}' not in data "
                f"(columns: {list(data.columns)})"
            )
        w = data[weights].to_numpy(dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
    expected = len(data) if n is None else n
    if w.ndim != 1 or w.shape[0] != expected:
        raise ValueError(
            f"weights must be a 1-D array of length {expected}; got shape {w.shape}"
        )
    if not np.isfinite(w).all():
        raise ValueError("weights contain non-finite values (NaN / Inf)")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    if not np.any(w > 0):
        raise ValueError("weights must contain at least one positive value")
    return w


def resolve_scores_and_target(
    sol: Any,
    data: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(y, scores)`` for ``sol`` applied to ``data``.

    Calls :func:`model_crafter.solve.predict` to compute scores; reads the
    target via ``sol.spec.target``. Both are returned as 1-D ``np.ndarray``s
    of dtype ``float``.
    """
    # Lazy import to avoid a hard import-time dependency from metrics on solve.
    from model_crafter.solve import predict as _predict

    target_name = getattr(getattr(sol, "spec", None), "target", None)
    if target_name is None:
        raise AttributeError(
            "sol.spec.target is required to resolve the target column for metrics"
        )
    if target_name not in data.columns:
        raise KeyError(
            f"target column '{target_name}' not in data "
            f"(columns: {list(data.columns)})"
        )
    y_series = pd.to_numeric(data[target_name], errors="raise")
    y = np.asarray(y_series, dtype=float)
    scores = np.asarray(_predict(sol, data), dtype=float)
    if y.shape != scores.shape:
        raise ValueError(
            f"target shape {y.shape} != prediction shape {scores.shape}"
        )
    if not np.isfinite(y).all():
        raise ValueError(
            f"target column '{target_name}' contains non-finite values; "
            "drop or impute before computing metrics"
        )
    if not np.isfinite(scores).all():
        raise ValueError(
            "predictions contain non-finite values; check the fitted Solution"
        )
    return y, scores


def check_binary_target(y: np.ndarray) -> None:
    """Verify ``y`` is binary 0/1. Raises ``ValueError`` if not.

    Discrimination + calibration metrics in DESIGN.md §3.3 are defined for
    binary classification (probability outputs). A regression spec whose
    target is continuous is rejected informatively.
    """
    uniq = np.unique(y[np.isfinite(y)])
    if not np.all(np.isin(uniq, (0.0, 1.0))):
        raise ValueError(
            "binary target required (values must be 0 or 1); got unique values "
            f"{uniq.tolist()}. Discrimination and calibration metrics are only "
            "defined for classification (DESIGN.md §3.3)."
        )


def split_pos_neg(
    y: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Partition ``(scores, weights)`` into positives and negatives by ``y``."""
    pos_mask = y == 1.0
    s_pos = scores[pos_mask]
    s_neg = scores[~pos_mask]
    if weights is None:
        return s_pos, s_neg, None, None
    return s_pos, s_neg, weights[pos_mask], weights[~pos_mask]


def weighted_mean(
    x: np.ndarray, weights: np.ndarray | None
) -> float:
    """Weighted mean. ``weights=None`` is uniform."""
    if weights is None:
        return float(np.mean(x))
    total = float(np.sum(weights))
    if total == 0.0:
        raise ValueError("weighted_mean: sum of weights is zero")
    return float(np.sum(weights * x) / total)


def weighted_quantile(
    x: np.ndarray, q: np.ndarray, weights: np.ndarray | None
) -> np.ndarray:
    """Weighted quantile via inverse-CDF on the empirical weighted CDF.

    Matches the unweighted ``np.quantile`` with ``method='linear'`` when
    ``weights=None``. For ``q=0`` and ``q=1`` returns ``min`` and ``max``.
    """
    x = np.asarray(x, dtype=float)
    q = np.atleast_1d(np.asarray(q, dtype=float))
    if weights is None:
        weights = np.ones_like(x)
    if x.size == 0:
        return np.full_like(q, np.nan, dtype=float)
    order = np.argsort(x)
    x_sorted = x[order]
    w_sorted = np.asarray(weights, dtype=float)[order]
    cum = np.cumsum(w_sorted)
    total = cum[-1]
    if total == 0.0:
        raise ValueError("weighted_quantile: sum of weights is zero")
    # Use the same "linear" interpolation convention np.quantile uses:
    # positions are h = q*(n-1) on the unweighted ranks. For weighted data
    # we map to cumulative-weight fractions in (0, 1].
    cdf = (cum - 0.5 * w_sorted) / total
    out = np.interp(q, cdf, x_sorted, left=x_sorted[0], right=x_sorted[-1])
    return out
