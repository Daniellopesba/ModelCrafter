r"""``over_time`` — time-indexed metric runner (DESIGN.md §3.2, AGENTS.md P3.B).

A fitted solution paired with a temporal splitter yields a sequence of
metric values, one per validation window. :func:`over_time` materialises
that sequence as a :class:`pandas.Series` keyed by the *midpoint of the
validation window*, which is the natural visualisation axis for PSI /
KS / AUC drift plots (DESIGN.md §3.3).

This is the building block underneath ``performance_over_time`` (P5.A) —
the runner here is intentionally minimal: a metric, a fitted solution,
the data, and the splitter.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd


def over_time(
    metric: Callable,
    sol: Any,
    data: pd.DataFrame,
    splitter: Any,
    *,
    weights: str | np.ndarray | pd.Series | None = None,
) -> pd.Series:
    r"""Run ``metric`` across ``splitter``'s validation windows.

    Parameters
    ----------
    metric
        Callable ``metric(sol, valid_window, weights=None) -> ResultLike``
        matching DESIGN.md §3.3.
    sol
        A fitted :class:`~model_crafter.solution.Solution` (or any value
        the metric can score).
    data
        Input frame; must contain the splitter's ``time_col``.
    splitter
        Any :class:`~model_crafter.validation.splitters.Splitter`.
    weights
        Sample weights — column name, array, or ``None``.

    Returns
    -------
    pandas.Series
        Index is the midpoint timestamp of each validation window;
        values are the metric outputs (floats). Series name is taken
        from ``metric.name`` / ``metric.__name__``.
    """
    if not callable(metric):
        raise TypeError("metric must be callable")
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be a DataFrame; got {type(data).__name__}")

    time_col = getattr(splitter, "time_col", None)
    if time_col is None:
        raise ValueError(
            "splitter has no time_col; over_time requires a temporal splitter"
        )

    full_weights = _resolve_weights(weights, data)
    metric_name = getattr(metric, "name", None) or getattr(
        metric, "__name__", "metric"
    )

    timestamps: list[pd.Timestamp] = []
    values: list[float] = []
    for _train, valid in splitter.split(data):
        if len(valid) == 0:
            continue
        valid_times = pd.to_datetime(valid[time_col])
        v_lo = valid_times.min()
        v_hi = valid_times.max()
        midpoint = v_lo + (v_hi - v_lo) / 2
        valid_w = (
            None
            if full_weights is None
            else pd.Series(full_weights, index=data.index)
            .reindex(valid.index)
            .to_numpy(dtype=float)
        )
        out = _call_metric(metric, sol, valid, valid_w)
        timestamps.append(midpoint)
        values.append(out)

    return pd.Series(values, index=pd.DatetimeIndex(timestamps), name=metric_name)


def _resolve_weights(
    weights: str | np.ndarray | pd.Series | None,
    data: pd.DataFrame,
) -> np.ndarray | None:
    if weights is None:
        return None
    if isinstance(weights, str):
        if weights not in data.columns:
            raise KeyError(
                f"weights column {weights!r} not in data (columns: "
                f"{list(data.columns)})"
            )
        return data[weights].to_numpy(dtype=float)
    arr = np.asarray(weights, dtype=float)
    if arr.shape[0] != len(data):
        raise ValueError(
            f"weights length {arr.shape[0]} != data length {len(data)}"
        )
    return arr


def _call_metric(
    fn: Callable,
    sol: Any,
    data: pd.DataFrame,
    weights: np.ndarray | None,
) -> float:
    try:
        out = fn(sol, data, weights=weights)
    except TypeError:
        out = fn(sol, data)
    if hasattr(out, "value"):
        return float(out.value)
    return float(out)
