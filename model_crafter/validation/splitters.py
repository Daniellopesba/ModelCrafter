r"""Temporal splitters (AGENTS.md Task P3.B, DESIGN.md §3.2).

A *splitter* partitions a time-indexed data frame into ``(train, valid)``
pairs that respect chronological order and an optional ``gap`` between the
end of training and the start of validation.

The contract is intentionally minimal — every splitter exposes:

* ``.split(df) -> Iterator[(train_df, valid_df)]`` — generator of fold pairs.
* ``time_col: str`` — name of the time column in ``df``.
* ``gap: pd.Timedelta`` — the buffer between train end and valid start.

These attributes are what :class:`~model_crafter.assumptions.temporal.NoTemporalLeakage`
inspects when verifying that a CV partition is leakage-free
(DESIGN.md §3.2: the gap exists because the label requires time to
mature — a 12-month default flag on a loan originated at ``t`` isn't
observable until ``t + 365D``).

Math / conventions
------------------

Let :math:`t_1 \leq t_2 \leq \dots \leq t_n` be the sorted timestamps in
the data and let :math:`g` be the ``gap``. A fold ``(T, V)`` is
*leakage-free* iff

.. math::

    \max_{i \in T} t_i + g \;\leq\; \min_{j \in V} t_j.

Every splitter in this module enforces that condition by construction;
:class:`~model_crafter.assumptions.temporal.NoTemporalLeakage` re-checks
it at run time so the assumption report contains an audit-grade record.

References
----------
ESL §7.10 — cross-validation in general.
López de Prado (2018) *Advances in Financial Machine Learning*, §7.4 —
purged k-fold cross-validation.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

__all__ = [
    "Splitter",
    "expanding_window",
    "purged_kfold",
    "rolling_window",
    "time_split",
]


# Protocol


@runtime_checkable
class Splitter(Protocol):
    """Minimal splitter surface.

    Attributes
    ----------
    time_col
        Column name of the timestamps.
    gap
        Pandas Timedelta enforced between train end and validation start.
        ``pd.Timedelta(0)`` means no gap.

    The presence of ``time_col`` and ``gap`` is part of the contract — the
    :class:`NoTemporalLeakage` assumption (DESIGN.md §3.2) reads them
    directly. Concrete implementations may carry additional metadata
    (``n_folds``, ``horizon``, ...) for diagnostics.
    """

    @property
    def time_col(self) -> str: ...

    @property
    def gap(self) -> pd.Timedelta: ...

    def split(self, df: pd.DataFrame) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]: ...


# Helpers — timedelta parsing + validation


def _to_timedelta(value: str | pd.Timedelta | None, *, name: str) -> pd.Timedelta:
    """Parse a pandas-offset alias (``"365D"``) or pass-through a
    :class:`pd.Timedelta`. ``None`` is rejected — callers must default
    explicitly when they want zero gap, so the parameter intent is loud.

    ``pd.NaT`` (the ``NaTType``) is also rejected: ``pd.Timedelta`` would
    parse e.g. ``float("nan")`` as ``NaT`` and we want a noisy error.
    """
    if value is None:
        raise ValueError(f"{name} is required (pass '0D' for no gap)")
    if isinstance(value, pd.Timedelta):
        return value
    try:
        td = pd.Timedelta(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"{name}={value!r} is not a valid pandas Timedelta / offset alias"
        ) from exc
    if not isinstance(td, pd.Timedelta):
        raise ValueError(
            f"{name}={value!r} parsed to {type(td).__name__} (not a valid Timedelta)"
        )
    return td


def _check_nonneg_td(td: pd.Timedelta, *, name: str) -> pd.Timedelta:
    if td < pd.Timedelta(0):
        raise ValueError(f"{name} must be non-negative; got {td}")
    return td


def _check_positive_td(td: pd.Timedelta, *, name: str) -> pd.Timedelta:
    if td <= pd.Timedelta(0):
        raise ValueError(f"{name} must be strictly positive; got {td}")
    return td


def _require_time_col(df: pd.DataFrame, time_col: str) -> pd.DatetimeIndex:
    if time_col not in df.columns:
        raise KeyError(
            f"time_col {time_col!r} not in data (columns: {list(df.columns)})"
        )
    ts = pd.to_datetime(df[time_col], errors="raise")
    return pd.DatetimeIndex(ts)


def _index_min(ts: pd.DatetimeIndex) -> pd.Timestamp:
    """Return ``ts.min()`` as a :class:`pd.Timestamp`, raising on NaT.

    Pyright models ``DatetimeIndex.min()`` as ``Timestamp | NaTType``; we
    handle the NaT branch explicitly so downstream Timestamp arithmetic
    is well-typed.
    """
    out = ts.min()
    if not isinstance(out, pd.Timestamp):
        raise ValueError("time column has no non-NaT values")
    return out


def _index_max(ts: pd.DatetimeIndex) -> pd.Timestamp:
    out = ts.max()
    if not isinstance(out, pd.Timestamp):
        raise ValueError("time column has no non-NaT values")
    return out


# time_split — single chronological split into k slices


def time_split(
    df: pd.DataFrame,
    *,
    time_col: str,
    ratios: Sequence[float],
) -> tuple[pd.DataFrame, ...]:
    """Single chronological split of ``df`` into ``len(ratios)`` slices.

    Rows are sorted by ``time_col`` and then sliced into contiguous,
    non-overlapping chunks whose sizes follow ``ratios``. ``ratios`` must
    be strictly positive and sum to 1.0 (within floating-point tolerance).

    The return is a tuple of DataFrames, one per requested slice, each
    with its original (re-sorted-by-time) row order preserved. Index
    values are not reset — callers can ``.reset_index(drop=True)`` if
    they prefer integer indexing.

    Parameters
    ----------
    df
        Input frame; must contain ``time_col``.
    time_col
        Column name of the timestamps.
    ratios
        Sequence of floats in :math:`(0, 1]` summing to 1.

    Raises
    ------
    KeyError
        If ``time_col`` is missing.
    ValueError
        If ``ratios`` is empty, contains a non-positive entry, or does
        not sum to 1.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a DataFrame; got {type(df).__name__}")
    if not ratios:
        raise ValueError("ratios must be a non-empty sequence")
    ratios_arr = np.asarray(ratios, dtype=float)
    if np.any(ratios_arr <= 0):
        raise ValueError(f"ratios must all be strictly positive; got {list(ratios)}")
    total = float(ratios_arr.sum())
    if not np.isclose(total, 1.0, atol=1e-9):
        raise ValueError(f"ratios must sum to 1.0; got sum={total}")

    times = _require_time_col(df, time_col)
    order = np.argsort(times.values, kind="stable")
    df_sorted = df.iloc[order]
    n = len(df_sorted)

    # Build cumulative integer boundaries from the ratios. Using
    # np.cumsum then floor keeps the last slice exact (n).
    cum = np.cumsum(ratios_arr)
    boundaries = np.rint(cum * n).astype(int)
    boundaries[-1] = n  # absorb floor rounding into the last slice

    pieces: list[pd.DataFrame] = []
    start = 0
    for end in boundaries:
        pieces.append(df_sorted.iloc[start:end])
        start = int(end)
    return tuple(pieces)


# expanding_window


@dataclass(frozen=True, slots=True)
class _ExpandingWindow:
    r"""Expanding-window walk-forward splitter (DESIGN.md §3.2).

    For each fold :math:`k`:

    * Training window: ``[t_min, train_end_k]`` where ``train_end_k``
      advances by ``horizon`` from one fold to the next.
    * Validation window: ``(train_end_k + gap, train_end_k + gap + horizon]``.

    All folds train from the start of the series; only the right edge
    of the train window moves. ``min_train`` sets the minimum span of
    the first fold's train window.
    """

    time_col: str
    n_folds: int
    horizon: pd.Timedelta
    gap: pd.Timedelta
    min_train: pd.Timedelta

    def split(
        self, df: pd.DataFrame
    ) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        times = _require_time_col(df, self.time_col)
        order = np.argsort(times.values, kind="stable")
        df_sorted = df.iloc[order]
        ts = pd.DatetimeIndex(pd.to_datetime(df_sorted[self.time_col]))
        t0 = _index_min(ts)
        # Fold k (1..n_folds): train_end = t0 + min_train + (k-1) * horizon.
        for k in range(self.n_folds):
            train_end = t0 + self.min_train + k * self.horizon
            valid_start = train_end + self.gap
            valid_end = valid_start + self.horizon
            train_mask = ts <= train_end
            valid_mask = (ts > valid_start) & (ts <= valid_end)
            # Pandas Timestamp comparison vs DatetimeIndex returns a numpy
            # bool array; let's coerce explicitly for indexing.
            train_df = df_sorted.iloc[np.asarray(train_mask)]
            valid_df = df_sorted.iloc[np.asarray(valid_mask)]
            yield train_df, valid_df


def expanding_window(
    *,
    time_col: str,
    n_folds: int,
    horizon: str | pd.Timedelta,
    gap: str | pd.Timedelta,
    min_train: str | pd.Timedelta | None = None,
) -> Splitter:
    """Construct an expanding-window splitter (DESIGN.md §3.2).

    Parameters
    ----------
    time_col
        Column name of the timestamps in the frame passed to ``split``.
    n_folds
        Number of folds (strictly positive).
    horizon
        Length of each validation window (pandas offset alias like
        ``"90D"`` or :class:`pd.Timedelta`).
    gap
        Buffer between train end and validation start. ``"0D"`` for none.
    min_train
        Minimum training span before the first fold. Defaults to one
        ``horizon`` (so the first train window has at least that much
        history).
    """
    if not isinstance(time_col, str) or not time_col:
        raise ValueError("time_col must be a non-empty string")
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1; got {n_folds}")
    horizon_td = _check_positive_td(_to_timedelta(horizon, name="horizon"),
                                    name="horizon")
    gap_td = _check_nonneg_td(_to_timedelta(gap, name="gap"), name="gap")
    if min_train is None:
        min_train_td = horizon_td
    else:
        min_train_td = _check_positive_td(
            _to_timedelta(min_train, name="min_train"), name="min_train"
        )
    return _ExpandingWindow(
        time_col=time_col,
        n_folds=int(n_folds),
        horizon=horizon_td,
        gap=gap_td,
        min_train=min_train_td,
    )


# rolling_window


@dataclass(frozen=True, slots=True)
class _RollingWindow:
    r"""Fixed-width sliding window splitter (DESIGN.md §3.2).

    For each fold :math:`k`:

    * Training window:
      ``(train_end_k - train_size, train_end_k]``.
    * Validation window:
      ``(train_end_k + gap, train_end_k + gap + horizon]``.

    Between folds, the window slides forward by ``step``. The number of
    folds is computed from the data span so the last validation window
    fits inside the data.
    """

    time_col: str
    train_size: pd.Timedelta
    horizon: pd.Timedelta
    step: pd.Timedelta
    gap: pd.Timedelta

    def split(
        self, df: pd.DataFrame
    ) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        times = _require_time_col(df, self.time_col)
        order = np.argsort(times.values, kind="stable")
        df_sorted = df.iloc[order]
        ts = pd.DatetimeIndex(pd.to_datetime(df_sorted[self.time_col]))
        t0 = _index_min(ts)
        t_max = _index_max(ts)
        # Start train_end at t0 + train_size and advance by step until
        # train_end + gap + horizon exceeds t_max.
        train_end = t0 + self.train_size
        while train_end + self.gap + self.horizon <= t_max + self.step:
            valid_start = train_end + self.gap
            valid_end = valid_start + self.horizon
            train_mask = (ts > train_end - self.train_size) & (ts <= train_end)
            valid_mask = (ts > valid_start) & (ts <= valid_end)
            if not np.any(np.asarray(train_mask)) or not np.any(np.asarray(valid_mask)):
                train_end = train_end + self.step
                continue
            yield df_sorted.iloc[np.asarray(train_mask)], df_sorted.iloc[
                np.asarray(valid_mask)
            ]
            train_end = train_end + self.step


def rolling_window(
    *,
    time_col: str,
    train_size: str | pd.Timedelta,
    horizon: str | pd.Timedelta,
    step: str | pd.Timedelta,
    gap: str | pd.Timedelta,
) -> Splitter:
    """Construct a rolling-window splitter (DESIGN.md §3.2).

    Parameters mirror :func:`expanding_window`, with ``train_size`` taking
    the place of ``min_train`` and ``step`` controlling the slide
    distance between folds.
    """
    if not isinstance(time_col, str) or not time_col:
        raise ValueError("time_col must be a non-empty string")
    train_size_td = _check_positive_td(
        _to_timedelta(train_size, name="train_size"), name="train_size"
    )
    horizon_td = _check_positive_td(_to_timedelta(horizon, name="horizon"),
                                    name="horizon")
    step_td = _check_positive_td(_to_timedelta(step, name="step"), name="step")
    gap_td = _check_nonneg_td(_to_timedelta(gap, name="gap"), name="gap")
    return _RollingWindow(
        time_col=time_col,
        train_size=train_size_td,
        horizon=horizon_td,
        step=step_td,
        gap=gap_td,
    )


# purged_kfold


@dataclass(frozen=True, slots=True)
class _PurgedKFold:
    r"""López de Prado–style purged k-fold splitter (DESIGN.md §3.2,
    *Advances in Financial Machine Learning*, §7.4).

    Time is partitioned into ``n_folds`` contiguous buckets. For each
    fold ``k`` the validation set is bucket ``k`` and the training set
    is all rows in the other buckets whose timestamp is *outside* the
    purge zone ``[v_lo - gap, v_hi + gap]`` around the validation
    window. The purge prevents label leakage from observations whose
    label horizon overlaps with the validation window.
    """

    time_col: str
    n_folds: int
    gap: pd.Timedelta

    def split(
        self, df: pd.DataFrame
    ) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        times = _require_time_col(df, self.time_col)
        order = np.argsort(times.values, kind="stable")
        df_sorted = df.iloc[order]
        ts = pd.DatetimeIndex(pd.to_datetime(df_sorted[self.time_col]))
        n = len(df_sorted)
        # Bucket assignment: equal-row buckets in time order so each fold
        # has the same number of validation rows.
        bucket_edges = np.linspace(0, n, num=self.n_folds + 1, dtype=int)
        for k in range(self.n_folds):
            lo, hi = int(bucket_edges[k]), int(bucket_edges[k + 1])
            if hi <= lo:
                continue
            valid_idx = np.arange(lo, hi)
            v_lo = ts[lo]
            v_hi = ts[hi - 1]
            train_mask = (ts < v_lo - self.gap) | (ts > v_hi + self.gap)
            train_mask_arr = np.asarray(train_mask)
            train_df = df_sorted.iloc[train_mask_arr]
            valid_df = df_sorted.iloc[valid_idx]
            yield train_df, valid_df


def purged_kfold(
    *,
    time_col: str,
    n_folds: int,
    gap: str | pd.Timedelta,
) -> Splitter:
    """Construct a purged k-fold splitter (DESIGN.md §3.2)."""
    if not isinstance(time_col, str) or not time_col:
        raise ValueError("time_col must be a non-empty string")
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2; got {n_folds}")
    gap_td = _check_nonneg_td(_to_timedelta(gap, name="gap"), name="gap")
    return _PurgedKFold(time_col=time_col, n_folds=int(n_folds), gap=gap_td)


# Common helper for downstream consumers (cross_validate / NoTemporalLeakage)


def splitter_time_col(splitter: Any) -> str | None:
    """Return ``splitter.time_col`` if it has one, else ``None``."""
    return getattr(splitter, "time_col", None)


_ZERO_TD: pd.Timedelta = pd.Timedelta(0)  # type: ignore[assignment]


def splitter_gap(splitter: Any) -> pd.Timedelta:
    """Return ``splitter.gap`` or :class:`pd.Timedelta(0)` if absent."""
    g = getattr(splitter, "gap", None)
    if g is None:
        return _ZERO_TD
    if isinstance(g, pd.Timedelta):
        return g
    out = pd.Timedelta(g)
    if not isinstance(out, pd.Timedelta):
        return _ZERO_TD
    return out
