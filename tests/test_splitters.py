"""Tests for temporal splitters (AGENTS.md Task P3.B, DESIGN.md §3.2).

Acceptance criteria covered here:

* ``expanding_window(time_col=..., gap="365D")`` produces folds where every
  validation window starts at least 365 days after its training window ends.
* ``rolling_window`` preserves a fixed training window size and respects
  ``gap``.
* ``purged_kfold`` purges a buffer around each fold's validation window.
* ``time_split`` produces chronological, non-overlapping slices in the
  requested ratios.

The splitter contract (DESIGN.md §3.2): each splitter exposes a ``.split(df)``
iterator that yields ``(train_df, valid_df)`` pairs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model_crafter.validation.splitters import (
    Splitter,
    expanding_window,
    purged_kfold,
    rolling_window,
    time_split,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _daily_panel(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """A daily-frequency synthetic panel with monotone time and a couple of
    feature columns. Shuffled-row-order on the way in so we exercise the
    splitter's sort path."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2020-01-01")
    dates = start + pd.to_timedelta(np.arange(n), unit="D")
    df = pd.DataFrame(
        {
            "origination_dt": dates,
            "x": rng.normal(size=n),
            "y": rng.normal(size=n),
        }
    )
    # Shuffle so the splitter must sort by time, not by row order.
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# time_split
# ---------------------------------------------------------------------------


def test_time_split_three_way_chronological():
    df = _daily_panel(n=120)
    train, valid, test = time_split(df, time_col="origination_dt",
                                    ratios=(0.7, 0.15, 0.15))
    # Sizes line up with ratios.
    assert len(train) == 84
    assert len(valid) == 18
    assert len(test) == 18
    # Chronological boundary: every train timestamp <= every valid timestamp.
    assert train["origination_dt"].max() < valid["origination_dt"].min()
    assert valid["origination_dt"].max() < test["origination_dt"].min()


def test_time_split_two_way():
    df = _daily_panel(n=100)
    train, holdout = time_split(df, time_col="origination_dt",
                                ratios=(0.8, 0.2))
    assert len(train) == 80
    assert len(holdout) == 20
    assert train["origination_dt"].max() < holdout["origination_dt"].min()


def test_time_split_rejects_invalid_ratios():
    df = _daily_panel(n=100)
    with pytest.raises(ValueError, match="ratios"):
        time_split(df, time_col="origination_dt", ratios=(0.5, 0.6))
    with pytest.raises(ValueError, match="ratios"):
        time_split(df, time_col="origination_dt", ratios=(0.0, 1.0))


def test_time_split_missing_column():
    df = _daily_panel(n=100)
    with pytest.raises(KeyError, match="not in data"):
        time_split(df, time_col="not_a_col", ratios=(0.7, 0.3))


# ---------------------------------------------------------------------------
# expanding_window
# ---------------------------------------------------------------------------


def test_expanding_window_respects_gap_acceptance():
    """ACCEPTANCE: every validation window starts at least ``gap`` after
    its training window ends (DESIGN.md §3.2)."""
    df = _daily_panel(n=2000)
    gap = pd.Timedelta("365D")
    splitter: Splitter = expanding_window(
        time_col="origination_dt",
        n_folds=5,
        horizon="90D",
        gap="365D",
        min_train="365D",
    )
    folds = list(splitter.split(df))
    assert len(folds) == 5
    for i, (train, valid) in enumerate(folds):
        assert len(train) > 0, f"fold {i}: empty train"
        assert len(valid) > 0, f"fold {i}: empty valid"
        train_end = train["origination_dt"].max()
        valid_start = valid["origination_dt"].min()
        assert valid_start >= train_end + gap, (
            f"fold {i}: gap violated: train_end={train_end!s}, "
            f"valid_start={valid_start!s}, gap={gap!s}"
        )


def test_expanding_window_train_set_grows():
    df = _daily_panel(n=2000)
    splitter = expanding_window(
        time_col="origination_dt",
        n_folds=4,
        horizon="120D",
        gap="30D",
        min_train="180D",
    )
    sizes = [len(train) for train, _ in splitter.split(df)]
    # Each train set must be strictly larger than the previous.
    assert sizes == sorted(sizes)
    assert sizes[0] < sizes[-1]


def test_expanding_window_horizon_fixed_length():
    df = _daily_panel(n=2000)
    horizon = pd.Timedelta("90D")
    splitter = expanding_window(
        time_col="origination_dt",
        n_folds=3,
        horizon="90D",
        gap="0D",
        min_train="90D",
    )
    for train, valid in splitter.split(df):
        v_min = valid["origination_dt"].min()
        v_max = valid["origination_dt"].max()
        # Validation span is at most one horizon (inclusive-exclusive).
        assert v_max - v_min < horizon


def test_expanding_window_carries_metadata():
    """``NoTemporalLeakage`` inspects ``time_col`` + ``gap`` from the
    splitter. The splitter exposes these as attributes."""
    splitter = expanding_window(
        time_col="origination_dt",
        n_folds=3,
        horizon="90D",
        gap="365D",
    )
    assert splitter.time_col == "origination_dt"
    assert splitter.gap == pd.Timedelta("365D")


# ---------------------------------------------------------------------------
# rolling_window
# ---------------------------------------------------------------------------


def test_rolling_window_fixed_train_size():
    df = _daily_panel(n=3000)
    train_size = pd.Timedelta("365D")
    splitter = rolling_window(
        time_col="origination_dt",
        train_size="365D",
        horizon="90D",
        step="90D",
        gap="0D",
    )
    folds = list(splitter.split(df))
    assert len(folds) > 0
    spans = []
    for train, _ in folds:
        span = train["origination_dt"].max() - train["origination_dt"].min()
        spans.append(span)
        assert span <= train_size + pd.Timedelta("1D")
    # All spans roughly equal (rolling, not expanding).
    assert max(spans) - min(spans) < pd.Timedelta("5D")


def test_rolling_window_respects_gap():
    df = _daily_panel(n=3000)
    gap = pd.Timedelta("60D")
    splitter = rolling_window(
        time_col="origination_dt",
        train_size="365D",
        horizon="30D",
        step="60D",
        gap="60D",
    )
    for i, (train, valid) in enumerate(splitter.split(df)):
        train_end = train["origination_dt"].max()
        valid_start = valid["origination_dt"].min()
        assert valid_start >= train_end + gap, f"fold {i}: gap violated"


def test_rolling_window_step_advances():
    df = _daily_panel(n=3000)
    splitter = rolling_window(
        time_col="origination_dt",
        train_size="365D",
        horizon="90D",
        step="90D",
        gap="0D",
    )
    starts = [train["origination_dt"].min() for train, _ in splitter.split(df)]
    assert len(starts) >= 2, "rolling window should yield more than one fold"
    diffs = [b - a for a, b in zip(starts[:-1], starts[1:], strict=True)]
    # Step is 90D (modulo discrete-day rounding to the available data).
    assert all(d > pd.Timedelta("60D") for d in diffs)


# ---------------------------------------------------------------------------
# purged_kfold
# ---------------------------------------------------------------------------


def test_purged_kfold_partitions_time():
    df = _daily_panel(n=1000)
    splitter = purged_kfold(
        time_col="origination_dt",
        n_folds=5,
        gap="30D",
    )
    folds = list(splitter.split(df))
    assert len(folds) == 5
    # Validation buckets are disjoint in time.
    valid_starts = [v["origination_dt"].min() for _, v in folds]
    valid_ends = [v["origination_dt"].max() for _, v in folds]
    # Sort by start to get them in time order.
    order = np.argsort([s.value for s in valid_starts])
    sorted_starts = [valid_starts[i] for i in order]
    sorted_ends = [valid_ends[i] for i in order]
    for i in range(len(folds) - 1):
        assert sorted_ends[i] < sorted_starts[i + 1]


def test_purged_kfold_purges_around_valid_window():
    df = _daily_panel(n=1000)
    gap = pd.Timedelta("30D")
    splitter = purged_kfold(
        time_col="origination_dt",
        n_folds=5,
        gap="30D",
    )
    for i, (train, valid) in enumerate(splitter.split(df)):
        v_lo = valid["origination_dt"].min()
        v_hi = valid["origination_dt"].max()
        train_times = train["origination_dt"]
        # No training row falls within [v_lo - gap, v_hi + gap].
        bad = train_times[(train_times >= v_lo - gap) & (train_times <= v_hi + gap)]
        assert len(bad) == 0, f"fold {i}: {len(bad)} training rows inside purge zone"


def test_purged_kfold_carries_metadata():
    splitter = purged_kfold(time_col="origination_dt", n_folds=4, gap="30D")
    assert splitter.time_col == "origination_dt"
    assert splitter.gap == pd.Timedelta("30D")
    assert getattr(splitter, "n_folds") == 4  # noqa: B009


# ---------------------------------------------------------------------------
# Argument validation (DESIGN.md §9.8 — eager input validation)
# ---------------------------------------------------------------------------


def test_expanding_window_rejects_bad_args():
    with pytest.raises(ValueError, match="n_folds"):
        expanding_window(
            time_col="t", n_folds=0, horizon="90D", gap="0D",
        )
    with pytest.raises(ValueError, match="horizon"):
        expanding_window(
            time_col="t", n_folds=3, horizon="-30D", gap="0D",
        )
    with pytest.raises(ValueError, match="gap"):
        expanding_window(
            time_col="t", n_folds=3, horizon="90D", gap="-1D",
        )


def test_purged_kfold_rejects_bad_args():
    with pytest.raises(ValueError, match="n_folds"):
        purged_kfold(time_col="t", n_folds=1, gap="0D")
