"""Tests for binning strategies (DESIGN.md §3.1, AGENTS.md Task P4.B).

Acceptance criteria quoted from the task brief:

5. ``mc.categorical(group_rare=0.01)`` groups levels appearing in
   < 1% of rows into a "RARE" bucket; the test pins this on a small
   example.
6. ``mc.manual(edges=[a, b, c])`` produces 4 bins (open on the left,
   closed on the right) and the implementation matches ``pd.cut``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model_crafter.terms.woe import (
    MISSING_BIN_LABEL,
    RARE_CATEGORY_LABEL,
    BinningResult,
    CategoricalBinning,
    categorical,
    manual,
    monotonic,
    tree_bins,
)

# ---------------------------------------------------------------------------
# Strategy constructors return frozen dataclass values
# ---------------------------------------------------------------------------


def test_monotonic_constructor_validates_min_bin_size():
    with pytest.raises(ValueError, match="min_bin_size"):
        monotonic(min_bin_size=0.0)
    with pytest.raises(ValueError, match="min_bin_size"):
        monotonic(min_bin_size=1.0)


def test_monotonic_constructor_validates_max_bins():
    with pytest.raises(ValueError, match="max_bins"):
        monotonic(max_bins=1)


def test_tree_constructor_validates_inputs():
    with pytest.raises(ValueError, match="max_leaves"):
        tree_bins(max_leaves=1)
    with pytest.raises(ValueError, match="min_samples_leaf"):
        tree_bins(min_samples_leaf=0.0)
    with pytest.raises(ValueError, match="min_samples_leaf"):
        tree_bins(min_samples_leaf=0.5)


def test_categorical_constructor_validates_group_rare():
    with pytest.raises(ValueError, match="group_rare"):
        categorical(group_rare=-0.1)
    with pytest.raises(ValueError, match="group_rare"):
        categorical(group_rare=1.0)


def test_manual_constructor_validates_edges():
    with pytest.raises(ValueError):
        manual(edges=[])
    with pytest.raises(ValueError, match="strictly increasing"):
        manual(edges=[1.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        manual(edges=[2.0, 1.0])
    with pytest.raises(ValueError, match="finite"):
        manual(edges=[1.0, float("inf")])


def test_strategy_values_are_frozen():
    s = monotonic()
    with pytest.raises((AttributeError, Exception)):
        # frozen dataclass — assignment must fail
        s.min_bin_size = 0.99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Manual binning — Acceptance #6
# ---------------------------------------------------------------------------


def test_manual_three_edges_yields_four_bins():
    """Acceptance #6: manual(edges=[a, b, c]) produces 4 bins, open-left/closed-right."""
    x = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    y = np.array([0, 1, 0, 1, 0, 1])
    strategy = manual(edges=[15.0, 35.0, 55.0])
    result = strategy.fit(x, y, column="x")
    # Without missing values, exactly 4 bins.
    assert result.n_bins == 4
    # Labels reflect (-inf, 15], (15, 35], (35, 55], (55, +inf].
    assert result.bin_labels[0] == "(-inf, 15]"
    assert result.bin_labels[1] == "(15, 35]"
    assert result.bin_labels[2] == "(35, 55]"
    assert result.bin_labels[3] == "(55, +inf]"


def test_manual_matches_pd_cut_assignment():
    """Acceptance #6: assignment matches pd.cut semantics (open-left, closed-right)."""
    x_arr = np.array([10.0, 15.0, 16.0, 35.0, 36.0, 55.0, 56.0])
    y = np.zeros_like(x_arr, dtype=int)
    edges = [15.0, 35.0, 55.0]
    strategy = manual(edges=edges)
    result = strategy.fit(pd.Series(x_arr), y, column="x")
    # Per pd.cut conventions:
    # 10.0 → bin 0; 15.0 → bin 0 (right-closed); 16.0 → bin 1;
    # 35.0 → bin 1; 36.0 → bin 2; 55.0 → bin 2; 56.0 → bin 3.
    from model_crafter.terms.woe import _assign_bin_index_numeric

    idx = _assign_bin_index_numeric(pd.Series(x_arr), result)
    assert list(idx) == [0, 0, 1, 1, 2, 2, 3]


def test_manual_counts_match_pd_cut_directly():
    """Verify our counting agrees with pd.cut on a regression-style example."""
    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(0, 1, n)
    y = (rng.random(n) < 0.5).astype(int)
    edges = [-0.5, 0.0, 0.5]
    result = manual(edges=edges).fit(pd.Series(x), y, column="x")
    # Reference via pd.cut.
    full_edges = [-np.inf] + edges + [np.inf]
    ref_bins = pd.cut(x, bins=full_edges, include_lowest=True, labels=False)
    ref_arr = np.asarray(ref_bins, dtype=int)
    for b in range(4):
        ev = int(((ref_arr == b) & (y == 1)).sum())
        ne = int(((ref_arr == b) & (y == 0)).sum())
        assert result.n_events[b] == ev
        assert result.n_nonevents[b] == ne


# ---------------------------------------------------------------------------
# Categorical binning — Acceptance #5
# ---------------------------------------------------------------------------


def test_categorical_groups_rare_levels_into_RARE_bucket():
    """Acceptance #5: levels < 1% of rows go to RARE."""
    # 100 rows: 50 "A", 49 "B", 1 "C". With group_rare=0.01, "C" is rare.
    # Build with group_rare such that 1/100 < group_rare (1%) means equal,
    # not less-than. Use group_rare=0.02 to make "C" qualify (1% < 2%).
    x = pd.Series(["A"] * 50 + ["B"] * 49 + ["C"] * 1)
    y = np.array([0] * 50 + [1] * 49 + [1])
    result = categorical(group_rare=0.02).fit(x, y, column="region")
    # Categories should be [B/A in freq order] + [RARE].
    assert RARE_CATEGORY_LABEL in result.categories
    assert "C" not in result.categories  # folded into RARE
    assert "A" in result.categories
    assert "B" in result.categories


def test_categorical_group_rare_zero_keeps_every_level():
    """group_rare=0 disables rare-grouping."""
    x = pd.Series(["A"] * 99 + ["C"] * 1)
    y = np.array([0] * 99 + [1])
    result = categorical(group_rare=0.0).fit(x, y, column="region")
    # No RARE bucket; both A and C survive as their own bins.
    assert RARE_CATEGORY_LABEL not in result.categories
    assert "A" in result.categories
    assert "C" in result.categories


def test_categorical_handles_missing_values():
    """NaN in a categorical feature gets a (Missing) bin."""
    x = pd.Series(["A", "A", None, "B", "B", None, "A", "B"])
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    result = categorical(group_rare=0.0).fit(x, y, column="region")
    assert result.has_missing_bin
    assert result.bin_labels[-1] == MISSING_BIN_LABEL


# ---------------------------------------------------------------------------
# Monotonic binning behaviour
# ---------------------------------------------------------------------------


def test_monotonic_falls_back_to_single_bin_on_constant_x():
    x = pd.Series([3.0] * 100)
    y = np.array([0, 1] * 50)
    result = monotonic(min_bin_size=0.10).fit(x, y, column="x")
    # All values are 3.0 → no useful interior edges → one bin.
    assert result.n_bins == 1


def test_monotonic_creates_missing_bin_when_nan_present():
    x = pd.Series([1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 30)
    y = np.array([0, 0, 0, 1, 1, 0, 1, 1, 1, 1] * 30)
    result = monotonic(min_bin_size=0.05).fit(x, y, column="x")
    assert result.has_missing_bin
    assert result.bin_labels[-1] == MISSING_BIN_LABEL


# ---------------------------------------------------------------------------
# Tree binning behaviour
# ---------------------------------------------------------------------------


def test_tree_binning_produces_at_most_max_leaves_bins():
    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(0, 1, n)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-x))).astype(int)
    result = tree_bins(max_leaves=4, min_samples_leaf=0.10).fit(
        pd.Series(x), y, column="x"
    )
    # n_bins counts numeric leaves + optional (Missing) bin.
    n_numeric = result.n_bins - (1 if result.has_missing_bin else 0)
    assert n_numeric <= 4


def test_tree_binning_respects_min_samples_leaf():
    rng = np.random.default_rng(1)
    n = 500
    x = rng.normal(0, 1, n)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-x))).astype(int)
    result = tree_bins(max_leaves=10, min_samples_leaf=0.10).fit(
        pd.Series(x), y, column="x"
    )
    min_count = int(0.10 * n)
    for ev, ne in zip(result.n_events, result.n_nonevents, strict=True):
        assert ev + ne >= min_count or ev + ne == 0  # 0 only if all-NaN; not here


# ---------------------------------------------------------------------------
# Strategy.fit returns a BinningResult value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "strategy",
    [
        monotonic(min_bin_size=0.10),
        tree_bins(max_leaves=4, min_samples_leaf=0.10),
        categorical(group_rare=0.0),
        manual(edges=[0.0]),
    ],
)
def test_strategy_fit_returns_binning_result(strategy):
    rng = np.random.default_rng(0)
    n = 200
    if isinstance(strategy, CategoricalBinning):
        x = pd.Series(rng.choice(["A", "B", "C"], size=n))
    else:
        x = pd.Series(rng.normal(0, 1, n))
    y = rng.integers(0, 2, n)
    result = strategy.fit(x, y, column="feat")
    assert isinstance(result, BinningResult)
    assert result.column == "feat"
    assert result.n_bins >= 1
    assert len(result.bin_labels) == result.n_bins
    assert len(result.n_events) == result.n_bins
    assert len(result.n_nonevents) == result.n_bins
    assert len(result.woe) == result.n_bins
