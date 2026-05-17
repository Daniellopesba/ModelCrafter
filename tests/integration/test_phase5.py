"""Phase 5 integration tests.

Quotes AGENTS.md Task P5.INTEG acceptance:

  The full §10 north-star example runs end-to-end — temporal + segmented +
  comparative performance analysis on top of the Phase-4 WoE-logistic
  pipeline.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import model_crafter as mc


def _panel(n: int = 600, seed: int = 19) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    income = rng.normal(size=n)
    age = rng.normal(size=n)
    tenure = rng.normal(size=n)
    segment = rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
    seg_effect = pd.Series(segment).map({"A": -0.2, "B": 0.1, "C": 0.3}).to_numpy()
    eta = -1.0 + 0.8 * income - 0.5 * age + 0.3 * tenure + seg_effect
    p = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, p).astype(float)
    origination_dt = pd.date_range("2022-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "default_12m": y,
            "income": income,
            "age": age,
            "tenure": tenure,
            "segment": segment,
            "origination_dt": origination_dt,
        }
    )


def _fitted_logistic_sol():
    df = _panel(n=600, seed=11)
    spec = mc.linear(
        target="default_12m",
        features=["income", "age", "tenure"],
        loss=mc.logistic,
        penalty=mc.l2(0.1),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = mc.solve(spec, df)
    return sol, df


def test_performance_over_time_summary_has_one_row_per_window() -> None:
    """mc.performance_over_time returns a TemporalPerformanceReport whose
    summary DataFrame has one row per validation window."""
    sol, df = _fitted_logistic_sol()
    splitter = mc.purged_kfold(time_col="origination_dt", n_folds=4, gap="0D")
    perf_t = mc.performance_over_time(sol, df, splitter)
    assert len(perf_t.summary) == 4
    assert len(perf_t.reports) == 4
    assert "auc" in perf_t.summary.columns


def test_performance_by_segment_aggregates_to_full_data() -> None:
    """Per-segment n_obs sums to the aggregate n_obs."""
    sol, df = _fitted_logistic_sol()
    perf_seg = mc.performance_by_segment(sol, df, by="segment")
    seg_total = sum(r.n_obs for r in perf_seg.segments.values())
    assert seg_total == perf_seg.aggregate.n_obs
    # Three segments populated, matching the data.
    assert set(perf_seg.segments.keys()) == {"A", "B", "C"}


def test_compare_two_logistic_solutions() -> None:
    """compare() returns a Comparison with pairwise DeLong p-values and
    per-solution PerformanceReports."""
    df = _panel(n=400, seed=23)
    spec_a = mc.linear(
        target="default_12m",
        features=["income", "age"],
        loss=mc.logistic,
        penalty=mc.l2(0.01),
    )
    spec_b = mc.linear(
        target="default_12m",
        features=["income", "age", "tenure"],
        loss=mc.logistic,
        penalty=mc.l2(1.0),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol_a = mc.solve(spec_a, df)
        sol_b = mc.solve(spec_b, df)

    cmp = mc.compare({"baseline": sol_a, "challenger": sol_b}, data=df)
    assert set(cmp.reports.keys()) == {"baseline", "challenger"}
    # DeLong p-values: 2×2 with NaN diagonal and symmetric off-diagonals.
    pvals = cmp.delong_pvalues
    assert pvals.shape == (2, 2)
    assert pd.isna(pvals.loc["baseline", "baseline"])
    assert pd.isna(pvals.loc["challenger", "challenger"])
    p_ab = pvals.loc["baseline", "challenger"]
    p_ba = pvals.loc["challenger", "baseline"]
    assert float(p_ab) == float(p_ba)
    # Wrapping the existing primitive — should match mc.delong_test exactly.
    direct = float(mc.delong_test(sol_a, sol_b, df))
    assert float(p_ab) == direct


def test_north_star_temporal_plus_segmented_plus_compare() -> None:
    """The §10 north-star sequence: solve → performance → over_time →
    by_segment → compare. Smoke test that everything composes."""
    df = _panel(n=500, seed=29)
    spec = mc.linear(
        target="default_12m",
        features=["income", "age", "tenure"],
        loss=mc.logistic,
        penalty=mc.l2(0.1),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = mc.solve(spec, df)

    perf = mc.performance(sol, df)
    assert perf.n_obs == len(df)

    splitter = mc.purged_kfold(time_col="origination_dt", n_folds=3, gap="0D")
    perf_t = mc.performance_over_time(sol, df, splitter)
    assert len(perf_t.reports) == 3

    perf_seg = mc.performance_by_segment(sol, df, by="segment")
    assert set(perf_seg.segments.keys()) == {"A", "B", "C"}

    cmp = mc.compare({"main": sol, "main_copy": sol}, data=df)
    # Identical solutions → AUCs identical → DeLong p-value should be 1.0 (or NaN if degenerate).
    p = cmp.delong_pvalues.loc["main", "main_copy"]
    assert pd.isna(p) or float(p) >= 0.99
