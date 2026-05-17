"""Tests for ``_repr_html_`` on every public value type (Task P6).

Acceptance criterion (DESIGN.md §8 Phase 6 / Task P6):

* Every listed value type defines ``_repr_html_``.
* For each, ``value._repr_html_()`` returns a non-empty string containing
  ``"<table"`` or ``"<div"``.
* A snapshot test pins the HTML for a small fixture of each (just check
  that key labels — e.g. "AUC", "Discrimination", "Segment", "p-value" —
  appear).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import model_crafter as mc
from model_crafter.assumptions import AssumptionReport
from model_crafter.inspect import BinningTable, binning_table
from model_crafter.performance import (
    PerformanceReport,
    compare,
    performance,
    performance_by_segment,
    performance_over_time,
)
from model_crafter.performance.by_segment import SegmentedPerformanceReport
from model_crafter.performance.compare import Comparison
from model_crafter.performance.over_time import TemporalPerformanceReport
from model_crafter.solution import BootstrappedSolution, SegmentedSolution
from model_crafter.spec import segmented
from model_crafter.validation.cross_validate import CVResult
from model_crafter.validation.tune import NestedCVResult, TuneResult

# ---------------------------------------------------------------------------
# Fixtures: a simple OLS + a logistic + a WoE fit so we can probe everything
# ---------------------------------------------------------------------------


@pytest.fixture
def ols_solution():
    rng = np.random.default_rng(0)
    n = 100
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = 1.0 + 0.7 * x1 - 0.4 * x2 + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    spec = mc.linear(target="y", features=["x1", "x2"], loss=mc.squared_error)
    return mc.solve(spec, df), df


@pytest.fixture
def logistic_solution_with_segment():
    rng = np.random.default_rng(1)
    n = 300
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    seg = rng.choice(["A", "B", "C"], size=n)
    eta = -0.2 + 0.5 * x1 - 0.3 * x2
    p = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, p).astype(float)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "segment": seg})
    spec = mc.linear(target="y", features=["x1", "x2"], loss=mc.logistic)
    return mc.solve(spec, df), df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_html_string(s: str) -> None:
    assert isinstance(s, str)
    assert len(s) > 0
    assert ("<div" in s) or ("<table" in s)


# ---------------------------------------------------------------------------
# Solution
# ---------------------------------------------------------------------------


def test_solution_repr_html(ols_solution):
    sol, _ = ols_solution
    h = sol._repr_html_()
    _assert_html_string(h)
    assert "Solution" in h
    # Coefficient names appear in the table.
    for col in sol.design_columns:
        assert col in h


# ---------------------------------------------------------------------------
# BootstrappedSolution
# ---------------------------------------------------------------------------


def test_bootstrapped_solution_repr_html(ols_solution):
    sol, df = ols_solution
    bs = mc.bootstrap(sol, df, n_boot=10, random_state=0)
    assert isinstance(bs, BootstrappedSolution)
    h = bs._repr_html_()
    _assert_html_string(h)
    assert "BootstrappedSolution" in h
    assert "CI" in h


# ---------------------------------------------------------------------------
# SegmentedSolution
# ---------------------------------------------------------------------------


def test_segmented_solution_repr_html(logistic_solution_with_segment):
    _, df = logistic_solution_with_segment
    base = mc.linear(target="y", features=["x1", "x2"], loss=mc.logistic)
    spec = segmented(by="segment", base=base)
    sol = mc.solve(spec, df)
    assert isinstance(sol, SegmentedSolution)
    h = sol._repr_html_()
    _assert_html_string(h)
    assert "SegmentedSolution" in h
    assert "Segment" in h
    for k in sol:
        assert k in h


# ---------------------------------------------------------------------------
# AssumptionReport
# ---------------------------------------------------------------------------


def test_assumption_report_repr_html(ols_solution):
    sol, _ = ols_solution
    rep = sol.assumptions
    assert isinstance(rep, AssumptionReport)
    h = rep._repr_html_()
    _assert_html_string(h)
    assert "AssumptionReport" in h


def test_assumption_report_repr_html_empty():
    rep = AssumptionReport(results=())
    h = rep._repr_html_()
    _assert_html_string(h)
    assert "empty" in h


# ---------------------------------------------------------------------------
# BinningTable
# ---------------------------------------------------------------------------


def test_binning_table_repr_html():
    rng = np.random.default_rng(2)
    n = 300
    income = rng.normal(50, 15, n)
    y = (rng.uniform(0, 1, n) < 1.0 / (1.0 + np.exp(-(income - 50) / 10))).astype(float)
    df = pd.DataFrame({"y": y, "income": income})
    spec = mc.linear(
        target="y",
        features=mc.woe("income", bins=mc.monotonic(min_bin_size=0.1, max_bins=4)),
        loss=mc.logistic,
    )
    from model_crafter.terms.woe import fit_binnings

    spec_fit = fit_binnings(spec, df)
    sol = mc.solve(spec_fit, df, on_violation="warn")
    table = binning_table(sol)
    assert isinstance(table, BinningTable)
    h = table._repr_html_()
    _assert_html_string(h)
    assert "BinningTable" in h
    assert "income" in h


def test_binning_table_repr_html_empty():
    table = BinningTable(tables={}, iv=pd.Series(dtype=float))
    h = table._repr_html_()
    _assert_html_string(h)
    assert "BinningTable" in h


# ---------------------------------------------------------------------------
# CVResult
# ---------------------------------------------------------------------------


def test_cv_result_repr_html():
    # Build a CVResult by hand to avoid running a CV loop here.
    fold_results = (
        {
            "train_period": (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")),
            "valid_period": (pd.Timestamp("2020-02-02"), pd.Timestamp("2020-03-01")),
            "metrics": {"auc": 0.78, "ks": 0.42},
            "solution": None,
        },
        {
            "train_period": (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-04-01")),
            "valid_period": (pd.Timestamp("2020-04-02"), pd.Timestamp("2020-05-01")),
            "metrics": {"auc": 0.81, "ks": 0.45},
            "solution": None,
        },
    )
    cv = CVResult(fold_results=fold_results, solutions=())
    h = cv._repr_html_()
    _assert_html_string(h)
    assert "CVResult" in h


# ---------------------------------------------------------------------------
# TuneResult / NestedCVResult
# ---------------------------------------------------------------------------


def test_tune_result_repr_html():
    # Build a minimal TuneResult by hand to avoid solver time.
    curve = pd.DataFrame(
        {"metric_mean": [0.5, 0.6], "metric_sd": [0.01, 0.01]},
        index=pd.Index([0.1, 0.2]),
    )
    tr = TuneResult(
        best_param=0.2,
        cv_curve=curve,
        solution=None,
        direction="maximize",
    )
    h = tr._repr_html_()
    _assert_html_string(h)
    assert "TuneResult" in h
    assert "CV curve" in h


def test_nested_cv_result_repr_html():
    outer = pd.DataFrame({"metric": [0.7, 0.72], "best_param": [0.1, 0.2]})
    ncv = NestedCVResult(
        outer_metric=outer,
        best_params=(0.1, 0.2),
        inner_curves=(),
    )
    h = ncv._repr_html_()
    _assert_html_string(h)
    assert "NestedCVResult" in h
    assert "Outer" in h


# ---------------------------------------------------------------------------
# PerformanceReport + sub-reports
# ---------------------------------------------------------------------------


def test_performance_report_repr_html(logistic_solution_with_segment):
    sol, df = logistic_solution_with_segment
    perf = performance(sol, df)
    assert isinstance(perf, PerformanceReport)
    h = perf._repr_html_()
    _assert_html_string(h)
    assert "PerformanceReport" in h
    assert "Discrimination" in h
    assert "Calibration" in h
    assert "Distribution" in h
    assert "AUC" in h


def test_performance_report_with_stability_repr_html(logistic_solution_with_segment):
    sol, df = logistic_solution_with_segment
    perf = performance(sol, df, reference=df)
    assert perf.stability is not None
    h = perf._repr_html_()
    _assert_html_string(h)
    assert "Stability" in h
    assert "PSI" in h


def test_discrimination_report_repr_html(logistic_solution_with_segment):
    sol, df = logistic_solution_with_segment
    perf = performance(sol, df)
    h = perf.discrimination._repr_html_()
    _assert_html_string(h)
    assert "Discrimination" in h
    assert "AUC" in h


def test_calibration_report_repr_html(logistic_solution_with_segment):
    sol, df = logistic_solution_with_segment
    perf = performance(sol, df)
    h = perf.calibration._repr_html_()
    _assert_html_string(h)
    assert "Calibration" in h
    assert "Brier" in h


def test_stability_report_repr_html(logistic_solution_with_segment):
    sol, df = logistic_solution_with_segment
    perf = performance(sol, df, reference=df)
    assert perf.stability is not None
    h = perf.stability._repr_html_()
    _assert_html_string(h)
    assert "Stability" in h
    assert "PSI" in h


def test_distribution_report_repr_html(logistic_solution_with_segment):
    sol, df = logistic_solution_with_segment
    perf = performance(sol, df)
    h = perf.distribution._repr_html_()
    _assert_html_string(h)
    assert "Distribution" in h


# ---------------------------------------------------------------------------
# TemporalPerformanceReport
# ---------------------------------------------------------------------------


def test_temporal_performance_report_repr_html(logistic_solution_with_segment):
    sol, df = logistic_solution_with_segment
    df = df.copy()
    df["t"] = pd.date_range("2020-01-01", periods=len(df), freq="D")
    splitter = mc.rolling_window(
        time_col="t",
        train_size="30D",
        horizon="20D",
        step="20D",
        gap="1D",
    )
    perf_t = performance_over_time(sol, df, splitter)
    assert isinstance(perf_t, TemporalPerformanceReport)
    h = perf_t._repr_html_()
    _assert_html_string(h)
    assert "TemporalPerformanceReport" in h


def test_temporal_performance_report_empty_repr_html():
    # Empty bundle round-trip.
    rep = TemporalPerformanceReport(summary=pd.DataFrame(), reports=())
    h = rep._repr_html_()
    _assert_html_string(h)
    assert "empty" in h


# ---------------------------------------------------------------------------
# SegmentedPerformanceReport
# ---------------------------------------------------------------------------


def test_segmented_performance_report_repr_html(logistic_solution_with_segment):
    sol, df = logistic_solution_with_segment
    seg = performance_by_segment(sol, df, by="segment")
    assert isinstance(seg, SegmentedPerformanceReport)
    h = seg._repr_html_()
    _assert_html_string(h)
    assert "SegmentedPerformanceReport" in h
    assert "Segment" in h
    assert "AUC" in h


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def test_comparison_repr_html(logistic_solution_with_segment):
    sol, df = logistic_solution_with_segment
    cmp = compare({"baseline": sol, "challenger": sol}, data=df)
    assert isinstance(cmp, Comparison)
    h = cmp._repr_html_()
    _assert_html_string(h)
    assert "Comparison" in h
    assert "AUC" in h
    assert "baseline" in h
    assert "challenger" in h


# ---------------------------------------------------------------------------
# Snapshot-style key-label checks
# ---------------------------------------------------------------------------


def test_html_has_critical_labels_for_jupyter(logistic_solution_with_segment):
    """The HTML must surface the labels a reviewer cares about."""
    sol, df = logistic_solution_with_segment
    perf = performance(sol, df, reference=df)
    html_full = perf._repr_html_()
    for needle in ("AUC", "Brier", "PSI", "Discrimination", "Calibration"):
        assert needle in html_full, f"missing {needle!r} in PerformanceReport HTML"


def test_segmented_html_has_segment_and_auc():
    """Critical labels on the segmented performance HTML."""
    rng = np.random.default_rng(33)
    n = 150
    x = rng.normal(0, 1, n)
    seg = rng.choice(["A", "B"], n)
    eta = -0.3 + 0.5 * x
    p = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, p).astype(float)
    df = pd.DataFrame({"y": y, "x": x, "segment": seg})
    sol = mc.solve(mc.linear(target="y", features=["x"], loss=mc.logistic), df)
    seg_rep = performance_by_segment(sol, df, by="segment")
    h = seg_rep._repr_html_()
    assert "Segment" in h
    assert "AUC" in h


# ---------------------------------------------------------------------------
# Solution._repr_html_ does not crash for SE = None
# ---------------------------------------------------------------------------


def test_solution_repr_html_handles_missing_se(ols_solution):
    sol, df = ols_solution
    # Build a lasso (no SE) version
    spec = mc.linear(
        target="y",
        features=["x1", "x2"],
        loss=mc.squared_error,
        penalty=mc.l1(0.1),
    )
    sol_l = mc.solve(spec, df)
    assert sol_l.coefficient_se is None
    h = sol_l._repr_html_()
    _assert_html_string(h)
    assert "Solution" in h
