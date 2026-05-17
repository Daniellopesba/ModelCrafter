"""Tests for ``mc.compare`` and the ``Comparison`` value (Task P5.C).

This file pins the acceptance criteria for AGENTS.md Task P5.C and the
relevant parts of DESIGN.md §8 Phase 5:

    Acceptance #2  (DESIGN.md §8 Phase 5): ``mc.delong_test`` matches the
    ``pROC::roc.test`` R reference to 1e-6 — verified at P3.D. P5.C wraps
    that primitive and surfaces the same p-value in ``Comparison.delong_pvalues``
    to floating-point equality.

    Acceptance #3  (DESIGN.md §8 Phase 5): ``mc.compare`` on two models
    trained with different lambdas on the same data shows AUC differences
    consistent with ``report_a.discrimination.auc.value -
    report_b.discrimination.auc.value`` on the same test set, including a
    DeLong p-value that matches ``mc.delong_test`` directly.

The structural fixture mirrors ``tests/test_performance.py``: small
Bernoulli data, two logistic-ridge solutions at different lambdas.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import model_crafter as mc
from model_crafter.performance import PerformanceReport, performance
from model_crafter.performance.compare import Comparison, compare, delong_test

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _bernoulli_data(rng: np.random.Generator, n: int = 800) -> pd.DataFrame:
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    p = 1.0 / (1.0 + np.exp(-(0.3 + 0.8 * x1 - 0.4 * x2)))
    y = rng.binomial(1, p).astype(float)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


@pytest.fixture
def two_solutions_and_data():
    """Two logistic-ridge solutions at different lambdas on the same data."""
    rng = np.random.default_rng(0)
    train = _bernoulli_data(rng, n=800)
    test = _bernoulli_data(rng, n=400)
    spec_a = mc.linear(
        target="y",
        features=["x1", "x2"],
        loss=mc.logistic,
        penalty=mc.l2(1e-4),
    )
    spec_b = mc.linear(
        target="y",
        features=["x1", "x2"],
        loss=mc.logistic,
        penalty=mc.l2(10.0),
    )
    sol_a = mc.solve(spec_a, train)
    sol_b = mc.solve(spec_b, train)
    return sol_a, sol_b, test


# ---------------------------------------------------------------------------
# Type structure
# ---------------------------------------------------------------------------


def test_compare_returns_Comparison(two_solutions_and_data):
    sol_a, sol_b, test = two_solutions_and_data
    cmp = compare({"a": sol_a, "b": sol_b}, test)
    assert isinstance(cmp, Comparison)
    assert set(cmp.reports.keys()) == {"a", "b"}
    assert isinstance(cmp.reports["a"], PerformanceReport)
    assert isinstance(cmp.reports["b"], PerformanceReport)


def test_compare_delong_pvalues_dataframe_shape(two_solutions_and_data):
    sol_a, sol_b, test = two_solutions_and_data
    cmp = compare({"a": sol_a, "b": sol_b}, test)
    df = cmp.delong_pvalues
    assert isinstance(df, pd.DataFrame)
    assert list(df.index) == ["a", "b"]
    assert list(df.columns) == ["a", "b"]
    # Diagonal is NaN (a model vs itself is not a meaningful pair).
    assert math.isnan(df.loc["a", "a"])
    assert math.isnan(df.loc["b", "b"])


def test_compare_delong_pvalues_symmetric(two_solutions_and_data):
    sol_a, sol_b, test = two_solutions_and_data
    cmp = compare({"a": sol_a, "b": sol_b}, test)
    df = cmp.delong_pvalues
    # Acceptance: the p-value is symmetric in the unordered pair.
    assert df.loc["a", "b"] == df.loc["b", "a"]


# ---------------------------------------------------------------------------
# Acceptance #2 / #3 — DeLong p-values match the primitive exactly
# ---------------------------------------------------------------------------


def test_compare_delong_pvalues_match_delong_test_primitive(two_solutions_and_data):
    """Acceptance: ``compare`` surfaces the same DeLong p-value as the
    standalone ``mc.delong_test`` primitive, to floating-point equality.
    """
    sol_a, sol_b, test = two_solutions_and_data
    cmp = compare({"a": sol_a, "b": sol_b}, test)
    direct = delong_test(sol_a, sol_b, test)
    # Floating-point equality (same primitive, no recomputation).
    assert cmp.delong_pvalues.loc["a", "b"] == direct.p_value
    assert cmp.delong_pvalues.loc["b", "a"] == direct.p_value


def test_compare_auc_difference_matches_reports(two_solutions_and_data):
    """Acceptance #3: AUC difference is consistent with the per-solution
    PerformanceReport AUC values evaluated on the same data.
    """
    sol_a, sol_b, test = two_solutions_and_data
    cmp = compare({"a": sol_a, "b": sol_b}, test)
    report_a = cmp.reports["a"]
    report_b = cmp.reports["b"]
    # The reports are built by the same primitives as mc.performance
    # called directly.
    perf_a = performance(sol_a, test)
    perf_b = performance(sol_b, test)
    assert report_a.discrimination.auc.value == perf_a.discrimination.auc.value
    assert report_b.discrimination.auc.value == perf_b.discrimination.auc.value
    # AUC differences computed from the per-report values exactly match the
    # delong primitive's diff (same scores, same target).
    direct = delong_test(sol_a, sol_b, test)
    diff_from_reports = (
        report_a.discrimination.auc.value - report_b.discrimination.auc.value
    )
    assert diff_from_reports == pytest.approx(direct.diff, abs=1e-12)


# ---------------------------------------------------------------------------
# Three-way comparison
# ---------------------------------------------------------------------------


def test_compare_three_models_square_pvalues(two_solutions_and_data):
    sol_a, sol_b, test = two_solutions_and_data
    spec_c = mc.linear(
        target="y",
        features=["x1", "x2"],
        loss=mc.logistic,
        penalty=mc.l2(0.5),
    )
    sol_c = mc.solve(spec_c, test)  # any fitted sol is fine for shape test
    sols = {"a": sol_a, "b": sol_b, "c": sol_c}
    cmp = compare(sols, test)
    df = cmp.delong_pvalues
    assert df.shape == (3, 3)
    assert list(df.index) == ["a", "b", "c"]
    # Diagonal all NaN.
    for name in ("a", "b", "c"):
        assert math.isnan(df.loc[name, name])
    # Off-diagonal symmetric.
    for i in ("a", "b", "c"):
        for j in ("a", "b", "c"):
            if i == j:
                continue
            assert df.loc[i, j] == df.loc[j, i]
    # And each off-diagonal entry equals the standalone primitive.
    for i, j in [("a", "b"), ("a", "c"), ("b", "c")]:
        expected = delong_test(sols[i], sols[j], test).p_value
        assert df.loc[i, j] == expected


# ---------------------------------------------------------------------------
# Non-uniform weights → DeLong falls back to NaN (P3.D limitation)
# ---------------------------------------------------------------------------


def test_compare_with_nonuniform_weights_falls_back_to_nan(two_solutions_and_data):
    """When weights are non-uniform, P3.D's ``delong_test`` raises
    ``NotImplementedError``. ``compare`` must catch that and fill
    ``delong_pvalues`` with NaN rather than raise, since the per-solution
    reports themselves remain computable.
    """
    sol_a, sol_b, test = two_solutions_and_data
    rng = np.random.default_rng(7)
    test = test.assign(w=rng.uniform(0.5, 2.0, len(test)))
    cmp = compare({"a": sol_a, "b": sol_b}, test, weights="w")
    df = cmp.delong_pvalues
    assert df.shape == (2, 2)
    # All entries (including off-diagonal) are NaN.
    for i in ("a", "b"):
        for j in ("a", "b"):
            assert math.isnan(df.loc[i, j])
    # Reports themselves were still built.
    assert isinstance(cmp.reports["a"], PerformanceReport)
    assert isinstance(cmp.reports["b"], PerformanceReport)


def test_compare_with_uniform_weights_still_computes_delong(two_solutions_and_data):
    """A uniform weights column (e.g. all 1.0) should *not* trigger the
    fallback — DeLong accepts uniform weights.
    """
    sol_a, sol_b, test = two_solutions_and_data
    test = test.assign(w=np.ones(len(test)))
    cmp = compare({"a": sol_a, "b": sol_b}, test, weights="w")
    df = cmp.delong_pvalues
    assert not math.isnan(df.loc["a", "b"])
    direct = delong_test(sol_a, sol_b, test, weights="w")
    assert df.loc["a", "b"] == direct.p_value


# ---------------------------------------------------------------------------
# repr layout (DESIGN.md §3.3 "Model comparison")
# ---------------------------------------------------------------------------


def test_compare_repr_contains_header_and_metrics(two_solutions_and_data):
    """The ``__repr__`` should match DESIGN.md §3.3's layout: a header
    line with ``n`` and the metric/solution table including AUC, KS,
    Brier, Log-loss, PSI, plus Δ and DeLong p-value columns.
    """
    sol_a, sol_b, test = two_solutions_and_data
    cmp = compare({"a": sol_a, "b": sol_b}, test)
    text = repr(cmp)
    assert "Comparison" in text
    # Header includes the observation count.
    assert "n=" in text
    # Section headers / metric labels per DESIGN.md §3.3.
    for label in ("AUC", "KS", "Brier", "Log-loss"):
        assert label in text
    # Solution names appear in the header row.
    assert "a" in text and "b" in text
    # Δ column and a p-value mention (DeLong).
    assert "Δ" in text or "Delta" in text
    assert "p-value" in text or "DeLong" in text


def test_compare_repr_mentions_weights_fallback(two_solutions_and_data):
    sol_a, sol_b, test = two_solutions_and_data
    rng = np.random.default_rng(11)
    test = test.assign(w=rng.uniform(0.5, 2.0, len(test)))
    cmp = compare({"a": sol_a, "b": sol_b}, test, weights="w")
    text = repr(cmp)
    # The repr documents the DeLong-unavailable case so users aren't
    # confused by NaNs.
    assert "DeLong" in text
    assert "weights" in text or "unavailable" in text.lower()


# ---------------------------------------------------------------------------
# Frozen + slots
# ---------------------------------------------------------------------------


def test_Comparison_is_frozen(two_solutions_and_data):
    sol_a, sol_b, test = two_solutions_and_data
    cmp = compare({"a": sol_a, "b": sol_b}, test)
    with pytest.raises((AttributeError, TypeError)):
        cmp.reports = {}  # type: ignore[misc]
