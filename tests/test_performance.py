"""Tests for ``mc.performance`` and the ``PerformanceReport`` value.

This file covers acceptance criterion 6 of Task P3.D (AGENTS.md):

  ``performance(sol, data)`` produces a ``PerformanceReport`` whose
  sub-reports contain values matching the individual primitive calls.

Plus the cross-cutting acceptance criterion 7 of Section §8 Phase 3
(DESIGN.md): the ``__repr__`` produces a single block matching the format
in §3.3.

The test approach (per AGENTS.md P3.D note): we build ``Solution`` objects
via the squared-error path because P3.A's ``logistic`` loss is being
developed in parallel. Predictions are roughly in ``[0, 1]`` for our
synthetic Bernoulli targets; calibration metrics clip internally per the
DESIGN.md contract.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import model_crafter as mc
from model_crafter.metrics import (
    auc,
    brier_score,
    calibration_curve,
    calibration_slope_intercept,
    cohens_d,
    cumulative_gains,
    ece,
    gini,
    ks,
    lift_table,
    log_loss,
    psi,
)
from model_crafter.performance import (
    CalibrationReport,
    DiscriminationReport,
    DistributionReport,
    PerformanceReport,
    StabilityReport,
    performance,
)


def _bernoulli_data(rng: np.random.Generator, n: int = 600):
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    p = 1.0 / (1.0 + np.exp(-(0.4 + 0.7 * x1 + 0.3 * x2)))
    y = rng.binomial(1, p).astype(float)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


@pytest.fixture
def fitted_sol_and_data():
    rng = np.random.default_rng(0)
    df = _bernoulli_data(rng)
    spec = mc.linear(
        target="y",
        features=["x1", "x2"],
        loss=mc.squared_error,
    )
    sol = mc.solve(spec, df)
    return sol, df


@pytest.fixture
def fitted_sol_with_reference():
    """Same dataset but with a separate "reference" frame for the PSI check."""
    rng = np.random.default_rng(1)
    train = _bernoulli_data(rng, n=600)
    test = _bernoulli_data(rng, n=400)
    spec = mc.linear(
        target="y",
        features=["x1", "x2"],
        loss=mc.squared_error,
    )
    sol = mc.solve(spec, train)
    return sol, train, test


# ---------------------------------------------------------------------------
# Type structure
# ---------------------------------------------------------------------------


def test_performance_returns_PerformanceReport(fitted_sol_and_data):
    sol, df = fitted_sol_and_data
    perf = performance(sol, df)
    assert isinstance(perf, PerformanceReport)
    assert isinstance(perf.discrimination, DiscriminationReport)
    assert isinstance(perf.calibration, CalibrationReport)
    assert perf.stability is None  # no reference provided
    assert isinstance(perf.distribution, DistributionReport)


def test_performance_with_reference_returns_StabilityReport(
    fitted_sol_with_reference,
):
    sol, train, test = fitted_sol_with_reference
    perf = performance(sol, test, reference=train)
    assert isinstance(perf.stability, StabilityReport)
    assert perf.stability.psi.value >= 0.0


# ---------------------------------------------------------------------------
# Acceptance #6: sub-reports match individual primitive calls.
# ---------------------------------------------------------------------------


def test_performance_subreports_match_primitives(fitted_sol_and_data):
    """Acceptance criterion P3.D #6: every sub-report value equals the
    value of the corresponding standalone primitive call.

    This is the contract that prevents the orchestration from silently
    drifting away from the user-facing primitives.
    """
    sol, df = fitted_sol_and_data
    perf = performance(sol, df, n_bins=10, n_deciles=10)

    # Discrimination
    assert math.isclose(perf.discrimination.auc.value, auc(sol, df).value, abs_tol=1e-12)
    assert math.isclose(
        perf.discrimination.gini.value, gini(sol, df).value, abs_tol=1e-12
    )
    assert math.isclose(
        perf.discrimination.ks.value, ks(sol, df).value, abs_tol=1e-12
    )
    assert math.isclose(
        perf.discrimination.cohens_d.value, cohens_d(sol, df).value, abs_tol=1e-12
    )

    # Calibration
    assert math.isclose(
        perf.calibration.brier.value, brier_score(sol, df).value, abs_tol=1e-12
    )
    assert math.isclose(
        perf.calibration.ece.value,
        ece(sol, df, n_bins=10).value,
        abs_tol=1e-12,
    )
    assert math.isclose(
        perf.calibration.log_loss.value, log_loss(sol, df).value, abs_tol=1e-12
    )
    cf = calibration_slope_intercept(sol, df)
    assert math.isclose(
        perf.calibration.slope_intercept.slope, cf.slope, abs_tol=1e-8
    )
    assert math.isclose(
        perf.calibration.slope_intercept.intercept, cf.intercept, abs_tol=1e-8
    )
    cc = calibration_curve(sol, df, n_bins=10)
    np.testing.assert_allclose(
        perf.calibration.curve.predicted, cc.predicted, atol=1e-12
    )
    np.testing.assert_allclose(
        perf.calibration.curve.observed, cc.observed, atol=1e-12
    )

    # Lift table + gains
    lt = lift_table(sol, df, n_deciles=10)
    pd.testing.assert_frame_equal(perf.lift_table.table, lt.table)
    cg = cumulative_gains(sol, df)
    np.testing.assert_allclose(
        perf.cumulative_gains.cum_population, cg.cum_population, atol=1e-12
    )
    np.testing.assert_allclose(
        perf.cumulative_gains.cum_captured, cg.cum_captured, atol=1e-12
    )


def test_performance_stability_matches_psi_primitive(fitted_sol_with_reference):
    sol, train, test = fitted_sol_with_reference
    perf = performance(sol, test, reference=train, psi_bins=10)
    # Compute PSI of train predictions (reference) vs test predictions
    # (current) via the public ``psi`` primitive.
    train_scores = mc.predict(sol, train).to_numpy()
    test_scores = mc.predict(sol, test).to_numpy()
    psi_ref = psi(train_scores, test_scores, bins=10)
    assert perf.stability is not None
    assert math.isclose(
        perf.stability.psi.value, psi_ref.value, abs_tol=1e-12
    )


# ---------------------------------------------------------------------------
# Header counts
# ---------------------------------------------------------------------------


def test_performance_header_counts(fitted_sol_and_data):
    sol, df = fitted_sol_and_data
    perf = performance(sol, df)
    assert perf.n_obs == len(df)
    assert perf.n_events == int(df["y"].sum())


# ---------------------------------------------------------------------------
# AUC DeLong CI
# ---------------------------------------------------------------------------


def test_performance_auc_ci_brackets_point_estimate(fitted_sol_and_data):
    sol, df = fitted_sol_and_data
    perf = performance(sol, df, auc_ci_level=0.95)
    ci = perf.discrimination.auc_ci
    assert ci is not None
    lo, hi = ci
    assert lo <= perf.discrimination.auc.value <= hi
    assert perf.discrimination.auc_se is not None
    assert perf.discrimination.auc_se > 0.0


# ---------------------------------------------------------------------------
# Repr layout matches DESIGN.md §3.3
# ---------------------------------------------------------------------------


def test_performance_repr_contains_required_sections(fitted_sol_with_reference):
    """Acceptance #7 (Phase 3 §8): the ``__repr__`` produces a single block
    matching the format in DESIGN.md §3.3.

    We assert structural presence of every line label in §3.3 — exact
    numerical formatting is verified by the individual metric reprs.
    """
    sol, train, test = fitted_sol_with_reference
    perf = performance(sol, test, reference=train)
    s = repr(perf)
    # Header
    assert s.startswith("PerformanceReport")
    assert "events=" in s
    # Discrimination
    assert "Discrimination" in s
    assert "AUC" in s and "DeLong" in s  # CI is present
    assert "Gini" in s
    assert "KS" in s and "at score" in s
    assert "Cohen's d" in s
    # Calibration
    assert "Calibration" in s
    assert "Brier" in s
    assert "ECE" in s and "bins)" in s
    assert "Log-loss" in s
    assert "Slope / Intercept" in s
    # Stability (only when reference is provided)
    assert "Stability" in s
    assert "PSI vs reference" in s
    # Distribution
    assert "Distribution" in s
    assert "Mean / Median" in s
    assert "Score range" in s


def test_performance_repr_no_stability_when_no_reference(fitted_sol_and_data):
    sol, df = fitted_sol_and_data
    perf = performance(sol, df)
    s = repr(perf)
    assert "Stability" not in s


# ---------------------------------------------------------------------------
# Weights pass through
# ---------------------------------------------------------------------------


def test_performance_with_weights_runs(fitted_sol_and_data):
    sol, df = fitted_sol_and_data
    df = df.assign(w=np.ones(len(df)))
    perf_w = performance(sol, df, weights="w")
    perf_u = performance(sol, df.drop(columns=["w"]))
    # Uniform weights => identical metric values to unweighted.
    assert math.isclose(
        perf_w.discrimination.auc.value, perf_u.discrimination.auc.value
    )
    assert math.isclose(
        perf_w.calibration.brier.value, perf_u.calibration.brier.value
    )
    assert math.isclose(
        perf_w.calibration.log_loss.value, perf_u.calibration.log_loss.value
    )


def test_performance_with_array_weights(fitted_sol_and_data):
    sol, df = fitted_sol_and_data
    rng = np.random.default_rng(2)
    w = rng.uniform(0.5, 2.0, len(df))
    perf = performance(sol, df, weights=w)
    assert perf.n_obs == len(df)


# ---------------------------------------------------------------------------
# Reference forms
# ---------------------------------------------------------------------------


def test_performance_reference_dataframe(fitted_sol_with_reference):
    sol, train, test = fitted_sol_with_reference
    perf = performance(sol, test, reference=train)
    assert perf.stability is not None


def test_performance_reference_array_of_scores(fitted_sol_with_reference):
    sol, train, test = fitted_sol_with_reference
    ref_scores = mc.predict(sol, train).to_numpy()
    perf = performance(sol, test, reference=ref_scores)
    assert perf.stability is not None


def test_performance_reference_invalid_raises(fitted_sol_and_data):
    sol, df = fitted_sol_and_data
    with pytest.raises(TypeError):
        performance(sol, df, reference=42)


# ---------------------------------------------------------------------------
# Distribution sub-report
# ---------------------------------------------------------------------------


def test_performance_distribution_summary(fitted_sol_and_data):
    sol, df = fitted_sol_and_data
    perf = performance(sol, df)
    scores = mc.predict(sol, df).to_numpy()
    d = perf.distribution
    assert math.isclose(d.mean, float(np.mean(scores)), abs_tol=1e-12)
    assert math.isclose(d.median, float(np.median(scores)), abs_tol=1e-12)
    assert math.isclose(d.p_min, float(np.min(scores)), abs_tol=1e-12)
    assert math.isclose(d.p_max, float(np.max(scores)), abs_tol=1e-12)
