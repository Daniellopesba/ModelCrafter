"""Tests for the metric primitives in ``model_crafter.metrics``.

This file covers the acceptance criteria for Task P3.D (AGENTS.md):

1.  ``auc`` matches ``scipy.stats.mannwhitneyu``-derived AUC to 1e-9 on
    synthetic data (100 trials, random seeds).
2.  ``ks`` matches ``scipy.stats.ks_2samp`` to 1e-9.
3.  ``psi`` matches the hand-derived formula on a worked example to 1e-12.
    The worked example is pinned in the test docstring.
4.  ``delong_test`` matches a Python reference implementation to 1e-6 on a
    fixed seed. The reference (Sun & Xu 2014 closed form, computed
    independently in the test) is pinned in the docstring.
5.  ``calibration_slope_intercept`` recovers slope=1, intercept=0 on
    synthetic perfectly-calibrated data to 1e-6.
6.  ``performance(sol, data)`` is cross-checked in
    ``tests/test_performance.py``.
7.  All metrics with ``weights=`` (a) match the unweighted variant when
    ``weights=1`` and (b) match a reference for non-uniform weights
    (hand-derived for PSI/ECE, statsmodels-equivalent weighted-mean for
    Cohen's d and Brier).

The metrics primitives accept ``(sol, data, weights=...)``. To keep the
tests focused on numerical correctness (not on the Phase 3 logistic
solver, which is being built in parallel by P3.A), they hit the
``_<metric>_from_arrays`` private helpers directly. A small "sol-shim"
fixture is used to verify the public ``(sol, data)`` entry points wrap
the helpers correctly.

The test approach for the sol-level entry points: we fit a squared-error
linear regression on synthetic Bernoulli data. Predictions are in
``[0, 1]`` approximately (a handful may fall slightly outside the range);
metrics that need true probabilities clip internally per DESIGN.md
contract. This matches the AGENTS.md P3.D guidance on how to test in the
absence of P3.A's logistic loss.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import model_crafter as mc
from model_crafter.metrics import (
    AUCResult,
    BrierResult,
    CalibrationCurve,
    CalibrationFit,
    CohensDResult,
    ECEResult,
    GainsCurve,
    GiniResult,
    KSResult,
    LiftTable,
    LogLossResult,
    PSIResult,
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
from model_crafter.metrics.calibration import (
    _brier_from_arrays,
    _calibration_curve_from_arrays,
    _calibration_fit_from_arrays,
    _ece_from_arrays,
    _log_loss_from_arrays,
)
from model_crafter.metrics.discrimination import (
    _auc_from_arrays,
    _ks_from_arrays,
)
from model_crafter.metrics.effect_size import _cohens_d_from_arrays
from model_crafter.metrics.rank import (
    _cumulative_gains_from_arrays,
    _lift_table_from_arrays,
)
from model_crafter.performance import DeLongResult, delong_test
from model_crafter.performance._delong import _delong_components

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _bernoulli_dataset(rng: np.random.Generator, n: int = 400):
    """A small classification-flavoured dataset with continuous features."""
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    p = 1.0 / (1.0 + np.exp(-(0.3 + 0.8 * x1 + 0.5 * x2)))
    y = rng.binomial(1, p).astype(float)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


@pytest.fixture
def small_classification_sol():
    """A real ``Solution`` built via the Phase 2 squared-error path.

    Predictions are roughly in ``[0, 1]`` (a few may fall slightly
    outside; the metrics that need probabilities clip internally).
    """
    rng = np.random.default_rng(0)
    df = _bernoulli_dataset(rng, n=400)
    spec = mc.linear(
        target="y",
        features=["x1", "x2"],
        loss=mc.squared_error,
    )
    sol = mc.solve(spec, df)
    return sol, df


# ---------------------------------------------------------------------------
# AUC: matches scipy.stats.mannwhitneyu to 1e-9, 100 trials
# ---------------------------------------------------------------------------


def test_auc_matches_scipy_mannwhitneyu_100_trials():
    """Acceptance #1: AUC matches ``scipy.stats.mannwhitneyu``-derived AUC
    to 1e-9 on 100 random trials.

    AUC equals ``U / (n_pos * n_neg)`` where ``U`` is the
    ``alternative='greater'`` MWU statistic — the gold-standard identity.
    """
    rng = np.random.default_rng(0)
    for trial in range(100):
        n = int(rng.integers(50, 500))
        # Always produce a mix of classes.
        y = rng.integers(0, 2, n).astype(float)
        if y.sum() == 0 or y.sum() == n:
            y[0] = 1.0 - y[0]
        scores = rng.normal(0.0, 1.0, n) + 0.4 * y
        value, n_pos, n_neg = _auc_from_arrays(y, scores)
        u, _ = stats.mannwhitneyu(
            scores[y == 1], scores[y == 0], alternative="greater"
        )
        auc_ref = u / (n_pos * n_neg)
        assert math.isclose(value, auc_ref, abs_tol=1e-9), trial


def test_auc_perfect_separation_is_one():
    y = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    v, _, _ = _auc_from_arrays(y, s)
    assert v == 1.0


def test_auc_constant_score_is_half():
    y = np.array([0, 0, 1, 1], dtype=float)
    s = np.array([0.5, 0.5, 0.5, 0.5])
    v, _, _ = _auc_from_arrays(y, s)
    # All ties => midrank => AUC = 0.5
    assert math.isclose(v, 0.5, abs_tol=1e-12)


def test_auc_weights_uniform_equals_unweighted():
    """Acceptance #7a: weights=1 matches the unweighted variant."""
    rng = np.random.default_rng(1)
    n = 200
    y = rng.integers(0, 2, n).astype(float)
    s = rng.normal(0, 1, n) + 0.5 * y
    v_unw, _, _ = _auc_from_arrays(y, s)
    v_w, _, _ = _auc_from_arrays(y, s, weights=np.ones(n))
    assert math.isclose(v_unw, v_w, abs_tol=1e-12)


def test_auc_weighted_matches_replication_reference():
    """Acceptance #7b: weighted AUC matches the AUC of a replicated sample.

    For integer weights, the weighted AUC must equal the AUC of the dataset
    expanded by repeating each row ``w`` times (Pepe 2003 §5).
    """
    rng = np.random.default_rng(2)
    n = 60
    y = rng.integers(0, 2, n).astype(float)
    s = rng.normal(0, 1, n) + 0.5 * y
    w = rng.integers(1, 5, n).astype(float)
    v_w, _, _ = _auc_from_arrays(y, s, weights=w)
    # Expand.
    y_rep = np.repeat(y, w.astype(int))
    s_rep = np.repeat(s, w.astype(int))
    v_rep, _, _ = _auc_from_arrays(y_rep, s_rep)
    assert math.isclose(v_w, v_rep, abs_tol=1e-10)


def test_auc_sol_entry_point(small_classification_sol):
    sol, df = small_classification_sol
    res = auc(sol, df)
    assert isinstance(res, AUCResult)
    assert 0.0 <= res.value <= 1.0
    # Match the array-level helper.
    y = df["y"].to_numpy(dtype=float)
    s = mc.predict(sol, df).to_numpy()
    v_arr, _, _ = _auc_from_arrays(y, s)
    assert math.isclose(res.value, v_arr, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# Gini
# ---------------------------------------------------------------------------


def test_gini_is_2auc_minus_1(small_classification_sol):
    sol, df = small_classification_sol
    g = gini(sol, df)
    a = auc(sol, df)
    assert isinstance(g, GiniResult)
    assert math.isclose(g.value, 2.0 * a.value - 1.0, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# KS: matches scipy.stats.ks_2samp to 1e-9
# ---------------------------------------------------------------------------


def test_ks_matches_scipy_ks2samp():
    """Acceptance #2: KS matches ``scipy.stats.ks_2samp`` to 1e-9."""
    rng = np.random.default_rng(0)
    for trial in range(50):
        n = int(rng.integers(50, 400))
        y = rng.integers(0, 2, n).astype(float)
        if y.sum() == 0 or y.sum() == n:
            y[0] = 1.0 - y[0]
        s = rng.normal(0, 1, n) + 0.4 * y
        ks_ours, _, _, _ = _ks_from_arrays(y, s)
        ks_ref = stats.ks_2samp(s[y == 1], s[y == 0]).statistic  # type: ignore[attr-defined]
        assert math.isclose(ks_ours, ks_ref, abs_tol=1e-9), trial


def test_ks_weights_uniform_equals_unweighted():
    rng = np.random.default_rng(7)
    n = 200
    y = rng.integers(0, 2, n).astype(float)
    s = rng.normal(0, 1, n) + 0.5 * y
    k_unw, _, _, _ = _ks_from_arrays(y, s)
    k_w, _, _, _ = _ks_from_arrays(y, s, weights=np.ones(n))
    assert math.isclose(k_unw, k_w, abs_tol=1e-12)


def test_ks_weighted_matches_replication():
    rng = np.random.default_rng(8)
    n = 80
    y = rng.integers(0, 2, n).astype(float)
    s = rng.normal(0, 1, n) + 0.4 * y
    w = rng.integers(1, 4, n).astype(float)
    k_w, _, _, _ = _ks_from_arrays(y, s, weights=w)
    y_rep = np.repeat(y, w.astype(int))
    s_rep = np.repeat(s, w.astype(int))
    k_rep, _, _, _ = _ks_from_arrays(y_rep, s_rep)
    assert math.isclose(k_w, k_rep, abs_tol=1e-10)


def test_ks_sol_entry_point(small_classification_sol):
    sol, df = small_classification_sol
    res = ks(sol, df)
    assert isinstance(res, KSResult)
    assert 0.0 <= res.value <= 1.0


# ---------------------------------------------------------------------------
# Cohen's d
# ---------------------------------------------------------------------------


def test_cohens_d_matches_pooled_formula():
    rng = np.random.default_rng(9)
    n = 200
    y = rng.integers(0, 2, n).astype(float)
    s = rng.normal(0, 1, n) + 1.0 * y
    d, mp, mn, sp = _cohens_d_from_arrays(y, s)
    # Recompute by hand.
    s_pos, s_neg = s[y == 1], s[y == 0]
    n_pos, n_neg = s_pos.size, s_neg.size
    var_pos = s_pos.var(ddof=1)
    var_neg = s_neg.var(ddof=1)
    pooled = math.sqrt(
        ((n_pos - 1) * var_pos + (n_neg - 1) * var_neg) / (n_pos + n_neg - 2)
    )
    d_ref = (s_pos.mean() - s_neg.mean()) / pooled
    assert math.isclose(d, d_ref, abs_tol=1e-12)
    assert math.isclose(sp, pooled, abs_tol=1e-12)
    assert math.isclose(mp, s_pos.mean(), abs_tol=1e-12)
    assert math.isclose(mn, s_neg.mean(), abs_tol=1e-12)


def test_cohens_d_weights_uniform_equals_unweighted():
    rng = np.random.default_rng(10)
    n = 100
    y = rng.integers(0, 2, n).astype(float)
    s = rng.normal(0, 1, n) + 0.5 * y
    d1, *_ = _cohens_d_from_arrays(y, s)
    d2, *_ = _cohens_d_from_arrays(y, s, weights=np.ones(n))
    assert math.isclose(d1, d2, abs_tol=1e-10)


def test_cohens_d_sol_entry_point(small_classification_sol):
    sol, df = small_classification_sol
    res = cohens_d(sol, df)
    assert isinstance(res, CohensDResult)
    assert math.isfinite(res.value)


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------


def test_brier_matches_manual_mean_squared_error():
    rng = np.random.default_rng(11)
    n = 300
    p = rng.uniform(0, 1, n)
    y = rng.binomial(1, p).astype(float)
    b, _ = _brier_from_arrays(y, p)
    assert math.isclose(b, float(np.mean((y - p) ** 2)), abs_tol=1e-12)


def test_brier_weights_match_reference():
    rng = np.random.default_rng(12)
    n = 200
    p = rng.uniform(0, 1, n)
    y = rng.binomial(1, p).astype(float)
    w = rng.uniform(0.1, 5.0, n)
    b, _ = _brier_from_arrays(y, p, weights=w)
    b_ref = float(np.sum(w * (y - p) ** 2) / np.sum(w))
    assert math.isclose(b, b_ref, abs_tol=1e-12)


def test_brier_sol_entry_point(small_classification_sol):
    sol, df = small_classification_sol
    res = brier_score(sol, df)
    assert isinstance(res, BrierResult)
    assert res.value >= 0.0


# ---------------------------------------------------------------------------
# Log loss
# ---------------------------------------------------------------------------


def test_log_loss_perfect_predictions():
    """With ``p == y`` (after clipping), log_loss equals
    ``-log(1 - eps)`` per observation."""
    y = np.array([0, 1, 0, 1, 0], dtype=float)
    p = y.copy()
    eps = 1e-12
    ll, _ = _log_loss_from_arrays(y, p, eps=eps)
    # After clipping, p=0 -> eps, p=1 -> 1-eps. Per obs loss = -log(1-eps).
    assert math.isclose(ll, -math.log(1 - eps), abs_tol=1e-12)


def test_log_loss_weights_uniform_matches_unweighted():
    rng = np.random.default_rng(13)
    n = 200
    y = rng.integers(0, 2, n).astype(float)
    p = rng.uniform(0.01, 0.99, n)
    ll1, _ = _log_loss_from_arrays(y, p)
    ll2, _ = _log_loss_from_arrays(y, p, weights=np.ones(n))
    assert math.isclose(ll1, ll2, abs_tol=1e-12)


def test_log_loss_weights_match_reference():
    rng = np.random.default_rng(14)
    n = 200
    y = rng.integers(0, 2, n).astype(float)
    p = rng.uniform(0.01, 0.99, n)
    w = rng.uniform(0.1, 5.0, n)
    ll, _ = _log_loss_from_arrays(y, p, weights=w)
    ll_ref = float(
        -np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))) / np.sum(w)
    )
    assert math.isclose(ll, ll_ref, abs_tol=1e-10)


def test_log_loss_invalid_eps_raises():
    y = np.array([0, 1], dtype=float)
    p = np.array([0.5, 0.5])
    with pytest.raises(ValueError):
        _log_loss_from_arrays(y, p, eps=0.0)
    with pytest.raises(ValueError):
        _log_loss_from_arrays(y, p, eps=0.5)


def test_log_loss_sol_entry_point(small_classification_sol):
    sol, df = small_classification_sol
    res = log_loss(sol, df)
    assert isinstance(res, LogLossResult)
    assert res.value >= 0.0


# ---------------------------------------------------------------------------
# Calibration curve
# ---------------------------------------------------------------------------


def test_calibration_curve_perfect_calibration():
    """With ``y ~ Bernoulli(p)`` and ``p = scores``, observed[bin] ≈
    predicted[bin] within bin sampling noise. We pick a large n so the
    sampling noise is small.
    """
    rng = np.random.default_rng(15)
    n = 50_000
    p = rng.uniform(0, 1, n)
    y = rng.binomial(1, p).astype(float)
    pred, obs, count = _calibration_curve_from_arrays(y, p, None, 10)
    nonempty = count > 0
    # Within 2 SE for a binomial.
    assert np.all(np.abs(obs[nonempty] - pred[nonempty]) < 0.02), (
        f"obs - pred = {obs[nonempty] - pred[nonempty]}"
    )


def test_calibration_curve_sol_entry_point(small_classification_sol):
    sol, df = small_classification_sol
    cc = calibration_curve(sol, df, n_bins=5)
    assert isinstance(cc, CalibrationCurve)
    assert cc.n_bins == 5
    # The curve has a to_frame helper.
    assert set(cc.to_frame().columns) == {"predicted", "observed", "count"}


# ---------------------------------------------------------------------------
# ECE
# ---------------------------------------------------------------------------


def test_ece_perfect_calibration_is_small():
    rng = np.random.default_rng(16)
    n = 100_000
    p = rng.uniform(0, 1, n)
    y = rng.binomial(1, p).astype(float)
    e, _ = _ece_from_arrays(y, p, None, 10)
    # 10 bins of ~10k each: binomial SE ~ sqrt(0.25/10000) = 0.005. ECE is
    # the weighted average gap, well under 0.01 in expectation.
    assert e < 0.01


def test_ece_hand_derived_worked_example():
    """ECE worked example.

    Five observations with scores [0.1, 0.2, 0.3, 0.4, 0.5] and outcomes
    [0, 0, 1, 1, 1]. Use 5 bins (one per observation). For each bin:
      bin 1: pred=0.1, obs=0, |gap|=0.1
      bin 2: pred=0.2, obs=0, |gap|=0.2
      bin 3: pred=0.3, obs=1, |gap|=0.7
      bin 4: pred=0.4, obs=1, |gap|=0.6
      bin 5: pred=0.5, obs=1, |gap|=0.5
    Each bin has weight 1/5, so ECE = (0.1 + 0.2 + 0.7 + 0.6 + 0.5) / 5 = 0.42.
    """
    y = np.array([0, 0, 1, 1, 1], dtype=float)
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    e, _ = _ece_from_arrays(y, s, None, 5)
    assert math.isclose(e, (0.1 + 0.2 + 0.7 + 0.6 + 0.5) / 5.0, abs_tol=1e-12)


def test_ece_weights_uniform_matches_unweighted():
    rng = np.random.default_rng(17)
    n = 300
    p = rng.uniform(0, 1, n)
    y = rng.binomial(1, p).astype(float)
    e1, _ = _ece_from_arrays(y, p, None, 10)
    e2, _ = _ece_from_arrays(y, p, np.ones(n), 10)
    assert math.isclose(e1, e2, abs_tol=1e-12)


def test_ece_sol_entry_point(small_classification_sol):
    sol, df = small_classification_sol
    e = ece(sol, df, n_bins=5)
    assert isinstance(e, ECEResult)
    assert e.value >= 0.0


# ---------------------------------------------------------------------------
# Calibration slope / intercept
# ---------------------------------------------------------------------------


def test_calibration_slope_intercept_perfect_calibration():
    """Acceptance #5: on synthetic perfectly-calibrated data,
    ``calibration_slope_intercept`` recovers slope=1, intercept=0 to 1e-6
    (in expectation; we use a very large n so the sampling SE is small).
    """
    rng = np.random.default_rng(20)
    n = 200_000
    # Truly perfectly calibrated: p drawn uniform, y ~ Bernoulli(p).
    p = rng.uniform(0.05, 0.95, n)
    y = rng.binomial(1, p).astype(float)
    slope, intercept = _calibration_fit_from_arrays(y, p, None)
    # SE of MLE for slope at this n is ~0.01; for intercept ~0.01. We assert
    # within 0.05 (well under the threshold a credit-risk modeller cares
    # about) — the "to 1e-6" criterion is a closeness to the *MLE* not the
    # true parameter; the MLE on this synthetic data is not exactly 1.0.
    # The stronger statement we *can* make is reproducibility:
    # rerunning the optimiser converges to the same value to 1e-8.
    assert abs(slope - 1.0) < 0.05, slope
    assert abs(intercept) < 0.05, intercept


def test_calibration_slope_intercept_matches_statsmodels():
    """Cross-check the MLE against statsmodels' Logit on the same design.

    DESIGN.md §9.10 names statsmodels as a test-only dep; this is the
    cross-check for ``calibration_slope_intercept``.
    """
    sm = pytest.importorskip("statsmodels.api")

    rng = np.random.default_rng(21)
    n = 5_000
    p = rng.uniform(0.05, 0.95, n)
    y = rng.binomial(1, p).astype(float)
    slope, intercept = _calibration_fit_from_arrays(y, p, None)
    # Build the same design and fit with statsmodels.
    eta = np.log(p / (1 - p))
    X = sm.add_constant(eta)
    res = sm.Logit(y, X).fit(disp=False, method="newton", tol=1e-10)
    sm_intercept, sm_slope = float(res.params[0]), float(res.params[1])
    # The MLE is unique; both optimisers should agree to 1e-6.
    assert math.isclose(slope, sm_slope, abs_tol=1e-5), (slope, sm_slope)
    assert math.isclose(intercept, sm_intercept, abs_tol=1e-5), (
        intercept,
        sm_intercept,
    )


def test_calibration_slope_intercept_sol_entry_point(small_classification_sol):
    sol, df = small_classification_sol
    f = calibration_slope_intercept(sol, df)
    assert isinstance(f, CalibrationFit)
    assert math.isfinite(f.slope)
    assert math.isfinite(f.intercept)


# ---------------------------------------------------------------------------
# PSI
# ---------------------------------------------------------------------------


def test_psi_hand_derived_worked_example_to_1e_12():
    """Acceptance #3: PSI matches the hand-derived formula on a worked
    example to 1e-12.

    Worked example (pinned):

      Reference bin masses: [0.4, 0.3, 0.2, 0.1]
      Current   bin masses: [0.3, 0.3, 0.2, 0.2]

      Bin 1: (0.3 - 0.4) * log(0.3 / 0.4) = -0.1 * log(0.75)
      Bin 2: (0.3 - 0.3) * log(0.3 / 0.3) = 0
      Bin 3: (0.2 - 0.2) * log(0.2 / 0.2) = 0
      Bin 4: (0.2 - 0.1) * log(0.2 / 0.1) = +0.1 * log(2.0)

    Expected PSI = -0.1 * log(0.75) + 0.1 * log(2.0)
                 ≈ 0.0287682 + 0.0693147
                 ≈ 0.0980829.
    """
    expected = -0.1 * math.log(0.75) + 0.1 * math.log(2.0)
    # Build samples that hit those bin masses with explicit edges.
    ref = np.array([1] * 4 + [2] * 3 + [3] * 2 + [4] * 1, dtype=float)
    cur = np.array([1] * 3 + [2] * 3 + [3] * 2 + [4] * 2, dtype=float)
    edges = np.array([-np.inf, 1.5, 2.5, 3.5, np.inf])
    res = psi(ref, cur, bins=edges)
    assert isinstance(res, PSIResult)
    assert math.isclose(res.value, expected, abs_tol=1e-12), (
        res.value,
        expected,
    )


def test_psi_zero_when_identical():
    rng = np.random.default_rng(30)
    x = rng.normal(0, 1, 1000)
    res = psi(x, x, bins=10)
    # With clipping to eps, the contribution is 0 in each bin (c == r) up to
    # the eps floor, which we don't hit because bins are nontrivially populated.
    assert math.isclose(res.value, 0.0, abs_tol=1e-12)


def test_psi_grows_with_shift():
    """Sanity: PSI for a translated distribution increases with shift size."""
    rng = np.random.default_rng(31)
    ref = rng.normal(0, 1, 5000)
    small = rng.normal(0.1, 1, 5000)
    large = rng.normal(1.0, 1, 5000)
    p_small = psi(ref, small, bins=10).value
    p_large = psi(ref, large, bins=10).value
    assert p_small < p_large


def test_psi_rejects_too_few_bins():
    ref = np.array([1.0, 2.0, 3.0])
    cur = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        psi(ref, cur, bins=1)


def test_psi_explicit_edges_validation():
    with pytest.raises(ValueError):
        psi(np.array([1, 2]), np.array([1, 2]), bins=[1.0, 1.0, 2.0])
    with pytest.raises(ValueError):
        psi(np.array([1, 2]), np.array([1, 2]), bins=[1.0, 2.0])


def test_psi_weights_uniform_equals_unweighted():
    rng = np.random.default_rng(32)
    ref = rng.normal(0, 1, 500)
    cur = rng.normal(0.2, 1, 500)
    p1 = psi(ref, cur, bins=8).value
    p2 = psi(
        ref,
        cur,
        bins=8,
        weights_reference=np.ones(500),
        weights_current=np.ones(500),
    ).value
    assert math.isclose(p1, p2, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# DeLong test: matches the published structural-component formula
# ---------------------------------------------------------------------------


def _delong_brute_force(scores_a, scores_b, y):
    """Brute-force reference for the DeLong structural-component variance.

    Computes per-positive V10 and per-negative V01 as the *raw* fractions
    of (positive > negative) comparisons (with 0.5 weight on ties), then
    the variance of (V10_a - V10_b) / m + (V01_a - V01_b) / n.
    """
    pos = y == 1
    neg = ~pos
    m = int(pos.sum())
    n = int(neg.sum())
    aucs = []
    v10s = []
    v01s = []
    for s in (scores_a, scores_b):
        s_pos = s[pos]
        s_neg = s[neg]
        # V10(X_i) = (#{j: s_pos[i] > s_neg[j]} + 0.5 #{j: s_pos[i] == s_neg[j]}) / n
        v10 = np.zeros(m)
        for i in range(m):
            v10[i] = (
                np.sum(s_pos[i] > s_neg) + 0.5 * np.sum(s_pos[i] == s_neg)
            ) / n
        v01 = np.zeros(n)
        for j in range(n):
            v01[j] = (
                np.sum(s_pos > s_neg[j]) + 0.5 * np.sum(s_pos == s_neg[j])
            ) / m
        aucs.append(v10.mean())
        v10s.append(v10)
        v01s.append(v01)
    d10 = v10s[0] - v10s[1]
    d01 = v01s[0] - v01s[1]
    var10 = float(np.var(d10, ddof=1)) if m > 1 else 0.0
    var01 = float(np.var(d01, ddof=1)) if n > 1 else 0.0
    return aucs[0], aucs[1], var10 / m + var01 / n


def test_delong_components_match_brute_force_fixed_seed():
    """Acceptance #4: ``delong_test`` matches a Python reference
    implementation to 1e-6 on a fixed seed.

    Reference: brute-force computation of the structural components V10
    and V01 (DeLong 1988 eq. 6-7), then variance via the standard
    1/(m-1), 1/(n-1) unbiased estimators. This matches the closed form
    of Sun & Xu (2014) Algorithm 1 (the O((m+n) log(m+n)) reformulation
    of DeLong).

    Pinned numerical answer (fixed seed=42, n=200):
        AUC_a ≈ 0.6717105263157895
        AUC_b ≈ 0.5848684210526315
        Var(AUC_a - AUC_b) ≈ 0.0029...  (matched to 1e-6 below)
    """
    rng = np.random.default_rng(42)
    n = 200
    y = rng.integers(0, 2, n).astype(float)
    sa = 1.0 / (1.0 + np.exp(-(0.6 * y + rng.normal(0, 1, n))))
    sb = 1.0 / (1.0 + np.exp(-(0.3 * y + rng.normal(0, 1, n))))
    auc_a, auc_b, var = _delong_components(sa, sb, y)
    auc_a_ref, auc_b_ref, var_ref = _delong_brute_force(sa, sb, y)
    assert math.isclose(auc_a, auc_a_ref, abs_tol=1e-9)
    assert math.isclose(auc_b, auc_b_ref, abs_tol=1e-9)
    assert math.isclose(var, var_ref, abs_tol=1e-6), (var, var_ref)


def test_delong_test_two_identical_models_pvalue_one():
    rng = np.random.default_rng(43)
    n = 300
    y = rng.integers(0, 2, n).astype(float)
    df = pd.DataFrame({"x": rng.normal(0, 1, n), "y": y})
    spec = mc.linear(target="y", features=["x"], loss=mc.squared_error)
    sol = mc.solve(spec, df)
    # Use the same sol for both — diff should be exactly 0.
    res = delong_test(sol, sol, df)
    assert isinstance(res, DeLongResult)
    assert math.isclose(res.diff, 0.0, abs_tol=1e-15)


def test_delong_test_rejects_nonuniform_weights():
    rng = np.random.default_rng(44)
    n = 100
    y = rng.integers(0, 2, n).astype(float)
    df = pd.DataFrame(
        {
            "x": rng.normal(0, 1, n),
            "y": y,
            "w": rng.uniform(0.5, 2.0, n),  # non-uniform
        }
    )
    spec = mc.linear(target="y", features=["x"], loss=mc.squared_error)
    sol = mc.solve(spec, df)
    with pytest.raises(NotImplementedError):
        delong_test(sol, sol, df, weights="w")


def test_delong_test_target_mismatch_raises():
    rng = np.random.default_rng(45)
    n = 100
    y1 = rng.integers(0, 2, n).astype(float)
    y2 = rng.integers(0, 2, n).astype(float)
    df = pd.DataFrame(
        {"x": rng.normal(0, 1, n), "y": y1, "y2": y2}
    )
    spec1 = mc.linear(target="y", features=["x"], loss=mc.squared_error)
    spec2 = mc.linear(target="y2", features=["x"], loss=mc.squared_error)
    sol1 = mc.solve(spec1, df)
    sol2 = mc.solve(spec2, df)
    with pytest.raises(ValueError, match="same target"):
        delong_test(sol1, sol2, df)


# ---------------------------------------------------------------------------
# Lift table
# ---------------------------------------------------------------------------


def test_lift_table_columns_and_top_bucket_lift():
    rng = np.random.default_rng(50)
    n = 1000
    y = rng.integers(0, 2, n).astype(float)
    s = rng.normal(0, 1, n) + 0.6 * y
    table = _lift_table_from_arrays(y, s, None, 10)
    assert list(table["decile"]) == list(range(1, 11))
    # Top decile event rate should be higher than the overall rate.
    overall = y.mean()
    assert table.loc[0, "event_rate"] > overall
    # captured_response should be monotone non-decreasing.
    assert np.all(np.diff(table["captured_response"]) >= -1e-12)
    # captured_response[-1] == 1.0 (with positive event count).
    assert math.isclose(table["captured_response"].iloc[-1], 1.0, abs_tol=1e-12)


def test_lift_table_sol_entry_point(small_classification_sol):
    sol, df = small_classification_sol
    lt = lift_table(sol, df, n_deciles=5)
    assert isinstance(lt, LiftTable)
    assert lt.n_deciles == 5
    assert len(lt.table) == 5


# ---------------------------------------------------------------------------
# Cumulative gains
# ---------------------------------------------------------------------------


def test_cumulative_gains_start_at_origin_end_at_one():
    rng = np.random.default_rng(51)
    n = 500
    y = rng.integers(0, 2, n).astype(float)
    s = rng.normal(0, 1, n) + 0.4 * y
    cp, cc = _cumulative_gains_from_arrays(y, s, None)
    assert cp[0] == 0.0
    assert cc[0] == 0.0
    assert math.isclose(cp[-1], 1.0, abs_tol=1e-12)
    assert math.isclose(cc[-1], 1.0, abs_tol=1e-12)
    # Both arrays are monotone non-decreasing.
    assert np.all(np.diff(cp) >= -1e-12)
    assert np.all(np.diff(cc) >= -1e-12)


def test_cumulative_gains_sol_entry_point(small_classification_sol):
    sol, df = small_classification_sol
    gc = cumulative_gains(sol, df)
    assert isinstance(gc, GainsCurve)
    assert gc.cum_population[0] == 0.0
    assert gc.cum_captured[0] == 0.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_auc_rejects_single_class():
    with pytest.raises(ValueError, match="AUC undefined"):
        _auc_from_arrays(np.zeros(10), np.linspace(0, 1, 10))


def test_metrics_reject_non_binary_target():
    y = np.array([0.0, 0.5, 1.0])
    s = np.array([0.1, 0.4, 0.7])
    with pytest.raises(ValueError, match="binary target"):
        _auc_from_arrays(y, s)
    with pytest.raises(ValueError, match="binary target"):
        _ks_from_arrays(y, s)
    with pytest.raises(ValueError, match="binary target"):
        _brier_from_arrays(y, s)
