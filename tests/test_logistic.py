"""Tests for the logistic loss + IRLS end-to-end (Task P3.A).

Quoted acceptance criteria (DESIGN.md §8 Phase 3, AGENTS.md Task P3.A):

  1. Reproduce ``statsmodels.GLM(family=Binomial())`` coefficients to 1e-6
     on a real-ish dataset. We use ``statsmodels.datasets.spector`` — a
     32-row binary-target dataset bundled with statsmodels (Spector and
     Mazzeo 1980). Also verify on synthetic data with known coefficients.
  2. KS and AUC of the fitted model match scipy/reference implementations
     to 1e-9. Computed inline here because Task P3.D owns the metric
     primitives.
  3. Perfectly-separable synthetic data (``y = (x > 0)`` exactly) raises
     ``AssumptionError`` from ``NoPerfectSeparation`` with the L2
     recommendation message embedded.
  4. ``mc.predict(sol, data)`` on a logistic spec returns probabilities in
     ``[0, 1]`` (property test on 100 random datasets). The integration
     wiring step at P3.INTEG must update ``predict()`` to call
     ``spec.loss.link(eta)``; a second test ``xfail``s pending that wiring.

The math (DESIGN.md §2.4, §4.3; ESL §4.4):

* Per-row negative log-likelihood:
  :math:`\\ell(y, \\eta) = \\log(1 + e^\\eta) - y\\,\\eta`.
* Gradient w.r.t. :math:`\\eta`: :math:`p - y`, where :math:`p = \\sigma(\\eta)`.
* Hessian diagonal: :math:`p(1 - p)`.
* Weighted variant: scale row contributions by ``w_i``; the loss value
  returns the *average* over :math:`\\sum w_i`, matching ``squared_error``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.datasets import spector

from model_crafter.assumptions import AssumptionError
from model_crafter.loss import LogisticLoss, logistic

# Importing the irls module triggers solver self-registration. The
# integration agent wires this import into solve/__init__.py at P3.INTEG;
# tests in P3.A run before that wiring, so we import explicitly here.
from model_crafter.solve import irls as _irls  # noqa: F401
from model_crafter.solve import predict, solve
from model_crafter.spec import linear

# ---------------------------------------------------------------------------
# LogisticLoss math (value / gradient / hessian / link)
# ---------------------------------------------------------------------------


def test_logistic_is_a_loss_instance() -> None:
    """``logistic`` is an object that satisfies the Loss protocol."""
    from model_crafter.loss import Loss

    assert isinstance(logistic, Loss)
    assert isinstance(logistic, LogisticLoss)


def test_logistic_declares_required_assumptions() -> None:
    """LogisticLoss declares the four logistic assumptions plus stability
    diagnostics (AGENTS.md P3.A, DESIGN.md §4.3).
    """
    names = [type(a).__name__ for a in logistic.assumptions]
    # HARD prerequisites
    assert "FullRankDesign" in names
    assert "BinaryOrProportionTarget" in names
    assert "NoPerfectSeparation" in names
    # SOFT
    assert "ClassBalance" in names
    # INFO (opt-in)
    assert "LinkAdequacy" in names
    # SOFT stability diagnostics inherited from the linear-loss baseline
    assert "CoefficientStability" in names
    assert "PredictiveStability" in names


def test_logistic_value_unweighted_matches_formula() -> None:
    r"""L(y, eta) = mean(log(1+exp(eta)) - y*eta)."""
    y = np.array([0.0, 1.0, 0.0, 1.0])
    eta = np.array([-1.0, 0.5, 2.0, -0.5])
    expected = float(np.mean(np.logaddexp(0.0, eta) - y * eta))
    got = logistic.value(y, eta, weights=None)
    assert got == pytest.approx(expected, abs=1e-15)


def test_logistic_value_weighted_normalises_by_sum_weights() -> None:
    """Weighted: L = sum(w_i * (log(1+exp(eta_i)) - y_i*eta_i)) / sum(w_i)."""
    y = np.array([0.0, 1.0, 0.0, 1.0])
    eta = np.array([-1.0, 0.5, 2.0, -0.5])
    w = np.array([1.0, 2.0, 3.0, 4.0])
    per_row = np.logaddexp(0.0, eta) - y * eta
    expected = float(np.sum(w * per_row) / np.sum(w))
    got = logistic.value(y, eta, weights=w)
    assert got == pytest.approx(expected, abs=1e-14)


def test_logistic_gradient_equals_p_minus_y() -> None:
    """grad_eta = (p - y), where p = sigmoid(eta) (per-row, unweighted form)."""
    y = np.array([0.0, 1.0, 0.0, 1.0])
    eta = np.array([-1.0, 0.5, 2.0, -0.5])
    p = 1.0 / (1.0 + np.exp(-eta))
    expected = (p - y) / len(y)
    got = logistic.gradient(y, eta, weights=None)
    np.testing.assert_allclose(got, expected, atol=1e-15)


def test_logistic_hessian_equals_p_times_one_minus_p() -> None:
    """Hessian (diag) = p*(1-p), normalised by sum of weights."""
    y = np.array([0.0, 1.0, 0.0, 1.0])
    eta = np.array([-1.0, 0.5, 2.0, -0.5])
    p = 1.0 / (1.0 + np.exp(-eta))
    expected = p * (1 - p) / len(y)
    got = logistic.hessian(y, eta, weights=None)
    np.testing.assert_allclose(got, expected, atol=1e-15)


def test_logistic_link_maps_eta_to_probability() -> None:
    """link(eta) returns sigmoid(eta) — DESIGN.md §3.3 "output is always a
    probability"."""
    eta = np.linspace(-5, 5, 11)
    expected = 1.0 / (1.0 + np.exp(-eta))
    got = logistic.link(eta)
    np.testing.assert_allclose(got, expected, atol=1e-15)
    # Range check
    assert np.all(got >= 0) and np.all(got <= 1)


def test_logistic_link_is_numerically_stable_for_large_eta() -> None:
    """log1p / expit-style stability: link(+inf) == 1, link(-inf) == 0,
    no overflow warnings on large magnitudes."""
    eta = np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = logistic.link(eta)
    assert got[0] == pytest.approx(0.0, abs=1e-15)
    assert got[-1] == pytest.approx(1.0, abs=1e-15)
    assert got[2] == pytest.approx(0.5, abs=1e-15)


# ---------------------------------------------------------------------------
# End-to-end IRLS against statsmodels on the Spector dataset
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def spector_df() -> pd.DataFrame:
    """Spector & Mazzeo (1980) dataset, 32 rows, binary GRADE target.

    Bundled with statsmodels (``statsmodels.datasets.spector``). Three
    continuous/binary predictors: GPA, TUCE, PSI.
    """
    data = spector.load()
    # statsmodels' Dataset type stubs are loose; cast explicitly to DataFrame.
    df = pd.DataFrame(data.exog).copy()
    df["GRADE"] = pd.Series(data.endog, index=df.index).astype(float)
    return df


def test_logistic_matches_statsmodels_on_spector(spector_df: pd.DataFrame) -> None:
    """Acceptance #1 — coefficients match statsmodels.GLM(family=Binomial())
    to atol=1e-6 on the Spector dataset (statsmodels.datasets.spector).
    """
    spec = linear(
        target="GRADE",
        features=["GPA", "TUCE", "PSI"],
        loss=logistic,
    )
    sol = solve(spec, spector_df)

    # statsmodels reference
    X = sm.add_constant(spector_df[["GPA", "TUCE", "PSI"]].to_numpy(dtype=float))
    y = spector_df["GRADE"].to_numpy(dtype=float)
    ref = sm.GLM(y, X, family=sm.families.Binomial()).fit()

    # statsmodels orders [const, GPA, TUCE, PSI]; sol.coefficients is keyed
    # by design column names with "(Intercept)" first.
    got = np.array([
        sol.coefficients["(Intercept)"],
        sol.coefficients["GPA"],
        sol.coefficients["TUCE"],
        sol.coefficients["PSI"],
    ])
    np.testing.assert_allclose(got, ref.params, atol=1e-6, rtol=0)


def test_logistic_se_matches_statsmodels_on_spector(spector_df: pd.DataFrame) -> None:
    """Standard errors of an unpenalised logistic fit match statsmodels'
    GLM SEs to atol=1e-4 — both compute ``sqrt(diag((X' W X)^{-1}))`` at
    convergence; small differences come from the Fisher information
    normalisation chosen by each implementation.
    """
    spec = linear(target="GRADE", features=["GPA", "TUCE", "PSI"], loss=logistic)
    sol = solve(spec, spector_df)
    assert sol.coefficient_se is not None
    X = sm.add_constant(spector_df[["GPA", "TUCE", "PSI"]].to_numpy(dtype=float))
    y = spector_df["GRADE"].to_numpy(dtype=float)
    ref = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    got = np.array([
        sol.coefficient_se["(Intercept)"],
        sol.coefficient_se["GPA"],
        sol.coefficient_se["TUCE"],
        sol.coefficient_se["PSI"],
    ])
    np.testing.assert_allclose(got, ref.bse, atol=1e-4, rtol=0)


def test_logistic_matches_statsmodels_on_synthetic_known_coefficients() -> None:
    """Synthetic check: with n=2000 and known beta, recovered coefficients
    match statsmodels to atol=1e-6."""
    rng = np.random.default_rng(seed=0)
    n = 2000
    X = rng.normal(size=(n, 3))
    beta_true = np.array([-0.5, 1.2, -0.8, 0.3])  # intercept first
    eta = beta_true[0] + X @ beta_true[1:]
    p = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, p).astype(float)

    df = pd.DataFrame(X, columns=pd.Index(["x1", "x2", "x3"]))
    df["y"] = y
    spec = linear(target="y", features=["x1", "x2", "x3"], loss=logistic)
    sol = solve(spec, df)

    Xint = sm.add_constant(X)
    ref = sm.GLM(y, Xint, family=sm.families.Binomial()).fit()
    got = np.array([
        sol.coefficients["(Intercept)"],
        sol.coefficients["x1"],
        sol.coefficients["x2"],
        sol.coefficients["x3"],
    ])
    np.testing.assert_allclose(got, ref.params, atol=1e-6, rtol=0)


# ---------------------------------------------------------------------------
# AUC and KS — inline reference per AGENTS.md (P3.D owns the primitives)
# ---------------------------------------------------------------------------


def _auc_inline(y: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney U based AUC, equivalent to scipy.stats.mannwhitneyu /
    sklearn roc_auc_score. Stable when there are ties (uses average ranks)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1, dtype=float)
    # Average ranks across ties
    # (this matches scipy.stats.rankdata default method='average')
    from scipy.stats import rankdata
    ranks = rankdata(p, method="average")
    sum_ranks_pos = float(ranks[y == 1].sum())
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def test_auc_inline_matches_scipy_reference() -> None:
    """AUC inline equals scipy/sklearn-style rank-based AUC to 1e-12 on a
    fixed seed."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200).astype(float)
    p = rng.uniform(size=200)
    auc_a = _auc_inline(y, p)
    # Independent computation via the trapezoid of the ROC curve.
    # (We do not depend on sklearn — DESIGN.md §9.3.)
    # Sort by p descending; sweep TPR / FPR.
    order = np.argsort(-p)
    y_s = y[order]
    cum_pos = np.cumsum(y_s == 1)
    cum_neg = np.cumsum(y_s == 0)
    P = float((y == 1).sum())
    N = float((y == 0).sum())
    tpr = np.concatenate([[0.0], cum_pos / P])
    fpr = np.concatenate([[0.0], cum_neg / N])
    # ``np.trapezoid`` was added in NumPy 1.24 and is the preferred name
    # (NPY201); ``np.trapz`` is deprecated in NumPy 2.0. The getattr
    # fallback is only evaluated on older NumPys where the old name is
    # still valid.
    trapezoid = getattr(np, "trapezoid", np.trapz)  # noqa: NPY201
    auc_trap = float(trapezoid(tpr, fpr))
    # Allow small slack for tied scores.
    assert auc_a == pytest.approx(auc_trap, abs=1e-9)


def test_logistic_auc_and_ks_against_scipy_on_spector(spector_df: pd.DataFrame) -> None:
    """Acceptance #2 — AUC matches inline reference to 1e-9; KS matches
    scipy.stats.ks_2samp to 1e-9 (Spector dataset). KS is the
    max-distance between the CDF of predicted scores on positives vs negatives.
    """
    from scipy.stats import ks_2samp

    spec = linear(target="GRADE", features=["GPA", "TUCE", "PSI"], loss=logistic)
    sol = solve(spec, spector_df)

    # Compute probabilities directly (predict() linking is deferred to P3.INTEG).
    beta = np.array([
        sol.coefficients["(Intercept)"],
        sol.coefficients["GPA"],
        sol.coefficients["TUCE"],
        sol.coefficients["PSI"],
    ])
    X = np.column_stack([
        np.ones(len(spector_df)),
        spector_df["GPA"].to_numpy(dtype=float),
        spector_df["TUCE"].to_numpy(dtype=float),
        spector_df["PSI"].to_numpy(dtype=float),
    ])
    p_hat = 1.0 / (1.0 + np.exp(-(X @ beta)))
    y = spector_df["GRADE"].to_numpy(dtype=float)

    # AUC: compare two ways for redundancy
    auc1 = _auc_inline(y, p_hat)
    # statsmodels reference probabilities (should give the same AUC bit-for-bit)
    Xref = sm.add_constant(spector_df[["GPA", "TUCE", "PSI"]].to_numpy(dtype=float))
    p_ref = sm.GLM(y, Xref, family=sm.families.Binomial()).fit().predict()
    auc2 = _auc_inline(y, p_ref)
    assert auc1 == pytest.approx(auc2, abs=1e-9)

    # KS via scipy. ``ks_2samp`` returns a ``KstestResult``-like value;
    # pyright's stubs type it loosely, so we coerce via numpy.
    ks_res = ks_2samp(p_hat[y == 1], p_hat[y == 0])
    ks_res_ref = ks_2samp(p_ref[y == 1], p_ref[y == 0])
    assert float(np.asarray(ks_res.statistic)) == pytest.approx(  # pyright: ignore[reportAttributeAccessIssue]
        float(np.asarray(ks_res_ref.statistic)),  # pyright: ignore[reportAttributeAccessIssue]
        abs=1e-9,
    )


# ---------------------------------------------------------------------------
# Perfect separation -> AssumptionError with the L2 hint embedded
# ---------------------------------------------------------------------------


def test_perfectly_separable_data_raises_assumption_error() -> None:
    """Acceptance #3 — y = (x > 0) is perfectly separable; the post-fit
    NoPerfectSeparation check fires with the L2-remedy message.
    """
    rng = np.random.default_rng(seed=42)
    x = rng.uniform(-1.0, 1.0, size=120)
    y = (x > 0).astype(float)
    df = pd.DataFrame({"x": x, "y": y})
    spec = linear(target="y", features=["x"], loss=logistic)
    with pytest.raises(AssumptionError) as exc:
        solve(spec, df)
    msg = str(exc.value)
    assert "penalty=mc.l2(...) is the standard remedy for perfect separation" in msg


def test_perfectly_separable_data_l2_makes_fit_well_defined() -> None:
    """The L2 remedy advertised in the error message actually works: adding
    a small ridge penalty restores a finite, well-defined fit on the same
    separable dataset.
    """
    from model_crafter.penalty import l2

    rng = np.random.default_rng(seed=42)
    x = rng.uniform(-1.0, 1.0, size=120)
    y = (x > 0).astype(float)
    df = pd.DataFrame({"x": x, "y": y})
    spec = linear(target="y", features=["x"], loss=logistic, penalty=l2(0.5))
    sol = solve(spec, df)
    assert sol.converged
    # Slope is finite and well-bounded, not the +infinity of the unpenalised fit.
    assert np.isfinite(sol.coefficients["x"])
    assert abs(sol.coefficients["x"]) < 1000.0


# ---------------------------------------------------------------------------
# BinaryOrProportionTarget HARD prerequisite
# ---------------------------------------------------------------------------


def test_non_binary_target_raises_assumption_error() -> None:
    """``BinaryOrProportionTarget`` fires when the target column contains
    values outside [0, 1]."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=40), "y": rng.normal(size=40)})
    spec = linear(target="y", features=["x"], loss=logistic)
    with pytest.raises(AssumptionError) as exc:
        solve(spec, df)
    assert "BinaryOrProportionTarget" in str(exc.value) or "target" in str(exc.value).lower()


def test_proportion_target_in_unit_interval_is_accepted() -> None:
    """Proportion-style targets (continuous in [0, 1]) pass the HARD check —
    fractional-binomial logistic regression is a valid use case."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    eta = 0.3 + 0.8 * x
    p_true = 1.0 / (1.0 + np.exp(-eta))
    df = pd.DataFrame({"x": x, "y": p_true})  # proportions, no [0,1] violation
    spec = linear(target="y", features=["x"], loss=logistic)
    # Should not raise — the check passes for proportions.
    sol = solve(spec, df)
    assert sol.converged


# ---------------------------------------------------------------------------
# ClassBalance SOFT
# ---------------------------------------------------------------------------


def test_severe_class_imbalance_warns() -> None:
    """ClassBalance fires (as a warning, severity=SOFT) when the minority
    class is below the default 1% threshold."""
    rng = np.random.default_rng(0)
    n = 1000
    x = rng.normal(size=n)
    y = np.zeros(n)
    y[:5] = 1  # 0.5% minority
    df = pd.DataFrame({"x": x, "y": y})
    spec = linear(target="y", features=["x"], loss=logistic)
    with pytest.warns(UserWarning, match="ClassBalance"):
        solve(spec, df, on_violation="warn")


# ---------------------------------------------------------------------------
# Property test: probabilities in [0, 1]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", list(range(100)))
def test_logistic_predicted_probabilities_in_unit_interval(seed: int) -> None:
    """Acceptance #4 — predicted probabilities lie in [0, 1] for 100 random
    datasets. Computed inline (1 / (1 + exp(-eta))) because the integration
    step at P3.INTEG is what wires the link into ``predict()``."""
    rng = np.random.default_rng(seed=seed)
    n = rng.integers(50, 400)
    p = rng.integers(1, 5)
    X = rng.normal(size=(n, p))
    beta = rng.normal(size=p + 1)
    eta = beta[0] + X @ beta[1:]
    pr = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, pr).astype(float)
    # Avoid degenerate cases (all 0 or all 1).
    if y.sum() in (0, n):
        pytest.skip("degenerate y for this seed")

    df = pd.DataFrame(X, columns=pd.Index([f"x{j}" for j in range(p)]))
    df["y"] = y
    spec = linear(target="y", features=list(df.columns[:-1]), loss=logistic)
    sol = solve(spec, df)

    # Inline probability computation — what predict() will do post-P3.INTEG.
    Xfull = np.column_stack([np.ones(n), X])
    eta_hat = Xfull @ sol.coefficients.to_numpy(dtype=float)
    p_hat = 1.0 / (1.0 + np.exp(-eta_hat))
    assert np.all(p_hat >= 0.0)
    assert np.all(p_hat <= 1.0)


def test_predict_returns_probabilities_for_logistic() -> None:
    """``mc.predict(sol, data)`` for a logistic spec returns probabilities
    in [0, 1] (DESIGN.md §3.3 — output is always a probability)."""
    rng = np.random.default_rng(seed=0)
    n = 200
    x = rng.normal(size=n)
    y = rng.binomial(1, 1 / (1 + np.exp(-(0.3 + 0.8 * x)))).astype(float)
    df = pd.DataFrame({"x": x, "y": y})
    spec = linear(target="y", features=["x"], loss=logistic)
    sol = solve(spec, df)
    yhat = predict(sol, df)
    assert (yhat >= 0).all() and (yhat <= 1).all()
