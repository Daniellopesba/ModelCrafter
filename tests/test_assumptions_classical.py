"""Tests for the classical-inference (INFO) assumptions (P1.B).

Acceptance criteria pinned here:

- "Unit tests for each concrete assumption: synthetic data triggers PASS,
  synthetic data triggers FAIL, with statistic values verified against a
  reference (scipy or hand-derived)."
- "``LowVIF`` is INFO severity and its ``suggestion`` field reads (verbatim):
  ``High collinearity detected (max VIF = {x:.1f}). ESL §3.4.1 recommends
  ridge or lasso regularization rather than feature pruning. Consider
  penalty=mc.l2(...) or penalty=mc.l1(...).``"
- "``classical_inference=False`` (default) suppresses INFO-level checks
  entirely; ``classical_inference=True`` runs them and includes them in the
  report at INFO severity."

For Breusch-Pagan and Durbin-Watson the statistic is cross-checked against
``statsmodels`` (test-only dep). For Shapiro-Wilk / Anderson-Darling we
cross-check against ``scipy.stats`` directly (which is what we use to compute
it).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

from model_crafter.assumptions import (
    Assumption,
    AssumptionReport,
    Homoscedasticity,
    Independence,
    LowVIF,
    ResidualNormality,
    Severity,
    run_assumptions,
)

# ---------------------------------------------------------------------------
# Minimal stub spec / solution shapes (documented as the contract with P1.A).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StubLoss:
    assumptions: tuple[Assumption, ...] = ()


@dataclass(frozen=True)
class StubTerm:
    name: str
    assumptions: tuple[Assumption, ...] = ()


@dataclass(frozen=True)
class StubSpec:
    target: str
    features: tuple[Any, ...]
    loss: StubLoss
    penalty: Any | None = None
    intercept: bool = True


@dataclass(frozen=True)
class StubSolution:
    coefficients: pd.Series
    coefficient_se: pd.Series | None = None
    design_columns: tuple[str, ...] = field(default_factory=tuple)


def _spec_with(features: tuple[str, ...], assumptions: tuple[Assumption, ...]) -> StubSpec:
    return StubSpec(
        target="y",
        features=tuple(StubTerm(c) for c in features),
        loss=StubLoss(assumptions=assumptions),
        intercept=True,
    )


def _ols_fit_solution(df: pd.DataFrame, features: tuple[str, ...]) -> StubSolution:
    """Fit OLS via the normal equations and pack it into a StubSolution.

    Independent of P1.A — we build the minimum solution-shaped object that
    the assumption layer reads.
    """
    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy() for c in features])
    y = df["y"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    cols = ("(Intercept)",) + features
    return StubSolution(
        coefficients=pd.Series(beta, index=list(cols)),
        coefficient_se=None,
        design_columns=cols,
    )


# ---------------------------------------------------------------------------
# ResidualNormality: Shapiro-Wilk for n < 5000, Anderson-Darling otherwise.
# ---------------------------------------------------------------------------


def test_residual_normality_passes_on_normal_residuals():
    # Seed 1 yields Shapiro p~0.71 — comfortably above alpha=0.05.
    rng = np.random.default_rng(1)
    n = 200
    x = rng.standard_normal(n)
    y = 2 * x + rng.standard_normal(n)  # Gaussian noise
    df = pd.DataFrame({"x": x, "y": y})
    sol = _ols_fit_solution(df, ("x",))
    a = ResidualNormality()
    res = a.check(_spec_with(("x",), (a,)), df, solution=sol)
    # Reference statistic from scipy directly.
    resid = y - (sol.coefficients["(Intercept)"] + sol.coefficients["x"] * x)
    ref_stat, ref_p = stats.shapiro(resid)
    assert res.statistic == pytest.approx(float(ref_stat), rel=1e-12, abs=1e-12)
    # Pass at the default 0.05 alpha
    assert res.passed is True
    assert ref_p > 0.05


def test_residual_normality_fails_on_skewed_residuals():
    rng = np.random.default_rng(1)
    n = 200
    x = rng.standard_normal(n)
    # Heavy lognormal noise -> non-normal residuals
    noise = rng.lognormal(mean=0.0, sigma=1.0, size=n) - np.exp(0.5)
    y = 2 * x + noise
    df = pd.DataFrame({"x": x, "y": y})
    sol = _ols_fit_solution(df, ("x",))
    a = ResidualNormality()
    res = a.check(_spec_with(("x",), (a,)), df, solution=sol)
    resid = y - (sol.coefficients["(Intercept)"] + sol.coefficients["x"] * x)
    ref_stat, ref_p = stats.shapiro(resid)
    assert res.statistic == pytest.approx(float(ref_stat), rel=1e-12, abs=1e-12)
    assert ref_p < 0.05
    assert res.passed is False


def test_residual_normality_uses_anderson_darling_above_5000():
    rng = np.random.default_rng(2)
    n = 6000
    x = rng.standard_normal(n)
    y = 2 * x + rng.standard_normal(n)
    df = pd.DataFrame({"x": x, "y": y})
    sol = _ols_fit_solution(df, ("x",))
    a = ResidualNormality()
    res = a.check(_spec_with(("x",), (a,)), df, solution=sol)
    resid = y - (sol.coefficients["(Intercept)"] + sol.coefficients["x"] * x)
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("ignore", FutureWarning)
        ad: Any = stats.anderson(resid, dist="norm")
    assert res.statistic == pytest.approx(float(ad.statistic), rel=1e-12, abs=1e-12)
    assert "anderson" in res.message.lower() or "ad" in res.message.lower()


# ---------------------------------------------------------------------------
# Homoscedasticity: Breusch-Pagan. Cross-check against statsmodels.het_breuschpagan.
# ---------------------------------------------------------------------------


def test_homoscedasticity_passes_on_constant_variance():
    rng = np.random.default_rng(3)
    n = 300
    x = rng.standard_normal(n)
    y = 2 * x + rng.standard_normal(n)
    df = pd.DataFrame({"x": x, "y": y})
    sol = _ols_fit_solution(df, ("x",))
    a = Homoscedasticity()
    res = a.check(_spec_with(("x",), (a,)), df, solution=sol)

    # Reference statistic from statsmodels.
    resid = y - (sol.coefficients["(Intercept)"] + sol.coefficients["x"] * x)
    X_with_const = sm.add_constant(np.column_stack([x]))
    bp_lm, bp_lm_p, _, _ = sm.stats.diagnostic.het_breuschpagan(resid, X_with_const)
    assert res.statistic == pytest.approx(float(bp_lm), rel=1e-10, abs=1e-10)
    assert bp_lm_p > 0.05
    assert res.passed is True


def test_homoscedasticity_fails_on_heteroscedastic_data():
    rng = np.random.default_rng(4)
    n = 500
    x = rng.uniform(0.1, 5.0, size=n)
    # Variance proportional to x^2 — strong heteroscedasticity
    eps = rng.standard_normal(n) * x
    y = 1.0 + 2.0 * x + eps
    df = pd.DataFrame({"x": x, "y": y})
    sol = _ols_fit_solution(df, ("x",))
    a = Homoscedasticity()
    res = a.check(_spec_with(("x",), (a,)), df, solution=sol)

    resid = y - (sol.coefficients["(Intercept)"] + sol.coefficients["x"] * x)
    X_with_const = sm.add_constant(np.column_stack([x]))
    bp_lm, bp_lm_p, _, _ = sm.stats.diagnostic.het_breuschpagan(resid, X_with_const)
    assert res.statistic == pytest.approx(float(bp_lm), rel=1e-10, abs=1e-10)
    assert bp_lm_p < 0.05
    assert res.passed is False


# ---------------------------------------------------------------------------
# Independence: Durbin-Watson, d = sum((e_t - e_{t-1})^2) / sum(e_t^2).
# ---------------------------------------------------------------------------


def test_independence_passes_on_iid_residuals():
    rng = np.random.default_rng(5)
    n = 200
    x = rng.standard_normal(n)
    y = 2 * x + rng.standard_normal(n)
    df = pd.DataFrame({"x": x, "y": y})
    sol = _ols_fit_solution(df, ("x",))
    a = Independence()
    res = a.check(_spec_with(("x",), (a,)), df, solution=sol)

    resid = y - (sol.coefficients["(Intercept)"] + sol.coefficients["x"] * x)
    ref_d = sm.stats.stattools.durbin_watson(resid)
    assert res.statistic == pytest.approx(float(ref_d), rel=1e-12, abs=1e-12)
    # d ~ 2 means no autocorrelation; default pass band is 1.5 <= d <= 2.5
    assert res.statistic is not None
    assert 1.5 <= res.statistic <= 2.5
    assert res.passed is True


def test_independence_fails_on_autocorrelated_residuals():
    rng = np.random.default_rng(6)
    n = 400
    x = rng.standard_normal(n)
    # AR(1) residuals with phi=0.9
    eps = np.zeros(n)
    eps[0] = rng.standard_normal()
    for t in range(1, n):
        eps[t] = 0.9 * eps[t - 1] + rng.standard_normal()
    y = 2 * x + eps
    df = pd.DataFrame({"x": x, "y": y})
    sol = _ols_fit_solution(df, ("x",))
    a = Independence()
    res = a.check(_spec_with(("x",), (a,)), df, solution=sol)
    resid = y - (sol.coefficients["(Intercept)"] + sol.coefficients["x"] * x)
    ref_d = sm.stats.stattools.durbin_watson(resid)
    assert res.statistic == pytest.approx(float(ref_d), rel=1e-12, abs=1e-12)
    # Strong positive autocorrelation -> d << 2
    assert res.statistic is not None
    assert res.statistic < 1.5
    assert res.passed is False


# ---------------------------------------------------------------------------
# LowVIF: max VIF across features; suggestion-string contract is verbatim.
# ---------------------------------------------------------------------------


def test_low_vif_passes_on_uncorrelated_features():
    rng = np.random.default_rng(7)
    n = 300
    X = rng.standard_normal((n, 3))
    df = pd.DataFrame(X, columns=pd.Index(["x1", "x2", "x3"]))
    df["y"] = X.sum(axis=1) + rng.standard_normal(n)
    a = LowVIF()
    res = a.check(_spec_with(("x1", "x2", "x3"), (a,)), df)

    # Hand-compute reference max VIF.
    vifs = []
    for j, col in enumerate(["x1", "x2", "x3"]):
        others = [c for c in ["x1", "x2", "x3"] if c != col]
        Xj = df[col].to_numpy()
        Xrest = np.column_stack([np.ones(n)] + [df[c].to_numpy() for c in others])
        beta, *_ = np.linalg.lstsq(Xrest, Xj, rcond=None)
        Xj_hat = Xrest @ beta
        ss_res = float(np.sum((Xj - Xj_hat) ** 2))
        ss_tot = float(np.sum((Xj - Xj.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot
        vifs.append(1.0 / (1.0 - r2))
    assert res.statistic == pytest.approx(max(vifs), rel=1e-10, abs=1e-10)
    assert res.passed is True


def test_low_vif_fails_and_emits_verbatim_suggestion():
    """ACCEPTANCE: LowVIF.suggestion is the exact ESL §3.4.1 string."""
    rng = np.random.default_rng(8)
    n = 300
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    # x3 is near-perfectly collinear with x1
    x3 = x1 + 0.01 * rng.standard_normal(n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    df["y"] = x1 + x2 + rng.standard_normal(n)
    a = LowVIF()
    res = a.check(_spec_with(("x1", "x2", "x3"), (a,)), df)
    assert res.passed is False
    assert res.severity is Severity.INFO
    assert res.statistic is not None and res.statistic > 10

    expected_suggestion = (
        f"High collinearity detected (max VIF = {res.statistic:.1f}). "
        "ESL §3.4.1 recommends ridge or lasso regularization rather than "
        "feature pruning. Consider penalty=mc.l2(...) or penalty=mc.l1(...)."
    )
    assert res.suggestion == expected_suggestion


# ---------------------------------------------------------------------------
# Wiring through run_assumptions: classical_inference flag gates INFO,
# and INFO results never warn or raise.
# ---------------------------------------------------------------------------


def test_classical_inference_flag_gates_info_checks_endtoend():
    """Without classical_inference=True, INFO checks are not in the report.
    With it, they are present at INFO severity."""
    rng = np.random.default_rng(9)
    n = 300
    x1 = rng.standard_normal(n)
    x2 = x1 + 0.001 * rng.standard_normal(n)  # collinear
    df = pd.DataFrame({"x1": x1, "x2": x2})
    df["y"] = x1 + rng.standard_normal(n)
    spec = _spec_with(("x1", "x2"), (LowVIF(),))

    rep_no = run_assumptions(spec, df)
    assert isinstance(rep_no, AssumptionReport)
    assert not any(r.name == "LowVIF" for r in rep_no.results)

    rep_yes = run_assumptions(spec, df, classical_inference=True)
    vif_results = [r for r in rep_yes.results if r.name == "LowVIF"]
    assert len(vif_results) == 1
    assert vif_results[0].severity is Severity.INFO
