"""Phase 3 integration tests.

Quotes AGENTS.md Task P3.INTEG acceptance criteria:

  - End-to-end logistic regression with ``mc.tune`` + ``mc.bootstrap`` +
    ``mc.performance`` runs on a real-ish dataset.
  - ``mc.solve(..., classical_inference=True)`` produces an
    ``AssumptionReport`` including ``LinkAdequacy`` for logistic
    regression.
  - Stability assumptions (``CoefficientStability``,
    ``PredictiveStability``) from P1.B's framework now run automatically
    when CV is available; verify they fire on an unstable dataset.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import model_crafter as mc
from model_crafter.assumptions import Severity

# ---------------------------------------------------------------------------
# Synthetic logistic dataset for end-to-end checks
# ---------------------------------------------------------------------------


def _make_credit_panel(n: int = 800, seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    income = rng.normal(loc=0.0, scale=1.0, size=n)
    age = rng.normal(loc=0.0, scale=1.0, size=n)
    tenure = rng.normal(loc=0.0, scale=1.0, size=n)
    eta = -1.5 + 1.2 * income - 0.8 * age + 0.4 * tenure
    p = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, p).astype(float)
    origination_dt = pd.date_range("2022-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "default_12m": y,
            "income": income,
            "age": age,
            "tenure": tenure,
            "origination_dt": origination_dt,
        }
    )


def test_logistic_predict_is_a_probability() -> None:
    """DESIGN.md §3.3: every model output is a probability."""
    df = _make_credit_panel(n=200)
    spec = mc.linear(
        target="default_12m",
        features=["income", "age", "tenure"],
        loss=mc.logistic,
    )
    sol = mc.solve(spec, df)
    yhat = mc.predict(sol, df)
    assert (yhat >= 0).all() and (yhat <= 1).all()


def test_logistic_with_classical_inference_includes_link_adequacy() -> None:
    """``classical_inference=True`` surfaces LinkAdequacy at INFO severity."""
    df = _make_credit_panel(n=400)
    spec = mc.linear(
        target="default_12m",
        features=["income", "age", "tenure"],
        loss=mc.logistic,
    )
    sol = mc.solve(spec, df, classical_inference=True)
    info_results = sol.assumptions.by_severity()[Severity.INFO]
    link_adequacy = next(
        (r for r in info_results if r.name == "LinkAdequacy"), None
    )
    assert link_adequacy is not None, (
        f"LinkAdequacy missing from INFO results; got {[r.name for r in info_results]}"
    )


def test_default_logistic_excludes_info_checks() -> None:
    """Without ``classical_inference=True``, INFO checks (incl. LinkAdequacy)
    don't appear (DESIGN.md §4)."""
    df = _make_credit_panel(n=200)
    spec = mc.linear(
        target="default_12m",
        features=["income", "age", "tenure"],
        loss=mc.logistic,
    )
    sol = mc.solve(spec, df)
    info_results = sol.assumptions.by_severity()[Severity.INFO]
    assert info_results == ()


def test_perfect_separation_raises_with_l2_hint() -> None:
    """Perfectly separable synthetic data raises AssumptionError pointing
    at penalty=mc.l2 (DESIGN.md §11 — separation handling)."""
    from model_crafter.assumptions import AssumptionError

    rng = np.random.default_rng(0)
    n = 60
    x = rng.normal(size=n)
    y = (x > 0).astype(float)  # perfect separation
    df = pd.DataFrame({"x": x, "y": y})
    spec = mc.linear(target="y", features=["x"], loss=mc.logistic)
    with pytest.raises(AssumptionError, match=r"penalty=mc\.l2"):
        mc.solve(spec, df)


# ---------------------------------------------------------------------------
# Tune + bootstrap + performance end-to-end (the §10 north star, lite)
# ---------------------------------------------------------------------------


def test_end_to_end_tune_then_bootstrap_then_performance() -> None:
    """Tune a logistic-ridge λ on a synthetic credit panel, bootstrap CIs,
    bundle a PerformanceReport. End-to-end smoke test."""
    df = _make_credit_panel(n=400, seed=11)
    features = ["income", "age", "tenure"]

    def build_spec(lam: float):
        return mc.linear(
            target="default_12m",
            features=features,
            loss=mc.logistic,
            penalty=mc.l2(lam),
        )

    grid = mc.log_grid(low=1e-3, high=1e1, n=6)
    splitter = mc.expanding_window(
        time_col="origination_dt",
        n_folds=3,
        horizon="60D",
        gap="0D",
        min_train="180D",
    )

    def auc_metric(sol, data, weights=None):
        return float(mc.auc(sol, data, weights=weights))

    with warnings.catch_warnings():
        # Stability warnings are expected on this tiny synthetic panel.
        warnings.simplefilter("ignore")
        tuned = mc.tune(
            spec_fn=build_spec,
            grid=grid,
            data=df,
            splitter=splitter,
            metric=auc_metric,
        )

    assert tuned.solution.converged
    assert np.all(np.isfinite(tuned.solution.coefficients.to_numpy()))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        boot = mc.bootstrap(tuned.solution, data=df, n_boot=25, random_state=42)
    ci = boot.coefficient_ci(level=0.95)
    assert list(ci.columns) == ["lower", "upper"]
    assert (ci["lower"] <= ci["upper"]).all()

    perf = mc.performance(tuned.solution, data=df)
    assert perf.n_obs == len(df)
    repr_str = repr(perf)
    assert "Discrimination" in repr_str
    assert "Calibration" in repr_str


def test_cross_validate_runs_logistic_with_temporal_splitter() -> None:
    """cross_validate(logistic spec, expanding_window splitter) returns
    a CVResult with per-fold metrics (DESIGN.md §3.2 plumbing)."""
    df = _make_credit_panel(n=400, seed=23)
    spec = mc.linear(
        target="default_12m",
        features=["income", "age", "tenure"],
        loss=mc.logistic,
    )
    splitter = mc.expanding_window(
        time_col="origination_dt",
        n_folds=3,
        horizon="60D",
        gap="30D",
        min_train="120D",
    )

    def auc_metric(sol, data, weights=None):
        return float(mc.auc(sol, data, weights=weights))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = mc.cross_validate(spec, df, splitter=splitter, metrics=[auc_metric])

    assert len(cv.solutions) == 3
    summary = cv.summary()
    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == 3
