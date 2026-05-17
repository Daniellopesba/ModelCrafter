"""Phase 4 integration tests.

Quotes AGENTS.md Task P4.INTEG acceptance:

  The §10 north-star example in DESIGN.md (the part that fits inside
  Phase 4 — WoE + logistic + walk-forward CV + bootstrap + performance)
  runs end-to-end on a public dataset. Each step is a single function
  call. The model output is always a probability via ``mc.predict(sol,
  data)``.

Also exercises:
  - mc.ns / mc.bs basis terms in a regression spec
  - mc.interact / mc.cross interaction algebra
  - mc.binning_table inspection on a WoE-logistic fit
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import model_crafter as mc


def _credit_panel(n: int = 600, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    income = rng.normal(loc=0.0, scale=1.0, size=n)
    age = rng.normal(loc=0.0, scale=1.0, size=n)
    tenure = rng.normal(loc=0.0, scale=1.0, size=n)
    region = rng.choice(["NE", "SE", "MW", "W"], size=n, p=[0.4, 0.3, 0.2, 0.1])
    region_effect = pd.Series(region).map(
        {"NE": -0.4, "SE": 0.1, "MW": 0.2, "W": 0.5}
    ).to_numpy()
    eta = -1.2 + 0.9 * income - 0.7 * age + 0.3 * tenure + region_effect
    p = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, p).astype(float)
    origination_dt = pd.date_range("2022-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "default_12m": y,
            "income": income,
            "age": age,
            "tenure": tenure,
            "region": region,
            "origination_dt": origination_dt,
        }
    )


def test_ns_term_in_a_squared_error_spec() -> None:
    """A natural cubic spline term participates in mc.linear + mc.solve."""
    rng = np.random.default_rng(0)
    n = 200
    x = rng.uniform(0, 10, size=n)
    y = np.sin(x) + 0.1 * rng.normal(size=n)
    df = pd.DataFrame({"x": x, "y": y})
    spec = mc.linear(target="y", features=mc.ns("x", df=5), loss=mc.squared_error)
    sol = mc.solve(spec, df)
    yhat = mc.predict(sol, df)
    # Spline should fit sin(x) much better than a single linear term.
    rss_spline = float(np.sum((y - yhat.to_numpy()) ** 2))
    linear_spec = mc.linear(target="y", features=["x"], loss=mc.squared_error)
    linear_sol = mc.solve(linear_spec, df)
    rss_linear = float(np.sum((y - mc.predict(linear_sol, df).to_numpy()) ** 2))
    assert rss_spline < rss_linear * 0.5, (
        f"spline RSS={rss_spline:.3f} should be much less than "
        f"linear RSS={rss_linear:.3f}"
    )


def test_interact_in_a_linear_spec() -> None:
    """mc.interact(a, b) and mc.cross(a, b) both produce solvable specs."""
    rng = np.random.default_rng(1)
    n = 200
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    y = 1.0 + 2.0 * a + 3.0 * b + 1.5 * (a * b) + 0.2 * rng.normal(size=n)
    df = pd.DataFrame({"a": a, "b": b, "y": y})

    spec_main_and_cross = mc.linear(
        target="y", features=mc.interact("a", "b"), loss=mc.squared_error
    )
    spec_cross_only = mc.linear(
        target="y", features=["a", "b"] + [mc.cross("a", "b")],
        loss=mc.squared_error,
    )
    sol1 = mc.solve(spec_main_and_cross, df)
    sol2 = mc.solve(spec_cross_only, df)
    # Both specs should produce a finite predictor; "interact" is a superset
    # of "a + b + cross(a, b)" so it should fit at least as well.
    rss1 = float(np.sum((y - mc.predict(sol1, df).to_numpy()) ** 2))
    rss2 = float(np.sum((y - mc.predict(sol2, df).to_numpy()) ** 2))
    assert np.isfinite(rss1) and np.isfinite(rss2)


def test_woe_logistic_walk_forward_bootstrap_performance() -> None:
    """The §10 north-star example: WoE + logistic + walk-forward CV +
    bootstrap + performance. All single function calls."""
    df = _credit_panel(n=600, seed=11)

    # P4.B's stateful WoE terms aren't yet auto-fit by mc.solve (a Phase 5
    # follow-up flagged in notes/P4.B.md). Use mc.fit_binnings to bake the
    # learned bins into the spec before solving.
    from model_crafter.terms.woe import fit_binnings

    features = (
        mc.woe("income", bins=mc.monotonic(min_bin_size=0.05))
        + mc.woe("age", bins=mc.monotonic(min_bin_size=0.05))
        + mc.woe("region", bins=mc.categorical(group_rare=0.01))
    )
    spec = mc.linear(
        target="default_12m",
        features=features,
        loss=mc.logistic,
        penalty=mc.l2(0.1),
    )
    spec_fitted = fit_binnings(spec, df)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = mc.solve(spec_fitted, df)

    # Predict is always a probability.
    yhat = mc.predict(sol, df)
    assert (yhat >= 0).all() and (yhat <= 1).all()

    # Binning table is inspectable.
    bt = mc.binning_table(sol)
    assert hasattr(bt, "tables")
    assert "income" in bt.tables
    assert "region" in bt.tables

    # Walk-forward CV runs.
    splitter = mc.expanding_window(
        time_col="origination_dt",
        n_folds=3,
        horizon="60D",
        gap="30D",
        min_train="180D",
    )

    def auc_metric(sol, data, weights=None):
        return float(mc.auc(sol, data, weights=weights))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = mc.cross_validate(
            spec_fitted, df, splitter=splitter, metrics=[auc_metric]
        )
    assert len(cv.solutions) == 3

    # Bootstrap CIs.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        boot = mc.bootstrap(sol, data=df, n_boot=15, random_state=42)
    ci = boot.coefficient_ci(level=0.95)
    assert list(ci.columns) == ["lower", "upper"]

    # PerformanceReport bundles everything.
    perf = mc.performance(sol, data=df)
    repr_str = repr(perf)
    assert "Discrimination" in repr_str
    assert "Calibration" in repr_str


def test_binned_vs_woe_have_different_parameter_count() -> None:
    """mc.binned produces k-1 columns per feature; mc.woe produces 1 column.
    This is the ESL-honest alternative discussed in DESIGN.md §3.1."""
    from model_crafter.terms.woe import fit_binnings

    df = _credit_panel(n=300, seed=3)
    woe_spec = mc.linear(
        target="default_12m",
        features=mc.woe("income", bins=mc.monotonic(min_bin_size=0.1, max_bins=5)),
        loss=mc.logistic,
        penalty=mc.l2(0.1),
    )
    binned_spec = mc.linear(
        target="default_12m",
        features=mc.binned("income", bins=mc.monotonic(min_bin_size=0.1, max_bins=5)),
        loss=mc.logistic,
        penalty=mc.l2(0.1),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        woe_sol = mc.solve(fit_binnings(woe_spec, df), df)
        binned_sol = mc.solve(fit_binnings(binned_spec, df), df)

    # WoE yields a single coefficient for income (plus intercept).
    # Binned yields k-1 indicator coefficients for income (plus intercept).
    woe_n = len(woe_sol.coefficients)
    binned_n = len(binned_sol.coefficients)
    assert binned_n > woe_n, (
        f"mc.binned should produce more coefs than mc.woe; got "
        f"binned={binned_n}, woe={woe_n}"
    )
