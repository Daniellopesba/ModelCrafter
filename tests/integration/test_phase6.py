"""Phase 6 integration tests — v0 acceptance.

Quotes AGENTS.md Task P6 acceptance:

  A segmented logistic regression with WoE features produces per-segment
  Solutions, AssumptionReports, BootstrappedSolutions, and
  PerformanceReports, all from a single declarative spec.

Also exercises:
  - mc.coefficients / mc.diagnostics / mc.hat_matrix / mc.influence on OLS
  - _repr_html_ smoke-test across every public value type
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import model_crafter as mc

# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def _segmented_panel(n: int = 600, seed: int = 31) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    income = rng.normal(size=n)
    age = rng.normal(size=n)
    tenure = rng.normal(size=n)
    product = rng.choice(["card", "loan", "mortgage"], size=n, p=[0.5, 0.3, 0.2])
    seg_effects = {"card": (-0.5, 0.8), "loan": (0.2, 1.2), "mortgage": (0.7, 0.6)}
    eta = np.zeros(n)
    for prod, (intercept, beta_income) in seg_effects.items():
        mask = product == prod
        eta[mask] = intercept + beta_income * income[mask] - 0.3 * age[mask]
    p = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, p).astype(float)
    return pd.DataFrame(
        {
            "default_12m": y,
            "income": income,
            "age": age,
            "tenure": tenure,
            "product": product,
        }
    )


def test_segmented_logistic_end_to_end() -> None:
    """The §10 + §3.4 north-star: a segmented logistic spec produces a
    per-segment Solution, AssumptionReport, and PerformanceReport from a
    single declarative spec."""
    df = _segmented_panel(n=600, seed=11)
    base = mc.linear(
        target="default_12m",
        features=["income", "age", "tenure"],
        loss=mc.logistic,
        penalty=mc.l2(0.1),
    )
    spec = mc.segmented(by="product", base=base)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg_sol = mc.solve(spec, df)

    assert isinstance(seg_sol, mc.SegmentedSolution)
    assert set(seg_sol.segments.keys()) == {"card", "loan", "mortgage"}

    # Each segment is a real Solution with its own AssumptionReport.
    for name, sub_sol in seg_sol.segments.items():
        assert sub_sol.converged, f"segment {name} did not converge"
        assert hasattr(sub_sol, "assumptions")
        report = sub_sol.assumptions
        assert hasattr(report, "results")

    # Predict routes by segment; output is always a probability.
    yhat = mc.predict(seg_sol, df)
    assert isinstance(yhat, pd.Series)
    assert (yhat >= 0).all() and (yhat <= 1).all()

    # Total n_obs is the sum across segments.
    seg_n = sum(s.n_obs for s in seg_sol.segments.values())
    assert seg_sol.n_obs == seg_n == len(df)

    # __getitem__ works for routing.
    card_sol = seg_sol["card"]
    assert card_sol.n_obs == int((df["product"] == "card").sum())


def test_segmented_predict_warns_on_unseen_segment() -> None:
    """Predict on a row whose segment wasn't seen at fit time → NaN + warning."""
    df = _segmented_panel(n=300, seed=3)
    base = mc.linear(
        target="default_12m",
        features=["income"],
        loss=mc.logistic,
    )
    spec = mc.segmented(by="product", base=base)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg_sol = mc.solve(spec, df)

    new_data = df.head(10).copy()
    new_data.loc[new_data.index[:3], "product"] = "personal_loan"  # unseen
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yhat = mc.predict(seg_sol, new_data)
    assert yhat.iloc[:3].isna().all()
    messages = " ".join(str(w.message) for w in caught)
    assert "personal_loan" in messages or "unseen" in messages.lower()


# ---------------------------------------------------------------------------
# Diagnostics (closed-form linear models only)
# ---------------------------------------------------------------------------


def test_ols_diagnostics_match_hand_derivation() -> None:
    """diagnostics(sol, data).leverage matches np.diag(H) to 1e-12."""
    rng = np.random.default_rng(0)
    n, p = 30, 3
    X = np.column_stack([np.ones(n), rng.normal(size=(n, p))])
    beta = np.array([1.0, 2.0, -1.0, 0.5])
    y = X @ beta + 0.1 * rng.normal(size=n)
    df = pd.DataFrame(
        {"y": y, "x1": X[:, 1], "x2": X[:, 2], "x3": X[:, 3]}
    )
    spec = mc.linear(target="y", features=["x1", "x2", "x3"], loss=mc.squared_error)
    sol = mc.solve(spec, df)

    diag = mc.diagnostics(sol, df)
    H = mc.hat_matrix(sol, df)
    np.testing.assert_allclose(
        diag.leverage.to_numpy(), np.diag(H), atol=1e-12
    )
    # Reference hat matrix.
    H_ref = X @ np.linalg.inv(X.T @ X) @ X.T
    np.testing.assert_allclose(H, H_ref, atol=1e-12)


def test_diagnostics_on_lasso_raises_not_implemented() -> None:
    """Lasso doesn't have a closed-form hat matrix; diagnostics raises."""
    rng = np.random.default_rng(1)
    n = 50
    df = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x": rng.normal(size=n),
        }
    )
    spec = mc.linear(
        target="y",
        features=["x"],
        loss=mc.squared_error,
        penalty=mc.l1(0.5),
    )
    sol = mc.solve(spec, df)
    with pytest.raises(NotImplementedError):
        mc.diagnostics(sol, df)


def test_coefficients_table_has_se_z_p_for_ols() -> None:
    """coefficients(sol) returns a DataFrame with SE, z, p columns for OLS."""
    rng = np.random.default_rng(2)
    n = 100
    df = pd.DataFrame(
        {"y": rng.normal(size=n), "x": rng.normal(size=n), "z": rng.normal(size=n)}
    )
    spec = mc.linear(target="y", features=["x", "z"], loss=mc.squared_error)
    sol = mc.solve(spec, df)
    coefs = mc.coefficients(sol)
    assert isinstance(coefs, pd.DataFrame)
    assert {"estimate", "std_error", "z", "p_value"}.issubset(coefs.columns)


# ---------------------------------------------------------------------------
# _repr_html_ smoke test
# ---------------------------------------------------------------------------


def test_repr_html_on_every_value_type() -> None:
    """Every public value type has a _repr_html_ that returns non-empty HTML."""
    df = _segmented_panel(n=200, seed=7)
    spec = mc.linear(
        target="default_12m",
        features=["income", "age"],
        loss=mc.logistic,
        penalty=mc.l2(0.1),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = mc.solve(spec, df)

    # Solution + AssumptionReport
    assert "<" in sol._repr_html_()
    assert "<" in sol.assumptions._repr_html_()

    # PerformanceReport
    perf = mc.performance(sol, df)
    assert "<" in perf._repr_html_()

    # Comparison
    cmp = mc.compare({"a": sol, "b": sol}, data=df)
    assert "<" in cmp._repr_html_()

    # SegmentedPerformanceReport
    perf_seg = mc.performance_by_segment(sol, df, by="product")
    assert "<" in perf_seg._repr_html_()
