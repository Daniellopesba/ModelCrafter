"""Phase 2 integration tests.

Quotes AGENTS.md Task P2.INTEG acceptance criteria:

  1. All P2.{A,B,C} acceptance criteria pass.
  2. ``mc.linear(..., loss=squared_error, penalty=l1(0.1)+l2(0.1))``
     solves end-to-end.
  3. ``ComparableFeatureScales`` warning fires on unscaled features
     by default.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import model_crafter as mc
from model_crafter.penalty import L1Penalty, L2Penalty, PenaltySum, l1, l2

PROSTATE = Path(__file__).resolve().parent.parent / "data" / "prostate.csv"


def _prostate_train() -> pd.DataFrame:
    df = pd.read_csv(PROSTATE)
    return df.loc[df["train"] == "T"].copy()


def test_dispatch_routes_to_four_distinct_solvers() -> None:
    """NoPenalty → OLS, L2 → ridge, L1 → CD lasso, L1+L2 → CD enet."""
    from model_crafter.loss import squared_error
    from model_crafter.solve._registry import get_solver

    assert get_solver(squared_error, mc.NoPenalty()).__name__ == "solve_ols"
    assert get_solver(squared_error, L2Penalty(lam=0.1)).__name__ == "solve_ridge_closed_form"
    assert get_solver(squared_error, L1Penalty(lam=0.1)).__name__ == "solve_lasso_cd"
    assert get_solver(squared_error, l1(0.1) + l2(0.1)).__name__ == "solve_enet_cd"


def test_elastic_net_end_to_end_on_prostate() -> None:
    """mc.linear with l1(0.1)+l2(0.1) solves end-to-end and returns a
    Solution with finite coefficients."""
    train = _prostate_train()
    features = ["lcavol", "lweight", "age", "lbph", "svi", "lcp", "gleason", "pgg45"]
    spec = mc.linear(
        target="lpsa",
        features=features,
        loss=mc.squared_error,
        penalty=l1(0.1) + l2(0.1),
    )
    sol = mc.solve(spec, train)
    assert sol.converged
    assert np.all(np.isfinite(sol.coefficients.to_numpy()))
    # Predict works on the same DataFrame.
    yhat = mc.predict(sol, train)
    assert isinstance(yhat, pd.Series)
    assert len(yhat) == len(train)
    assert np.all(np.isfinite(yhat.to_numpy()))


def test_ridge_recovers_ols_at_zero_lambda() -> None:
    """ridge with λ=0 matches OLS coefficients to numerical tolerance."""
    train = _prostate_train()
    features = ["lcavol", "lweight", "age", "lbph"]

    ols_spec = mc.linear(target="lpsa", features=features, loss=mc.squared_error)
    ridge_spec = mc.linear(
        target="lpsa", features=features, loss=mc.squared_error, penalty=l2(0.0)
    )
    sol_ols = mc.solve(ols_spec, train)
    sol_ridge = mc.solve(ridge_spec, train)
    np.testing.assert_allclose(
        sol_ridge.coefficients.to_numpy(),
        sol_ols.coefficients.to_numpy(),
        atol=1e-8,
    )


def test_lasso_at_large_lambda_is_sparse() -> None:
    """Lasso with very large λ zeros every coefficient (ESL §3.4.2)."""
    train = _prostate_train()
    features = ["lcavol", "lweight", "age", "lbph"]
    # A λ far above λ_max guarantees the all-zero solution.
    spec = mc.linear(
        target="lpsa",
        features=features,
        loss=mc.squared_error,
        penalty=l1(1e3),
    )
    sol = mc.solve(spec, train)
    non_intercept = sol.coefficients.drop(labels=["(Intercept)"], errors="ignore")
    np.testing.assert_array_equal(non_intercept.to_numpy(), np.zeros(len(non_intercept)))


def test_penalty_sum_flattens() -> None:
    """l1 + l2 + l1 flattens to a 3-part PenaltySum (no nesting)."""
    p = l1(0.1) + l2(0.2) + l1(0.05)
    assert isinstance(p, PenaltySum)
    assert len(list(p)) == 3


def test_penalty_plus_term_raises_with_helpful_message() -> None:
    """Penalty + Term is a programming error pointing at features=/penalty=."""
    from model_crafter.terms.base import RawTerm

    with pytest.raises(TypeError, match="features=|penalty="):
        _ = l1(0.1) + RawTerm(name="income")


def test_comparable_feature_scales_warns_on_unscaled_features() -> None:
    """A spec with std ratio > 100 emits ComparableFeatureScales warning (SOFT)."""
    rng = np.random.default_rng(seed=42)
    n = 200
    df = pd.DataFrame(
        {
            "small": rng.normal(scale=1.0, size=n),
            "huge": rng.normal(scale=1000.0, size=n),  # std ratio ~1000
            "y": rng.normal(size=n),
        }
    )
    spec = mc.linear(
        target="y",
        features=["small", "huge"],
        loss=mc.squared_error,
        penalty=l2(0.1),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mc.solve(spec, df)
    messages = " ".join(str(w.message) for w in caught)
    assert "ComparableFeatureScales" in messages, (
        f"Expected ComparableFeatureScales warning, got: {messages!r}"
    )
