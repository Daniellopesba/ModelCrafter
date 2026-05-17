"""End-to-end OLS acceptance tests for Task P1.A.

Quoted acceptance criterion (DESIGN.md §8 Phase 1, AGENTS.md Task P1.A):

  1. *Reproduce ESL §3.2 prostate cancer OLS coefficients to 1e-8 against
     R's lm(). Coefficients, SEs, residual standard error, R² all match.*

The classroom-published values in ESL Table 3.2 are rounded to two decimals,
so this file verifies the OLS implementation in two complementary ways:

* against the printed ESL Table 3.2 values to ``atol=5e-3`` (the rounding
  granularity), and
* against the high-precision values produced by ``statsmodels.OLS`` on the
  same prepared design matrix to ``atol=1e-8``.

statsmodels is a declared *test-only* dependency per DESIGN.md §9.10 / CLAUDE.md
rule 3, used here exactly as the standard reference implementation. The
preprocessing follows ``prostate.info.txt`` from Hastie/Tibshirani/Friedman:
predictors are scaled to mean 0 and unit variance over the **full** 97-row
data, and OLS is fit on the 67-row training subset.

The other two P1.A acceptance criteria — rank-deficient design raises
``AssumptionError`` naming offending columns, and ``mc.predict`` returns a
Series aligned with ``new_data.index`` — are exercised by the remaining tests
in this file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from model_crafter.assumptions import AssumptionError
from model_crafter.loss import squared_error
from model_crafter.solve import predict, solve
from model_crafter.spec import linear

# ---------------------------------------------------------------------------
# Fixture: the ESL prostate dataset, prepared as ESL §3.2 / prostate.info.txt
# specifies.
# ---------------------------------------------------------------------------


PROSTATE_PREDICTORS = (
    "lcavol",
    "lweight",
    "age",
    "lbph",
    "svi",
    "lcp",
    "gleason",
    "pgg45",
)


@pytest.fixture(scope="module")
def prostate() -> pd.DataFrame:
    """Return the ESL prostate dataset with predictors standardized on all 97 rows."""
    path = Path(__file__).parent / "data" / "prostate.csv"
    df = pd.read_csv(path)
    # ESL preprocessing (prostate.info.txt): scale(x, TRUE, TRUE) over all 97 rows.
    X = df[list(PROSTATE_PREDICTORS)].astype(float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    df = df.copy()
    df[list(PROSTATE_PREDICTORS)] = (X - mu) / sd
    return df


@pytest.fixture(scope="module")
def prostate_train(prostate: pd.DataFrame) -> pd.DataFrame:
    """The 67-row training subset used for ESL Table 3.2."""
    mask = prostate["train"] == "T"
    train = prostate.loc[mask].reset_index(drop=True)
    return train


@pytest.fixture(scope="module")
def statsmodels_reference(prostate_train: pd.DataFrame):
    """High-precision OLS fit via statsmodels for the 1e-8 cross-check."""
    X = prostate_train[list(PROSTATE_PREDICTORS)].to_numpy(dtype=float)
    y = prostate_train["lpsa"].to_numpy(dtype=float)
    return sm.OLS(y, sm.add_constant(X)).fit()


# ---------------------------------------------------------------------------
# Acceptance: ESL §3.2 printed values match to printed precision
# ---------------------------------------------------------------------------


# ESL Table 3.2 (2nd ed), prostate cancer training fit. Columns: Term,
# Coefficient, Std Error. (Z scores are computable from these and not pinned.)
ESL_TABLE_3_2 = {
    "(Intercept)": (2.46, 0.09),
    "lcavol": (0.68, 0.13),
    "lweight": (0.26, 0.10),
    "age": (-0.14, 0.10),
    "lbph": (0.21, 0.10),
    "svi": (0.31, 0.12),
    "lcp": (-0.29, 0.15),
    "gleason": (-0.02, 0.15),
    "pgg45": (0.27, 0.15),
}


def test_prostate_coefficients_match_esl_table_3_2(prostate_train: pd.DataFrame) -> None:
    """ESL Table 3.2 coefficients and SEs match to the 2-dp printed precision."""
    spec = linear(
        target="lpsa",
        features=list(PROSTATE_PREDICTORS),
        loss=squared_error,
    )
    sol = solve(spec, data=prostate_train)
    assert sol.converged
    assert sol.coefficient_se is not None

    # Intercept column is named "(Intercept)" by convention.
    coefs = sol.coefficients
    ses = sol.coefficient_se
    for name, (b_expected, se_expected) in ESL_TABLE_3_2.items():
        assert name in coefs.index, f"missing coefficient: {name}"
        assert coefs[name] == pytest.approx(b_expected, abs=5e-3), (
            f"{name}: got {coefs[name]:.6f}, expected ESL {b_expected:.2f}"
        )
        assert ses[name] == pytest.approx(se_expected, abs=5e-3), (
            f"{name} SE: got {ses[name]:.6f}, expected ESL {se_expected:.2f}"
        )


# ---------------------------------------------------------------------------
# Acceptance: 1e-8 match against statsmodels (R's lm() proxy)
# ---------------------------------------------------------------------------


def test_prostate_coefficients_match_statsmodels_to_1e_8(
    prostate_train: pd.DataFrame, statsmodels_reference
) -> None:
    """Coefficients match statsmodels.OLS to absolute tolerance 1e-8."""
    spec = linear(
        target="lpsa",
        features=list(PROSTATE_PREDICTORS),
        loss=squared_error,
    )
    sol = solve(spec, data=prostate_train)

    expected_names = ["(Intercept)", *PROSTATE_PREDICTORS]
    expected_coefs = np.asarray(statsmodels_reference.params, dtype=float)
    got_coefs = np.asarray(sol.coefficients.reindex(expected_names), dtype=float)
    np.testing.assert_allclose(got_coefs, expected_coefs, atol=1e-8, rtol=0)


def test_prostate_standard_errors_match_statsmodels_to_1e_8(
    prostate_train: pd.DataFrame, statsmodels_reference
) -> None:
    """Coefficient SEs match statsmodels.OLS to absolute tolerance 1e-8."""
    spec = linear(
        target="lpsa",
        features=list(PROSTATE_PREDICTORS),
        loss=squared_error,
    )
    sol = solve(spec, data=prostate_train)

    assert sol.coefficient_se is not None  # OLS always produces SEs
    expected_names = ["(Intercept)", *PROSTATE_PREDICTORS]
    expected_se = np.asarray(statsmodels_reference.bse, dtype=float)
    got_se = np.asarray(sol.coefficient_se.reindex(expected_names), dtype=float)
    np.testing.assert_allclose(got_se, expected_se, atol=1e-8, rtol=0)


def test_prostate_residual_standard_error_matches_statsmodels_to_1e_8(
    prostate_train: pd.DataFrame, statsmodels_reference
) -> None:
    r"""Residual standard error :math:`\hat\sigma = \sqrt{\mathrm{RSS}/(n-p)}` matches."""
    spec = linear(
        target="lpsa",
        features=list(PROSTATE_PREDICTORS),
        loss=squared_error,
    )
    sol = solve(spec, data=prostate_train)
    expected_rse = float(np.sqrt(statsmodels_reference.scale))
    got_rse = float(sol.solver_info["residual_std_error"])
    assert got_rse == pytest.approx(expected_rse, abs=1e-8)


def test_prostate_rsquared_matches_statsmodels_to_1e_8(
    prostate_train: pd.DataFrame, statsmodels_reference
) -> None:
    """R^2 matches statsmodels.OLS."""
    spec = linear(
        target="lpsa",
        features=list(PROSTATE_PREDICTORS),
        loss=squared_error,
    )
    sol = solve(spec, data=prostate_train)
    expected_r2 = float(statsmodels_reference.rsquared)
    got_r2 = float(sol.solver_info["r_squared"])
    assert got_r2 == pytest.approx(expected_r2, abs=1e-8)


# ---------------------------------------------------------------------------
# Acceptance: rank-deficient design raises AssumptionError naming columns
# ---------------------------------------------------------------------------


def test_rank_deficient_design_raises_assumption_error() -> None:
    """A linearly dependent column triggers FullRankDesign and names the column."""
    rng = np.random.default_rng(0)
    n = 50
    df = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )
    # x3 is an exact linear combination of x1 and x2 — rank deficient.
    df["x3"] = 2.0 * df["x1"] - 3.0 * df["x2"]

    spec = linear(target="y", features=["x1", "x2", "x3"], loss=squared_error)
    with pytest.raises(AssumptionError) as excinfo:
        solve(spec, data=df)
    assert "x3" in str(excinfo.value) or "x1" in str(excinfo.value) or "x2" in str(excinfo.value), (
        f"AssumptionError message must name at least one offending column; got: {excinfo.value!s}"
    )


def test_rank_deficient_due_to_intercept_collinearity_raises() -> None:
    """A constant non-intercept column is collinear with the intercept and raises."""
    df = pd.DataFrame(
        {
            "y": np.linspace(0, 1, 20),
            "x1": np.linspace(-1, 1, 20),
            "const_col": np.ones(20),  # collinear with intercept
        }
    )
    spec = linear(target="y", features=["x1", "const_col"], loss=squared_error)
    with pytest.raises(AssumptionError) as excinfo:
        solve(spec, data=df, on_violation="raise")
    assert "const_col" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Acceptance: mc.predict returns a Series aligned with new_data.index
# ---------------------------------------------------------------------------


def test_predict_returns_pd_series_aligned_to_new_data_index() -> None:
    """predict's output index equals new_data.index, including non-default indices."""
    rng = np.random.default_rng(42)
    n_train = 100
    train = pd.DataFrame(
        {
            "y": rng.normal(size=n_train),
            "x1": rng.normal(size=n_train),
            "x2": rng.normal(size=n_train),
        }
    )
    spec = linear(target="y", features=["x1", "x2"], loss=squared_error)
    sol = solve(spec, data=train)

    new_data = pd.DataFrame(
        {"x1": [0.1, 0.2, 0.3], "x2": [1.0, -1.0, 0.0]},
        index=pd.Index(["a", "b", "c"], name="loan_id"),
    )
    p = predict(sol, new_data)
    assert isinstance(p, pd.Series)
    assert list(p.index) == ["a", "b", "c"]
    assert p.index.name == "loan_id"
    assert len(p) == 3


def test_predict_matches_manual_eta_computation() -> None:
    """y_hat for OLS equals beta0 + sum_j x_j * beta_j numerically."""
    rng = np.random.default_rng(123)
    n = 200
    df = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )
    spec = linear(target="y", features=["x1", "x2"], loss=squared_error)
    sol = solve(spec, data=df)

    new_data = pd.DataFrame({"x1": [1.0, 2.0], "x2": [-1.0, 3.0]})
    p = predict(sol, new_data)
    b0 = float(sol.coefficients["(Intercept)"])
    b1 = float(sol.coefficients["x1"])
    b2 = float(sol.coefficients["x2"])
    expected = b0 + b1 * np.asarray(new_data["x1"]) + b2 * np.asarray(new_data["x2"])
    np.testing.assert_allclose(np.asarray(p), expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Weighted OLS smoke test (DESIGN.md §9.6: weights everywhere)
# ---------------------------------------------------------------------------


def test_weighted_ols_matches_statsmodels_wls() -> None:
    """Passing weights to solve() reproduces statsmodels.WLS to 1e-10."""
    rng = np.random.default_rng(7)
    n = 200
    x = rng.normal(size=n)
    y = 1.0 + 2.0 * x + rng.normal(size=n)
    w = rng.uniform(0.1, 2.0, size=n)
    df = pd.DataFrame({"y": y, "x": x, "w": w})

    spec = linear(target="y", features=["x"], loss=squared_error)
    sol = solve(spec, data=df, weights="w")

    ref = sm.WLS(y, sm.add_constant(x), weights=w).fit()  # pyright: ignore[reportArgumentType]
    expected = np.asarray(ref.params, dtype=float)
    got = np.asarray(sol.coefficients.reindex(["(Intercept)", "x"]), dtype=float)
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_weighted_ols_accepts_array_weights() -> None:
    """Weights can be supplied as a numpy array directly."""
    rng = np.random.default_rng(8)
    n = 50
    df = pd.DataFrame({"y": rng.normal(size=n), "x": rng.normal(size=n)})
    w = rng.uniform(0.5, 1.5, size=n)
    spec = linear(target="y", features=["x"], loss=squared_error)
    sol_arr = solve(spec, data=df, weights=w)
    sol_str = solve(spec, data=df.assign(w=w), weights="w")
    np.testing.assert_allclose(
        np.asarray(sol_arr.coefficients), np.asarray(sol_str.coefficients), atol=1e-14
    )


# ---------------------------------------------------------------------------
# Assumption integration: solution carries a passing FullRankDesign result
# ---------------------------------------------------------------------------


def test_solution_assumptions_contain_full_rank_design_pass(
    prostate_train: pd.DataFrame,
) -> None:
    """After solving, sol.assumptions includes a passing FullRankDesign result."""
    spec = linear(
        target="lpsa", features=list(PROSTATE_PREDICTORS), loss=squared_error
    )
    sol = solve(spec, data=prostate_train)
    names = [r.name for r in sol.assumptions.results]
    assert "FullRankDesign" in names
    fr = next(r for r in sol.assumptions.results if r.name == "FullRankDesign")
    assert fr.passed is True


# ---------------------------------------------------------------------------
# Predict requires all feature columns; missing column raises naming the col
# ---------------------------------------------------------------------------


def test_predict_missing_feature_column_names_column() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {"y": rng.normal(size=20), "x1": rng.normal(size=20), "x2": rng.normal(size=20)}
    )
    spec = linear(target="y", features=["x1", "x2"], loss=squared_error)
    sol = solve(spec, data=df)
    with pytest.raises(KeyError, match="x2"):
        predict(sol, pd.DataFrame({"x1": [0.0]}))
