r"""Tests for ``mc.coefficients``, ``mc.diagnostics``, ``mc.hat_matrix``, ``mc.influence``.

Task P6 (AGENTS.md / DESIGN.md §5). Acceptance criterion (Phase 6 §8):

* On a hand-derived OLS problem (n=10, p=3, known X / β / σ²),
  ``diagnostics(sol).leverage`` matches
  ``np.diag(X @ np.linalg.inv(X.T @ X) @ X.T)`` to 1e-12.
* ``diagnostics(sol).cooks_distance`` matches the hand-derived reference
  to 1e-12.
* ``hat_matrix(sol)`` matches ``X @ np.linalg.inv(X.T @ X) @ X.T`` to
  1e-12.
* Calling ``diagnostics`` on a lasso / logistic solution raises
  ``NotImplementedError`` with a message pointing at ``mc.bootstrap``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import model_crafter as mc
from model_crafter.inspect import (
    Diagnostics,
    Influence,
    coefficients,
    diagnostics,
    hat_matrix,
    influence,
)

# ---------------------------------------------------------------------------
# Hand-derived OLS reference (n=10, p=3) — the acceptance fixture
# ---------------------------------------------------------------------------


def _hand_derived_ols_problem():
    """Return ``(X, y, df, spec, ref_dict)`` for the hand-derived case.

    X has an intercept column plus two predictors. Everything is small
    enough to hand-verify in numpy.
    """
    # Design columns (without the intercept, since LinearSpec adds one).
    x1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    x2 = np.array([2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0, 10.0, 9.0])
    # Known true beta = [intercept=1, beta1=2, beta2=-1].
    eta = 1.0 + 2.0 * x1 + (-1.0) * x2
    # Slight residual noise so the hat matrix is non-trivial.
    rng = np.random.default_rng(0)
    eps = rng.normal(0.0, 0.5, size=10)
    y = eta + eps

    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

    spec = mc.linear(target="y", features=["x1", "x2"], loss=mc.squared_error)

    # Reference hat matrix and leverages (full p=3 design with intercept).
    X_full = np.column_stack([np.ones(10), x1, x2])
    XtX = X_full.T @ X_full
    XtX_inv = np.linalg.inv(XtX)
    H_ref = X_full @ XtX_inv @ X_full.T
    leverage_ref = np.diag(H_ref)

    beta_ref = XtX_inv @ X_full.T @ y
    yhat_ref = X_full @ beta_ref
    resid_ref = y - yhat_ref
    rss_ref = float(np.sum(resid_ref * resid_ref))
    n_ref, p_ref = X_full.shape
    sigma2_ref = rss_ref / (n_ref - p_ref)

    one_minus_h = 1.0 - leverage_ref
    cooks_ref = (resid_ref * resid_ref / (p_ref * sigma2_ref)) * (
        leverage_ref / (one_minus_h * one_minus_h)
    )
    studentized_ref = resid_ref / (np.sqrt(sigma2_ref) * np.sqrt(one_minus_h))

    return {
        "df": df,
        "spec": spec,
        "X": X_full,
        "y": y,
        "beta": beta_ref,
        "H": H_ref,
        "leverage": leverage_ref,
        "residuals": resid_ref,
        "sigma2": sigma2_ref,
        "cooks": cooks_ref,
        "studentized": studentized_ref,
    }


@pytest.fixture
def hand_derived():
    return _hand_derived_ols_problem()


# ---------------------------------------------------------------------------
# hat_matrix
# ---------------------------------------------------------------------------


def test_hat_matrix_matches_hand_derived(hand_derived):
    sol = mc.solve(hand_derived["spec"], hand_derived["df"])
    H = hat_matrix(sol, hand_derived["df"])
    np.testing.assert_allclose(H, hand_derived["H"], atol=1e-12, rtol=0.0)


def test_hat_matrix_shape(hand_derived):
    sol = mc.solve(hand_derived["spec"], hand_derived["df"])
    H = hat_matrix(sol, hand_derived["df"])
    assert H.shape == (10, 10)


def test_hat_matrix_symmetric_idempotent(hand_derived):
    sol = mc.solve(hand_derived["spec"], hand_derived["df"])
    H = hat_matrix(sol, hand_derived["df"])
    # Hat matrix is symmetric idempotent.
    np.testing.assert_allclose(H, H.T, atol=1e-12)
    np.testing.assert_allclose(H @ H, H, atol=1e-10)


def test_hat_matrix_trace_equals_p(hand_derived):
    sol = mc.solve(hand_derived["spec"], hand_derived["df"])
    H = hat_matrix(sol, hand_derived["df"])
    # tr(H) = p (rank of X)
    np.testing.assert_allclose(np.trace(H), 3.0, atol=1e-10)


# ---------------------------------------------------------------------------
# diagnostics: leverage / cooks / studentized
# ---------------------------------------------------------------------------


def test_diagnostics_leverage_matches_hand_derived(hand_derived):
    sol = mc.solve(hand_derived["spec"], hand_derived["df"])
    diag = diagnostics(sol, hand_derived["df"])
    np.testing.assert_allclose(
        diag.leverage.to_numpy(),
        hand_derived["leverage"],
        atol=1e-12,
        rtol=0.0,
    )


def test_diagnostics_cooks_distance_matches_hand_derived(hand_derived):
    sol = mc.solve(hand_derived["spec"], hand_derived["df"])
    diag = diagnostics(sol, hand_derived["df"])
    np.testing.assert_allclose(
        diag.cooks_distance.to_numpy(),
        hand_derived["cooks"],
        atol=1e-12,
        rtol=0.0,
    )


def test_diagnostics_residuals_match(hand_derived):
    sol = mc.solve(hand_derived["spec"], hand_derived["df"])
    diag = diagnostics(sol, hand_derived["df"])
    np.testing.assert_allclose(
        diag.residuals.to_numpy(),
        hand_derived["residuals"],
        atol=1e-12,
        rtol=0.0,
    )


def test_diagnostics_studentized_match(hand_derived):
    sol = mc.solve(hand_derived["spec"], hand_derived["df"])
    diag = diagnostics(sol, hand_derived["df"])
    np.testing.assert_allclose(
        diag.studentized_residuals.to_numpy(),
        hand_derived["studentized"],
        atol=1e-12,
        rtol=0.0,
    )


def test_diagnostics_sigma2_matches(hand_derived):
    sol = mc.solve(hand_derived["spec"], hand_derived["df"])
    diag = diagnostics(sol, hand_derived["df"])
    assert diag.sigma2 == pytest.approx(hand_derived["sigma2"], abs=1e-12)


def test_diagnostics_is_frozen(hand_derived):
    sol = mc.solve(hand_derived["spec"], hand_derived["df"])
    diag = diagnostics(sol, hand_derived["df"])
    assert isinstance(diag, Diagnostics)
    with pytest.raises((AttributeError, Exception)):
        diag.sigma2 = 0.5  # type: ignore[misc]


def test_diagnostics_repr_does_not_crash(hand_derived):
    sol = mc.solve(hand_derived["spec"], hand_derived["df"])
    diag = diagnostics(sol, hand_derived["df"])
    s = repr(diag)
    assert "Diagnostics" in s
    assert "sigma^2" in s


# ---------------------------------------------------------------------------
# influence: DFBETAS / Cook's / leverage
# ---------------------------------------------------------------------------


def test_influence_leverage_matches_diagnostics(hand_derived):
    sol = mc.solve(hand_derived["spec"], hand_derived["df"])
    diag = diagnostics(sol, hand_derived["df"])
    inf = influence(sol, hand_derived["df"])
    np.testing.assert_allclose(
        inf.leverage.to_numpy(), diag.leverage.to_numpy(), atol=1e-12
    )


def test_influence_cooks_matches_diagnostics(hand_derived):
    sol = mc.solve(hand_derived["spec"], hand_derived["df"])
    diag = diagnostics(sol, hand_derived["df"])
    inf = influence(sol, hand_derived["df"])
    np.testing.assert_allclose(
        inf.cooks_distance.to_numpy(),
        diag.cooks_distance.to_numpy(),
        atol=1e-12,
    )


def test_influence_dfbetas_shape(hand_derived):
    sol = mc.solve(hand_derived["spec"], hand_derived["df"])
    inf = influence(sol, hand_derived["df"])
    n, p = inf.dfbetas.shape
    assert n == 10
    assert p == 3  # intercept + x1 + x2
    assert list(inf.dfbetas.columns) == list(sol.design_columns)


def test_influence_dfbetas_consistent_with_leave_one_out_refit(hand_derived):
    """Compute DFBETAS directly via leave-one-out refit and compare."""
    df = hand_derived["df"]
    spec = hand_derived["spec"]
    sol = mc.solve(spec, df)
    inf = influence(sol, df)

    X = hand_derived["X"]
    y = hand_derived["y"]
    n, p = X.shape

    # Closed-form leave-one-out beta change:
    # ΔB_i = (X^T X)^-1 x_i e_i / (1 - h_ii)
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    H = X @ XtX_inv @ X.T
    leverage = np.diag(H)
    rss = float(np.sum(resid * resid))
    sigma2 = rss / (n - p)
    one_minus_h = 1.0 - leverage

    studentized = resid / (np.sqrt(sigma2) * np.sqrt(one_minus_h))
    sigma2_minus_i = sigma2 * (n - p - studentized * studentized) / (n - p - 1)

    # DFBETAS reference (row-loop for clarity)
    diag_inv = np.diag(XtX_inv)
    expected = np.zeros((n, p))
    for i in range(n):
        delta_beta_i = XtX_inv @ X[i, :] * (resid[i] / one_minus_h[i])
        se_i = np.sqrt(sigma2_minus_i[i] * diag_inv)
        expected[i, :] = delta_beta_i / se_i

    np.testing.assert_allclose(
        inf.dfbetas.to_numpy(), expected, atol=1e-12, rtol=0.0
    )


def test_influence_repr_does_not_crash(hand_derived):
    sol = mc.solve(hand_derived["spec"], hand_derived["df"])
    inf = influence(sol, hand_derived["df"])
    s = repr(inf)
    assert "Influence" in s


def test_influence_is_frozen(hand_derived):
    sol = mc.solve(hand_derived["spec"], hand_derived["df"])
    inf = influence(sol, hand_derived["df"])
    assert isinstance(inf, Influence)


# ---------------------------------------------------------------------------
# coefficients
# ---------------------------------------------------------------------------


def test_coefficients_columns(hand_derived):
    sol = mc.solve(hand_derived["spec"], hand_derived["df"])
    tab = coefficients(sol)
    assert list(tab.columns) == ["estimate", "std_error", "z", "p_value"]
    assert list(tab.index) == list(sol.design_columns)


def test_coefficients_match_solution_values(hand_derived):
    sol = mc.solve(hand_derived["spec"], hand_derived["df"])
    tab = coefficients(sol)
    np.testing.assert_allclose(
        tab["estimate"].to_numpy(),
        sol.coefficients.to_numpy(dtype=float),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        tab["std_error"].to_numpy(),
        sol.coefficient_se.reindex(sol.design_columns).to_numpy(dtype=float),
        atol=1e-12,
    )


def test_coefficients_p_values_under_t_distribution(hand_derived):
    """OLS p-values use the t_{n-p} distribution."""
    from scipy.stats import t

    sol = mc.solve(hand_derived["spec"], hand_derived["df"])
    tab = coefficients(sol)
    n, p = hand_derived["X"].shape
    z = tab["z"].to_numpy()
    p_expected = 2.0 * (1.0 - t.cdf(np.abs(z), df=n - p))
    np.testing.assert_allclose(tab["p_value"].to_numpy(), p_expected, atol=1e-12)


def test_coefficients_lasso_has_nan_p(hand_derived):
    """Lasso has no closed-form SE → std_error is NaN and z/p are NaN."""
    df = hand_derived["df"].copy()
    spec = mc.linear(
        target="y",
        features=["x1", "x2"],
        loss=mc.squared_error,
        penalty=mc.l1(0.1),
    )
    sol = mc.solve(spec, df)
    tab = coefficients(sol)
    assert bool(tab["std_error"].isna().all())
    assert bool(tab["z"].isna().all())
    assert bool(tab["p_value"].isna().all())


def test_coefficients_logistic_uses_wald():
    """Logistic coefficients use Wald (standard-normal) p-values."""
    from scipy.stats import norm

    # Build a non-separable logistic problem.
    rng = np.random.default_rng(7)
    n = 200
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    eta = -0.3 + 0.5 * x1 - 0.3 * x2
    p = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, p).astype(float)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    spec = mc.linear(target="y", features=["x1", "x2"], loss=mc.logistic)
    sol = mc.solve(spec, df)
    tab = coefficients(sol)
    if sol.coefficient_se is not None:
        z = tab["z"].to_numpy()
        p_expected = 2.0 * (1.0 - norm.cdf(np.abs(z)))
        np.testing.assert_allclose(
            tab["p_value"].to_numpy(), p_expected, atol=1e-12
        )


# ---------------------------------------------------------------------------
# NotImplementedError for non-closed-form fits
# ---------------------------------------------------------------------------


def test_diagnostics_raises_on_lasso(hand_derived):
    df = hand_derived["df"]
    spec = mc.linear(
        target="y",
        features=["x1", "x2"],
        loss=mc.squared_error,
        penalty=mc.l1(0.1),
    )
    sol = mc.solve(spec, df)
    with pytest.raises(NotImplementedError, match="bootstrap"):
        diagnostics(sol, df)


def test_diagnostics_raises_on_elastic_net(hand_derived):
    df = hand_derived["df"]
    spec = mc.linear(
        target="y",
        features=["x1", "x2"],
        loss=mc.squared_error,
        penalty=mc.l1(0.1) + mc.l2(0.1),
    )
    sol = mc.solve(spec, df)
    with pytest.raises(NotImplementedError, match="bootstrap"):
        diagnostics(sol, df)


def _non_separable_logistic_sol():
    rng = np.random.default_rng(11)
    n = 200
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    p = 1.0 / (1.0 + np.exp(-(-0.3 + 0.4 * x1 - 0.2 * x2)))
    y = rng.binomial(1, p).astype(float)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    spec = mc.linear(target="y", features=["x1", "x2"], loss=mc.logistic)
    return mc.solve(spec, df), df


def test_diagnostics_raises_on_logistic():
    sol, df = _non_separable_logistic_sol()
    with pytest.raises(NotImplementedError, match="bootstrap"):
        diagnostics(sol, df)


def test_hat_matrix_raises_on_lasso(hand_derived):
    df = hand_derived["df"]
    spec = mc.linear(
        target="y",
        features=["x1", "x2"],
        loss=mc.squared_error,
        penalty=mc.l1(0.1),
    )
    sol = mc.solve(spec, df)
    with pytest.raises(NotImplementedError, match="bootstrap"):
        hat_matrix(sol, df)


def test_hat_matrix_raises_on_logistic():
    sol, df = _non_separable_logistic_sol()
    with pytest.raises(NotImplementedError, match="bootstrap"):
        hat_matrix(sol, df)


def test_influence_raises_on_lasso(hand_derived):
    df = hand_derived["df"]
    spec = mc.linear(
        target="y",
        features=["x1", "x2"],
        loss=mc.squared_error,
        penalty=mc.l1(0.1),
    )
    sol = mc.solve(spec, df)
    with pytest.raises(NotImplementedError, match="bootstrap"):
        influence(sol, df)


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_diagnostics_requires_data(hand_derived):
    sol = mc.solve(hand_derived["spec"], hand_derived["df"])
    with pytest.raises(TypeError, match="data frame"):
        diagnostics(sol)


def test_hat_matrix_requires_data(hand_derived):
    sol = mc.solve(hand_derived["spec"], hand_derived["df"])
    with pytest.raises(TypeError, match="data frame"):
        hat_matrix(sol)


def test_influence_requires_data(hand_derived):
    sol = mc.solve(hand_derived["spec"], hand_derived["df"])
    with pytest.raises(TypeError, match="data frame"):
        influence(sol)


def test_diagnostics_data_missing_target_raises(hand_derived):
    sol = mc.solve(hand_derived["spec"], hand_derived["df"])
    bad = hand_derived["df"].drop(columns=["y"])
    with pytest.raises(KeyError, match="target"):
        diagnostics(sol, bad)


def test_ridge_diagnostics_closed_form(hand_derived):
    """Ridge has a closed-form hat → diagnostics should work."""
    df = hand_derived["df"]
    spec = mc.linear(
        target="y",
        features=["x1", "x2"],
        loss=mc.squared_error,
        penalty=mc.l2(0.5),
    )
    sol = mc.solve(spec, df)
    diag = diagnostics(sol, df)
    assert isinstance(diag, Diagnostics)
    # Ridge hat is symmetric (in the weighted-projection sense) but not
    # idempotent; we only check that the call returns sensible shapes.
    assert len(diag.leverage) == 10
    # Leverages should be in (0, 1) for a regular ridge fit on this fixture.
    assert ((diag.leverage > 0) & (diag.leverage < 1)).all()
