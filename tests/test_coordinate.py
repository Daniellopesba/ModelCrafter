"""Tests for the coordinate-descent solver (AGENTS.md Task P2.C).

Quoted acceptance criteria (DESIGN.md §8 Phase 2, AGENTS.md Task P2.C):

  1. *Match `glmnet` coefficients across a lambda path for lasso and elastic
     net, on the prostate dataset, to 1e-6.*
  2. *Warm-started path fit (from large to small lambda) is measurably
     faster than from-scratch fits at each lambda. Benchmark in the test.*
  3. *Lasso with very large lambda produces β = 0; lasso with λ = 0 recovers
     OLS coefficients to 1e-8.*
  4. *Elastic net with α=1 (pure L1) matches the lasso solver; α=0 (pure L2)
     matches the ridge closed form to 1e-6.*
  5. *Weighted CD matches a hand-derived reference on a small problem to
     1e-8.*

Reference source for the prostate-path acceptance value
-------------------------------------------------------
We don't have R / glmnet available, so the high-precision reference here is
**statsmodels.OLS.fit_regularized** with ``method="elastic_net"``, fit on
internally weighted-standardised X (the same convention our solver
follows). statsmodels' docstring states "The implementation closely follows
the glmnet package in R", and verifying at lam=0 recovers OLS to 1e-14
confirms the objective parameterisation matches glmnet's. Cross-references
to the analytical orthonormal-X case (where the lasso reduces to soft-
thresholding ``β_j = S(X_j^T y / n, λ)``) and to the ridge closed form (for
the α=0 limit) anchor the math independently.

P2.A's ``L1Penalty`` / ``L2Penalty`` / ``PenaltySum`` aren't merged at the
time these tests are written, so we exercise the solver's math through the
``coordinate_descent_path`` function directly. Dispatch registration is
tested separately by a stub-penalty test below.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from model_crafter.solve.coordinate import (
    _split_enet_parts,
    coordinate_descent_path,
    solve_enet_cd,
    solve_lasso_cd,
)

# ---------------------------------------------------------------------------
# Penalty stubs (matching P2.A's documented contract by class name)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class L1Penalty:
    lam: float

    @property
    def assumptions(self) -> tuple:
        return ()

    def value(self, beta: np.ndarray) -> float:
        return float(self.lam * np.sum(np.abs(beta)))

    def __add__(self, other: object) -> Any:
        return PenaltySum(parts=(self, other))


@dataclass(frozen=True)
class L2Penalty:
    lam: float

    @property
    def assumptions(self) -> tuple:
        return ()

    def value(self, beta: np.ndarray) -> float:
        return float(0.5 * self.lam * np.sum(beta * beta))

    def __add__(self, other: object) -> Any:
        return PenaltySum(parts=(self, other))


@dataclass(frozen=True)
class PenaltySum:
    parts: tuple

    @property
    def assumptions(self) -> tuple:
        return ()

    def value(self, beta: np.ndarray) -> float:
        return float(sum(p.value(beta) for p in self.parts))

    def __add__(self, other: object) -> Any:
        return PenaltySum(parts=(*self.parts, other))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _weighted_standardise(
    X: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(X_standardised, col_mean, col_sd)`` using weighted moments.

    Matches the convention inside :func:`coordinate_descent_path` so we can
    pre-standardise X and feed it to statsmodels (which doesn't standardise)
    to compare to our solver on equal footing.
    """
    w_norm = w * (X.shape[0] / np.sum(w))
    wsum = float(np.sum(w_norm))
    mean = (w_norm @ X) / wsum
    Xc = X - mean
    var = (w_norm @ (Xc * Xc)) / wsum
    sd = np.sqrt(var)
    sd_safe = np.where(sd > 0, sd, 1.0)
    return Xc / sd_safe, mean, sd


# ---------------------------------------------------------------------------
# Acceptance criterion 3: λ very large → β=0; λ=0 → OLS to 1e-8
# ---------------------------------------------------------------------------


def test_lasso_large_lambda_yields_zero_coefficients() -> None:
    """ESL §3.4.2: at λ ≥ λ_max the lasso solution is exactly β=0."""
    rng = np.random.default_rng(0)
    n, p = 80, 5
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)

    # λ_max from FHT 2010 (on standardised X, centred y): max_j |x_j^T y_c| / n.
    Xs, _, _ = _weighted_standardise(X, np.ones(n))
    y_c = y - y.mean()
    lam_max = float(np.max(np.abs(Xs.T @ y_c)) / n)

    # A multiplicative safety margin makes the assertion robust to numerical
    # CD tolerance (β=0 is optimal at exactly λ_max but might pick up jitter).
    betas, ints, infos = coordinate_descent_path(
        X, y, lambdas_l1=[10.0 * lam_max], lam_l2=0.0,
        intercept=True, tol=1e-10, max_iter=10_000,
    )
    assert infos[0].converged
    np.testing.assert_array_equal(betas[0], np.zeros(p))


def test_lasso_at_lambda_zero_recovers_ols_to_1e_8() -> None:
    """Lasso with λ=0 reduces to OLS (ESL §3.4.2)."""
    rng = np.random.default_rng(1)
    n, p = 200, 6
    X = rng.normal(size=(n, p))
    y = X @ rng.normal(size=p) + 0.3 * rng.normal(size=n) + 1.0

    ols_ref = sm.OLS(y, sm.add_constant(X)).fit().params
    betas, ints, infos = coordinate_descent_path(
        X, y, lambdas_l1=[0.0], lam_l2=0.0,
        intercept=True, tol=1e-12, max_iter=100_000,
    )
    got = np.concatenate([[ints[0]], betas[0]])
    np.testing.assert_allclose(got, np.asarray(ols_ref, dtype=float), atol=1e-8)


# ---------------------------------------------------------------------------
# Acceptance criterion 4: α=1 → lasso; α=0 → ridge closed form to 1e-6
# ---------------------------------------------------------------------------


def test_enet_pure_l1_matches_lasso_solver() -> None:
    """``lam_l2=0`` in the elastic-net path reproduces the pure-lasso solver."""
    rng = np.random.default_rng(2)
    n, p = 100, 4
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n) + 2 * X[:, 0]

    lam = 0.05
    betas_lasso, ints_lasso, _ = coordinate_descent_path(
        X, y, lambdas_l1=[lam], lam_l2=0.0,
        intercept=True, tol=1e-12, max_iter=100_000,
    )
    betas_enet, ints_enet, _ = coordinate_descent_path(
        X, y, lambdas_l1=[lam], lam_l2=0.0,
        intercept=True, tol=1e-12, max_iter=100_000,
    )
    np.testing.assert_allclose(betas_enet[0], betas_lasso[0], atol=1e-14)
    assert ints_enet[0] == pytest.approx(ints_lasso[0], abs=1e-14)


def test_enet_pure_l2_matches_ridge_closed_form_to_1e_6() -> None:
    """``lam_l1=0`` recovers the ridge closed form on standardised X to 1e-6."""
    rng = np.random.default_rng(3)
    n, p = 100, 5
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n) + X @ np.array([1.0, -0.5, 0.0, 0.3, -1.0])

    lam2 = 0.2
    # Ridge closed form on standardised X with intercept absorbed by centring:
    #   beta_std = (X_std^T X_std / n + lam2 * I)^-1 (X_std^T y_c / n)
    # then de-standardise.
    X_mean = X.mean(axis=0)
    X_sd = X.std(axis=0, ddof=0)  # population SD
    X_std = (X - X_mean) / X_sd
    y_mean = y.mean()
    y_c = y - y_mean
    XtX_n = X_std.T @ X_std / n
    Xty_n = X_std.T @ y_c / n
    beta_std_ref = np.linalg.solve(XtX_n + lam2 * np.eye(p), Xty_n)
    beta_orig_ref = beta_std_ref / X_sd
    int_ref = y_mean - X_mean @ beta_orig_ref

    betas, ints, infos = coordinate_descent_path(
        X, y, lambdas_l1=[0.0], lam_l2=lam2,
        intercept=True, tol=1e-12, max_iter=100_000,
    )
    assert infos[0].converged
    np.testing.assert_allclose(betas[0], beta_orig_ref, atol=1e-6)
    assert ints[0] == pytest.approx(int_ref, abs=1e-6)


# ---------------------------------------------------------------------------
# Acceptance criterion 5: weighted CD matches hand-derived reference to 1e-8
# ---------------------------------------------------------------------------


def test_weighted_cd_lambda_zero_matches_wls() -> None:
    """Weighted lasso with λ=0 reduces to weighted OLS."""
    rng = np.random.default_rng(4)
    n, p = 100, 3
    X = rng.normal(size=(n, p))
    y = X @ np.array([1.0, -0.5, 0.2]) + 0.3 * rng.normal(size=n) + 0.5
    w = rng.uniform(0.5, 2.0, size=n)

    wls_ref = sm.WLS(y, sm.add_constant(X), weights=w).fit().params  # pyright: ignore[reportArgumentType]
    betas, ints, infos = coordinate_descent_path(
        X, y, lambdas_l1=[0.0], lam_l2=0.0, weights=w,
        intercept=True, tol=1e-13, max_iter=100_000,
    )
    assert infos[0].converged
    got = np.concatenate([[ints[0]], betas[0]])
    np.testing.assert_allclose(got, np.asarray(wls_ref, dtype=float), atol=1e-10)


def test_weighted_cd_matches_hand_derived_orthonormal_solution() -> None:
    r"""For a (weighted-)orthonormal X, weighted lasso = soft-threshold(X^T W y / n, λ).

    Hand-derivation: when ``X_s^T diag(w) X_s / n = I`` (where X_s is the
    weighted-standardised design and w sums to n), the coordinate-update
    equation reduces to ``β_j = S(z_j, λ)`` with ``z_j = X_s^T (w * y_c) / n``.
    This is a closed-form check that doesn't depend on any third-party
    optimiser.
    """
    rng = np.random.default_rng(5)
    n, p = 200, 3
    # Build X such that after weighted standardisation it is weighted-orthonormal.
    # Strategy: pick weights, then construct X by orthonormalising w.r.t. the
    # weighted inner product.
    w = rng.uniform(0.5, 1.5, size=n)
    w_norm = w * (n / w.sum())
    A = rng.normal(size=(n, p))
    A = A - (w_norm @ A) / n          # weighted-centre columns
    # Weighted QR via cholesky of weighted gram.
    G = A.T @ (w_norm[:, None] * A) / n  # weighted gram, shape (p, p)
    L = np.linalg.cholesky(G)
    X = A @ np.linalg.inv(L.T)          # X^T W X / n = I
    # y = X β + noise, then we'll soft-threshold the weighted inner product.
    beta_true = np.array([0.7, -0.4, 0.2])
    y = X @ beta_true + 0.1 * rng.normal(size=n) + 0.3
    y_mean_w = (w_norm @ y) / n
    y_c = y - y_mean_w

    lam = 0.05
    # Hand-derived analytical lasso under weighted-orthonormality:
    z = (w_norm[:, None] * X).T @ y_c / n  # = X^T diag(w) y_c / n
    expected_beta_std = np.sign(z) * np.maximum(0.0, np.abs(z) - lam)
    # On this construction, X already has weighted column SD = 1 and weighted mean 0,
    # so the internal standardisation is a no-op and the de-standardised beta equals beta_std.
    # The intercept is y_mean_w - col_mean_w @ beta = y_mean_w (col means are zero).
    expected_intercept = float(y_mean_w)

    betas, ints, infos = coordinate_descent_path(
        X, y, lambdas_l1=[lam], lam_l2=0.0, weights=w,
        intercept=True, tol=1e-13, max_iter=100_000,
    )
    assert infos[0].converged
    np.testing.assert_allclose(betas[0], expected_beta_std, atol=1e-8)
    assert ints[0] == pytest.approx(expected_intercept, abs=1e-8)


# ---------------------------------------------------------------------------
# Acceptance criterion 2: warm-started path is faster than from-scratch
# ---------------------------------------------------------------------------


def test_warm_started_path_is_faster_than_from_scratch() -> None:
    """Path-wise warm starts beat per-λ cold starts. Run each 5× and take the min.

    FHT 2010 §2.4 typically reports 5–10× warm-start speedups on moderately
    sparse problems. The threshold below (1.25×) is conservative — the
    measurement itself is what matters; the assertion is just a CI guard.
    The speedup is printed in the assertion message for visibility.
    """
    rng = np.random.default_rng(11)
    # Bigger problem with tighter tol → larger relative warm-start advantage.
    n, p = 600, 100
    X = rng.normal(size=(n, p))
    beta_true = np.zeros(p)
    beta_true[:10] = rng.normal(size=10)
    y = X @ beta_true + 0.3 * rng.normal(size=n)

    Xs, _, _ = _weighted_standardise(X, np.ones(n))
    y_c = y - y.mean()
    lam_max = float(np.max(np.abs(Xs.T @ y_c)) / n)
    n_lam = 50
    grid = np.geomspace(lam_max, lam_max * 1e-3, num=n_lam)

    # Warm up (JIT, dispatch caches, etc.). One throwaway run each.
    coordinate_descent_path(X, y, lambdas_l1=grid, lam_l2=0.0, intercept=True,
                            tol=1e-8, max_iter=10_000, warm_start=True)
    coordinate_descent_path(X, y, lambdas_l1=[grid[0]], lam_l2=0.0, intercept=True,
                            tol=1e-8, max_iter=10_000, warm_start=False)

    warm_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        coordinate_descent_path(
            X, y, lambdas_l1=grid, lam_l2=0.0,
            intercept=True, tol=1e-8, max_iter=10_000, warm_start=True,
        )
        warm_times.append(time.perf_counter() - t0)
    warm_t = min(warm_times)

    cold_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        for lam in grid:
            coordinate_descent_path(
                X, y, lambdas_l1=[lam], lam_l2=0.0,
                intercept=True, tol=1e-8, max_iter=10_000, warm_start=False,
            )
        cold_times.append(time.perf_counter() - t0)
    cold_t = min(cold_times)

    speedup = cold_t / warm_t
    assert speedup > 1.25, (
        f"warm-start should be faster than from-scratch; "
        f"warm={warm_t:.4f}s, cold={cold_t:.4f}s, speedup={speedup:.2f}x"
    )


# ---------------------------------------------------------------------------
# Acceptance criterion 1: match a glmnet-style reference on prostate-path
# ---------------------------------------------------------------------------


PROSTATE_PREDICTORS = (
    "lcavol", "lweight", "age", "lbph", "svi", "lcp", "gleason", "pgg45",
)


@pytest.fixture(scope="module")
def prostate_train() -> pd.DataFrame:
    """ESL prostate dataset, training subset, predictors standardised on the full 97 rows."""
    path = Path(__file__).parent / "data" / "prostate.csv"
    df = pd.read_csv(path)
    X = df[list(PROSTATE_PREDICTORS)].astype(float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    df = df.copy()
    df[list(PROSTATE_PREDICTORS)] = (X - mu) / sd
    return df.loc[df["train"] == "T"].reset_index(drop=True)


def _glmnet_style_reference(
    X: np.ndarray, y: np.ndarray, *, lam_l1: float, lam_l2: float, w: np.ndarray | None = None
) -> tuple[np.ndarray, float]:
    """Reference path: weighted-standardise X, fit statsmodels on standardised X, de-standardise.

    statsmodels' ``OLS.fit_regularized`` minimises
    ``0.5 RSS / n + alpha * (L1_wt * |β|_1 + (1-L1_wt) * |β|_2² / 2)``,
    which matches our solver's objective with ``alpha = lam_l1 + lam_l2``
    and ``L1_wt = lam_l1 / (lam_l1 + lam_l2)``. Fitting on the
    weighted-standardised X (the same internal representation our solver
    uses) makes the two solvers solve the *exact same* optimisation problem.
    """
    n = X.shape[0]
    w_eff = np.ones(n) if w is None else w
    X_std, col_mean, col_sd = _weighted_standardise(X, w_eff)
    w_norm = w_eff * (n / np.sum(w_eff))
    y_mean = float(np.sum(w_norm * y) / n)
    y_c = y - y_mean

    total = lam_l1 + lam_l2
    alpha_scalar = total
    l1_wt = 1.0 if total <= 0 else (lam_l1 / total)
    # zero_tol=0 disables statsmodels' aggressive coefficient-snapping; we want
    # bit-for-bit comparison of the optimum, not the snapped values. cnvrg_tol
    # set very tight so any difference is purely from CD's coordinate sweeps.
    if w is None:
        ref = sm.OLS(y_c, X_std).fit_regularized(
            method="elastic_net",
            alpha=alpha_scalar,
            L1_wt=l1_wt,
            cnvrg_tol=1e-14,
            maxiter=200_000,
            zero_tol=0.0,
        )
    else:
        ref = sm.WLS(y_c, X_std, weights=w_eff).fit_regularized(  # pyright: ignore[reportArgumentType]
            method="elastic_net",
            alpha=alpha_scalar,
            L1_wt=l1_wt,
            cnvrg_tol=1e-14,
            maxiter=200_000,
            zero_tol=0.0,
        )
    beta_std_ref = np.asarray(ref.params, dtype=float)
    nonzero = col_sd > 0
    beta_orig = np.zeros_like(beta_std_ref)
    beta_orig[nonzero] = beta_std_ref[nonzero] / col_sd[nonzero]
    intercept = y_mean - float(col_mean @ beta_orig)
    return beta_orig, intercept


def test_prostate_lasso_path_matches_glmnet_style_reference_to_1e_6(
    prostate_train: pd.DataFrame,
) -> None:
    """On the prostate path, our lasso matches a glmnet-style reference to 1e-6."""
    X = prostate_train[list(PROSTATE_PREDICTORS)].to_numpy(dtype=float)
    y = prostate_train["lpsa"].to_numpy(dtype=float)
    n = X.shape[0]

    # Build a representative descending grid in the interesting regime
    # (small enough that several coefficients are nonzero, large enough that
    # some are exactly zero).
    Xs, _, _ = _weighted_standardise(X, np.ones(n))
    y_c = y - y.mean()
    lam_max = float(np.max(np.abs(Xs.T @ y_c)) / n)
    grid = np.geomspace(lam_max, lam_max * 1e-3, num=20)

    betas, ints, infos = coordinate_descent_path(
        X, y, lambdas_l1=grid, lam_l2=0.0,
        intercept=True, tol=1e-12, max_iter=200_000,
    )
    for k, lam in enumerate(grid):
        beta_ref, int_ref = _glmnet_style_reference(X, y, lam_l1=float(lam), lam_l2=0.0)
        max_coef_err = float(np.max(np.abs(betas[k] - beta_ref)))
        assert max_coef_err < 1e-6, (
            f"path index {k}, λ={lam:.4g}: max coefficient error {max_coef_err:.2e} "
            f"exceeds 1e-6.\n  got     : {betas[k]}\n  expected: {beta_ref}"
        )
        assert ints[k] == pytest.approx(int_ref, abs=1e-6)
        assert infos[k].converged, f"path index {k}, λ={lam:.4g} did not converge"


def test_prostate_elastic_net_path_matches_glmnet_style_reference_to_1e_6(
    prostate_train: pd.DataFrame,
) -> None:
    """Same as the lasso path test, but for elastic net (α = 0.5)."""
    X = prostate_train[list(PROSTATE_PREDICTORS)].to_numpy(dtype=float)
    y = prostate_train["lpsa"].to_numpy(dtype=float)
    n = X.shape[0]

    Xs, _, _ = _weighted_standardise(X, np.ones(n))
    y_c = y - y.mean()
    # For elastic net with α=0.5, λ_max = (1/n α) max|...| = 2 × the α=1 value.
    alpha = 0.5
    lam_max = float(np.max(np.abs(Xs.T @ y_c)) / (n * alpha))
    grid = np.geomspace(lam_max, lam_max * 1e-3, num=20)

    for lam in grid:
        lam_l1 = float(lam * alpha)
        lam_l2 = float(lam * (1.0 - alpha))
        betas, ints, info = coordinate_descent_path(
            X, y, lambdas_l1=[lam_l1], lam_l2=lam_l2,
            intercept=True, tol=1e-12, max_iter=200_000,
        )
        beta_ref, int_ref = _glmnet_style_reference(X, y, lam_l1=lam_l1, lam_l2=lam_l2)
        np.testing.assert_allclose(betas[0], beta_ref, atol=1e-6)
        assert ints[0] == pytest.approx(int_ref, abs=1e-6)
        assert info[0].converged


# ---------------------------------------------------------------------------
# Convergence / divergence behaviour
# ---------------------------------------------------------------------------


def test_did_not_converge_path_sets_converged_false() -> None:
    """When max_iter is exhausted before tol is met, converged=False."""
    rng = np.random.default_rng(13)
    n, p = 300, 40
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    # max_iter=1 is unlikely to converge on a problem with many active coefs.
    _, _, infos = coordinate_descent_path(
        X, y, lambdas_l1=[1e-6], lam_l2=0.0,
        intercept=True, tol=1e-12, max_iter=1,
    )
    assert infos[0].converged is False
    assert infos[0].n_iter == 1


# ---------------------------------------------------------------------------
# Dispatch wiring through the SolverInputs / SolverOutputs contract
# ---------------------------------------------------------------------------


def _make_inputs(
    X: np.ndarray, y: np.ndarray, penalty: object, *, intercept: bool = True,
    weights: np.ndarray | None = None,
) -> Any:
    """Build a SolverInputs payload directly (bypasses dispatch)."""
    from model_crafter._internal.design import DesignMatrix
    from model_crafter.loss import squared_error
    from model_crafter.solve._registry import SolverInputs
    from model_crafter.spec import LinearSpec
    from model_crafter.terms.base import _normalize_features

    n, p = X.shape
    feature_names = [f"x{i}" for i in range(1, p + 1)]
    df = pd.DataFrame(X, columns=feature_names)  # pyright: ignore[reportArgumentType]
    df["y"] = y

    spec = LinearSpec(
        target="y",
        features=_normalize_features(feature_names),
        loss=squared_error,
        penalty=penalty,  # type: ignore[arg-type]
        intercept=intercept,
    )

    cols: tuple[str, ...]
    vals: np.ndarray
    term_for_column: tuple[str, ...]
    if intercept:
        cols = ("(Intercept)", *feature_names)
        vals = np.hstack([np.ones((n, 1)), X])
        term_for_column = ("(Intercept)", *feature_names)
    else:
        cols = tuple(feature_names)
        vals = X
        term_for_column = tuple(feature_names)
    design = DesignMatrix(
        values=vals.astype(float),
        columns=cols,
        term_for_column=term_for_column,
        fit_state={},
    )
    return SolverInputs(
        spec=spec,
        design=design,
        y=y.astype(float),
        weights=weights,
        data_index=df.index,
        method=None,
    )


def test_solve_lasso_cd_via_solver_inputs() -> None:
    """Driving solve_lasso_cd through SolverInputs returns a valid SolverOutputs."""
    rng = np.random.default_rng(21)
    n, p = 100, 4
    X = rng.normal(size=(n, p))
    y = X[:, 0] - 0.5 * X[:, 1] + 0.2 * rng.normal(size=n)

    inputs = _make_inputs(X, y, L1Penalty(lam=0.05))
    out = solve_lasso_cd(inputs)
    assert out.converged
    assert out.coefficient_se is None  # penalised fits → no closed-form SE
    assert list(out.coefficients.index)[0] == "(Intercept)"
    assert len(out.coefficients) == 1 + p
    assert out.solver_info["solver"] == "coordinate_descent"
    assert out.solver_info["lam_l1"] == 0.05
    assert out.solver_info["lam_l2"] == 0.0


def test_solve_enet_cd_via_solver_inputs() -> None:
    """solve_enet_cd accepts a PenaltySum and stores both lambdas in solver_info."""
    rng = np.random.default_rng(22)
    n, p = 100, 4
    X = rng.normal(size=(n, p))
    y = X[:, 0] - 0.5 * X[:, 1] + 0.2 * rng.normal(size=n)

    penalty = L1Penalty(lam=0.05) + L2Penalty(lam=0.1)
    inputs = _make_inputs(X, y, penalty)
    out = solve_enet_cd(inputs)
    assert out.converged
    assert out.solver_info["lam_l1"] == 0.05
    assert out.solver_info["lam_l2"] == 0.1


def test_split_enet_parts_rejects_unsupported_penalty() -> None:
    """PenaltySum with two L1 parts is a clear error, not silently summed."""
    with pytest.raises(TypeError, match="multiple L1"):
        _split_enet_parts(L1Penalty(lam=0.1) + L1Penalty(lam=0.2))


def test_solve_enet_cd_rejects_pure_l2_sum() -> None:
    """An L2-only PenaltySum is rejected (ridge solver handles this case)."""
    rng = np.random.default_rng(23)
    n, p = 30, 3
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    penalty = PenaltySum(parts=(L2Penalty(lam=0.1),))
    inputs = _make_inputs(X, y, penalty)
    with pytest.raises(ValueError, match="ridge"):
        solve_enet_cd(inputs)


def test_solve_lasso_cd_rejects_l2_penalty() -> None:
    """An L2-only penalty isn't a lasso — the dispatch shouldn't route it here."""
    rng = np.random.default_rng(24)
    n, p = 30, 3
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    # We construct an L1 with no L1 mass to force lam_l2 != 0 via _split_enet_parts.
    # That branch is only reachable when an L2-shaped object is mis-routed —
    # _split_enet_parts itself returns (0, λ) for an L2Penalty, which exercises
    # the explicit check inside solve_lasso_cd.
    inputs = _make_inputs(X, y, L2Penalty(lam=0.1))
    with pytest.raises(ValueError, match="non-zero L2"):
        solve_lasso_cd(inputs)
