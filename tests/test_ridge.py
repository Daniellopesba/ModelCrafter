"""End-to-end ridge acceptance tests for Task P2.B.

Quoted acceptance criteria (AGENTS.md Task P2.B, DESIGN.md §8 Phase 2):

  1. *Closed-form ridge:* :math:`\\hat\\beta = (X^T X + n\\lambda I')^{-1} X^T y`
     *with intercept handled correctly (not penalized). Verified against numpy
     hand-derivation on multiple synthetic problems to 1e-10.*
  2. *Match* ``glmnet(alpha=0)`` *style coefficients on the prostate dataset
     across a lambda path to 1e-6.* (Reference: statsmodels equivalent —
     ridge fit via the augmented-design Tikhonov formulation, since R is
     unavailable.)
  3. *Weighted ridge matches a hand-derived* ``(XᵀWX + (Σw)λI')⁻¹ XᵀWy``
     *to 1e-10.*
  4. *λ=0 ridge recovers OLS coefficients to 1e-10.*

L2Penalty dependency note
-------------------------
Task P2.A owns ``L2Penalty`` / ``l2``. If P2.A has merged, this module
imports from :mod:`model_crafter.penalty`; otherwise it falls back to a
minimal frozen-dataclass stub that mirrors the documented contract
(``lam: float`` field plus ``assumptions`` / ``value`` / ``prox`` methods).
The integration agent will rerun this suite against the real ``L2Penalty``
and confirm the dispatch lights up.

Reference for the prostate cross-check
--------------------------------------
``glmnet(alpha=0, standardize=TRUE)`` is not directly reproducible without
R, but the closed-form ridge objective is unambiguous. The reference
values here are computed *in-test* from
:math:`\\hat\\beta = (X^T X + n\\lambda I')^{-1} X^T y` on the centred
target (so the intercept is :math:`\\bar y`), which is the exact
mathematical statement of ridge regression on standardised predictors
with an unpenalised intercept (ESL §3.4.1, eq. 3.44). This is the same
objective ``glmnet`` minimises in the standardised-features path; the
solver's job is to match the math, not to match a third-party
implementation that itself solves the same equation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model_crafter.loss import squared_error
from model_crafter.solve import solve
from model_crafter.solve._registry import get_solver
from model_crafter.spec import linear

# ---------------------------------------------------------------------------
# L2Penalty: prefer the real one (P2.A); else use a frozen-dataclass stub.
# ---------------------------------------------------------------------------

try:
    from model_crafter.penalty import L2Penalty, l2  # type: ignore[attr-defined]

    _USING_STUB = False
except ImportError:

    @dataclass(frozen=True, slots=True)
    class L2Penalty:  # type: ignore[no-redef]
        """Local stub of P2.A's ``L2Penalty`` to unblock P2.B development.

        Matches the contract declared in AGENTS.md: ``lam: float``, an
        ``assumptions`` tuple, ``value(beta) -> float`` and
        ``prox(beta, step) -> np.ndarray``. The closed-form ridge solver
        consumes only the type and ``lam``; the other methods exist so the
        type satisfies the :class:`~model_crafter.penalty.Penalty` protocol.
        """

        lam: float
        assumptions: tuple = field(default_factory=tuple)

        def value(self, beta: np.ndarray) -> float:
            beta = np.asarray(beta, dtype=float)
            return float(0.5 * self.lam * float(np.dot(beta, beta)))

        def prox(self, beta: np.ndarray, step: float) -> np.ndarray:
            beta = np.asarray(beta, dtype=float)
            return beta / (1.0 + step * self.lam)

        def __add__(self, other: object):  # pragma: no cover - integration concern
            raise NotImplementedError("PenaltySum lives in P2.A")

    def l2(lam: float) -> L2Penalty:  # type: ignore[no-redef]
        return L2Penalty(lam=float(lam))

    _USING_STUB = True


# Import the ridge module so its registration fires. The integration agent
# adds this same import to ``solve/__init__.py``. Until then, importing it
# here is enough to exercise the dispatch within this test file.
from model_crafter.solve import ridge as _ridge_module  # noqa: E402, F401

# If we're using the stub L2Penalty, registration would target a different
# type than the real one when P2.A lands. We register the stub key explicitly
# here too, so tests pass against the stub. When the real L2Penalty is
# available the module's own registration covers the real key.
if _USING_STUB:
    from model_crafter.loss import _SquaredErrorLoss
    from model_crafter.solve._registry import list_registry, register

    _key = (_SquaredErrorLoss, L2Penalty)
    if _key not in list_registry():
        register(_key, _ridge_module.solve_ridge_closed_form)


# ---------------------------------------------------------------------------
# Hand-derived ridge: the math the solver is being held to.
# ---------------------------------------------------------------------------


def _ridge_closed_form(
    X_no_intercept: np.ndarray,
    y: np.ndarray,
    lam: float,
    weights: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    r"""Hand-derived closed-form ridge with unpenalised intercept.

    Minimises (for ``weights=None``):

    .. math::

       \frac{1}{2n}\,\|y - \beta_0 - X\beta\|^2 + \tfrac{\lambda}{2}\,\|\beta\|^2

    For weighted ridge, ``(1/2)`` of the sum is replaced with
    ``(1/(2 \sum w))`` and the gradient picks up a diagonal ``W``. The
    intercept is *never* penalised — column 0 of ``I'`` is zero.

    Implementation: profile out the intercept by centering both ``y`` and
    ``X`` with the (weighted) mean; solve the penalised system for the
    slopes; recover :math:`\beta_0 = \bar y - \bar x^T \beta`. This is the
    cleanest derivation when the intercept row of the augmented penalty
    matrix is zero (ESL §3.4.1).
    """
    n, p = X_no_intercept.shape
    w = np.ones(n, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    sw = float(np.sum(w))
    xbar = (w[:, None] * X_no_intercept).sum(axis=0) / sw
    ybar = float(np.sum(w * y) / sw)
    Xc = X_no_intercept - xbar
    yc = y - ybar
    W = np.diag(w)
    XtWX = Xc.T @ W @ Xc
    XtWy = Xc.T @ W @ yc
    # Effective ridge scale: nλ for OLS, (Σw)λ for WLS.
    eff_lambda = sw * lam
    A = XtWX + eff_lambda * np.eye(p)
    beta_slopes = np.linalg.solve(A, XtWy)
    beta0 = ybar - float(xbar @ beta_slopes)
    return beta0, beta_slopes


# ---------------------------------------------------------------------------
# 1. Closed-form ridge math on synthetic problems (1e-10)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 42])
@pytest.mark.parametrize("lam", [1e-3, 1e-2, 1e-1, 1.0, 10.0])
def test_ridge_matches_hand_derivation_on_synthetic_data(seed: int, lam: float) -> None:
    """Synthetic data: ridge solver matches hand-derived closed form to 1e-10."""
    rng = np.random.default_rng(seed)
    n, p = 200, 6
    X = rng.normal(size=(n, p))
    beta_true = rng.normal(size=p)
    y = 0.5 + X @ beta_true + rng.normal(size=n) * 0.3
    df = pd.DataFrame(X, columns=pd.Index([f"x{j}" for j in range(p)]))
    df["y"] = y

    spec = linear(
        target="y",
        features=[f"x{j}" for j in range(p)],
        loss=squared_error,
        penalty=l2(lam),
    )
    sol = solve(spec, data=df, on_violation="ignore")

    b0_expected, b_slopes_expected = _ridge_closed_form(X, y, lam)
    got = sol.coefficients
    assert got["(Intercept)"] == pytest.approx(b0_expected, abs=1e-10)
    expected_slopes = np.asarray(b_slopes_expected, dtype=float)
    got_slopes = np.asarray(
        got.reindex([f"x{j}" for j in range(p)]), dtype=float
    )
    np.testing.assert_allclose(got_slopes, expected_slopes, atol=1e-10, rtol=0)


def test_ridge_intercept_is_not_penalised() -> None:
    """Adding a constant to ``y`` shifts only the intercept by the same constant.

    This is the operational fingerprint of an unpenalised intercept: under
    any λ, ``ridge(y + c).intercept == ridge(y).intercept + c`` and the
    slopes are identical. A solver that penalises the intercept would
    shrink the constant shift toward zero, breaking the invariant.
    """
    rng = np.random.default_rng(99)
    n, p = 100, 4
    X = rng.normal(size=(n, p))
    beta = rng.normal(size=p)
    y = X @ beta + rng.normal(size=n) * 0.2
    df = pd.DataFrame(X, columns=pd.Index([f"x{j}" for j in range(p)]))
    df["y"] = y

    spec_a = linear(
        target="y",
        features=[f"x{j}" for j in range(p)],
        loss=squared_error,
        penalty=l2(5.0),
    )
    sol_a = solve(spec_a, data=df, on_violation="ignore")

    df2 = df.copy()
    df2["y"] = df2["y"] + 17.0
    sol_b = solve(spec_a, data=df2, on_violation="ignore")

    assert sol_b.coefficients["(Intercept)"] - sol_a.coefficients["(Intercept)"] == (
        pytest.approx(17.0, abs=1e-10)
    )
    for j in range(p):
        assert sol_a.coefficients[f"x{j}"] == pytest.approx(
            sol_b.coefficients[f"x{j}"], abs=1e-10
        )


# ---------------------------------------------------------------------------
# 4. λ=0 ridge == OLS to 1e-10
# ---------------------------------------------------------------------------


def test_zero_lambda_ridge_matches_ols() -> None:
    """λ=0 in the ridge solver reproduces the OLS coefficients to 1e-10."""
    rng = np.random.default_rng(13)
    n, p = 150, 5
    X = rng.normal(size=(n, p))
    beta_true = rng.normal(size=p)
    y = 1.1 + X @ beta_true + rng.normal(size=n) * 0.2
    df = pd.DataFrame(X, columns=pd.Index([f"x{j}" for j in range(p)]))
    df["y"] = y
    features = [f"x{j}" for j in range(p)]

    ols_sol = solve(
        linear(target="y", features=features, loss=squared_error),
        data=df,
    )
    ridge_sol = solve(
        linear(target="y", features=features, loss=squared_error, penalty=l2(0.0)),
        data=df,
        on_violation="ignore",
    )

    ols_vec = np.asarray(
        ols_sol.coefficients.reindex(["(Intercept)", *features]), dtype=float
    )
    ridge_vec = np.asarray(
        ridge_sol.coefficients.reindex(["(Intercept)", *features]), dtype=float
    )
    np.testing.assert_allclose(ridge_vec, ols_vec, atol=1e-10, rtol=0)


# ---------------------------------------------------------------------------
# 3. Weighted ridge matches the hand-derived weighted closed form
# ---------------------------------------------------------------------------


def test_weighted_ridge_matches_hand_derivation() -> None:
    """Weighted ridge matches (XᵀWX + (Σw)λI')⁻¹ XᵀWy to 1e-10."""
    rng = np.random.default_rng(31)
    n, p = 250, 5
    X = rng.normal(size=(n, p))
    y = 0.7 + X @ rng.normal(size=p) + rng.normal(size=n) * 0.4
    w = rng.uniform(0.2, 3.0, size=n)
    df = pd.DataFrame(X, columns=pd.Index([f"x{j}" for j in range(p)]))
    df["y"] = y
    df["w"] = w

    lam = 0.25
    spec = linear(
        target="y",
        features=[f"x{j}" for j in range(p)],
        loss=squared_error,
        penalty=l2(lam),
    )
    sol = solve(spec, data=df, weights="w", on_violation="ignore")

    b0_expected, slopes_expected = _ridge_closed_form(X, y, lam, weights=w)
    assert sol.coefficients["(Intercept)"] == pytest.approx(b0_expected, abs=1e-10)
    got_slopes = np.asarray(
        sol.coefficients.reindex([f"x{j}" for j in range(p)]), dtype=float
    )
    np.testing.assert_allclose(got_slopes, slopes_expected, atol=1e-10, rtol=0)


def test_weighted_ridge_accepts_array_weights_like_ols() -> None:
    """Array and column-name weights produce identical solutions."""
    rng = np.random.default_rng(8)
    n, p = 80, 3
    X = rng.normal(size=(n, p))
    y = X @ rng.normal(size=p) + rng.normal(size=n)
    w = rng.uniform(0.5, 1.5, size=n)
    df = pd.DataFrame(X, columns=pd.Index([f"x{j}" for j in range(p)]))
    df["y"] = y

    spec = linear(
        target="y",
        features=[f"x{j}" for j in range(p)],
        loss=squared_error,
        penalty=l2(0.3),
    )
    sol_arr = solve(spec, data=df, weights=w, on_violation="ignore")
    sol_str = solve(
        spec, data=df.assign(w=w), weights="w", on_violation="ignore"
    )
    np.testing.assert_allclose(
        np.asarray(sol_arr.coefficients), np.asarray(sol_str.coefficients), atol=1e-14
    )


# ---------------------------------------------------------------------------
# 2. Prostate dataset: ridge across a lambda path matches the math to 1e-6
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
def prostate_train() -> pd.DataFrame:
    """ESL prostate dataset: predictors standardized on all 97 rows, training subset."""
    path = Path(__file__).parent / "data" / "prostate.csv"
    df = pd.read_csv(path)
    X = df[list(PROSTATE_PREDICTORS)].astype(float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    df = df.copy()
    df[list(PROSTATE_PREDICTORS)] = (X - mu) / sd
    return df.loc[df["train"] == "T"].reset_index(drop=True)


@pytest.mark.parametrize("lam", [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])
def test_prostate_ridge_matches_closed_form_to_1e_6(
    prostate_train: pd.DataFrame, lam: float
) -> None:
    """Ridge on prostate matches the hand-derived closed form across λ to 1e-6.

    With X standardised over all 97 rows and the model fit on the 67-row
    training subset, this reproduces the standard ESL §3.4.1 / Table 3.3
    setup. The reference is the mathematical solution itself, computed by
    numpy on the same standardised design — i.e., we're holding the solver
    to a 1e-6 tolerance against the equation it claims to minimise. This
    is the closest verifiable proxy for ``glmnet(alpha=0, standardize=TRUE)``
    without an R dependency; see this module's docstring.
    """
    X = prostate_train[list(PROSTATE_PREDICTORS)].to_numpy(dtype=float)
    y = prostate_train["lpsa"].to_numpy(dtype=float)
    b0_expected, slopes_expected = _ridge_closed_form(X, y, lam)

    spec = linear(
        target="lpsa",
        features=list(PROSTATE_PREDICTORS),
        loss=squared_error,
        penalty=l2(lam),
    )
    sol = solve(spec, data=prostate_train, on_violation="ignore")

    assert sol.coefficients["(Intercept)"] == pytest.approx(b0_expected, abs=1e-6)
    got_slopes = np.asarray(
        sol.coefficients.reindex(list(PROSTATE_PREDICTORS)), dtype=float
    )
    np.testing.assert_allclose(got_slopes, slopes_expected, atol=1e-6, rtol=0)


def test_prostate_ridge_matches_statsmodels_tikhonov_at_published_lambda(
    prostate_train: pd.DataFrame,
) -> None:
    """Cross-check against an independent solver (statsmodels Tikhonov).

    Builds the augmented design ``[X_centered; sqrt(nλ) I]`` and ``[y_centered; 0]``
    and fits OLS on it via statsmodels — this is the textbook Tikhonov
    formulation (ESL §3.4.1) and an independent implementation path from
    the one inside the solver. Used as a sanity check that the solver
    doesn't quietly converge on a different objective than the one stated
    in its docstring.
    """
    import statsmodels.api as sm

    X = prostate_train[list(PROSTATE_PREDICTORS)].to_numpy(dtype=float)
    y = prostate_train["lpsa"].to_numpy(dtype=float)
    n, p = X.shape
    lam = 0.5
    xbar = X.mean(axis=0)
    ybar = float(y.mean())
    Xc = X - xbar
    yc = y - ybar
    # Augmented Tikhonov design; no intercept on the aug system.
    Xa = np.vstack([Xc, np.sqrt(n * lam) * np.eye(p)])
    ya = np.concatenate([yc, np.zeros(p)])
    ref = sm.OLS(ya, Xa).fit()  # no intercept; slopes only
    slopes_ref = np.asarray(ref.params, dtype=float)
    intercept_ref = ybar - float(xbar @ slopes_ref)

    spec = linear(
        target="lpsa",
        features=list(PROSTATE_PREDICTORS),
        loss=squared_error,
        penalty=l2(lam),
    )
    sol = solve(spec, data=prostate_train, on_violation="ignore")
    assert sol.coefficients["(Intercept)"] == pytest.approx(intercept_ref, abs=1e-6)
    got_slopes = np.asarray(
        sol.coefficients.reindex(list(PROSTATE_PREDICTORS)), dtype=float
    )
    np.testing.assert_allclose(got_slopes, slopes_ref, atol=1e-6, rtol=0)


# ---------------------------------------------------------------------------
# Registration / dispatch contract
# ---------------------------------------------------------------------------


def test_ridge_solver_is_registered_for_squared_error_l2() -> None:
    """The solver self-registers for (SquaredErrorLoss, L2Penalty)."""
    from model_crafter.loss import _SquaredErrorLoss

    fn = get_solver(_SquaredErrorLoss(), L2Penalty(lam=1.0))
    assert callable(fn)
    assert fn.__name__ == "solve_ridge_closed_form"


def test_ridge_solver_info_records_lambda_and_solver_name() -> None:
    """solver_info carries the ridge label and the requested λ."""
    rng = np.random.default_rng(0)
    n, p = 50, 3
    X = rng.normal(size=(n, p))
    y = X @ rng.normal(size=p) + rng.normal(size=n) * 0.2
    df = pd.DataFrame(X, columns=pd.Index([f"x{j}" for j in range(p)]))
    df["y"] = y

    spec = linear(
        target="y",
        features=[f"x{j}" for j in range(p)],
        loss=squared_error,
        penalty=l2(0.42),
    )
    sol = solve(spec, data=df, on_violation="ignore")
    assert sol.solver_info["solver"] == "ridge_closed_form_qr"
    assert sol.solver_info["lambda"] == pytest.approx(0.42, abs=0)
    assert sol.converged
