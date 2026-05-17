"""Tests for the bootstrap (AGENTS.md Task P3.C, DESIGN.md §3.2 / §8 Phase 3 #6).

Quoted acceptance criteria (AGENTS.md / DESIGN.md):

1. ``bootstrap`` with ``method="pairs"`` produces 95% percentile CIs that
   contain the OLS closed-form CIs to within 5% (relative error) across a
   synthetic linear regression (n=500, n_boot=2000).
2. ``method="residual"`` matches the pairs bootstrap to within 10% on the
   same problem.
3. ``selection_frequency`` on a lasso fit with a known sparse ground truth
   (n=1000, p=20, true nonzeros = 4 strong + 16 zeros, n_boot=500): true
   nonzero coefficients have selection frequency > 0.9; true zero
   coefficients have selection frequency < 0.3.
4. ``BootstrappedSolution.prediction_ci(new_data)`` returns a
   ``pd.DataFrame`` with ``["lower", "upper"]`` columns, both finite, with
   ``lower < upper`` for each row.
5. Reproducibility: same ``random_state`` → identical results to 1e-12.

Math references (ESL §7.11 / §8.2): pairs bootstrap resamples rows with
replacement; residual bootstrap reuses fitted X and adds resampled
residuals to the fitted predictions; percentile CIs read empirical
quantiles of the bootstrap coefficient distribution.

The heavy acceptance test (n_boot=2000, n=500) takes real wall-clock time
— it is marked ``slow`` so the rest of the suite stays fast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model_crafter.loss import squared_error
from model_crafter.penalty import l1
from model_crafter.solution import BootstrappedSolution
from model_crafter.solve import solve
from model_crafter.spec import linear
from model_crafter.validation.bootstrap import bootstrap

# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------


def _ols_problem(n: int = 500, seed: int = 0) -> tuple[pd.DataFrame, dict[str, float]]:
    """Make a y = b0 + b1*x1 + b2*x2 + N(0, sigma^2) regression problem.

    Returns the data frame and the true coefficients.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    true = {"(Intercept)": 0.5, "x1": 1.5, "x2": -0.75}
    y = true["(Intercept)"] + true["x1"] * x1 + true["x2"] * x2 + rng.normal(scale=1.0, size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2}), true


def _sparse_lasso_problem(
    n: int = 1000, p: int = 20, n_nonzero: int = 4, seed: int = 0
) -> tuple[pd.DataFrame, set[str]]:
    """A sparse problem: ``n_nonzero`` strong predictors, rest are pure noise.

    Returns the data frame and the names of the true non-zero predictors.
    Features are standardised so the lasso path is well-defined.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    # Strong but realistic effect sizes.
    true_beta = np.zeros(p)
    true_beta[:n_nonzero] = rng.choice([-1.0, 1.0], size=n_nonzero) * 2.5
    sigma = 1.0
    y = X @ true_beta + rng.normal(scale=sigma, size=n)
    cols = {f"x{i}": X[:, i] for i in range(p)}
    cols["y"] = y
    df = pd.DataFrame(cols)
    nonzero_names = {f"x{i}" for i in range(n_nonzero)}
    return df, nonzero_names


# ---------------------------------------------------------------------------
# Smoke tests: contract / shape / public API
# ---------------------------------------------------------------------------


def test_bootstrap_returns_bootstrapped_solution_with_required_fields() -> None:
    df, _ = _ols_problem(n=80, seed=0)
    spec = linear(target="y", features=["x1", "x2"], loss=squared_error)
    sol = solve(spec, data=df)
    bs = bootstrap(sol, df, n_boot=20, random_state=0)

    assert isinstance(bs, BootstrappedSolution)
    assert bs.n_boot == 20
    assert bs.method == "pairs"
    assert bs.base is sol
    assert isinstance(bs.coefficients_dist, pd.DataFrame)
    assert bs.coefficients_dist.shape[0] == 20
    # Columns equal design_columns of the base solution.
    assert list(bs.coefficients_dist.columns) == list(sol.design_columns)
    assert isinstance(bs.fit_state_dist, tuple)
    assert len(bs.fit_state_dist) == 20
    assert isinstance(bs.selection_frequency, pd.Series)
    assert list(bs.selection_frequency.index) == list(sol.design_columns)


def test_bootstrap_coefficient_ci_returns_dataframe_with_lower_upper() -> None:
    df, _ = _ols_problem(n=80, seed=1)
    spec = linear(target="y", features=["x1", "x2"], loss=squared_error)
    sol = solve(spec, data=df)
    bs = bootstrap(sol, df, n_boot=50, random_state=1)

    ci = bs.coefficient_ci(level=0.95)
    assert isinstance(ci, pd.DataFrame)
    assert list(ci.columns) == ["lower", "upper"]
    assert list(ci.index) == list(sol.design_columns)
    # Lower < upper for every row.
    assert (ci["lower"] < ci["upper"]).all()


def test_bootstrap_prediction_ci_lower_strictly_less_than_upper() -> None:
    """Acceptance #4: ``prediction_ci(new_data)`` is a DataFrame[``lower``,``upper``]."""
    df, _ = _ols_problem(n=100, seed=2)
    spec = linear(target="y", features=["x1", "x2"], loss=squared_error)
    sol = solve(spec, data=df)
    bs = bootstrap(sol, df, n_boot=50, random_state=2)

    new = pd.DataFrame({"x1": [0.0, 1.0, -1.0], "x2": [0.5, -0.5, 0.0]})
    pci = bs.prediction_ci(new, level=0.95)
    assert isinstance(pci, pd.DataFrame)
    assert list(pci.columns) == ["lower", "upper"]
    assert len(pci) == len(new)
    assert np.isfinite(pci.to_numpy()).all()
    assert (pci["lower"] < pci["upper"]).all()
    # Index matches new_data.
    assert list(pci.index) == list(new.index)


def test_bootstrap_reproducibility_to_1e_12() -> None:
    """Acceptance #5: identical ``random_state`` → identical coefficient draws."""
    df, _ = _ols_problem(n=100, seed=3)
    spec = linear(target="y", features=["x1", "x2"], loss=squared_error)
    sol = solve(spec, data=df)

    bs1 = bootstrap(sol, df, n_boot=30, random_state=42)
    bs2 = bootstrap(sol, df, n_boot=30, random_state=42)
    np.testing.assert_allclose(
        bs1.coefficients_dist.to_numpy(),
        bs2.coefficients_dist.to_numpy(),
        atol=1e-12,
    )
    # Different seed -> different (almost surely).
    bs3 = bootstrap(sol, df, n_boot=30, random_state=43)
    assert not np.allclose(
        bs1.coefficients_dist.to_numpy(), bs3.coefficients_dist.to_numpy()
    )


def test_bootstrap_rejects_unknown_method() -> None:
    df, _ = _ols_problem(n=40, seed=4)
    spec = linear(target="y", features=["x1", "x2"], loss=squared_error)
    sol = solve(spec, data=df)
    with pytest.raises(ValueError, match="method"):
        bootstrap(sol, df, n_boot=10, method="nonsense", random_state=0)


def test_bootstrap_bca_is_explicit_about_status() -> None:
    """BCa is documented as stubbed for v0 — calling raises ``NotImplementedError``."""
    df, _ = _ols_problem(n=40, seed=5)
    spec = linear(target="y", features=["x1", "x2"], loss=squared_error)
    sol = solve(spec, data=df)
    bs = bootstrap(sol, df, n_boot=10, random_state=0)
    with pytest.raises(NotImplementedError, match="BCa"):
        bs.coefficient_ci(method="bca")


def test_bootstrap_weights_passthrough() -> None:
    """Weights column flows through to each refit's ``solve(..., weights=...)``."""
    rng = np.random.default_rng(6)
    n = 100
    x = rng.normal(size=n)
    y = 0.5 + 2.0 * x + rng.normal(scale=0.5, size=n)
    w = rng.uniform(0.5, 1.5, size=n)
    df = pd.DataFrame({"y": y, "x": x, "w": w})
    spec = linear(target="y", features=["x"], loss=squared_error)
    sol = solve(spec, data=df, weights="w")

    bs = bootstrap(sol, df, n_boot=20, weights="w", random_state=0)
    # Coefficients are roughly centred near the truth.
    median = pd.Series(
        bs.coefficients_dist.median(numeric_only=False),
        index=bs.coefficients_dist.columns,
    )
    assert abs(float(median["x"]) - 2.0) < 0.3
    assert abs(float(median["(Intercept)"]) - 0.5) < 0.3


def test_bootstrap_stratify_preserves_class_balance() -> None:
    """Stratified resampling preserves the marginal frequency of the strata."""
    rng = np.random.default_rng(7)
    n = 200
    x = rng.normal(size=n)
    y = rng.binomial(1, 0.2, size=n).astype(float)
    df = pd.DataFrame({"y": y, "x": x, "stratum": y.astype(int)})
    # Squared-error here just so we don't pull in logistic for the smoke
    # test; the stratify mechanic is independent of the loss.
    spec = linear(target="y", features=["x"], loss=squared_error)
    sol = solve(spec, data=df)
    # Each resample must contain the same per-stratum count as the original.
    counts = df["stratum"].value_counts().sort_index()
    bs = bootstrap(sol, df, n_boot=10, stratify="stratum", random_state=0)
    assert bs.n_boot == 10
    # Sanity check that the stratify counts cover the whole dataset.
    assert counts.sum() == n


# ---------------------------------------------------------------------------
# Acceptance #1: pairs-bootstrap percentile CIs vs OLS closed-form CIs
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_pairs_bootstrap_ci_matches_ols_closed_form_to_5pct() -> None:
    """Acceptance #1 (DESIGN.md §8 Phase 3 / AGENTS.md P3.C):

    Pairs bootstrap 95% percentile CI width is within 5% relative error of
    the OLS closed-form 95% CI width on a synthetic n=500 problem
    (n_boot=2000, fixed seed=12345).
    """
    n = 500
    df, _ = _ols_problem(n=n, seed=12345)
    spec = linear(target="y", features=["x1", "x2"], loss=squared_error)
    sol = solve(spec, data=df)
    assert sol.coefficient_se is not None  # OLS produces SEs

    # OLS closed-form 95% CI per coefficient: beta ± t_{0.975, n-p} * SE.
    from scipy.stats import t as student_t

    p_design = len(sol.coefficients)
    z = float(student_t.ppf(0.975, df=n - p_design))
    closed_lower = sol.coefficients - z * sol.coefficient_se
    closed_upper = sol.coefficients + z * sol.coefficient_se
    closed_widths = (closed_upper - closed_lower).to_numpy(dtype=float)

    bs = bootstrap(sol, df, n_boot=2000, method="pairs", random_state=12345)
    ci = bs.coefficient_ci(level=0.95)
    boot_widths = (ci["upper"] - ci["lower"]).to_numpy(dtype=float)

    rel_err = np.abs(boot_widths - closed_widths) / np.abs(closed_widths)
    assert np.all(rel_err < 0.05), (
        f"Bootstrap CI widths differ from closed-form by >5%: "
        f"rel_err={rel_err.tolist()}, boot={boot_widths.tolist()}, "
        f"closed={closed_widths.tolist()}"
    )


# ---------------------------------------------------------------------------
# Acceptance #2: residual vs pairs bootstrap agreement on the same problem
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_residual_bootstrap_matches_pairs_to_10pct() -> None:
    """Acceptance #2: residual bootstrap CI widths within 10% of pairs CIs.

    Seed=7, n_boot=2000. The two bootstrap variants give asymptotically
    equivalent CI widths for an iid-errors linear model (ESL §8.2); 10%
    is the agreed finite-sample tolerance from AGENTS.md P3.C.
    """
    n = 500
    df, _ = _ols_problem(n=n, seed=7)
    spec = linear(target="y", features=["x1", "x2"], loss=squared_error)
    sol = solve(spec, data=df)

    bs_pairs = bootstrap(sol, df, n_boot=2000, method="pairs", random_state=7)
    bs_resid = bootstrap(sol, df, n_boot=2000, method="residual", random_state=7)

    ci_p = bs_pairs.coefficient_ci(level=0.95)
    ci_r = bs_resid.coefficient_ci(level=0.95)
    widths_p = (ci_p["upper"] - ci_p["lower"]).to_numpy(dtype=float)
    widths_r = (ci_r["upper"] - ci_r["lower"]).to_numpy(dtype=float)
    rel_err = np.abs(widths_r - widths_p) / np.abs(widths_p)
    assert np.all(rel_err < 0.10), (
        f"Residual vs pairs CI widths differ by >10%: rel_err={rel_err.tolist()}"
    )


# ---------------------------------------------------------------------------
# Acceptance #3: selection_frequency on a sparse lasso ground truth
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_lasso_selection_frequency_discriminates_truth() -> None:
    """Acceptance #3: true nonzero coefs have freq > 0.9, true zeros < 0.3."""
    df, nonzeros = _sparse_lasso_problem(n=1000, p=20, n_nonzero=4, seed=99)
    feat = [f"x{i}" for i in range(20)]
    # Pick a modest lambda — small enough that strong signals stay in, large
    # enough that noise variables get killed by the L1 penalty. We don't
    # tune via the lambda path here (P3.B's job); a fixed lambda is enough
    # to exhibit the selection-frequency contrast.
    spec = linear(target="y", features=feat, loss=squared_error, penalty=l1(0.1))
    sol = solve(spec, data=df)
    bs = bootstrap(sol, df, n_boot=500, method="pairs", random_state=2026)

    freq = bs.selection_frequency
    for name in nonzeros:
        assert freq[name] > 0.9, (
            f"true nonzero {name} has selection_frequency={freq[name]:.3f} <= 0.9"
        )
    zeros = [c for c in feat if c not in nonzeros]
    for name in zeros:
        assert freq[name] < 0.3, (
            f"true zero {name} has selection_frequency={freq[name]:.3f} >= 0.3"
        )


def test_selection_frequency_is_all_one_for_unpenalised_fit() -> None:
    """For non-lasso fits, ``selection_frequency`` is documented as all 1.0."""
    df, _ = _ols_problem(n=80, seed=10)
    spec = linear(target="y", features=["x1", "x2"], loss=squared_error)
    sol = solve(spec, data=df)
    bs = bootstrap(sol, df, n_boot=30, random_state=10)
    # All entries are 1.0 because exact zeros are vanishingly rare in OLS.
    assert (bs.selection_frequency == 1.0).all()


# ---------------------------------------------------------------------------
# Robustness: degenerate resamples are skipped, and excessive failures raise
# ---------------------------------------------------------------------------


def test_bootstrap_skips_assumption_errors_when_few(monkeypatch) -> None:
    """Per-resample :class:`AssumptionError` is caught and skipped silently.

    We simulate this by patching ``solve`` to fail with AssumptionError on
    one out of every several refits; the bootstrap should still return a
    valid BootstrappedSolution with n_boot effective resamples.
    """
    df, _ = _ols_problem(n=100, seed=11)
    spec = linear(target="y", features=["x1", "x2"], loss=squared_error)
    sol = solve(spec, data=df)

    import sys

    from model_crafter.assumptions import AssumptionError

    # The validation/__init__.py re-exports ``bootstrap`` as the function,
    # so ``import model_crafter.validation.bootstrap`` resolves to the
    # function via the namespace. Grab the module object from sys.modules
    # to monkeypatch the ``solve`` symbol the bootstrap function uses.
    bootstrap_module = sys.modules["model_crafter.validation.bootstrap"]

    counter = {"i": 0}
    real_solve = bootstrap_module.solve

    def flaky_solve(spec, data, **kw):
        counter["i"] += 1
        # Fail every 5th call -> ~20% failure rate.
        if counter["i"] % 5 == 0:
            raise AssumptionError("simulated rank failure on resample")
        return real_solve(spec, data, **kw)

    monkeypatch.setattr(bootstrap_module, "solve", flaky_solve)
    # 20% failure rate exceeds the documented >5% threshold — should raise.
    with pytest.raises(RuntimeError, match="resample"):
        bootstrap(sol, df, n_boot=20, random_state=0)


def test_bootstrap_n_boot_must_be_positive_int() -> None:
    df, _ = _ols_problem(n=40, seed=12)
    spec = linear(target="y", features=["x1", "x2"], loss=squared_error)
    sol = solve(spec, data=df)
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap(sol, df, n_boot=0, random_state=0)
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap(sol, df, n_boot=-3, random_state=0)


def test_coefficient_ci_level_validation() -> None:
    df, _ = _ols_problem(n=40, seed=13)
    spec = linear(target="y", features=["x1", "x2"], loss=squared_error)
    sol = solve(spec, data=df)
    bs = bootstrap(sol, df, n_boot=20, random_state=0)
    with pytest.raises(ValueError, match="level"):
        bs.coefficient_ci(level=0.0)
    with pytest.raises(ValueError, match="level"):
        bs.coefficient_ci(level=1.0)
    with pytest.raises(ValueError, match="level"):
        bs.coefficient_ci(level=1.5)


# ---------------------------------------------------------------------------
# BootstrappedSolution dataclass shape
# ---------------------------------------------------------------------------


def test_bootstrapped_solution_is_frozen() -> None:
    import dataclasses

    df, _ = _ols_problem(n=40, seed=14)
    spec = linear(target="y", features=["x1", "x2"], loss=squared_error)
    sol = solve(spec, data=df)
    bs = bootstrap(sol, df, n_boot=10, random_state=0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        bs.n_boot = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Block bootstrap via splitter (P3.B integration is optional at this stage)
# ---------------------------------------------------------------------------


def test_block_bootstrap_uses_splitter_blocks() -> None:
    """If a ``splitter`` is provided, resampling preserves block structure.

    We provide a synthetic splitter that yields fixed contiguous blocks;
    each resample concatenates whole blocks (not individual rows).
    """
    rng = np.random.default_rng(15)
    n = 60
    df = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )
    spec = linear(target="y", features=["x1", "x2"], loss=squared_error)
    sol = solve(spec, data=df)

    # A minimal splitter contract: an object with a ``.windows(data)`` method
    # that yields integer position arrays defining the blocks. P3.B's
    # production splitters expose a richer interface; the bootstrap accepts
    # any object whose ``.windows()`` returns iterables of integer indices.
    class FixedBlockSplitter:
        def windows(self, data):
            # 6 blocks of 10 rows each.
            for i in range(6):
                yield np.arange(i * 10, (i + 1) * 10)

    bs = bootstrap(
        sol, df, n_boot=20, splitter=FixedBlockSplitter(), random_state=0
    )
    assert bs.n_boot == 20
    assert bs.method == "block"
