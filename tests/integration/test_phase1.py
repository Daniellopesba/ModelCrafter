"""Phase 1 integration tests.

Quotes AGENTS.md Task P1.INTEG acceptance criteria:

  1. All P1.A and P1.B acceptance criteria still pass.
  2. ``mc.solve(spec, data)`` end-to-end on the prostate dataset produces
     a ``Solution`` with an ``assumptions`` field that includes a passing
     ``FullRankDesign`` result.
  3. ``mc.solve(spec, data, classical_inference=True)`` on a
     heteroscedastic synthetic dataset produces an ``AssumptionReport``
     containing a ``Homoscedasticity`` INFO-level result with the
     Breusch-Pagan statistic. Without ``classical_inference=True``,
     classical checks are absent.
  4. Public API surface in ``model_crafter/__init__.py`` matches what's
     imported in the reference example in ``DESIGN.md`` §10 *for Phase 1
     scope* (``mc.linear``, ``mc.solve``, ``mc.predict``,
     ``mc.squared_error``, ``mc.check_assumptions``, ``mc.NoPenalty``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import model_crafter as mc
from model_crafter.assumptions import (
    AssumptionError,
    Homoscedasticity,
    Severity,
)

PROSTATE = Path(__file__).resolve().parent.parent / "data" / "prostate.csv"


def test_public_api_surface_phase1() -> None:
    """Phase-1 scope exports exactly the names DESIGN.md §10 uses here."""
    expected = {
        "NoPenalty",
        "__version__",
        "check_assumptions",
        "linear",
        "predict",
        "solve",
        "squared_error",
    }
    assert set(mc.__all__) == expected
    for name in expected:
        assert hasattr(mc, name), f"model_crafter.{name} missing"


def test_solve_prostate_attaches_passing_fullrankdesign() -> None:
    """End-to-end OLS on prostate; Solution.assumptions has FullRankDesign PASS."""
    df = pd.read_csv(PROSTATE)
    train = df.loc[df["train"] == "T"].copy()
    features = ["lcavol", "lweight", "age", "lbph", "svi", "lcp", "gleason", "pgg45"]

    spec = mc.linear(target="lpsa", features=features, loss=mc.squared_error)
    sol = mc.solve(spec, train)

    by_sev = sol.assumptions.by_severity()
    hard_results = by_sev[Severity.HARD]
    full_rank = next((r for r in hard_results if r.name == "FullRankDesign"), None)
    assert full_rank is not None, "FullRankDesign missing from HARD results"
    assert full_rank.passed, f"FullRankDesign should pass on prostate; got {full_rank}"


def test_default_solve_excludes_info_checks() -> None:
    """Without classical_inference=True, INFO-level checks must not appear."""
    df = pd.read_csv(PROSTATE)
    train = df.loc[df["train"] == "T"].copy()
    spec = mc.linear(
        target="lpsa",
        features=["lcavol", "lweight", "age"],
        loss=mc.squared_error,
    )
    sol = mc.solve(spec, train)
    info_results = sol.assumptions.by_severity()[Severity.INFO]
    assert info_results == (), (
        f"INFO checks must be absent by default; got {info_results}"
    )


def test_classical_inference_emits_homoscedasticity_with_bp_statistic() -> None:
    """On a heteroscedastic synthetic dataset, classical_inference=True yields
    a Homoscedasticity INFO result whose statistic is the Breusch-Pagan stat."""
    rng = np.random.default_rng(seed=20260517)
    n = 500
    x = rng.normal(size=n)
    # Variance grows quadratically with x — strongly heteroscedastic.
    noise = rng.normal(scale=1.0 + np.abs(x) * 2.0, size=n)
    y = 1.5 + 2.0 * x + noise
    data = pd.DataFrame({"y": y, "x": x})

    spec = mc.linear(target="y", features=["x"], loss=mc.squared_error)
    sol = mc.solve(spec, data, classical_inference=True)

    info_results = sol.assumptions.by_severity()[Severity.INFO]
    homo = next((r for r in info_results if r.name == Homoscedasticity().name), None)
    assert homo is not None, (
        f"Homoscedasticity INFO result missing; INFO results were {info_results}"
    )
    assert homo.statistic is not None, "Breusch-Pagan statistic must be reported"
    assert homo.statistic > 0, f"BP statistic should be positive; got {homo.statistic}"


def test_predict_returns_aligned_series() -> None:
    """mc.predict returns a Series aligned with new_data.index (P1.A contract)."""
    df = pd.read_csv(PROSTATE)
    train = df.loc[df["train"] == "T"].copy()
    test = df.loc[df["train"] == "F"].copy()
    spec = mc.linear(
        target="lpsa",
        features=["lcavol", "lweight", "age"],
        loss=mc.squared_error,
    )
    sol = mc.solve(spec, train)
    yhat = mc.predict(sol, test)
    assert isinstance(yhat, pd.Series)
    assert list(yhat.index) == list(test.index)


def test_check_assumptions_standalone_reports_without_raising() -> None:
    """mc.check_assumptions runs the framework and never raises, even on a
    rank-deficient design (DESIGN.md §10 — regulator pass)."""
    rng = np.random.default_rng(seed=1)
    n = 100
    x = rng.normal(size=n)
    data = pd.DataFrame(
        {
            "y": x + rng.normal(scale=0.1, size=n),
            "x": x,
            "x_copy": x,  # exact duplicate — rank-deficient with intercept
        }
    )
    spec = mc.linear(
        target="y", features=["x", "x_copy"], loss=mc.squared_error
    )
    report = mc.check_assumptions(spec, data, classical_inference=True)
    # Must not raise; must include a failing FullRankDesign result.
    hard = report.by_severity()[Severity.HARD]
    full_rank = next((r for r in hard if r.name == "FullRankDesign"), None)
    assert full_rank is not None
    assert not full_rank.passed


def test_rank_deficient_design_raises_via_solve() -> None:
    """solve() must raise AssumptionError on rank deficiency by default."""
    rng = np.random.default_rng(seed=2)
    n = 50
    x = rng.normal(size=n)
    data = pd.DataFrame(
        {"y": x + rng.normal(scale=0.1, size=n), "x": x, "x_copy": x}
    )
    spec = mc.linear(
        target="y", features=["x", "x_copy"], loss=mc.squared_error
    )
    with pytest.raises(AssumptionError):
        mc.solve(spec, data)
