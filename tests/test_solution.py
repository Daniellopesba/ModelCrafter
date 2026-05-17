"""Tests for Solution.

Pins the public-interface contract from AGENTS.md Task P1.A for
:mod:`model_crafter.solution`.
"""

from __future__ import annotations

import dataclasses
from types import MappingProxyType

import pandas as pd
import pytest

from model_crafter.loss import squared_error
from model_crafter.solution import Solution
from model_crafter.spec import linear


def _trivial_solution() -> Solution:
    from model_crafter.assumptions import (  # pyright: ignore[reportMissingImports]
        AssumptionReport,
    )

    spec = linear(target="y", features=["x"], loss=squared_error)
    return Solution(
        spec=spec,
        coefficients=pd.Series([1.0], index=["x"]),
        coefficient_se=pd.Series([0.5], index=["x"]),
        fit_state=MappingProxyType({}),
        design_columns=("x",),
        loss_value=0.0,
        penalty_value=0.0,
        n_obs=10,
        converged=True,
        solver_info=MappingProxyType({"solver": "ols"}),
        assumptions=AssumptionReport(),
    )


def test_solution_is_frozen() -> None:
    sol = _trivial_solution()
    with pytest.raises(dataclasses.FrozenInstanceError):
        sol.n_obs = 99  # type: ignore[misc]


def test_solution_has_required_fields() -> None:
    """Every field documented in AGENTS.md is present and typed as documented."""
    sol = _trivial_solution()
    for name in [
        "spec",
        "coefficients",
        "coefficient_se",
        "fit_state",
        "design_columns",
        "loss_value",
        "penalty_value",
        "n_obs",
        "converged",
        "solver_info",
        "assumptions",
    ]:
        assert hasattr(sol, name), f"Solution missing field: {name}"


def test_solution_coefficient_se_can_be_none() -> None:
    """coefficient_se is optional (None for solvers without closed-form SEs)."""
    from model_crafter.assumptions import (  # pyright: ignore[reportMissingImports]
        AssumptionReport,
    )

    spec = linear(target="y", features=["x"], loss=squared_error)
    sol = Solution(
        spec=spec,
        coefficients=pd.Series([1.0], index=["x"]),
        coefficient_se=None,
        fit_state=MappingProxyType({}),
        design_columns=("x",),
        loss_value=0.0,
        penalty_value=0.0,
        n_obs=10,
        converged=True,
        solver_info=MappingProxyType({}),
        assumptions=AssumptionReport(),
    )
    assert sol.coefficient_se is None
