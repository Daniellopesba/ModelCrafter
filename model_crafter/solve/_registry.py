"""Solver dispatch registry.

Per DESIGN.md §9.4 the dispatch in :mod:`model_crafter.solve` is the only
place that picks a solver. To allow Phase 2's ridge and lasso solvers to
ship in parallel without editing the same file (see AGENTS.md "Dispatch
ownership" note for P2.B/P2.C coordination), each solver module
*registers itself on import*::

    from model_crafter.solve._registry import register
    from model_crafter.loss import _SquaredErrorLoss
    from model_crafter.penalty import NoPenalty
    register((_SquaredErrorLoss, NoPenalty), solve_ols)

The registry is a plain dict keyed by ``(type(loss), type(penalty))``.
``solve(...)`` looks up the function and calls it with a normalised
``SolverInputs`` payload.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from model_crafter._internal.design import DesignMatrix
from model_crafter.spec import LinearSpec

__all__ = ["SolverInputs", "SolverOutputs", "get_solver", "list_registry", "register"]


@dataclass(frozen=True, slots=True)
class SolverInputs:
    """Everything a solver needs, in normalised numerical form.

    A solver does not see the original data frame. It receives a
    materialised :class:`DesignMatrix`, the target vector, optional weights,
    and the original spec (for solver hyperparameters and method hints).
    """

    spec: LinearSpec
    design: DesignMatrix
    y: np.ndarray
    weights: np.ndarray | None
    data_index: pd.Index
    method: str | None


@dataclass(frozen=True, slots=True)
class SolverOutputs:
    """What a solver returns. Solve() wraps this into a Solution."""

    coefficients: pd.Series
    coefficient_se: pd.Series | None
    fit_state: Mapping[str, Any]
    loss_value: float
    penalty_value: float
    n_obs: int
    converged: bool
    solver_info: Mapping[str, Any]


SolverFn = Callable[[SolverInputs], SolverOutputs]


_REGISTRY: dict[tuple[type, type], SolverFn] = {}


def register(key: tuple[type, type], fn: SolverFn) -> None:
    """Register ``fn`` as the solver for ``key = (LossType, PenaltyType)``.

    Raises ``ValueError`` on duplicate registration so two Phase-2 modules
    cannot silently shadow each other.
    """
    if key in _REGISTRY:
        raise ValueError(
            f"solver already registered for {key!r}: {_REGISTRY[key].__name__}"
        )
    _REGISTRY[key] = fn


def get_solver(loss: Any, penalty: Any) -> SolverFn:
    """Return the registered solver for ``(type(loss), type(penalty))``.

    Raises ``LookupError`` with a helpful message if no solver is registered.
    """
    key = (type(loss), type(penalty))
    if key not in _REGISTRY:
        available = sorted((k[0].__name__, k[1].__name__) for k in _REGISTRY)
        raise LookupError(
            f"no solver registered for (loss={type(loss).__name__}, "
            f"penalty={type(penalty).__name__}). Registered: {available}"
        )
    return _REGISTRY[key]


def list_registry() -> dict[tuple[type, type], SolverFn]:
    """Return a shallow copy of the registry (introspection only)."""
    return dict(_REGISTRY)
