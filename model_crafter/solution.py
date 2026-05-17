"""Solution dataclass.

A :class:`Solution` is the immutable artefact of fitting. Per DESIGN.md §2.1
there is no fitted/unfitted duality: a ``Solution`` is what falls out when
:func:`~model_crafter.solve.solve` is applied to a
:class:`~model_crafter.spec.LinearSpec` and data.

The ``assumptions`` field carries an :class:`AssumptionReport` (defined by
P1.B) summarising the prerequisite and stability checks that ran at solve
time. P1.A treats this type as opaque — it imports the type from
``model_crafter.assumptions`` and never reaches into the assumption
framework's internals.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from model_crafter.spec import LinearSpec

if TYPE_CHECKING:
    from model_crafter.assumptions import (  # pyright: ignore[reportMissingImports]
        AssumptionReport,
    )

__all__ = ["Solution"]


@dataclass(frozen=True, slots=True)
class Solution:
    """Immutable result of solving a :class:`LinearSpec` against data.

    See AGENTS.md Task P1.A for the field contract.
    """

    spec: LinearSpec
    coefficients: pd.Series
    coefficient_se: pd.Series | None
    fit_state: Mapping[str, Any]
    design_columns: tuple[str, ...]
    loss_value: float
    penalty_value: float
    n_obs: int
    converged: bool
    solver_info: Mapping[str, Any]
    assumptions: AssumptionReport

    def __post_init__(self) -> None:
        # Defensive validation. Failure here usually means a solver
        # constructed a Solution with mismatched coefficient names; the
        # error message names the disagreement.
        if not isinstance(self.spec, LinearSpec):
            raise TypeError(
                f"spec must be a LinearSpec; got {type(self.spec).__name__}"
            )
        if not isinstance(self.coefficients, pd.Series):
            raise TypeError("coefficients must be a pandas Series")
        if self.coefficient_se is not None and not isinstance(
            self.coefficient_se, pd.Series
        ):
            raise TypeError("coefficient_se must be a pandas Series or None")
        if list(self.coefficients.index) != list(self.design_columns):
            raise ValueError(
                "coefficients.index must equal design_columns; "
                f"got coefficients={list(self.coefficients.index)} vs "
                f"design_columns={list(self.design_columns)}"
            )
        if self.coefficient_se is not None and list(
            self.coefficient_se.index
        ) != list(self.design_columns):
            raise ValueError(
                "coefficient_se.index must equal design_columns; "
                f"got coefficient_se={list(self.coefficient_se.index)} vs "
                f"design_columns={list(self.design_columns)}"
            )
        if self.n_obs < 0:
            raise ValueError(f"n_obs must be non-negative; got {self.n_obs}")
