r"""Penalty primitives.

Phase 1 only ships :class:`NoPenalty` (the additive identity for the penalty
algebra). :func:`l1`, :func:`l2`, and :class:`PenaltySum` arrive in Phase 2
(AGENTS.md Task P2.A) and extend this module.

Per DESIGN.md §2.3, ``+`` composes penalties:

>>> from model_crafter.penalty import NoPenalty
>>> NoPenalty() + NoPenalty()  # doctest: +ELLIPSIS
NoPenalty()

A ``Penalty`` plus a ``Term`` is a programming error and raises
``TypeError`` with a message pointing at ``features=`` vs ``penalty=``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = ["NoPenalty", "Penalty"]


@runtime_checkable
class Penalty(Protocol):
    """Minimal penalty surface.

    Per DESIGN.md §6.1, ``value(beta)`` is required and ``prox(beta, step)``
    is required when the penalty participates in proximal optimisation
    (Phase 2). ``NoPenalty`` implements ``value`` only.

    ``assumptions`` is declared as a read-only property so frozen
    dataclasses (whose fields are read-only) satisfy the protocol.
    """

    @property
    def assumptions(self) -> tuple: ...

    def value(self, beta: np.ndarray) -> float: ...

    def __add__(self, other: object) -> Penalty: ...


@dataclass(frozen=True, slots=True)
class NoPenalty:
    r"""The identity element of the penalty algebra.

    :math:`R(\beta) = 0` for all :math:`\beta`. Returned by the default
    constructor of :class:`~model_crafter.spec.LinearSpec` when no
    regularization is requested.
    """

    @property
    def assumptions(self) -> tuple:
        return ()

    def value(self, beta: np.ndarray) -> float:
        # Validate type, but the return is unconditionally zero.
        np.asarray(beta, dtype=float)  # raises if non-array-like
        return 0.0

    def __add__(self, other: object) -> Penalty:
        # Importing terms.base here would create a cycle on package import in
        # some loaders; use duck typing.
        if hasattr(other, "expand") and hasattr(other, "name"):
            raise TypeError(
                "cannot add a Term to a Penalty — pass terms via features= and "
                "penalties via penalty= (DESIGN.md §2.3)"
            )
        if isinstance(other, NoPenalty):
            return NoPenalty()
        if isinstance(other, Penalty):
            return other  # NoPenalty is the additive identity
        raise TypeError(f"unsupported operand for Penalty +: {type(other).__name__}")

    def __radd__(self, other: object) -> Penalty:
        return self.__add__(other)

    def __repr__(self) -> str:
        return "NoPenalty()"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, NoPenalty)

    def __hash__(self) -> int:
        return hash(("NoPenalty",))
