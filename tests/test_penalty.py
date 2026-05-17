"""Tests for the Penalty protocol and NoPenalty.

Pins the public-interface contract from AGENTS.md Task P1.A for
:mod:`model_crafter.penalty`. ``l1``, ``l2`` and ``PenaltySum`` belong to
Task P2.A; this module ships only ``NoPenalty`` and the protocol.
"""

from __future__ import annotations

import numpy as np
import pytest

from model_crafter.penalty import NoPenalty, Penalty


def test_no_penalty_is_a_penalty() -> None:
    """NoPenalty() satisfies the Penalty protocol."""
    p = NoPenalty()
    assert isinstance(p, Penalty)


def test_no_penalty_has_empty_assumptions() -> None:
    """NoPenalty has no assumptions to declare."""
    p = NoPenalty()
    assert hasattr(p, "assumptions")
    assert p.assumptions == ()


def test_no_penalty_value_is_zero() -> None:
    """NoPenalty.value(beta) == 0 for any beta."""
    p = NoPenalty()
    beta = np.array([1.0, -2.0, 3.0])
    assert p.value(beta) == 0.0
    assert p.value(np.zeros(0)) == 0.0


def test_no_penalty_addition_with_term_raises_typeerror() -> None:
    """A Penalty + Term must be a TypeError pointing at the right argument."""
    from model_crafter.terms.base import RawTerm

    with pytest.raises(TypeError, match="features=.*penalty="):
        _ = NoPenalty() + RawTerm("x")  # type: ignore[operator]


def test_no_penalty_is_frozen() -> None:
    """NoPenalty is immutable."""
    p = NoPenalty()
    with pytest.raises((AttributeError, Exception)):
        p.foo = 1  # type: ignore[attr-defined]
