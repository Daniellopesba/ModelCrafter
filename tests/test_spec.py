"""Tests for LinearSpec and the mc.linear constructor.

Pins AGENTS.md Task P1.A's public-interface contract for
:mod:`model_crafter.spec`.
"""

from __future__ import annotations

import dataclasses

import pytest

from model_crafter.loss import squared_error
from model_crafter.penalty import NoPenalty
from model_crafter.spec import LinearSpec, linear
from model_crafter.terms.base import RawTerm, Term


def test_linear_returns_linearspec() -> None:
    """``mc.linear(...)`` returns a LinearSpec."""
    s = linear(target="y", features=["x1", "x2"], loss=squared_error)
    assert isinstance(s, LinearSpec)


def test_linear_default_penalty_is_no_penalty() -> None:
    s = linear(target="y", features=["x1"], loss=squared_error)
    assert isinstance(s.penalty, NoPenalty)


def test_linear_default_intercept_is_true() -> None:
    s = linear(target="y", features=["x1"], loss=squared_error)
    assert s.intercept is True


def test_linear_normalizes_string_features() -> None:
    """A string and a list-of-strings produce the same features tuple."""
    s = linear(target="y", features="x1", loss=squared_error)
    assert tuple(t.name for t in s.features) == ("x1",)
    assert all(isinstance(t, Term) for t in s.features)


def test_linear_normalizes_list_features() -> None:
    s = linear(target="y", features=["x1", "x2", "x3"], loss=squared_error)
    assert tuple(t.name for t in s.features) == ("x1", "x2", "x3")


def test_linear_normalizes_termsum_features() -> None:
    s = linear(
        target="y", features=RawTerm("a") + RawTerm("b") + "c", loss=squared_error
    )
    assert tuple(t.name for t in s.features) == ("a", "b", "c")


def test_linear_features_is_a_tuple() -> None:
    s = linear(target="y", features=["x1", "x2"], loss=squared_error)
    assert isinstance(s.features, tuple)


def test_linear_target_must_be_str() -> None:
    with pytest.raises(TypeError, match="target"):
        linear(target=42, features=["x"], loss=squared_error)  # type: ignore[arg-type]


def test_linear_target_must_be_non_empty() -> None:
    with pytest.raises(ValueError, match="target"):
        linear(target="", features=["x"], loss=squared_error)


def test_linear_features_must_be_non_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        linear(target="y", features=[], loss=squared_error)


def test_linear_loss_required() -> None:
    """loss is mandatory and must satisfy the Loss protocol."""
    with pytest.raises(TypeError):
        linear(target="y", features=["x"], loss="logistic")  # type: ignore[arg-type]


def test_linearspec_is_frozen() -> None:
    s = linear(target="y", features=["x"], loss=squared_error)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.target = "z"  # type: ignore[misc]


def test_linearspec_is_hashable() -> None:
    """Frozen specs are hashable so they can serve as dispatch keys."""
    s = linear(target="y", features=["x"], loss=squared_error)
    hash(s)  # does not raise
