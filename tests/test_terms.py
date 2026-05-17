"""Tests for Term, RawTerm, TermSum, _promote, _normalize_features.

These exercise the term primitives that compose the ``features=`` argument of
``mc.linear``. The acceptance criterion they support is from AGENTS.md Task P1.A,
specifically the public-interface contract for :mod:`model_crafter.terms.base`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model_crafter.terms.base import (
    ExpandedTerm,
    RawTerm,
    Term,
    TermSum,
    _normalize_features,
    _promote,
)


@pytest.fixture()
def toy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "income": [1.0, 2.0, 3.0, 4.0],
            "age": [10.0, 20.0, 30.0, 40.0],
            "tenure": [0.5, 1.5, 2.5, 3.5],
        }
    )


# ---------------------------------------------------------------------------
# RawTerm
# ---------------------------------------------------------------------------


def test_rawterm_name_is_column_name() -> None:
    """A RawTerm("x") has ``name == "x"``."""
    t = RawTerm("income")
    assert t.name == "income"


def test_rawterm_expand_returns_single_column(toy_df: pd.DataFrame) -> None:
    """RawTerm.expand returns an ExpandedTerm with one column equal to the data column."""
    t = RawTerm("income")
    exp = t.expand(toy_df, fit_state=None)
    assert isinstance(exp, ExpandedTerm)
    assert exp.columns == ("income",)
    np.testing.assert_array_equal(exp.values[:, 0], toy_df["income"].to_numpy(dtype=float))


def test_rawterm_expand_missing_column_raises(toy_df: pd.DataFrame) -> None:
    """RawTerm.expand names the missing column in its error message."""
    t = RawTerm("not_there")
    with pytest.raises(KeyError, match="not_there"):
        t.expand(toy_df, fit_state=None)


def test_rawterm_is_frozen() -> None:
    """RawTerm instances are immutable."""
    t = RawTerm("income")
    with pytest.raises((AttributeError, Exception)):
        t.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TermSum (and __add__)
# ---------------------------------------------------------------------------


def test_termsum_via_addition_flattens() -> None:
    """Adding three RawTerms with ``+`` produces a TermSum with three flat parts."""
    s = RawTerm("a") + RawTerm("b") + RawTerm("c")
    assert isinstance(s, TermSum)
    assert tuple(t.name for t in s.terms) == ("a", "b", "c")


def test_termsum_addition_with_string_promotes() -> None:
    """``RawTerm("a") + "b"`` promotes the string and yields a TermSum."""
    s = RawTerm("a") + "b"
    assert isinstance(s, TermSum)
    assert tuple(t.name for t in s.terms) == ("a", "b")


def test_termsum_addition_left_string_promotes() -> None:
    """``"a" + RawTerm("b")`` (reverse add) also works via ``__radd__``."""
    s: TermSum = "a" + RawTerm("b")  # type: ignore[operator]
    assert isinstance(s, TermSum)
    assert tuple(t.name for t in s.terms) == ("a", "b")


def test_termsum_addition_with_termsum_flattens() -> None:
    """``TermSum + TermSum`` produces a single flat TermSum, no nesting."""
    s1 = RawTerm("a") + RawTerm("b")
    s2 = RawTerm("c") + RawTerm("d")
    s = s1 + s2
    assert isinstance(s, TermSum)
    assert tuple(t.name for t in s.terms) == ("a", "b", "c", "d")
    for t in s.terms:
        assert not isinstance(t, TermSum)


def test_termsum_expand_concatenates(toy_df: pd.DataFrame) -> None:
    """TermSum.expand concatenates per-term expansions column-wise."""
    s = RawTerm("income") + RawTerm("age")
    exp = s.expand(toy_df, fit_state=None)
    assert exp.columns == ("income", "age")
    np.testing.assert_array_equal(exp.values[:, 0], toy_df["income"].to_numpy(dtype=float))
    np.testing.assert_array_equal(exp.values[:, 1], toy_df["age"].to_numpy(dtype=float))


# ---------------------------------------------------------------------------
# _promote
# ---------------------------------------------------------------------------


def test_promote_string_returns_rawterm() -> None:
    """_promote("col") returns a RawTerm named "col"."""
    t = _promote("col")
    assert isinstance(t, RawTerm)
    assert t.name == "col"


def test_promote_term_returns_self_identity() -> None:
    """_promote(term) is the same Term instance (no copy)."""
    rt = RawTerm("x")
    assert _promote(rt) is rt


def test_promote_rejects_other_types() -> None:
    """_promote raises TypeError for non-(str|Term) inputs."""
    with pytest.raises(TypeError):
        _promote(3.14)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _normalize_features
# ---------------------------------------------------------------------------


def test_normalize_features_string_to_single_term_tuple() -> None:
    """A bare string becomes a one-element tuple."""
    out = _normalize_features("income")
    assert tuple(t.name for t in out) == ("income",)
    assert all(isinstance(t, Term) for t in out)


def test_normalize_features_term_to_singleton_tuple() -> None:
    """A bare Term becomes a one-element tuple."""
    out = _normalize_features(RawTerm("income"))
    assert tuple(t.name for t in out) == ("income",)


def test_normalize_features_termsum_flattens() -> None:
    """A TermSum is flattened into its constituent terms."""
    out = _normalize_features(RawTerm("a") + RawTerm("b") + RawTerm("c"))
    assert tuple(t.name for t in out) == ("a", "b", "c")


def test_normalize_features_list_of_strings_promotes_each() -> None:
    """A list/tuple/iterable of strings and terms normalizes each element."""
    out = _normalize_features(["income", "age", RawTerm("tenure")])
    assert tuple(t.name for t in out) == ("income", "age", "tenure")


def test_normalize_features_empty_raises() -> None:
    """An empty feature list is rejected at normalization time."""
    with pytest.raises(ValueError, match="empty"):
        _normalize_features([])


def test_normalize_features_rejects_non_iterables() -> None:
    """Other types raise TypeError with a helpful message."""
    with pytest.raises(TypeError):
        _normalize_features(3.14)  # type: ignore[arg-type]
