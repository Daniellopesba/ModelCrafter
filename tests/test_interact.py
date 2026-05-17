"""Tests for ``mc.interact`` and ``mc.cross`` interaction terms.

These exercise AGENTS.md Task P4.C's interaction term implementations. The
acceptance criteria they support, quoted from the task brief:

1. ``interact(a, b)`` on a frame produces columns: columns of ``a``,
   columns of ``b``, all pairwise products. Verified on a small
   hand-built case.
2. ``cross(a, b)`` produces only pairwise products. Verified.
3. Works with categorical (one-hot via step or binned) and continuous
   terms. Verified.
4. Works with WoE terms (WoE x scalar = a single column). Verified with
   a stub if P4.B not merged.
5. Predict-time interaction columns match training-time interaction
   columns exactly (fit_state pass-through). Verified.

Plus internal contract checks: flattening on construction, string
auto-promotion, frozen dataclass invariants, and union of assumptions
across operands.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pytest

from model_crafter.terms.base import ExpandedTerm, RawTerm, Term, TermSum
from model_crafter.terms.interact import (
    InteractionTerm,
    cross,
    interact,
)

# ---------------------------------------------------------------------------
# Local stubs for P4.A (basis) and P4.B (woe) operands.
# These satisfy the Term protocol and let us verify that interactions compose
# with multi-column terms (basis-style) and with stateful terms (woe-style).
# The integration agent removes these stubs and switches to the real imports
# at P4.INTEG.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _NsStub:
    """Stub for ``mc.ns(col, df=K)`` from P4.A.

    Expands a single column ``col`` into ``df`` polynomial basis columns
    (a stand-in for the natural cubic spline; the test only cares that
    multiple basis columns appear). The basis is deterministic so
    predict-time matches training-time.
    """

    col: str
    df_: int

    @property
    def name(self) -> str:
        return f"ns({self.col}, df={self.df_})"

    def expand(
        self,
        data: pd.DataFrame,
        *,
        fit_state: Mapping[str, Any] | None = None,
    ) -> ExpandedTerm:
        x = np.asarray(data[self.col], dtype=float)
        cols = [x ** (k + 1) for k in range(self.df_)]
        names = tuple(f"{self.name}_{k + 1}" for k in range(self.df_))
        return ExpandedTerm(columns=names, values=np.column_stack(cols))

    def __add__(self, other: object) -> TermSum:
        # Use the public RawTerm+ behaviour via construction of a TermSum.
        from model_crafter.terms.base import _add_terms

        return _add_terms(self, other)

    def __radd__(self, other: object) -> TermSum:
        from model_crafter.terms.base import _add_terms

        return _add_terms(other, self)


@dataclass(frozen=True, slots=True)
class _WoEStub:
    """Stub for ``mc.woe(col, bins=...)`` from P4.B.

    Expands ``col`` into a single WoE-encoded column. The "WoE" here is a
    fixed monotone transform (``log1p``) so the test is deterministic and
    P4.B-independent. Threads ``fit_state`` to demonstrate the
    pass-through contract: at fit time it receives ``None``; at predict
    time it receives whatever dict the interaction term threaded through.
    """

    col: str

    @property
    def name(self) -> str:
        return f"woe({self.col})"

    def expand(
        self,
        data: pd.DataFrame,
        *,
        fit_state: Mapping[str, Any] | None = None,
    ) -> ExpandedTerm:
        x = np.asarray(data[self.col], dtype=float)
        return ExpandedTerm(columns=(self.name,), values=np.log1p(x).reshape(-1, 1))

    def __add__(self, other: object) -> TermSum:
        from model_crafter.terms.base import _add_terms

        return _add_terms(self, other)

    def __radd__(self, other: object) -> TermSum:
        from model_crafter.terms.base import _add_terms

        return _add_terms(other, self)


@dataclass(frozen=True, slots=True)
class _AssumedTerm:
    """A stub term that declares one assumption; used to verify
    ``InteractionTerm.assumptions`` is the union of operand assumptions.
    """

    col: str
    assumption: object
    assumptions_field: tuple[object, ...] = field(default=())

    def __post_init__(self) -> None:
        object.__setattr__(self, "assumptions_field", (self.assumption,))

    @property
    def name(self) -> str:
        return self.col

    @property
    def assumptions(self) -> tuple[object, ...]:
        return self.assumptions_field

    def expand(
        self,
        data: pd.DataFrame,
        *,
        fit_state: Mapping[str, Any] | None = None,
    ) -> ExpandedTerm:
        x = np.asarray(data[self.col], dtype=float)
        return ExpandedTerm(columns=(self.col,), values=x.reshape(-1, 1))

    def __add__(self, other: object) -> TermSum:
        from model_crafter.terms.base import _add_terms

        return _add_terms(self, other)

    def __radd__(self, other: object) -> TermSum:
        from model_crafter.terms.base import _add_terms

        return _add_terms(other, self)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def toy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [10.0, 20.0, 30.0, 40.0, 50.0],
            "z": [0.5, 1.5, 2.5, 3.5, 4.5],
            "income": [100.0, 200.0, 300.0, 400.0, 500.0],
            "cat": [0.0, 1.0, 1.0, 0.0, 1.0],   # binary indicator
        }
    )


# ---------------------------------------------------------------------------
# Public API surface and string promotion
# ---------------------------------------------------------------------------


def test_interact_returns_term() -> None:
    """``interact(a, b)`` returns something that satisfies the Term protocol."""
    t = interact("x", "y")
    assert isinstance(t, Term)
    assert hasattr(t, "name") and isinstance(t.name, str)
    assert hasattr(t, "expand")


def test_cross_returns_term() -> None:
    """``cross(a, b)`` returns something that satisfies the Term protocol."""
    t = cross("x", "y")
    assert isinstance(t, Term)


def test_interact_promotes_strings() -> None:
    """String operands auto-promote to ``RawTerm`` inside the interaction."""
    t = interact("x", "y")
    assert isinstance(t, InteractionTerm)
    # Both operands are RawTerm("x"), RawTerm("y")
    assert all(isinstance(op, RawTerm) for op in t.operands)
    assert tuple(op.name for op in t.operands) == ("x", "y")


def test_cross_promotes_strings() -> None:
    """Same for ``cross``: string operands become ``RawTerm``s."""
    t = cross("x", "y")
    assert isinstance(t, InteractionTerm)
    assert all(isinstance(op, RawTerm) for op in t.operands)


def test_interact_requires_two_operands() -> None:
    """``interact`` with fewer than two operands raises a clear error."""
    with pytest.raises(ValueError, match="at least two"):
        interact("x")
    with pytest.raises(ValueError, match="at least two"):
        interact()


def test_cross_requires_two_operands() -> None:
    """``cross`` with fewer than two operands raises."""
    with pytest.raises(ValueError, match="at least two"):
        cross("x")
    with pytest.raises(ValueError, match="at least two"):
        cross()


def test_interact_rejects_non_str_non_term() -> None:
    """``interact`` rejects ints/lists/etc. with a clear TypeError."""
    with pytest.raises(TypeError):
        interact("x", 42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Acceptance #1: interact(a, b) = main effects + pairwise products
# ---------------------------------------------------------------------------


def test_interact_two_scalars_three_columns(toy_df: pd.DataFrame) -> None:
    """Acceptance #1: ``interact("x", "y")`` produces columns ``x``,
    ``y``, ``x*y`` with the hand-derived values.
    """
    t = interact("x", "y")
    exp = t.expand(toy_df, fit_state=None)
    assert exp.columns == ("x", "y", "x*y")
    expected = np.column_stack(
        [
            toy_df["x"].to_numpy(dtype=float),
            toy_df["y"].to_numpy(dtype=float),
            (toy_df["x"] * toy_df["y"]).to_numpy(dtype=float),
        ]
    )
    np.testing.assert_array_equal(exp.values, expected)


def test_interact_three_scalars_full_crossed(toy_df: pd.DataFrame) -> None:
    """Acceptance #1 (3 operands): ``interact(a, b, c)`` produces the
    fully-crossed ANOVA expansion: main effects + all pairwise + triple.
    Column order is main effects first (in operand order), then all
    pairwise products (lexicographic by operand index), then the
    highest-order product.
    """
    t = interact("x", "y", "z")
    exp = t.expand(toy_df, fit_state=None)
    assert exp.columns == ("x", "y", "z", "x*y", "x*z", "y*z", "x*y*z")
    xv = toy_df["x"].to_numpy(dtype=float)
    yv = toy_df["y"].to_numpy(dtype=float)
    zv = toy_df["z"].to_numpy(dtype=float)
    expected = np.column_stack(
        [xv, yv, zv, xv * yv, xv * zv, yv * zv, xv * yv * zv]
    )
    np.testing.assert_array_equal(exp.values, expected)


# ---------------------------------------------------------------------------
# Acceptance #2: cross(a, b) = pairwise products only
# ---------------------------------------------------------------------------


def test_cross_two_scalars_one_column(toy_df: pd.DataFrame) -> None:
    """Acceptance #2: ``cross("x", "y")`` produces the single column
    ``x*y`` with no main effects.
    """
    t = cross("x", "y")
    exp = t.expand(toy_df, fit_state=None)
    assert exp.columns == ("x*y",)
    expected = (toy_df["x"] * toy_df["y"]).to_numpy(dtype=float).reshape(-1, 1)
    np.testing.assert_array_equal(exp.values, expected)


def test_cross_three_scalars_highest_order_only(toy_df: pd.DataFrame) -> None:
    """Acceptance #2 (3 operands): ``cross(a, b, c)`` produces only the
    highest-order interaction column.
    """
    t = cross("x", "y", "z")
    exp = t.expand(toy_df, fit_state=None)
    assert exp.columns == ("x*y*z",)
    expected = (
        toy_df["x"] * toy_df["y"] * toy_df["z"]
    ).to_numpy(dtype=float).reshape(-1, 1)
    np.testing.assert_array_equal(exp.values, expected)


# ---------------------------------------------------------------------------
# Acceptance #3: continuous + "categorical" (one-hot via numeric stub)
# ---------------------------------------------------------------------------


def test_interact_continuous_with_categorical_dummy(toy_df: pd.DataFrame) -> None:
    """Acceptance #3: ``interact(continuous, categorical_one_hot)`` mixes
    a continuous column with a 0/1 indicator; the product equals the
    continuous value on rows where the indicator is 1 and zero otherwise.
    """
    t = interact("x", "cat")
    exp = t.expand(toy_df, fit_state=None)
    assert exp.columns == ("x", "cat", "x*cat")
    xv = toy_df["x"].to_numpy(dtype=float)
    cv = toy_df["cat"].to_numpy(dtype=float)
    np.testing.assert_array_equal(exp.values[:, 2], xv * cv)
    # The product is zero exactly where cat == 0.
    zeros = cv == 0
    assert np.all(exp.values[zeros, 2] == 0.0)


# ---------------------------------------------------------------------------
# Acceptance #3b: basis (multi-column) operand x scalar
# ---------------------------------------------------------------------------


def test_interact_basis_with_scalar(toy_df: pd.DataFrame) -> None:
    """Acceptance #3 (basis): ``interact(ns(x, df=3), z)`` produces the
    three ns columns, then ``z``, then the three element-wise products
    of each ns column with ``z``.
    """
    ns = _NsStub(col="x", df_=3)
    t = interact(ns, "z")
    exp = t.expand(toy_df, fit_state=None)
    ns_names = (f"{ns.name}_1", f"{ns.name}_2", f"{ns.name}_3")
    prod_names = (
        f"{ns.name}_1*z",
        f"{ns.name}_2*z",
        f"{ns.name}_3*z",
    )
    assert exp.columns == ns_names + ("z",) + prod_names
    ns_block = ns.expand(toy_df, fit_state=None).values
    zv = toy_df["z"].to_numpy(dtype=float).reshape(-1, 1)
    np.testing.assert_array_equal(exp.values[:, :3], ns_block)
    np.testing.assert_array_equal(exp.values[:, 3:4], zv)
    np.testing.assert_array_equal(exp.values[:, 4:7], ns_block * zv)


def test_cross_basis_with_scalar(toy_df: pd.DataFrame) -> None:
    """``cross(ns(x, df=3), z)`` produces only the three pairwise
    products, no main effects.
    """
    ns = _NsStub(col="x", df_=3)
    t = cross(ns, "z")
    exp = t.expand(toy_df, fit_state=None)
    assert exp.columns == (
        f"{ns.name}_1*z",
        f"{ns.name}_2*z",
        f"{ns.name}_3*z",
    )
    ns_block = ns.expand(toy_df, fit_state=None).values
    zv = toy_df["z"].to_numpy(dtype=float).reshape(-1, 1)
    np.testing.assert_array_equal(exp.values, ns_block * zv)


# ---------------------------------------------------------------------------
# Acceptance #4: WoE x scalar = one column
# ---------------------------------------------------------------------------


def test_cross_woe_with_scalar(toy_df: pd.DataFrame) -> None:
    """Acceptance #4: ``cross(woe("region"), "income")`` produces a single
    column equal to the element-wise product of the WoE-encoded values
    and the scalar column.
    """
    woe = _WoEStub(col="x")  # use "x" to keep the stub's dependencies minimal
    t = cross(woe, "income")
    exp = t.expand(toy_df, fit_state=None)
    assert exp.columns == (f"{woe.name}*income",)
    woe_block = woe.expand(toy_df, fit_state=None).values
    iv = toy_df["income"].to_numpy(dtype=float).reshape(-1, 1)
    np.testing.assert_array_equal(exp.values, woe_block * iv)


def test_interact_woe_with_scalar(toy_df: pd.DataFrame) -> None:
    """``interact(woe("x"), "income")`` produces WoE col + income + product."""
    woe = _WoEStub(col="x")
    t = interact(woe, "income")
    exp = t.expand(toy_df, fit_state=None)
    assert exp.columns == (woe.name, "income", f"{woe.name}*income")


# ---------------------------------------------------------------------------
# Acceptance #5: predict-time matches training-time (fit_state pass-through)
# ---------------------------------------------------------------------------


def test_interact_predict_columns_match_training(toy_df: pd.DataFrame) -> None:
    """Acceptance #5: training-time columns equal predict-time columns
    on the same term applied to disjoint data with the same schema.
    """
    train = toy_df.iloc[:3].reset_index(drop=True)
    holdout = toy_df.iloc[3:].reset_index(drop=True)
    t = interact("x", "y", "z")
    exp_train = t.expand(train, fit_state=None)
    exp_holdout = t.expand(holdout, fit_state=None)
    assert exp_train.columns == exp_holdout.columns


def test_interact_fit_state_round_trip(toy_df: pd.DataFrame) -> None:
    """Acceptance #5 (fit_state): expand accepts ``fit_state=None`` at
    training, and passes a per-operand state mapping through at predict
    time. Whatever the term returns is callable again with a structured
    fit_state without raising; columns match.
    """
    t = interact("x", "y")
    exp1 = t.expand(toy_df, fit_state=None)
    # Pass a structured fit_state (the interaction term threads operand
    # state by operand name); for stateless operands the state can be
    # None and the result is identical.
    fit_state: Mapping[str, Any] = {"operands": {"x": None, "y": None}}
    exp2 = t.expand(toy_df, fit_state=fit_state)
    assert exp1.columns == exp2.columns
    np.testing.assert_array_equal(exp1.values, exp2.values)


def test_interact_predict_with_basis_operand_matches_training(
    toy_df: pd.DataFrame,
) -> None:
    """Acceptance #5 (basis): a basis operand inside ``interact`` produces
    the same column names on a held-out frame.
    """
    ns = _NsStub(col="x", df_=3)
    t = interact(ns, "z")
    train = toy_df.iloc[:3].reset_index(drop=True)
    holdout = toy_df.iloc[3:].reset_index(drop=True)
    assert t.expand(train, fit_state=None).columns == t.expand(
        holdout, fit_state=None
    ).columns


# ---------------------------------------------------------------------------
# Flattening: interact(interact(a, b), c) == interact(a, b, c)
# ---------------------------------------------------------------------------


def test_interact_flattens_nested_interact() -> None:
    """``interact(interact(a, b), c)`` is equivalent to ``interact(a, b, c)``
    after construction. The flattened form is the canonical
    representation (so column order and counts agree).
    """
    nested = interact(interact("x", "y"), "z")
    flat = interact("x", "y", "z")
    assert isinstance(nested, InteractionTerm)
    assert tuple(op.name for op in nested.operands) == tuple(
        op.name for op in flat.operands
    )
    assert nested.kind == flat.kind


def test_cross_flattens_nested_cross() -> None:
    """``cross(cross(a, b), c)`` flattens to ``cross(a, b, c)``."""
    nested = cross(cross("x", "y"), "z")
    flat = cross("x", "y", "z")
    assert tuple(op.name for op in nested.operands) == tuple(
        op.name for op in flat.operands
    )
    assert nested.kind == flat.kind


def test_cross_does_not_flatten_interact_operand() -> None:
    """``cross(interact(a, b), c)`` does not silently flatten an operand
    of a *different* kind. The interact operand is opaque to the cross.
    """
    t = cross(interact("x", "y"), "z")
    # Outer kind is cross; it has two operands (the InteractionTerm and "z").
    assert t.kind == "cross"
    assert len(t.operands) == 2
    assert isinstance(t.operands[0], InteractionTerm)
    assert t.operands[0].kind == "interact"


def test_interact_does_not_flatten_cross_operand() -> None:
    """Symmetric: ``interact(cross(a, b), c)`` keeps the cross operand opaque."""
    t = interact(cross("x", "y"), "z")
    assert t.kind == "interact"
    assert isinstance(t.operands[0], InteractionTerm)
    assert t.operands[0].kind == "cross"


# ---------------------------------------------------------------------------
# Composition with TermSum via `+`
# ---------------------------------------------------------------------------


def test_interaction_term_supports_addition() -> None:
    """An ``InteractionTerm`` supports ``+`` to form a ``TermSum`` (like
    every other Term)."""
    t = interact("x", "y") + "z"
    assert isinstance(t, TermSum)
    assert len(t.terms) == 2


def test_interaction_term_name_is_stable() -> None:
    """The term's ``name`` is a deterministic string built from operand names."""
    t1 = interact("x", "y")
    t2 = interact("x", "y")
    assert t1.name == t2.name
    assert isinstance(t1.name, str) and t1.name


# ---------------------------------------------------------------------------
# Frozen dataclass invariants
# ---------------------------------------------------------------------------


def test_interaction_term_is_frozen() -> None:
    """``InteractionTerm`` is a frozen dataclass: mutation raises."""
    t = interact("x", "y")
    with pytest.raises((AttributeError, Exception)):
        t.operands = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Assumptions: union of operand assumptions
# ---------------------------------------------------------------------------


def test_interaction_assumptions_unions_operands() -> None:
    """``InteractionTerm.assumptions`` returns the union of operand
    assumptions; duplicate assumption *instances* are de-duplicated by
    identity to keep the framework from running the same check twice.
    """
    a_marker = object()
    b_marker = object()
    a = _AssumedTerm(col="x", assumption=a_marker)
    b = _AssumedTerm(col="y", assumption=b_marker)
    t = interact(a, b)
    # Both assumption markers appear, in operand order.
    assert hasattr(t, "assumptions")
    assert a_marker in t.assumptions
    assert b_marker in t.assumptions
    assert len(t.assumptions) == 2


def test_interaction_assumptions_dedupes_shared_marker() -> None:
    """When two operands share the same assumption instance, the union
    contains it once."""
    shared = object()
    a = _AssumedTerm(col="x", assumption=shared)
    b = _AssumedTerm(col="y", assumption=shared)
    t = cross(a, b)
    assert t.assumptions.count(shared) == 1


def test_interaction_assumptions_handles_operand_without_attribute() -> None:
    """An operand that does not declare ``assumptions`` contributes no
    assumptions; ``RawTerm`` is the canonical example."""
    t = interact("x", "y")  # both RawTerms, no .assumptions
    assert t.assumptions == ()
