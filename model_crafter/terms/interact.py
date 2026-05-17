"""Interaction terms — ``interact`` (main effects + products) and ``cross``
(products only).

The two factories produce values of one frozen dataclass
:class:`InteractionTerm` that satisfies the :class:`~model_crafter.terms.base.Term`
protocol. The ``kind`` field discriminates between the two expansions.

Algebra
-------

For operands :math:`a, b, \\ldots, k` whose expansions produce column
blocks :math:`A, B, \\ldots, K` (each a matrix of one or more columns),

* ``interact(a, b, ..., k)`` expands to the *fully crossed* ANOVA model:
  every main effect, every pairwise product, every triple-wise product,
  up to the single highest-order product. Concretely:

  .. math::

      \\text{interact}(a, b, c) \\;=\\; A \\;\\cup\\; B \\;\\cup\\; C
          \\;\\cup\\; A \\odot B \\;\\cup\\; A \\odot C \\;\\cup\\; B \\odot C
          \\;\\cup\\; A \\odot B \\odot C,

  where :math:`A \\odot B` is the Khatri–Rao-style elementwise product over
  the Cartesian product of columns: ``len(A) * len(B)`` columns whose
  ``(i, j)``-th entry is ``A[:, i] * B[:, j]``.

* ``cross(a, b, ..., k)`` expands to *only* the highest-order interaction.
  For two operands that means the pairwise products :math:`A \\odot B`;
  for three operands the triple-wise products :math:`A \\odot B \\odot C`;
  and so on. No main effects, no lower-order interactions.

Column names follow a deterministic convention: pairwise and higher-order
columns are joined with ``*`` between sub-column names, so
``interact("x", "y")`` produces columns ``x``, ``y``, ``x*y`` and
``cross(ns("x", df=3), "z")`` produces ``ns(x, df=3)_1*z`` and so on.

ESL references
--------------

* §5 — basis-function expansions, of which an interaction column is the
  product of two such basis functions.
* §9.4 — MARS, whose central operation is precisely a product of hinge
  functions: ``cross(hinge(x, knot), hinge(y, knot))``.

Assumptions
-----------

An interaction term *introduces* no new assumptions of its own. It
*inherits* the assumptions of its operands: see
:attr:`InteractionTerm.assumptions`. Each declared assumption from an
operand is propagated to the assumption framework verbatim. Duplicate
assumption *instances* (same Python object) shared across operands are
de-duplicated so the framework does not run the same check twice.

Composition
-----------

Interactions of the same ``kind`` flatten on construction. Specifically::

    interact(interact(a, b), c)   ==  interact(a, b, c)
    cross(cross(a, b), c)         ==  cross(a, b, c)

Mixed-kind nesting is opaque: ``cross(interact(a, b), c)`` keeps the
``interact(a, b)`` operand as a single multi-column basis and computes
its product with ``c``.

String operands auto-promote to :class:`~model_crafter.terms.base.RawTerm`
via :func:`~model_crafter.terms.base._promote`. Multi-column operands
(``ns``, ``bs``, ``poly``, ``woe``, ``binned``, ``step``) are supported
because the expansion is defined over operand *blocks*, not single
columns.

``fit_state`` pass-through
--------------------------

The :meth:`InteractionTerm.expand` method receives a single
``fit_state`` mapping (per the :class:`Term` protocol) and threads
per-operand state through to each operand's own ``.expand`` call. The
contract is::

    fit_state = {"operands": {operand_name: operand_state, ...}}

If ``fit_state`` is ``None`` (training time) every operand is called
with ``fit_state=None``. If the mapping is present, each operand
receives its own slice keyed by its ``name``. Operands without an
entry receive ``None``. This matches the contract used by
:func:`~model_crafter._internal.design.build_design` — the top-level
``fit_state`` for a single term is whatever the term needs to round-trip
between fit and predict.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Literal

import numpy as np
import pandas as pd

from model_crafter.terms.base import (
    ExpandedTerm,
    Term,
    TermSum,
    _add_terms,
    _promote,
)

_Kind = Literal["interact", "cross"]


@dataclass(frozen=True, slots=True)
class InteractionTerm:
    """A multi-operand interaction term.

    Constructed by :func:`interact` or :func:`cross`; clients should
    prefer those factories over calling the constructor directly.

    Attributes
    ----------
    operands:
        The terms being interacted, in declaration order. Strings are
        promoted to :class:`~model_crafter.terms.base.RawTerm` by the
        factories before reaching the constructor.
    kind:
        ``"interact"`` produces main effects + every order of
        interaction; ``"cross"`` produces only the highest-order
        interaction. See module docstring for the algebra.
    """

    operands: tuple[Term, ...]
    kind: _Kind

    def __post_init__(self) -> None:
        if len(self.operands) < 2:
            raise ValueError(
                f"InteractionTerm requires at least two operands; got {len(self.operands)}"
            )
        for op in self.operands:
            if not isinstance(op, Term):
                raise TypeError(
                    f"InteractionTerm operand is not a Term: {op!r} "
                    f"(type {type(op).__name__})"
                )

    # Term protocol

    @property
    def name(self) -> str:
        inner = ", ".join(op.name for op in self.operands)
        return f"{self.kind}({inner})"

    @property
    def assumptions(self) -> tuple[Any, ...]:
        """Union of operand assumptions, preserving order, deduplicated
        by object identity.

        See module docstring (Assumptions). The framework runs each
        operand's assumptions verbatim; the interaction itself adds
        nothing new.
        """
        seen_ids: set[int] = set()
        out: list[Any] = []
        for op in self.operands:
            for assumption in getattr(op, "assumptions", ()):
                if id(assumption) in seen_ids:
                    continue
                seen_ids.add(id(assumption))
                out.append(assumption)
        return tuple(out)

    def expand(
        self,
        data: pd.DataFrame,
        *,
        fit_state: Mapping[str, Any] | None = None,
    ) -> ExpandedTerm:
        """Materialize the interaction's design-matrix block.

        See module docstring for the algebra and for the ``fit_state``
        contract.
        """
        operand_states = _operand_states(fit_state)
        # Expand each operand independently. Each block is a 2-D array.
        blocks: list[tuple[tuple[str, ...], np.ndarray]] = []
        for op in self.operands:
            sub_state = operand_states.get(op.name)
            sub = op.expand(data, fit_state=sub_state)
            blocks.append((sub.columns, sub.values))

        if self.kind == "cross":
            columns, values = _khatri_rao_block(blocks)
        else:
            columns, values = _fully_crossed_block(blocks)
        return ExpandedTerm(columns=columns, values=values)

    def __add__(self, other: object) -> TermSum:
        return _add_terms(self, other)

    def __radd__(self, other: object) -> TermSum:
        return _add_terms(other, self)


# Factories


def interact(*cols: str | Term) -> InteractionTerm:
    """Construct an ``interact`` term: main effects + all interactions.

    Parameters
    ----------
    *cols:
        Two or more column names or :class:`Term` values. Strings are
        promoted to :class:`~model_crafter.terms.base.RawTerm`.

    Returns
    -------
    InteractionTerm
        A frozen interaction term of ``kind="interact"``.

    Raises
    ------
    ValueError
        If fewer than two operands are passed.
    TypeError
        If any operand is neither a string nor a Term.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    >>> t = interact("x", "y")
    >>> exp = t.expand(df, fit_state=None)
    >>> exp.columns
    ('x', 'y', 'x*y')

    Composition flattens automatically::

        interact(interact("x", "y"), "z") == interact("x", "y", "z")
    """
    return _build("interact", cols)


def cross(*cols: str | Term) -> InteractionTerm:
    """Construct a ``cross`` term: only the highest-order interaction.

    Parameters
    ----------
    *cols:
        Two or more column names or :class:`Term` values. Strings are
        promoted to :class:`~model_crafter.terms.base.RawTerm`.

    Returns
    -------
    InteractionTerm
        A frozen interaction term of ``kind="cross"``.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    >>> t = cross("x", "y")
    >>> exp = t.expand(df, fit_state=None)
    >>> exp.columns
    ('x*y',)
    """
    return _build("cross", cols)


# Internal helpers


def _build(kind: _Kind, raw_operands: tuple[str | Term, ...]) -> InteractionTerm:
    """Promote, flatten same-kind nesting, validate, construct."""
    if len(raw_operands) < 2:
        raise ValueError(
            f"{kind}(...) requires at least two operands; got {len(raw_operands)}"
        )
    promoted: list[Term] = []
    for item in raw_operands:
        term = _promote(item)
        if isinstance(term, InteractionTerm) and term.kind == kind:
            # Same-kind nested interaction: flatten its operands into ours.
            promoted.extend(term.operands)
        else:
            promoted.append(term)
    return InteractionTerm(operands=tuple(promoted), kind=kind)


def _operand_states(
    fit_state: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    """Extract the ``operands`` sub-mapping from a fit_state.

    The contract: ``fit_state = {"operands": {operand_name: state, ...}}``.
    A missing or ``None`` ``fit_state`` yields an empty mapping (every
    operand receives ``fit_state=None``).
    """
    if fit_state is None:
        return {}
    operands = fit_state.get("operands")
    if operands is None:
        return {}
    if not isinstance(operands, Mapping):
        raise TypeError(
            "InteractionTerm.expand: fit_state['operands'] must be a Mapping; "
            f"got {type(operands).__name__}"
        )
    return operands


def _fully_crossed_block(
    blocks: list[tuple[tuple[str, ...], np.ndarray]],
) -> tuple[tuple[str, ...], np.ndarray]:
    """Assemble the ``interact`` expansion from per-operand blocks.

    Output order:

    1. Each operand's main-effect block, in operand order.
    2. All pairwise Khatri–Rao products
       (operand-index pairs in lexicographic order ``(i, j)`` with
       ``i < j``).
    3. All triple-wise Khatri–Rao products in lexicographic order.
    4. ... up to the single all-operands highest-order Khatri–Rao
       product.

    Within each Khatri–Rao block the column order matches
    :func:`_khatri_rao_pair` / :func:`_khatri_rao_block` (operand 1's
    columns vary slowest).
    """
    n_operands = len(blocks)
    col_names: list[str] = []
    value_parts: list[np.ndarray] = []

    # 1) Main effects.
    for cols, vals in blocks:
        col_names.extend(cols)
        value_parts.append(vals)

    # 2..k) All k-wise interactions for k = 2, ..., n_operands.
    for k in range(2, n_operands + 1):
        for idx in combinations(range(n_operands), k):
            sub_blocks = [blocks[i] for i in idx]
            sub_cols, sub_vals = _khatri_rao_block(sub_blocks)
            col_names.extend(sub_cols)
            value_parts.append(sub_vals)

    values = np.hstack(value_parts)
    return tuple(col_names), values


def _khatri_rao_block(
    blocks: list[tuple[tuple[str, ...], np.ndarray]],
) -> tuple[tuple[str, ...], np.ndarray]:
    """Reduce a list of operand blocks to their full Khatri–Rao product.

    The result for ``k`` operands is one column per element of the
    Cartesian product of their column indices. Column 0 of operand 0
    varies slowest; column ``-1`` of operand ``-1`` varies fastest. The
    resulting column name is the ``*``-joined concatenation of the
    contributing sub-column names.

    Two operands is the common case (pairwise interaction). For three
    or more operands the function reduces left-to-right.
    """
    if not blocks:
        raise ValueError("_khatri_rao_block requires at least one block")
    cols, vals = blocks[0]
    for next_cols, next_vals in blocks[1:]:
        cols, vals = _khatri_rao_pair(cols, vals, next_cols, next_vals)
    return cols, vals


def _khatri_rao_pair(
    left_cols: tuple[str, ...],
    left_vals: np.ndarray,
    right_cols: tuple[str, ...],
    right_vals: np.ndarray,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Pairwise Khatri–Rao (row-wise Kronecker) product of two blocks.

    Given ``L = (n, p)`` and ``R = (n, q)``, returns ``(n, p * q)`` with
    columns ordered such that ``L``'s columns vary slowest::

        out[:, i * q + j] = L[:, i] * R[:, j]

    Column names are ``f"{left_cols[i]}*{right_cols[j]}"``.
    """
    if left_vals.shape[0] != right_vals.shape[0]:
        raise ValueError(
            "Khatri-Rao operands have different row counts: "
            f"{left_vals.shape[0]} vs {right_vals.shape[0]}"
        )
    p = left_vals.shape[1]
    q = right_vals.shape[1]
    # Broadcasting trick: (n, p, 1) * (n, 1, q) -> (n, p, q), reshape to (n, p*q).
    prod = (left_vals[:, :, None] * right_vals[:, None, :]).reshape(
        left_vals.shape[0], p * q
    )
    names: list[str] = []
    for i in range(p):
        for j in range(q):
            names.append(f"{left_cols[i]}*{right_cols[j]}")
    return tuple(names), prod
