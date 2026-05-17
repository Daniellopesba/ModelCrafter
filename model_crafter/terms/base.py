"""Term protocol, RawTerm, TermSum, ExpandedTerm and feature normalization.

The ``Term`` protocol is the smallest unit of the linear predictor. A term
knows two things:

1. Its ``name``, used to label and identify it in error messages and in the
   coefficient table of a :class:`~model_crafter.solution.Solution`.
2. How to ``expand`` itself into an :class:`ExpandedTerm` — a small value
   carrying the resulting design-matrix columns (with names) for a given
   ``data`` frame.

Composition uses ``+``. Per DESIGN.md §2.2, ``+`` between terms produces a
:class:`TermSum`. ``TermSum``'s ``__add__`` is associative and *flattens* —
``(a + b) + (c + d)`` produces a single :class:`TermSum` with four parts, not
a nested tree. Strings auto-promote to :class:`RawTerm` on either side of
``+`` so the call site reads like the equation.

ESL §3.2 frames a linear predictor as
:math:`f(x) = \\beta_0 + \\sum_j h_j(x) \\beta_j` for chosen basis functions
``h_j``. A ``Term`` *is* the parameterised choice of one such ``h_j``; a
``TermSum`` is a list of choices. Phase 4 adds smoother bases (``ns``, ``bs``,
``poly``, ``hinge``, ``woe``); this module only ships the trivial
column-reference case.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

__all__ = [
    "ExpandedTerm",
    "RawTerm",
    "Term",
    "TermSum",
    "_normalize_features",
    "_promote",
]


@dataclass(frozen=True, slots=True)
class ExpandedTerm:
    """The expansion of a :class:`Term` against a data frame.

    Attributes
    ----------
    columns:
        One name per output column. For a :class:`RawTerm` this is just the
        source column name; for basis terms it includes basis-function indices.
    values:
        Float64 ``(n_obs, len(columns))`` array of expanded values. Always a
        2-D array even when there is only one column, so downstream code can
        ``np.hstack`` term expansions unconditionally.
    """

    columns: tuple[str, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError(
                f"ExpandedTerm.values must be 2-D; got shape {self.values.shape}"
            )
        if self.values.shape[1] != len(self.columns):
            raise ValueError(
                f"ExpandedTerm: {len(self.columns)} column names but values has "
                f"{self.values.shape[1]} columns"
            )


@runtime_checkable
class Term(Protocol):
    """The minimal protocol every feature term implements.

    See :mod:`model_crafter.terms.base` module docstring for the framing.

    ``name`` is declared as a read-only property so frozen dataclasses
    (whose attributes are effectively read-only) satisfy the protocol.
    """

    @property
    def name(self) -> str: ...

    def expand(
        self,
        data: pd.DataFrame,
        *,
        fit_state: Mapping[str, object] | None,
    ) -> ExpandedTerm:
        """Expand the term against ``data``.

        ``fit_state`` carries any per-term state learned at solve time
        (e.g. WoE bin edges); a stateless term ignores it.
        """
        ...

    def __add__(self, other: object) -> TermSum:
        """Compose two terms into a :class:`TermSum`."""
        ...


@dataclass(frozen=True, slots=True)
class RawTerm:
    """A term that copies one column of ``data`` verbatim.

    The most common ``Term``. ``RawTerm("income")`` is what the string
    ``"income"`` promotes to when it appears in ``features=``.
    """

    name: str

    def expand(
        self,
        data: pd.DataFrame,
        *,
        fit_state: Mapping[str, object] | None = None,
    ) -> ExpandedTerm:
        if self.name not in data.columns:
            raise KeyError(
                f"RawTerm refers to column '{self.name}' which is not in the data frame "
                f"(available columns: {list(data.columns)})"
            )
        series = pd.to_numeric(data[self.name], errors="raise")
        col = np.asarray(series, dtype=float)
        return ExpandedTerm(columns=(self.name,), values=col.reshape(-1, 1))

    def __add__(self, other: object) -> TermSum:
        return _add_terms(self, other)

    def __radd__(self, other: object) -> TermSum:
        return _add_terms(other, self)


@dataclass(frozen=True, slots=True)
class TermSum:
    """An ordered, flattened sum of :class:`Term` values.

    Composition uses ``+``; iteration over ``.terms`` yields the constituent
    terms in declaration order.
    """

    terms: tuple[Term, ...] = field(default=())

    def __post_init__(self) -> None:
        if not self.terms:
            raise ValueError("TermSum requires at least one term")
        for t in self.terms:
            if isinstance(t, TermSum):
                raise ValueError(
                    "TermSum must be constructed already-flattened; use _add_terms or '+'"
                )
            if not isinstance(t, Term):
                raise TypeError(f"TermSum part is not a Term: {t!r}")

    @property
    def name(self) -> str:
        return " + ".join(t.name for t in self.terms)

    def expand(
        self,
        data: pd.DataFrame,
        *,
        fit_state: Mapping[str, object] | None = None,
    ) -> ExpandedTerm:
        parts = [t.expand(data, fit_state=fit_state) for t in self.terms]
        cols: tuple[str, ...] = tuple(c for p in parts for c in p.columns)
        values = np.hstack([p.values for p in parts])
        return ExpandedTerm(columns=cols, values=values)

    def __add__(self, other: object) -> TermSum:
        return _add_terms(self, other)

    def __radd__(self, other: object) -> TermSum:
        return _add_terms(other, self)

    def __iter__(self):
        return iter(self.terms)


# Promotion and addition helpers


def _promote(x: object) -> Term:
    """Promote a string to a :class:`RawTerm`; pass terms through unchanged.

    Any other type raises ``TypeError``. The function is intentionally narrow:
    silent broadcasting of arbitrary array-likes into terms is exactly the kind
    of magic DESIGN.md §9.8 forbids.
    """
    if isinstance(x, str):
        return RawTerm(x)
    if isinstance(x, Term):
        return x
    raise TypeError(
        f"Expected a column name (str) or a Term; got {type(x).__name__}: {x!r}"
    )


def _add_terms(left: object, right: object) -> TermSum:
    """Implementation of ``Term + Term`` (with string promotion and flattening).

    Both sides are coerced via :func:`_promote`. :class:`TermSum`s are
    flattened so the result is always a one-level tuple.
    """
    # Defer Penalty-vs-Term type error to penalty.py, which knows about Penalty.
    # Here we just promote and flatten.
    lterm = _promote(left)
    rterm = _promote(right)
    parts: list[Term] = []
    parts.extend(lterm.terms if isinstance(lterm, TermSum) else (lterm,))
    parts.extend(rterm.terms if isinstance(rterm, TermSum) else (rterm,))
    return TermSum(terms=tuple(parts))


def _normalize_features(
    f: str | Term | Iterable[str | Term],
) -> tuple[Term, ...]:
    """Coerce the ``features=`` argument of :func:`~model_crafter.spec.linear` to a tuple.

    Accepts:

    * a single string (promoted),
    * a single :class:`Term` (or :class:`TermSum`, which is flattened),
    * an iterable of strings/Terms (each promoted; nested TermSums are
      flattened).

    Returns a tuple ``(t1, t2, ...)`` in declaration order. The tuple is
    guaranteed non-empty; if the input is empty, ``ValueError`` is raised
    with a message that names the offending argument (per DESIGN.md §9.8).
    """
    if isinstance(f, str):
        return (RawTerm(f),)
    if isinstance(f, TermSum):
        return tuple(f.terms)
    if isinstance(f, Term):
        return (f,)
    # Iterable case. Materialize once so we can fail loudly on empties without
    # consuming a generator twice.
    try:
        items = list(f)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(
            f"features must be a str, Term, or iterable of str/Term; got "
            f"{type(f).__name__}: {f!r}"
        ) from exc
    if not items:
        raise ValueError("features is empty; supply at least one column or Term")
    out: list[Term] = []
    for item in items:
        term = _promote(item)
        if isinstance(term, TermSum):
            out.extend(term.terms)
        else:
            out.append(term)
    return tuple(out)
