r"""Penalty primitives.

Phase 1 ships :class:`NoPenalty` (the additive identity for the penalty
algebra). Phase 2 (AGENTS.md Task P2.A) extends this module with
:class:`L1Penalty`, :class:`L2Penalty`, :class:`PenaltySum`, and the
constructors :func:`l1` and :func:`l2`.

Per DESIGN.md §2.3, ``+`` composes penalties:

>>> from model_crafter.penalty import NoPenalty, l1, l2
>>> NoPenalty() + NoPenalty()  # doctest: +ELLIPSIS
NoPenalty()
>>> p = l1(0.1) + l2(0.2)
>>> len(p)
2

A ``Penalty`` plus a ``Term`` is a programming error and raises
``TypeError`` with a message pointing at ``features=`` vs ``penalty=``.

Math (the docstrings of the concrete penalties also state these):

* L1 (lasso, ESL §3.4.2)
  :math:`R(\beta) = \lambda \sum_j |\beta_j|`.
  Prox at step :math:`s`: soft-thresholding,
  :math:`\mathrm{prox}(\beta_j) = \mathrm{sign}(\beta_j)\,
  \max(|\beta_j| - \lambda s,\, 0)`.

* L2 (ridge, ESL §3.4.1, glmnet parameterisation)
  :math:`R(\beta) = \tfrac{\lambda}{2} \sum_j \beta_j^2`.
  Prox: shrinkage,
  :math:`\mathrm{prox}(\beta_j) = \beta_j / (1 + \lambda s)`.

* Elastic net (L1 + L2; ESL §3.4.5).
  Prox uses the closed form from Parikh & Boyd §6.5.2:
  :math:`\mathrm{prox}(\beta) = \mathrm{soft\_threshold}\!\bigl(
  \beta / (1 + \lambda_2 s),\ \lambda_1 s\bigr)`.

* ``PenaltySum`` value sums part values; prox of a generic sum is the
  composition of part proxes (right-to-left). For the L1 + L2 pair we
  detect the elastic-net structure and apply the closed form.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "L1Penalty",
    "L2Penalty",
    "NoPenalty",
    "Penalty",
    "PenaltySum",
    "l1",
    "l2",
]


# ---------------------------------------------------------------------------
# Helpers / shared error path
# ---------------------------------------------------------------------------


_TERM_REDIRECT_MSG = (
    "cannot add a Term to a Penalty — pass terms via features= and "
    "penalties via penalty= (DESIGN.md §2.3)"
)


def _is_term(other: object) -> bool:
    """Duck-typed Term test that avoids importing ``terms.base`` (would
    risk circular imports at package init)."""
    return hasattr(other, "expand") and hasattr(other, "name")


def _check_lambda(lam: float, *, kind: str) -> float:
    """Validate ``lam``: must be a finite, non-negative float."""
    lam_f = float(lam)
    if not np.isfinite(lam_f):
        raise ValueError(
            f"{kind} requires a finite lam; got {lam_f!r}"
        )
    if lam_f < 0:
        raise ValueError(
            f"{kind} requires lam >= 0 (ESL §3.4 — lambda is non-negative); "
            f"got {lam_f}"
        )
    return lam_f


def _comparable_feature_scales_tuple() -> tuple:
    """Lazy import of :class:`ComparableFeatureScales` to avoid a circular
    dependency between :mod:`model_crafter.penalty` and
    :mod:`model_crafter.assumptions` at package init."""
    from model_crafter.assumptions import ComparableFeatureScales

    return (ComparableFeatureScales(),)


def _soft_threshold(x: np.ndarray, t: float) -> np.ndarray:
    r"""Element-wise soft-thresholding:
    :math:`\mathrm{sign}(x) \cdot \max(|x| - t,\ 0)`."""
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Penalty(Protocol):
    """Minimal penalty surface (DESIGN.md §6.1).

    Every penalty exposes:

    * ``assumptions`` — a tuple of :class:`~model_crafter.assumptions.Assumption`
      instances the framework runs at solve time.
    * ``value(beta)`` — the scalar penalty :math:`R(\\beta)`.
    * ``prox(beta, step)`` — the proximal operator at the given step
      (used by Phase 2's coordinate-descent solver). The identity
      penalty :class:`NoPenalty` returns ``beta`` unchanged.
    * ``__add__(other)`` — composition; returns a :class:`PenaltySum`
      (flattened), or the non-identity operand when one side is
      :class:`NoPenalty`. ``Penalty + Term`` raises :class:`TypeError`.

    ``assumptions`` is declared as a read-only property so frozen
    dataclasses (whose fields are read-only) satisfy the protocol.
    """

    @property
    def assumptions(self) -> tuple: ...

    def value(self, beta: np.ndarray) -> float: ...

    def prox(self, beta: np.ndarray, step: float) -> np.ndarray: ...

    def __add__(self, other: object) -> Penalty: ...


# ---------------------------------------------------------------------------
# NoPenalty — the identity element (P1.A)
# ---------------------------------------------------------------------------


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

    def prox(self, beta: np.ndarray, step: float) -> np.ndarray:
        """Prox of the zero penalty is the identity map.

        Provided so callers in proximal solvers can treat ``NoPenalty`` and
        ``L1Penalty(0.0)`` symmetrically without a type switch."""
        return np.asarray(beta, dtype=float).copy()

    def __add__(self, other: object) -> Penalty:
        # Importing terms.base here would create a cycle on package import in
        # some loaders; use duck typing.
        if _is_term(other):
            raise TypeError(_TERM_REDIRECT_MSG)
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


# ---------------------------------------------------------------------------
# L1Penalty
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class L1Penalty:
    r"""Lasso (L1) penalty (ESL §3.4.2).

    .. math::

        R(\beta) \;=\; \lambda \sum_j |\beta_j|

    Proximal operator at step :math:`s` is the element-wise soft-threshold
    (ESL §3.4.2; Parikh & Boyd §6.5.2):

    .. math::

        \mathrm{prox}_{sR}(\beta)_j
            \;=\; \mathrm{sign}(\beta_j)\,
                  \max\bigl(|\beta_j| - \lambda s,\ 0\bigr).

    Declared assumption: :class:`~model_crafter.assumptions.ComparableFeatureScales`
    (SOFT) — L1 is scale-sensitive; standardise features first
    (ESL §3.4.1).
    """

    lam: float

    def __post_init__(self) -> None:
        # Frozen dataclass: write through object.__setattr__.
        validated = _check_lambda(self.lam, kind="l1")
        object.__setattr__(self, "lam", validated)

    @property
    def assumptions(self) -> tuple:
        return _comparable_feature_scales_tuple()

    def value(self, beta: np.ndarray) -> float:
        b = np.asarray(beta, dtype=float)
        return float(self.lam * np.sum(np.abs(b)))

    def prox(self, beta: np.ndarray, step: float) -> np.ndarray:
        b = np.asarray(beta, dtype=float)
        return _soft_threshold(b, self.lam * float(step))

    def __add__(self, other: object) -> Penalty:
        return _penalty_add(self, other)

    def __radd__(self, other: object) -> Penalty:
        # Only triggered when ``other`` is a non-Penalty without its own
        # __add__ implementation, e.g., ``0 + l1(0.1)`` would land here.
        # Defer to NoPenalty()'s rules for the identity-element case.
        return _penalty_add(self, other)

    def __repr__(self) -> str:
        return f"L1Penalty(lam={self.lam!r})"


# ---------------------------------------------------------------------------
# L2Penalty
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class L2Penalty:
    r"""Ridge (L2) penalty in the glmnet half-norm parameterisation
    (ESL §3.4.1).

    .. math::

        R(\beta) \;=\; \frac{\lambda}{2} \sum_j \beta_j^2

    Proximal operator at step :math:`s` is element-wise shrinkage:

    .. math::

        \mathrm{prox}_{sR}(\beta)_j \;=\; \frac{\beta_j}{1 + \lambda s}.

    The factor of one-half is the conventional ``glmnet`` parameterisation
    (Friedman et al. 2010); ESL §3.4.1's :math:`\beta^T \beta` version is
    equivalent up to a rescaling of :math:`\lambda`.

    Declared assumption: :class:`~model_crafter.assumptions.ComparableFeatureScales`
    (SOFT) — L2 is scale-sensitive; standardise features first
    (ESL §3.4.1).
    """

    lam: float

    def __post_init__(self) -> None:
        validated = _check_lambda(self.lam, kind="l2")
        object.__setattr__(self, "lam", validated)

    @property
    def assumptions(self) -> tuple:
        return _comparable_feature_scales_tuple()

    def value(self, beta: np.ndarray) -> float:
        b = np.asarray(beta, dtype=float)
        return float(0.5 * self.lam * (b @ b))

    def prox(self, beta: np.ndarray, step: float) -> np.ndarray:
        b = np.asarray(beta, dtype=float)
        return b / (1.0 + self.lam * float(step))

    def __add__(self, other: object) -> Penalty:
        return _penalty_add(self, other)

    def __radd__(self, other: object) -> Penalty:
        return _penalty_add(self, other)

    def __repr__(self) -> str:
        return f"L2Penalty(lam={self.lam!r})"


# ---------------------------------------------------------------------------
# PenaltySum
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PenaltySum:
    r"""Sum of penalty primitives (DESIGN.md §2.3).

    Math
    ----
    For parts :math:`R_1, \dots, R_k`,

    .. math::

        R(\beta) \;=\; \sum_{i=1}^{k} R_i(\beta).

    Proximal operator
    -----------------
    The proximal operator of a sum is, in general, not the sum of the
    proximal operators. For two compatible parts the prox of the sum can
    be evaluated by composing the parts' proxes in a specific order; for
    arbitrary sums, ``PenaltySum.prox`` applies the parts' proxes in
    declaration order. The composition is exact when the parts are
    *separable* in the same coordinate system (the L1/L2 case) and is the
    standard "operator splitting" approximation otherwise.

    Two special cases get the closed form (used by Phase 2's
    coordinate-descent solver, P2.C):

    * **Elastic net** (a single L1 + a single L2):
      :math:`\mathrm{prox}(\beta) = \mathrm{soft\_threshold}\!\bigl(
      \beta / (1 + \lambda_2 s),\ \lambda_1 s\bigr)`
      (Parikh & Boyd §6.5.2).

    * **Multiple L1 / L2 of the same kind**: combined into a single L1
      (sum of thresholds) or L2 (sum of lambdas) before the composition.

    Composition algebra
    -------------------
    ``+`` is flattening: ``(l1 + l2) + l1`` produces
    ``PenaltySum(parts=(l1, l2, l1))``, not a nested sum. ``+`` with a
    ``Term`` raises :class:`TypeError` pointing at ``features=`` vs
    ``penalty=`` (DESIGN.md §2.3).

    Assumptions
    -----------
    The ``assumptions`` tuple is the *deduplicated* union of part
    assumptions: equal assumption instances collapse so the framework
    doesn't run the same check twice (L1 and L2 both declare
    :class:`~model_crafter.assumptions.ComparableFeatureScales`).
    """

    parts: tuple = field()

    def __post_init__(self) -> None:
        # Defensive normalisation: coerce to tuple and validate that every
        # part exposes the minimal Penalty surface (`value` + `__add__`).
        parts = tuple(self.parts)
        for p in parts:
            if not hasattr(p, "value") or not hasattr(p, "__add__"):
                raise TypeError(
                    f"PenaltySum part {p!r} is not a Penalty "
                    "(missing value/__add__)"
                )
        # Enforce flatness: no nested PenaltySum inside parts.
        if any(isinstance(p, PenaltySum) for p in parts):
            raise ValueError(
                "PenaltySum.parts must be flat — use `+` to flatten nesting"
            )
        object.__setattr__(self, "parts", parts)

    @property
    def assumptions(self) -> tuple:
        seen: list = []
        for p in self.parts:
            for a in getattr(p, "assumptions", ()) or ():
                # Equal frozen-dataclass instances compare equal — that's
                # the dedup signal we want. Linear scan is fine for the
                # handful of assumptions a penalty declares.
                if a not in seen:
                    seen.append(a)
        return tuple(seen)

    def value(self, beta: np.ndarray) -> float:
        b = np.asarray(beta, dtype=float)
        return float(sum(p.value(b) for p in self.parts))

    def prox(self, beta: np.ndarray, step: float) -> np.ndarray:
        r"""Proximal operator of the sum.

        Strategy
        --------
        1. Combine multiple L1 (or L2) parts into one by summing lambdas.
        2. If after combining we have exactly one L1 and one L2 part, use
           the elastic-net closed form
           ``soft_threshold(beta / (1 + l2*s), l1*s)``.
        3. Otherwise compose the parts' proxes in declaration order
           (standard operator splitting).
        """
        b = np.asarray(beta, dtype=float).copy()
        step_f = float(step)

        # 1. Combine like terms.
        l1_lam = 0.0
        l2_lam = 0.0
        others: list = []
        for p in self.parts:
            if isinstance(p, L1Penalty):
                l1_lam += p.lam
            elif isinstance(p, L2Penalty):
                l2_lam += p.lam
            elif isinstance(p, NoPenalty):
                continue  # identity
            else:
                others.append(p)

        # 2. Elastic-net closed form when ``others`` is empty.
        if not others:
            if l1_lam == 0.0 and l2_lam == 0.0:
                return b
            if l1_lam == 0.0:
                return b / (1.0 + l2_lam * step_f)
            if l2_lam == 0.0:
                return _soft_threshold(b, l1_lam * step_f)
            return _soft_threshold(b / (1.0 + l2_lam * step_f), l1_lam * step_f)

        # 3. Operator splitting for arbitrary additional parts. Apply the
        # combined L1/L2 first (if any), then each additional part in
        # declaration order via its prox.
        if l2_lam > 0.0:
            b = b / (1.0 + l2_lam * step_f)
        if l1_lam > 0.0:
            b = _soft_threshold(b, l1_lam * step_f)
        for p in others:
            prox_fn = getattr(p, "prox", None)
            if prox_fn is None:
                raise TypeError(
                    f"PenaltySum.prox: part {p!r} has no `prox` method"
                )
            b = np.asarray(prox_fn(b, step_f), dtype=float)
        return b

    def __add__(self, other: object) -> Penalty:
        return _penalty_add(self, other)

    def __radd__(self, other: object) -> Penalty:
        return _penalty_add(self, other)

    def __iter__(self):
        return iter(self.parts)

    def __len__(self) -> int:
        return len(self.parts)

    def __repr__(self) -> str:
        return f"PenaltySum(parts={self.parts!r})"


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------


def l1(lam: float) -> L1Penalty:
    r"""Construct a lasso (L1) penalty :math:`R(\beta) = \lambda \sum_j |\beta_j|`.

    Parameters
    ----------
    lam
        Non-negative regularisation strength. Larger values shrink more
        coefficients to exactly zero (ESL §3.4.2).
    """
    return L1Penalty(lam=lam)


def l2(lam: float) -> L2Penalty:
    r"""Construct a ridge (L2) penalty
    :math:`R(\beta) = \tfrac{\lambda}{2} \sum_j \beta_j^2`.

    The half-norm is the glmnet parameterisation (Friedman et al. 2010);
    ESL §3.4.1 uses :math:`\beta^T \beta` which differs only by a factor
    of two in the chosen lambda.

    Parameters
    ----------
    lam
        Non-negative regularisation strength. Larger values shrink
        coefficients toward zero without forcing exact zeros (ESL §3.4.1).
    """
    return L2Penalty(lam=lam)


# ---------------------------------------------------------------------------
# Shared addition logic (flattening + identity handling + Term redirect)
# ---------------------------------------------------------------------------


def _penalty_add(left: Penalty, other: object) -> Penalty:
    """Implement ``+`` for atomic penalties and ``PenaltySum``.

    Rules (DESIGN.md §2.3):

    * ``Penalty + Term`` -> ``TypeError`` pointing at ``features=``/
      ``penalty=``.
    * ``Penalty + NoPenalty`` -> ``Penalty`` (identity).
    * ``Penalty + Penalty`` -> ``PenaltySum`` with flattened parts.
    """
    if _is_term(other):
        raise TypeError(_TERM_REDIRECT_MSG)
    if isinstance(other, NoPenalty):
        return left
    if not isinstance(other, Penalty):
        raise TypeError(
            f"unsupported operand for Penalty +: {type(other).__name__}"
        )

    left_parts = (
        tuple(left.parts) if isinstance(left, PenaltySum) else (left,)
    )
    right_parts = (
        tuple(other.parts) if isinstance(other, PenaltySum) else (other,)
    )
    return PenaltySum(parts=left_parts + right_parts)
