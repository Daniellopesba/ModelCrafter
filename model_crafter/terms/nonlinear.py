r"""Non-spline basis terms: orthogonal polynomial, step, and MARS hinges.

Math
----

**Orthogonal polynomial (``poly``).** Centre :math:`x` at its mean,
form the Vandermonde :math:`V_{ij} = (x_i - \bar x)^j` for
:math:`j = 0, \dots, d`, take the QR factorisation :math:`V = QR`, and
return :math:`Q_{:, 1:}` (degree :math:`d` orthogonal polynomials with
the constant dropped). The fit-time state stores :math:`\bar x` and
:math:`R` so predict-time produces aligned columns from the same linear
combinations of monomials. Matches R's ``poly()`` to 1e-12.

**Step (``step``).** Given sorted breakpoints
:math:`b_1 < \cdots < b_k`, define :math:`k+1` bins
:math:`(-\infty, b_1], (b_1, b_2], \dots, (b_k, \infty)`. The basis is
:math:`k` indicator columns; the first bin is the reference (dropped
for identifiability when an intercept is present).

**Hinge (``hinge``, ESL §9.4.1).** For knot :math:`t` and
direction "left": :math:`(t - x)_+`; "right": :math:`(x - t)_+`. One
column. Stateless beyond the fit-time boundary record (used by the
support assumption).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from model_crafter.assumptions.stability import SupportContainsPredictData
from model_crafter.terms._basis_common import (
    _BasisExpandedTerm,
    _freeze_state,
    _x_series,
)
from model_crafter.terms.base import TermSum, _add_terms

__all__ = ["hinge", "poly", "step"]


# Orthogonal polynomial.


@dataclass(frozen=True, slots=True)
class _PolyTerm:
    r"""Orthogonal polynomial basis (matches R's ``poly()`` to 1e-12).

    Algorithm:

    1. Centre :math:`x` at its mean :math:`\bar x`.
    2. Form the Vandermonde with :math:`V_{ij} = (x_i - \bar x)^j`.
    3. QR decompose: :math:`V = QR`.
    4. Return :math:`Q_{:, 1:d+1}` (constant dropped).

    Predict time re-forms :math:`V` against the *training* mean and
    multiplies by :math:`R^{-1}` so the resulting columns lie on exactly
    the same orthogonal-polynomial axes.
    """

    col: str
    degree: int
    name: str = field(init=False)
    assumptions: tuple = field(init=False)

    def __post_init__(self) -> None:
        if self.degree < 1:
            raise ValueError(f"poly requires degree >= 1; got {self.degree}")
        object.__setattr__(self, "name", f"poly({self.col})")
        object.__setattr__(self, "assumptions", (SupportContainsPredictData(),))

    def expand(
        self,
        data: pd.DataFrame,
        *,
        fit_state: Mapping[str, Any] | None = None,
    ) -> _BasisExpandedTerm:
        from scipy.linalg import solve_triangular

        x = _x_series(data, self.col)
        if fit_state is None:
            xbar = float(np.mean(x))
            xc = x - xbar
            V = np.column_stack([xc**d for d in range(self.degree + 1)])
            Q, R = np.linalg.qr(V)
            values = Q[:, 1:]
        else:
            xbar = float(fit_state["xbar"])
            R = np.asarray(fit_state["R"], dtype=float)
            xc = x - xbar
            V = np.column_stack([xc**d for d in range(self.degree + 1)])
            # V = Q R  ⇒  Q = V @ R^{-1}; with R upper-triangular,
            # solve_triangular(R.T, V.T, lower=True).T computes V @ R^{-1}.
            Q = solve_triangular(R.T, V.T, lower=True).T
            values = Q[:, 1:]

        columns = tuple(f"poly({self.col})_{i + 1}" for i in range(values.shape[1]))
        state = _freeze_state(
            {
                "xbar": xbar,
                "R": R,
                "degree": int(self.degree),
            }
        )
        return _BasisExpandedTerm(columns=columns, values=values, state=state)

    def __add__(self, other: object) -> TermSum:
        return _add_terms(self, other)

    def __radd__(self, other: object) -> TermSum:
        return _add_terms(other, self)


def poly(col: str, degree: int) -> _PolyTerm:
    """Orthogonal polynomial basis term (R's ``poly()``).

    Produces ``degree`` columns. The basis is *orthogonal* — even at
    high degree the design matrix stays well-conditioned because the
    columns are constructed from the QR factorisation of the centred
    Vandermonde rather than the naive monomial basis.
    """
    return _PolyTerm(col=col, degree=int(degree))


# Step.


@dataclass(frozen=True, slots=True)
class _StepTerm:
    r"""Piecewise-constant basis.

    Given sorted breakpoints :math:`b_1 < \cdots < b_k`, the basis has
    :math:`k` indicator columns marking membership in
    :math:`(b_1, b_2], \dots, (b_k, \infty)`. The leading bin
    :math:`(-\infty, b_1]` is the reference, dropped for
    identifiability when an intercept is present.
    """

    col: str
    breakpoints: tuple[float, ...]
    name: str = field(init=False)
    assumptions: tuple = field(init=False)

    def __post_init__(self) -> None:
        bps = tuple(float(b) for b in self.breakpoints)
        if len(bps) == 0:
            raise ValueError("step requires at least one breakpoint")
        if any(bps[i] >= bps[i + 1] for i in range(len(bps) - 1)):
            raise ValueError(
                f"step breakpoints must be strictly sorted; got {bps}"
            )
        object.__setattr__(self, "breakpoints", bps)
        object.__setattr__(self, "name", f"step({self.col})")
        object.__setattr__(self, "assumptions", (SupportContainsPredictData(),))

    def expand(
        self,
        data: pd.DataFrame,
        *,
        fit_state: Mapping[str, Any] | None = None,
    ) -> _BasisExpandedTerm:
        x = _x_series(data, self.col)
        bps = np.asarray(self.breakpoints, dtype=float)
        # searchsorted side='left' returns smallest i with bps[i] >= x.
        # That gives 0 for x <= b_1, 1 for b_1 < x <= b_2, ..., k for x > b_k —
        # exactly the bin numbering we want.
        bin_idx = np.searchsorted(bps, x, side="left")
        n_bins = len(bps) + 1
        n = len(x)
        full = np.zeros((n, n_bins), dtype=float)
        full[np.arange(n), bin_idx] = 1.0
        # Drop bin 0 (reference).
        values = full[:, 1:]
        columns = tuple(f"step({self.col})_bin{i}" for i in range(1, n_bins))
        # boundary_knots reflect training min/max so SupportContainsPredictData
        # treats predict-time extrapolation the same as for splines.
        state = _freeze_state(
            {
                "breakpoints": bps.tolist(),
                "boundary_knots": (float(np.min(x)), float(np.max(x))),
            }
        )
        return _BasisExpandedTerm(columns=columns, values=values, state=state)

    def __add__(self, other: object) -> TermSum:
        return _add_terms(self, other)

    def __radd__(self, other: object) -> TermSum:
        return _add_terms(other, self)


def step(col: str, breakpoints: Sequence[float]) -> _StepTerm:
    """Piecewise-constant basis term.

    ``breakpoints`` must be strictly increasing. Produces
    ``len(breakpoints)`` indicator columns; the bin
    :math:`(-\\infty, b_1]` is the reference (dropped for
    identifiability).
    """
    return _StepTerm(col=col, breakpoints=tuple(float(b) for b in breakpoints))


# Hinge (MARS).


@dataclass(frozen=True, slots=True)
class _HingeTerm:
    r"""MARS-style hinge (ESL §9.4.1).

    One column:

    * ``direction='left'``  → :math:`(t - x)_+`
    * ``direction='right'`` → :math:`(x - t)_+`

    The knot ``t`` is fixed at construction; no fit-time learning is
    required.
    """

    col: str
    knot: float
    direction: str
    name: str = field(init=False)
    assumptions: tuple = field(init=False)

    def __post_init__(self) -> None:
        if self.direction not in ("left", "right"):
            raise ValueError(
                f"hinge direction must be 'left' or 'right'; got {self.direction!r}"
            )
        object.__setattr__(self, "name", f"hinge({self.col}@{self.knot:g}:{self.direction})")
        object.__setattr__(self, "assumptions", (SupportContainsPredictData(),))

    def expand(
        self,
        data: pd.DataFrame,
        *,
        fit_state: Mapping[str, Any] | None = None,
    ) -> _BasisExpandedTerm:
        x = _x_series(data, self.col)
        if self.direction == "left":
            values = np.maximum(0.0, self.knot - x).reshape(-1, 1)
        else:
            values = np.maximum(0.0, x - self.knot).reshape(-1, 1)
        columns = (self.name,)
        if fit_state is None:
            boundary = (float(np.min(x)), float(np.max(x)))
        else:
            boundary = tuple(fit_state["boundary_knots"])  # type: ignore[assignment]
        state = _freeze_state(
            {
                "boundary_knots": (float(boundary[0]), float(boundary[1])),
                "knot": float(self.knot),
                "direction": self.direction,
            }
        )
        return _BasisExpandedTerm(columns=columns, values=values, state=state)

    def __add__(self, other: object) -> TermSum:
        return _add_terms(self, other)

    def __radd__(self, other: object) -> TermSum:
        return _add_terms(other, self)


def hinge(col: str, knot: float, direction: str) -> _HingeTerm:
    """MARS-style hinge term (ESL §9.4.1).

    ``direction='left'`` produces :math:`(t - x)_+`; ``'right'``
    produces :math:`(x - t)_+`.
    """
    return _HingeTerm(col=col, knot=float(knot), direction=direction)
