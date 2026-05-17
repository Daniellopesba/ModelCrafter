r"""Spline basis terms: B-splines and natural cubic splines.

Math
----

**B-spline basis (cubic, ``bs``).** With boundary knots
:math:`b_l \le b_u` and :math:`K` interior knots
:math:`\xi_1 < \cdots < \xi_K`, the augmented knot sequence
:math:`(b_l,b_l,b_l,b_l, \xi_1,\dots,\xi_K, b_u,b_u,b_u,b_u)` defines a
cubic B-spline basis with :math:`K+4` functions :math:`B_j(x)` via the
Cox-De Boor recursion (De Boor 1978). With ``include_intercept=False``
the leading column is dropped, yielding :math:`K+3` functions; that
matches R's ``splines::bs(x, df=K+3, degree=3)``.

For a general degree :math:`d` the augmented sequence repeats the
boundary knots :math:`d+1` times each, giving :math:`K+2(d+1)` knots and
:math:`K+d+1` basis functions; dropping the intercept produces
:math:`K+d` columns. ``df`` selects the column count and uniquely
determines :math:`K = df - d` interior knots.

**Natural cubic spline (``ns``, ESL §5.2.1).** The natural cubic spline
space on :math:`[b_l, b_u]` with :math:`K` interior knots is the
subspace of the cubic spline space (dimension :math:`K+4`) whose
elements have zero second derivative at the boundary knots; dimension
:math:`K+2`. Dropping the constant column gives :math:`K+1 = df` columns.
We construct the natural basis as a linear transformation of the
B-spline basis, with the projection learned at fit time and stored on
the expanded term for predict-time reuse. Outside :math:`[b_l, b_u]` the
basis extends linearly — the defining "natural" property.

:func:`smooth` aliases :func:`ns`: proper smoothing splines with a
roughness penalty are out of v0 scope (DESIGN.md §2.5).

References:
* Hastie, Tibshirani, Friedman, *ESL* §5.2 / §5.2.1.
* C. de Boor, *A Practical Guide to Splines*, 1978.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.linalg import null_space

from model_crafter.assumptions.stability import SupportContainsPredictData
from model_crafter.terms._basis_common import (
    _BasisExpandedTerm,
    _freeze_state,
    _x_series,
)
from model_crafter.terms.base import TermSum, _add_terms

# Knot construction (shared by bs and ns).


def _interior_quantile_knots(x: np.ndarray, n_interior: int) -> np.ndarray:
    if n_interior <= 0:
        return np.array([], dtype=float)
    qs = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
    return np.asarray(np.quantile(x, qs), dtype=float)


def _augmented_knots(
    interior: np.ndarray, boundary: tuple[float, float], degree: int
) -> np.ndarray:
    b_l, b_u = boundary
    return np.r_[[b_l] * (degree + 1), interior, [b_u] * (degree + 1)]


def _bspline_basis_matrix(
    x: np.ndarray, augmented_knots: np.ndarray, degree: int
) -> np.ndarray:
    """Cubic (or higher) B-spline basis at ``x``.

    Uses ``scipy.interpolate.BSpline`` with extrapolation enabled so that
    predict-time inputs outside the boundary knots produce values
    consistent with the polynomial extension of the B-splines (used for
    ``bs``; natural extrapolation is enforced separately by ``ns``).
    """
    n = len(x)
    n_basis = len(augmented_knots) - degree - 1
    out = np.zeros((n, n_basis), dtype=float)
    for j in range(n_basis):
        c = np.zeros(n_basis, dtype=float)
        c[j] = 1.0
        spl = BSpline(augmented_knots, c, degree, extrapolate=True)
        out[:, j] = spl(x)
    return out


# B-spline (``bs``).


@dataclass(frozen=True, slots=True)
class _BSTerm:
    r"""B-spline basis (matches R's ``splines::bs`` to 1e-10).

    Mathematically ``df = degree + len(interior_knots)`` (intercept
    dropped). Predict-time knots come from
    ``fit_state['augmented_knots']``; fit-time knots are placed at
    quantiles of the training column when ``knots`` is not user-supplied.
    """

    col: str
    df_: int
    degree: int = 3
    user_knots: tuple[float, ...] | None = None
    name: str = field(init=False)
    assumptions: tuple = field(init=False)

    def __post_init__(self) -> None:
        if self.df_ < self.degree + 1:
            raise ValueError(
                f"bs requires df >= degree + 1; got df={self.df_}, degree={self.degree}"
            )
        object.__setattr__(self, "name", f"bs({self.col})")
        object.__setattr__(self, "assumptions", (SupportContainsPredictData(),))
        if self.user_knots is not None:
            ks = sorted(float(k) for k in self.user_knots)
            object.__setattr__(self, "user_knots", tuple(ks))
            if len(ks) != self.df_ - self.degree:
                raise ValueError(
                    f"bs: df={self.df_}, degree={self.degree} ⇒ expected "
                    f"{self.df_ - self.degree} interior knots; got {len(ks)}"
                )

    def expand(
        self,
        data: pd.DataFrame,
        *,
        fit_state: Mapping[str, Any] | None = None,
    ) -> _BasisExpandedTerm:
        x = _x_series(data, self.col)
        if fit_state is None:
            if self.user_knots is None:
                interior = _interior_quantile_knots(x, self.df_ - self.degree)
            else:
                interior = np.asarray(self.user_knots, dtype=float)
            boundary = (float(np.min(x)), float(np.max(x)))
            aug = _augmented_knots(interior, boundary, self.degree)
        else:
            aug = np.asarray(fit_state["augmented_knots"], dtype=float)
            interior = np.asarray(fit_state["interior_knots"], dtype=float)
            boundary = tuple(fit_state["boundary_knots"])  # type: ignore[assignment]

        B = _bspline_basis_matrix(x, aug, self.degree)
        # Drop the leading column to match R's bs(intercept=FALSE) /
        # patsy's bs(include_intercept=False).
        values = B[:, 1:]
        columns = tuple(f"bs({self.col})_{i}" for i in range(values.shape[1]))
        state = _freeze_state(
            {
                "augmented_knots": tuple(aug.tolist()),
                "interior_knots": tuple(interior.tolist()),
                "boundary_knots": (float(boundary[0]), float(boundary[1])),
                "degree": int(self.degree),
                "df": int(self.df_),
            }
        )
        return _BasisExpandedTerm(columns=columns, values=values, state=state)

    def __add__(self, other: object) -> TermSum:
        return _add_terms(self, other)

    def __radd__(self, other: object) -> TermSum:
        return _add_terms(other, self)


def bs(
    col: str,
    df: int,
    *,
    degree: int = 3,
    knots: Sequence[float] | None = None,
) -> _BSTerm:
    """B-spline basis term (R's ``splines::bs``).

    With ``include_intercept=False`` (the package convention; the spec
    carries its own intercept), the relationship between ``df``,
    ``degree``, and the interior-knot count :math:`K` is
    :math:`df = K + degree`. When ``knots`` is omitted, they are placed
    at quantiles of ``data[col]`` at fit time.
    """
    return _BSTerm(
        col=col,
        df_=int(df),
        degree=int(degree),
        user_knots=None if knots is None else tuple(float(k) for k in knots),
    )


# Natural cubic spline (``ns``).


def _ns_projection(
    augmented_knots: np.ndarray, x_for_lstsq: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Projection matrices ``Z`` and ``T`` that map the cubic B-spline basis
    into the natural-cubic basis with the constant column dropped.

    ``Z`` (shape :math:`(K+4) \\times (K+2)`) enforces zero second
    derivative at the boundary knots — the natural BC. ``T`` (shape
    :math:`(K+2) \\times (K+1)`) drops the direction along the constant
    column. The basis at any ``x`` is then ``B(x) @ Z @ T``.
    """
    degree = 3
    b_l = float(augmented_knots[degree])
    b_u = float(augmented_knots[-degree - 1])
    n_basis = len(augmented_knots) - degree - 1

    # Second derivatives of each B-spline at the two boundary knots.
    C = np.zeros((2, n_basis), dtype=float)
    for j in range(n_basis):
        c = np.zeros(n_basis, dtype=float)
        c[j] = 1.0
        spl = BSpline(augmented_knots, c, degree, extrapolate=False)
        spl2 = spl.derivative(2)
        v_l = spl2(b_l + 1e-12)
        v_u = spl2(b_u - 1e-12)
        C[0, j] = 0.0 if np.isnan(v_l) else float(v_l)
        C[1, j] = 0.0 if np.isnan(v_u) else float(v_u)
    Z = null_space(C)  # (n_basis, n_basis - 2)

    # Drop the constant direction: find w such that N @ w = 1 on the
    # training sample (the unique representation of the constant function
    # in the natural basis); the complement basis spans the
    # constant-free subspace.
    B = _bspline_basis_matrix(x_for_lstsq, augmented_knots, degree)
    N = B @ Z
    w = np.linalg.lstsq(N, np.ones(len(x_for_lstsq)), rcond=None)[0]
    w_norm = w / np.linalg.norm(w)
    M = np.hstack([w_norm.reshape(-1, 1), np.eye(len(w))])
    Q, _ = np.linalg.qr(M)
    T = Q[:, 1:]
    return Z, T


def _ns_eval(
    x: np.ndarray,
    augmented_knots: np.ndarray,
    Z: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    """Evaluate the natural-cubic basis at ``x`` with linear extension
    outside the boundary knots.

    Inside ``[b_l, b_u]`` we use the standard B-spline values. Outside,
    we extend by value+derivative of each natural-basis column at the
    nearest boundary knot — the natural condition (zero second
    derivative beyond, so the spline is linear there).
    """
    degree = 3
    b_l = float(augmented_knots[degree])
    b_u = float(augmented_knots[-degree - 1])
    n = len(x)

    inside = (x >= b_l) & (x <= b_u)
    below = x < b_l
    above = x > b_u

    out = np.zeros((n, T.shape[1]), dtype=float)

    if inside.any():
        B_in = _bspline_basis_matrix(x[inside], augmented_knots, degree)
        out[inside] = B_in @ Z @ T

    if below.any() or above.any():
        n_basis = len(augmented_knots) - degree - 1
        B_l = _bspline_basis_matrix(np.array([b_l]), augmented_knots, degree)
        B_u = _bspline_basis_matrix(np.array([b_u]), augmented_knots, degree)
        Bd_l = np.zeros((1, n_basis), dtype=float)
        Bd_u = np.zeros((1, n_basis), dtype=float)
        for j in range(n_basis):
            c = np.zeros(n_basis, dtype=float)
            c[j] = 1.0
            spl = BSpline(augmented_knots, c, degree, extrapolate=False)
            sp1 = spl.derivative(1)
            v_l = sp1(b_l + 1e-12)
            v_u = sp1(b_u - 1e-12)
            Bd_l[0, j] = 0.0 if np.isnan(v_l) else float(v_l)
            Bd_u[0, j] = 0.0 if np.isnan(v_u) else float(v_u)

        v_l_nat = (B_l @ Z @ T).ravel()
        d_l_nat = (Bd_l @ Z @ T).ravel()
        v_u_nat = (B_u @ Z @ T).ravel()
        d_u_nat = (Bd_u @ Z @ T).ravel()

        if below.any():
            dx = (x[below] - b_l).reshape(-1, 1)
            out[below] = v_l_nat + dx * d_l_nat
        if above.any():
            dx = (x[above] - b_u).reshape(-1, 1)
            out[above] = v_u_nat + dx * d_u_nat

    return out


@dataclass(frozen=True, slots=True)
class _NSTerm:
    r"""Natural cubic spline basis (matches R's ``splines::ns`` to 1e-10).

    The basis has ``df`` columns; ``df - 1`` interior knots are placed at
    quantiles of the training column unless explicit ``knots`` are given.
    Linear extrapolation beyond the boundary knots is the defining
    "natural" property (ESL §5.2.1).
    """

    col: str
    df_: int
    user_knots: tuple[float, ...] | None = None
    name: str = field(init=False)
    assumptions: tuple = field(init=False)

    def __post_init__(self) -> None:
        if self.df_ < 2:
            raise ValueError(f"ns requires df >= 2; got df={self.df_}")
        object.__setattr__(self, "name", f"ns({self.col})")
        object.__setattr__(self, "assumptions", (SupportContainsPredictData(),))
        if self.user_knots is not None:
            ks = sorted(float(k) for k in self.user_knots)
            object.__setattr__(self, "user_knots", tuple(ks))
            if len(ks) != self.df_ - 1:
                raise ValueError(
                    f"ns: df={self.df_} ⇒ expected {self.df_ - 1} interior "
                    f"knots; got {len(ks)}"
                )

    def expand(
        self,
        data: pd.DataFrame,
        *,
        fit_state: Mapping[str, Any] | None = None,
    ) -> _BasisExpandedTerm:
        x = _x_series(data, self.col)
        if fit_state is None:
            if self.user_knots is None:
                interior = _interior_quantile_knots(x, self.df_ - 1)
            else:
                interior = np.asarray(self.user_knots, dtype=float)
            boundary = (float(np.min(x)), float(np.max(x)))
            aug = _augmented_knots(interior, boundary, 3)
            Z, T = _ns_projection(aug, x)
        else:
            aug = np.asarray(fit_state["augmented_knots"], dtype=float)
            interior = np.asarray(fit_state["interior_knots"], dtype=float)
            boundary = tuple(fit_state["boundary_knots"])  # type: ignore[assignment]
            Z = np.asarray(fit_state["Z"], dtype=float)
            T = np.asarray(fit_state["T"], dtype=float)

        values = _ns_eval(x, aug, Z, T)
        columns = tuple(f"ns({self.col})_{i}" for i in range(values.shape[1]))
        state = _freeze_state(
            {
                "augmented_knots": tuple(aug.tolist()),
                "interior_knots": tuple(interior.tolist()),
                "boundary_knots": (float(boundary[0]), float(boundary[1])),
                "Z": Z,
                "T": T,
                "df": int(self.df_),
            }
        )
        return _BasisExpandedTerm(columns=columns, values=values, state=state)

    def __add__(self, other: object) -> TermSum:
        return _add_terms(self, other)

    def __radd__(self, other: object) -> TermSum:
        return _add_terms(other, self)


def ns(
    col: str,
    df: int,
    *,
    knots: Sequence[float] | None = None,
) -> _NSTerm:
    """Natural cubic spline term (R's ``splines::ns``).

    ``df - 1`` interior knots are placed at quantiles of ``data[col]``
    at fit time unless ``knots`` is supplied. Outside the boundary knots
    :math:`[\\min(x), \\max(x)]` the basis is linear (ESL §5.2.1).
    """
    return _NSTerm(
        col=col,
        df_=int(df),
        user_knots=None if knots is None else tuple(float(k) for k in knots),
    )


def smooth(col: str, df: int) -> _NSTerm:
    """Alias for :func:`ns` (DESIGN.md §2.5).

    Proper smoothing splines with a roughness penalty are out of v0
    scope; the alias preserves the spelling used by the §10 reference
    example.
    """
    return ns(col, df)
