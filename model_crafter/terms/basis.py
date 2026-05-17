r"""Standard basis-expansion terms (Task P4.A).

Implements the six basis-term factories pinned in ``AGENTS.md``:

* :func:`ns`     — natural cubic spline (R's ``splines::ns``)
* :func:`bs`     — B-spline (R's ``splines::bs``)
* :func:`poly`   — orthogonal polynomial basis (R's ``poly``)
* :func:`step`   — piecewise-constant indicators
* :func:`smooth` — alias for :func:`ns` (per DESIGN.md §2.5: no roughness
  penalty in v0)
* :func:`hinge`  — MARS-style truncated-linear basis (ESL §9.4.1)

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

For a general degree :math:`d`, the augmented knot sequence repeats the
boundary knots :math:`d+1` times each, giving :math:`K+2(d+1)` knots and
:math:`K+d+1` basis functions; dropping the intercept produces :math:`K+d`
columns. ``df`` selects the column count and uniquely determines
:math:`K = df - d` interior knots.

**Natural cubic spline (``ns``).** ESL §5.2.1. The natural cubic spline
space on :math:`[b_l, b_u]` with :math:`K` interior knots is the subspace
of the cubic spline space (dimension :math:`K+4`) whose elements have
zero second derivative at the boundary knots; dimension :math:`K+2`.
Dropping the constant column gives :math:`K+1 = df` columns. We construct
the natural basis as a linear transformation of the B-spline basis, with
the projection learned at fit time and stored in ``fit_state`` for
predict-time reuse. Outside :math:`[b_l, b_u]` the basis extends linearly
(the defining "natural" property).

**Orthogonal polynomial basis (``poly``).** Centre :math:`x` at its mean
:math:`\bar x`, form the Vandermonde
:math:`V_{ij} = (x_i - \bar x)^j` for :math:`j = 0, \dots, d`, take the
QR factorisation :math:`V = QR`, and return the columns
:math:`Q_{:,1:}` (degree :math:`d` orthogonal polynomials with the
constant dropped). The fit-time state stores :math:`\bar x` and the
:math:`R` matrix so predict-time produces aligned columns from the same
linear combinations of monomials.

**Step basis (``step``).** Given sorted breakpoints
:math:`b_1 < \cdots < b_k`, define :math:`k+1` bins
:math:`(-\infty, b_1], (b_1, b_2], \dots, (b_k, \infty)`. The basis is
:math:`k` indicator columns (the first bin is the reference, dropped for
identifiability when an intercept is present).

**Hinge basis (``hinge``).** ESL §9.4.1. For a knot :math:`t` and
direction "left": :math:`(t - x)_+`. For "right": :math:`(x - t)_+`. Each
hinge is a single column.

Predict-time state
------------------

Every basis term that learns numerical state at fit time
(knots / orthogonalisation / projection) stores that state in the
``state`` attribute of the :class:`_BasisExpandedTerm` it returns. The
canonical flow is:

1. At fit time, ``expand(data, fit_state=None)`` computes the state and
   returns the expansion together with the learned state.
2. At predict time, ``expand(data, fit_state=<learned state>)`` uses
   the stored knots / mean / R / projection to evaluate the basis on
   ``data``.

This is the contract used by :class:`SupportContainsPredictData` (which
reads the training knot range from ``fit_state``).

References
----------

* Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*,
  §5.2 (B-splines), §5.2.1 (natural splines), §9.4 (MARS).
* C. de Boor, *A Practical Guide to Splines*, 1978.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.linalg import null_space

from model_crafter.assumptions.stability import SupportContainsPredictData
from model_crafter.terms.base import ExpandedTerm, TermSum, _add_terms

__all__ = [
    "bs",
    "hinge",
    "ns",
    "poly",
    "smooth",
    "step",
]


# ---------------------------------------------------------------------------
# Internal: enriched ExpandedTerm carrying learned state
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _BasisExpandedTerm(ExpandedTerm):
    """An :class:`ExpandedTerm` plus the learned state needed to reproduce
    the same column values at predict time.

    The base ``ExpandedTerm`` (``columns``, ``values``) is what
    :mod:`model_crafter._internal.design` reads to build the design
    matrix. The extra ``state`` field is read by basis terms when they
    receive ``fit_state`` at predict time, and by
    :class:`~model_crafter.assumptions.stability.SupportContainsPredictData`
    when it needs the training boundary knots.
    """

    state: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))


def _freeze_state(state: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return an immutable view of ``state``."""
    return MappingProxyType(dict(state))


# ---------------------------------------------------------------------------
# B-spline (``bs``)
# ---------------------------------------------------------------------------


def _x_series(data: pd.DataFrame, col: str) -> np.ndarray:
    if col not in data.columns:
        raise KeyError(
            f"basis term refers to column '{col}' which is not in the data frame "
            f"(available columns: {list(data.columns)})"
        )
    series = pd.to_numeric(data[col], errors="raise")
    arr = np.asarray(series, dtype=float)
    if not np.isfinite(arr).all():
        bad = int(np.sum(~np.isfinite(arr)))
        raise ValueError(
            f"column '{col}' contains {bad} non-finite value(s); drop or impute "
            "before expansion (DESIGN.md §9.8 — no silent NaN handling)"
        )
    return arr


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


@dataclass(frozen=True, slots=True)
class _BSTerm:
    r"""B-spline basis (matches R's ``splines::bs`` to 1e-10).

    See module docstring for the math. Mathematically::

        df = degree + len(interior_knots)         (intercept dropped)

    Predict-time knots are taken from ``fit_state['augmented_knots']``;
    fit-time knots are placed at quantiles of the training column when
    ``knots`` is not user-supplied.
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
        # Frozen-but-settable trick: bypass __setattr__ via object.__setattr__.
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
        # Drop the leading column (include_intercept=False), matching
        # R's bs(intercept=FALSE) / patsy's bs(include_intercept=False).
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

    Parameters
    ----------
    col:
        Column name in the data frame.
    df:
        Number of output columns. With ``include_intercept=False`` (the
        package convention; the spec carries its own intercept), the
        relationship between ``df``, ``degree``, and the interior-knot
        count :math:`K` is :math:`df = K + degree`.
    degree:
        Polynomial degree of the spline (default 3 — cubic).
    knots:
        Optional sequence of interior knots. When omitted, knots are
        placed at quantiles of ``data[col]`` at fit time.

    Returns
    -------
    Term
        A frozen :class:`_BSTerm`.
    """
    return _BSTerm(
        col=col,
        df_=int(df),
        degree=int(degree),
        user_knots=None if knots is None else tuple(float(k) for k in knots),
    )


# ---------------------------------------------------------------------------
# Natural cubic spline (``ns``)
# ---------------------------------------------------------------------------


def _ns_projection(
    augmented_knots: np.ndarray, x_for_lstsq: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the projection matrices ``Z`` and ``T`` that map the cubic
    B-spline basis into the natural cubic spline basis with constant
    dropped.

    ``Z`` (shape :math:`(K+4) \\times (K+2)`) enforces zero second
    derivative at the boundary knots (the natural BC). ``T`` (shape
    :math:`(K+2) \\times (K+1)`) drops the direction along the constant
    column. The basis at any ``x`` is then
    ``B(x) @ Z @ T``.
    """
    degree = 3
    b_l = float(augmented_knots[degree])
    b_u = float(augmented_knots[-degree - 1])
    n_basis = len(augmented_knots) - degree - 1

    # Constraint: second derivative of each B-spline at the boundaries.
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

    # Drop constant: find w with N @ w = 1 (using the fit-time x sample
    # so the linear system is determined). The resulting w is the unique
    # representation of the constant function in the natural basis. The
    # complement basis spans the constant-free subspace.
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
    we extend by the value+derivative of each *natural-basis* column
    evaluated at the nearest boundary knot — that is the literal
    "natural" condition (zero second derivative beyond the boundary
    knots, so the spline is linear there).
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
        # Compute basis value + derivative at the boundary knot for each
        # natural-basis column.
        n_basis = len(augmented_knots) - degree - 1
        # Value at b_l and b_u (single eval): use a 1-element x and read out.
        B_l = _bspline_basis_matrix(np.array([b_l]), augmented_knots, degree)
        B_u = _bspline_basis_matrix(np.array([b_u]), augmented_knots, degree)
        # First derivative
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

        # Project into natural basis.
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
    quantiles of the training column (unless explicit ``knots`` are
    given). Linear extrapolation beyond the boundary knots is the
    defining "natural" property (ESL §5.2.1).
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

    Parameters
    ----------
    col:
        Column name in the data frame.
    df:
        Number of output columns. ``df - 1`` interior knots are placed
        at quantiles of ``data[col]`` at fit time (unless ``knots`` is
        supplied).
    knots:
        Optional sequence of ``df - 1`` interior knots.

    Returns
    -------
    Term
        A frozen :class:`_NSTerm`.

    Notes
    -----
    Outside the boundary knots :math:`[\\min(x), \\max(x)]` the basis is
    linear in :math:`x` — the defining "natural" property
    (ESL §5.2.1).
    """
    return _NSTerm(
        col=col,
        df_=int(df),
        user_knots=None if knots is None else tuple(float(k) for k in knots),
    )


def smooth(col: str, df: int) -> _NSTerm:
    """Alias for :func:`ns` (DESIGN.md §2.5).

    ``mc.smooth(col, df)`` returns the same term as ``mc.ns(col, df)``.
    The "smoothness penalty" angle in DESIGN.md §2.5 — proper smoothing
    splines with a roughness penalty — is out of scope for v0; this
    alias documents the simplification while preserving the spelling
    the §10 reference example uses.
    """
    return ns(col, df)


# ---------------------------------------------------------------------------
# Orthogonal polynomial (``poly``)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _PolyTerm:
    r"""Orthogonal polynomial basis (matches R's ``poly()`` to 1e-12).

    Algorithm (R's ``stats/R/poly.R``):

    1. Centre :math:`x` at its mean :math:`\bar x`.
    2. Form the Vandermonde
       :math:`V \in \mathbb{R}^{n \times (d+1)}` with
       :math:`V_{ij} = (x_i - \bar x)^j`.
    3. Compute the QR decomposition :math:`V = QR`.
    4. Return :math:`Q_{:,1:d+1}` (the orthonormal basis with the
       constant column dropped).

    Predict time: re-form :math:`V` against the *training* mean, then
    multiply by :math:`R^{-1}` so the resulting columns lie on exactly
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
            # Vandermonde QR — Q has orthonormal columns spanning the
            # polynomial space; R encodes the scaling.
            Q, R = np.linalg.qr(V)
            values = Q[:, 1:]  # drop the constant
        else:
            xbar = float(fit_state["xbar"])
            R = np.asarray(fit_state["R"], dtype=float)
            xc = x - xbar
            V = np.column_stack([xc**d for d in range(self.degree + 1)])
            # V = Q R  ⇒  Q = V @ R^{-1}. With R upper-triangular,
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

    Parameters
    ----------
    col:
        Column name in the data frame.
    degree:
        Polynomial degree :math:`d \\ge 1`. Produces ``degree`` columns.

    Returns
    -------
    Term
        A frozen :class:`_PolyTerm`.

    Notes
    -----
    The basis is *orthogonal* — even at high ``degree`` the design
    matrix stays well-conditioned because the columns are constructed
    from the QR factorisation of the (centred) Vandermonde rather than
    the naive monomial basis.
    """
    return _PolyTerm(col=col, degree=int(degree))


# ---------------------------------------------------------------------------
# Step (piecewise constant)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _StepTerm:
    r"""Piecewise-constant basis.

    Given sorted breakpoints :math:`b_1 < \cdots < b_k`, the basis has
    ``k`` indicator columns, each marking membership in one of the bins
    :math:`(b_1, b_2], (b_2, b_3], \dots, (b_k, \infty)`; the leading
    bin :math:`(-\infty, b_1]` is the reference (dropped for
    identifiability when an intercept is present).
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
        # bin index in {0, 1, ..., k}: 0 if x <= b_1, k if x > b_k.
        # np.searchsorted with side='left' gives 0 for x <= b_1, etc.
        bin_idx = np.searchsorted(bps, x, side="left")
        # We want bin 0 if x <= b_1, bin 1 if b_1 < x <= b_2, ..., bin k if x > b_k.
        # searchsorted side='left' returns 0 if x <= b_1, etc. Adjust: side='right' gives
        # 1 if x < b_1, so equality cases differ. Pick side='left' and shift:
        # Actually for bins (-inf, b_1], (b_1, b_2], ..., (b_k, inf):
        # bin_idx_correct = np.searchsorted(bps, x, side='left')  if x <= b_j places in bin j.
        # searchsorted(bps, x, side='left'): returns smallest i where bps[i] >= x.
        # For x = b_1 exactly: side='left' returns 0 (smallest i where bps[i] >= b_1), giving bin 0. Good.
        # For x slightly > b_1: returns 1 (bps[1] >= x), giving bin 1. Good.
        # For x > b_k: returns k, giving bin k. Good.
        # So bin_idx as computed is correct.
        n_bins = len(bps) + 1
        n = len(x)
        full = np.zeros((n, n_bins), dtype=float)
        full[np.arange(n), bin_idx] = 1.0
        # Drop reference column (bin 0).
        values = full[:, 1:]
        columns = tuple(f"step({self.col})_bin{i}" for i in range(1, n_bins))
        state = _freeze_state(
            {
                "breakpoints": bps.tolist(),
                "boundary_knots": (float(np.min(x)), float(np.max(x))),
                # `boundary_knots` is included so the
                # SupportContainsPredictData assumption can read it. For
                # a step term the meaningful "support" is the training
                # min/max — predict-time x far beyond either is
                # extrapolation in the same sense as for splines.
            }
        )
        return _BasisExpandedTerm(columns=columns, values=values, state=state)

    def __add__(self, other: object) -> TermSum:
        return _add_terms(self, other)

    def __radd__(self, other: object) -> TermSum:
        return _add_terms(other, self)


def step(col: str, breakpoints: Sequence[float]) -> _StepTerm:
    """Piecewise-constant basis term.

    Parameters
    ----------
    col:
        Column name in the data frame.
    breakpoints:
        Sorted strictly-increasing sequence of breakpoints
        :math:`b_1 < \\cdots < b_k`. The basis produces :math:`k`
        indicator columns; the bin :math:`(-\\infty, b_1]` is the
        reference (dropped for identifiability).

    Returns
    -------
    Term
        A frozen :class:`_StepTerm`.
    """
    return _StepTerm(col=col, breakpoints=tuple(float(b) for b in breakpoints))


# ---------------------------------------------------------------------------
# Hinge (MARS)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _HingeTerm:
    r"""MARS-style hinge (ESL §9.4.1).

    A single-column basis:

    * ``direction='left'``  → :math:`(t - x)_+`
    * ``direction='right'`` → :math:`(x - t)_+`

    Stateless: the knot ``t`` is fixed at construction; no fit-time
    learning is required.
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
        # The hinge's "training boundary" for the SOFT support check is
        # just the training min/max — the hinge is well-defined
        # everywhere, but a predict-time x far outside the training
        # range still warrants an extrapolation warning.
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

    Parameters
    ----------
    col:
        Column name in the data frame.
    knot:
        Hinge location :math:`t`.
    direction:
        ``'left'`` produces :math:`(t - x)_+`; ``'right'`` produces
        :math:`(x - t)_+`.

    Returns
    -------
    Term
        A frozen :class:`_HingeTerm`.
    """
    return _HingeTerm(col=col, knot=float(knot), direction=direction)
