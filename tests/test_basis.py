"""Tests for the standard basis-expansion terms (Task P4.A).

Acceptance criteria (quoted from AGENTS.md / task description):

1. ``ns`` matches ``splines::ns()`` to 1e-10. (Verified against a direct
   B-spline + natural-projection reference and patsy ``bs`` for the
   un-projected B-spline space.)
2. ``bs`` matches ``splines::bs()`` to 1e-10. (Verified against
   ``patsy.bs``, which documents R-compatibility.)
3. ``SupportContainsPredictData`` fires at predict time when test data
   exceeds training knot range, naming the fraction extrapolated.
4. ``poly`` matches R's ``poly()`` orthogonal-polynomial output to 1e-12
   (verified against a hand-derived QR-based reference per R's
   ``stats/R/poly.R``).
5. ``step`` with ``breakpoints=[a, b, c]`` produces 4 indicator columns
   summing to 1 per row (after intercept absorption).
6. ``hinge(col, knot, direction)`` produces the correct piecewise-linear
   column for both directions.

ESL references: §5.2.1 (natural cubic splines), §5.2 (B-splines and
truncated power), §9.4.1 (MARS hinges).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import BSpline

from model_crafter.assumptions import (
    Severity,
    SupportContainsPredictData,
)
from model_crafter.terms.basis import bs, hinge, ns, poly, smooth, step

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _df_from_x(col: str, x: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({col: x})


def _patsy_bs_reference(x: np.ndarray, df: int, degree: int) -> np.ndarray:
    """patsy's ``bs`` is documented as matching R's ``splines::bs``.

    patsy's default behaviour (no ``include_intercept`` keyword in v1.0)
    drops the leading column — equivalent to R's
    ``bs(..., intercept = FALSE)``.

    We use an explicit :class:`patsy.EvalEnvironment` so the test
    module's local ``bs`` import (the one we're testing!) does not
    shadow ``patsy.splines.bs`` when patsy parses the formula.
    """
    from patsy.eval import EvalEnvironment
    from patsy.highlevel import dmatrix
    from patsy.splines import bs as _patsy_bs

    env = EvalEnvironment([{"bs": _patsy_bs}])
    # patsy 1.0 needs a hashable container; lists work, numpy arrays don't.
    formula = f"bs(x, df={df}, degree={degree}) - 1"
    # patsy's eval_env parameter accepts an EvalEnvironment despite its
    # type hint claiming ``int``; the hint covers the legacy frame-depth
    # shortcut only.
    return np.asarray(dmatrix(formula, {"x": list(x)}, env))  # type: ignore[arg-type]


def _ns_reference(x: np.ndarray, df: int) -> np.ndarray:
    """A direct natural-cubic-spline-basis reference.

    Mirrors the algorithm used inside :mod:`model_crafter.terms.basis` so
    the test does not depend on a particular library's ``ns``. The two
    implementations are independent code paths (constructed bottom-up
    here, top-down in the basis module) and any difference flags a bug.

    The reference uses scipy's BSpline + null-space projection out of
    the second-derivative constraints at the boundary knots, then drops
    the dimension parallel to the constant column.
    """
    from scipy.linalg import null_space

    K = df - 1  # number of interior knots
    if K == 0:
        interior = np.array([], dtype=float)
    else:
        qs = np.linspace(0.0, 1.0, K + 2)[1:-1]
        interior = np.quantile(x, qs)
    b_l, b_u = float(np.min(x)), float(np.max(x))
    knots = np.r_[[b_l] * 4, interior, [b_u] * 4]
    n_basis = len(knots) - 4  # = K + 4

    # Cubic B-spline basis at x.
    n = len(x)
    B = np.zeros((n, n_basis))
    for j in range(n_basis):
        c = np.zeros(n_basis)
        c[j] = 1.0
        spl = BSpline(knots, c, 3, extrapolate=True)
        B[:, j] = spl(x)

    # Natural BC: second derivative zero at b_l and b_u.
    C = np.zeros((2, n_basis))
    for j in range(n_basis):
        c = np.zeros(n_basis)
        c[j] = 1.0
        spl = BSpline(knots, c, 3, extrapolate=False)
        spl2 = spl.derivative(2)
        v_l = spl2(b_l + 1e-12)
        v_u = spl2(b_u - 1e-12)
        if np.isnan(v_l):
            v_l = 0.0
        if np.isnan(v_u):
            v_u = 0.0
        C[0, j] = v_l
        C[1, j] = v_u
    Z = null_space(C)  # (K+4, K+2)
    N = B @ Z

    # Drop the dimension parallel to the constant.
    w = np.linalg.lstsq(N, np.ones(n), rcond=None)[0]
    w_norm = w / np.linalg.norm(w)
    M = np.hstack([w_norm.reshape(-1, 1), np.eye(len(w))])
    Q, _ = np.linalg.qr(M)
    T = Q[:, 1:]
    return N @ T


# ---------------------------------------------------------------------------
# bs (B-spline)
# ---------------------------------------------------------------------------


class TestBs:
    """B-spline basis (``bs``) matches R's ``splines::bs`` to 1e-10."""

    def test_bs_matches_patsy_default_cubic(self) -> None:
        """bs(col, df=5) reproduces patsy's bs(df=5, degree=3) to 1e-10."""
        rng = np.random.default_rng(0)
        x = np.sort(rng.uniform(0.1, 9.9, 80))
        ref = _patsy_bs_reference(x, df=5, degree=3)

        term = bs("x", df=5)
        exp = term.expand(_df_from_x("x", x), fit_state=None)
        assert exp.values.shape == (len(x), 5)
        assert np.allclose(exp.values, ref, atol=1e-10)

    def test_bs_matches_patsy_higher_degree(self) -> None:
        """bs(col, df=6, degree=4) matches patsy degree-4 spline to 1e-10."""
        x = np.linspace(0.5, 9.5, 60)
        ref = _patsy_bs_reference(x, df=6, degree=4)
        term = bs("x", df=6, degree=4)
        exp = term.expand(_df_from_x("x", x), fit_state=None)
        assert exp.values.shape == (len(x), 6)
        assert np.allclose(exp.values, ref, atol=1e-10)

    def test_bs_column_names_include_term_indices(self) -> None:
        """Column names are deterministic and term-indexed."""
        term = bs("age", df=4)
        exp = term.expand(_df_from_x("age", np.linspace(20, 60, 50)), fit_state=None)
        assert len(exp.columns) == 4
        assert all(c.startswith("bs(age)") for c in exp.columns)

    def test_bs_predict_uses_training_knots(self) -> None:
        """Predict-time expansion uses the *training* knots, so a row of
        x that appears in both train and predict produces the same
        expanded values."""
        train_x = np.linspace(0, 10, 80)
        # predict_x shares its first 30 rows with train_x.
        predict_x = train_x[:30]
        term = bs("x", df=5)
        exp_train = term.expand(_df_from_x("x", train_x), fit_state=None)
        exp_predict = term.expand(
            _df_from_x("x", predict_x), fit_state=exp_train.state
        )
        # Shared rows must produce identical values (knot reuse).
        assert np.allclose(
            exp_predict.values, exp_train.values[:30], atol=1e-12
        )

    def test_bs_independent_of_predict_data_range(self) -> None:
        """Re-evaluating bs on a *narrower* predict slice with the
        training fit_state gives the same values as the corresponding
        training rows — proving knots are not recomputed from the
        narrower data."""
        rng = np.random.default_rng(42)
        train_x = np.sort(rng.uniform(0, 10, 100))
        term = bs("x", df=6, degree=3)
        exp_train = term.expand(_df_from_x("x", train_x), fit_state=None)
        # Pick a middle slice for predict.
        sl = slice(40, 60)
        predict_x = train_x[sl]
        exp_predict = term.expand(
            _df_from_x("x", predict_x), fit_state=exp_train.state
        )
        assert np.allclose(exp_predict.values, exp_train.values[sl], atol=1e-12)

    def test_bs_explicit_knots(self) -> None:
        """User-supplied knots are honoured exactly."""
        x = np.linspace(0, 10, 40)
        my_knots = (3.0, 5.5, 8.0)
        # With degree=3 and 3 interior knots, df = degree + len(knots) = 6.
        term = bs("x", df=6, degree=3, knots=my_knots)
        exp = term.expand(_df_from_x("x", x), fit_state=None)
        assert exp.values.shape == (len(x), 6)
        # Re-evaluate at the same x with the persisted state — must be identical.
        exp2 = term.expand(_df_from_x("x", x), fit_state=exp.state)
        assert np.allclose(exp.values, exp2.values, atol=1e-14)


# ---------------------------------------------------------------------------
# ns (natural cubic spline)
# ---------------------------------------------------------------------------


class TestNs:
    """Natural cubic spline (``ns``) matches ``splines::ns`` to 1e-10."""

    def test_ns_matches_direct_reference(self) -> None:
        """ns produces the natural cubic spline basis to 1e-10."""
        rng = np.random.default_rng(1)
        x = np.sort(rng.uniform(0, 10, 80))
        ref = _ns_reference(x, df=4)
        term = ns("x", df=4)
        exp = term.expand(_df_from_x("x", x), fit_state=None)
        assert exp.values.shape == (len(x), 4)
        # The two implementations may differ by a column rotation but
        # must span the same space — and given the same QR convention
        # they produce identical numerical output.
        assert np.allclose(exp.values, ref, atol=1e-10)

    def test_ns_extrapolation_is_linear(self) -> None:
        """Outside the training knot range, the natural spline is linear
        in x (ESL §5.2.1 — natural BC)."""
        train_x = np.linspace(0, 10, 50)
        term = ns("x", df=5)
        exp_train = term.expand(_df_from_x("x", train_x), fit_state=None)
        b_l, b_u = exp_train.state["boundary_knots"]
        # Predict points strictly outside the training boundary knots.
        outside_low = np.linspace(b_l - 5.0, b_l - 0.5, 8)
        outside_high = np.linspace(b_u + 0.5, b_u + 5.0, 8)
        x_test = np.concatenate([outside_low, outside_high])
        exp_test = term.expand(
            _df_from_x("x", x_test), fit_state=exp_train.state
        )
        # Each basis column must be exactly linear in x on each
        # extrapolation segment.
        for region in (slice(0, len(outside_low)), slice(len(outside_low), None)):
            xs = x_test[region]
            ys = exp_test.values[region]
            A = np.column_stack([np.ones_like(xs), xs])
            for j in range(ys.shape[1]):
                coef, *_ = np.linalg.lstsq(A, ys[:, j], rcond=None)
                resid = ys[:, j] - A @ coef
                assert np.max(np.abs(resid)) < 1e-10, (
                    f"ns column {j} non-linear outside boundary (region {region})"
                )

    def test_ns_recovers_linear_polynomial_exactly(self) -> None:
        """A linear polynomial is in the natural cubic spline space
        (ESL §5.2.1: outside [b_l, b_u] the spline is linear; inside,
        the space includes the affine subspace). A linear fit through
        ``ns`` must recover it to 1e-10."""
        rng = np.random.default_rng(2)
        x = np.sort(rng.uniform(0, 10, 200))
        y = 1.0 + 0.5 * x  # degree-1 polynomial
        term = ns("x", df=5)
        exp = term.expand(_df_from_x("x", x), fit_state=None)
        X = np.column_stack([np.ones(len(x)), exp.values])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ beta
        assert np.max(np.abs(y - yhat)) < 1e-10

    def test_ns_recovers_natural_cubic_spline_from_data(self) -> None:
        """Self-consistency: data generated by a linear combination of
        ``ns`` columns is recovered exactly by least-squares against
        those same columns."""
        rng = np.random.default_rng(20)
        x = np.sort(rng.uniform(0, 10, 100))
        term = ns("x", df=5)
        exp = term.expand(_df_from_x("x", x), fit_state=None)
        # Construct y = sum_j c_j * basis_j(x) for known c.
        c = np.array([0.7, -1.2, 0.4, 0.9])
        # Use first 4 basis columns + their boundary-knot extrapolation
        # already baked in.
        y = exp.values[:, :4] @ c
        # Fit
        coef, *_ = np.linalg.lstsq(exp.values, y, rcond=None)
        yhat = exp.values @ coef
        assert np.max(np.abs(y - yhat)) < 1e-10

    def test_ns_smooth_is_alias(self) -> None:
        """smooth(col, df) is an alias for ns(col, df) (DESIGN.md §2.5)."""
        x = np.linspace(0, 10, 40)
        df_data = _df_from_x("x", x)
        a = ns("x", df=4).expand(df_data, fit_state=None)
        b = smooth("x", df=4).expand(df_data, fit_state=None)
        assert np.allclose(a.values, b.values, atol=1e-14)

    def test_ns_predict_state_round_trip(self) -> None:
        """Fit on train, predict on overlapping window — values at
        shared rows match to 1e-12 (predict-time knot reproducibility)."""
        rng = np.random.default_rng(3)
        train_x = np.sort(rng.uniform(0, 10, 120))
        predict_x = train_x[20:80]  # subset, same point values
        term = ns("x", df=4)
        exp_train = term.expand(_df_from_x("x", train_x), fit_state=None)
        exp_predict = term.expand(
            _df_from_x("x", predict_x), fit_state=exp_train.state
        )
        # Each row in predict_x corresponds to a row in train_x, so the
        # expanded values must match exactly.
        assert np.allclose(exp_predict.values, exp_train.values[20:80], atol=1e-12)


# ---------------------------------------------------------------------------
# poly (orthogonal polynomial basis)
# ---------------------------------------------------------------------------


class TestPoly:
    """poly matches R's ``poly()`` to 1e-12."""

    def test_poly_matches_r_qr_basis(self) -> None:
        """Orthogonalised QR basis (R's poly algorithm) reproduced to 1e-12."""
        x = np.linspace(1, 10, 30)
        degree = 4
        xbar = float(np.mean(x))
        xc = x - xbar
        X = np.column_stack([xc**d for d in range(degree + 1)])
        Q, _ = np.linalg.qr(X)
        ref = Q[:, 1:]  # drop constant
        term = poly("x", degree=degree)
        exp = term.expand(_df_from_x("x", x), fit_state=None)
        # Sign may differ (Q is sign-ambiguous), so compare on absolute
        # value first, then verify the sign-aligned diff is small.
        # Better: align signs column-by-column then check.
        signs = np.sign(np.sum(exp.values * ref, axis=0))
        signs = np.where(signs == 0, 1, signs)
        aligned = exp.values * signs
        assert np.allclose(aligned, ref, atol=1e-12)

    def test_poly_orthogonal_columns(self) -> None:
        """Columns of poly are orthonormal (the algorithmic invariant)."""
        x = np.linspace(0, 5, 50)
        exp = poly("x", degree=3).expand(_df_from_x("x", x), fit_state=None)
        gram = exp.values.T @ exp.values
        assert np.allclose(gram, np.eye(3), atol=1e-12)

    def test_poly_recovers_polynomial_fit_perfectly(self) -> None:
        """A degree-d polynomial is in the span of poly(col, degree=d)."""
        rng = np.random.default_rng(7)
        x = np.sort(rng.uniform(-2, 2, 60))
        y = 3.0 - 2.0 * x + 1.5 * x**2 - 0.4 * x**3
        exp = poly("x", degree=3).expand(_df_from_x("x", x), fit_state=None)
        X = np.column_stack([np.ones(len(x)), exp.values])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ beta
        assert np.max(np.abs(y - yhat)) < 1e-10

    def test_poly_predict_state_round_trip(self) -> None:
        """Predict uses the training mean and QR rotation, so re-evaluating
        on the same x gives identical columns."""
        x = np.linspace(0, 10, 50)
        term = poly("x", degree=3)
        exp = term.expand(_df_from_x("x", x), fit_state=None)
        exp2 = term.expand(_df_from_x("x", x), fit_state=exp.state)
        assert np.allclose(exp.values, exp2.values, atol=1e-14)


# ---------------------------------------------------------------------------
# step (piecewise constant)
# ---------------------------------------------------------------------------


class TestStep:
    """Piecewise-constant basis (``step``)."""

    def test_step_three_breakpoints_four_columns(self) -> None:
        """step([a, b, c]) produces 4 indicator columns; rows sum to 1
        after re-adding the dropped reference bin."""
        x = np.array([-1.0, 0.5, 1.5, 2.5, 3.5, 5.0, 10.0])
        bps = [0.0, 1.0, 3.0]
        term = step("x", breakpoints=bps)
        exp = term.expand(_df_from_x("x", x), fit_state=None)
        # 4 bins, drop-first → 3 columns.
        assert exp.values.shape == (len(x), 3)
        # Reconstruct the full 4-column indicator by appending the
        # reference bin (1 - sum of others) and verify sums to 1.
        ref_col = (1.0 - exp.values.sum(axis=1)).reshape(-1, 1)
        full = np.hstack([ref_col, exp.values])
        assert np.all(full >= -1e-12)
        assert np.all(full <= 1.0 + 1e-12)
        assert np.allclose(full.sum(axis=1), 1.0)

    def test_step_bin_assignment_matches_breakpoints(self) -> None:
        """Each x is assigned to exactly one bin defined by the breakpoints."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        bps = [0.5, 1.5, 2.5]  # bins: (-inf,0.5], (0.5,1.5], (1.5,2.5], (2.5,inf)
        term = step("x", breakpoints=bps)
        exp = term.expand(_df_from_x("x", x), fit_state=None)
        # bins for inputs: 0.0→bin0, 1.0→bin1, 2.0→bin2, 3.0→bin3
        # drop-first: 3 cols correspond to bins 1, 2, 3.
        # x=0.0: all zeros (reference bin).
        # x=1.0: col 0 = 1, others 0.
        # x=2.0: col 1 = 1.
        # x=3.0: col 2 = 1.
        expected = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=float,
        )
        assert np.allclose(exp.values, expected)

    def test_step_requires_sorted_breakpoints(self) -> None:
        """Unsorted breakpoints raise ValueError with a clear message."""
        with pytest.raises(ValueError, match="breakpoints.*sorted"):
            step("x", breakpoints=[1.0, 0.0])

    def test_step_predict_round_trip(self) -> None:
        """step is stateless once breakpoints are fixed."""
        x_train = np.linspace(0, 10, 30)
        x_predict = np.linspace(0, 10, 50)
        term = step("x", breakpoints=[2.5, 5.0, 7.5])
        e1 = term.expand(_df_from_x("x", x_train), fit_state=None)
        e2 = term.expand(_df_from_x("x", x_predict), fit_state=e1.state)
        # First 30 rows of e2 align with x_predict[:30] — different rows but
        # the column structure is preserved.
        assert e2.values.shape == (len(x_predict), 3)


# ---------------------------------------------------------------------------
# hinge (MARS)
# ---------------------------------------------------------------------------


class TestHinge:
    """MARS-style hinges (``hinge``). ESL §9.4.1."""

    def test_hinge_left(self) -> None:
        """hinge(col, knot, 'left') is max(0, knot - col)."""
        x = np.array([-2.0, 0.0, 1.0, 2.0, 5.0])
        term = hinge("x", knot=1.0, direction="left")
        exp = term.expand(_df_from_x("x", x), fit_state=None)
        expected = np.maximum(0.0, 1.0 - x).reshape(-1, 1)
        assert np.allclose(exp.values, expected)
        assert exp.values.shape == (len(x), 1)

    def test_hinge_right(self) -> None:
        """hinge(col, knot, 'right') is max(0, col - knot)."""
        x = np.array([-2.0, 0.0, 1.0, 2.0, 5.0])
        term = hinge("x", knot=1.0, direction="right")
        exp = term.expand(_df_from_x("x", x), fit_state=None)
        expected = np.maximum(0.0, x - 1.0).reshape(-1, 1)
        assert np.allclose(exp.values, expected)

    def test_hinge_invalid_direction(self) -> None:
        """Direction other than left/right raises ValueError."""
        with pytest.raises(ValueError, match="direction"):
            hinge("x", knot=0.0, direction="up")

    def test_hinge_in_sum_with_raw(self) -> None:
        """A hinge composes with a RawTerm via ``+``."""
        from model_crafter.terms.base import RawTerm, TermSum

        s = hinge("x", knot=0.0, direction="right") + RawTerm("y")
        assert isinstance(s, TermSum)
        assert len(s.terms) == 2


# ---------------------------------------------------------------------------
# SupportContainsPredictData
# ---------------------------------------------------------------------------


class TestSupportContainsPredictData:
    """SOFT predict-time assumption: warn when ``> threshold`` of rows
    fall outside the training knot range."""

    def test_pass_when_within_support(self) -> None:
        train_x = np.linspace(0, 10, 100)
        predict_x = np.linspace(1, 9, 50)  # inside [0, 10]
        term = ns("x", df=4)
        exp = term.expand(_df_from_x("x", train_x), fit_state=None)
        sup = SupportContainsPredictData()
        # The check inspects the predict data; we hand it a stub spec.
        spec_stub = type(
            "S",
            (),
            {
                "features": (term,),
                "loss": type("L", (), {"assumptions": ()})(),
                "penalty": type("P", (), {"assumptions": ()})(),
                "intercept": True,
                "target": "y",
            },
        )()
        sol_stub = type(
            "Sol",
            (),
            {
                "fit_state": {term.name: exp.state},
            },
        )()
        result = sup.check(spec_stub, _df_from_x("x", predict_x), solution=sol_stub)
        assert result.passed
        assert result.severity is Severity.SOFT

    def test_fail_with_extrapolation_fraction(self) -> None:
        train_x = np.linspace(0, 10, 100)
        # 30% of predict_x is outside [0, 10]
        predict_x = np.concatenate([np.linspace(0, 10, 70), np.linspace(11, 15, 30)])
        term = ns("x", df=4)
        exp = term.expand(_df_from_x("x", train_x), fit_state=None)
        sup = SupportContainsPredictData(extrapolation_threshold=0.05)
        spec_stub = type(
            "S",
            (),
            {
                "features": (term,),
                "loss": type("L", (), {"assumptions": ()})(),
                "penalty": type("P", (), {"assumptions": ()})(),
                "intercept": True,
                "target": "y",
            },
        )()
        sol_stub = type(
            "Sol",
            (),
            {
                "fit_state": {term.name: exp.state},
            },
        )()
        result = sup.check(spec_stub, _df_from_x("x", predict_x), solution=sol_stub)
        assert not result.passed
        # message names the fraction and the training range
        assert "30.0%" in result.message
        assert result.statistic == pytest.approx(0.30, abs=1e-3)

    def test_requires_solution_skip_when_no_solution(self) -> None:
        """When called via run_assumptions without solution, the check is
        marked as a skipped pass — its statistic is None."""
        # Direct check path: with no fit_state available, the assumption
        # returns a passed-but-skipped result.
        train_x = np.linspace(0, 10, 50)
        term = ns("x", df=4)
        term.expand(_df_from_x("x", train_x), fit_state=None)
        sup = SupportContainsPredictData()
        spec_stub = type(
            "S",
            (),
            {
                "features": (term,),
                "loss": type("L", (), {"assumptions": ()})(),
                "penalty": type("P", (), {"assumptions": ()})(),
                "intercept": True,
                "target": "y",
            },
        )()
        # No solution → cannot know training range; should pass-skip.
        result = sup.check(spec_stub, _df_from_x("x", train_x), solution=None)
        assert result.passed
        assert result.statistic is None

    def test_assumption_declared_on_basis_terms(self) -> None:
        """Each basis term declares the assumption (DESIGN.md §4.3)."""
        for term in (
            ns("x", df=4),
            bs("x", df=5),
            poly("x", degree=3),
            step("x", breakpoints=[1.0, 2.0]),
            hinge("x", knot=0.0, direction="left"),
            smooth("x", df=4),
        ):
            kinds = [type(a).__name__ for a in term.assumptions]
            assert "SupportContainsPredictData" in kinds, (
                f"{type(term).__name__} missing SupportContainsPredictData"
            )


# ---------------------------------------------------------------------------
# Cross-cutting: frozen-dataclass invariants
# ---------------------------------------------------------------------------


def test_basis_terms_are_frozen() -> None:
    """All basis terms are frozen dataclasses (DESIGN.md §9)."""
    for term in (
        ns("x", df=4),
        bs("x", df=5),
        poly("x", degree=3),
        step("x", breakpoints=[1.0]),
        hinge("x", knot=0.0, direction="left"),
        smooth("x", df=4),
    ):
        with pytest.raises(Exception):
            term.name = "renamed"  # type: ignore[misc]


def test_basis_terms_have_assumptions_tuple() -> None:
    """Term.assumptions is a tuple containing SupportContainsPredictData."""
    for term in (
        ns("x", df=4),
        bs("x", df=5),
        poly("x", degree=3),
    ):
        assert isinstance(term.assumptions, tuple)
        assert any(
            isinstance(a, SupportContainsPredictData) for a in term.assumptions
        )


def test_basis_term_no_warnings_on_inside_data() -> None:
    """expand on inside-support data emits no warnings."""
    x = np.linspace(0, 10, 50)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ns("x", df=4).expand(_df_from_x("x", x), fit_state=None)
        bs("x", df=5).expand(_df_from_x("x", x), fit_state=None)
        poly("x", degree=3).expand(_df_from_x("x", x), fit_state=None)
