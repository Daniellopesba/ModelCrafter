"""Tests for the assumption framework core (P1.B).

Acceptance criteria from DESIGN.md §8 Phase 1 / AGENTS.md Task P1.B that this
file pins:

- "Unit tests for each concrete assumption: synthetic data triggers PASS,
  synthetic data triggers FAIL, with statistic values verified against a
  reference (scipy or hand-derived). For HARD assumptions verify the failure
  raises ``AssumptionError`` under ``on_violation='raise'``."
- "``run_assumptions`` raises ``AssumptionError`` on HARD failure when
  ``on_violation='raise'`` (the default). Warns on SOFT failure. Silently
  records INFO results."
- "``classical_inference=False`` (default) suppresses INFO checks entirely;
  ``classical_inference=True`` runs them and includes them at INFO severity."

The classical-inference checks themselves get their own pass/fail tests in
``tests/test_assumptions_classical.py``; here we test the framework and the
HARD prerequisite (``FullRankDesign``) plus the stability stubs.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pytest

from model_crafter.assumptions import (
    Assumption,
    AssumptionError,
    AssumptionReport,
    CheckResult,
    CoefficientStability,
    FullRankDesign,
    Homoscedasticity,
    Independence,
    LowVIF,
    PredictiveStability,
    ResidualNormality,
    Severity,
    check_assumptions,
    run_assumptions,
)

# ---------------------------------------------------------------------------
# Test fixtures: minimal stub spec/solution/loss types.
#
# P1.B does not depend on P1.A's concrete LinearSpec / Solution; we build
# the minimum surface our framework reads. Documented attributes consumed:
#
# spec:
#   target: str         (column name)
#   features: tuple     (the feature columns / Term objects with .name)
#   loss: object        (must expose .assumptions: tuple[Assumption, ...])
#   penalty: object|None (optional; .assumptions if present)
#   intercept: bool     (whether to prepend an intercept column)
#
# solution (when needed):
#   coefficients: pd.Series       (indexed by design column name)
#   coefficient_se: pd.Series|None
#   design_columns: tuple[str,...] (column names in order; intercept if used)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StubLoss:
    """Minimal stub of a ``Loss`` that declares an ``assumptions`` tuple."""

    assumptions: tuple[Assumption, ...] = ()


@dataclass(frozen=True)
class StubPenalty:
    assumptions: tuple[Assumption, ...] = ()


@dataclass(frozen=True)
class StubTerm:
    name: str
    assumptions: tuple[Assumption, ...] = ()


@dataclass(frozen=True)
class StubSpec:
    target: str
    features: tuple[Any, ...]
    loss: StubLoss
    penalty: StubPenalty | None = None
    intercept: bool = True


@dataclass(frozen=True)
class StubSolution:
    coefficients: pd.Series
    coefficient_se: pd.Series | None = None
    design_columns: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Test 1: CheckResult is a frozen dataclass with the contract fields.
# ---------------------------------------------------------------------------


def test_check_result_is_frozen_with_contract_fields():
    """CheckResult exposes (name, severity, passed, message, statistic,
    threshold, suggestion) and is immutable."""
    r = CheckResult(
        name="X",
        severity=Severity.HARD,
        passed=True,
        message="ok",
        statistic=1.0,
        threshold=0.5,
        suggestion=None,
    )
    assert r.name == "X"
    assert r.severity is Severity.HARD
    assert r.passed is True
    assert r.message == "ok"
    assert r.statistic == 1.0
    assert r.threshold == 0.5
    assert r.suggestion is None

    with pytest.raises((AttributeError, TypeError, Exception)):  # frozen
        r.passed = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Test 2: Severity enum has HARD/SOFT/INFO members with documented values.
# ---------------------------------------------------------------------------


def test_severity_enum_members():
    """Severity enum has HARD/SOFT/INFO members from DESIGN.md §4.1."""
    assert Severity.HARD.value == "hard"
    assert Severity.SOFT.value == "soft"
    assert Severity.INFO.value == "info"


# ---------------------------------------------------------------------------
# Test 3: AssumptionReport groups by severity, exposes passed(), and has a
#          repr that shows pass/fail status per result.
# ---------------------------------------------------------------------------


def test_assumption_report_by_severity_and_passed():
    """AssumptionReport.by_severity groups results; passed() is True iff
    no HARD or SOFT failure."""
    r1 = CheckResult("a", Severity.HARD, True, "ok", None, None, None)
    r2 = CheckResult("b", Severity.SOFT, True, "ok", None, None, None)
    r3 = CheckResult("c", Severity.INFO, False, "fail-info", 3.0, 1.0, None)
    rep = AssumptionReport(results=(r1, r2, r3))
    grouped = rep.by_severity()
    assert grouped[Severity.HARD] == (r1,)
    assert grouped[Severity.SOFT] == (r2,)
    assert grouped[Severity.INFO] == (r3,)
    # INFO failure does not flip passed() — INFO is report-only.
    assert rep.passed() is True

    r4 = CheckResult("d", Severity.SOFT, False, "soft-fail", None, None, None)
    rep2 = AssumptionReport(results=(r1, r4))
    assert rep2.passed() is False


def test_assumption_report_repr_contains_per_result_status():
    r1 = CheckResult("Rank", Severity.HARD, True, "rank ok", 8.0, 8.0, None)
    r2 = CheckResult("VIF", Severity.INFO, False, "VIF high", 11.0, 10.0, None)
    rep = AssumptionReport(results=(r1, r2))
    text = repr(rep)
    assert "AssumptionReport" in text
    assert "Rank" in text
    assert "VIF" in text
    # Visual pass/fail markers
    assert "pass" in text.lower() or "PASS" in text or "ok" in text.lower()
    assert "fail" in text.lower() or "FAIL" in text


# ---------------------------------------------------------------------------
# Test 4: FullRankDesign passes on full-rank data and fails (and raises under
#          on_violation="raise") on a rank-deficient design that has linearly
#          dependent columns. Failing message names the offending columns.
# ---------------------------------------------------------------------------


def _full_rank_frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((50, 3))
    return pd.DataFrame(X, columns=["x1", "x2", "x3"]).assign(
        y=lambda d: d["x1"] + d["x2"] - 0.5 * d["x3"] + rng.standard_normal(50)
    )


def _rank_deficient_frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((50, 2))
    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["x3"] = df["x1"] + df["x2"]  # dependent
    df["y"] = df["x1"] - df["x2"] + rng.standard_normal(50)
    return df


def _spec(features: tuple[str, ...], assumptions: tuple[Assumption, ...]) -> StubSpec:
    loss = StubLoss(assumptions=assumptions)
    feats = tuple(StubTerm(name=c) for c in features)
    return StubSpec(target="y", features=feats, loss=loss, intercept=True)


def test_full_rank_design_passes_on_full_rank():
    a = FullRankDesign()
    spec = _spec(("x1", "x2", "x3"), (a,))
    res = a.check(spec, _full_rank_frame())
    assert res.severity is Severity.HARD
    assert res.passed is True
    # Statistic is the rank deficit (0 when full rank).
    assert res.statistic == 0


def test_full_rank_design_fails_and_names_offending_columns():
    a = FullRankDesign()
    spec = _spec(("x1", "x2", "x3"), (a,))
    res = a.check(spec, _rank_deficient_frame())
    assert res.passed is False
    assert res.severity is Severity.HARD
    # Rank deficit of 1.
    assert res.statistic == 1
    # The dependent column "x3" must be named.
    assert "x3" in res.message


def test_run_assumptions_raises_on_hard_failure_by_default():
    """run_assumptions raises AssumptionError on HARD failure with
    on_violation='raise' (which is the default per the spec)."""
    spec = _spec(("x1", "x2", "x3"), (FullRankDesign(),))
    # Default
    with pytest.raises(AssumptionError):
        run_assumptions(spec, _rank_deficient_frame())
    # Explicit
    with pytest.raises(AssumptionError):
        run_assumptions(spec, _rank_deficient_frame(), on_violation="raise")


def test_run_assumptions_warn_on_violation_does_not_raise_hard():
    """on_violation='warn' converts HARD failures into warnings + a recorded
    failure (does not raise)."""
    spec = _spec(("x1", "x2", "x3"), (FullRankDesign(),))
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        rep = run_assumptions(spec, _rank_deficient_frame(), on_violation="warn")
    assert any("FullRankDesign" in str(w.message) for w in wlist)
    assert any(r.name == "FullRankDesign" and not r.passed for r in rep.results)


def test_run_assumptions_ignore_silently_records():
    spec = _spec(("x1", "x2", "x3"), (FullRankDesign(),))
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        rep = run_assumptions(spec, _rank_deficient_frame(), on_violation="ignore")
    assert not any("FullRankDesign" in str(w.message) for w in wlist)
    assert any(r.name == "FullRankDesign" and not r.passed for r in rep.results)


# ---------------------------------------------------------------------------
# Test 5: SOFT failures warn (and report) but never raise — even under the
#          default on_violation='raise'.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _AlwaysFailingSoft:
    name: str = "_AlwaysFailingSoft"
    severity: Severity = Severity.SOFT
    requires_solution: bool = False
    requires_cv: bool = False

    def describe(self) -> str:
        return "test-only assumption that always fails"

    def check(self, spec, data, *, solution=None, cv=None) -> CheckResult:
        return CheckResult(
            name=self.name,
            severity=Severity.SOFT,
            passed=False,
            message="forced failure",
            statistic=None,
            threshold=None,
            suggestion=None,
        )


def test_run_assumptions_warns_on_soft_failure_under_default_raise():
    spec = _spec(("x1",), (_AlwaysFailingSoft(),))
    df = pd.DataFrame({"x1": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]})
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        rep = run_assumptions(spec, df)  # default on_violation='raise'
    assert any("forced failure" in str(w.message) for w in wlist)
    assert rep.passed() is False


# ---------------------------------------------------------------------------
# Test 6: INFO results are silently recorded (no warning, no raise) and are
#          GATED by classical_inference=True.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _AlwaysFailingInfo:
    name: str = "_AlwaysFailingInfo"
    severity: Severity = Severity.INFO
    requires_solution: bool = False
    requires_cv: bool = False

    def describe(self) -> str:
        return "test-only INFO assumption that always fails"

    def check(self, spec, data, *, solution=None, cv=None) -> CheckResult:
        return CheckResult(
            name=self.name,
            severity=Severity.INFO,
            passed=False,
            message="info noise",
            statistic=1.23,
            threshold=1.0,
            suggestion=None,
        )


def test_info_checks_suppressed_without_classical_inference_flag():
    spec = _spec(("x1",), (_AlwaysFailingInfo(),))
    df = pd.DataFrame({"x1": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]})
    rep = run_assumptions(spec, df)
    assert all(r.name != "_AlwaysFailingInfo" for r in rep.results)


def test_info_checks_run_with_classical_inference_flag_and_dont_warn():
    spec = _spec(("x1",), (_AlwaysFailingInfo(),))
    df = pd.DataFrame({"x1": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]})
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        rep = run_assumptions(spec, df, classical_inference=True)
    # No warning was emitted for the INFO failure.
    assert not any("info noise" in str(w.message) for w in wlist)
    # But it was recorded at INFO severity.
    info = [r for r in rep.results if r.name == "_AlwaysFailingInfo"]
    assert len(info) == 1
    assert info[0].severity is Severity.INFO
    assert info[0].passed is False


def test_check_assumptions_wraps_run_assumptions_in_ignore_mode():
    """check_assumptions returns the report without raising on HARD
    failures — it's the standalone reporter."""
    spec = _spec(("x1", "x2", "x3"), (FullRankDesign(),))
    rep = check_assumptions(spec, _rank_deficient_frame())
    assert any(r.name == "FullRankDesign" and not r.passed for r in rep.results)


# ---------------------------------------------------------------------------
# Test 7: stability stubs — when requires_cv=True and cv is None, the check
#          returns a passed=True result with an explanatory message ("skipped:
#          no CV available"), not a silent skip and not a raise.
# ---------------------------------------------------------------------------


def test_coefficient_stability_skips_without_cv():
    a = CoefficientStability()
    spec = _spec(("x1",), (a,))
    df = pd.DataFrame({"x1": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]})
    res = a.check(spec, df, cv=None)
    assert a.requires_cv is True
    assert res.passed is True
    assert "skipped" in res.message.lower() or "no cv" in res.message.lower()


def test_predictive_stability_skips_without_cv():
    a = PredictiveStability()
    spec = _spec(("x1",), (a,))
    df = pd.DataFrame({"x1": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]})
    res = a.check(spec, df, cv=None)
    assert a.requires_cv is True
    assert res.passed is True
    assert "skipped" in res.message.lower() or "no cv" in res.message.lower()


# ---------------------------------------------------------------------------
# Test 8: suppress= drops named assumption classes.
# ---------------------------------------------------------------------------


def test_suppress_drops_named_assumption_classes():
    spec = _spec(
        ("x1", "x2", "x3"),
        (FullRankDesign(), _AlwaysFailingSoft()),
    )
    rep = run_assumptions(
        spec,
        _full_rank_frame(),
        suppress=(_AlwaysFailingSoft,),
    )
    assert all(r.name != "_AlwaysFailingSoft" for r in rep.results)


# ---------------------------------------------------------------------------
# Test 9: requires_solution-gated checks return a "skipped" pass when no
#          solution is provided. Stays consistent with the "no silent
#          failures" rule.
# ---------------------------------------------------------------------------


def test_residual_normality_skipped_without_solution():
    a = ResidualNormality()
    spec = _spec(("x1",), (a,))
    df = pd.DataFrame({"x1": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]})
    assert a.requires_solution is True
    res = a.check(spec, df, solution=None)
    assert res.passed is True
    assert "skipped" in res.message.lower() or "no solution" in res.message.lower()


def test_homoscedasticity_skipped_without_solution():
    a = Homoscedasticity()
    spec = _spec(("x1",), (a,))
    df = pd.DataFrame({"x1": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]})
    assert a.requires_solution is True
    res = a.check(spec, df, solution=None)
    assert res.passed is True


def test_independence_skipped_without_solution():
    a = Independence()
    spec = _spec(("x1",), (a,))
    df = pd.DataFrame({"x1": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]})
    assert a.requires_solution is True
    res = a.check(spec, df, solution=None)
    assert res.passed is True


def test_low_vif_does_not_require_solution():
    # VIF reads off the design only; no fitted coefficients needed.
    a = LowVIF()
    assert a.requires_solution is False


# ---------------------------------------------------------------------------
# Test 10: invalid on_violation raises ValueError early (no silent default).
# ---------------------------------------------------------------------------


def test_run_assumptions_rejects_invalid_on_violation():
    spec = _spec(("x1",), (FullRankDesign(),))
    df = pd.DataFrame({"x1": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]})
    with pytest.raises(ValueError):
        run_assumptions(spec, df, on_violation="explode")
