"""Assumption framework for ``model_crafter``.

This module implements the three-tier assumption protocol described in
``DESIGN.md`` §4:

* **HARD** — prerequisites the math literally requires. Violations raise
  :class:`AssumptionError` by default.
* **SOFT** — stability diagnostics (ESL §7 in spirit). Violations warn.
* **INFO** — classical-inference checks (Shapiro-Wilk, Breusch-Pagan,
  Durbin-Watson, VIF). Opt in via ``classical_inference=True``; never
  warn or raise.

The framework operates on a ``spec`` (any object exposing the documented
attributes — see :class:`Assumption` below for the contract) and
``data`` (a ``pd.DataFrame``).

Spec contract (the attributes the framework reads):

* ``spec.target: str`` — name of the target column in ``data``.
* ``spec.features: Iterable`` — feature terms. Each entry may be a string
  column name or a ``Term``-like object with a ``.name`` attribute.
* ``spec.loss`` — must expose ``assumptions: tuple[Assumption, ...]``.
* ``spec.penalty`` (optional) — may expose ``assumptions: tuple[Assumption, ...]``.
* ``spec.intercept: bool`` — whether the design includes an intercept
  (defaults to ``True`` when missing).

Solution contract (only used when ``requires_solution=True``):

* ``solution.coefficients: pd.Series`` — indexed by design column name.
* ``solution.coefficient_se: pd.Series | None``.
* ``solution.design_columns: tuple[str, ...]`` — column order.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from model_crafter.assumptions._types import (
    Assumption,
    AssumptionError,
    CheckResult,
    Severity,
)
from model_crafter.assumptions.classical import (
    Homoscedasticity,
    Independence,
    LowVIF,
    ResidualNormality,
)
from model_crafter.assumptions.logistic import (
    BinaryOrProportionTarget,
    ClassBalance,
    LinkAdequacy,
    NoPerfectSeparation,
)
from model_crafter.assumptions.prerequisites import FullRankDesign
from model_crafter.assumptions.stability import (
    CoefficientStability,
    ComparableFeatureScales,
    PredictiveStability,
)
from model_crafter.assumptions.temporal import NoTemporalLeakage


@dataclass(frozen=True, slots=True)
class AssumptionReport:
    """An ordered, immutable collection of :class:`CheckResult`s."""

    results: tuple[CheckResult, ...]

    def by_severity(self) -> dict[Severity, tuple[CheckResult, ...]]:
        """Group results by severity. All three severities always appear
        as keys (possibly with empty tuples)."""
        out: dict[Severity, list[CheckResult]] = {
            Severity.HARD: [],
            Severity.SOFT: [],
            Severity.INFO: [],
        }
        for r in self.results:
            out[r.severity].append(r)
        return {k: tuple(v) for k, v in out.items()}

    def passed(self) -> bool:
        """True iff no HARD or SOFT failure. INFO failures do not flip
        ``passed`` — INFO is report-only."""
        return all(
            r.passed
            for r in self.results
            if r.severity in (Severity.HARD, Severity.SOFT)
        )

    def __repr__(self) -> str:
        if not self.results:
            return "AssumptionReport(empty)"
        grouped = self.by_severity()
        lines = ["AssumptionReport"]
        for sev in (Severity.HARD, Severity.SOFT, Severity.INFO):
            items = grouped[sev]
            if not items:
                continue
            lines.append(f"  [{sev.value.upper()}]")
            for r in items:
                marker = "PASS" if r.passed else "FAIL"
                stat_str = (
                    f"  stat={r.statistic:.4g}" if r.statistic is not None else ""
                )
                lines.append(
                    f"    {marker}  {r.name:30s} {r.message}{stat_str}"
                )
                if r.suggestion:
                    lines.append(f"           -> {r.suggestion}")
        return "\n".join(lines)


__all__ = [
    "Assumption",
    "AssumptionError",
    "AssumptionReport",
    "BinaryOrProportionTarget",
    "CheckResult",
    "ClassBalance",
    "CoefficientStability",
    "ComparableFeatureScales",
    "FullRankDesign",
    "Homoscedasticity",
    "Independence",
    "LinkAdequacy",
    "LowVIF",
    "NoPerfectSeparation",
    "NoTemporalLeakage",
    "PredictiveStability",
    "ResidualNormality",
    "Severity",
    "check_assumptions",
    "run_assumptions",
]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


_VALID_ON_VIOLATION = ("raise", "warn", "ignore")


def _collect_assumptions(spec: Any) -> tuple[Assumption, ...]:
    """Gather assumptions declared by ``spec.loss``, ``spec.penalty``, and
    each term in ``spec.features``, in declared order."""
    collected: list[Assumption] = []
    loss = getattr(spec, "loss", None)
    if loss is not None:
        for a in getattr(loss, "assumptions", ()) or ():
            collected.append(a)
    penalty = getattr(spec, "penalty", None)
    if penalty is not None:
        for a in getattr(penalty, "assumptions", ()) or ():
            collected.append(a)
    features = getattr(spec, "features", ()) or ()
    for term in features:
        for a in getattr(term, "assumptions", ()) or ():
            collected.append(a)
    return tuple(collected)


def _skipped_for_solution(a: Assumption, solution: Any | None) -> CheckResult | None:
    if getattr(a, "requires_solution", False) and solution is None:
        return CheckResult(
            name=getattr(a, "name", type(a).__name__),
            severity=getattr(a, "severity", Severity.INFO),
            passed=True,
            message="skipped: no solution available (post-fit check)",
            statistic=None,
            threshold=None,
            suggestion=None,
        )
    return None


def _skipped_for_cv(a: Assumption, cv: Any | None) -> CheckResult | None:
    if getattr(a, "requires_cv", False) and cv is None:
        return CheckResult(
            name=getattr(a, "name", type(a).__name__),
            severity=getattr(a, "severity", Severity.SOFT),
            passed=True,
            message="skipped: no CV available (stability check)",
            statistic=None,
            threshold=None,
            suggestion=None,
        )
    return None


def run_assumptions(
    spec: Any,
    data: Any,
    *,
    solution: Any | None = None,
    cv: Any | None = None,
    on_violation: str = "raise",
    suppress: Iterable[type] = (),
    classical_inference: bool = False,
) -> AssumptionReport:
    """Run all assumptions declared on ``spec`` and return a report.

    Parameters
    ----------
    spec
        Any object exposing the spec contract (see module docstring).
    data
        A ``pd.DataFrame`` (or ``DataFrame``-like).
    solution
        A fitted solution. When ``None``, post-fit checks
        (``requires_solution=True``) are skipped with an explanatory pass.
    cv
        A cross-validation object. When ``None``, stability checks
        (``requires_cv=True``) are skipped with an explanatory pass.
    on_violation
        ``"raise"`` (default) raises :class:`AssumptionError` on the first
        HARD failure. ``"warn"`` converts HARD failures into warnings.
        ``"ignore"`` records the failure but neither warns nor raises.
        SOFT failures *always* warn (never raise), except under
        ``"ignore"``. INFO results are silent.
    suppress
        Iterable of assumption classes (not instances) to drop entirely.
    classical_inference
        When ``False`` (default), INFO-severity assumptions are dropped
        before running — they don't appear in the report. When ``True``,
        they run and are recorded at INFO severity.
    """
    if on_violation not in _VALID_ON_VIOLATION:
        raise ValueError(
            f"on_violation must be one of {_VALID_ON_VIOLATION!r}; "
            f"got {on_violation!r}"
        )

    suppress_set = tuple(suppress)
    assumptions = _collect_assumptions(spec)

    results: list[CheckResult] = []
    for a in assumptions:
        if isinstance(a, suppress_set):
            continue
        sev = getattr(a, "severity", Severity.INFO)
        if sev is Severity.INFO and not classical_inference:
            continue

        skipped = _skipped_for_solution(a, solution) or _skipped_for_cv(a, cv)
        if skipped is not None:
            results.append(skipped)
            continue

        try:
            res = a.check(spec, data, solution=solution, cv=cv)
        except AssumptionError:
            raise
        except Exception as exc:  # noqa: BLE001
            res = CheckResult(
                name=getattr(a, "name", type(a).__name__),
                severity=sev,
                passed=False,
                message=f"check raised {type(exc).__name__}: {exc}",
                statistic=None,
                threshold=None,
                suggestion=None,
            )

        results.append(res)

        if not res.passed:
            if res.severity is Severity.HARD:
                if on_violation == "raise":
                    raise AssumptionError(f"{res.name}: {res.message}")
                if on_violation == "warn":
                    warnings.warn(f"{res.name}: {res.message}", stacklevel=2)
                # "ignore" => silently recorded
            elif res.severity is Severity.SOFT:
                if on_violation != "ignore":
                    warnings.warn(f"{res.name}: {res.message}", stacklevel=2)
            # INFO => silent by contract.

    return AssumptionReport(results=tuple(results))


def check_assumptions(
    spec: Any,
    data: Any,
    *,
    solution: Any | None = None,
    classical_inference: bool = False,
) -> AssumptionReport:
    """Standalone assumption reporter.

    Equivalent to :func:`run_assumptions` with ``on_violation='ignore'`` —
    it returns the full report without raising. Used for regulatory
    documentation passes (see DESIGN.md §10).
    """
    return run_assumptions(
        spec,
        data,
        solution=solution,
        on_violation="ignore",
        classical_inference=classical_inference,
    )
