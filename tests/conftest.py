"""Test configuration for Task P1.A.

P1.A depends on P1.B's ``model_crafter.assumptions`` public interface for
``Solution.assumptions``, ``solve(...)`` plumbing, and the ``FullRankDesign``
prerequisite. Per AGENTS.md, P1.A treats those names as **opaque** and must
not import P1.B's internal modules.

To keep P1.A's test suite green while P1.B is developed in parallel, this
conftest installs a minimal stub of the assumptions interface into
``sys.modules`` *before* any production module that imports it is loaded.
The stub is intentionally minimal: it implements just enough to satisfy
P1.A's plumbing.

When P1.B merges, this stub is no longer needed; the integration agent
deletes this file (or replaces it with a thin compatibility shim that
re-exports the real module).
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from enum import Enum
from typing import Any


def _install_assumptions_stub() -> None:
    if "model_crafter.assumptions" in sys.modules:
        # Real (or another stub) already present; do nothing.
        return

    mod = types.ModuleType("model_crafter.assumptions")

    class Severity(Enum):
        HARD = "hard"
        SOFT = "soft"
        INFO = "info"

    @dataclass(frozen=True)
    class CheckResult:
        name: str
        severity: Severity
        passed: bool
        message: str
        statistic: float | None = None
        threshold: float | None = None
        suggestion: str | None = None

    @dataclass(frozen=True)
    class AssumptionReport:
        results: tuple = ()

        def by_severity(self) -> dict:
            out: dict[Severity, list[CheckResult]] = {s: [] for s in Severity}
            for r in self.results:
                out[r.severity].append(r)
            return {k: tuple(v) for k, v in out.items()}

        def passed(self) -> bool:
            return all(r.passed for r in self.results)

        def __repr__(self) -> str:
            if not self.results:
                return "AssumptionReport(empty)"
            return "AssumptionReport(" + ", ".join(
                f"{r.name}={'PASS' if r.passed else 'FAIL'}" for r in self.results
            ) + ")"

    class AssumptionError(Exception):
        """Raised when a HARD assumption is violated and on_violation='raise'."""

    @dataclass(frozen=True)
    class FullRankDesign:
        """Stub for the HARD prerequisite asserting design matrix is full rank.

        The real implementation lives in P1.B. The stub just exposes the
        symbol so `squared_error.assumptions` can reference it.
        """

        name: str = "FullRankDesign"
        severity: Severity = Severity.HARD
        requires_solution: bool = False
        requires_cv: bool = False

        def describe(self) -> str:
            return "Design matrix has full column rank."

        def check(
            self,
            spec: Any,
            data: Any,
            *,
            solution: Any = None,
            cv: Any = None,
            design: Any = None,
            offending_columns: tuple[str, ...] | None = None,
        ) -> CheckResult:
            # In the real impl the check inspects the design matrix.
            # The stub trusts the caller (solve) to do that check and pass
            # in offending_columns when a violation is found.
            if offending_columns:
                return CheckResult(
                    name="FullRankDesign",
                    severity=Severity.HARD,
                    passed=False,
                    message=(
                        "Design matrix is rank-deficient; columns linearly dependent "
                        f"on others: {list(offending_columns)}"
                    ),
                )
            return CheckResult(
                name="FullRankDesign",
                severity=Severity.HARD,
                passed=True,
                message="design matrix is full column rank",
            )

    def run_assumptions(
        spec: Any,
        data: Any,
        *,
        solution: Any = None,
        cv: Any = None,
        on_violation: str = "warn",
        suppress: tuple = (),
        classical_inference: bool = False,
        design: Any = None,
        offending_columns: tuple[str, ...] | None = None,
    ) -> AssumptionReport:
        results: list[CheckResult] = []
        # Collect HARD assumptions from the loss + penalty + terms.
        sources: list[Any] = []
        if spec is not None:
            sources.append(getattr(spec, "loss", None))
            sources.append(getattr(spec, "penalty", None))
            for t in getattr(spec, "features", ()) or ():
                sources.append(t)
        seen: set[type] = set()
        for src in sources:
            if src is None:
                continue
            for assumption in getattr(src, "assumptions", ()) or ():
                if type(assumption) in seen:
                    continue
                seen.add(type(assumption))
                if type(assumption) in suppress:
                    continue
                if isinstance(assumption, FullRankDesign):
                    res = assumption.check(
                        spec, data, solution=solution, design=design,
                        offending_columns=offending_columns,
                    )
                else:
                    # Other stub assumptions just pass.
                    res = CheckResult(
                        name=getattr(assumption, "name", type(assumption).__name__),
                        severity=getattr(assumption, "severity", Severity.SOFT),
                        passed=True,
                        message="(stub) ok",
                    )
                results.append(res)
                if not res.passed and res.severity == Severity.HARD and on_violation == "raise":
                    raise AssumptionError(res.message)
        return AssumptionReport(results=tuple(results))

    def check_assumptions(
        spec: Any,
        data: Any,
        *,
        solution: Any = None,
        classical_inference: bool = False,
    ) -> AssumptionReport:
        return run_assumptions(
            spec, data, solution=solution,
            on_violation="ignore",
            classical_inference=classical_inference,
        )

    mod.Severity = Severity  # type: ignore[attr-defined]
    mod.CheckResult = CheckResult  # type: ignore[attr-defined]
    mod.AssumptionReport = AssumptionReport  # type: ignore[attr-defined]
    mod.AssumptionError = AssumptionError  # type: ignore[attr-defined]
    mod.FullRankDesign = FullRankDesign  # type: ignore[attr-defined]
    mod.run_assumptions = run_assumptions  # type: ignore[attr-defined]
    mod.check_assumptions = check_assumptions  # type: ignore[attr-defined]
    mod.__all__ = [  # type: ignore[attr-defined]
        "AssumptionError",
        "AssumptionReport",
        "CheckResult",
        "FullRankDesign",
        "Severity",
        "check_assumptions",
        "run_assumptions",
    ]

    sys.modules["model_crafter.assumptions"] = mod


_install_assumptions_stub()
