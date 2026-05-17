"""Core types for the assumption framework.

This module exists so that concrete assumption modules
(:mod:`model_crafter.assumptions.prerequisites`,
:mod:`~.stability`, :mod:`~.classical`) can import :class:`Severity`,
:class:`CheckResult`, :class:`Assumption`, and :class:`AssumptionError`
without a circular import through the package's ``__init__``.

End users should import these from :mod:`model_crafter.assumptions`, not
from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class Severity(Enum):
    """Three-tier severity for assumption checks (DESIGN.md §4.1)."""

    HARD = "hard"
    SOFT = "soft"
    INFO = "info"


class AssumptionError(Exception):
    """Raised when a HARD assumption fails under ``on_violation='raise'``."""


@dataclass(frozen=True, slots=True)
class CheckResult:
    """The outcome of running a single :class:`Assumption`.

    Fields:

    name
        Human-readable name of the check (matches the assumption's
        ``name``).
    severity
        The :class:`Severity` of the check.
    passed
        ``True`` iff the check did not detect a violation. By convention a
        check that cannot run (missing solution / missing CV) returns
        ``passed=True`` with an explanatory message — silent failures are
        not allowed (DESIGN.md §9).
    message
        Short, human-readable description of the outcome.
    statistic
        The numerical test statistic, when applicable (e.g., Shapiro-Wilk
        W, Breusch-Pagan LM, Durbin-Watson d, max VIF, rank deficit).
    threshold
        The cutoff used for the pass/fail decision, when applicable.
    suggestion
        A short remediation hint (used by :class:`~.classical.LowVIF` to
        point at regularization per ESL §3.4.1).
    """

    name: str
    severity: Severity
    passed: bool
    message: str
    statistic: float | None
    threshold: float | None
    suggestion: str | None


@runtime_checkable
class Assumption(Protocol):
    """Protocol every concrete assumption implements (DESIGN.md §4.1).

    Attributes (read via ``@property`` so frozen-dataclass implementations
    satisfy the protocol cleanly):

    name
        Stable identifier, used for ``suppress=`` filtering and report
        keys.
    severity
        :class:`Severity` of the assumption.
    requires_solution
        ``True`` for post-fit checks (e.g., residual-based). The
        orchestrator passes ``solution=None`` until the model has been
        solved; the assumption's :meth:`check` is expected to return a
        "skipped" pass in that case.
    requires_cv
        ``True`` for stability checks. The orchestrator passes
        ``cv=None`` until cross-validation is available (Phase 3); again,
        the check is expected to return a "skipped" pass.

    Methods:

    describe()
        One-line description of what the assumption tests.
    check(spec, data, *, solution=None, cv=None)
        Run the check and return a :class:`CheckResult`.
    """

    @property
    def name(self) -> str: ...

    @property
    def severity(self) -> Severity: ...

    @property
    def requires_solution(self) -> bool: ...

    @property
    def requires_cv(self) -> bool: ...

    def describe(self) -> str: ...

    def check(
        self,
        spec: Any,
        data: Any,
        *,
        solution: Any | None = None,
        cv: Any | None = None,
    ) -> CheckResult: ...
