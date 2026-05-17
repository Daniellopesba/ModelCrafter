r"""Weight-of-Evidence assumptions (DESIGN.md §4.3).

Four checks declared by :class:`~model_crafter.terms.woe.WoETerm` (and a
subset by :class:`~model_crafter.terms.woe.BinnedTerm`):

* :class:`AtLeastOneEventPerBin` — HARD. Each bin has at least one event
  and at least one non-event. Without this WoE is undefined (or infinite,
  modulo the Laplace smoothing the package applies).
* :class:`MinimumBinSize` — HARD. Each bin contains at least
  ``min_fraction * n`` rows; without this the bin's contribution is
  noise.
* :class:`MonotonicEventRate` — HARD when the binning strategy is
  monotonic. Event rates across ordered bins must be sign-consistent (the
  WoE encoding's defining promise).
* :class:`WoEMonotonicityPreserved` — SOFT, post-fit. The joint logistic
  regression coefficient on a WoE-encoded column is positive, meaning the
  multivariate fit agrees with the univariate WoE encoding. A negative
  coefficient signals a sign-reversal in the joint model and is a useful
  Simpson's-paradox detector (DESIGN.md §3.1).

References
----------
* Siddiqi (2006), *Credit Risk Scorecards*, §6.
* Anderson (2007), *The Credit Scoring Toolkit*.
* Hastie / Tibshirani / Friedman, *ESL* (2nd ed.), §5.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from model_crafter.assumptions._types import CheckResult, Severity

__all__ = [
    "AtLeastOneEventPerBin",
    "MinimumBinSize",
    "MonotonicEventRate",
    "WoEMonotonicityPreserved",
]


def _binning_terms(spec: Any) -> list[Any]:
    """Return the list of WoE/Binned terms in ``spec.features`` (lazy import)."""
    from model_crafter.terms.woe import BinnedTerm, WoETerm

    return [
        t
        for t in getattr(spec, "features", ()) or ()
        if isinstance(t, (WoETerm, BinnedTerm))
    ]


@dataclass(frozen=True, slots=True)
class AtLeastOneEventPerBin:
    r"""Every WoE / binned-term bin contains at least one event and one non-event.

    Severity: HARD. Without this the un-smoothed WoE is :math:`\pm \infty`;
    with Laplace smoothing it is merely large, but the bin's information
    value is then driven by the smoothing constant rather than the data.
    Coarsen the binning or drop the offending bin.

    Pre-smoothing counts come from each term's
    :class:`~model_crafter.terms.woe.BinningResult`.
    """

    name: str = "AtLeastOneEventPerBin"
    severity: Severity = Severity.HARD
    requires_solution: bool = False
    requires_cv: bool = False

    def describe(self) -> str:
        return "Each WoE/binned-term bin has >= 1 event and >= 1 non-event."

    def check(
        self,
        spec: Any,
        data: Any,
        *,
        solution: Any | None = None,
        cv: Any | None = None,
    ) -> CheckResult:
        terms = _binning_terms(spec)
        if not terms:
            return _passed_skip(self, "no WoE/binned terms to check")

        offenders: list[str] = []
        for t in terms:
            r = getattr(t, "fitted", None)
            if r is None:
                continue  # term not yet fitted; another check (or solve) will catch it
            for label, ev, ne in zip(r.bin_labels, r.n_events, r.n_nonevents, strict=True):
                if ev <= 0 or ne <= 0:
                    offenders.append(f"{t.column}: bin {label!r} (events={ev}, non-events={ne})")
        if offenders:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                passed=False,
                message=(
                    "bins with zero events or zero non-events: "
                    + "; ".join(offenders)
                ),
                statistic=float(len(offenders)),
                threshold=0.0,
                suggestion=(
                    "Coarsen the binning strategy (e.g. larger min_bin_size) "
                    "or use mc.manual(...) to merge problematic bins. "
                    "Laplace smoothing keeps the math finite but does not fix "
                    "the underlying lack of signal."
                ),
            )
        return CheckResult(
            name=self.name,
            severity=self.severity,
            passed=True,
            message=f"all bins have >= 1 event and >= 1 non-event across {len(terms)} term(s)",
            statistic=0.0,
            threshold=0.0,
            suggestion=None,
        )


@dataclass(frozen=True, slots=True)
class MinimumBinSize:
    r"""Every WoE / binned-term bin contains at least ``min_fraction * n`` rows.

    Severity: HARD. The default 5% threshold matches the credit-industry
    rule of thumb (Siddiqi 2006). The check uses the post-fit per-bin
    counts stored on each term.
    """

    min_fraction: float = 0.05
    name: str = "MinimumBinSize"
    severity: Severity = Severity.HARD
    requires_solution: bool = False
    requires_cv: bool = False

    def describe(self) -> str:
        return f"Each WoE/binned-term bin has >= {self.min_fraction:.0%} of rows."

    def check(
        self,
        spec: Any,
        data: Any,
        *,
        solution: Any | None = None,
        cv: Any | None = None,
    ) -> CheckResult:
        terms = _binning_terms(spec)
        if not terms:
            return _passed_skip(self, "no WoE/binned terms to check")

        offenders: list[str] = []
        worst_frac = 1.0
        for t in terms:
            r = getattr(t, "fitted", None)
            if r is None:
                continue
            n_total = sum(r.n_events) + sum(r.n_nonevents)
            if n_total == 0:
                continue
            for label, ev, ne in zip(r.bin_labels, r.n_events, r.n_nonevents, strict=True):
                frac = (ev + ne) / n_total
                if frac < self.min_fraction:
                    offenders.append(
                        f"{t.column}: bin {label!r} has {frac:.1%} of rows"
                    )
                worst_frac = min(worst_frac, frac)
        if offenders:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                passed=False,
                message=(
                    f"bins below {self.min_fraction:.0%} threshold: "
                    + "; ".join(offenders)
                ),
                statistic=float(worst_frac),
                threshold=float(self.min_fraction),
                suggestion=(
                    "Use a coarser binning strategy (higher min_bin_size) "
                    "or merge low-volume bins via mc.manual(...). Tiny bins "
                    "produce unstable WoE values that overfit the training set."
                ),
            )
        return CheckResult(
            name=self.name,
            severity=self.severity,
            passed=True,
            message=(
                f"smallest bin holds {worst_frac:.1%} of rows "
                f"(threshold {self.min_fraction:.0%})"
            ),
            statistic=float(worst_frac),
            threshold=float(self.min_fraction),
            suggestion=None,
        )


@dataclass(frozen=True, slots=True)
class MonotonicEventRate:
    r"""Event rates across ordered bins are monotonic (sign-consistent).

    Severity: HARD when the binning strategy is monotonic; otherwise the
    check passes trivially. Numeric monotonicity is checked on the
    natural ordering of the bins. The ``(Missing)`` bin (if present) is
    excluded from the monotonicity test — missingness rarely has a
    monotone relationship with the underlying variable and is not part
    of the strategy's promise.
    """

    name: str = "MonotonicEventRate"
    severity: Severity = Severity.HARD
    requires_solution: bool = False
    requires_cv: bool = False

    def describe(self) -> str:
        return "Event rates across ordered bins are monotonic for mc.monotonic strategies."

    def check(
        self,
        spec: Any,
        data: Any,
        *,
        solution: Any | None = None,
        cv: Any | None = None,
    ) -> CheckResult:
        # Lazy import to avoid circular dependency.
        from model_crafter.terms.woe import MonotonicBinning

        terms = _binning_terms(spec)
        applicable = [
            t
            for t in terms
            if isinstance(getattr(t, "binning", None), MonotonicBinning)
        ]
        if not applicable:
            return _passed_skip(
                self, "no monotonic-binning WoE/binned terms to check"
            )

        offenders: list[str] = []
        for t in applicable:
            r = getattr(t, "fitted", None)
            if r is None:
                continue
            # Exclude the (Missing) bin when present.
            n = r.n_bins - (1 if r.has_missing_bin else 0)
            if n <= 1:
                continue
            rates = np.asarray(r.event_rate[:n], dtype=float)
            diffs = np.diff(rates)
            inc = np.all(diffs >= 0)
            dec = np.all(diffs <= 0)
            if not (inc or dec):
                offenders.append(
                    f"{t.column}: event rates {np.round(rates, 4).tolist()} are not monotonic"
                )
        if offenders:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                passed=False,
                message="non-monotonic event rate(s): " + "; ".join(offenders),
                statistic=float(len(offenders)),
                threshold=0.0,
                suggestion=(
                    "Tighten min_bin_size or fall back to mc.binned(...) (ESL §5.2) "
                    "if the underlying relationship is genuinely non-monotonic."
                ),
            )
        return CheckResult(
            name=self.name,
            severity=self.severity,
            passed=True,
            message=f"event rates monotonic across {len(applicable)} term(s)",
            statistic=0.0,
            threshold=0.0,
            suggestion=None,
        )


@dataclass(frozen=True, slots=True)
class WoEMonotonicityPreserved:
    r"""The joint coefficient on each WoE-encoded column is positive (post-fit).

    Severity: SOFT. A negative joint coefficient signals that the
    multivariate model has flipped the direction of evidence relative to
    the univariate WoE encoding — typically caused by a confounder
    (Simpson's-paradox style; DESIGN.md §3.1). The user is nudged toward
    :func:`mc.binned` so the joint fit can learn per-bin coefficients
    independently of the marginal sign.

    Implementation: reads the joint coefficient from
    ``solution.coefficients[term.column]``. Only fires for
    :class:`~model_crafter.terms.woe.WoETerm`; bin-indicator terms do not
    carry a single WoE coefficient.
    """

    name: str = "WoEMonotonicityPreserved"
    severity: Severity = Severity.SOFT
    requires_solution: bool = True
    requires_cv: bool = False

    def describe(self) -> str:
        return (
            "Joint coefficient on each WoE-encoded column is positive — "
            "univariate and multivariate evidence agree."
        )

    def check(
        self,
        spec: Any,
        data: Any,
        *,
        solution: Any | None = None,
        cv: Any | None = None,
    ) -> CheckResult:
        # Lazy import for the WoETerm type check.
        from model_crafter.terms.woe import WoETerm

        if solution is None:
            return _passed_skip(self, "no solution available (post-fit check)")

        terms = [
            t
            for t in getattr(spec, "features", ()) or ()
            if isinstance(t, WoETerm)
        ]
        if not terms:
            return _passed_skip(self, "no WoE terms in spec")

        coefs = getattr(solution, "coefficients", None)
        if coefs is None:
            return _passed_skip(self, "solution lacks .coefficients")

        offenders: list[str] = []
        smallest = float("inf")
        for t in terms:
            try:
                beta = float(coefs[t.column])
            except (KeyError, ValueError):
                continue
            smallest = min(smallest, beta)
            if beta < 0:
                offenders.append(f"{t.column}: joint beta = {beta:.4f} (< 0)")
        if not np.isfinite(smallest):
            return _passed_skip(self, "no matching WoE coefficients found in solution")
        if offenders:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                passed=False,
                message=(
                    "joint coefficient(s) on WoE-encoded column(s) are negative: "
                    + "; ".join(offenders)
                ),
                statistic=float(smallest),
                threshold=0.0,
                suggestion=(
                    "The multivariate effect contradicts the univariate WoE encoding; "
                    "consider mc.binned(...) to let the joint model learn bin "
                    "coefficients independently."
                ),
            )
        return CheckResult(
            name=self.name,
            severity=self.severity,
            passed=True,
            message=(
                f"all WoE-encoded coefficients >= 0 "
                f"(smallest = {smallest:.4f})"
            ),
            statistic=float(smallest),
            threshold=0.0,
            suggestion=None,
        )


# Helpers


def _passed_skip(check: Any, message: str) -> CheckResult:
    return CheckResult(
        name=check.name,
        severity=check.severity,
        passed=True,
        message=f"skipped: {message}",
        statistic=None,
        threshold=None,
        suggestion=None,
    )
