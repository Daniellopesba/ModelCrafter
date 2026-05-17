"""Logistic-loss assumptions (DESIGN.md §4.3, AGENTS.md Task P3.A).

Four checks live here, declared on :class:`~model_crafter.loss.LogisticLoss`:

* :class:`BinaryOrProportionTarget` — HARD prerequisite. The target column
  is either binary (values in :math:`\\{0, 1\\}`) or a proportion (values
  in :math:`[0, 1]`). The latter is allowed because fractional-binomial
  logistic regression is a valid use case (rate models, aggregated
  micro-tabulations).
* :class:`NoPerfectSeparation` — HARD, post-fit. Fires when the fitted
  coefficient magnitude or non-convergence signals perfect separation.
  The error message embeds the ESL §4.4.2 ridge remedy verbatim, so the
  user is told the fix at the point of failure.
* :class:`ClassBalance` — SOFT. The minority-class fraction is at least
  ``min_minority`` (default 1%).
* :class:`LinkAdequacy` — INFO. Opt-in classical-inference check. A
  Hosmer-Lemeshow-style decile chi-squared test using only numpy/scipy
  (statsmodels is not a runtime dependency).

The Hosmer-Lemeshow check is the FDA / regulatory documentation tool;
held-out predictive performance via the :class:`PerformanceReport`
remains the package's preferred adequacy diagnostic (DESIGN.md §4.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import chi2

from model_crafter.assumptions._common import materialise_design
from model_crafter.assumptions._types import CheckResult, Severity

# BinaryOrProportionTarget (HARD)


@dataclass(frozen=True, slots=True)
class BinaryOrProportionTarget:
    """Target column is binary or a proportion in :math:`[0, 1]`.

    Two acceptable shapes:

    * **Binary** — every value is in :math:`\\{0, 1\\}` (after coercion to
      float; the column can be int, float, or bool dtype).
    * **Proportion** — every value lies in :math:`[0, 1]`.

    Constants or NaN/inf values fail the check.
    """

    name: str = "BinaryOrProportionTarget"
    severity: Severity = Severity.HARD
    requires_solution: bool = False
    requires_cv: bool = False

    def describe(self) -> str:
        return "target column is binary {0,1} or a proportion in [0, 1]"

    def check(
        self,
        spec: Any,
        data: Any,
        *,
        solution: Any | None = None,
        cv: Any | None = None,
    ) -> CheckResult:
        target = spec.target
        if target not in data.columns:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                passed=False,
                message=f"target column '{target}' not found in data",
                statistic=None,
                threshold=None,
                suggestion=None,
            )
        series = pd.to_numeric(data[target], errors="coerce")
        y: np.ndarray = np.asarray(series, dtype=float)
        if not np.isfinite(y).all():
            bad = int(np.sum(~np.isfinite(y)))
            return CheckResult(
                name=self.name,
                severity=self.severity,
                passed=False,
                message=(
                    f"target '{target}' has {bad} non-finite value(s); "
                    "drop or impute before solving (DESIGN.md §9.8)."
                ),
                statistic=float(bad),
                threshold=0.0,
                suggestion=None,
            )

        # Check range [0, 1] first; if all values are in {0, 1}, it's binary.
        ymin = float(np.min(y))
        ymax = float(np.max(y))
        if ymin < 0.0 or ymax > 1.0:
            # Sample a few violating values for the message.
            mask = (y < 0.0) | (y > 1.0)
            bad_values = sorted({float(v) for v in y[mask][:5]})
            return CheckResult(
                name=self.name,
                severity=self.severity,
                passed=False,
                message=(
                    f"target '{target}' has values outside [0, 1] "
                    f"(min={ymin:.4g}, max={ymax:.4g}); examples: {bad_values}. "
                    "Use loss=mc.squared_error for continuous regression "
                    "or recode the target as binary/proportion."
                ),
                statistic=max(abs(ymin), abs(ymax)),
                threshold=1.0,
                suggestion=None,
            )

        is_binary = bool(np.all((y == 0.0) | (y == 1.0)))
        kind = "binary" if is_binary else "proportion"
        return CheckResult(
            name=self.name,
            severity=self.severity,
            passed=True,
            message=f"target '{target}' is {kind} (range [{ymin:.3g}, {ymax:.3g}])",
            statistic=None,
            threshold=None,
            suggestion=None,
        )


# NoPerfectSeparation (HARD, post-fit)


_L2_REMEDY = (
    "penalty=mc.l2(...) is the standard remedy for perfect separation "
    "(ESL §4.4.2)."
)


@dataclass(frozen=True, slots=True)
class NoPerfectSeparation:
    """The fitted logistic model is not in the perfect-separation regime.

    Two converging symptoms can trigger failure:

    1. **Large coefficient magnitude.** Any fitted slope has magnitude
       exceeding ``coef_magnitude_max`` (default :math:`10^3`).
       Perfect-separation MLEs are unbounded; in a finite-iteration IRLS
       run they accumulate at very large magnitudes before the iter cap.
    2. **Saturated probability predictions.** The fraction of rows for
       which the fitted probability is within ``prob_eps`` of 0 or 1
       exceeds ``saturated_frac_max``. Truly separable data drives every
       prediction to within machine precision of the boundary; a
       well-conditioned logistic fit rarely saturates more than a small
       fraction of points. This catches the case where IRLS's
       ``coef_magnitude_max`` threshold isn't hit because the iter cap
       interrupted the runaway early, but every observation is already
       perfectly classified.

    Either symptom is sufficient to fail the check. The post-fit pass in
    ``solve/__init__.py`` hands the check a ``SimpleNamespace``-style stub
    rather than the full :class:`~model_crafter.solution.Solution`, so we
    cannot rely on a ``converged`` attribute being present — the
    probability-saturation signal compensates for that.

    The failure message embeds the ESL §4.4.2 ridge remedy verbatim:

        ``"penalty=mc.l2(...) is the standard remedy for perfect
        separation (ESL §4.4.2)."``

    so the user is told the fix at the point of failure. (DESIGN.md §11
    "Numerical stability in IRLS for separable data" — the lean is
    "detect and raise; nudge the user toward ``penalty=mc.l2(...)``".)
    """

    coef_magnitude_max: float = 1e3
    prob_eps: float = 1e-4
    saturated_frac_max: float = 0.95
    name: str = "NoPerfectSeparation"
    severity: Severity = Severity.HARD
    requires_solution: bool = True
    requires_cv: bool = False

    def describe(self) -> str:
        return (
            "fitted logistic coefficients are bounded and predicted "
            "probabilities are not uniformly saturated (ESL §4.4.2)"
        )

    def check(
        self,
        spec: Any,
        data: Any,
        *,
        solution: Any | None = None,
        cv: Any | None = None,
    ) -> CheckResult:
        if solution is None:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                passed=True,
                message="skipped: no solution available (post-fit check)",
                statistic=None,
                threshold=None,
                suggestion=None,
            )

        # Inspect the *slopes* — the intercept can legitimately be very
        # negative for severely imbalanced classes.
        coefs: pd.Series = solution.coefficients
        slopes = coefs.drop(labels="(Intercept)", errors="ignore")
        max_mag = (
            0.0
            if len(slopes) == 0
            else float(np.max(np.abs(slopes.to_numpy(dtype=float))))
        )

        # Compute saturated-prediction fraction. This requires rebuilding
        # the design matrix and computing eta = X @ beta. We do the
        # cheapest possible materialisation (numpy + spec.target column).
        sat_frac = float("nan")
        try:
            X, columns = materialise_design(spec, data)
            beta = np.asarray(
                [coefs[c] for c in columns], dtype=float
            )
            eta = X @ beta
            p_hat = expit(eta)
            saturated = (p_hat < self.prob_eps) | (p_hat > 1.0 - self.prob_eps)
            sat_frac = float(saturated.mean())
        except (KeyError, ValueError):
            # Materialisation failure — leave sat_frac as NaN and fall
            # through to the magnitude-only check.
            pass

        magnitude_failure = max_mag > self.coef_magnitude_max
        saturation_failure = (
            not np.isnan(sat_frac) and sat_frac > self.saturated_frac_max
        )

        if magnitude_failure or saturation_failure:
            offenders = [
                f"{name}={slopes[name]:.3g}"
                for name in slopes.index
                if abs(float(slopes[name]))
                > min(self.coef_magnitude_max / 10.0, 10.0)
            ]
            return CheckResult(
                name=self.name,
                severity=self.severity,
                passed=False,
                message=(
                    "perfect (or near-perfect) separation detected: "
                    f"max|slope|={max_mag:.3g}, "
                    f"saturated_fraction={sat_frac:.3f}. "
                    f"Offending coefficients: {offenders}. " + _L2_REMEDY
                ),
                statistic=max_mag,
                threshold=self.coef_magnitude_max,
                suggestion=_L2_REMEDY,
            )

        return CheckResult(
            name=self.name,
            severity=self.severity,
            passed=True,
            message=(
                f"max|slope|={max_mag:.3g} <= {self.coef_magnitude_max}, "
                f"saturated_fraction={sat_frac:.3f}"
            ),
            statistic=max_mag,
            threshold=self.coef_magnitude_max,
            suggestion=None,
        )


# ClassBalance (SOFT)


@dataclass(frozen=True, slots=True)
class ClassBalance:
    """Minority-class fraction is at least ``min_minority``.

    For a binary target :math:`y` with positive-class fraction
    :math:`p = \\bar y`, the *minority* fraction is :math:`\\min(p, 1-p)`.
    SOFT — fires as a warning when the minority is below ``min_minority``
    (default 1%); this is a *data sanity* check, not a math prerequisite.

    For proportion targets the same statistic is informative (it measures
    the mass concentration near the boundaries).
    """

    min_minority: float = 0.01
    name: str = "ClassBalance"
    severity: Severity = Severity.SOFT
    requires_solution: bool = False
    requires_cv: bool = False

    def describe(self) -> str:
        return f"minority-class fraction >= {self.min_minority}"

    def check(
        self,
        spec: Any,
        data: Any,
        *,
        solution: Any | None = None,
        cv: Any | None = None,
    ) -> CheckResult:
        target = spec.target
        series = pd.to_numeric(data[target], errors="coerce")
        y: np.ndarray = np.asarray(series, dtype=float)
        # NaN / inf are caught by BinaryOrProportionTarget; we guard
        # defensively here.
        if not np.isfinite(y).all():
            return CheckResult(
                name=self.name,
                severity=self.severity,
                passed=True,
                message="skipped: target has non-finite values",
                statistic=None,
                threshold=self.min_minority,
                suggestion=None,
            )
        p = float(np.mean(y))
        minority = float(min(p, 1.0 - p))
        passed = minority >= self.min_minority
        return CheckResult(
            name=self.name,
            severity=self.severity,
            passed=passed,
            message=(
                f"minority fraction = {minority:.4g} "
                f"(threshold {self.min_minority})"
            ),
            statistic=minority,
            threshold=self.min_minority,
            suggestion=(
                None
                if passed
                else (
                    "Severe class imbalance — consider stratified sampling, "
                    "class weights (``weights=``), or aggregating rare classes."
                )
            ),
        )


# LinkAdequacy (INFO, opt-in) — Hosmer-Lemeshow decile chi-square


@dataclass(frozen=True, slots=True)
class LinkAdequacy:
    r"""Hosmer-Lemeshow goodness-of-fit (ESL §4.4 / Hosmer & Lemeshow 1980).

    Bins observations into ``n_bins`` deciles of predicted probability,
    then forms the statistic

    .. math::

        \widehat{C}
        \;=\; \sum_{g=1}^{G} \frac{(O_g - E_g)^2}{E_g\,(1 - \bar p_g)},

    where :math:`O_g` and :math:`E_g` are observed and expected positive
    counts in group :math:`g`, and :math:`\bar p_g` is the mean predicted
    probability in that group. Under the null (correct link, well-specified
    model) :math:`\widehat{C}` is approximately :math:`\chi^2_{G-2}`.

    Passes when the p-value exceeds ``alpha`` (default 0.05).

    INFO severity — this is classical-inference output for regulatory
    documentation (DESIGN.md §4.3). Held-out predictive validity via
    :class:`~model_crafter.assumptions.PredictiveStability` remains the
    package's preferred adequacy diagnostic.

    The check is implementable without statsmodels (numpy + scipy only);
    statsmodels' diagnostics module is acceptable as a *test* dependency
    but not at runtime.
    """

    n_bins: int = 10
    alpha: float = 0.05
    name: str = "LinkAdequacy"
    severity: Severity = Severity.INFO
    requires_solution: bool = True
    requires_cv: bool = False

    def describe(self) -> str:
        return (
            f"Hosmer-Lemeshow chi^2 on {self.n_bins} probability deciles "
            f"(alpha={self.alpha})"
        )

    def check(
        self,
        spec: Any,
        data: Any,
        *,
        solution: Any | None = None,
        cv: Any | None = None,
    ) -> CheckResult:
        if solution is None:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                passed=True,
                message="skipped: no solution available (post-fit check)",
                statistic=None,
                threshold=None,
                suggestion=None,
            )
        # Build the design and compute predicted probabilities. The check
        # lives at INFO severity, so a materialisation failure shouldn't
        # break the whole report — return a graceful skip with the error
        # explained.
        try:
            X, columns = materialise_design(spec, data)
            beta = np.asarray(
                [solution.coefficients[c] for c in columns], dtype=float
            )
            eta = X @ beta
            p_hat = expit(eta)
            y_series = pd.to_numeric(data[spec.target], errors="coerce")
            y = np.asarray(y_series, dtype=float)
        except (KeyError, ValueError) as exc:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                passed=True,
                message=f"skipped: could not materialise predictions ({exc})",
                statistic=None,
                threshold=self.alpha,
                suggestion=None,
            )

        # Decile binning by predicted probability. ``qcut`` may fail when
        # ``p_hat`` is concentrated (few distinct values); handle by
        # collapsing duplicate edges.
        try:
            bins = pd.qcut(p_hat, q=self.n_bins, duplicates="drop")
        except ValueError as exc:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                passed=True,
                message=f"skipped: cannot bin predicted probabilities ({exc})",
                statistic=None,
                threshold=self.alpha,
                suggestion=None,
            )

        df = pd.DataFrame({"y": y, "p": p_hat, "bin": bins})
        grouped = df.groupby("bin", observed=True)
        Og = grouped["y"].sum().to_numpy(dtype=float)
        Eg = grouped["p"].sum().to_numpy(dtype=float)
        pg = grouped["p"].mean().to_numpy(dtype=float)

        # Effective number of groups after merging duplicates.
        G = int(Og.shape[0])
        if G < 3:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                passed=True,
                message=(
                    f"skipped: too few effective probability bins "
                    f"(G={G}); concentrate predictions or use more data"
                ),
                statistic=None,
                threshold=self.alpha,
                suggestion=None,
            )

        denom = Eg * (1.0 - pg)
        # Avoid divide-by-zero in degenerate-bin edge cases.
        mask = denom > 0
        if not mask.any():
            return CheckResult(
                name=self.name,
                severity=self.severity,
                passed=True,
                message="skipped: degenerate Hosmer-Lemeshow denominator",
                statistic=None,
                threshold=self.alpha,
                suggestion=None,
            )
        stat = float(np.sum(((Og - Eg) ** 2)[mask] / denom[mask]))
        dof = G - 2
        if dof < 1:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                passed=True,
                message=f"skipped: insufficient degrees of freedom (G={G})",
                statistic=stat,
                threshold=self.alpha,
                suggestion=None,
            )
        p_value = float(chi2.sf(stat, dof))
        passed = p_value > self.alpha
        return CheckResult(
            name=self.name,
            severity=self.severity,
            passed=passed,
            message=(
                f"Hosmer-Lemeshow chi^2 = {stat:.3f} (G={G}, df={dof}, "
                f"p={p_value:.4g}, alpha={self.alpha})"
            ),
            statistic=stat,
            threshold=self.alpha,
            suggestion=(
                None
                if passed
                else (
                    "Predicted probabilities depart from observed event rates "
                    "in some deciles. Consider basis expansions (Phase 4) for "
                    "non-linear effects or refitting with adjusted features."
                )
            ),
        )
