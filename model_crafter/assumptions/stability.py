"""SOFT stability assumptions (DESIGN.md §4.3).

These are held-out / resampling diagnostics — ESL §7 applied as automatic
checks. They require a CV object to actually run; until P3.B ships the
CV runner, the implementations are deliberate stubs that return a
``passed=True`` result with an explanatory message when ``cv`` is
``None`` (DESIGN.md §9, "no silent failures").

Currently implemented as stubs:

* :class:`CoefficientStability` — max(SD/|mean|) across CV folds vs
  threshold.
* :class:`PredictiveStability` — CV metric SD relative to mean vs
  threshold.

When ``cv`` is provided (Phase 3 onwards), the real checks compute the
documented statistic and compare against the threshold.

Fully implemented (no CV needed):

* :class:`ComparableFeatureScales` — declared by ``L1Penalty`` and
  ``L2Penalty`` (DESIGN.md §4.3). Computes the ratio of max to min
  column standard deviation across non-intercept design columns and
  warns when it exceeds ``std_ratio_max``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from model_crafter.assumptions._common import materialise_design
from model_crafter.assumptions._types import CheckResult, Severity


@dataclass(frozen=True, slots=True)
class CoefficientStability:
    """Coefficients are stable across CV folds (ESL §7).

    Statistic: ``max_j SD_j / |mean_j|`` (the coefficient of variation of
    the coefficient of variation, taken across folds). When this exceeds
    ``cv_cv_max``, the joint fit is too sensitive to the held-out fold —
    a sign of high variance or unstable selection.

    Threshold: ``cv_cv_max`` (default 0.2, per DESIGN.md §4.1's example).
    """

    cv_cv_max: float = 0.2
    name: str = "CoefficientStability"
    severity: Severity = Severity.SOFT
    requires_solution: bool = True
    requires_cv: bool = True

    def describe(self) -> str:
        return (
            f"max coefficient CV across CV folds <= {self.cv_cv_max} "
            "(ESL §7 stability)"
        )

    def check(
        self,
        spec: Any,
        data: Any,
        *,
        solution: Any | None = None,
        cv: Any | None = None,
    ) -> CheckResult:
        # Orchestration handles the ``cv is None`` skip path, but we
        # re-check defensively for direct callers.
        if cv is None:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                passed=True,
                message="skipped: no CV available (stability check)",
                statistic=None,
                threshold=self.cv_cv_max,
                suggestion=None,
            )
        return _real_coefficient_stability(self, cv)


@dataclass(frozen=True, slots=True)
class PredictiveStability:
    """Predictive metric is stable across CV folds (ESL §7).

    Statistic: ``SD(metric_folds) / mean(metric_folds)``. When this
    exceeds ``metric_cv_max``, predictive performance is too sensitive
    to the held-out fold to support a confident point estimate.

    Threshold: ``metric_cv_max`` (default 0.1, per DESIGN.md §4.1's
    example).
    """

    metric_cv_max: float = 0.1
    name: str = "PredictiveStability"
    severity: Severity = Severity.SOFT
    requires_solution: bool = True
    requires_cv: bool = True

    def describe(self) -> str:
        return (
            f"CV metric SD / mean <= {self.metric_cv_max} "
            "(ESL §7 stability)"
        )

    def check(
        self,
        spec: Any,
        data: Any,
        *,
        solution: Any | None = None,
        cv: Any | None = None,
    ) -> CheckResult:
        if cv is None:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                passed=True,
                message="skipped: no CV available (stability check)",
                statistic=None,
                threshold=self.metric_cv_max,
                suggestion=None,
            )
        return _real_predictive_stability(self, cv)


# ---------------------------------------------------------------------------
# CV-backed implementations.
#
# These are deliberately small: they expect ``cv`` to expose either a
# ``.coefficients`` DataFrame (rows = folds) or a ``.metrics`` dict /
# DataFrame depending on the check. P3.B will pin the CVResult shape;
# until then, the implementations read the simplest documented shape so
# downstream tests can exercise the real path without waiting on P3.B.
# ---------------------------------------------------------------------------


def _coefficient_matrix(cv: Any) -> pd.DataFrame:
    """Best-effort extraction of fold coefficients.

    Accepted shapes (most specific first):

    * ``cv.coefficients`` — a ``pd.DataFrame`` with one row per fold.
    * ``cv.fold_results`` — a sequence of objects each with a
      ``.coefficients`` ``pd.Series``.
    """
    if hasattr(cv, "coefficients"):
        coefs = cv.coefficients
        if isinstance(coefs, pd.DataFrame):
            return coefs
    fold_results = getattr(cv, "fold_results", None)
    if fold_results is not None:
        rows = []
        for f in fold_results:
            sol = getattr(f, "solution", f)
            rows.append(sol.coefficients)
        return pd.DataFrame(rows).reset_index(drop=True)
    raise AttributeError(
        "CoefficientStability needs cv.coefficients (DataFrame) or "
        "cv.fold_results with .coefficients per fold."
    )


def _metric_series(cv: Any) -> pd.Series:
    """Best-effort extraction of fold metrics (one float per fold)."""
    metrics = getattr(cv, "metrics", None)
    if metrics is not None:
        if isinstance(metrics, pd.Series):
            return metrics.astype(float)
        if isinstance(metrics, pd.DataFrame) and metrics.shape[1] == 1:
            return metrics.iloc[:, 0].astype(float)
    fold_results = getattr(cv, "fold_results", None)
    if fold_results is not None:
        values = []
        for f in fold_results:
            m = getattr(f, "metric", None)
            if m is None:
                m = getattr(f, "value", None)
            if m is None:
                continue
            values.append(float(m))
        if values:
            return pd.Series(values)
    raise AttributeError(
        "PredictiveStability needs cv.metrics (Series) or "
        "cv.fold_results with .metric per fold."
    )


def _real_coefficient_stability(spec: CoefficientStability, cv: Any) -> CheckResult:
    coefs = _coefficient_matrix(cv)
    means = cast(pd.Series, coefs.mean(axis=0)).abs()
    sds = coefs.std(axis=0, ddof=1)
    # Guard against zero means (intercept-free terms with tiny means).
    safe_means = means.where(means > 0, np.nan)
    cv_of_cv = (sds / safe_means).dropna()
    max_stat = float(cv_of_cv.max()) if len(cv_of_cv) else 0.0
    passed = max_stat <= spec.cv_cv_max
    return CheckResult(
        name=spec.name,
        severity=spec.severity,
        passed=passed,
        message=(
            f"max(SD/|mean|) across folds = {max_stat:.3f} "
            f"(threshold {spec.cv_cv_max})"
        ),
        statistic=max_stat,
        threshold=spec.cv_cv_max,
        suggestion=(
            None
            if passed
            else "Coefficients vary substantially across folds — consider "
            "penalty=mc.l2(...) (ESL §3.4.1) or revisiting feature selection."
        ),
    )


def _real_predictive_stability(spec: PredictiveStability, cv: Any) -> CheckResult:
    metrics = _metric_series(cv)
    # ``.mean()`` / ``.std()`` on a 1-D Series return a numpy scalar; cast
    # explicitly to silence pyright's "Series | float" inference.
    mean = float(np.asarray(metrics.mean()).item())
    sd = float(np.asarray(metrics.std(ddof=1)).item())
    if mean == 0:
        stat = float("nan")
        passed = False
    else:
        stat = sd / abs(mean)
        passed = stat <= spec.metric_cv_max
    return CheckResult(
        name=spec.name,
        severity=spec.severity,
        passed=passed,
        message=(
            f"CV metric SD/mean = {stat:.3f} (threshold {spec.metric_cv_max})"
        ),
        statistic=stat,
        threshold=spec.metric_cv_max,
        suggestion=(
            None
            if passed
            else "Predictive performance is unstable across folds — consider "
            "more data, a temporal splitter, or stronger regularization."
        ),
    )


# ---------------------------------------------------------------------------
# ComparableFeatureScales (declared by L1Penalty and L2Penalty)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ComparableFeatureScales:
    r"""Feature scales are comparable enough for L1/L2 to be meaningful.

    L1 and L2 penalties shrink coefficients toward zero by an amount that
    does **not** depend on the column's units (ESL §3.4.1). When one
    feature's standard deviation is orders of magnitude larger than
    another's, the same lambda imposes very different effective
    regularisation on the two coefficients, and the resulting fit becomes
    a function of arbitrary scaling choices rather than the data.

    Statistic
    ---------
    .. math::

        \mathrm{ratio} \;=\; \frac{\max_j \mathrm{std}(X_{:,j})}
                                  {\min_j \mathrm{std}(X_{:,j})}

    taken across the *non-intercept* design columns. Constant columns
    (std = 0) are excluded from the ratio because the intercept also has
    std = 0 and a single sentinel comparison is meaningless. The check
    fires (``passed=False``) when ``ratio > std_ratio_max``.

    Severity
    --------
    SOFT — emitting a warning at solve time when the user has supplied an
    L1 or L2 penalty against features whose scales differ by more than
    two orders of magnitude (default ``std_ratio_max=100``). The
    suggestion field tells the user to standardise.
    """

    std_ratio_max: float = 100.0
    name: str = "ComparableFeatureScales"
    severity: Severity = Severity.SOFT
    requires_solution: bool = False
    requires_cv: bool = False

    def describe(self) -> str:
        return (
            f"max(std)/min(std) across non-intercept feature columns "
            f"<= {self.std_ratio_max} (ESL §3.4.1 — L1/L2 are scale-sensitive)"
        )

    def check(
        self,
        spec: Any,
        data: Any,
        *,
        solution: Any | None = None,
        cv: Any | None = None,
    ) -> CheckResult:
        X, columns = materialise_design(spec, data)
        # Per-column std (population sd with ddof=0; the ratio is invariant
        # to ddof, but ddof=0 avoids NaN when n=1).
        stds = X.std(axis=0, ddof=0)

        # Identify non-intercept, non-constant columns. The intercept is a
        # constant column by construction; if any other column has std=0
        # the user has a degenerate feature and `FullRankDesign` will catch
        # it as a HARD failure — here we just exclude such columns from
        # the scale comparison.
        non_intercept_mask = np.array(
            [c != "(Intercept)" for c in columns], dtype=bool
        )
        positive_std_mask = stds > 0
        keep = non_intercept_mask & positive_std_mask
        n_kept = int(keep.sum())

        if n_kept <= 1:
            # 0 or 1 non-intercept feature: nothing to compare.
            return CheckResult(
                name=self.name,
                severity=self.severity,
                passed=True,
                message=(
                    f"only {n_kept} non-intercept feature column(s); "
                    "scale comparison is degenerate"
                ),
                statistic=None,
                threshold=self.std_ratio_max,
                suggestion=None,
            )

        kept_stds = stds[keep]
        ratio = float(np.max(kept_stds) / np.min(kept_stds))
        passed = ratio <= self.std_ratio_max
        message = (
            f"feature std ratio = {ratio:.1f} "
            f"(threshold {self.std_ratio_max:.1f}); "
            "consider scaling features before applying L1/L2."
        )
        return CheckResult(
            name=self.name,
            severity=self.severity,
            passed=passed,
            message=message,
            statistic=ratio,
            threshold=self.std_ratio_max,
            suggestion=(
                None
                if passed
                else (
                    "Standardise features (zero mean, unit std) before "
                    "fitting penalised models — L1/L2 are scale-sensitive "
                    "(ESL §3.4.1)."
                )
            ),
        )
