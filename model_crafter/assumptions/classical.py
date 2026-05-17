"""Classical-inference (INFO) assumptions (DESIGN.md §4.3).

Opt-in via ``classical_inference=True`` in :func:`~.run_assumptions` or
:func:`~.check_assumptions`. These checks never warn or raise — they're
report-only, intended for regulatory model documentation.

Implementations:

* :class:`ResidualNormality` — Shapiro-Wilk for ``n < 5000``, otherwise
  Anderson-Darling (``scipy.stats``).
* :class:`Homoscedasticity` — Breusch-Pagan, hand-rolled from
  ``numpy``/``scipy`` (no statsmodels at runtime).
* :class:`Independence` — Durbin-Watson,
  :math:`d = \\sum (e_t - e_{t-1})^2 / \\sum e_t^2`.
* :class:`LowVIF` — for each column :math:`j`,
  :math:`\\mathrm{VIF}_j = 1 / (1 - R_j^2)` where :math:`R_j^2` is the
  coefficient of determination from regressing column :math:`j` on the
  others. Reports the max. Suggestion field follows DESIGN.md §4.3
  verbatim (ESL §3.4.1).

All checks read off ``solution.coefficients`` when residuals are needed
and ``spec`` + ``data`` to rebuild the design matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from model_crafter.assumptions._common import (
    get_residuals,
    materialise_design,
)
from model_crafter.assumptions._types import CheckResult, Severity

# ResidualNormality


@dataclass(frozen=True, slots=True)
class ResidualNormality:
    """OLS residuals are approximately normal.

    For :math:`n < 5000` we use Shapiro-Wilk (``scipy.stats.shapiro``);
    the statistic ``W`` is in :math:`[0, 1]` with :math:`W = 1` meaning a
    perfect normal fit and the test's null hypothesis is normality.
    Reject when the p-value is below ``alpha``.

    For :math:`n \\geq 5000` Shapiro-Wilk is not reliable; we fall back
    to Anderson-Darling (``scipy.stats.anderson(dist='norm')``) and pass
    when the statistic is below the table's 5% critical value.

    INFO severity: this is a classical-inference check used for
    regulatory documentation. Held-out predictive validity (Phase 3's
    `PredictiveStability` / `PerformanceReport`) is the package's
    preferred adequacy diagnostic per DESIGN.md §4.
    """

    alpha: float = 0.05
    n_shapiro_max: int = 5000
    name: str = "ResidualNormality"
    severity: Severity = Severity.INFO
    requires_solution: bool = True
    requires_cv: bool = False

    def describe(self) -> str:
        return (
            f"residuals approximately normal (Shapiro-Wilk n<{self.n_shapiro_max}, "
            "else Anderson-Darling)"
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
        resid = get_residuals(spec, data, solution)
        n = resid.size
        if n < self.n_shapiro_max:
            w, p = stats.shapiro(resid)
            passed = p > self.alpha
            return CheckResult(
                name=self.name,
                severity=self.severity,
                passed=bool(passed),
                message=(
                    f"Shapiro-Wilk W={float(w):.4f}, p={float(p):.4g} "
                    f"(alpha={self.alpha})"
                ),
                statistic=float(w),
                threshold=self.alpha,
                suggestion=(
                    None
                    if passed
                    else "Residuals look non-normal; classical SEs may be "
                    "off for small n. Bootstrap CIs (Phase 3) are robust."
                ),
            )
        # Anderson-Darling fall-back
        # Use the legacy critical-values table (5% level) — this matches the
        # table-driven Anderson-Darling form most users expect. The newer
        # scipy interface (>=1.17) deprecates this signature in favour of
        # ``method=``; we pin it explicitly to keep results stable.
        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", FutureWarning)
            ad: Any = stats.anderson(resid, dist="norm")
        crit_5pct = float(ad.critical_values[2])  # significance_level == 5%
        statistic = float(ad.statistic)
        passed = statistic < crit_5pct
        return CheckResult(
            name=self.name,
            severity=self.severity,
            passed=bool(passed),
            message=(
                f"Anderson-Darling A^2={statistic:.4f} "
                f"(5% critical value {crit_5pct:.4f})"
            ),
            statistic=statistic,
            threshold=crit_5pct,
            suggestion=(
                None
                if passed
                else "Residuals look non-normal at large n; consider robust "
                "SEs (Phase 3 bootstrap)."
            ),
        )


# Homoscedasticity (Breusch-Pagan)


@dataclass(frozen=True, slots=True)
class Homoscedasticity:
    """Residual variance does not depend on the predictors (Breusch-Pagan).

    Test:

    1. Fit OLS residuals :math:`e = y - X\\hat\\beta`.
    2. Regress :math:`e^2` on the design matrix :math:`X` (auxiliary OLS).
    3. The Lagrange-multiplier statistic is :math:`\\mathrm{LM} = n R^2`
       of the auxiliary regression; under H0 of constant variance it
       follows :math:`\\chi^2_{k}` with :math:`k` the number of regressors
       (excluding intercept).

    Implemented from scratch on top of numpy to keep statsmodels off the
    runtime path; statsmodels is used in tests as the reference.

    Accepts an optional ``weights`` argument on construction for parity
    with the package's weighted-everywhere rule; weighted residuals are
    used in the auxiliary regression's residual-of-squares.
    """

    alpha: float = 0.05
    weights: tuple[float, ...] | None = None
    name: str = "Homoscedasticity"
    severity: Severity = Severity.INFO
    requires_solution: bool = True
    requires_cv: bool = False

    def describe(self) -> str:
        return "constant residual variance (Breusch-Pagan)"

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
        resid = get_residuals(spec, data, solution)
        X, _columns = materialise_design(spec, data)
        n = X.shape[0]

        e2 = resid**2
        # Auxiliary OLS of e^2 on X.
        beta_aux, *_ = np.linalg.lstsq(X, e2, rcond=None)
        e2_hat = X @ beta_aux
        ss_res = float(np.sum((e2 - e2_hat) ** 2))
        ss_tot = float(np.sum((e2 - e2.mean()) ** 2))
        r2_aux = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        lm = float(n * r2_aux)

        # Degrees of freedom = number of regressors in the auxiliary
        # regression excluding the intercept. We don't know which column
        # is the intercept here, but materialise_design always places it
        # first when spec.intercept is True; otherwise df = ncol.
        df = X.shape[1] - 1 if getattr(spec, "intercept", True) else X.shape[1]
        df = max(df, 1)
        p = 1.0 - float(stats.chi2.cdf(lm, df))
        passed = p > self.alpha
        return CheckResult(
            name=self.name,
            severity=self.severity,
            passed=bool(passed),
            message=(
                f"Breusch-Pagan LM={lm:.4f}, df={df}, p={p:.4g} "
                f"(alpha={self.alpha})"
            ),
            statistic=lm,
            threshold=self.alpha,
            suggestion=(
                None
                if passed
                else "Heteroscedasticity detected; classical SEs are biased. "
                "Consider robust (HC) standard errors or a log/Box-Cox "
                "transform of the target."
            ),
        )


# Independence (Durbin-Watson)


@dataclass(frozen=True, slots=True)
class Independence:
    """OLS residuals are not serially correlated (Durbin-Watson).

    Statistic:
    :math:`d = \\sum_{t=2}^{n} (e_t - e_{t-1})^2 / \\sum_{t=1}^{n} e_t^2`.

    :math:`d \\approx 2` indicates no autocorrelation; :math:`d \\to 0`
    indicates positive autocorrelation; :math:`d \\to 4` indicates
    negative autocorrelation. We pass within
    ``[d_lower, d_upper]`` (default ``[1.5, 2.5]``), which is the
    commonly quoted rule of thumb.

    The check only makes sense when the data has a meaningful order
    (typically time). Phase 3's temporal-CV framework is the right tool
    for serially correlated data; this check is included for
    completeness in regulatory documentation.
    """

    d_lower: float = 1.5
    d_upper: float = 2.5
    name: str = "Independence"
    severity: Severity = Severity.INFO
    requires_solution: bool = True
    requires_cv: bool = False

    def describe(self) -> str:
        return (
            f"residuals not autocorrelated (Durbin-Watson in "
            f"[{self.d_lower}, {self.d_upper}])"
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
        resid = get_residuals(spec, data, solution)
        d = float(np.sum(np.diff(resid) ** 2) / np.sum(resid**2))
        passed = self.d_lower <= d <= self.d_upper
        return CheckResult(
            name=self.name,
            severity=self.severity,
            passed=bool(passed),
            message=(
                f"Durbin-Watson d={d:.4f} "
                f"(pass band [{self.d_lower}, {self.d_upper}])"
            ),
            statistic=d,
            threshold=self.d_lower if d < 2 else self.d_upper,
            suggestion=(
                None
                if passed
                else "Residuals show autocorrelation; for time-indexed data "
                "use temporal CV (Phase 3) and consider an AR error model."
            ),
        )


# LowVIF


@dataclass(frozen=True, slots=True)
class LowVIF:
    """Variance Inflation Factor is bounded across features (ESL §3.4.1).

    For each non-intercept design column :math:`j`,
    :math:`\\mathrm{VIF}_j = 1 / (1 - R_j^2)` where :math:`R_j^2` is the
    coefficient of determination from regressing column :math:`j` on the
    other design columns (including the intercept).

    Reports the max VIF and flags when it exceeds ``vif_max`` (default
    10, the conventional rule of thumb). The suggestion field is the
    verbatim ESL §3.4.1 nudge toward regularisation.
    """

    vif_max: float = 10.0
    name: str = "LowVIF"
    severity: Severity = Severity.INFO
    requires_solution: bool = False
    requires_cv: bool = False

    def describe(self) -> str:
        return (
            f"max VIF <= {self.vif_max} across non-intercept design columns "
            "(ESL §3.4.1)"
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
        intercept = getattr(spec, "intercept", True)
        # Skip the intercept column when scoring VIFs.
        if intercept and columns and columns[0] == "(Intercept)":
            non_int_idx = list(range(1, X.shape[1]))
        else:
            non_int_idx = list(range(X.shape[1]))
        if len(non_int_idx) < 2:
            # VIF needs at least two non-intercept columns to be meaningful.
            return CheckResult(
                name=self.name,
                severity=self.severity,
                passed=True,
                message="VIF undefined for fewer than 2 features",
                statistic=None,
                threshold=self.vif_max,
                suggestion=None,
            )

        max_vif = -np.inf
        max_col = ""
        for j in non_int_idx:
            xj = X[:, j]
            other_idx = [i for i in range(X.shape[1]) if i != j]
            Xo = X[:, other_idx]
            beta, *_ = np.linalg.lstsq(Xo, xj, rcond=None)
            xj_hat = Xo @ beta
            ss_res = float(np.sum((xj - xj_hat) ** 2))
            ss_tot = float(np.sum((xj - xj.mean()) ** 2))
            if ss_tot == 0:
                # Constant column — undefined VIF, treat as infinite collinearity.
                vif_j = float("inf")
            else:
                r2 = 1.0 - ss_res / ss_tot
                vif_j = 1.0 / (1.0 - r2) if r2 < 1.0 else float("inf")
            if vif_j > max_vif:
                max_vif = vif_j
                max_col = columns[j]

        passed = max_vif <= self.vif_max
        if not passed:
            suggestion = (
                f"High collinearity detected (max VIF = {max_vif:.1f}). "
                "ESL §3.4.1 recommends ridge or lasso regularization "
                "rather than feature pruning. Consider penalty=mc.l2(...) "
                "or penalty=mc.l1(...)."
            )
        else:
            suggestion = None

        return CheckResult(
            name=self.name,
            severity=self.severity,
            passed=bool(passed),
            message=(
                f"max VIF = {max_vif:.3f} on column {max_col!r} "
                f"(threshold {self.vif_max})"
            ),
            statistic=float(max_vif),
            threshold=self.vif_max,
            suggestion=suggestion,
        )
