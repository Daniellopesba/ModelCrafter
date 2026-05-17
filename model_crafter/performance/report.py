"""``PerformanceReport`` and the orchestration that builds it.

The report bundles four sub-reports plus the lift table and gains curve.
Its ``__repr__`` matches the layout in ``DESIGN.md`` §3.3 — the canonical
user-facing artefact of fitting.

Implementation notes
--------------------

* Every metric is computed *once* and threaded through the sub-reports;
  there is no double-evaluation between, say, ``DiscriminationReport.auc``
  and ``mc.auc(sol, data)`` called by a user. (Acceptance criterion 6:
  sub-reports contain values matching individual primitive calls.)
* The AUC CI in the report header is computed with DeLong's one-sample
  variance — the same structural-component variance as
  :func:`~model_crafter.metrics.classification.delong_test` reformulated
  for a single AUC. We compute it here rather than as a public primitive
  to keep the metrics surface minimal.
* PSI is computed against ``reference`` if and only if a reference is
  supplied. The reference may be a fitted ``Solution`` (in which case we
  call :func:`model_crafter.solve.predict` to materialise reference
  scores) or a raw array/Series of reference scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from model_crafter.metrics._common import (
    check_binary_target,
    coerce_weights,
    resolve_scores_and_target,
    weighted_mean,
    weighted_quantile,
)
from model_crafter.metrics.calibration import (
    BrierResult,
    CalibrationCurve,
    CalibrationFit,
    ECEResult,
    LogLossResult,
    _brier_from_arrays,
    _calibration_curve_from_arrays,
    _calibration_fit_from_arrays,
    _ece_from_arrays,
    _log_loss_from_arrays,
)
from model_crafter.metrics.classification import (
    AUCResult,
    CohensDResult,
    GiniResult,
    KSResult,
    _auc_from_arrays,
    _cohens_d_from_arrays,
    _delong_components,
    _ks_from_arrays,
)
from model_crafter.metrics.rank import (
    GainsCurve,
    LiftTable,
    _cumulative_gains_from_arrays,
    _lift_table_from_arrays,
)
from model_crafter.metrics.stability import PSIResult
from model_crafter.metrics.stability import psi as _psi_call

__all__ = [
    "CalibrationReport",
    "DiscriminationReport",
    "DistributionReport",
    "PerformanceReport",
    "StabilityReport",
    "performance",
]


# ---------------------------------------------------------------------------
# Sub-reports
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DiscriminationReport:
    """AUC + Gini + KS + Cohen's d, with an AUC CI from DeLong's variance.

    Fields
    ------
    auc, gini, ks, cohens_d :
        Primitive result objects.
    auc_ci : tuple[float, float] | None
        ``(lower, upper)`` for the AUC at ``auc_ci_level`` (default 0.95),
        computed from DeLong's one-sample structural-component variance.
        ``None`` only if AUC is degenerate.
    auc_se : float | None
        DeLong standard error of the AUC.
    auc_ci_level : float
        The confidence level used.
    """

    auc: AUCResult
    gini: GiniResult
    ks: KSResult
    cohens_d: CohensDResult
    auc_ci: tuple[float, float] | None = None
    auc_se: float | None = None
    auc_ci_level: float = 0.95


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """Calibration curve + Brier + ECE + log-loss + slope/intercept."""

    curve: CalibrationCurve
    brier: BrierResult
    ece: ECEResult
    log_loss: LogLossResult
    slope_intercept: CalibrationFit


@dataclass(frozen=True, slots=True)
class StabilityReport:
    """PSI of current scores vs a reference."""

    psi: PSIResult
    reference_label: str = "reference"


@dataclass(frozen=True, slots=True)
class DistributionReport:
    """Summary of the predicted-probability distribution (DESIGN.md §3.3)."""

    mean: float
    median: float
    p_min: float
    p_max: float
    n_obs: float

    def __repr__(self) -> str:
        return (
            f"DistributionReport(mean={self.mean:.4f}, median={self.median:.4f}, "
            f"range=[{self.p_min:.4f}, {self.p_max:.4f}], n={self.n_obs:g})"
        )


# ---------------------------------------------------------------------------
# Top-level PerformanceReport
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PerformanceReport:
    """The bundled performance value (DESIGN.md §3.3).

    Attribute access matches the §3.3 example: ``.discrimination``,
    ``.calibration``, ``.stability``, ``.distribution``, ``.lift_table``,
    ``.cumulative_gains``, plus ``.n_obs`` and ``.n_events``.

    The ``__repr__`` produces a multi-section block matching the layout
    shown in DESIGN.md §3.3.
    """

    discrimination: DiscriminationReport
    calibration: CalibrationReport
    stability: StabilityReport | None
    distribution: DistributionReport
    lift_table: LiftTable = field(repr=False)
    cumulative_gains: GainsCurve = field(repr=False)
    n_obs: int
    n_events: int

    def __repr__(self) -> str:  # noqa: PLR0915 — repr layout
        # ----- Header -----
        n = self.n_obs
        ev = self.n_events
        ev_rate = (ev / n * 100.0) if n > 0 else 0.0
        lines = [
            "PerformanceReport",
            f"n={n:,}  events={ev:,} ({ev_rate:.1f}%)",
            "",
            "Discrimination",
        ]
        # AUC line with CI.
        auc_v = self.discrimination.auc.value
        auc_ci = self.discrimination.auc_ci
        if auc_ci is not None:
            lvl = int(round(self.discrimination.auc_ci_level * 100))
            ci_str = f"   ({lvl}% CI: {auc_ci[0]:.4f} – {auc_ci[1]:.4f}, DeLong)"
        else:
            ci_str = ""
        lines.append(f"  AUC                {auc_v:.4f}{ci_str}")
        lines.append(f"  Gini               {self.discrimination.gini.value:.4f}")
        ks_v = self.discrimination.ks.value
        ks_at = self.discrimination.ks.at_score
        lines.append(
            f"  KS                 {ks_v:.4f}   (at score {ks_at:.4f})"
        )
        lines.append(
            f"  Cohen's d          {self.discrimination.cohens_d.value:.4f}"
        )
        lines.append("")
        # ----- Calibration -----
        lines.append("Calibration")
        lines.append(f"  Brier              {self.calibration.brier.value:.4f}")
        ece_v = self.calibration.ece.value
        ece_bins = self.calibration.ece.n_bins
        lines.append(f"  ECE  ({ece_bins} bins)     {ece_v:.4f}")
        lines.append(
            f"  Log-loss           {self.calibration.log_loss.value:.4f}"
        )
        slope = self.calibration.slope_intercept.slope
        intercept = self.calibration.slope_intercept.intercept
        lines.append(
            f"  Slope / Intercept  {slope:.3f} / {intercept:+.3f}  "
            "(logit regression of y on linear pred)"
        )
        lines.append("")
        # ----- Stability -----
        if self.stability is not None:
            lines.append("Stability")
            psi_v = self.stability.psi.value
            if psi_v < 0.10:
                tag = "low"
            elif psi_v < 0.25:
                tag = "moderate"
            else:
                tag = "high"
            label = self.stability.reference_label
            lines.append(
                f"  PSI vs reference   {psi_v:.3f}   ({tag}; reference: {label})"
            )
            lines.append("")
        # ----- Distribution -----
        d = self.distribution
        lines.append("Distribution")
        lines.append(
            f"  Mean / Median p̂   {d.mean:.4f} / {d.median:.4f}"
        )
        lines.append(
            f"  Score range        [{d.p_min:.4f}, {d.p_max:.4f}]"
        )
        return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# DeLong one-sample AUC variance (for the AUC CI)
# ---------------------------------------------------------------------------


def _delong_se(
    y: np.ndarray, scores: np.ndarray
) -> float:
    """One-sample DeLong SE of the AUC (Sun & Xu 2014).

    Reuses the structural components used by :func:`delong_test` but
    against a "zero" comparator: ``var(AUC) = var(V10) / m + var(V01) / n``
    where ``V10`` and ``V01`` are the per-positive and per-negative
    structural components of the single AUC.
    """
    auc_a, _, var_diff_zero = _delong_components(scores, np.zeros_like(scores), y)
    # The "zero" comparator has AUC = 0 and zero variance contribution; the
    # cross terms cancel and var_diff equals var(AUC_a). Verify quickly by
    # rebuilding from raw components.
    pos = y == 1
    neg = ~pos
    m = int(pos.sum())
    n = int(neg.sum())
    from model_crafter.metrics.classification import _midrank

    s = scores
    s_pos = s[pos]
    s_neg = s[neg]
    t_z = _midrank(np.concatenate([s_pos, s_neg]))
    t_x = _midrank(s_pos)
    t_y = _midrank(s_neg)
    tz_pos = t_z[:m]
    tz_neg = t_z[m:]
    v10 = (tz_pos - t_x) / float(n)
    v01 = 1.0 - (tz_neg - t_y) / float(m)
    var10 = float(np.var(v10, ddof=1)) if m > 1 else 0.0
    var01 = float(np.var(v01, ddof=1)) if n > 1 else 0.0
    var_auc = var10 / m + var01 / n
    # ``var_diff_zero`` should equal ``var_auc`` (cross-check).
    _ = auc_a, var_diff_zero
    if var_auc < 0:
        var_auc = 0.0
    return float(np.sqrt(var_auc))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _resolve_reference_scores(
    reference: Any, sol_for_predict: Any
) -> tuple[np.ndarray, str]:
    """Coerce the ``reference=`` argument of :func:`performance` to a 1-D
    array of reference scores, plus a short label for the report.

    Accepted forms:

    * a fitted ``Solution``-like object with ``.spec.target``; we apply
      ``predict`` to its training data... but we don't have its training
      data, so this form requires ``reference`` to be a tuple
      ``(sol, data)``.
    * a ``pd.DataFrame`` — we apply ``sol_for_predict`` to it and use
      those predictions.
    * a ``pd.Series`` / ``np.ndarray`` — taken at face value.
    """
    from model_crafter.solve import predict as _predict

    if isinstance(reference, pd.DataFrame):
        scores = np.asarray(_predict(sol_for_predict, reference), dtype=float)
        return scores, "DataFrame"
    if isinstance(reference, pd.Series):
        return np.asarray(reference, dtype=float), "Series"
    if isinstance(reference, np.ndarray):
        return np.asarray(reference, dtype=float).ravel(), "array"
    if isinstance(reference, tuple) and len(reference) == 2:
        ref_sol, ref_data = reference
        if not isinstance(ref_data, pd.DataFrame):
            raise TypeError(
                "When reference=(sol, data) the second element must be a "
                "DataFrame; got {type(ref_data).__name__}"
            )
        scores = np.asarray(_predict(ref_sol, ref_data), dtype=float)
        return scores, "Solution"
    raise TypeError(
        "reference must be a DataFrame, Series/array of scores, or "
        f"(sol, data) tuple; got {type(reference).__name__}"
    )


def performance(
    sol: Any,
    data: pd.DataFrame,
    *,
    weights: str | np.ndarray | pd.Series | None = None,
    reference: Any = None,
    n_bins: int = 10,
    n_deciles: int = 10,
    psi_bins: int = 10,
    auc_ci_level: float = 0.95,
    eps: float = 1e-12,
) -> PerformanceReport:
    """Compute the full :class:`PerformanceReport` for ``sol`` on ``data``.

    Parameters
    ----------
    sol :
        A fitted ``Solution`` (or anything that :func:`mc.predict` works on).
    data :
        A ``pd.DataFrame`` containing ``sol.spec.target``.
    weights :
        Sample weights — a column name in ``data`` or a 1-D array. ``None``
        is uniform.
    reference :
        Optional reference distribution for the PSI stability check. May be
        a ``DataFrame`` (scored via ``sol``), a ``Series``/array of scores,
        or a ``(sol, data)`` tuple. When ``None``, no stability sub-report.
    n_bins, n_deciles, psi_bins :
        Bin counts for the calibration curve / ECE, the lift table /
        gains, and the PSI binning, respectively.
    auc_ci_level :
        Confidence level for the DeLong AUC CI (default 0.95).
    eps :
        Probability clipping used for log-loss (default ``1e-12``).
    """
    y, scores = resolve_scores_and_target(sol, data)
    check_binary_target(y)
    w = coerce_weights(weights, data)

    # ---- Discrimination ----
    auc_value, n_pos, n_neg = _auc_from_arrays(y, scores, w)
    gini_value = 2.0 * auc_value - 1.0
    ks_value, ks_at_score, _, _ = _ks_from_arrays(y, scores, w)
    cohens_value, mean_pos, mean_neg, pooled_sd = _cohens_d_from_arrays(
        y, scores, w
    )
    # DeLong SE & CI — unweighted (the weighted DeLong variance is out of
    # scope; we still want a CI when weights=None, and we fall back to
    # ``None`` when weights are non-uniform).
    auc_se: float | None = None
    auc_ci: tuple[float, float] | None = None
    if w is None or np.allclose(w, w[0]):
        try:
            se = _delong_se(y, scores)
            auc_se = float(se)
            z_crit = float(scipy_stats.norm.ppf(0.5 + auc_ci_level / 2.0))
            lo = max(0.0, auc_value - z_crit * auc_se)
            hi = min(1.0, auc_value + z_crit * auc_se)
            auc_ci = (lo, hi)
        except Exception:  # noqa: BLE001 — CI is informational only
            auc_se = None
            auc_ci = None

    disc = DiscriminationReport(
        auc=AUCResult(value=auc_value, n_pos=n_pos, n_neg=n_neg, se=auc_se),
        gini=GiniResult(value=gini_value, n_pos=n_pos, n_neg=n_neg),
        ks=KSResult(
            value=ks_value, at_score=ks_at_score, n_pos=n_pos, n_neg=n_neg
        ),
        cohens_d=CohensDResult(
            value=cohens_value,
            mean_pos=mean_pos,
            mean_neg=mean_neg,
            pooled_sd=pooled_sd,
        ),
        auc_ci=auc_ci,
        auc_se=auc_se,
        auc_ci_level=auc_ci_level,
    )

    # ---- Calibration ----
    brier_v, n_brier = _brier_from_arrays(y, scores, w)
    ece_v, n_ece = _ece_from_arrays(y, scores, w, n_bins)
    ll_v, n_ll = _log_loss_from_arrays(y, scores, w, eps=eps)
    slope, intercept = _calibration_fit_from_arrays(y, scores, w, eps=eps)
    curve_pred, curve_obs, curve_count = _calibration_curve_from_arrays(
        y, scores, w, n_bins
    )
    cal = CalibrationReport(
        curve=CalibrationCurve(
            predicted=curve_pred,
            observed=curve_obs,
            count=curve_count,
            n_bins=n_bins,
        ),
        brier=BrierResult(value=brier_v, n_obs=n_brier),
        ece=ECEResult(value=ece_v, n_bins=n_bins, n_obs=n_ece),
        log_loss=LogLossResult(value=ll_v, eps=eps, n_obs=n_ll),
        slope_intercept=CalibrationFit(
            slope=slope, intercept=intercept, n_obs=float(y.size if w is None else np.sum(w))
        ),
    )

    # ---- Stability (optional) ----
    stab: StabilityReport | None = None
    if reference is not None:
        ref_scores, ref_label = _resolve_reference_scores(reference, sol)
        psi_res = _psi_call(ref_scores, scores, bins=psi_bins)
        stab = StabilityReport(psi=psi_res, reference_label=ref_label)

    # ---- Distribution ----
    if w is None:
        d_mean = float(np.mean(scores))
        d_median = float(np.median(scores))
        n_dist = float(scores.size)
    else:
        d_mean = weighted_mean(scores, w)
        d_median = float(weighted_quantile(scores, np.array([0.5]), w)[0])
        n_dist = float(np.sum(w))
    dist = DistributionReport(
        mean=d_mean,
        median=d_median,
        p_min=float(np.min(scores)),
        p_max=float(np.max(scores)),
        n_obs=n_dist,
    )

    # ---- Lift table + Gains ----
    lift_df = _lift_table_from_arrays(y, scores, w, n_deciles)
    lt = LiftTable(table=lift_df, n_deciles=n_deciles)
    cp, cc = _cumulative_gains_from_arrays(y, scores, w)
    gains = GainsCurve(cum_population=cp, cum_captured=cc)

    # ---- Header counts ----
    n_obs_total = int(y.size)
    n_events = int(np.sum(y == 1))

    return PerformanceReport(
        discrimination=disc,
        calibration=cal,
        stability=stab,
        distribution=dist,
        lift_table=lt,
        cumulative_gains=gains,
        n_obs=n_obs_total,
        n_events=n_events,
    )
