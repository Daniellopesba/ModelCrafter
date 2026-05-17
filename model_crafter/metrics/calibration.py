"""Calibration metrics: Brier, ECE, calibration curve, log-loss, slope/intercept.

All primitives accept ``(sol, data, weights=...)`` and return rich-`__repr__`
result objects (DESIGN.md §3.3). Probabilities are clipped to
``[eps, 1 - eps]`` (default ``eps = 1e-12``) before entering any log so
log-loss and ECE remain finite when a fitted spec assigns a probability of
exactly 0 or 1.

References
----------
* Brier, G. W. (1950). *Verification of forecasts expressed in terms of
  probability.* Monthly Weather Review 78(1): 1-3.
* Naeini, M. P., Cooper, G. F. and Hauskrecht, M. (2015). *Obtaining
  well calibrated probabilities using Bayesian binning.* AAAI 2015.
  (Expected Calibration Error, equal-frequency binning.)
* Cox, D. R. (1958). *Two further applications of a model for binary
  regression.* Biometrika 45(3-4): 562-565. (Calibration slope &
  intercept via logistic regression of y on the linear predictor.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize

from model_crafter.metrics._common import (
    DEFAULT_EPS,
    check_binary_target,
    coerce_weights,
    resolve_scores_and_target,
)

__all__ = [
    "BrierResult",
    "CalibrationCurve",
    "CalibrationFit",
    "ECEResult",
    "LogLossResult",
    "brier_score",
    "calibration_curve",
    "calibration_slope_intercept",
    "ece",
    "log_loss",
]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BrierResult:
    """Brier score (Brier 1950): ``mean((y - p)^2)``.

    Lower is better; ``0`` is perfect probabilities, ``0.25`` is the score
    of predicting ``0.5`` everywhere.
    """

    value: float
    n_obs: float

    def __float__(self) -> float:
        return float(self.value)

    def __repr__(self) -> str:
        return f"Brier = {self.value:.4f}  (n={self.n_obs:g})"


@dataclass(frozen=True, slots=True)
class ECEResult:
    """Expected Calibration Error (Naeini et al. 2015).

    Equal-frequency binning:
    :math:`\\sum_b (n_b / n) \\cdot | \\bar y_b - \\bar p_b |`.
    """

    value: float
    n_bins: int
    n_obs: float

    def __float__(self) -> float:
        return float(self.value)

    def __repr__(self) -> str:
        return (
            f"ECE = {self.value:.4f}  "
            f"(n_bins={self.n_bins}, n={self.n_obs:g})"
        )


@dataclass(frozen=True, slots=True)
class LogLossResult:
    """Negative log-likelihood per observation (binary cross-entropy).

    :math:`- \\frac{1}{n} \\sum_i [y_i \\log p_i + (1 - y_i) \\log (1 - p_i)]`
    with ``p`` clipped to ``[eps, 1 - eps]``.
    """

    value: float
    eps: float
    n_obs: float

    def __float__(self) -> float:
        return float(self.value)

    def __repr__(self) -> str:
        return f"Log-loss = {self.value:.4f}  (eps={self.eps:g}, n={self.n_obs:g})"


@dataclass(frozen=True, slots=True)
class CalibrationCurve:
    """A binned reliability curve (DESIGN.md §3.3).

    ``predicted`` is the mean predicted probability per bin; ``observed``
    is the (weighted) event rate per bin; ``count`` is the (weighted)
    count per bin.

    A perfectly calibrated model has ``observed ≈ predicted`` for every bin.
    """

    predicted: np.ndarray = field(repr=False)
    observed: np.ndarray = field(repr=False)
    count: np.ndarray = field(repr=False)
    n_bins: int

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "predicted": self.predicted,
                "observed": self.observed,
                "count": self.count,
            }
        )

    def __repr__(self) -> str:
        lines = [f"CalibrationCurve ({self.n_bins} bins)"]
        lines.append("  bin  predicted  observed     count")
        for i, (p, o, c) in enumerate(
            zip(self.predicted, self.observed, self.count, strict=True),
            start=1,
        ):
            lines.append(f"  {i:>3d}  {p:>9.4f}  {o:>8.4f}  {c:>8.1f}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class CalibrationFit:
    """Calibration slope and intercept (Cox 1958).

    Fitted by logistic regression of ``y`` on the linear predictor
    :math:`\\eta = \\log(p / (1 - p))`. A perfectly calibrated model has
    ``slope = 1``, ``intercept = 0``.
    """

    slope: float
    intercept: float
    n_obs: float

    def __float__(self) -> float:
        return float(self.slope)

    def __repr__(self) -> str:
        return (
            f"Calibration fit: slope = {self.slope:.4f}, "
            f"intercept = {self.intercept:+.4f}  (n={self.n_obs:g})"
        )


# ---------------------------------------------------------------------------
# Brier
# ---------------------------------------------------------------------------


def _brier_from_arrays(
    y: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float, float]:
    """Return ``(brier_score, n)``."""
    check_binary_target(y)
    # No clipping for Brier — squared error is well-defined for any score.
    err2 = (y - scores) ** 2
    if weights is None:
        return float(np.mean(err2)), float(y.size)
    n_eff = float(np.sum(weights))
    return float(np.sum(weights * err2) / n_eff), n_eff


def brier_score(
    sol: Any,
    data: pd.DataFrame,
    *,
    weights: str | np.ndarray | pd.Series | None = None,
) -> BrierResult:
    """Brier score: mean squared error between probabilities and outcomes."""
    y, scores = resolve_scores_and_target(sol, data)
    w = coerce_weights(weights, data)
    value, n_obs = _brier_from_arrays(y, scores, w)
    return BrierResult(value=value, n_obs=n_obs)


# ---------------------------------------------------------------------------
# Log loss
# ---------------------------------------------------------------------------


def _log_loss_from_arrays(
    y: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    eps: float = DEFAULT_EPS,
) -> tuple[float, float]:
    """Return ``(log_loss, n)``.

    Clips ``scores`` to ``[eps, 1 - eps]`` to avoid ``-inf`` when a model
    predicts an exact 0 or 1.
    """
    check_binary_target(y)
    if eps <= 0.0 or eps >= 0.5:
        raise ValueError(f"eps must be in (0, 0.5); got {eps}")
    p = np.clip(scores, eps, 1.0 - eps)
    ll = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    if weights is None:
        return float(np.mean(ll)), float(y.size)
    n_eff = float(np.sum(weights))
    return float(np.sum(weights * ll) / n_eff), n_eff


def log_loss(
    sol: Any,
    data: pd.DataFrame,
    *,
    weights: str | np.ndarray | pd.Series | None = None,
    eps: float = DEFAULT_EPS,
) -> LogLossResult:
    """Negative log-likelihood per observation (binary cross-entropy)."""
    y, scores = resolve_scores_and_target(sol, data)
    w = coerce_weights(weights, data)
    value, n_obs = _log_loss_from_arrays(y, scores, w, eps=eps)
    return LogLossResult(value=value, eps=eps, n_obs=n_obs)


# ---------------------------------------------------------------------------
# Calibration curve
# ---------------------------------------------------------------------------


def _bin_by_quantile(
    scores: np.ndarray,
    weights: np.ndarray | None,
    n_bins: int,
) -> np.ndarray:
    """Return bin assignment (0..n_bins-1) by equal-frequency binning.

    For weighted data the cumulative weight is used to determine the
    breakpoints. Ties may cause some bins to contain more mass than the
    nominal ``1/n_bins``; the implementation prefers fewer effective
    bins over splitting ties (matches statsmodels' calibration plot).
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1; got {n_bins}")
    n = scores.size
    if weights is None:
        weights = np.ones(n, dtype=float)
    total = float(np.sum(weights))
    # Sort by score, accumulate weight, assign bin where cum weight
    # falls into [b/n_bins, (b+1)/n_bins].
    order = np.argsort(scores, kind="mergesort")
    cum = np.cumsum(weights[order]) / total
    # Find bin index = floor(cum * n_bins), clamped to [0, n_bins-1].
    # Using "previous mass" so a point at the boundary falls into the
    # lower bin (deterministic, weight-fraction-stable).
    cum_prev = cum - weights[order] / total
    # ``+ 1e-12`` absorbs the floating-point error that can turn 2.0 into
    # 1.9999...998 when ``cum_prev * n_bins`` should land on an integer
    # boundary (e.g. uniform weights with n divisible by n_bins).
    bin_sorted = np.floor(cum_prev * n_bins + 1e-12).astype(int)
    bin_sorted = np.clip(bin_sorted, 0, n_bins - 1)
    out = np.empty(n, dtype=int)
    out[order] = bin_sorted
    return out


def _calibration_curve_from_arrays(
    y: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray | None,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(predicted, observed, count)`` arrays for a reliability curve."""
    check_binary_target(y)
    bins = _bin_by_quantile(scores, weights, n_bins)
    w = np.ones_like(scores, dtype=float) if weights is None else weights
    predicted = np.zeros(n_bins, dtype=float)
    observed = np.zeros(n_bins, dtype=float)
    count = np.zeros(n_bins, dtype=float)
    # Vectorise the per-bin reductions.
    np.add.at(count, bins, w)
    np.add.at(predicted, bins, w * scores)
    np.add.at(observed, bins, w * y)
    # Avoid division by zero: empty bins emit NaN.
    nonempty = count > 0
    predicted[nonempty] /= count[nonempty]
    observed[nonempty] /= count[nonempty]
    predicted[~nonempty] = np.nan
    observed[~nonempty] = np.nan
    return predicted, observed, count


def calibration_curve(
    sol: Any,
    data: pd.DataFrame,
    *,
    n_bins: int = 10,
    weights: str | np.ndarray | pd.Series | None = None,
) -> CalibrationCurve:
    """Equal-frequency reliability curve (DESIGN.md §3.3)."""
    y, scores = resolve_scores_and_target(sol, data)
    w = coerce_weights(weights, data)
    predicted, observed, count = _calibration_curve_from_arrays(
        y, scores, w, n_bins
    )
    return CalibrationCurve(
        predicted=predicted, observed=observed, count=count, n_bins=n_bins
    )


# ---------------------------------------------------------------------------
# Expected Calibration Error
# ---------------------------------------------------------------------------


def _ece_from_arrays(
    y: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray | None,
    n_bins: int,
) -> tuple[float, float]:
    """Return ``(ece, n_eff)``."""
    predicted, observed, count = _calibration_curve_from_arrays(
        y, scores, weights, n_bins
    )
    total = float(np.sum(count))
    nonempty = count > 0
    gaps = np.abs(observed[nonempty] - predicted[nonempty])
    weights_b = count[nonempty] / total
    return float(np.sum(weights_b * gaps)), total


def ece(
    sol: Any,
    data: pd.DataFrame,
    *,
    n_bins: int = 10,
    weights: str | np.ndarray | pd.Series | None = None,
) -> ECEResult:
    """Expected Calibration Error (equal-frequency binning)."""
    y, scores = resolve_scores_and_target(sol, data)
    w = coerce_weights(weights, data)
    value, n_eff = _ece_from_arrays(y, scores, w, n_bins)
    return ECEResult(value=value, n_bins=n_bins, n_obs=n_eff)


# ---------------------------------------------------------------------------
# Calibration slope and intercept
# ---------------------------------------------------------------------------


def _calibration_fit_from_arrays(
    y: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray | None,
    *,
    eps: float = DEFAULT_EPS,
) -> tuple[float, float]:
    """Return ``(slope, intercept)`` from a logistic regression of ``y`` on
    the linear predictor :math:`\\eta = \\log(p / (1 - p))`.

    Implemented via :func:`scipy.optimize.minimize` (Newton-CG) on the
    binary cross-entropy. Statsmodels is the test-only cross-check;
    deliberately not a runtime import (DESIGN.md §9.10).
    """
    check_binary_target(y)
    p = np.clip(scores, eps, 1.0 - eps)
    eta = np.log(p / (1.0 - p))
    w = (
        np.ones_like(y, dtype=float)
        if weights is None
        else np.asarray(weights, dtype=float)
    )

    # Design matrix [1, eta]; coefficients beta = (intercept, slope).
    X = np.column_stack([np.ones_like(eta), eta])

    def neg_log_lik(beta: np.ndarray) -> float:
        z = np.asarray(X @ beta, dtype=float)
        # Numerically stable log(1 + e^z) via logaddexp.
        log1p_e = np.logaddexp(0.0, z)
        # nll = sum w * (log1p(e^z) - y * z)
        return float(np.sum(w * (log1p_e - y * z)))

    def grad(beta: np.ndarray) -> np.ndarray:
        z = np.asarray(X @ beta, dtype=float)
        # Stable sigmoid: sigma(z) = 1 / (1 + exp(-|z|)) with sign tracking,
        # equivalent to expit(z).
        from scipy.special import expit

        sig = expit(z)
        resid = sig - y
        return (X.T * w) @ resid

    res = optimize.minimize(
        neg_log_lik,
        x0=np.array([0.0, 1.0]),
        jac=grad,
        method="L-BFGS-B",
        options={"ftol": 1e-12, "gtol": 1e-10},
    )
    intercept, slope = float(res.x[0]), float(res.x[1])
    return slope, intercept


def calibration_slope_intercept(
    sol: Any,
    data: pd.DataFrame,
    *,
    weights: str | np.ndarray | pd.Series | None = None,
    eps: float = DEFAULT_EPS,
) -> CalibrationFit:
    """Calibration slope and intercept (Cox 1958).

    Fits the logistic regression :math:`P(y=1) = \\sigma(\\alpha + \\beta \\eta)`
    where :math:`\\eta = \\log(p / (1 - p))` is the linear predictor implied
    by ``sol``'s probabilities on ``data``. Perfect calibration recovers
    ``slope = 1, intercept = 0``.
    """
    y, scores = resolve_scores_and_target(sol, data)
    w = coerce_weights(weights, data)
    slope, intercept = _calibration_fit_from_arrays(y, scores, w, eps=eps)
    n_obs = float(y.size if w is None else np.sum(w))
    return CalibrationFit(slope=slope, intercept=intercept, n_obs=n_obs)
