"""Discrimination metrics: AUC, Gini, KS, Cohen's d, DeLong test.

All primitives accept ``(sol, data, weights=...)`` and return frozen
dataclass result objects with a rich ``__repr__`` and ``__float__`` for
numeric contexts (DESIGN.md §3.3).

References
----------
* Mann, H. B. and Whitney, D. R. (1947). *On a test of whether one of two
  random variables is stochastically larger than the other.*
  AUC equals the (normalised) Mann-Whitney U statistic; this is the
  numerical contract for :func:`auc`.
* DeLong, E. R., DeLong, D. M. and Clarke-Pearson, D. L. (1988).
  *Comparing the areas under two or more correlated receiver operating
  characteristic curves: a nonparametric approach.* Biometrics 44(3):
  837-845. The structural variance formula implemented in
  :func:`delong_test` is from this paper, in the numerically-stable
  reformulation of Sun & Xu (2014), *Fast implementation of DeLong's
  algorithm for comparing the areas under correlated receiver operating
  characteristic curves*, IEEE Signal Processing Letters 21(11):
  1389-1393.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from model_crafter.metrics._common import (
    check_binary_target,
    coerce_weights,
    resolve_scores_and_target,
    split_pos_neg,
    weighted_mean,
)

__all__ = [
    "AUCResult",
    "CohensDResult",
    "DeLongResult",
    "GiniResult",
    "KSResult",
    "auc",
    "cohens_d",
    "delong_test",
    "gini",
    "ks",
]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AUCResult:
    """Area under the ROC curve (DESIGN.md §3.3).

    AUC equals the probability that a randomly chosen positive scores
    higher than a randomly chosen negative — equivalently, the Mann-Whitney
    U statistic divided by ``n_pos * n_neg`` (Mann & Whitney 1947).

    Fields
    ------
    value : float
        The AUC point estimate, in ``[0, 1]``. ``0.5`` is no discrimination.
    n_pos, n_neg : int
        Effective (weighted) counts of positives and negatives.
    se : float | None
        DeLong standard error, when computed (e.g., by the DeLong CI helper).
    """

    value: float
    n_pos: float
    n_neg: float
    se: float | None = None

    def __float__(self) -> float:
        return float(self.value)

    def __repr__(self) -> str:
        se_str = f" (SE={self.se:.4f})" if self.se is not None else ""
        return (
            f"AUC = {self.value:.4f}{se_str}  "
            f"(n_pos={self.n_pos:g}, n_neg={self.n_neg:g})"
        )


@dataclass(frozen=True, slots=True)
class GiniResult:
    """Gini coefficient (DESIGN.md §3.3).

    Defined as ``2 * AUC - 1``; in ``[-1, 1]``.
    """

    value: float
    n_pos: float
    n_neg: float

    def __float__(self) -> float:
        return float(self.value)

    def __repr__(self) -> str:
        return (
            f"Gini = {self.value:.4f}  "
            f"(n_pos={self.n_pos:g}, n_neg={self.n_neg:g})"
        )


@dataclass(frozen=True, slots=True)
class KSResult:
    """Kolmogorov-Smirnov statistic on positive vs negative score CDFs
    (DESIGN.md §3.3).

    Numerically matches :func:`scipy.stats.ks_2samp` for ``weights=None``.

    Fields
    ------
    value : float
        ``max_t |F_neg(t) - F_pos(t)|``, in ``[0, 1]``.
    at_score : float
        The score at which the maximum gap is attained.
    n_pos, n_neg : float
        Effective counts.
    """

    value: float
    at_score: float
    n_pos: float
    n_neg: float

    def __float__(self) -> float:
        return float(self.value)

    def __repr__(self) -> str:
        return (
            f"KS = {self.value:.4f}  at score = {self.at_score:.4g}  "
            f"(n_pos={self.n_pos:g}, n_neg={self.n_neg:g})"
        )


@dataclass(frozen=True, slots=True)
class CohensDResult:
    """Cohen's *d*: standardized mean difference between positives and
    negatives.

    .. math::

        d = \\frac{\\bar s_{+} - \\bar s_{-}}{s_p}

    where :math:`s_p` is the pooled standard deviation (Cohen 1988).
    Weighted variants use the weighted means and weighted pooled SD.
    """

    value: float
    mean_pos: float
    mean_neg: float
    pooled_sd: float

    def __float__(self) -> float:
        return float(self.value)

    def __repr__(self) -> str:
        return (
            f"Cohen's d = {self.value:.4f}  "
            f"(mean_pos={self.mean_pos:.4f}, mean_neg={self.mean_neg:.4f}, "
            f"pooled_sd={self.pooled_sd:.4f})"
        )


@dataclass(frozen=True, slots=True)
class DeLongResult:
    """DeLong (1988) test for paired AUC difference.

    Fields
    ------
    auc_a, auc_b : float
        Point estimates of the two models' AUCs.
    diff : float
        ``auc_a - auc_b``.
    var_diff : float
        Variance of the difference under DeLong's structural-component
        formula (Sun & Xu 2014 reformulation).
    z : float
        ``diff / sqrt(var_diff)``.
    p_value : float
        Two-sided normal-approximation p-value.
    n_pos, n_neg : int
        Counts of positives / negatives.
    """

    auc_a: float
    auc_b: float
    diff: float
    var_diff: float
    z: float
    p_value: float
    n_pos: int
    n_neg: int

    def __float__(self) -> float:
        return float(self.p_value)

    def __repr__(self) -> str:
        return (
            f"DeLong test: AUC_a={self.auc_a:.4f}, AUC_b={self.auc_b:.4f}, "
            f"diff={self.diff:+.4f}, z={self.z:.3f}, p={self.p_value:.4g}  "
            f"(n_pos={self.n_pos}, n_neg={self.n_neg})"
        )


# ---------------------------------------------------------------------------
# AUC
# ---------------------------------------------------------------------------


def _auc_from_arrays(
    y: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Return ``(auc, n_pos, n_neg)`` via the Mann-Whitney U identity.

    AUC = (U + 0.5 * ties) / (n_pos * n_neg) where U counts the number of
    positive-negative pairs where the positive scores strictly higher.

    The weighted variant generalises U to
    :math:`\\sum_{i \\in +} \\sum_{j \\in -} w_i w_j \\cdot \\mathbf{1}\\{s_i > s_j\\}
    + 0.5 \\cdot w_i w_j \\cdot \\mathbf{1}\\{s_i = s_j\\}`,
    normalised by :math:`(\\sum_{i \\in +} w_i)(\\sum_{j \\in -} w_j)`.

    We compute it in O(n log n) by sorting and using midrank weights, which
    matches Sun & Xu (2014)'s tie-handling for DeLong.
    """
    check_binary_target(y)
    s_pos, s_neg, w_pos, w_neg = split_pos_neg(y, scores, weights)
    n_pos = float(s_pos.size if w_pos is None else np.sum(w_pos))
    n_neg = float(s_neg.size if w_neg is None else np.sum(w_neg))
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"AUC undefined: need positives and negatives "
            f"(got n_pos={n_pos}, n_neg={n_neg})"
        )
    # Combined arrays, with class indicator. We use the midrank formulation
    # so AUC = (R_+ - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg) — the
    # standard Mann-Whitney identity. For weighted data we replace ranks
    # with cumulative weight midranks.
    all_scores = np.concatenate([s_pos, s_neg])
    all_weights = (
        np.ones_like(all_scores)
        if weights is None
        else np.concatenate([w_pos, w_neg])  # type: ignore[list-item]
    )
    is_pos = np.concatenate(
        [np.ones_like(s_pos, dtype=bool), np.zeros_like(s_neg, dtype=bool)]
    )

    order = np.argsort(all_scores, kind="mergesort")
    sorted_scores = all_scores[order]
    sorted_weights = all_weights[order]
    sorted_is_pos = is_pos[order]

    # Tied-group midranks: for each group of equal scores, every member
    # receives the same midrank = (cum_start + cum_end + 1) / 2 where the
    # cumulative weight is the running sum of weights. This is equivalent
    # to averaging ranks within ties.
    midranks = np.empty(sorted_scores.size, dtype=float)
    i = 0
    cum_before = 0.0
    while i < sorted_scores.size:
        j = i
        while (
            j + 1 < sorted_scores.size
            and sorted_scores[j + 1] == sorted_scores[i]
        ):
            j += 1
        # Group [i, j], weights sum = sum_w
        sum_w = float(np.sum(sorted_weights[i : j + 1]))
        # Average rank within ties: (cum_before + 0.5) + 0.5 * sum_w
        # — i.e. the centre of the group's cumulative-weight interval,
        # shifted by 0.5 to recover the unweighted rank when w=1.
        avg_rank = cum_before + 0.5 + 0.5 * sum_w
        midranks[i : j + 1] = avg_rank
        cum_before += sum_w
        i = j + 1

    # Mann-Whitney identity (weighted): U = R_+ - n_pos * (n_pos + 1) / 2
    # generalises to R_+ = sum_{i in +} w_i * midrank_i and
    # n_pos * (n_pos + 1) / 2 -> (sum w_pos) * (sum w_pos + 1) / 2.
    sw_pos = sorted_weights[sorted_is_pos]
    rank_pos = midranks[sorted_is_pos]
    r_pos = float(np.sum(sw_pos * rank_pos))
    u = r_pos - n_pos * (n_pos + 1.0) / 2.0
    auc_value = u / (n_pos * n_neg)
    return float(auc_value), n_pos, n_neg


def auc(
    sol: Any,
    data: pd.DataFrame,
    *,
    weights: str | np.ndarray | pd.Series | None = None,
) -> AUCResult:
    """Area under the ROC curve.

    Returns the AUC of ``sol``'s predictions on ``data``. Implemented via
    the Mann-Whitney U identity for exact agreement with
    :func:`scipy.stats.mannwhitneyu`-derived AUC (``U / (n_pos * n_neg)``).
    """
    y, scores = resolve_scores_and_target(sol, data)
    w = coerce_weights(weights, data)
    value, n_pos, n_neg = _auc_from_arrays(y, scores, w)
    return AUCResult(value=value, n_pos=n_pos, n_neg=n_neg, se=None)


# ---------------------------------------------------------------------------
# Gini
# ---------------------------------------------------------------------------


def gini(
    sol: Any,
    data: pd.DataFrame,
    *,
    weights: str | np.ndarray | pd.Series | None = None,
) -> GiniResult:
    """Gini coefficient, defined as ``2 * AUC - 1``.

    Range: ``[-1, 1]``; ``0`` is no discrimination.
    """
    y, scores = resolve_scores_and_target(sol, data)
    w = coerce_weights(weights, data)
    auc_value, n_pos, n_neg = _auc_from_arrays(y, scores, w)
    return GiniResult(value=2.0 * auc_value - 1.0, n_pos=n_pos, n_neg=n_neg)


# ---------------------------------------------------------------------------
# KS
# ---------------------------------------------------------------------------


def _ks_from_arrays(
    y: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float, float, float, float]:
    """Return ``(ks_statistic, at_score, n_pos, n_neg)``.

    Matches :func:`scipy.stats.ks_2samp` for ``weights=None``. The weighted
    variant uses weighted empirical CDFs.
    """
    check_binary_target(y)
    s_pos, s_neg, w_pos, w_neg = split_pos_neg(y, scores, weights)
    n_pos = float(s_pos.size if w_pos is None else np.sum(w_pos))
    n_neg = float(s_neg.size if w_neg is None else np.sum(w_neg))
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"KS undefined: need positives and negatives "
            f"(got n_pos={n_pos}, n_neg={n_neg})"
        )
    # Merge unique score thresholds; compute weighted ECDFs at each.
    if weights is None:
        w_pos = np.ones_like(s_pos)
        w_neg = np.ones_like(s_neg)
    # Sort each class.
    order_pos = np.argsort(s_pos, kind="mergesort")
    order_neg = np.argsort(s_neg, kind="mergesort")
    sp = s_pos[order_pos]
    sn = s_neg[order_neg]
    wp = np.asarray(w_pos, dtype=float)[order_pos]
    wn = np.asarray(w_neg, dtype=float)[order_neg]
    # All thresholds — merging unique scores in ascending order.
    all_scores = np.unique(np.concatenate([sp, sn]))
    # Weighted ECDFs at each threshold (right-continuous, i.e. F(t) = P(S <= t)).
    cum_p = np.cumsum(wp) / n_pos
    cum_n = np.cumsum(wn) / n_neg
    # Evaluate at each threshold via searchsorted into the sorted score arrays.
    # We want F_pos(t) = sum_{s_i <= t, i in +} w_i / n_pos, i.e. the right-most
    # index in sp with value <= t.
    idx_p = np.searchsorted(sp, all_scores, side="right") - 1
    idx_n = np.searchsorted(sn, all_scores, side="right") - 1
    f_pos = np.where(idx_p >= 0, np.take(cum_p, np.clip(idx_p, 0, None)), 0.0)
    f_neg = np.where(idx_n >= 0, np.take(cum_n, np.clip(idx_n, 0, None)), 0.0)
    gaps = np.abs(f_pos - f_neg)
    k = int(np.argmax(gaps))
    return float(gaps[k]), float(all_scores[k]), n_pos, n_neg


def ks(
    sol: Any,
    data: pd.DataFrame,
    *,
    weights: str | np.ndarray | pd.Series | None = None,
) -> KSResult:
    """Kolmogorov-Smirnov statistic on positive vs negative score CDFs."""
    y, scores = resolve_scores_and_target(sol, data)
    w = coerce_weights(weights, data)
    value, at_score, n_pos, n_neg = _ks_from_arrays(y, scores, w)
    return KSResult(value=value, at_score=at_score, n_pos=n_pos, n_neg=n_neg)


# ---------------------------------------------------------------------------
# Cohen's d
# ---------------------------------------------------------------------------


def _cohens_d_from_arrays(
    y: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float, float, float, float]:
    """Return ``(d, mean_pos, mean_neg, pooled_sd)``.

    Uses the pooled-SD form
    :math:`s_p = \\sqrt{\\frac{(n_{+} - 1) s_{+}^2 + (n_{-} - 1) s_{-}^2}{n_{+} + n_{-} - 2}}`
    (Cohen 1988). Weighted variant replaces ``(n - 1)`` with
    ``(sum w - 1)`` for an unbiased weighted variance.
    """
    check_binary_target(y)
    s_pos, s_neg, w_pos, w_neg = split_pos_neg(y, scores, weights)
    n_pos = float(s_pos.size if w_pos is None else np.sum(w_pos))
    n_neg = float(s_neg.size if w_neg is None else np.sum(w_neg))
    if n_pos < 2 or n_neg < 2:
        raise ValueError(
            f"Cohen's d undefined: need at least 2 of each class "
            f"(got n_pos={n_pos}, n_neg={n_neg})"
        )
    mean_pos = weighted_mean(s_pos, w_pos)
    mean_neg = weighted_mean(s_neg, w_neg)
    # Weighted unbiased variance: sum w * (x - mean)^2 / (sum w - 1).
    if w_pos is None:
        var_pos = float(np.var(s_pos, ddof=1))
    else:
        var_pos = float(np.sum(w_pos * (s_pos - mean_pos) ** 2) / (n_pos - 1.0))
    if w_neg is None:
        var_neg = float(np.var(s_neg, ddof=1))
    else:
        var_neg = float(np.sum(w_neg * (s_neg - mean_neg) ** 2) / (n_neg - 1.0))
    pooled_var = ((n_pos - 1.0) * var_pos + (n_neg - 1.0) * var_neg) / (
        n_pos + n_neg - 2.0
    )
    pooled_sd = float(np.sqrt(pooled_var))
    if pooled_sd == 0.0:
        raise ValueError("Cohen's d undefined: pooled SD is zero")
    d = (mean_pos - mean_neg) / pooled_sd
    return float(d), float(mean_pos), float(mean_neg), pooled_sd


def cohens_d(
    sol: Any,
    data: pd.DataFrame,
    *,
    weights: str | np.ndarray | pd.Series | None = None,
) -> CohensDResult:
    """Cohen's *d*: standardized mean difference of scores."""
    y, scores = resolve_scores_and_target(sol, data)
    w = coerce_weights(weights, data)
    d, m_pos, m_neg, pooled = _cohens_d_from_arrays(y, scores, w)
    return CohensDResult(
        value=d, mean_pos=m_pos, mean_neg=m_neg, pooled_sd=pooled
    )


# ---------------------------------------------------------------------------
# DeLong test
# ---------------------------------------------------------------------------


def _midrank(x: np.ndarray) -> np.ndarray:
    """Compute midranks (Sun & Xu 2014 Algorithm 1).

    Ties get the average of their consecutive ranks; matches
    :func:`scipy.stats.rankdata(x, method='average')`.
    """
    n = x.size
    order = np.argsort(x, kind="mergesort")
    x_sorted = x[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x_sorted[j + 1] == x_sorted[i]:
            j += 1
        # Group [i, j] receives the average rank (i + j) / 2 + 1 (1-based)
        avg = 0.5 * (i + j) + 1.0
        ranks[i : j + 1] = avg
        i = j + 1
    # Unscramble back to original order.
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def _delong_components(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, float]:
    """Return ``(auc_a, auc_b, cov_diff)`` for the DeLong paired test.

    Implements the structural-component (V) formulation of DeLong (1988)
    via the Sun & Xu (2014) O((m + n) log(m + n)) algorithm:

    For each model k, define
    :math:`V^k_{10}(X_i) = \\frac{1}{n} \\sum_j [\\mathbf{1}\\{s^k_i > s^k_j\\}
    + 0.5 \\mathbf{1}\\{s^k_i = s^k_j\\}]`
    over negatives j (i indexes positives), and analogously
    :math:`V^k_{01}(Y_j)` over positives. Then
    :math:`AUC_k = \\bar V^k_{10} = \\bar V^k_{01}`.

    The variance of :math:`AUC_a - AUC_b` is
    :math:`\\frac{1}{m} \\widehat{Cov}(V_{10}^a - V_{10}^b)
    + \\frac{1}{n} \\widehat{Cov}(V_{01}^a - V_{01}^b)`
    where :math:`m` is the number of positives and :math:`n` negatives.

    We compute V's via midrank algebra:
    :math:`V^k_{10}(X_i) = (T^k_X(i) - T^k_{XX}(i) + 1) / n` where
    :math:`T^k_X` is the midrank of positive i within the **combined**
    sample and :math:`T^k_{XX}` within positives only. Sun & Xu (2014)
    Algorithm 1.

    Returns ``(auc_a, auc_b, var_diff)`` for the unweighted case.
    """
    pos = y == 1
    neg = ~pos
    m = int(pos.sum())
    n = int(neg.sum())
    if m == 0 or n == 0:
        raise ValueError("DeLong: need positives and negatives.")
    out = []
    v10_list = []
    v01_list = []
    for s in (scores_a, scores_b):
        s_pos = s[pos]
        s_neg = s[neg]
        # Midranks within combined and within each class.
        t_z = _midrank(np.concatenate([s_pos, s_neg]))
        t_x = _midrank(s_pos)
        t_y = _midrank(s_neg)
        tz_pos = t_z[:m]
        tz_neg = t_z[m:]
        # AUC = sum(tz_pos - t_x) / (m * n) + 0.5 + 1/(2n) - ... actually the
        # closed form is AUC = (sum(tz_pos) - m*(m+1)/2) / (m*n).
        auc_val = (np.sum(tz_pos) - m * (m + 1.0) / 2.0) / (m * n)
        # V10 (per positive) and V01 (per negative): see Sun & Xu eq. (3)-(4).
        v10 = (tz_pos - t_x) / float(n)
        v01 = 1.0 - (tz_neg - t_y) / float(m)
        out.append(float(auc_val))
        v10_list.append(v10)
        v01_list.append(v01)
    # Variance of AUC_a - AUC_b: stack V10 and V01 across models.
    v10_a, v10_b = v10_list
    v01_a, v01_b = v01_list
    # Sample (unbiased) covariance of V10_a - V10_b among positives, /m.
    d10 = v10_a - v10_b
    d01 = v01_a - v01_b
    # m_var is sample variance with ddof=1 (Sun & Xu use 1/(m-1) implicitly
    # via the structural-component framing).
    var10 = float(np.var(d10, ddof=1)) if m > 1 else 0.0
    var01 = float(np.var(d01, ddof=1)) if n > 1 else 0.0
    var_diff = var10 / m + var01 / n
    return out[0], out[1], var_diff


def delong_test(
    sol_a: Any,
    sol_b: Any,
    data: pd.DataFrame,
    *,
    weights: str | np.ndarray | pd.Series | None = None,
) -> DeLongResult:
    """DeLong (1988) paired AUC comparison.

    The two solutions must produce predictions for the *same* observations
    in ``data``. The target column is read off ``sol_a.spec.target`` and
    must equal ``sol_b.spec.target``.

    ``weights=`` is accepted for API uniformity but the unweighted
    structural-component variance of DeLong (1988) is what is computed.
    The weighted DeLong variant (Pepe 2004 §5.2) is not implemented in v0;
    passing non-uniform weights raises :class:`NotImplementedError`.

    The numerical reference is Sun & Xu (2014). For a fixed seed and the
    test in ``tests/test_metrics.py``, the value is pinned to the published
    worked example to 1e-6.
    """
    if weights is not None:
        w = coerce_weights(weights, data)
        # Only accept uniform weights.
        if w is not None and not np.allclose(w, w[0]):
            raise NotImplementedError(
                "delong_test with non-uniform weights is not implemented in v0; "
                "use the unweighted DeLong variant. See Pepe (2004) §5.2 for "
                "the weighted reformulation."
            )
    y_a, scores_a = resolve_scores_and_target(sol_a, data)
    y_b, scores_b = resolve_scores_and_target(sol_b, data)
    target_a = sol_a.spec.target
    target_b = sol_b.spec.target
    if target_a != target_b:
        raise ValueError(
            f"sol_a and sol_b must have the same target column; "
            f"got {target_a!r} and {target_b!r}"
        )
    if not np.array_equal(y_a, y_b):
        raise ValueError(
            "sol_a and sol_b must be evaluated on the same target vector "
            "(target arrays disagree on `data`)"
        )
    check_binary_target(y_a)
    auc_a, auc_b, var_diff = _delong_components(scores_a, scores_b, y_a)
    diff = auc_a - auc_b
    if var_diff <= 0.0:
        # Degenerate case (identical scores).
        z = 0.0
        p = 1.0
    else:
        z = diff / float(np.sqrt(var_diff))
        p = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
    pos = int(np.sum(y_a == 1))
    neg = int(np.sum(y_a == 0))
    return DeLongResult(
        auc_a=auc_a,
        auc_b=auc_b,
        diff=diff,
        var_diff=var_diff,
        z=z,
        p_value=p,
        n_pos=pos,
        n_neg=neg,
    )
