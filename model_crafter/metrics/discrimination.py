"""Rank-based discrimination metrics: AUC, Gini, KS.

AUC is the Mann-Whitney U statistic normalised by ``n_pos * n_neg``:

    AUC = P(s_+ > s_-) + 0.5 * P(s_+ = s_-)

equivalently the probability that a randomly chosen positive scores
higher than a randomly chosen negative. With sample weights it becomes

    AUC_w = (sum_{i in +} sum_{j in -} w_i w_j [1{s_i > s_j} + 0.5 * 1{s_i = s_j}])
            / (sum w_+ * sum w_-).

Gini is the affine reparametrisation ``2 * AUC - 1``.

KS is the maximum absolute gap between the weighted empirical CDFs of
positives and negatives — Kolmogorov-Smirnov on conditional score
distributions. Matches ``scipy.stats.ks_2samp`` in the unweighted case.

References:

* Mann, H. B. and Whitney, D. R. (1947).
* Sun & Xu (2014) — tie-handling via midrank algebra; we use the same
  midrank convention for the weighted AUC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from model_crafter.metrics._common import (
    check_binary_target,
    coerce_weights,
    resolve_scores_and_target,
    split_pos_neg,
)

__all__ = [
    "AUCResult",
    "GiniResult",
    "KSResult",
    "auc",
    "gini",
    "ks",
]


@dataclass(frozen=True, slots=True)
class AUCResult:
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


def _auc_from_arrays(
    y: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Return ``(auc, n_pos, n_neg)`` via weighted Mann-Whitney U.

    Sorting once and sweeping tied groups gives O(n log n). Midrank
    weighting recovers the unweighted rank when ``w = 1``; with non-uniform
    weights it averages over cumulative-weight intervals, which is the
    weighted analogue of ``scipy.stats.rankdata(method='average')``.
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
        sum_w = float(np.sum(sorted_weights[i : j + 1]))
        avg_rank = cum_before + 0.5 + 0.5 * sum_w
        midranks[i : j + 1] = avg_rank
        cum_before += sum_w
        i = j + 1

    sw_pos = sorted_weights[sorted_is_pos]
    rank_pos = midranks[sorted_is_pos]
    r_pos = float(np.sum(sw_pos * rank_pos))
    u = r_pos - n_pos * (n_pos + 1.0) / 2.0
    return float(u / (n_pos * n_neg)), n_pos, n_neg


def auc(
    sol: Any,
    data: pd.DataFrame,
    *,
    weights: str | np.ndarray | pd.Series | None = None,
) -> AUCResult:
    y, scores = resolve_scores_and_target(sol, data)
    w = coerce_weights(weights, data)
    value, n_pos, n_neg = _auc_from_arrays(y, scores, w)
    return AUCResult(value=value, n_pos=n_pos, n_neg=n_neg, se=None)


def gini(
    sol: Any,
    data: pd.DataFrame,
    *,
    weights: str | np.ndarray | pd.Series | None = None,
) -> GiniResult:
    y, scores = resolve_scores_and_target(sol, data)
    w = coerce_weights(weights, data)
    auc_value, n_pos, n_neg = _auc_from_arrays(y, scores, w)
    return GiniResult(value=2.0 * auc_value - 1.0, n_pos=n_pos, n_neg=n_neg)


def _ks_from_arrays(
    y: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float, float, float, float]:
    """Return ``(ks, at_score, n_pos, n_neg)`` from weighted ECDFs.

    The two empirical CDFs are evaluated at every unique merged score; the
    KS statistic is the max of their absolute difference.
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
    if weights is None:
        w_pos = np.ones_like(s_pos)
        w_neg = np.ones_like(s_neg)
    order_pos = np.argsort(s_pos, kind="mergesort")
    order_neg = np.argsort(s_neg, kind="mergesort")
    sp = s_pos[order_pos]
    sn = s_neg[order_neg]
    wp = np.asarray(w_pos, dtype=float)[order_pos]
    wn = np.asarray(w_neg, dtype=float)[order_neg]
    all_scores = np.unique(np.concatenate([sp, sn]))
    cum_p = np.cumsum(wp) / n_pos
    cum_n = np.cumsum(wn) / n_neg
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
    y, scores = resolve_scores_and_target(sol, data)
    w = coerce_weights(weights, data)
    value, at_score, n_pos, n_neg = _ks_from_arrays(y, scores, w)
    return KSResult(value=value, at_score=at_score, n_pos=n_pos, n_neg=n_neg)
