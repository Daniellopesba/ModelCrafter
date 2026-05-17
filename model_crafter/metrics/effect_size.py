"""Cohen's *d* — the standardised mean difference between two groups.

    d = (mean_+ - mean_-) / s_p

with pooled SD

    s_p = sqrt( ((n_+ - 1) s_+^2 + (n_- - 1) s_-^2) / (n_+ + n_- - 2) ).

Cohen (1988). The weighted variant replaces ``(n - 1)`` with ``(sum w - 1)``
for an unbiased weighted variance.

Discrimination metrics (AUC, KS) measure rank separation; Cohen's *d*
measures separation in raw score units. They answer related but distinct
questions, hence the separate module.
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
    weighted_mean,
)

__all__ = ["CohensDResult", "cohens_d"]


@dataclass(frozen=True, slots=True)
class CohensDResult:
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


def _cohens_d_from_arrays(
    y: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float, float, float, float]:
    """Return ``(d, mean_pos, mean_neg, pooled_sd)`` (Cohen 1988)."""
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
    return float((mean_pos - mean_neg) / pooled_sd), float(mean_pos), float(mean_neg), pooled_sd


def cohens_d(
    sol: Any,
    data: pd.DataFrame,
    *,
    weights: str | np.ndarray | pd.Series | None = None,
) -> CohensDResult:
    y, scores = resolve_scores_and_target(sol, data)
    w = coerce_weights(weights, data)
    d, m_pos, m_neg, pooled = _cohens_d_from_arrays(y, scores, w)
    return CohensDResult(
        value=d, mean_pos=m_pos, mean_neg=m_neg, pooled_sd=pooled
    )
