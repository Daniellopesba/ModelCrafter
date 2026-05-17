"""Midrank and structural-component primitives behind DeLong's test.

These are shared between :func:`~model_crafter.performance.compare.delong_test`
(paired AUC comparison, public) and the AUC CI computed inside
:class:`PerformanceReport` (single-model variance). Sun & Xu (2014)
*Fast implementation of DeLong's algorithm for comparing the areas under
correlated receiver operating characteristic curves*, IEEE Signal Processing
Letters 21(11): 1389-1393, Algorithm 1.

The two functions exposed here are private: they take raw arrays, not
``Solution`` objects, and they don't accept weights. The weighted DeLong
variant (Pepe 2004 §5.2) is out of v0 scope; consumers in this package
gate non-uniform weights at the public-API layer.
"""

from __future__ import annotations

import numpy as np

__all__ = ["_delong_components", "_midrank"]


def _midrank(x: np.ndarray) -> np.ndarray:
    """Midrank of ``x`` (ties get the average of their consecutive ranks).

    Matches ``scipy.stats.rankdata(x, method='average')``. Implemented
    in O(n log n) via a single mergesort plus a tied-group sweep.
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
        ranks[i : j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def _delong_components(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, float]:
    """Return ``(auc_a, auc_b, var_diff)`` via Sun & Xu (2014) Algorithm 1.

    For each model k, the structural components are

        V^k_{10}(X_i) = (T^k_X(i) - T^k_{XX}(i)) / n
        V^k_{01}(Y_j) = 1 - (T^k_Y(j) - T^k_{YY}(j)) / m

    where T^k_X is the midrank of positive i within the combined sample,
    T^k_{XX} the midrank within positives only, and analogously for Y.
    AUC_k equals the mean of V^k_{10}; the variance of AUC_a - AUC_b is

        var(V_{10}^a - V_{10}^b, ddof=1) / m
            + var(V_{01}^a - V_{01}^b, ddof=1) / n.

    ``ddof=1`` matches the structural-component framing in Sun & Xu and
    pins ``tests/test_metrics.py`` against ``pROC::roc.test`` to 1e-6.
    """
    pos = y == 1
    neg = ~pos
    m = int(pos.sum())
    n = int(neg.sum())
    if m == 0 or n == 0:
        raise ValueError("DeLong: need positives and negatives.")

    aucs: list[float] = []
    v10s: list[np.ndarray] = []
    v01s: list[np.ndarray] = []
    for s in (scores_a, scores_b):
        s_pos = s[pos]
        s_neg = s[neg]
        t_z = _midrank(np.concatenate([s_pos, s_neg]))
        t_x = _midrank(s_pos)
        t_y = _midrank(s_neg)
        tz_pos = t_z[:m]
        tz_neg = t_z[m:]
        auc_val = (np.sum(tz_pos) - m * (m + 1.0) / 2.0) / (m * n)
        v10 = (tz_pos - t_x) / float(n)
        v01 = 1.0 - (tz_neg - t_y) / float(m)
        aucs.append(float(auc_val))
        v10s.append(v10)
        v01s.append(v01)

    d10 = v10s[0] - v10s[1]
    d01 = v01s[0] - v01s[1]
    var10 = float(np.var(d10, ddof=1)) if m > 1 else 0.0
    var01 = float(np.var(d01, ddof=1)) if n > 1 else 0.0
    var_diff = var10 / m + var01 / n
    return aucs[0], aucs[1], var_diff
