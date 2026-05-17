"""Rank-based metrics: lift table and cumulative-gains curve.

DESIGN.md §3.3 specifies that the :class:`PerformanceReport` exposes a
lift table and a cumulative-gains curve as separate top-level fields,
so they need their own primitives.

* :func:`lift_table` cuts observations into ``n_deciles`` (default 10)
  equal-weight buckets *by predicted score, descending*, then computes:
    - the (weighted) mean predicted score per bucket
    - the (weighted) event rate per bucket
    - the lift = ``event_rate / overall_event_rate``
    - the cumulative captured response = ``cum_events / total_events``
* :func:`cumulative_gains` returns the cumulative-population vs
  cumulative-captured-events curve.

References
----------
* Lewis, M. (2005). *Comparative evaluation of model selection methods
  for direct marketing.* Discusses decile lift tables.
* Hand, D. J. (2009). *Measuring classifier performance: a coherent
  alternative to the area under the ROC curve.* Machine Learning 77:
  103-123. Discusses the gains curve as a Lorenz-style summary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from model_crafter.metrics._common import (
    check_binary_target,
    coerce_weights,
    resolve_scores_and_target,
)

__all__ = ["GainsCurve", "LiftTable", "cumulative_gains", "lift_table"]


# Result dataclasses


@dataclass(frozen=True, slots=True)
class LiftTable:
    """Decile lift table (DESIGN.md §3.3).

    Columns: ``decile`` (1 = highest score), ``mean_score``, ``event_rate``,
    ``lift``, ``captured_response``.
    """

    table: pd.DataFrame = field(repr=False)
    n_deciles: int

    def __repr__(self) -> str:
        lines = [f"LiftTable ({self.n_deciles} buckets)"]
        lines.append(
            "  dec  mean_score  event_rate    lift   captured"
        )
        for _, row in self.table.iterrows():
            lines.append(
                f"  {int(row['decile']):>3d}  {row['mean_score']:>10.4f}  "
                f"{row['event_rate']:>10.4f}  {row['lift']:>6.2f}  "
                f"{row['captured_response']:>9.4f}"
            )
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class GainsCurve:
    """Cumulative gains curve (DESIGN.md §3.3).

    Two parallel arrays:

    * ``cum_population`` — cumulative population fraction (sorted by score
      descending). Includes a leading ``0.0``.
    * ``cum_captured`` — cumulative captured event fraction.
      Includes a leading ``0.0``.

    The first non-trivial point is at fraction ``1/n``; the last is at
    ``(1.0, 1.0)``.
    """

    cum_population: np.ndarray = field(repr=False)
    cum_captured: np.ndarray = field(repr=False)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "cum_population": self.cum_population,
                "cum_captured": self.cum_captured,
            }
        )

    def __repr__(self) -> str:
        # Print a few percentile checkpoints.
        checkpoints = (0.10, 0.20, 0.50, 1.00)
        lines = ["GainsCurve"]
        lines.append("  pop_frac  captured_frac")
        for c in checkpoints:
            # Find first index where cum_population >= c.
            idx = int(np.searchsorted(self.cum_population, c, side="left"))
            idx = min(idx, self.cum_population.size - 1)
            lines.append(
                f"  {self.cum_population[idx]:>8.2f}  {self.cum_captured[idx]:>13.4f}"
            )
        return "\n".join(lines)


# Lift table


def _lift_table_from_arrays(
    y: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray | None,
    n_deciles: int,
) -> pd.DataFrame:
    """Cut ``(y, scores, weights)`` into ``n_deciles`` equal-weight buckets
    by score (descending) and tabulate.

    The buckets are equal-weight on cumulative weight (so weighted obs lift
    is correct). The "decile" index is 1 for the highest-scoring bucket.
    """
    check_binary_target(y)
    if n_deciles < 1:
        raise ValueError(f"n_deciles must be >= 1; got {n_deciles}")
    w = (
        np.ones_like(scores, dtype=float)
        if weights is None
        else np.asarray(weights, dtype=float)
    )
    # Sort descending by score; stable so ties keep their original order.
    order = np.argsort(-scores, kind="mergesort")
    s_sorted = scores[order]
    y_sorted = y[order]
    w_sorted = w[order]
    total_w = float(np.sum(w_sorted))
    if total_w == 0.0:
        raise ValueError("lift table: total weight is zero")
    cum_w = np.cumsum(w_sorted)
    # Use "previous mass" to assign bucket: every observation whose cum-fraction
    # *before* it sits in [b/n_deciles, (b+1)/n_deciles) gets bucket b. The
    # ``+ 1e-12`` absorbs floating-point error on exact integer boundaries.
    cum_w_prev = cum_w - w_sorted
    bucket = np.floor((cum_w_prev / total_w) * n_deciles + 1e-12).astype(int)
    bucket = np.clip(bucket, 0, n_deciles - 1)
    # Aggregate per bucket.
    mean_score = np.zeros(n_deciles)
    event_rate = np.zeros(n_deciles)
    bucket_weight = np.zeros(n_deciles)
    bucket_events = np.zeros(n_deciles)
    np.add.at(bucket_weight, bucket, w_sorted)
    np.add.at(bucket_events, bucket, w_sorted * y_sorted)
    np.add.at(mean_score, bucket, w_sorted * s_sorted)
    # Normalise.
    safe = bucket_weight > 0
    mean_score[safe] /= bucket_weight[safe]
    event_rate[safe] = bucket_events[safe] / bucket_weight[safe]
    mean_score[~safe] = np.nan
    event_rate[~safe] = np.nan
    overall_event_rate = float(np.sum(w_sorted * y_sorted) / total_w)
    lift = (
        np.full(n_deciles, np.nan)
        if overall_event_rate == 0.0
        else np.where(safe, event_rate / overall_event_rate, np.nan)
    )
    # captured_response = cumulative events / total events (highest first).
    total_events = float(np.sum(w_sorted * y_sorted))
    cum_events = np.cumsum(bucket_events)
    captured = (
        np.full(n_deciles, np.nan)
        if total_events == 0.0
        else cum_events / total_events
    )
    table = pd.DataFrame(
        {
            "decile": np.arange(1, n_deciles + 1, dtype=int),
            "mean_score": mean_score,
            "event_rate": event_rate,
            "lift": lift,
            "captured_response": captured,
            "n": bucket_weight,
        }
    )
    return table


def lift_table(
    sol: Any,
    data: pd.DataFrame,
    *,
    n_deciles: int = 10,
    weights: str | np.ndarray | pd.Series | None = None,
) -> LiftTable:
    """Decile lift table sorted by score descending.

    Bucket 1 is the highest-scoring fraction. ``lift`` is the bucket's
    event rate divided by the overall event rate; ``captured_response`` is
    the cumulative share of events captured by all buckets up to and
    including this one.
    """
    y, scores = resolve_scores_and_target(sol, data)
    w = coerce_weights(weights, data)
    table = _lift_table_from_arrays(y, scores, w, n_deciles)
    return LiftTable(table=table, n_deciles=n_deciles)


# Cumulative gains


def _cumulative_gains_from_arrays(
    y: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(cum_population, cum_captured)`` arrays.

    Both arrays have length ``n + 1`` and start at ``(0, 0)``. The final
    entry is ``(1, 1)`` unless ``total_events == 0`` (in which case
    cum_captured is ``NaN``).
    """
    check_binary_target(y)
    w = (
        np.ones_like(scores, dtype=float)
        if weights is None
        else np.asarray(weights, dtype=float)
    )
    order = np.argsort(-scores, kind="mergesort")
    w_sorted = w[order]
    y_sorted = y[order]
    total_w = float(np.sum(w_sorted))
    total_events = float(np.sum(w_sorted * y_sorted))
    cum_pop = np.concatenate(([0.0], np.cumsum(w_sorted) / total_w))
    if total_events == 0.0:
        cum_cap = np.full(cum_pop.size, np.nan)
        cum_cap[0] = 0.0
    else:
        cum_cap = np.concatenate(
            ([0.0], np.cumsum(w_sorted * y_sorted) / total_events)
        )
    return cum_pop, cum_cap


def cumulative_gains(
    sol: Any,
    data: pd.DataFrame,
    *,
    weights: str | np.ndarray | pd.Series | None = None,
) -> GainsCurve:
    """Cumulative gains curve (cum population fraction vs cum captured events).

    The curve starts at ``(0, 0)`` and ends at ``(1, 1)``; a perfect model
    rises immediately to capture all events in the top portion of the
    score-sorted sample.
    """
    y, scores = resolve_scores_and_target(sol, data)
    w = coerce_weights(weights, data)
    cum_pop, cum_cap = _cumulative_gains_from_arrays(y, scores, w)
    return GainsCurve(cum_population=cum_pop, cum_captured=cum_cap)
