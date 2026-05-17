"""Per-segment performance reports — Task P5.B.

This module implements :func:`performance_by_segment`, a pure orchestration
layer over :func:`model_crafter.performance.report.performance`. For every
unique value of a ``by`` column in the held-out evaluation frame we compute
a :class:`~model_crafter.performance.PerformanceReport`, and we bundle the
per-segment reports together with an *aggregate* report over the whole frame.

The split is at *evaluation time only*: ``sol`` is a single fitted
:class:`~model_crafter.solution.Solution`. No re-fitting per segment
happens here — that is :class:`mc.segmented` (Phase 6, DESIGN.md §3.4). The
segmentation here is the deployment-monitoring counterpart of
:func:`mc.performance_over_time` (Task P5.A): "how does my single model
perform on each slice of the population?".

DESIGN.md §3.4 names this artefact explicitly:

    ``mc.performance_by_segment(sol, data=test)`` — per-segment reports.

Behaviour notes (documented in ``notes/P5.B.md``):

* Numeric segment keys are coerced to ``str`` in the result dict so
  downstream tabulation (``pd.DataFrame.from_dict``) gets stable, sortable
  column labels.
* Empty segments — categorical levels with zero observations after
  filtering — are silently dropped rather than reported. Reporting a
  ``PerformanceReport`` over an empty frame is meaningless; advertising the
  drop in :func:`groupby` semantics keeps the API tight.
* Unknown ``by`` columns raise :class:`KeyError` with the offending name.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Any

import pandas as pd

from model_crafter.performance.report import (
    PerformanceReport,
    performance,
)

__all__ = [
    "SegmentedPerformanceReport",
    "performance_by_segment",
]


# The bundled value


@dataclass(frozen=True, slots=True)
class SegmentedPerformanceReport:
    """Per-segment performance bundle (DESIGN.md §3.4).

    Attributes
    ----------
    segments : dict[str, PerformanceReport]
        One entry per unique value of the segmentation column. Keys are
        stringified for stable downstream usage; values are full
        :class:`~model_crafter.performance.PerformanceReport`\\ s computed
        on the segment's slice of the evaluation frame.
    aggregate : PerformanceReport
        The unsegmented full-data report — the same value
        :func:`mc.performance` would return — included for context.
    """

    segments: dict[str, PerformanceReport]
    aggregate: PerformanceReport

    # Repr — aggregate first, then a compact per-segment summary
    def __repr__(self) -> str:
        lines: list[str] = ["SegmentedPerformanceReport"]
        agg = self.aggregate
        ev_rate = (
            (agg.n_events / agg.n_obs * 100.0) if agg.n_obs > 0 else 0.0
        )
        lines.append(
            f"  aggregate: n={agg.n_obs:,}  events={agg.n_events:,} "
            f"({ev_rate:.1f}%)"
        )
        lines.append(
            "             "
            f"AUC={agg.discrimination.auc.value:.4f}  "
            f"KS={agg.discrimination.ks.value:.4f}  "
            f"Brier={agg.calibration.brier.value:.4f}"
        )
        if self.segments:
            # Stable, sortable ordering on the stringified keys.
            keys = sorted(self.segments.keys())
            key_w = max(len(k) for k in keys)
            lines.append("")
            lines.append(f"  segments ({len(keys)}):")
            for k in keys:
                sub = self.segments[k]
                lines.append(
                    f"    {k:<{key_w}}  "
                    f"n={sub.n_obs:>6,}  "
                    f"AUC={sub.discrimination.auc.value:.4f}  "
                    f"KS={sub.discrimination.ks.value:.4f}  "
                    f"Brier={sub.calibration.brier.value:.4f}"
                )
        else:
            lines.append("  segments: (none)")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        agg = self.aggregate
        ev_rate = (agg.n_events / agg.n_obs * 100.0) if agg.n_obs > 0 else 0.0
        rows = []
        for k in sorted(self.segments.keys()):
            sub = self.segments[k]
            rows.append(
                "<tr>"
                f"<td>{html.escape(k)}</td>"
                f"<td>{sub.n_obs:,}</td>"
                f"<td>{sub.discrimination.auc.value:.4f}</td>"
                f"<td>{sub.discrimination.ks.value:.4f}</td>"
                f"<td>{sub.calibration.brier.value:.4f}</td>"
                "</tr>"
            )
        body = (
            "<table class='mc-segmented-performance'>"
            "<thead><tr><th>Segment</th><th>n_obs</th>"
            "<th>AUC</th><th>KS</th><th>Brier</th></tr></thead>"
            f"<tbody>{''.join(rows) or '<tr><td colspan=5>(none)</td></tr>'}</tbody>"
            "</table>"
        )
        return (
            "<div class='mc-segmented-performance-report'>"
            "<strong>SegmentedPerformanceReport</strong> "
            f"<span>aggregate n={agg.n_obs:,}, events={agg.n_events:,} "
            f"({ev_rate:.1f}%), AUC={agg.discrimination.auc.value:.4f}</span>"
            f"{body}"
            "</div>"
        )


# Orchestration


def performance_by_segment(
    sol: Any,
    data: pd.DataFrame,
    by: str,
    *,
    weights: str | None = None,
    reference: pd.DataFrame | None = None,
) -> SegmentedPerformanceReport:
    """Compute per-segment :class:`PerformanceReport`\\ s plus an aggregate.

    The ``sol`` is a *single* fitted ``Solution``; segmentation happens on
    the held-out ``data`` only. This is the deployment-monitoring view
    described in DESIGN.md §3.4 — orthogonal to :class:`mc.segmented`,
    which re-fits per segment in Phase 6.

    Parameters
    ----------
    sol :
        A fitted :class:`~model_crafter.solution.Solution`.
    data :
        Held-out evaluation frame containing ``sol.spec.target`` and the
        segmentation column ``by``.
    by :
        Column name in ``data``. Each unique value defines one segment.
    weights :
        Column name in ``data`` whose values are sample weights, passed
        through to :func:`mc.performance` per segment and on the aggregate.
        ``None`` is uniform.
    reference :
        Optional reference frame for the PSI stability sub-report; passed
        through unchanged to :func:`mc.performance` for *every* segment and
        the aggregate. See :func:`mc.performance` for accepted forms.

    Returns
    -------
    SegmentedPerformanceReport

    Raises
    ------
    KeyError
        If ``by`` is not a column of ``data``.
    """
    if by not in data.columns:
        raise KeyError(
            f"by={by!r} is not a column of data; got {list(data.columns)!r}"
        )

    # Aggregate first — that's our "all rows" reference and the field on
    # the bundled value.
    aggregate = performance(
        sol, data, weights=weights, reference=reference
    )

    # ``observed=True`` skips Categorical levels with no rows — that's our
    # "drop empty segments silently" contract.
    segments: dict[str, PerformanceReport] = {}
    for key, slice_ in data.groupby(by, observed=True, sort=False):
        if len(slice_) == 0:
            # Defensive — ``observed=True`` already handles Categorical
            # gaps, but a non-categorical column with NaN handling could
            # still surface an empty group depending on pandas version.
            continue
        seg_key = _stringify_key(key)
        segments[seg_key] = performance(
            sol, slice_, weights=weights, reference=reference
        )

    return SegmentedPerformanceReport(
        segments=segments, aggregate=aggregate
    )


# Helpers


def _stringify_key(key: Any) -> str:
    """Coerce a groupby key to a string while preserving sensible
    formatting for common numeric types (so ``0`` becomes ``"0"`` not
    ``"0.0"`` if the column is integer-typed).
    """
    # ``str`` is identity on str. ``str(np.int64(0))`` gives ``"0"``;
    # ``str(np.float64(0.5))`` gives ``"0.5"``. Pandas hands us numpy
    # scalars from ``groupby`` so we lean on numpy's repr.
    return str(key)
