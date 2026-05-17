r""":func:`performance_over_time` — temporal :class:`PerformanceReport` bundles.

DESIGN.md §3.3 ("Temporal performance") pins the deployment-monitoring
view of a fitted model: a single :class:`~model_crafter.solution.Solution`
evaluated across a temporal splitter's *validation* windows, producing a
:class:`PerformanceReport` per window and a time-indexed summary frame for
quick visual inspection of AUC / KS / Brier / PSI drift.

This is the temporal sibling of :func:`mc.performance` — the model is
fixed, the data slides. The structure mirrors the spec for ``mc.over_time``
(P3.B) but bundles full :class:`PerformanceReport`\ s rather than a single
metric series.

Math notes (DESIGN.md §3.3)
---------------------------

For each validation window :math:`W_i` produced by ``splitter.split(data)``
we compute the full performance bundle on :math:`W_i` and record the
window's *midpoint* timestamp

.. math::

    \tau_i = \min_{j \in W_i} t_j +
             \tfrac{1}{2}\bigl(\max_{j \in W_i} t_j - \min_{j \in W_i} t_j\bigr)

as the row label of the resulting summary frame. PSI is computed against
``reference`` (a fixed anchor) on every window, so PSI drift is what the
column captures — the reference does not slide.

The model is fixed across windows; ``weights`` is resolved per window
because the weight column lives on the window's frame, not on the model.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from model_crafter.performance.report import PerformanceReport, performance

if TYPE_CHECKING:  # pragma: no cover — type-checker only
    from matplotlib.figure import Figure

__all__ = ["TemporalPerformanceReport", "performance_over_time"]


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TemporalPerformanceReport:
    r"""Time-indexed bundle of :class:`PerformanceReport` values.

    Fields
    ------
    summary :
        ``DataFrame`` indexed by the midpoint of each validation window
        (``DatetimeIndex``). Columns: ``n_obs``, ``n_events``, ``auc``,
        ``ks``, ``brier``, ``log_loss``, ``mean_p``, plus ``psi`` when a
        reference was supplied. Each row is the headline numbers from
        the per-window :class:`PerformanceReport`.
    reports :
        Tuple of :class:`PerformanceReport`\ s, one per validation
        window, in chronological order.

    See DESIGN.md §3.3 (Temporal performance) for the pen-and-paper
    description and §10's north-star example for the call site.
    """

    summary: pd.DataFrame
    reports: tuple[PerformanceReport, ...]

    def plot(self) -> "Figure":  # noqa: UP037 — Figure is TYPE_CHECKING-only  # pragma: no cover
        """Diagnostic plot: AUC + KS over time on the primary axis, PSI
        on a secondary axis when present.

        Lazy ``matplotlib`` import — the package itself does not depend
        on matplotlib; we raise a friendly :class:`ImportError` when the
        helper is invoked without it installed. The numerical primitives
        are unaffected.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover — import-time path
            raise ImportError(
                "TemporalPerformanceReport.plot() requires matplotlib; "
                "install it with `pip install matplotlib`."
            ) from exc

        fig, ax = plt.subplots(figsize=(8, 4))
        x = self.summary.index
        ax.plot(x, self.summary["auc"], marker="o", label="AUC")
        ax.plot(x, self.summary["ks"], marker="s", label="KS")
        ax.set_ylabel("AUC / KS")
        ax.set_xlabel("validation window midpoint")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        if "psi" in self.summary.columns:
            ax2 = ax.twinx()
            ax2.plot(
                x,
                self.summary["psi"],
                marker="^",
                color="tab:red",
                label="PSI",
            )
            ax2.axhline(0.25, color="tab:red", linestyle="--", alpha=0.4)
            ax2.set_ylabel("PSI (vs reference)")
            # Combined legend so PSI shows up alongside AUC/KS.
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc="best")
        else:
            ax.legend(loc="best")
        ax.set_title("Performance over time")
        fig.tight_layout()
        return fig

    def __repr__(self) -> str:
        if len(self.reports) == 0:
            return "TemporalPerformanceReport(empty)"
        n_windows = len(self.reports)
        header = (
            f"TemporalPerformanceReport ({n_windows} window"
            f"{'s' if n_windows != 1 else ''})"
        )
        # Render the summary frame compactly: 4-decimal floats, ISO dates.
        with pd.option_context(
            "display.float_format",
            lambda v: f"{v:.4f}",
            "display.max_rows",
            32,
            "display.width",
            120,
        ):
            body = self.summary.to_string()
        # One-line per-window AUC/KS/PSI summary beneath the frame.
        bullets = []
        for ts, rep in zip(self.summary.index, self.reports, strict=True):
            auc_v = rep.discrimination.auc.value
            ks_v = rep.discrimination.ks.value
            brier_v = rep.calibration.brier.value
            psi_str = (
                f"  PSI={rep.stability.psi.value:.3f}"
                if rep.stability is not None
                else ""
            )
            # pd.Timestamp(ts) round-trips a datetime-like index entry
            # into a Timestamp with an isoformat() method.
            ts_str = pd.Timestamp(ts).isoformat()  # type: ignore[attr-defined]
            bullets.append(
                f"  {ts_str}  AUC={auc_v:.4f}  KS={ks_v:.4f}  "
                f"Brier={brier_v:.4f}{psi_str}  "
                f"(n={rep.n_obs}, events={rep.n_events})"
            )
        return "\n".join([header, "", body, "", *bullets]).rstrip()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _iter_valid_windows(
    splitter: Any, data: pd.DataFrame
) -> Iterable[pd.DataFrame]:
    """Yield the *validation* slices of ``splitter.split(data)``.

    Empty validation windows are skipped silently (matching P3.B's
    :func:`over_time` contract). A splitter without a ``time_col``
    attribute is rejected — temporal performance is meaningless without
    a time axis.
    """
    if not hasattr(splitter, "split"):
        raise TypeError(
            f"splitter must expose .split(df); got {type(splitter).__name__}"
        )
    time_col = getattr(splitter, "time_col", None)
    if time_col is None:
        raise ValueError(
            "performance_over_time requires a temporal splitter "
            "(splitter.time_col must be set)"
        )
    if time_col not in data.columns:
        raise KeyError(
            f"splitter.time_col={time_col!r} not in data "
            f"(columns: {list(data.columns)})"
        )
    for _train, valid in splitter.split(data):
        if len(valid) == 0:
            continue
        yield valid


def _window_midpoint(window: pd.DataFrame, time_col: str) -> pd.Timestamp:
    """Compute the midpoint timestamp of ``window`` along ``time_col``.

    Matches the convention used by :func:`model_crafter.validation.over_time`
    so the indices line up across the two helpers.
    """
    ts = pd.to_datetime(window[time_col])
    v_lo = ts.min()
    v_hi = ts.max()
    return v_lo + (v_hi - v_lo) / 2


def performance_over_time(
    sol: Any,
    data: pd.DataFrame,
    splitter: Any,
    *,
    reference: pd.DataFrame | None = None,
    weights: str | None = None,
) -> TemporalPerformanceReport:
    r"""Compute a :class:`PerformanceReport` for each validation window
    produced by ``splitter`` and bundle them into a
    :class:`TemporalPerformanceReport`.

    The model is fixed: ``sol`` is *not* refit per window. Only the
    *validation* slices of ``splitter.split(data)`` are consumed — for
    each one we call :func:`mc.performance` to materialise discrimination,
    calibration, stability, and distribution sub-reports. ``reference``,
    if given, is the *same* anchor across every window so PSI tracks
    population drift over time (DESIGN.md §3.3 "Temporal performance").

    Parameters
    ----------
    sol :
        A fitted :class:`~model_crafter.solution.Solution`-like value
        accepted by :func:`mc.predict`.
    data :
        A :class:`pandas.DataFrame` containing the splitter's
        ``time_col``, the target column referenced by ``sol.spec.target``,
        and (if used) the ``weights`` column.
    splitter :
        Any temporal splitter conforming to P3.B's contract
        (``splitter.split(data) -> Iterator[(train, valid)]`` and a
        ``time_col`` attribute).
    reference :
        Optional :class:`pandas.DataFrame` used to compute PSI drift on
        each window. The reference is scored once per window via
        :func:`mc.predict`. Pass ``None`` to omit the stability sub-
        report (and the ``psi`` summary column).
    weights :
        Optional column name in ``data``. The corresponding column in
        each validation window is passed straight through to
        :func:`mc.performance`.

    Returns
    -------
    TemporalPerformanceReport
        A frozen value with ``summary`` (time-indexed
        :class:`pandas.DataFrame`) and ``reports`` (tuple of per-window
        :class:`PerformanceReport`\ s).

    Raises
    ------
    TypeError
        If ``splitter`` does not implement ``.split()`` or ``data`` is
        not a :class:`pandas.DataFrame`.
    ValueError
        If the splitter lacks a ``time_col``.
    KeyError
        If ``time_col`` or ``weights`` is missing from ``data``.

    Notes
    -----
    See DESIGN.md §3.3 for the user-facing description and §10 for the
    north-star integration with ``mc.expanding_window`` / ``mc.tune``.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be a DataFrame; got {type(data).__name__}")
    if weights is not None and not isinstance(weights, str):
        # The pinned contract is ``weights: str | None`` — column name only.
        # Anything else risks ambiguity about whether it indexes ``data`` or
        # the per-window slice; we keep the surface narrow.
        raise TypeError(
            f"weights must be a column name (str) or None; got "
            f"{type(weights).__name__}"
        )
    if weights is not None and weights not in data.columns:
        raise KeyError(
            f"weights column {weights!r} not in data "
            f"(columns: {list(data.columns)})"
        )
    time_col = getattr(splitter, "time_col", None)
    if time_col is None:
        raise ValueError(
            "performance_over_time requires a temporal splitter "
            "(splitter.time_col must be set)"
        )

    midpoints: list[pd.Timestamp] = []
    reports: list[PerformanceReport] = []
    summary_rows: list[dict[str, float]] = []
    for valid in _iter_valid_windows(splitter, data):
        rep = performance(
            sol,
            valid,
            weights=weights,
            reference=reference,
        )
        midpoints.append(_window_midpoint(valid, time_col))
        reports.append(rep)
        row: dict[str, float] = {
            "n_obs": float(rep.n_obs),
            "n_events": float(rep.n_events),
            "auc": float(rep.discrimination.auc.value),
            "ks": float(rep.discrimination.ks.value),
            "brier": float(rep.calibration.brier.value),
            "log_loss": float(rep.calibration.log_loss.value),
            "mean_p": float(rep.distribution.mean),
        }
        if rep.stability is not None:
            row["psi"] = float(rep.stability.psi.value)
        summary_rows.append(row)

    columns: list[str] = [
        "n_obs",
        "n_events",
        "auc",
        "ks",
        "brier",
        "log_loss",
        "mean_p",
    ]
    if reference is not None:
        columns.append("psi")
    if len(summary_rows) == 0:
        summary = pd.DataFrame(
            {c: pd.Series(dtype=float) for c in columns},
            index=pd.DatetimeIndex([], name=time_col),
        )
    else:
        summary = pd.DataFrame.from_records(summary_rows, columns=columns)
        # n_obs / n_events are conceptually integer counts; cast for nice
        # printing while keeping floats in the DataFrame value-space.
        summary["n_obs"] = summary["n_obs"].astype(np.int64)
        summary["n_events"] = summary["n_events"].astype(np.int64)
        summary.index = pd.DatetimeIndex(midpoints, name=time_col)
    return TemporalPerformanceReport(summary=summary, reports=tuple(reports))
