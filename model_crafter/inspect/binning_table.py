r"""WoE / bin-indicator inspection.

Walks ``sol.spec.features`` for :class:`WoETerm` and :class:`BinnedTerm`
values, materialises their stored :class:`BinningResult` into per-feature
DataFrames, and computes the per-bin IV contribution

.. math::

    \mathrm{IV}_b = (p^{(1)}_b - p^{(0)}_b) \cdot \mathrm{WoE}_b

alongside the column totals. Industry IV rules of thumb (Siddiqi 2006,
Anderson 2007): < 0.02 useless, 0.02-0.1 weak, 0.1-0.3 medium, 0.3-0.5
strong, > 0.5 suspiciously high (likely target leakage). The package
reports the values; it does not enforce these thresholds.
"""

from __future__ import annotations

import html
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

__all__ = ["BinningTable", "binning_table"]


@dataclass(frozen=True, slots=True)
class BinningTable:
    """Per-term binning summary plus a total-IV Series.

    ``tables`` maps each WoE/binned column name to a DataFrame with
    columns ``["bin", "n", "n_events", "event_rate", "woe", "iv"]``
    (``iv`` is the per-bin contribution to total IV). ``iv`` is the
    Series of total IV per column.
    """

    tables: Mapping[str, pd.DataFrame]
    iv: pd.Series

    def __post_init__(self) -> None:
        for col, df in self.tables.items():
            expected = ["bin", "n", "n_events", "event_rate", "woe", "iv"]
            if list(df.columns) != expected:
                raise ValueError(
                    f"binning_table['{col}'] must have columns {expected}; "
                    f"got {list(df.columns)}"
                )

    def __repr__(self) -> str:
        if not self.tables:
            return "BinningTable(empty)"
        lines = ["BinningTable"]
        for col, df in self.tables.items():
            iv = float(self.iv[col]) if col in self.iv.index else 0.0
            lines.append(f"  {col}  (IV = {iv:.4f})")
            lines.append("  " + str(df).replace("\n", "\n  "))
            lines.append("")
        return "\n".join(lines).rstrip()

    def _repr_html_(self) -> str:
        if not self.tables:
            return "<div class='mc-binning-table'>BinningTable (empty)</div>"
        sections = []
        for col, df in self.tables.items():
            iv_val = float(self.iv[col]) if col in self.iv.index else 0.0
            sections.append(
                f"<h4>{html.escape(str(col))} "
                f"<span>(IV = {iv_val:.4f})</span></h4>"
                + df.to_html(border=0, classes="mc-binning-table")
            )
        return (
            "<div class='mc-binning-table'>"
            "<strong>BinningTable</strong>"
            + "".join(sections)
            + "</div>"
        )


def binning_table(sol: Any) -> BinningTable:
    """Return the :class:`BinningTable` for a fitted solution.

    Walks ``sol.spec.features``, picks WoE/binned terms, and reads each
    term's :class:`BinningResult` (either off ``.fitted`` or off
    ``sol.fit_state[term.column]['result']``). Non-binning terms are
    skipped.
    """
    from model_crafter.terms.binning import _SMOOTHING, BinningResult
    from model_crafter.terms.woe import BinnedTerm, WoETerm

    spec = getattr(sol, "spec", None)
    if spec is None:
        raise TypeError(
            f"binning_table expected a Solution-like object with a `.spec`; got "
            f"{type(sol).__name__}"
        )

    tables: dict[str, pd.DataFrame] = {}
    iv_per_col: dict[str, float] = {}

    for term in getattr(spec, "features", ()) or ():
        if not isinstance(term, (WoETerm, BinnedTerm)):
            continue
        result = getattr(term, "fitted", None)
        if result is None:
            # Solution may have stashed it under fit_state[term.name].
            fit_state = getattr(sol, "fit_state", None) or {}
            cand = fit_state.get(term.column)
            if isinstance(cand, Mapping):
                inner = cand.get("result")
                result = inner if isinstance(inner, BinningResult) else None
        if result is None:
            continue

        n_events = list(result.n_events)
        n_nonevents = list(result.n_nonevents)
        bin_labels = list(result.bin_labels)
        n_totals = [e + ne for e, ne in zip(n_events, n_nonevents, strict=True)]
        event_rates = [(e / nt) if nt > 0 else 0.0 for e, nt in zip(n_events, n_totals, strict=True)]

        # Per-bin IV contribution: (p_event_b - p_nonevent_b) * WoE_b,
        # using the same Laplace-smoothed proportions the term used.
        e_smooth = [e + _SMOOTHING for e in n_events]
        ne_smooth = [ne + _SMOOTHING for ne in n_nonevents]
        e_tot = sum(e_smooth)
        ne_tot = sum(ne_smooth)
        per_bin_iv = [
            (es / e_tot - nes / ne_tot) * w
            for es, nes, w in zip(e_smooth, ne_smooth, result.woe, strict=True)
        ]

        tables[term.column] = pd.DataFrame(
            {
                "bin": bin_labels,
                "n": n_totals,
                "n_events": n_events,
                "event_rate": event_rates,
                "woe": list(result.woe),
                "iv": per_bin_iv,
            }
        )
        iv_per_col[term.column] = float(result.iv)

    iv_series = pd.Series(iv_per_col, name="iv", dtype=float)
    return BinningTable(tables=tables, iv=iv_series)
