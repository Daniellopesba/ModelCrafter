"""``Comparison`` and the ``compare`` orchestration (Task P5.C).

DESIGN.md §3.3 "Model comparison" pins the value type:

    cmp = mc.compare({"baseline": sol_v1, "challenger": sol_v2},
                     data=test, weights="sample_weight")
    print(cmp)
    # Comparison on test (n=42,193)
    #
    #                       baseline    challenger    Δ          p-value
    #   AUC                 0.8123      0.8217        +0.0094    0.003 (DeLong)
    #   KS                  0.4781      0.4892        +0.0111    —
    #   Brier               0.0392      0.0388        −0.0004    —
    #   Log-loss            0.1564      0.1552        −0.0012    —
    #   PSI vs train        0.024       0.031         +0.007     —

The implementation is pure orchestration: for each named solution we
delegate to :func:`model_crafter.performance.performance`, and for each
unordered pair we delegate to
:func:`model_crafter.metrics.classification.delong_test`. The DeLong
p-value table is square and symmetric, with NaN on the diagonal (a
model vs itself is not a meaningful pair).

DeLong with non-uniform weights raises ``NotImplementedError`` (P3.D
limitation — see Pepe 2004 §5.2 for the weighted reformulation). When
that fires we fall back to an all-NaN ``delong_pvalues`` table; the
per-solution ``PerformanceReport`` values remain fully populated, so the
metric table in the ``__repr__`` is still informative.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from model_crafter.metrics.classification import delong_test
from model_crafter.performance.report import PerformanceReport, performance

__all__ = ["Comparison", "compare"]


# ---------------------------------------------------------------------------
# Value type
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Comparison:
    """Side-by-side performance bundle for two or more solutions.

    DESIGN.md §3.3 — the comparison value type. The ``reports`` mapping
    keeps a :class:`PerformanceReport` per named solution; the
    ``delong_pvalues`` DataFrame is square, indexed and labelled by
    solution name, with the diagonal set to ``NaN``.

    See :func:`compare` for the orchestration.
    """

    reports: dict[str, PerformanceReport]
    delong_pvalues: pd.DataFrame

    # ------------------------------------------------------------------
    # repr (DESIGN.md §3.3 "Model comparison")
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # noqa: PLR0915 — repr layout
        names = list(self.reports.keys())
        if not names:
            return "Comparison(<empty>)"
        baseline = names[0]
        base_report = self.reports[baseline]
        n = base_report.n_obs
        n_events = base_report.n_events

        # Collect the metric rows: (label, getter) where getter pulls the
        # raw float (or ``None`` if not available on the sub-report).
        def _auc(r: PerformanceReport) -> float | None:
            return r.discrimination.auc.value

        def _ks(r: PerformanceReport) -> float | None:
            return r.discrimination.ks.value

        def _brier(r: PerformanceReport) -> float | None:
            return r.calibration.brier.value

        def _log_loss(r: PerformanceReport) -> float | None:
            return r.calibration.log_loss.value

        def _psi(r: PerformanceReport) -> float | None:
            return r.stability.psi.value if r.stability is not None else None

        rows: list[tuple[str, Any]] = [
            ("AUC", _auc),
            ("KS", _ks),
            ("Brier", _brier),
            ("Log-loss", _log_loss),
            ("PSI vs reference", _psi),
        ]

        # Column widths.
        label_w = max(len(label) for label, _ in rows) + 2
        name_w = max(max(len(nm) for nm in names), 10)
        diff_w = 9
        pval_w = 16

        delong_unavailable = bool(self.delong_pvalues.isna().to_numpy().all())

        # ---- Header ----
        header_parts = [f"{'':<{label_w}}"]
        for nm in names:
            header_parts.append(f"{nm:>{name_w}}  ")
        header_parts.append(f"{'Δ':>{diff_w}}  ")
        header_parts.append(f"{'p-value':>{pval_w}}")
        header_line = "".join(header_parts).rstrip()

        ev_rate = (n_events / n * 100.0) if n > 0 else 0.0
        lines: list[str] = [
            f"Comparison (n={n:,}, events={n_events:,}, {ev_rate:.1f}%)",
            f"  baseline = {baseline!r}",
            "",
            header_line,
        ]

        # ---- Rows ----
        for label, getter in rows:
            cells = [f"  {label:<{label_w - 2}}"]
            base_v = getter(base_report)
            # Per-solution columns.
            for nm in names:
                v = getter(self.reports[nm])
                if v is None:
                    cells.append(f"{'—':>{name_w}}  ")
                else:
                    cells.append(f"{v:>{name_w}.4f}  ")
            # Δ column (vs baseline). For the baseline itself we leave blank.
            if len(names) >= 2 and base_v is not None:
                # Use the *second* solution as the "challenger" for the Δ
                # column to match the §3.3 two-solution layout. With more
                # than two we still show the second-vs-baseline diff and
                # the full matrix lives in ``delong_pvalues``.
                challenger_v = getter(self.reports[names[1]])
                if challenger_v is not None:
                    diff = challenger_v - base_v
                    cells.append(f"{diff:>+{diff_w}.4f}  ")
                else:
                    cells.append(f"{'—':>{diff_w}}  ")
            else:
                cells.append(f"{'—':>{diff_w}}  ")
            # p-value column — only meaningful for AUC.
            if label == "AUC" and len(names) >= 2 and not delong_unavailable:
                p = self.delong_pvalues.loc[baseline, names[1]]
                if isinstance(p, float) and not np.isnan(p):
                    cells.append(f"{p:>{pval_w - 9}.4f} (DeLong)")
                else:
                    cells.append(f"{'—':>{pval_w}}")
            else:
                cells.append(f"{'—':>{pval_w}}")
            lines.append("".join(cells).rstrip())

        # ---- Footer notes ----
        if delong_unavailable:
            lines.append("")
            lines.append(
                "  DeLong unavailable with non-uniform weights "
                "(see Pepe 2004 §5.2)."
            )
        elif len(names) > 2:
            lines.append("")
            lines.append(
                "  Pairwise DeLong p-values available in .delong_pvalues "
                "(square, symmetric, NaN diagonal)."
            )
        return "\n".join(lines).rstrip()

    def _repr_html_(self) -> str:
        names = list(self.reports.keys())
        if not names:
            return "<div class='mc-comparison'>Comparison(empty)</div>"

        # Build a metric × solution table.
        def _row(label: str, getter) -> str:
            cells = [f"<th>{html.escape(label)}</th>"]
            for nm in names:
                v = getter(self.reports[nm])
                cells.append(
                    f"<td>{v:.4f}</td>"
                    if v is not None
                    else "<td>&mdash;</td>"
                )
            return "<tr>" + "".join(cells) + "</tr>"

        rows = [
            _row("AUC", lambda r: r.discrimination.auc.value),
            _row("KS", lambda r: r.discrimination.ks.value),
            _row("Brier", lambda r: r.calibration.brier.value),
            _row("Log-loss", lambda r: r.calibration.log_loss.value),
            _row(
                "PSI vs reference",
                lambda r: r.stability.psi.value if r.stability is not None else None,
            ),
        ]
        header = (
            "<thead><tr><th>metric</th>"
            + "".join(f"<th>{html.escape(n)}</th>" for n in names)
            + "</tr></thead>"
        )
        return (
            "<div class='mc-comparison'>"
            "<strong>Comparison</strong>"
            "<table class='mc-comparison'>"
            f"{header}<tbody>{''.join(rows)}</tbody>"
            "</table></div>"
        )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _weights_are_uniform(
    weights: str | np.ndarray | pd.Series | None,
    data: pd.DataFrame,
) -> bool:
    """Return True iff ``weights`` would produce a uniform weight vector.

    ``None`` is uniform by definition. A column or array is uniform when
    all entries are equal. We probe upfront (mirroring the check inside
    :func:`delong_test`) so we can decide whether to call DeLong at all.
    """
    if weights is None:
        return True
    if isinstance(weights, str):
        w_arr = np.asarray(data[weights], dtype=float)
    else:
        w_arr = np.asarray(weights, dtype=float)
    if w_arr.size == 0:
        return True
    return bool(np.allclose(w_arr, w_arr[0]))


def compare(
    solutions: dict[str, Any],
    data: pd.DataFrame,
    *,
    weights: str | np.ndarray | pd.Series | None = None,
) -> Comparison:
    """Side-by-side comparison of two or more fitted solutions on ``data``.

    DESIGN.md §3.3 — the model-comparison entry point.

    Parameters
    ----------
    solutions :
        Mapping of name → fitted ``Solution``-like value. Each solution
        must produce predictions for every row of ``data`` and share the
        same target column (verified indirectly by
        :func:`~model_crafter.metrics.classification.delong_test`).
    data :
        A ``pd.DataFrame`` containing the shared target column.
    weights :
        Optional sample weights — a column name in ``data`` or an array.
        Per-solution :class:`PerformanceReport` values respect these
        weights. DeLong's test is unweighted in v0; with non-uniform
        weights the pairwise p-value table is filled with NaN and the
        ``__repr__`` flags the omission.

    Returns
    -------
    Comparison
        ``reports`` carries one :class:`PerformanceReport` per solution;
        ``delong_pvalues`` is a square symmetric DataFrame of pairwise
        DeLong p-values (NaN diagonal).
    """
    if not solutions:
        raise ValueError("compare() requires at least one named solution.")
    names = list(solutions.keys())

    # ---- Per-solution PerformanceReports ----
    reports: dict[str, PerformanceReport] = {
        name: performance(sol, data, weights=weights)
        for name, sol in solutions.items()
    }

    # ---- Pairwise DeLong p-values ----
    pvalue_arr = np.full((len(names), len(names)), np.nan, dtype=float)
    if _weights_are_uniform(weights, data):
        for i, j in combinations(range(len(names)), 2):
            name_a, name_b = names[i], names[j]
            sol_a, sol_b = solutions[name_a], solutions[name_b]
            try:
                res = delong_test(sol_a, sol_b, data, weights=weights)
            except NotImplementedError:
                # Defensive: keep NaN if the primitive ever rejects what
                # we thought was uniform (e.g. an edge-case probe).
                continue
            pvalue_arr[i, j] = res.p_value
            pvalue_arr[j, i] = res.p_value
    # Diagonal stays NaN (a model vs itself is not a meaningful pair).
    name_index = pd.Index(names)
    pvalues = pd.DataFrame(pvalue_arr, index=name_index, columns=name_index)

    return Comparison(reports=reports, delong_pvalues=pvalues)
