"""Paired model comparison.

This module hosts two things:

1. :func:`delong_test`, the public DeLong (1988) paired AUC test —
   conceptually a discrimination metric but co-located with its primary
   consumer because the structural-component variance machinery
   (:mod:`._delong`) is also what powers the AUC CI inside
   ``PerformanceReport``.
2. :func:`compare`, the orchestrator that takes a ``{name: solution}``
   dict and returns a :class:`Comparison` bundling one
   :class:`PerformanceReport` per solution plus a square symmetric
   matrix of pairwise DeLong p-values (NaN diagonal).

DESIGN.md §3.3 pins the layout of ``Comparison.__repr__``.

DeLong's weighted variant (Pepe 2004 §5.2) is out of v0 scope. With
non-uniform weights :func:`delong_test` raises ``NotImplementedError``;
:func:`compare` catches this and emits an all-NaN ``delong_pvalues``
table while keeping every per-solution :class:`PerformanceReport`
fully populated.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from model_crafter.metrics._common import (
    check_binary_target,
    coerce_weights,
    resolve_scores_and_target,
)
from model_crafter.performance._delong import _delong_components
from model_crafter.performance.report import PerformanceReport, performance


@dataclass(frozen=True, slots=True)
class DeLongResult:
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


def delong_test(
    sol_a: Any,
    sol_b: Any,
    data: pd.DataFrame,
    *,
    weights: str | np.ndarray | pd.Series | None = None,
) -> DeLongResult:
    """Paired DeLong (1988) AUC comparison on a single held-out set.

    Both solutions are scored on the same rows of ``data``; the target
    column is read off ``sol_a.spec.target`` and must equal
    ``sol_b.spec.target``. The numerical reference is Sun & Xu (2014);
    ``tests/test_metrics.py`` pins us to ``pROC::roc.test`` at 1e-6.

    Non-uniform weights raise :class:`NotImplementedError` — see the
    module docstring.
    """
    if weights is not None:
        w = coerce_weights(weights, data)
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


@dataclass(frozen=True, slots=True)
class Comparison:
    """Side-by-side performance bundle for two or more solutions.

    ``reports`` keeps one :class:`PerformanceReport` per name; the
    ``delong_pvalues`` DataFrame is square, indexed and labelled by
    solution name, with NaN on the diagonal.
    """

    reports: dict[str, PerformanceReport]
    delong_pvalues: pd.DataFrame

    def __repr__(self) -> str:  # noqa: PLR0915 — repr layout
        names = list(self.reports.keys())
        if not names:
            return "Comparison(<empty>)"
        baseline = names[0]
        base_report = self.reports[baseline]
        n = base_report.n_obs
        n_events = base_report.n_events

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

        label_w = max(len(label) for label, _ in rows) + 2
        name_w = max(max(len(nm) for nm in names), 10)
        diff_w = 9
        pval_w = 16

        delong_unavailable = bool(self.delong_pvalues.isna().to_numpy().all())

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

        for label, getter in rows:
            cells = [f"  {label:<{label_w - 2}}"]
            base_v = getter(base_report)
            for nm in names:
                v = getter(self.reports[nm])
                if v is None:
                    cells.append(f"{'—':>{name_w}}  ")
                else:
                    cells.append(f"{v:>{name_w}.4f}  ")
            # Δ uses solution-2 vs baseline (the §3.3 two-solution layout);
            # with more than two solutions the full matrix lives in
            # ``delong_pvalues``.
            if len(names) >= 2 and base_v is not None:
                challenger_v = getter(self.reports[names[1]])
                if challenger_v is not None:
                    diff = challenger_v - base_v
                    cells.append(f"{diff:>+{diff_w}.4f}  ")
                else:
                    cells.append(f"{'—':>{diff_w}}  ")
            else:
                cells.append(f"{'—':>{diff_w}}  ")
            if label == "AUC" and len(names) >= 2 and not delong_unavailable:
                p = self.delong_pvalues.loc[baseline, names[1]]
                if isinstance(p, float) and not np.isnan(p):
                    cells.append(f"{p:>{pval_w - 9}.4f} (DeLong)")
                else:
                    cells.append(f"{'—':>{pval_w}}")
            else:
                cells.append(f"{'—':>{pval_w}}")
            lines.append("".join(cells).rstrip())

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


def _weights_are_uniform(
    weights: str | np.ndarray | pd.Series | None,
    data: pd.DataFrame,
) -> bool:
    """True iff ``weights`` would collapse to a uniform vector.

    Probed upfront so we can decide whether to call DeLong at all; matches
    the gate inside :func:`delong_test`.
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
    """Bundle per-solution :class:`PerformanceReport` values with a pairwise
    DeLong p-value matrix (DESIGN.md §3.3).

    Each solution must produce predictions for every row of ``data`` and
    share a target column. With non-uniform weights the DeLong table is
    all-NaN — Pepe (2004) §5.2 for the weighted reformulation.
    """
    if not solutions:
        raise ValueError("compare() requires at least one named solution.")
    names = list(solutions.keys())

    reports: dict[str, PerformanceReport] = {
        name: performance(sol, data, weights=weights)
        for name, sol in solutions.items()
    }

    pvalue_arr = np.full((len(names), len(names)), np.nan, dtype=float)
    if _weights_are_uniform(weights, data):
        for i, j in combinations(range(len(names)), 2):
            name_a, name_b = names[i], names[j]
            sol_a, sol_b = solutions[name_a], solutions[name_b]
            try:
                res = delong_test(sol_a, sol_b, data, weights=weights)
            except NotImplementedError:
                continue
            pvalue_arr[i, j] = res.p_value
            pvalue_arr[j, i] = res.p_value
    name_index = pd.Index(names)
    pvalues = pd.DataFrame(pvalue_arr, index=name_index, columns=name_index)

    return Comparison(reports=reports, delong_pvalues=pvalues)
