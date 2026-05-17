"""HARD prerequisite assumptions (DESIGN.md §4.3).

These are conditions the math literally requires. Failures raise
:class:`~model_crafter.assumptions.AssumptionError` by default.

Currently implemented:

* :class:`FullRankDesign` — the design matrix's column rank equals its
  column count. Declared by every linear loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from model_crafter.assumptions._common import materialise_design
from model_crafter.assumptions._types import CheckResult, Severity


@dataclass(frozen=True, slots=True)
class FullRankDesign:
    """The design matrix has full column rank.

    For a design matrix :math:`X \\in \\mathbb{R}^{n \\times p}` (including the
    intercept column, when present), rank-deficiency means there exist
    columns that are linear combinations of the others — the normal
    equations are singular and OLS is not uniquely defined (ESL §3.2).

    The check identifies offending columns via successive QR
    decomposition: it inspects ``|R_{ii}|`` from the QR factorisation and
    flags any column whose diagonal entry is below a relative tolerance
    of ``max(|R_{ii}|) * 1e-10``.
    """

    name: str = "FullRankDesign"
    severity: Severity = Severity.HARD
    requires_solution: bool = False
    requires_cv: bool = False

    def describe(self) -> str:
        return (
            "Design matrix has full column rank — no exact linear "
            "dependence among predictors (including intercept)."
        )

    def check(
        self,
        spec: Any,
        data: Any,
        *,
        solution: Any | None = None,
        cv: Any | None = None,
    ) -> CheckResult:
        X, columns = materialise_design(spec, data)
        n_cols = X.shape[1]

        # numpy rank is good enough for the pass/fail decision...
        rank = int(np.linalg.matrix_rank(X))
        deficit = n_cols - rank

        if deficit == 0:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                passed=True,
                message=f"design matrix is full rank ({rank}/{n_cols} columns)",
                statistic=0,
                threshold=0,
                suggestion=None,
            )

        # ...but we need to *name* the offending columns. Use a pivoted-QR
        # style sweep: each new column that's already in the span of the
        # previous columns is the dependent one.
        offenders = _identify_dependent_columns(X, columns)

        msg = (
            f"design matrix is rank-deficient (rank={rank}, columns={n_cols}, "
            f"deficit={deficit}); dependent columns: {', '.join(offenders)}"
        )
        return CheckResult(
            name=self.name,
            severity=self.severity,
            passed=False,
            message=msg,
            statistic=deficit,
            threshold=0,
            suggestion=(
                "Drop one of the dependent columns or, if multicollinearity "
                "is the issue, consider penalty=mc.l2(...) (ESL §3.4.1)."
            ),
        )


def _identify_dependent_columns(X: np.ndarray, columns: tuple[str, ...]) -> list[str]:
    """Return the list of columns that are linearly dependent on a smaller
    subset, in the order they appear.

    Strategy: walk columns left-to-right. At each step, check whether the
    new column is in the span of the columns already accepted (rank does
    not increase). If so, it's redundant.
    """
    accepted_idx: list[int] = []
    offenders: list[str] = []
    for j in range(X.shape[1]):
        candidate_idx = accepted_idx + [j]
        rank_before = len(accepted_idx)
        rank_after = int(np.linalg.matrix_rank(X[:, candidate_idx]))
        if rank_after > rank_before:
            accepted_idx.append(j)
        else:
            offenders.append(columns[j])
    return offenders


