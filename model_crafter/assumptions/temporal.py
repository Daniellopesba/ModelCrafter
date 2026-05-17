r"""Temporal HARD assumptions (DESIGN.md §3.2, §4.3, AGENTS.md Task P3.B).

Currently implemented:

* :class:`NoTemporalLeakage` — for every fold in a CV partition, the
  training window's last timestamp plus ``gap`` does not exceed the
  validation window's first timestamp.

This check is HARD: a leaky CV partition reports performance the model
will not actually achieve in production. The check fires inside
:func:`model_crafter.validation.cross_validate.cross_validate` *and*
runs in standalone form when callers pass a ``cv=...`` carrying the
partitioned folds (DESIGN.md §3.2: the assumption framework is the
audit-grade record).

Math
----
For folds :math:`(T_k, V_k)_{k=1}^K`, the check verifies

.. math::

    \max_{i \in T_k} t_i + g \;\leq\; \min_{j \in V_k} t_j
    \qquad \text{for every } k = 1, \dots, K.

Violations name the offending fold index and the two offending
timestamps so the operator can debug.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from model_crafter.assumptions._types import CheckResult, Severity

__all__ = ["NoTemporalLeakage"]


@dataclass(frozen=True, slots=True)
class NoTemporalLeakage:
    """HARD assumption: every fold's train window precedes its validation
    window by at least ``gap``.

    The assumption operates on a ``cv`` object carrying a sequence of
    ``(train_df, valid_df)`` pairs and (optionally) the splitter's
    ``time_col``/``gap`` metadata. The expected ``cv`` shape is one of:

    * An object with ``.folds`` — a sequence of ``(train_df, valid_df)``
      pairs (this is what :func:`cross_validate` constructs internally
      before solving).
    * A :class:`~model_crafter.validation.cross_validate.CVResult` with
      a ``.folds`` accessor synthesised from ``fold_results``.

    The check reads ``time_col`` and ``gap`` from the ``cv.splitter``
    attribute when available; otherwise it falls back to comparing
    ``time_col`` against the spec's documented ``time_col`` (set on the
    splitter, not on the spec).

    Severity is HARD because a leaky CV partition reports performance
    the model will not actually achieve in production — the assumption
    is a prerequisite for the math of held-out evaluation.
    """

    name: str = "NoTemporalLeakage"
    severity: Severity = Severity.HARD
    requires_solution: bool = False
    requires_cv: bool = True

    def describe(self) -> str:
        return (
            "For every fold, max(train[time_col]) + gap <= "
            "min(valid[time_col]) (DESIGN.md §3.2)."
        )

    def check(
        self,
        spec: Any,
        data: Any,
        *,
        solution: Any | None = None,
        cv: Any | None = None,
    ) -> CheckResult:
        if cv is None:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                passed=True,
                message="skipped: no CV available (NoTemporalLeakage is a CV-aware check)",
                statistic=None,
                threshold=None,
                suggestion=None,
            )

        time_col, gap = _read_metadata(cv)
        if time_col is None:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                passed=False,
                message=(
                    "no time_col is available on the CV/splitter; "
                    "NoTemporalLeakage cannot run"
                ),
                statistic=None,
                threshold=None,
                suggestion=(
                    "Use a temporal splitter (expanding_window / rolling_window / "
                    "purged_kfold) whose .time_col attribute identifies the "
                    "timestamp column (DESIGN.md §3.2)."
                ),
            )

        folds = _extract_folds(cv)
        if folds is None:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                passed=False,
                message=(
                    "cv object exposes no .folds sequence; NoTemporalLeakage "
                    "cannot iterate over fold pairs"
                ),
                statistic=None,
                threshold=None,
                suggestion=None,
            )

        for k, (train, valid) in enumerate(folds):
            train_end = pd.to_datetime(train[time_col]).max()
            valid_start = pd.to_datetime(valid[time_col]).min()
            if pd.isna(train_end) or pd.isna(valid_start):
                continue
            if valid_start < train_end + gap:
                return CheckResult(
                    name=self.name,
                    severity=self.severity,
                    passed=False,
                    message=(
                        f"fold {k}: train_end={train_end!s} + gap={gap!s} "
                        f"> valid_start={valid_start!s} — leakage"
                    ),
                    statistic=float((train_end + gap - valid_start).total_seconds()),
                    threshold=0.0,
                    suggestion=(
                        "Increase the splitter's gap or use a temporal splitter "
                        "that purges around validation windows (purged_kfold). "
                        "DESIGN.md §3.2."
                    ),
                )

        return CheckResult(
            name=self.name,
            severity=self.severity,
            passed=True,
            message=(
                f"every fold respects train_end + gap ({gap!s}) <= valid_start"
            ),
            statistic=0.0,
            threshold=0.0,
            suggestion=None,
        )


# Helpers — duck-typed access to the cv/splitter contract.


_ZERO_TD: pd.Timedelta = pd.Timedelta(0)  # type: ignore[assignment]


def _read_metadata(cv: Any) -> tuple[str | None, pd.Timedelta]:
    """Return ``(time_col, gap)`` from a cv-like object.

    The function looks (in order) at: ``cv.splitter`` (the most common
    shape — :class:`CVResult` carries the splitter), then at ``cv``
    itself (a raw :class:`Splitter` is also acceptable for direct
    callers).
    """
    sp = getattr(cv, "splitter", cv)
    time_col = getattr(sp, "time_col", None)
    gap_val = getattr(sp, "gap", _ZERO_TD)
    if isinstance(gap_val, pd.Timedelta):
        gap: pd.Timedelta = gap_val
    else:
        parsed = pd.Timedelta(gap_val)
        gap = parsed if isinstance(parsed, pd.Timedelta) else _ZERO_TD
    return time_col, gap


def _extract_folds(cv: Any):
    """Return an iterable of ``(train_df, valid_df)`` pairs, or ``None``.

    Accepted shapes (most specific first):

    * ``cv.folds`` — sequence/iterable of pairs.
    * ``cv.fold_results`` — sequence of dicts with ``train`` / ``valid``
      keys (the :class:`CVResult` shape).
    """
    folds = getattr(cv, "folds", None)
    if folds is not None:
        return folds
    fold_results = getattr(cv, "fold_results", None)
    if fold_results is not None:
        pairs = []
        for fr in fold_results:
            train = fr.get("train") if isinstance(fr, dict) else getattr(fr, "train", None)
            valid = fr.get("valid") if isinstance(fr, dict) else getattr(fr, "valid", None)
            if train is None or valid is None:
                return None
            pairs.append((train, valid))
        return pairs
    return None
