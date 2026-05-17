"""Plumbing shared by every basis-expansion term.

The three pieces here are deliberately tiny:

* :class:`_BasisExpandedTerm` — the ``ExpandedTerm`` subclass that adds a
  ``state`` field. The base ``ExpandedTerm`` (``columns``, ``values``) is
  what ``_internal/design.py`` reads to build the design matrix; ``state``
  is what basis terms thread through at predict time and what
  :class:`SupportContainsPredictData` reads to recover training boundary
  knots.
* :func:`_freeze_state` — wraps a dict in ``MappingProxyType`` so the
  state on a frozen ExpandedTerm is also immutable.
* :func:`_x_series` — pull one numeric column out of a DataFrame and
  validate it (DESIGN.md §9.8 — no silent NaN handling).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from model_crafter.terms.base import ExpandedTerm

__all__ = ["_BasisExpandedTerm", "_freeze_state", "_x_series"]


@dataclass(frozen=True, slots=True)
class _BasisExpandedTerm(ExpandedTerm):
    """``ExpandedTerm`` carrying the learned state needed for predict-time replay."""

    state: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))


def _freeze_state(state: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(state))


def _x_series(data: pd.DataFrame, col: str) -> np.ndarray:
    if col not in data.columns:
        raise KeyError(
            f"basis term refers to column '{col}' which is not in the data frame "
            f"(available columns: {list(data.columns)})"
        )
    series = pd.to_numeric(data[col], errors="raise")
    arr = np.asarray(series, dtype=float)
    if not np.isfinite(arr).all():
        bad = int(np.sum(~np.isfinite(arr)))
        raise ValueError(
            f"column '{col}' contains {bad} non-finite value(s); drop or impute "
            "before expansion (DESIGN.md §9.8 — no silent NaN handling)"
        )
    return arr
