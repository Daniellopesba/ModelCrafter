"""Design matrix construction from a tuple of Term values.

Given a :class:`~model_crafter.spec.LinearSpec` and a data frame, this module
produces the :math:`n \\times p` numerical design matrix :math:`X` and the
matching column-name vector. It also captures any per-term fit state learned
during expansion (Phase 4: WoE bin edges, basis knots, etc.; Phase 1's
:class:`RawTerm` is stateless).

The intercept is *not* declared as a term — it is a structural part of the
spec (``intercept=True`` by default). When present, it is the first column
of :math:`X` and is conventionally named ``"(Intercept)"``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from model_crafter.spec import LinearSpec

INTERCEPT_NAME = "(Intercept)"

@dataclass(frozen=True, slots=True)
class DesignMatrix:
    """A numerical design matrix bundled with its column metadata.

    Attributes
    ----------
    values:
        The ``(n_obs, n_cols)`` float64 array.
    columns:
        One name per column. Column 0 is ``"(Intercept)"`` when
        ``spec.intercept`` is true.
    term_for_column:
        For each column index, the name of the term it originated from.
        Lets the rank-deficiency checker name the offending *terms* rather
        than just the columns.
    fit_state:
        Per-term learned state captured during expansion (empty for
        :class:`RawTerm`-only specs in Phase 1).
    """

    values: np.ndarray
    columns: tuple[str, ...]
    term_for_column: tuple[str, ...]
    fit_state: Mapping[str, Any]

    @property
    def n_obs(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_cols(self) -> int:
        return int(self.values.shape[1])


def build_design(
    spec: LinearSpec,
    data: pd.DataFrame,
    *,
    fit_state: Mapping[str, Mapping[str, Any]] | None = None,
) -> DesignMatrix:
    """Materialize :math:`X` from ``spec`` and ``data``.

    Parameters
    ----------
    spec:
        The :class:`LinearSpec`. Only ``features``, ``intercept`` are read.
    data:
        Source data frame. The target column is *not* read by this function.
    fit_state:
        Optional mapping of per-term fit state. When ``None``, terms expand
        without any learned state (correct for Phase 1's
        :class:`~model_crafter.terms.base.RawTerm`).

    Returns
    -------
    DesignMatrix
        A bundle of the values, column names, and per-column originating
        term names.

    Raises
    ------
    ValueError
        If two terms produce the same expanded column name, or any
        expansion returns zero columns.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be a pandas DataFrame; got {type(data).__name__}")
    n = len(data)
    blocks: list[np.ndarray] = []
    names: list[str] = []
    term_for_col: list[str] = []
    next_fit_state: dict[str, Any] = {}

    if spec.intercept:
        blocks.append(np.ones((n, 1), dtype=float))
        names.append(INTERCEPT_NAME)
        term_for_col.append(INTERCEPT_NAME)

    for term in spec.features:
        per_term_state = (fit_state or {}).get(term.name)
        exp = term.expand(data, fit_state=per_term_state)
        if exp.values.shape[0] != n:
            raise ValueError(
                f"term '{term.name}' produced {exp.values.shape[0]} rows but data has {n}"
            )
        if exp.values.shape[1] == 0:
            raise ValueError(f"term '{term.name}' produced no columns")
        blocks.append(exp.values.astype(float, copy=False))
        names.extend(exp.columns)
        term_for_col.extend([term.name] * exp.values.shape[1])
        # Capture any state the term wants to round-trip to predict time.
        # RawTerm has none; later term types can add via ExpandedTerm extension.

    # Duplicate-name guard. Better caught here than as a silent collision
    # later in the coefficient table.
    seen: dict[str, int] = {}
    for c in names:
        seen[c] = seen.get(c, 0) + 1
    dups = sorted(c for c, k in seen.items() if k > 1)
    if dups:
        raise ValueError(
            f"design matrix has duplicate column names: {dups}. Two terms expanded "
            "to the same column name."
        )

    X = np.hstack(blocks) if blocks else np.zeros((n, 0))
    if not np.isfinite(X).all():
        # Per DESIGN.md §9.8: no silent NaN/inf handling.
        # Identify offending columns by name.
        bad_cols = [names[j] for j in range(X.shape[1]) if not np.isfinite(X[:, j]).all()]
        raise ValueError(
            f"design matrix contains non-finite values in columns: {bad_cols}"
        )
    return DesignMatrix(
        values=X,
        columns=tuple(names),
        term_for_column=tuple(term_for_col),
        fit_state=next_fit_state,
    )
