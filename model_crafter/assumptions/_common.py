"""Internal helpers shared across assumption modules.

Design-matrix materialisation and residual extraction used by both
prerequisite and classical-inference checks.

Spec contract (only the attributes used here):

* ``spec.target`` — name of the target column.
* ``spec.features`` — iterable of strings or objects with a ``.name`` attr.
* ``spec.intercept`` — bool (default True if missing).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "design_column_names",
    "get_residuals",
    "materialise_design",
]


def _feature_columns(spec: Any) -> tuple[str, ...]:
    """Return the data column names referenced by ``spec.features``.

    Strings are taken verbatim; non-strings are assumed to be ``Term``-like
    and have a ``.name`` attribute (the AGENTS.md contract). For now, each
    Term's name is treated as a single design column — basis expansion is
    a Phase 4 concern and will be handled by P1.A's ``_internal/design.py``
    once integrated.
    """
    cols: list[str] = []
    for f in spec.features:
        if isinstance(f, str):
            cols.append(f)
        else:
            cols.append(f.name)
    return tuple(cols)


def design_column_names(spec: Any) -> tuple[str, ...]:
    """Return the design column names, prepending ``(Intercept)`` if the
    spec has ``intercept=True``."""
    feats = _feature_columns(spec)
    if getattr(spec, "intercept", True):
        return ("(Intercept)",) + feats
    return feats


def materialise_design(spec: Any, data: Any) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build the design matrix and the matching column-name tuple from a
    spec + DataFrame.

    Phase 1 only handles "raw column" features. Basis expansion / WoE
    encoding come from later phases and are the responsibility of P1.A's
    ``_internal/design.py``. The assumption layer needs its own minimal
    materialisation so that it can run *before* a solution exists.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"materialise_design requires a pandas DataFrame; got {type(data).__name__}"
        )
    feats = _feature_columns(spec)
    missing = [c for c in feats if c not in data.columns]
    if missing:
        raise KeyError(f"feature columns not found in data: {missing}")
    X = data[list(feats)].to_numpy(dtype=float, copy=False)
    if getattr(spec, "intercept", True):
        X = np.column_stack([np.ones(X.shape[0]), X])
        columns = ("(Intercept)",) + feats
    else:
        columns = feats
    return X, columns


def get_residuals(spec: Any, data: Any, solution: Any) -> np.ndarray:
    """Compute residuals ``y - X @ beta`` from a fitted solution.

    Reads ``solution.coefficients`` (a ``pd.Series`` indexed by design
    column name) and rebuilds ``X`` via :func:`materialise_design`. This
    avoids depending on a particular solver implementation.
    """
    X, columns = materialise_design(spec, data)
    coefs = solution.coefficients
    try:
        beta = np.asarray([coefs[c] for c in columns], dtype=float)
    except KeyError as e:
        raise KeyError(
            f"solution.coefficients is missing design column {e!r}; "
            f"expected one of {columns}"
        ) from e
    y = np.asarray(data[spec.target], dtype=float)
    return y - X @ beta
