r"""Solver dispatch entry point.

Public surface:

* :func:`solve` — apply a :class:`~model_crafter.spec.LinearSpec` to data
  and return a :class:`~model_crafter.solution.Solution`.
* :func:`predict` — apply a fitted ``Solution`` to new data and return a
  prediction Series aligned with ``new_data.index``.

The two functions implement DESIGN.md §2.1's "three verbs": ``spec``
(constructed elsewhere), ``solve``, ``predict``.

Dispatch happens on ``(type(spec.loss), type(spec.penalty))`` via the
registry in :mod:`model_crafter.solve._registry`. Solver modules
self-register on import; importing this package imports the OLS solver so
the registry is populated before the first ``solve`` call.

Per DESIGN.md §4.2, ``solve`` runs the assumption framework around the
numerical solve: HARD prerequisite checks happen *before* solving (so a
rank-deficient design raises before any compute), and post-fit checks run
*after*. Both rely on P1.B's framework, which P1.A treats as opaque.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from model_crafter._internal.design import INTERCEPT_NAME, build_design
from model_crafter._internal.linalg import find_rank_deficient_columns
from model_crafter.solution import Solution
from model_crafter.solve import ols as _ols  # noqa: F401 — self-registering import
from model_crafter.solve._registry import (
    SolverInputs,
    SolverOutputs,
    get_solver,
)
from model_crafter.spec import LinearSpec

__all__ = ["predict", "solve"]


def _coerce_weights(
    weights: str | np.ndarray | pd.Series | None,
    data: pd.DataFrame,
) -> np.ndarray | None:
    """Resolve ``weights=`` to a 1-D numpy array (or ``None`` for uniform).

    Strings are looked up in ``data``; arrays/Series are validated for shape.
    A weights column of all zeros, NaNs, or negatives is a HARD error
    (DESIGN.md §9.8 — no silent NaN handling).
    """
    if weights is None:
        return None
    if isinstance(weights, str):
        if weights not in data.columns:
            raise KeyError(
                f"weights column '{weights}' not in data (columns: {list(data.columns)})"
            )
        w = data[weights].to_numpy(dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.shape[0] != len(data):
        raise ValueError(
            f"weights must be a 1-D array of length {len(data)}; got shape {w.shape}"
        )
    if not np.isfinite(w).all():
        raise ValueError("weights contain non-finite values (NaN / Inf)")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    if not np.any(w > 0):
        raise ValueError("weights must contain at least one positive value")
    return w


def _extract_y(spec: LinearSpec, data: pd.DataFrame) -> np.ndarray:
    if spec.target not in data.columns:
        raise KeyError(
            f"target column '{spec.target}' not in data (columns: {list(data.columns)})"
        )
    series = pd.to_numeric(data[spec.target], errors="raise")
    y: np.ndarray = np.asarray(series, dtype=float)
    if not np.isfinite(y).all():
        bad = int(np.sum(~np.isfinite(y)))
        raise ValueError(
            f"target column '{spec.target}' contains {bad} non-finite value(s); "
            "drop or impute before solving (DESIGN.md §9.8 — no silent NaN handling)"
        )
    return y


def solve(
    spec: LinearSpec,
    data: pd.DataFrame,
    *,
    weights: str | np.ndarray | pd.Series | None = None,
    method: str | None = None,
    on_violation: str = "raise",
    suppress: tuple = (),
    classical_inference: bool = False,
    stability_splitter: Any = None,
) -> Solution:
    """Fit ``spec`` to ``data`` and return a :class:`Solution`.

    See DESIGN.md §2.4 for the dispatch rules, §4.2 for the assumption flow.

    Parameters
    ----------
    spec:
        A :class:`LinearSpec` constructed via :func:`~model_crafter.spec.linear`.
    data:
        The training data frame.
    weights:
        Sample weights — a column name (str) in ``data``, a 1-D array, or
        ``None`` (uniform). DESIGN.md §9.6 requires weights to be supported
        everywhere.
    method:
        Solver override; ``None`` lets the dispatch pick.
    on_violation:
        ``"raise"`` (default) raises :class:`AssumptionError` on HARD
        violations; ``"warn"`` warns and continues; ``"ignore"`` records the
        result silently.
    suppress:
        A tuple of assumption *types* to skip.
    classical_inference:
        When ``True``, the INFO-level classical tests (Shapiro-Wilk,
        Breusch-Pagan, VIF, etc.) are run and included in
        ``sol.assumptions``. Default ``False``.
    stability_splitter:
        Splitter for the SOFT stability checks (not used in Phase 1; reserved
        for Phase 3+).
    """
    from model_crafter.assumptions import (  # pyright: ignore[reportMissingImports]
        AssumptionError,
        run_assumptions,
    )

    if not isinstance(spec, LinearSpec):
        raise TypeError(f"spec must be a LinearSpec; got {type(spec).__name__}")
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be a pandas DataFrame; got {type(data).__name__}")

    # Materialise the design matrix and target.
    design = build_design(spec, data)
    y = _extract_y(spec, data)
    w = _coerce_weights(weights, data)

    # ---- HARD prerequisite: full-rank design ----
    offending = find_rank_deficient_columns(design.values, design.columns)
    # Map column names back to term names for a friendlier message.
    offending_terms = tuple(
        dict.fromkeys(
            design.term_for_column[design.columns.index(c)] for c in offending
        )
    )

    if offending:
        # Run the assumption framework with the rank result so the message,
        # severity, and report shape come from P1.B (the opaque interface).
        try:
            run_assumptions(
                spec,
                data,
                on_violation=on_violation,
                suppress=suppress,
                classical_inference=classical_inference,
                design=design,
                offending_columns=offending,
            )
        except TypeError:
            # The real P1.B framework may not accept design= / offending_columns=
            # kwargs at the call boundary; fall back to building the report
            # ourselves and raising directly.
            cols_display = list(offending)
            terms_display = list(offending_terms)
            raise AssumptionError(
                "FullRankDesign violated: design matrix is rank-deficient. "
                f"Offending columns: {cols_display}. "
                f"Originating terms: {terms_display}. "
                "Drop one of the collinear features or add a penalty "
                "(ESL §3.4.1: ridge / lasso is the principled response)."
            ) from None
        else:
            # If the framework's HARD check fired and returned (e.g. on_violation
            # != "raise"), we still cannot continue with a singular X. Construct
            # the error message ourselves so callers using on_violation="warn"
            # do not get a numerical NaN explosion downstream.
            cols_display = list(offending)
            terms_display = list(offending_terms)
            raise AssumptionError(
                "FullRankDesign violated: design matrix is rank-deficient. "
                f"Offending columns: {cols_display}. "
                f"Originating terms: {terms_display}. "
                "Drop one of the collinear features or add a penalty "
                "(ESL §3.4.1: ridge / lasso is the principled response)."
            )

    # ---- Numerical solve ----
    solver_fn = get_solver(spec.loss, spec.penalty)
    inputs = SolverInputs(
        spec=spec,
        design=design,
        y=y,
        weights=w,
        data_index=data.index,
        method=method,
    )
    out: SolverOutputs = solver_fn(inputs)

    # ---- Post-fit assumption pass (HARD + SOFT + optional INFO) ----
    report = run_assumptions(
        spec,
        data,
        solution=None,  # post-fit checks not in scope for Phase 1
        on_violation=on_violation,
        suppress=suppress,
        classical_inference=classical_inference,
    )

    return Solution(
        spec=spec,
        coefficients=out.coefficients,
        coefficient_se=out.coefficient_se,
        fit_state=out.fit_state,
        design_columns=tuple(out.coefficients.index),
        loss_value=out.loss_value,
        penalty_value=out.penalty_value,
        n_obs=out.n_obs,
        converged=out.converged,
        solver_info=out.solver_info,
        assumptions=report,
    )


def predict(sol: Solution, new_data: pd.DataFrame) -> pd.Series:
    """Apply a fitted ``sol`` to ``new_data`` and return :math:`\\hat y`.

    Returns a :class:`pandas.Series` aligned with ``new_data.index``. For
    Phase 1's squared-error loss the output is :math:`X\\beta`; future
    losses (logistic) will return probabilities. DESIGN.md §3.3 makes that
    the package's universal output convention.
    """
    if not isinstance(sol, Solution):
        raise TypeError(f"sol must be a Solution; got {type(sol).__name__}")
    if not isinstance(new_data, pd.DataFrame):
        raise TypeError(
            f"new_data must be a pandas DataFrame; got {type(new_data).__name__}"
        )

    # Build the design matrix with the same intercept + term shape.
    design = build_design(sol.spec, new_data, fit_state=sol.fit_state)
    # Sanity: columns must match the training design columns exactly.
    if design.columns != sol.design_columns:
        raise ValueError(
            "predict-time design columns do not match the training columns:\n"
            f"  training: {list(sol.design_columns)}\n"
            f"  predict:  {list(design.columns)}"
        )

    beta = sol.coefficients.reindex(list(design.columns)).to_numpy(dtype=float)
    yhat = design.values @ beta
    return pd.Series(yhat, index=new_data.index, name=sol.spec.target)


# Re-export the intercept name so other modules can reference it without
# digging into _internal.
_ = INTERCEPT_NAME
