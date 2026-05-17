"""Spec-kind predicates and the bootstrap-pointer message.

Coefficient tables and closed-form diagnostics both need to know what
flavour of Solution they're looking at: OLS, closed-form ridge, or
something else (lasso / elastic-net / logistic). The predicates are
trivial but they live in one place so the answer is consistent across
the inspect package.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _solution_is_ols(sol: Any) -> bool:
    """True iff the spec is squared-error + NoPenalty (closed-form OLS)."""
    from model_crafter.loss import _SquaredErrorLoss
    from model_crafter.penalty import NoPenalty

    spec = getattr(sol, "spec", None)
    if spec is None:
        return False
    return isinstance(spec.loss, _SquaredErrorLoss) and isinstance(
        spec.penalty, NoPenalty
    )


def _solution_is_ridge(sol: Any) -> bool:
    """True iff the spec is squared-error + L2 only (closed-form ridge)."""
    try:
        from model_crafter.loss import _SquaredErrorLoss
        from model_crafter.penalty import L2Penalty
    except ImportError:  # pragma: no cover
        return False

    spec = getattr(sol, "spec", None)
    if spec is None:
        return False
    return isinstance(spec.loss, _SquaredErrorLoss) and isinstance(
        spec.penalty, L2Penalty
    )


def _solution_supports_closed_form_hat(sol: Any) -> bool:
    return _solution_is_ols(sol) or _solution_is_ridge(sol)


def _solution_is_logistic(sol: Any) -> bool:
    """True iff the spec uses :class:`LogisticLoss`."""
    from model_crafter.loss import LogisticLoss

    spec = getattr(sol, "spec", None)
    if spec is None:
        return False
    return isinstance(spec.loss, LogisticLoss)


def _coef_se_from_solution(sol: Any) -> pd.Series | None:
    se = getattr(sol, "coefficient_se", None)
    if se is None:
        return None
    return se


def _bootstrap_pointer(what: str) -> str:
    return (
        f"{what} is only defined for closed-form linear models "
        "(OLS or closed-form ridge). For lasso, elastic net, or logistic "
        "regression use `mc.bootstrap(sol, data)` (ESL §7.11) — the "
        "bootstrap is the recommended uncertainty / influence diagnostic "
        "for non-closed-form fits."
    )
