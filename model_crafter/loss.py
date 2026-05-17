r"""Loss functions.

This module defines the :class:`Loss` protocol — the small surface that
every solver-loss combination implements — and the singleton
``squared_error`` for ordinary least squares.

The protocol mirrors the standard three-tuple of optimization: value,
gradient, Hessian. The Hessian for the squared-error loss is a constant
diagonal of weights (uniform 1.0 when ``weights=None``), so we expose it as a
1-D array rather than a full matrix. Phase 3 will introduce
:class:`LogisticLoss`, whose Hessian depends on :math:`\eta`.

Every concrete loss declares an ``assumptions`` tuple per DESIGN.md §4 — the
list of HARD prerequisites and SOFT stability diagnostics that the assumption
framework (P1.B) runs at solve time. For the squared-error loss the
prerequisite is :class:`~model_crafter.assumptions.FullRankDesign`. We
resolve that symbol lazily so that this module remains importable in
isolation from P1.B; the actual access happens via ``squared_error.assumptions``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = ["Loss", "squared_error"]


@runtime_checkable
class Loss(Protocol):
    """Minimal loss surface.

    The protocol intentionally omits a fitted-state hook; losses are
    stateless values.

    Notes
    -----
    Every method accepts ``weights`` and treats ``None`` as uniform. Per
    DESIGN.md §9.6 (Sample weights everywhere or nowhere), this is mandatory.
    ``assumptions`` is exposed as a read-only property so concrete losses
    can either declare a class-level tuple or compute it lazily.
    """

    @property
    def assumptions(self) -> tuple: ...

    def value(
        self,
        y: np.ndarray,
        eta: np.ndarray,
        weights: np.ndarray | None,
    ) -> float: ...

    def gradient(
        self,
        y: np.ndarray,
        eta: np.ndarray,
        weights: np.ndarray | None,
    ) -> np.ndarray: ...

    def hessian(
        self,
        y: np.ndarray,
        eta: np.ndarray,
        weights: np.ndarray | None,
    ) -> np.ndarray: ...


def _normalize_weights(weights: np.ndarray | None, n: int) -> np.ndarray:
    """Validate ``weights`` and broadcast ``None`` to a vector of ones.

    Per DESIGN.md §9.8, no silent broadcasting: ``weights`` (when given) must
    have length ``n``.
    """
    if weights is None:
        return np.ones(n, dtype=float)
    w = np.asarray(weights, dtype=float)
    if w.shape != (n,):
        raise ValueError(
            f"weights must be 1-D of length {n}; got shape {w.shape}"
        )
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    if not np.any(w > 0):
        raise ValueError("weights must contain at least one positive value")
    return w


class _SquaredErrorLoss:
    r"""The squared-error loss for ordinary / weighted least squares.

    .. math::

        \mathcal{L}(y, \eta; w) \;=\; \frac{1}{2 \sum_i w_i} \sum_i w_i (y_i - \eta_i)^2

    With ``weights=None`` this is the standard mean-squared-error objective
    ESL §3.2 minimises for ordinary least squares; with weights it is the
    weighted least squares objective of ESL §3.2.3.

    The Hessian with respect to the linear predictor :math:`\eta` is the
    diagonal matrix of weights divided by ``sum(w)``; we return the diagonal
    as a 1-D array so the linear-algebra layer can avoid building a dense
    diagonal.
    """

    @property
    def assumptions(self) -> tuple:
        from model_crafter.assumptions import (
            FullRankDesign,
            Homoscedasticity,
            Independence,
            LowVIF,
            ResidualNormality,
        )

        return (
            FullRankDesign(),
            ResidualNormality(),
            Homoscedasticity(),
            Independence(),
            LowVIF(),
        )

    def value(
        self,
        y: np.ndarray,
        eta: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> float:
        y_arr: np.ndarray = np.asarray(y, dtype=float)
        eta_arr: np.ndarray = np.asarray(eta, dtype=float)
        if y_arr.shape != eta_arr.shape:
            raise ValueError(f"y shape {y_arr.shape} != eta shape {eta_arr.shape}")
        w = _normalize_weights(weights, n=y_arr.shape[0])
        resid = y_arr - eta_arr
        return float(0.5 * np.sum(w * resid * resid) / np.sum(w))

    def gradient(
        self,
        y: np.ndarray,
        eta: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        y_arr: np.ndarray = np.asarray(y, dtype=float)
        eta_arr: np.ndarray = np.asarray(eta, dtype=float)
        if y_arr.shape != eta_arr.shape:
            raise ValueError(f"y shape {y_arr.shape} != eta shape {eta_arr.shape}")
        w = _normalize_weights(weights, n=y_arr.shape[0])
        # d/d eta of (1/(2 sum w)) * sum w (y - eta)^2 = -(w * (y - eta)) / sum(w)
        resid: np.ndarray = np.subtract(y_arr, eta_arr)
        scale = float(np.sum(w))
        out: np.ndarray = np.multiply(w, resid) / scale
        return np.negative(out)

    def hessian(
        self,
        y: np.ndarray,
        eta: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        y_arr: np.ndarray = np.asarray(y, dtype=float)
        eta_arr: np.ndarray = np.asarray(eta, dtype=float)
        if y_arr.shape != eta_arr.shape:
            raise ValueError(f"y shape {y_arr.shape} != eta shape {eta_arr.shape}")
        w = _normalize_weights(weights, n=y_arr.shape[0])
        # d^2/d eta^2 = w / sum(w) (diagonal)
        return w / float(np.sum(w))

    def __repr__(self) -> str:
        return "squared_error"

    # Make the singleton hashable and identifiable in the dispatch table.
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, _SquaredErrorLoss)

    def __hash__(self) -> int:
        return hash(("_SquaredErrorLoss",))


# Public singleton. Use the type for dispatch; the value for the API.
squared_error: Loss = _SquaredErrorLoss()
