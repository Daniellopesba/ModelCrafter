r"""Loss functions.

This module defines the :class:`Loss` protocol — the small surface that
every solver-loss combination implements — and the singletons
``squared_error`` and ``logistic``.

The protocol mirrors the standard three-tuple of optimization: value,
gradient, Hessian. Both losses expose a Hessian that is diagonal in
:math:`\eta` and so return a 1-D array per observation rather than a dense
matrix; the linear-algebra layer multiplies through by the design matrix.

Every concrete loss declares an ``assumptions`` tuple per DESIGN.md §4 — the
list of HARD prerequisites and SOFT stability diagnostics that the assumption
framework (P1.B) runs at solve time. For the squared-error loss the
prerequisite is :class:`~model_crafter.assumptions.FullRankDesign`. We
resolve symbols lazily so this module remains importable in isolation from
P1.B; the actual access happens via ``squared_error.assumptions`` and
``logistic.assumptions``.

The :class:`LogisticLoss` is introduced by Task P3.A (DESIGN.md §8 Phase 3).
Its ``link`` method maps the linear predictor :math:`\eta` to a probability
:math:`p = \sigma(\eta) = 1/(1 + e^{-\eta})` — the package's universal
"output is always a probability" guarantee (DESIGN.md §3.3).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from scipy.special import expit

__all__ = ["LogisticLoss", "Loss", "logistic", "squared_error"]


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


# Logistic loss (Task P3.A)


def _logaddexp_stable(eta: np.ndarray) -> np.ndarray:
    r"""Compute :math:`\log(1 + e^{\eta})` element-wise without overflow.

    Equivalent to ``numpy.logaddexp(0, eta)`` but written explicitly so the
    intent is clear at call sites. For very negative ``eta`` returns 0
    exactly; for very positive ``eta`` returns ``eta`` to numerical precision.
    """
    return np.logaddexp(0.0, eta)


class LogisticLoss:
    r"""Logistic (binary cross-entropy) loss (ESL §4.4).

    For a binary target :math:`y \in \{0, 1\}` and linear predictor
    :math:`\eta = X\beta`, the per-row negative log-likelihood is

    .. math::

        \ell(y, \eta) \;=\; \log\!\bigl(1 + e^{\eta}\bigr)\;-\; y\,\eta .

    Aggregated (with optional sample weights :math:`w_i`),

    .. math::

        \mathcal{L}(y, \eta; w)
        \;=\; \frac{1}{\sum_i w_i}\,
              \sum_i w_i \bigl[\log(1 + e^{\eta_i}) - y_i\,\eta_i\bigr].

    Setting :math:`w_i = 1` recovers the standard mean negative
    log-likelihood. The convention of dividing by :math:`\sum_i w_i`
    mirrors :class:`_SquaredErrorLoss` so the two losses report numerically
    comparable per-observation values.

    Derivatives with respect to :math:`\eta` (used by the IRLS solver):

    .. math::

        \frac{\partial\ell}{\partial\eta_i}
            \;=\; p_i - y_i,
        \qquad
        \frac{\partial^2\ell}{\partial\eta_i^2}
            \;=\; p_i\,(1 - p_i),
        \qquad
        p_i = \sigma(\eta_i) = \frac{1}{1 + e^{-\eta_i}}.

    These are the **per-observation** derivatives of the *aggregated* loss
    divided by :math:`\sum_i w_i`; the solver multiplies by :math:`w_i`
    when it builds the working weights for IRLS (see
    :mod:`model_crafter.solve.irls`).

    The :meth:`link` method maps :math:`\eta \to p = \sigma(\eta)` and is
    the canonical way to convert a fitted linear predictor into the
    probability that :func:`~model_crafter.solve.predict` must return per
    DESIGN.md §3.3 ("output is always a probability"). It is numerically
    stable for ``|eta|`` up to ``1e3`` via :func:`scipy.special.expit`.

    Assumptions (DESIGN.md §4.3, AGENTS.md P3.A)
    --------------------------------------------
    HARD:
        * :class:`~model_crafter.assumptions.FullRankDesign`
        * :class:`~model_crafter.assumptions.BinaryOrProportionTarget`
        * :class:`~model_crafter.assumptions.NoPerfectSeparation`
          (post-fit, ``requires_solution=True``)
    SOFT:
        * :class:`~model_crafter.assumptions.ClassBalance` (``min_minority=0.01``)
        * :class:`~model_crafter.assumptions.CoefficientStability`
        * :class:`~model_crafter.assumptions.PredictiveStability`
    INFO (opt-in via ``classical_inference=True``):
        * :class:`~model_crafter.assumptions.LinkAdequacy`
    """

    @property
    def assumptions(self) -> tuple:
        from model_crafter.assumptions import (
            BinaryOrProportionTarget,
            ClassBalance,
            CoefficientStability,
            FullRankDesign,
            LinkAdequacy,
            NoPerfectSeparation,
            PredictiveStability,
        )

        return (
            FullRankDesign(),
            BinaryOrProportionTarget(),
            NoPerfectSeparation(),
            ClassBalance(min_minority=0.01),
            CoefficientStability(),
            PredictiveStability(),
            LinkAdequacy(),
        )

    # Link (Phase 3 wires this into mc.predict at P3.INTEG)
    def link(self, eta: np.ndarray) -> np.ndarray:
        r"""Inverse-link from linear predictor to probability.

        :math:`p = \sigma(\eta) = 1/(1+e^{-\eta})`. Uses
        :func:`scipy.special.expit` for numerical stability — the result
        is in :math:`[0, 1]` for any finite ``eta``.

        DESIGN.md §3.3 requires that :func:`~model_crafter.solve.predict`
        return probabilities for classification losses. The integration
        step at P3.INTEG wires this method into ``predict()`` (see
        ``notes/P3.A.md``).
        """
        return np.asarray(expit(np.asarray(eta, dtype=float)), dtype=float)

    # Loss / gradient / Hessian (the Loss protocol)
    def value(
        self,
        y: np.ndarray,
        eta: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> float:
        y_arr: np.ndarray = np.asarray(y, dtype=float)
        eta_arr: np.ndarray = np.asarray(eta, dtype=float)
        if y_arr.shape != eta_arr.shape:
            raise ValueError(
                f"y shape {y_arr.shape} != eta shape {eta_arr.shape}"
            )
        w = _normalize_weights(weights, n=y_arr.shape[0])
        per_row = _logaddexp_stable(eta_arr) - y_arr * eta_arr
        return float(np.sum(w * per_row) / np.sum(w))

    def gradient(
        self,
        y: np.ndarray,
        eta: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        y_arr: np.ndarray = np.asarray(y, dtype=float)
        eta_arr: np.ndarray = np.asarray(eta, dtype=float)
        if y_arr.shape != eta_arr.shape:
            raise ValueError(
                f"y shape {y_arr.shape} != eta shape {eta_arr.shape}"
            )
        w = _normalize_weights(weights, n=y_arr.shape[0])
        p = self.link(eta_arr)
        scale = float(np.sum(w))
        return np.asarray((w * (p - y_arr)) / scale, dtype=float)

    def hessian(
        self,
        y: np.ndarray,
        eta: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        y_arr: np.ndarray = np.asarray(y, dtype=float)
        eta_arr: np.ndarray = np.asarray(eta, dtype=float)
        if y_arr.shape != eta_arr.shape:
            raise ValueError(
                f"y shape {y_arr.shape} != eta shape {eta_arr.shape}"
            )
        w = _normalize_weights(weights, n=y_arr.shape[0])
        p = self.link(eta_arr)
        return np.asarray((w * p * (1.0 - p)) / float(np.sum(w)), dtype=float)

    def __repr__(self) -> str:
        return "logistic"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, LogisticLoss)

    def __hash__(self) -> int:
        return hash(("LogisticLoss",))


# Public singleton.
logistic: LogisticLoss = LogisticLoss()
