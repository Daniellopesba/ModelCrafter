r"""IRLS plumbing shared by the ridge-IRLS solver and the prox-CD outer loop.

The proximal-Newton solver (:mod:`.prox_cd`) is an IRLS *outer loop*
wrapped around a weighted-elastic-net inner solve — it reuses the
working-response construction and convergence criterion the plain /
ridge IRLS use. Those four helpers live here so the two consumers
agree on conventions.

The working response, working weights, and intercept initialisation are

.. math::

    p &= \sigma(\eta), \\
    z &= \eta + (y - p) / \bigl(p (1 - p)\bigr), \\
    W &= \mathrm{diag}\bigl(w \odot p (1 - p)\bigr), \\
    \beta_0^{(0)} &= \mathrm{logit}(\bar y_w),

with :math:`p` clipped to :math:`[\epsilon, 1 - \epsilon]` to keep the
working response finite as iterates begin to saturate (a separation
symptom). FHT 2010 eq. 18.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from model_crafter._internal.design import INTERCEPT_NAME

__all__ = [
    "_MAX_ITER_DEFAULT",
    "_P_CLIP",
    "_TOL_DEFAULT",
    "_build_working_response",
    "_initialise_eta",
    "_intercept_index",
    "_normalize_weights",
    "_relative_beta_change",
]


_TOL_DEFAULT = 1e-8
_MAX_ITER_DEFAULT = 100
_P_CLIP = 1e-12  # keep p strictly inside (0, 1) so z stays finite.


def _normalize_weights(weights: np.ndarray | None, n: int) -> np.ndarray:
    """Coerce ``weights`` to an ``(n,)`` float vector, ``None`` → ones.

    Same convention as the squared-error solvers (no normalisation to
    sum-to-n — IRLS uses the absolute weights).
    """
    if weights is None:
        return np.ones(n, dtype=float)
    w = np.asarray(weights, dtype=float)
    if w.shape != (n,):
        raise ValueError(f"weights shape {w.shape} != ({n},)")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    if not np.any(w > 0):
        raise ValueError("weights must contain at least one positive value")
    return w


def _intercept_index(columns: tuple[str, ...]) -> int | None:
    try:
        return columns.index(INTERCEPT_NAME)
    except ValueError:
        return None


def _initialise_eta(y: np.ndarray, w: np.ndarray, has_intercept: bool) -> float:
    r"""Intercept-only MLE :math:`\mathrm{logit}(\bar y_w)`.

    Seeding :math:`\eta` here makes IRLS converge in one step for the
    intercept-only case and shaves iterations off richer specs.
    """
    if not has_intercept:
        return 0.0
    p = float(np.sum(w * y) / np.sum(w))
    # Clip so logit stays finite on degenerate y (all 0 / all 1).
    p_clipped = float(min(max(p, _P_CLIP), 1.0 - _P_CLIP))
    return float(np.log(p_clipped / (1.0 - p_clipped)))


def _build_working_response(
    eta: np.ndarray, y: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """One IRLS step's ``(z, working_weights)``.

    Returns ``z = eta + (y - p) / (p (1 - p))`` and ``w * p (1 - p)`` —
    FHT 2010 eq. 18. ``p`` is clipped by :data:`_P_CLIP` so division
    stays finite when iterates saturate.
    """
    p = expit(eta)
    p = np.clip(p, _P_CLIP, 1.0 - _P_CLIP)
    var = p * (1.0 - p)
    z = eta + (y - p) / var
    return z, w * var


def _relative_beta_change(beta_new: np.ndarray, beta_old: np.ndarray) -> float:
    """``||β_new − β_old|| / max(||β_old||, 1)`` — the convergence statistic."""
    denom = max(float(np.linalg.norm(beta_old)), 1.0)
    return float(np.linalg.norm(beta_new - beta_old) / denom)
