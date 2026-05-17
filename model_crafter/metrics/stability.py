"""Population Stability Index (PSI).

Unlike the discrimination + calibration metrics, ``psi`` takes two raw
distributions (reference vs current) rather than a fitted ``Solution``.
This matches DESIGN.md §3.3's call signature: ``mc.psi(reference, current,
bins=10)``.

PSI is defined (Karakoulas 2004, Siddiqi 2006) as

.. math::

    \\text{PSI} = \\sum_b (p^{cur}_b - p^{ref}_b)
                  \\cdot \\log \\frac{p^{cur}_b}{p^{ref}_b}

over equal-frequency or fixed-edge bins. Both bin masses are clipped to a
small positive ``eps`` to keep the log finite when a bin is empty in one
sample. The same bin edges are applied to both samples — the bins are
determined from the reference, then applied to the current.

Conventional interpretation (Siddiqi 2006):

* PSI < 0.10 — insignificant change
* 0.10 ≤ PSI < 0.25 — moderate shift, investigate
* PSI ≥ 0.25 — significant shift, retrain
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from model_crafter.metrics._common import coerce_weights

__all__ = ["PSIResult", "psi"]


_PSI_EPS = 1e-6
"""Floor for bin mass before entering the log (prevents log(0))."""


@dataclass(frozen=True, slots=True)
class PSIResult:
    """Population Stability Index over a binning of two distributions.

    Fields
    ------
    value : float
        The PSI scalar (sum of bin contributions).
    bins : int
        Number of bins used.
    edges : np.ndarray
        The bin edges applied to both samples (``len = bins + 1``).
    contributions : np.ndarray
        Per-bin contribution ``(p_cur - p_ref) * log(p_cur / p_ref)``.
    ref_pct : np.ndarray
        Reference bin mass fractions.
    cur_pct : np.ndarray
        Current bin mass fractions.
    """

    value: float
    bins: int
    edges: np.ndarray = field(repr=False)
    contributions: np.ndarray = field(repr=False)
    ref_pct: np.ndarray = field(repr=False)
    cur_pct: np.ndarray = field(repr=False)

    def __float__(self) -> float:
        return float(self.value)

    def __repr__(self) -> str:
        # Interpretation per Siddiqi (2006).
        if self.value < 0.10:
            tag = "low"
        elif self.value < 0.25:
            tag = "moderate"
        else:
            tag = "high — retrain"
        return f"PSI = {self.value:.4f}  ({tag}, {self.bins} bins)"


def _to_array(x: Any, name: str) -> np.ndarray:
    """Coerce a Series, DataFrame column, or array-like into a 1-D float array."""
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(
                f"{name}: a DataFrame must have exactly one column; got "
                f"shape {x.shape}. Pass a Series or column."
            )
        x = x.iloc[:, 0]
    if isinstance(x, pd.Series):
        x = x.to_numpy(dtype=float)
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(arr).all():
        raise ValueError(
            f"{name} contains non-finite values; PSI requires finite scores"
        )
    return arr


def _edges_from_reference(
    reference: np.ndarray,
    bins: int | Sequence[float] | np.ndarray,
    ref_weights: np.ndarray | None,
) -> np.ndarray:
    """Build bin edges from ``reference`` (equal-frequency) or accept explicit
    edges via ``bins``.
    """
    if isinstance(bins, int):
        if bins < 2:
            raise ValueError(f"bins must be >= 2; got {bins}")
        # Equal-frequency edges on reference. Use ``np.quantile`` when
        # weights are uniform (or absent) so the "uniform = unweighted"
        # contract holds bitwise; fall back to ``weighted_quantile`` only
        # when weights are genuinely non-uniform.
        qs = np.linspace(0.0, 1.0, bins + 1)
        if ref_weights is None or np.allclose(ref_weights, ref_weights[0]):
            edges = np.quantile(reference, qs)
        else:
            from model_crafter.metrics._common import weighted_quantile

            edges = weighted_quantile(reference, qs, ref_weights)
        # Outer edges set to -inf/+inf so out-of-range current points are
        # captured cleanly.
        edges = np.asarray(edges, dtype=float)
        edges[0] = -np.inf
        edges[-1] = np.inf
        # Deduplicate the interior edges (ties may collapse them); guarantee
        # strictly increasing interior to keep np.digitize sane.
        interior = edges[1:-1]
        if interior.size:
            # Force strictly increasing by adding tiny jitter when needed.
            for k in range(1, interior.size):
                if interior[k] <= interior[k - 1]:
                    interior[k] = np.nextafter(interior[k - 1], np.inf)
            edges = np.concatenate(([-np.inf], interior, [np.inf]))
        return edges
    arr = np.asarray(bins, dtype=float).ravel()
    if arr.size < 3:
        raise ValueError(
            f"explicit edges must have >= 3 entries (>= 2 bins); got {arr.size}"
        )
    if not np.all(np.diff(arr) > 0):
        raise ValueError("explicit edges must be strictly increasing")
    return arr


def _bin_mass(
    x: np.ndarray,
    edges: np.ndarray,
    weights: np.ndarray | None,
) -> np.ndarray:
    """Return the (weighted) mass fraction of ``x`` in each bin defined by
    ``edges``.

    Bin assignment uses ``digitize(x, edges, right=False)`` then offsets so
    bins are ``0..n_bins-1``.
    """
    # digitize with edges = [-inf, e1, e2, ..., +inf] gives indices in
    # [1, n_bins+1]; subtract 1 so bins are [0, n_bins-1].
    n_bins = edges.size - 1
    idx = np.digitize(x, edges[1:-1], right=False)  # gives [0, n_bins-1]
    if weights is None:
        counts = np.bincount(idx, minlength=n_bins).astype(float)
        total = float(x.size)
    else:
        counts = np.bincount(idx, weights=weights, minlength=n_bins).astype(float)
        total = float(np.sum(weights))
    if total == 0.0:
        raise ValueError("bin mass: total weight is zero")
    return counts / total


def _psi_from_pcts(
    ref_pct: np.ndarray, cur_pct: np.ndarray, eps: float = _PSI_EPS
) -> tuple[float, np.ndarray]:
    """Return ``(psi, contributions)``.

    Each bin mass is floored at ``eps`` to keep ``log`` finite. The
    contributions are ``(c - r) * log(c / r)`` per bin.
    """
    r = np.maximum(ref_pct, eps)
    c = np.maximum(cur_pct, eps)
    contributions = (c - r) * np.log(c / r)
    return float(np.sum(contributions)), contributions


def psi(
    reference: pd.Series | np.ndarray | pd.DataFrame,
    current: pd.Series | np.ndarray | pd.DataFrame,
    bins: int | Sequence[float] | np.ndarray = 10,
    *,
    weights_reference: str
    | np.ndarray
    | pd.Series
    | None = None,
    weights_current: str | np.ndarray | pd.Series | None = None,
    eps: float = _PSI_EPS,
) -> PSIResult:
    """Population Stability Index between two distributions.

    Parameters
    ----------
    reference : array-like or Series or 1-column DataFrame
        The "baseline" distribution. Bin edges are taken from this sample.
    current : array-like or Series or 1-column DataFrame
        The "current" distribution to compare against the reference.
    bins : int or sequence of floats
        ``int``: equal-frequency bins on the reference (default ``10``).
        Sequence: explicit edges (length = ``n_bins + 1``).
    weights_reference, weights_current :
        Optional weights. A string is looked up in a ``DataFrame`` input;
        otherwise a 1-D array of matching length.
    eps :
        Floor for bin masses before entering ``log`` (default ``1e-6``).

    Returns
    -------
    PSIResult
    """
    ref = _to_array(reference, "reference")
    cur = _to_array(current, "current")

    # Resolve weights. The string-column path only applies when the input
    # itself is a DataFrame, but we accept arrays/Series here too.
    def _coerce(w: Any, host: Any, n: int) -> np.ndarray | None:
        if w is None:
            return None
        if isinstance(w, str):
            if not isinstance(host, pd.DataFrame):
                raise TypeError(
                    "string weights require a DataFrame input for the "
                    "corresponding distribution"
                )
            return coerce_weights(w, host)
        arr = np.asarray(w, dtype=float).ravel()
        if arr.shape != (n,):
            raise ValueError(
                f"weights shape {arr.shape} != distribution length {n}"
            )
        if not np.isfinite(arr).all():
            raise ValueError("weights contain non-finite values")
        if np.any(arr < 0):
            raise ValueError("weights must be non-negative")
        if not np.any(arr > 0):
            raise ValueError("weights must contain at least one positive value")
        return arr

    w_ref = _coerce(weights_reference, reference, ref.size)
    w_cur = _coerce(weights_current, current, cur.size)

    edges = _edges_from_reference(ref, bins, w_ref)
    ref_pct = _bin_mass(ref, edges, w_ref)
    cur_pct = _bin_mass(cur, edges, w_cur)
    value, contribs = _psi_from_pcts(ref_pct, cur_pct, eps=eps)
    return PSIResult(
        value=value,
        bins=edges.size - 1,
        edges=edges,
        contributions=contribs,
        ref_pct=ref_pct,
        cur_pct=cur_pct,
    )
