r"""WoE-encoded and bin-indicator basis terms (DESIGN.md §3.1).

The two Term values here both consume a :class:`BinningResult` from
:mod:`.binning`:

* :class:`WoETerm` produces a single column whose value for row ``i`` is
  :math:`\mathrm{WoE}_{\mathrm{bin}(x_i)}`. The joint logistic regression
  learns one global coefficient that scales these values; in a properly-
  binned column that coefficient should be :math:`\approx +1`, which is
  what :class:`WoEMonotonicityPreserved` verifies post-fit.
* :class:`BinnedTerm` is ESL §5.2's step-function basis: :math:`k-1`
  indicator columns (drop-first), one coefficient per non-reference bin.
  Higher-dimensional than WoE and pairs naturally with :func:`mc.l2` to
  keep the joint fit identifiable.

Missing-value policy (DESIGN.md §11)
------------------------------------

Every WoE/binned term carries an explicit ``(Missing)`` bin when NaN is
present at fit time. At predict time, NaN rows go to that bin (and a
warning fires if no ``(Missing)`` bin was learned). For categorical WoE,
unseen categories at predict time get :math:`\mathrm{WoE} = 0` (neutral
evidence). For numeric WoE, predict-time values outside the training
range fall into the nearest edge bin.

Why :func:`fit_binnings` exists
-------------------------------

``_internal/design.py``'s ``build_design`` calls
``term.expand(data, fit_state=...)``. This is enough for **predict time**
(the round-trip works because :func:`predict` threads ``sol.fit_state``
through), but **at solve time** the design code does not capture state
produced by terms and the solver dispatch does not pass ``y`` to
``build_design``. Binning is a supervised learning step.

Resolution without modifying ``_internal/design.py``: WoE/Binned terms
carry their fitted bin definitions on the term itself. The user (or
``solve``) calls :func:`fit_binnings(spec, data)` to produce a new spec
whose binning terms have their bins baked in. ``expand()`` then reads
bins off the term, not off ``fit_state``. ``fit_state`` is *also*
populated as a courtesy so :func:`~model_crafter.inspect.binning_table`
can read it via the standard solution plumbing.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
import pandas as pd

from model_crafter.assumptions.woe import (
    AtLeastOneEventPerBin,
    MinimumBinSize,
    MonotonicEventRate,
    WoEMonotonicityPreserved,
)
from model_crafter.terms.base import ExpandedTerm
from model_crafter.terms.binning import (
    _SMOOTHING,
    MISSING_BIN_LABEL,
    RARE_CATEGORY_LABEL,
    Binning,
    BinningResult,
    CategoricalBinning,
    ManualBinning,
    MonotonicBinning,
    TreeBinning,
    categorical,
    manual,
    monotonic,
    tree_bins,
)

__all__ = [
    "BinnedTerm",
    "Binning",
    "BinningResult",
    "CategoricalBinning",
    "ManualBinning",
    "MISSING_BIN_LABEL",
    "MissingPolicy",
    "MonotonicBinning",
    "RARE_CATEGORY_LABEL",
    "TreeBinning",
    "WoETerm",
    "_SMOOTHING",
    "binned",
    "categorical",
    "fit_binnings",
    "manual",
    "monotonic",
    "tree_bins",
    "woe",
]


# Predict-time policy for handling missing / unseen / out-of-range values.
MissingPolicy = Literal["nearest", "zero", "missing_bin"]


# ---------------------------------------------------------------------------
# WoETerm and BinnedTerm.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WoETerm:
    r"""WoE-encoded basis term: one column, one joint coefficient.

    Construct with :func:`woe(col, bins=...)`. At spec-construction time
    the term carries only the column name and the binning strategy; its
    bin definitions are learned by :func:`fit_binnings(spec, data)`, which
    returns a new spec whose ``WoETerm.fitted`` field is populated.

    Once fitted, ``expand()`` returns a single column whose value for row
    :math:`i` is :math:`\mathrm{WoE}_{\mathrm{bin}(x_i)}`.
    """

    column: str
    binning: Binning
    fitted: BinningResult | None = None

    assumptions: tuple[Any, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.column, str) or not self.column:
            raise ValueError(f"WoETerm.column must be a non-empty str; got {self.column!r}")
        if not isinstance(self.binning, (MonotonicBinning, TreeBinning, CategoricalBinning, ManualBinning)):
            raise TypeError(
                f"WoETerm.binning must be a Binning strategy "
                f"(mc.monotonic / mc.tree_bins / mc.categorical / mc.manual); "
                f"got {type(self.binning).__name__}"
            )
        if not self.assumptions:
            object.__setattr__(
                self,
                "assumptions",
                (
                    AtLeastOneEventPerBin(),
                    MinimumBinSize(),
                    MonotonicEventRate(),
                    WoEMonotonicityPreserved(),
                ),
            )

    @property
    def name(self) -> str:
        # Single design column, named after the source feature.
        return self.column

    def expand(
        self,
        data: pd.DataFrame,
        *,
        fit_state: Mapping[str, Any] | None = None,
    ) -> ExpandedTerm:
        result = _resolve_fit(self, fit_state)
        x: pd.Series = data[self.column]  # type: ignore[assignment]
        woe_vals = _assign_woe(x, result)
        return ExpandedTerm(
            columns=(self.column,),
            values=woe_vals.reshape(-1, 1),
        )

    def __add__(self, other: object) -> Any:
        from model_crafter.terms.base import _add_terms
        return _add_terms(self, other)

    def __radd__(self, other: object) -> Any:
        from model_crafter.terms.base import _add_terms
        return _add_terms(other, self)

    def _fit(
        self,
        data: pd.DataFrame,
        y: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> WoETerm:
        """Return a new :class:`WoETerm` with ``fitted`` populated."""
        if self.column not in data.columns:
            raise KeyError(
                f"WoETerm refers to column '{self.column}' which is not in the data frame"
            )
        x: pd.Series = data[self.column]  # type: ignore[assignment]
        result = self.binning.fit(
            x, y, weights=weights, column=self.column
        )
        return replace(self, fitted=result)


@dataclass(frozen=True, slots=True)
class BinnedTerm:
    r"""Bin-indicator basis term: ``k-1`` indicator columns, one coef per bin.

    Construct with :func:`binned(col, bins=...)`. Drop-first encoding: the
    first learned bin is the reference. Pairs well with :func:`mc.l2`
    (DESIGN.md §3.1).

    Column-name convention: ``"{col}__{label}"``, where ``label`` is the
    bin's human-readable label. The reference bin is implicit in the
    intercept.
    """

    column: str
    binning: Binning
    fitted: BinningResult | None = None

    assumptions: tuple[Any, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.column, str) or not self.column:
            raise ValueError(f"BinnedTerm.column must be a non-empty str; got {self.column!r}")
        if not isinstance(self.binning, (MonotonicBinning, TreeBinning, CategoricalBinning, ManualBinning)):
            raise TypeError(
                f"BinnedTerm.binning must be a Binning strategy; "
                f"got {type(self.binning).__name__}"
            )
        if not self.assumptions:
            object.__setattr__(
                self,
                "assumptions",
                (
                    AtLeastOneEventPerBin(),
                    MinimumBinSize(),
                ),
            )

    @property
    def name(self) -> str:
        # The *term* name (for fit_state lookup and error messages) is the
        # source column. Individual expanded columns get suffixes.
        return self.column

    def expand(
        self,
        data: pd.DataFrame,
        *,
        fit_state: Mapping[str, Any] | None = None,
    ) -> ExpandedTerm:
        result = _resolve_fit(self, fit_state)
        x: pd.Series = data[self.column]  # type: ignore[assignment]
        bin_idx = _assign_bin_index(x, result)
        k = result.n_bins
        if k <= 1:
            raise ValueError(
                f"BinnedTerm('{self.column}'): only {k} bin(s) were learned; "
                "need at least 2 to produce indicators (the reference is dropped). "
                "Consider a coarser binning strategy or use mc.woe instead."
            )
        n = len(data)
        # Drop bin 0 as the reference.
        ind = np.zeros((n, k - 1), dtype=float)
        for j in range(1, k):
            ind[:, j - 1] = (bin_idx == j).astype(float)
        cols = tuple(f"{self.column}__{result.bin_labels[j]}" for j in range(1, k))
        return ExpandedTerm(columns=cols, values=ind)

    def __add__(self, other: object) -> Any:
        from model_crafter.terms.base import _add_terms
        return _add_terms(self, other)

    def __radd__(self, other: object) -> Any:
        from model_crafter.terms.base import _add_terms
        return _add_terms(other, self)

    def _fit(
        self,
        data: pd.DataFrame,
        y: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> BinnedTerm:
        if self.column not in data.columns:
            raise KeyError(
                f"BinnedTerm refers to column '{self.column}' which is not in the data frame"
            )
        x: pd.Series = data[self.column]  # type: ignore[assignment]
        result = self.binning.fit(
            x, y, weights=weights, column=self.column
        )
        return replace(self, fitted=result)


# ---------------------------------------------------------------------------
# Public constructors.
# ---------------------------------------------------------------------------


def woe(col: str, bins: Binning) -> WoETerm:
    """Construct a WoE-encoded term — single coefficient on a WoE column.

    ``bins`` is a binning strategy value from :mod:`.binning`
    (:func:`monotonic`, :func:`tree_bins`, :func:`categorical`, or
    :func:`manual`). See DESIGN.md §3.1.
    """
    return WoETerm(column=col, binning=bins)


def binned(col: str, bins: Binning) -> BinnedTerm:
    """Construct a bin-indicator term — one coefficient per non-reference bin.

    See DESIGN.md §3.1 (ESL §5.2 framing) for when to prefer this over
    :func:`woe`. The first learned bin is dropped as the reference; for
    full identifiability pair with ``mc.l2(...)``.
    """
    return BinnedTerm(column=col, binning=bins)


# ---------------------------------------------------------------------------
# Spec-level fitting helper.
# ---------------------------------------------------------------------------


def fit_binnings(
    spec: Any,
    data: pd.DataFrame,
    *,
    weights: str | np.ndarray | None = None,
) -> Any:
    """Return a new :class:`LinearSpec` with WoE/Binned terms fitted on ``data``.

    Walks ``spec.features``, calls each binning term's strategy on the
    target column, and replaces those terms with their fitted equivalents.
    Non-binning terms pass through untouched.
    """
    from model_crafter.spec import LinearSpec

    if not isinstance(spec, LinearSpec):
        raise TypeError(f"spec must be a LinearSpec; got {type(spec).__name__}")
    if spec.target not in data.columns:
        raise KeyError(
            f"target column '{spec.target}' not in data (columns: {list(data.columns)})"
        )
    y = np.asarray(pd.to_numeric(data[spec.target], errors="raise"), dtype=float)

    w: np.ndarray | None
    if weights is None:
        w = None
    elif isinstance(weights, str):
        w = data[weights].to_numpy(dtype=float)
    else:
        w = np.asarray(weights, dtype=float)

    new_features: list[Any] = []
    for term in spec.features:
        if isinstance(term, (WoETerm, BinnedTerm)):
            new_features.append(term._fit(data, y, weights=w))
        else:
            new_features.append(term)
    return replace(spec, features=tuple(new_features))


# ---------------------------------------------------------------------------
# Predict-time assignment helpers.
# ---------------------------------------------------------------------------


def _resolve_fit(
    term: WoETerm | BinnedTerm,
    fit_state: Mapping[str, Any] | None,
) -> BinningResult:
    """Resolve the BinningResult for ``term``.

    Preference order: a BinningResult attached to the term (set by
    :func:`fit_binnings`), then the ``fit_state`` mapping (used at predict
    time when the integrator stuffs the term's state into ``sol.fit_state``).
    Raises a clear error if neither is available.
    """
    if term.fitted is not None:
        return term.fitted
    if fit_state is not None:
        if isinstance(fit_state, BinningResult):
            return fit_state
        if isinstance(fit_state, Mapping):
            cand = fit_state.get("result")
            if isinstance(cand, BinningResult):
                return cand
    raise ValueError(
        f"WoE/binned term '{term.column}' has no fitted bins. "
        "Call `spec = mc.fit_binnings(spec, data)` before solving — see "
        "model_crafter/terms/woe.py module docstring for the integration note."
    )


def _assign_bin_index(x: pd.Series, result: BinningResult) -> np.ndarray:
    """Assign each row of ``x`` to a bin index in ``[0, n_bins)``.

    Handles missing values (route to the ``(Missing)`` bin if present;
    otherwise route to the nearest bin and emit a warning) and out-of-
    range / unseen categories.
    """
    if result.kind == "numeric":
        return _assign_bin_index_numeric(x, result)
    return _assign_bin_index_categorical(x, result)


def _assign_bin_index_numeric(x: pd.Series, result: BinningResult) -> np.ndarray:
    x_arr = np.asarray(pd.to_numeric(x, errors="coerce"), dtype=float)
    nan_mask = np.isnan(x_arr)
    n = len(x_arr)
    n_numeric_bins = len(result.edges) + 1  # interior edges → bins
    out = np.empty(n, dtype=int)
    if n_numeric_bins > 0:
        edges = np.asarray(result.edges, dtype=float)
        full_edges = np.concatenate(([-np.inf], edges, [np.inf]))
        idx = pd.cut(x_arr, bins=full_edges, include_lowest=True, labels=False)
        idx_arr = np.asarray(idx, dtype=float)
        # pd.cut returns NaN for NaN inputs.
        out_clean_mask = ~nan_mask
        out[out_clean_mask] = idx_arr[out_clean_mask].astype(int)
    if nan_mask.any():
        if result.has_missing_bin:
            out[nan_mask] = result.n_bins - 1  # last bin is (Missing)
        else:
            warnings.warn(
                f"column '{result.column}': NaN encountered at predict time but no "
                "(Missing) bin was learned at fit time; routing to nearest edge bin (bin 0).",
                stacklevel=3,
            )
            out[nan_mask] = 0
    return out


def _assign_bin_index_categorical(
    x: pd.Series, result: BinningResult
) -> np.ndarray:
    n = len(x)
    out = np.empty(n, dtype=int)
    cat_to_idx = {c: i for i, c in enumerate(result.categories)}
    rare_idx = cat_to_idx.get(RARE_CATEGORY_LABEL)
    missing_idx = result.n_bins - 1 if result.has_missing_bin else None

    x_values = pd.Series(x).astype("object").to_numpy()
    for i, v in enumerate(x_values):
        if pd.isna(v):
            if missing_idx is not None:
                out[i] = missing_idx
            else:
                warnings.warn(
                    f"column '{result.column}': NaN encountered at predict time but no "
                    "(Missing) bin was learned at fit time; routing to bin 0.",
                    stacklevel=3,
                )
                out[i] = 0
        elif v in cat_to_idx:
            out[i] = cat_to_idx[v]
        elif rare_idx is not None:
            out[i] = rare_idx
        else:
            warnings.warn(
                f"column '{result.column}': unseen category {v!r} at predict time; "
                "assigning WoE=0 (neutral evidence). DESIGN.md §11.",
                stacklevel=3,
            )
            out[i] = 0
    return out


def _assign_woe(x: pd.Series, result: BinningResult) -> np.ndarray:
    """Return the WoE value per row.

    Unlike :func:`_assign_bin_index`, unseen categorical levels at predict
    time get :math:`\\mathrm{WoE} = 0` (neutral evidence; DESIGN.md §11)
    rather than being mapped to bin 0's WoE.
    """
    if result.kind == "categorical":
        n = len(x)
        woe_vals = np.zeros(n, dtype=float)
        cat_to_idx = {c: i for i, c in enumerate(result.categories)}
        rare_idx = cat_to_idx.get(RARE_CATEGORY_LABEL)
        missing_idx = result.n_bins - 1 if result.has_missing_bin else None
        woe_arr = np.asarray(result.woe, dtype=float)
        x_values = pd.Series(x).astype("object").to_numpy()
        for i, v in enumerate(x_values):
            if pd.isna(v):
                if missing_idx is not None:
                    woe_vals[i] = woe_arr[missing_idx]
                else:
                    warnings.warn(
                        f"column '{result.column}': NaN encountered at predict time but no "
                        "(Missing) bin was learned at fit time; assigning WoE=0.",
                        stacklevel=3,
                    )
                    woe_vals[i] = 0.0
            elif v in cat_to_idx:
                woe_vals[i] = woe_arr[cat_to_idx[v]]
            elif rare_idx is not None:
                woe_vals[i] = woe_arr[rare_idx]
            else:
                warnings.warn(
                    f"column '{result.column}': unseen category {v!r} at predict time; "
                    "assigning WoE=0 (neutral evidence). DESIGN.md §11.",
                    stacklevel=3,
                )
                woe_vals[i] = 0.0
        return woe_vals
    # Numeric: just look up bin index → WoE.
    idx = _assign_bin_index(x, result)
    return np.asarray(result.woe, dtype=float)[idx]
