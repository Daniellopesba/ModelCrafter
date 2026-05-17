r"""Weight-of-Evidence and bin-indicator terms with their binning strategies.

This module implements DESIGN.md §3.1: the credit-industry-default
:func:`woe` term (single coefficient on a WoE-encoded basis), the
ESL §5.2-honest alternative :func:`binned` (one coefficient per bin), and
the four binning *strategies* that produce bin edges from data:
:func:`monotonic`, :func:`tree_bins`, :func:`categorical`, :func:`manual`.

Math
----

For a binary target :math:`y \in \{0, 1\}` and a feature :math:`x`
discretised into bins :math:`B_1, \ldots, B_k`, the Weight of Evidence for
bin :math:`b` is (Siddiqi 2006, §6):

.. math::

    \mathrm{WoE}_b \;=\; \log\!\Bigl(
        \tfrac{n^{(1)}_b / n^{(1)}_\bullet}{n^{(0)}_b / n^{(0)}_\bullet}
    \Bigr)

where :math:`n^{(1)}_b` and :math:`n^{(0)}_b` are event and non-event
counts in bin :math:`b` and :math:`n^{(\cdot)}_\bullet` are the
column-totals. Laplace smoothing (``+0.5`` events / non-events per bin)
handles empty cells: a bin with zero events would otherwise produce
:math:`-\infty`, which is mathematically right but operationally useless.

The **WoE-encoded basis** (``mc.woe``) produces a single column whose
value for row ``i`` is :math:`\mathrm{WoE}_{\mathrm{bin}(x_i)}`. The
joint logistic regression learns one global coefficient that scales these
values; in a properly-binned column that coefficient should be
:math:`\approx +1`, which is what the :class:`WoEMonotonicityPreserved`
post-fit check verifies.

The **bin-indicator basis** (``mc.binned``) is ESL §5.2's step-function
basis: :math:`k-1` indicator columns (drop-first), one coefficient per
non-reference bin. Higher-dimensional than WoE and pairs naturally with
``mc.l2(...)`` to keep the joint fit identifiable.

The **Information Value** per WoE term is the single scalar

.. math::

    \mathrm{IV} \;=\; \sum_b
        \bigl(p^{(1)}_b - p^{(0)}_b\bigr) \cdot \mathrm{WoE}_b

where :math:`p^{(\cdot)}_b = n^{(\cdot)}_b / n^{(\cdot)}_\bullet`. Industry
rules of thumb (Siddiqi 2006, Anderson 2007): IV < 0.02 useless,
0.02–0.1 weak, 0.1–0.3 medium, 0.3–0.5 strong, > 0.5 suspiciously high
(likely target leakage). The package reports the value; it does not
enforce these thresholds.

Missing-value handling (DESIGN.md §11 open question)
-----------------------------------------------------

Every WoE/binned term carries an explicit ``(Missing)`` bin when NaN is
present at fit time. At predict time, NaN rows go to that bin (and a
warning is emitted via the assumption framework if no ``(Missing)`` bin
was learned). For categorical WoE, unseen categories at predict time get
``WoE = 0`` (neutral evidence). For numeric WoE, predict-time values
outside the training range fall into the nearest edge bin (closed-on-both-
sides extrapolation — the conservative choice for credit risk).

Integration with ``_internal/design.py``
----------------------------------------

The current ``build_design`` calls ``term.expand(data, fit_state=...)``
where ``fit_state`` is keyed by term name. This is enough for **predict
time** (the round-trip works because :func:`predict` threads
``sol.fit_state`` through), but **at solve time** the design code does
*not* capture state produced by terms, and the solver dispatch does *not*
pass ``y`` to ``build_design``. Binning is a supervised learning step.

Resolution (without modifying ``_internal/design.py``): WoE/Binned terms
are *values* that carry their fitted bin definitions on the term itself.
The user (or, post-integration, ``solve``) calls
:func:`fit_binnings(spec, data)` to produce a new :class:`LinearSpec`
whose binning terms have their bins baked in. ``expand()`` then reads bins
off the term, not off ``fit_state``. ``fit_state`` is *also* populated as
a courtesy so :func:`~model_crafter.inspect.binning_table` can read it
via the standard solution plumbing.

See ``notes/P4.B.md`` for the design.py extension flagged for
integration: ``ExpandedTerm`` needs a ``state`` field so terms can
round-trip learned state through ``build_design``, and ``solve()`` needs
to call ``fit_binnings`` (or equivalent) before ``build_design``.

References
----------
* Siddiqi, N. (2006). *Credit Risk Scorecards*. Wiley.
* Anderson, R. (2007). *The Credit Scoring Toolkit*. Oxford.
* Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*
  (2nd ed.), §5.2 (step-function basis expansions).
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
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

__all__ = [
    "BinnedTerm",
    "Binning",
    "BinningResult",
    "CategoricalBinning",
    "ManualBinning",
    "MissingPolicy",
    "MonotonicBinning",
    "TreeBinning",
    "WoETerm",
    "binned",
    "categorical",
    "fit_binnings",
    "manual",
    "monotonic",
    "tree_bins",
    "woe",
]


MISSING_BIN_LABEL = "(Missing)"
RARE_CATEGORY_LABEL = "RARE"
_SMOOTHING = 0.5  # Laplace smoothing in events/non-events per bin.

# Predict-time policy for handling missing / unseen / out-of-range values.
MissingPolicy = Literal["nearest", "zero", "missing_bin"]


# ---------------------------------------------------------------------------
# BinningResult: the value produced by a strategy.fit(x, y) call.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BinningResult:
    r"""The output of binning one feature column against a binary target.

    Carries everything ``expand()`` and :func:`binning_table` need.

    Attributes
    ----------
    column
        Source data column name.
    kind
        ``"numeric"`` (edges-based) or ``"categorical"`` (label-based).
    edges
        For ``kind="numeric"``: the bin edges as a strictly increasing 1-D
        array. ``k`` bins are produced from ``k + 1`` edges, with the
        convention that bin :math:`i` covers :math:`(edges[i],
        edges[i+1]]` (open-on-left, closed-on-right; matches
        :func:`pandas.cut`). The first edge is :math:`-\infty` and the
        last is :math:`+\infty` after fitting, so all training values are
        contained.
    categories
        For ``kind="categorical"``: the list of categories (one per bin),
        already in their fit-time order. The optional rare-bucket label
        ``"RARE"`` is included if rare-grouping was applied.
    bin_labels
        Human-readable label per bin (e.g. ``"(-inf, 25.0]"`` or
        ``"south"``); used in the ``BinningTable`` and in
        :class:`BinnedTerm` column names.
    has_missing_bin
        Whether NaN was observed at fit time. When ``True``, the
        ``(Missing)`` bucket is the last entry in ``n_events``,
        ``n_nonevents``, ``woe``, etc., and ``bin_labels[-1] ==
        "(Missing)"``.
    n_events, n_nonevents
        ``(n_bins,)`` int arrays of event / non-event counts, post-fit
        and pre-smoothing.
    woe
        ``(n_bins,)`` array of WoE per bin (with smoothing).
    iv
        Scalar Information Value for the entire feature.
    """

    column: str
    kind: Literal["numeric", "categorical"]
    edges: tuple[float, ...]  # empty for categorical
    categories: tuple[Any, ...]  # empty for numeric
    bin_labels: tuple[str, ...]
    has_missing_bin: bool
    n_events: tuple[int, ...]
    n_nonevents: tuple[int, ...]
    woe: tuple[float, ...]
    iv: float

    @property
    def n_bins(self) -> int:
        return len(self.bin_labels)

    @property
    def event_rate(self) -> tuple[float, ...]:
        return tuple(
            (e / (e + ne)) if (e + ne) > 0 else 0.0
            for e, ne in zip(self.n_events, self.n_nonevents, strict=True)
        )

    @property
    def n_total(self) -> tuple[int, ...]:
        return tuple(
            e + ne for e, ne in zip(self.n_events, self.n_nonevents, strict=True)
        )


# ---------------------------------------------------------------------------
# Binning strategies (values, not classes — user calls mc.monotonic() etc.)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MonotonicBinning:
    r"""Supervised monotonic binning for numeric features.

    Algorithm (Siddiqi 2006, ch. 6):

    1. Start from a fine percentile-based discretisation (initial bins =
       ``max_bins``).
    2. Greedily merge adjacent bins until *every* bin has
       :math:`\geq` ``min_bin_size`` :math:`\cdot n` rows.
    3. Greedily merge adjacent bins until event rates are monotonic across
       ordered bins (the sign of monotonicity — increasing or decreasing —
       is chosen to maximise the resulting IV).

    Missing values get a dedicated ``(Missing)`` bin if any NaN is present
    in ``x`` at fit time.

    Parameters
    ----------
    min_bin_size
        Minimum fraction of rows per bin (default ``0.05``).
    max_bins
        Maximum number of fine bins before merging (default ``20``).
    """

    min_bin_size: float = 0.05
    max_bins: int = 20
    kind: Literal["monotonic"] = "monotonic"

    def __post_init__(self) -> None:
        if not (0.0 < self.min_bin_size < 1.0):
            raise ValueError(
                f"min_bin_size must be in (0, 1); got {self.min_bin_size}"
            )
        if self.max_bins < 2:
            raise ValueError(f"max_bins must be >= 2; got {self.max_bins}")

    def fit(
        self,
        x: pd.Series,
        y: np.ndarray,
        *,
        weights: np.ndarray | None = None,
        column: str = "",
    ) -> BinningResult:
        return _fit_monotonic(
            x,
            y,
            weights=weights,
            column=column or (str(x.name) if x.name is not None else ""),
            min_bin_size=self.min_bin_size,
            max_bins=self.max_bins,
        )


@dataclass(frozen=True, slots=True)
class TreeBinning:
    r"""Decision-tree binning: greedy single-feature CART for numeric x.

    Implementation note: this is a *single-feature* tree fit, no sklearn.
    At each node we evaluate every candidate split (the midpoints between
    sorted unique values) and pick the one minimising the weighted Gini
    impurity of the children. We stop when ``max_leaves`` leaves exist or
    no split keeps both children at :math:`\geq` ``min_samples_leaf``
    of the parent.

    Missing values are routed to a dedicated ``(Missing)`` bin (they are
    not used for split selection).
    """

    max_leaves: int = 10
    min_samples_leaf: float = 0.05
    kind: Literal["tree"] = "tree"

    def __post_init__(self) -> None:
        if self.max_leaves < 2:
            raise ValueError(f"max_leaves must be >= 2; got {self.max_leaves}")
        if not (0.0 < self.min_samples_leaf < 0.5):
            raise ValueError(
                f"min_samples_leaf must be in (0, 0.5); got {self.min_samples_leaf}"
            )

    def fit(
        self,
        x: pd.Series,
        y: np.ndarray,
        *,
        weights: np.ndarray | None = None,
        column: str = "",
    ) -> BinningResult:
        return _fit_tree(
            x,
            y,
            weights=weights,
            column=column or (str(x.name) if x.name is not None else ""),
            max_leaves=self.max_leaves,
            min_samples_leaf=self.min_samples_leaf,
        )


@dataclass(frozen=True, slots=True)
class CategoricalBinning:
    r"""Group rare categories together; each remaining level is its own bin.

    Levels whose row fraction is strictly less than ``group_rare`` are
    folded into a single ``"RARE"`` bucket. Missing values get their own
    ``(Missing)`` bin if present.
    """

    group_rare: float = 0.01
    kind: Literal["categorical"] = "categorical"

    def __post_init__(self) -> None:
        if not (0.0 <= self.group_rare < 1.0):
            raise ValueError(
                f"group_rare must be in [0, 1); got {self.group_rare}"
            )

    def fit(
        self,
        x: pd.Series,
        y: np.ndarray,
        *,
        weights: np.ndarray | None = None,
        column: str = "",
    ) -> BinningResult:
        return _fit_categorical(
            x,
            y,
            weights=weights,
            column=column or (str(x.name) if x.name is not None else ""),
            group_rare=self.group_rare,
        )


@dataclass(frozen=True, slots=True)
class ManualBinning:
    r"""User-supplied bin edges. No supervised fitting; only validation.

    ``edges`` is a strictly increasing sequence of interior break points;
    they are extended with :math:`-\infty` and :math:`+\infty` so all real
    values are contained. ``len(bins) == len(edges) + 1`` after fitting.

    Matches :func:`pandas.cut` semantics: bins are open-on-left,
    closed-on-right.
    """

    edges: tuple[float, ...]
    kind: Literal["manual"] = "manual"

    def __post_init__(self) -> None:
        edges = tuple(float(e) for e in self.edges)
        if len(edges) < 1:
            raise ValueError("manual edges requires at least one break point")
        if any(np.isnan(e) or np.isinf(e) for e in edges):
            raise ValueError("manual edges must be finite")
        if any(b <= a for a, b in zip(edges, edges[1:], strict=False)):
            raise ValueError(f"manual edges must be strictly increasing; got {edges}")
        # Re-assign as a tuple (frozen dataclass; bypass via object.__setattr__).
        object.__setattr__(self, "edges", edges)

    def fit(
        self,
        x: pd.Series,
        y: np.ndarray,
        *,
        weights: np.ndarray | None = None,
        column: str = "",
    ) -> BinningResult:
        return _fit_manual(
            x,
            y,
            weights=weights,
            column=column or (str(x.name) if x.name is not None else ""),
            edges=self.edges,
        )


# Union alias used in public signatures.
Binning = MonotonicBinning | TreeBinning | CategoricalBinning | ManualBinning


# ---------------------------------------------------------------------------
# Public binning-strategy constructors (the value-returning factories).
# ---------------------------------------------------------------------------


def monotonic(min_bin_size: float = 0.05, max_bins: int = 20) -> MonotonicBinning:
    """Construct a :class:`MonotonicBinning` strategy.

    See :class:`MonotonicBinning` for the algorithm. Default
    ``min_bin_size=0.05`` matches the credit-industry rule of thumb that
    every WoE bin should be supportable by at least 5% of the training
    sample.
    """
    return MonotonicBinning(min_bin_size=min_bin_size, max_bins=max_bins)


def tree_bins(
    max_leaves: int = 10, min_samples_leaf: float = 0.05
) -> TreeBinning:
    """Construct a :class:`TreeBinning` (greedy single-feature CART) strategy."""
    return TreeBinning(max_leaves=max_leaves, min_samples_leaf=min_samples_leaf)


def categorical(group_rare: float = 0.01) -> CategoricalBinning:
    """Construct a :class:`CategoricalBinning` strategy (rare-group then bin)."""
    return CategoricalBinning(group_rare=group_rare)


def manual(edges: Sequence[float]) -> ManualBinning:
    """Construct a :class:`ManualBinning` strategy from user-supplied edges.

    Per :func:`pandas.cut`, bins are open-on-left and closed-on-right.
    ``edges=[a, b, c]`` produces 4 bins: ``(-inf, a]``, ``(a, b]``,
    ``(b, c]``, ``(c, +inf]``.
    """
    return ManualBinning(edges=tuple(float(e) for e in edges))


# ---------------------------------------------------------------------------
# WoETerm and BinnedTerm: the actual Term values used in features=.
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

    See module docstring for the math (Siddiqi 2006; ESL §5.2).
    """

    column: str
    binning: Binning
    fitted: BinningResult | None = None

    # Assumptions declared per DESIGN.md §4.3.
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
        # Single design column named after the source feature. This is the
        # WoE-encoded scalar.
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
    *first* learned bin is the reference. Pairs well with :func:`mc.l2`
    (per DESIGN.md §3.1).

    Column-name convention: ``"{col}__{label}"``, where ``label`` is the
    bin's human-readable label (e.g. ``"(25.0, 35.0]"`` or
    ``"south"``). The reference bin is implicit in the intercept.
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
        # Drop-first: indicators for bins 1..k-1 (k = result.n_bins).
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

    See DESIGN.md §3.1 for the motivation. ``bins`` is a binning strategy
    value (use :func:`monotonic`, :func:`tree_bins`, :func:`categorical`,
    or :func:`manual`).
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
# Spec-level fitting helper (the bridge until _internal/design.py learns
# to thread y through to terms).
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

    See module docstring for why this lives outside ``solve()``.
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
# Internals: strategy implementations.
# ---------------------------------------------------------------------------


def _check_y(y: np.ndarray) -> None:
    if y.ndim != 1:
        raise ValueError(f"target y must be 1-D; got shape {y.shape}")
    uniq = np.unique(y[~np.isnan(y)]) if np.any(np.isnan(y)) else np.unique(y)
    if not np.all((uniq == 0) | (uniq == 1)):
        raise ValueError(
            f"WoE/binned terms require a binary {{0, 1}} target; got unique values {uniq}"
        )


def _percentile_edges(x_clean: np.ndarray, n_bins: int) -> np.ndarray:
    """Edges from equal-frequency (percentile) discretisation.

    Duplicate edges (caused by point masses) are collapsed; the resulting
    bin count may be less than ``n_bins``.
    """
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    # Interior edges only.
    interior = np.quantile(x_clean, qs[1:-1])
    # Remove duplicates while preserving order.
    interior = np.unique(interior)
    return interior


def _events_per_bin_numeric(
    x_clean: np.ndarray,
    y_clean: np.ndarray,
    w_clean: np.ndarray | None,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Event / non-event counts per bin given interior edges."""
    # pd.cut on the interior edges with -inf/+inf padding.
    full_edges = np.concatenate(([-np.inf], edges, [np.inf]))
    # Per pandas: bins are (a, b]; the leftmost is closed via include_lowest=True.
    bin_idx = pd.cut(x_clean, bins=full_edges, include_lowest=True, labels=False)
    bin_idx = np.asarray(bin_idx, dtype=int)
    k = len(full_edges) - 1
    if w_clean is None:
        ev = np.bincount(bin_idx[y_clean == 1], minlength=k).astype(float)
        nev = np.bincount(bin_idx[y_clean == 0], minlength=k).astype(float)
    else:
        ev = np.zeros(k, dtype=float)
        nev = np.zeros(k, dtype=float)
        for i, b in enumerate(bin_idx):
            if y_clean[i] == 1:
                ev[b] += w_clean[i]
            else:
                nev[b] += w_clean[i]
    return ev, nev


def _compute_woe_and_iv(
    n_events: np.ndarray, n_nonevents: np.ndarray
) -> tuple[np.ndarray, float]:
    """WoE per bin + total IV, with Laplace smoothing."""
    e_smooth = n_events + _SMOOTHING
    ne_smooth = n_nonevents + _SMOOTHING
    e_tot = e_smooth.sum()
    ne_tot = ne_smooth.sum()
    p_e = e_smooth / e_tot
    p_ne = ne_smooth / ne_tot
    woe = np.log(p_e / p_ne)
    iv = float(np.sum((p_e - p_ne) * woe))
    return woe, iv


def _label_numeric(edges_with_inf: np.ndarray) -> tuple[str, ...]:
    labels: list[str] = []
    for lo, hi in zip(edges_with_inf[:-1], edges_with_inf[1:], strict=True):
        lo_s = "-inf" if np.isneginf(lo) else f"{lo:.4g}"
        hi_s = "+inf" if np.isposinf(hi) else f"{hi:.4g}"
        labels.append(f"({lo_s}, {hi_s}]")
    return tuple(labels)


def _build_result_numeric(
    column: str,
    edges_with_inf: np.ndarray,
    n_events: np.ndarray,
    n_nonevents: np.ndarray,
    has_missing: bool,
    n_events_missing: int = 0,
    n_nonevents_missing: int = 0,
) -> BinningResult:
    labels = list(_label_numeric(edges_with_inf))
    ev_all = list(n_events)
    nev_all = list(n_nonevents)
    if has_missing:
        labels.append(MISSING_BIN_LABEL)
        ev_all.append(float(n_events_missing))
        nev_all.append(float(n_nonevents_missing))
    ev_arr = np.asarray(ev_all, dtype=float)
    nev_arr = np.asarray(nev_all, dtype=float)
    woe, iv = _compute_woe_and_iv(ev_arr, nev_arr)
    # Edges stored as the interior edges (without ±inf padding) so users
    # see the actual learned cuts.
    interior_edges = tuple(float(e) for e in edges_with_inf[1:-1])
    return BinningResult(
        column=column,
        kind="numeric",
        edges=interior_edges,
        categories=(),
        bin_labels=tuple(labels),
        has_missing_bin=has_missing,
        n_events=tuple(int(round(v)) for v in ev_arr),
        n_nonevents=tuple(int(round(v)) for v in nev_arr),
        woe=tuple(float(v) for v in woe),
        iv=iv,
    )


def _fit_monotonic(
    x: pd.Series,
    y: np.ndarray,
    *,
    weights: np.ndarray | None,
    column: str,
    min_bin_size: float,
    max_bins: int,
) -> BinningResult:
    y = np.asarray(y, dtype=float)
    _check_y(y)
    x_arr = np.asarray(pd.to_numeric(x, errors="coerce"), dtype=float)
    nan_mask = np.isnan(x_arr)
    has_missing = bool(nan_mask.any())
    x_clean = x_arr[~nan_mask]
    y_clean = y[~nan_mask]
    w_clean = None if weights is None else weights[~nan_mask]
    n_total = len(x_arr)
    if len(x_clean) == 0:
        raise ValueError(f"column '{column}': all values are NaN; cannot bin")

    # Step 1: start with up to max_bins percentile bins. Empty array means
    # ``x`` was essentially constant — fall back to a single bin.
    interior = _percentile_edges(x_clean, max_bins)
    interior_arr = (
        np.array([], dtype=float) if len(interior) == 0 else interior
    )

    # Step 2: enforce min_bin_size by greedy merge.
    min_count = max(1, int(np.floor(min_bin_size * n_total)))
    interior_arr = _enforce_min_size(
        x_clean, y_clean, w_clean, interior_arr, min_count
    )

    # Step 3: enforce monotone event rate by greedy merge.
    interior_arr = _enforce_monotone(
        x_clean, y_clean, w_clean, interior_arr
    )

    edges_with_inf = np.concatenate(([-np.inf], interior_arr, [np.inf]))
    ev, nev = _events_per_bin_numeric(x_clean, y_clean, w_clean, interior_arr)

    n_events_missing = 0
    n_nonevents_missing = 0
    if has_missing:
        y_missing = y[nan_mask]
        if weights is None:
            n_events_missing = int((y_missing == 1).sum())
            n_nonevents_missing = int((y_missing == 0).sum())
        else:
            w_missing = weights[nan_mask]
            n_events_missing = int(round(float(w_missing[y_missing == 1].sum())))
            n_nonevents_missing = int(round(float(w_missing[y_missing == 0].sum())))

    return _build_result_numeric(
        column=column,
        edges_with_inf=edges_with_inf,
        n_events=ev,
        n_nonevents=nev,
        has_missing=has_missing,
        n_events_missing=n_events_missing,
        n_nonevents_missing=n_nonevents_missing,
    )


def _enforce_min_size(
    x_clean: np.ndarray,
    y_clean: np.ndarray,
    w_clean: np.ndarray | None,
    interior: np.ndarray,
    min_count: int,
) -> np.ndarray:
    """Greedy merge of adjacent bins whose total count is below min_count."""
    while True:
        ev, nev = _events_per_bin_numeric(x_clean, y_clean, w_clean, interior)
        counts = ev + nev
        if len(counts) <= 1 or np.all(counts >= min_count):
            return interior
        # Find the smallest bin and merge with smaller neighbour.
        i = int(np.argmin(counts))
        if i == 0:
            drop = 0  # drop edge between bin 0 and 1
        elif i == len(counts) - 1:
            drop = len(interior) - 1
        else:
            # Merge with the smaller neighbour.
            left = counts[i - 1]
            right = counts[i + 1]
            drop = i - 1 if left <= right else i
        interior = np.delete(interior, drop)
        if len(interior) == 0:
            return interior


def _enforce_monotone(
    x_clean: np.ndarray,
    y_clean: np.ndarray,
    w_clean: np.ndarray | None,
    interior: np.ndarray,
) -> np.ndarray:
    """Greedy merge until event rates are monotone (the IV-maximising direction wins)."""
    if len(interior) == 0:
        return interior
    # Try both monotone directions; keep whichever needs fewer merges (and ties → higher IV).
    best_a = _enforce_monotone_direction(
        x_clean, y_clean, w_clean, interior.copy(), increasing=True
    )
    best_b = _enforce_monotone_direction(
        x_clean, y_clean, w_clean, interior.copy(), increasing=False
    )
    iv_a = _iv_of(x_clean, y_clean, w_clean, best_a)
    iv_b = _iv_of(x_clean, y_clean, w_clean, best_b)
    return best_a if iv_a >= iv_b else best_b


def _iv_of(
    x_clean: np.ndarray,
    y_clean: np.ndarray,
    w_clean: np.ndarray | None,
    interior: np.ndarray,
) -> float:
    ev, nev = _events_per_bin_numeric(x_clean, y_clean, w_clean, interior)
    _, iv = _compute_woe_and_iv(ev, nev)
    return iv


def _enforce_monotone_direction(
    x_clean: np.ndarray,
    y_clean: np.ndarray,
    w_clean: np.ndarray | None,
    interior: np.ndarray,
    *,
    increasing: bool,
) -> np.ndarray:
    while True:
        ev, nev = _events_per_bin_numeric(x_clean, y_clean, w_clean, interior)
        # Event rate per bin (smoothed to avoid 0/0 dominating decisions).
        rates = (ev + _SMOOTHING) / (ev + nev + 2 * _SMOOTHING)
        if len(rates) <= 1:
            return interior
        diffs = np.diff(rates)
        violations = (
            np.where(diffs < 0)[0] if increasing else np.where(diffs > 0)[0]
        )
        if len(violations) == 0:
            return interior
        # Merge the worst-violating adjacent pair (largest |delta| against direction).
        worst = int(violations[np.argmax(np.abs(diffs[violations]))])
        # Drop the edge between bin `worst` and `worst+1` → drop interior[worst].
        if len(interior) == 0:
            return interior
        interior = np.delete(interior, worst)


# ---------------------------------------------------------------------------
# Tree binning.
# ---------------------------------------------------------------------------


def _gini(p_event: float) -> float:
    return 2.0 * p_event * (1.0 - p_event)


def _split_gain(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray | None,
    threshold: float,
) -> tuple[float, int, int]:
    """Return (weighted Gini *decrease*, left count, right count) for a split."""
    left_mask = x <= threshold
    right_mask = ~left_mask
    if w is None:
        wl = float(left_mask.sum())
        wr = float(right_mask.sum())
        if wl == 0 or wr == 0:
            return -np.inf, int(wl), int(wr)
        pl = float(y[left_mask].sum()) / wl
        pr = float(y[right_mask].sum()) / wr
    else:
        wl = float(w[left_mask].sum())
        wr = float(w[right_mask].sum())
        if wl == 0 or wr == 0:
            return -np.inf, int(wl), int(wr)
        pl = float((y[left_mask] * w[left_mask]).sum()) / wl
        pr = float((y[right_mask] * w[right_mask]).sum()) / wr
    n = wl + wr
    parent_p = (wl * pl + wr * pr) / n
    parent_gini = _gini(parent_p)
    child_gini = (wl / n) * _gini(pl) + (wr / n) * _gini(pr)
    return parent_gini - child_gini, int(wl), int(wr)


def _fit_tree(
    x: pd.Series,
    y: np.ndarray,
    *,
    weights: np.ndarray | None,
    column: str,
    max_leaves: int,
    min_samples_leaf: float,
) -> BinningResult:
    y = np.asarray(y, dtype=float)
    _check_y(y)
    x_arr = np.asarray(pd.to_numeric(x, errors="coerce"), dtype=float)
    nan_mask = np.isnan(x_arr)
    has_missing = bool(nan_mask.any())
    x_clean = x_arr[~nan_mask]
    y_clean = y[~nan_mask]
    w_clean = None if weights is None else weights[~nan_mask]
    n_total = len(x_clean)
    if n_total == 0:
        raise ValueError(f"column '{column}': all values are NaN; cannot bin")
    min_leaf = max(1, int(np.floor(min_samples_leaf * n_total)))

    # Maintain a sorted list of interior thresholds. Start with no splits.
    thresholds: list[float] = []
    # Region representation: a sorted list of (lo, hi) pairs as indices into
    # the unique sorted values. Simpler: at each step, evaluate the best
    # split inside each current bin.

    # Sort once for efficiency.
    sort_idx = np.argsort(x_clean)
    xs = x_clean[sort_idx]
    ys = y_clean[sort_idx]
    ws = None if w_clean is None else w_clean[sort_idx]

    def best_split_in_segment(
        a: int, b: int
    ) -> tuple[float, float]:
        """Return (best_gain, best_threshold) for the segment xs[a:b]."""
        if b - a < 2:
            return -np.inf, np.nan
        seg_x = xs[a:b]
        seg_y = ys[a:b]
        seg_w = None if ws is None else ws[a:b]
        # Candidate thresholds: midpoints between consecutive unique values.
        uniq = np.unique(seg_x)
        if len(uniq) < 2:
            return -np.inf, np.nan
        midpoints = (uniq[:-1] + uniq[1:]) / 2.0
        best_gain = -np.inf
        best_thr = np.nan
        for t in midpoints:
            gain, n_left, n_right = _split_gain(seg_x, seg_y, seg_w, float(t))
            if n_left < min_leaf or n_right < min_leaf:
                continue
            if gain > best_gain:
                best_gain = gain
                best_thr = float(t)
        return best_gain, best_thr

    # Iteratively split the segment with the largest gain.
    segments: list[tuple[int, int]] = [(0, n_total)]
    while len(segments) < max_leaves:
        best_seg_i = -1
        best_gain = -np.inf
        best_thr = np.nan
        for i, (a, b) in enumerate(segments):
            g, t = best_split_in_segment(a, b)
            if g > best_gain:
                best_gain = g
                best_thr = t
                best_seg_i = i
        if not np.isfinite(best_gain) or best_seg_i < 0:
            break
        # Apply the split: replace segment with two halves.
        a, b = segments.pop(best_seg_i)
        # Find split index.
        split_idx = a + int(np.searchsorted(xs[a:b], best_thr, side="right"))
        segments.insert(best_seg_i, (a, split_idx))
        segments.insert(best_seg_i + 1, (split_idx, b))
        thresholds.append(best_thr)
        thresholds.sort()

    interior = np.array(sorted(thresholds), dtype=float)
    edges_with_inf = np.concatenate(([-np.inf], interior, [np.inf]))
    ev, nev = _events_per_bin_numeric(x_clean, y_clean, w_clean, interior)

    n_events_missing = 0
    n_nonevents_missing = 0
    if has_missing:
        y_missing = y[nan_mask]
        if weights is None:
            n_events_missing = int((y_missing == 1).sum())
            n_nonevents_missing = int((y_missing == 0).sum())
        else:
            w_missing = weights[nan_mask]
            n_events_missing = int(round(float(w_missing[y_missing == 1].sum())))
            n_nonevents_missing = int(round(float(w_missing[y_missing == 0].sum())))

    return _build_result_numeric(
        column=column,
        edges_with_inf=edges_with_inf,
        n_events=ev,
        n_nonevents=nev,
        has_missing=has_missing,
        n_events_missing=n_events_missing,
        n_nonevents_missing=n_nonevents_missing,
    )


# ---------------------------------------------------------------------------
# Categorical binning.
# ---------------------------------------------------------------------------


def _fit_categorical(
    x: pd.Series,
    y: np.ndarray,
    *,
    weights: np.ndarray | None,
    column: str,
    group_rare: float,
) -> BinningResult:
    y = np.asarray(y, dtype=float)
    _check_y(y)
    # Treat NaN as missing.
    x_arr = pd.Series(x).astype("object")
    nan_mask = x_arr.isna().to_numpy()
    has_missing = bool(nan_mask.any())

    n_total = len(x_arr)
    # Compute per-category frequency on the non-missing rows.
    freq: dict[Any, int] = {}
    for val, m in zip(x_arr.to_numpy(), nan_mask, strict=True):
        if m:
            continue
        freq[val] = freq.get(val, 0) + 1
    # Categories appearing in < group_rare * n_total become "RARE".
    threshold = group_rare * n_total
    kept: list[Any] = []
    rare: list[Any] = []
    for cat, count in freq.items():
        if count < threshold and group_rare > 0:
            rare.append(cat)
        else:
            kept.append(cat)
    # Stable order: kept by descending frequency (industry convention so the
    # most-common category is the dropped reference in BinnedTerm).
    kept.sort(key=lambda c: -freq[c])
    categories: list[Any] = list(kept)
    if rare:
        categories.append(RARE_CATEGORY_LABEL)

    bin_labels = [str(c) for c in categories]

    # Count events / non-events per category.
    rare_set = set(rare)

    def _bucket_of(v: Any) -> Any:
        if v in rare_set:
            return RARE_CATEGORY_LABEL
        return v

    n_ev = np.zeros(len(categories), dtype=float)
    n_ne = np.zeros(len(categories), dtype=float)
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    for i in range(n_total):
        if nan_mask[i]:
            continue
        bucket = _bucket_of(x_arr.iloc[i])
        idx = cat_to_idx[bucket]
        w_i = 1.0 if weights is None else float(weights[i])
        if y[i] == 1:
            n_ev[idx] += w_i
        else:
            n_ne[idx] += w_i

    n_events_missing = 0
    n_nonevents_missing = 0
    if has_missing:
        bin_labels.append(MISSING_BIN_LABEL)
        if weights is None:
            n_events_missing = int(np.sum((y == 1) & nan_mask))
            n_nonevents_missing = int(np.sum((y == 0) & nan_mask))
        else:
            n_events_missing = int(
                round(float(np.sum(weights[(y == 1) & nan_mask])))
            )
            n_nonevents_missing = int(
                round(float(np.sum(weights[(y == 0) & nan_mask])))
            )

    ev_all = list(n_ev) + ([float(n_events_missing)] if has_missing else [])
    nev_all = list(n_ne) + ([float(n_nonevents_missing)] if has_missing else [])
    ev_arr = np.asarray(ev_all, dtype=float)
    nev_arr = np.asarray(nev_all, dtype=float)
    woe, iv = _compute_woe_and_iv(ev_arr, nev_arr)

    return BinningResult(
        column=column,
        kind="categorical",
        edges=(),
        categories=tuple(categories),
        bin_labels=tuple(bin_labels),
        has_missing_bin=has_missing,
        n_events=tuple(int(round(v)) for v in ev_arr),
        n_nonevents=tuple(int(round(v)) for v in nev_arr),
        woe=tuple(float(v) for v in woe),
        iv=iv,
    )


# ---------------------------------------------------------------------------
# Manual binning.
# ---------------------------------------------------------------------------


def _fit_manual(
    x: pd.Series,
    y: np.ndarray,
    *,
    weights: np.ndarray | None,
    column: str,
    edges: tuple[float, ...],
) -> BinningResult:
    y = np.asarray(y, dtype=float)
    _check_y(y)
    x_arr = np.asarray(pd.to_numeric(x, errors="coerce"), dtype=float)
    nan_mask = np.isnan(x_arr)
    has_missing = bool(nan_mask.any())
    x_clean = x_arr[~nan_mask]
    y_clean = y[~nan_mask]
    w_clean = None if weights is None else weights[~nan_mask]

    interior = np.asarray(edges, dtype=float)
    edges_with_inf = np.concatenate(([-np.inf], interior, [np.inf]))
    ev, nev = _events_per_bin_numeric(x_clean, y_clean, w_clean, interior)

    n_events_missing = 0
    n_nonevents_missing = 0
    if has_missing:
        y_missing = y[nan_mask]
        if weights is None:
            n_events_missing = int((y_missing == 1).sum())
            n_nonevents_missing = int((y_missing == 0).sum())
        else:
            w_missing = weights[nan_mask]
            n_events_missing = int(round(float(w_missing[y_missing == 1].sum())))
            n_nonevents_missing = int(round(float(w_missing[y_missing == 0].sum())))

    return _build_result_numeric(
        column=column,
        edges_with_inf=edges_with_inf,
        n_events=ev,
        n_nonevents=nev,
        has_missing=has_missing,
        n_events_missing=n_events_missing,
        n_nonevents_missing=n_nonevents_missing,
    )


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
        # fit_state may be the per-term state dict, or it may contain a
        # ``result`` key. Be permissive.
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
        # The training-time interior edges; clip against ±inf is implicit.
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
                # Unseen missing — route to bin 0 (neutral) with warning.
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
            # Unseen level and no RARE bucket. WoE = 0 → bin 0; warn.
            warnings.warn(
                f"column '{result.column}': unseen category {v!r} at predict time; "
                "assigning WoE=0 (neutral evidence). DESIGN.md §11.",
                stacklevel=3,
            )
            out[i] = 0
    return out


def _assign_woe(x: pd.Series, result: BinningResult) -> np.ndarray:
    """Return the WoE value per row, handling unseen categories specially.

    Unlike :func:`_assign_bin_index` (which routes unseen to bin 0), unseen
    categorical levels at predict time get :math:`\\mathrm{WoE} = 0`
    (neutral evidence; DESIGN.md §11).
    """
    if result.kind == "categorical":
        # We need a different path that yields 0 for unseen levels rather
        # than mapping them to bin 0's WoE.
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
