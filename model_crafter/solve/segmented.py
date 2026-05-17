r"""Per-segment solve + predict for :class:`SegmentedSpec` (Phase 6).

DESIGN.md §3.4 pins the contract: a :class:`SegmentedSpec` groups data by
its ``by`` column, applies the base spec to each segment independently, and
bundles the per-segment :class:`Solution`\ s into a
:class:`SegmentedSolution`. ESL §3.7 is the methodological reference — the
segmented model is conceptually a piecewise-constant interaction with the
segment column where the per-segment coefficient vector is free.

The dispatch in :mod:`model_crafter.solve.__init__` calls
:func:`solve_segmented` when ``isinstance(spec, SegmentedSpec)`` and
:func:`predict_segmented` from :func:`predict` when given a
:class:`SegmentedSolution`. Importing the parent ``solve`` package triggers
this module's import; the only public symbols are the two functions, which
are intentionally not re-exported from :mod:`model_crafter.solve` (the
single entry points are ``solve`` and ``predict``).

Notes on routing semantics
--------------------------
* Segments are produced by ``data.groupby(spec.by, observed=True,
  sort=False)``. ``observed=True`` skips empty Categorical levels — the
  same convention used in
  :func:`model_crafter.performance.by_segment.performance_by_segment` so
  upstream code can mix the two with consistent expectations.
* Empty segments after groupby are silently dropped.
* At predict time, a row whose segment value is not in
  ``sol.segments`` (an unseen segment) receives ``NaN`` and a warning
  naming the unseen segment values and the number of affected rows.
* Segment keys on the returned :class:`SegmentedSolution` are stringified
  for stability across pandas/numpy scalar types (``np.int64(1) → '1'``).
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from model_crafter.solution import SegmentedSolution, Solution
from model_crafter.spec import SegmentedSpec


def _stringify_key(key: Any) -> str:
    """Coerce a groupby key to a string (mirrors performance.by_segment)."""
    return str(key)


def _needs_binning_fit(base: Any) -> bool:
    """True iff ``base.features`` contains an unfitted WoE / binned term.

    A WoE / binned term carries its learned ``BinningResult`` on
    ``.fitted``; an unfitted instance has ``fitted is None``. The check
    is silent when no WoE terms are present.
    """
    try:
        from model_crafter.terms.woe import BinnedTerm, WoETerm  # noqa: PLC0415
    except ImportError:  # pragma: no cover - circular safety
        return False
    for term in getattr(base, "features", ()) or ():
        if isinstance(term, (WoETerm, BinnedTerm)) and getattr(term, "fitted", None) is None:
            return True
    return False


def solve_segmented(
    spec: SegmentedSpec,
    data: pd.DataFrame,
    *,
    weights: str | np.ndarray | pd.Series | None = None,
    method: str | None = None,
    on_violation: str = "raise",
    suppress: tuple = (),
    classical_inference: bool = False,
    stability_splitter: Any = None,
) -> SegmentedSolution:
    r"""Fit ``spec.base`` independently within each ``spec.by`` segment.

    The signature mirrors :func:`model_crafter.solve.solve` exactly — every
    keyword argument is passed through to each per-segment ``solve`` call,
    so the assumption configuration (``on_violation``, ``suppress``,
    ``classical_inference``, ``stability_splitter``) applies uniformly
    across segments. Weights are resolved per segment from the column
    name; an array/Series of weights is sliced by the segment row index.

    Parameters
    ----------
    spec
        A :class:`SegmentedSpec`. The ``by`` column must exist in ``data``.
    data
        Training data. Must contain ``spec.by``, ``spec.base.target``, and
        every column required by ``spec.base.features``.
    weights
        Sample weights — column name (str), 1-D array, :class:`pandas.Series`,
        or ``None`` (uniform). When a non-string array is supplied it must
        be aligned with ``data.index``; the per-segment slice indexes into
        it positionally to preserve row alignment.
    method
        Optional method override; passed through unchanged.
    on_violation, suppress, classical_inference, stability_splitter
        Pass-through to each per-segment :func:`solve`. See
        :func:`model_crafter.solve.solve` for the semantics.

    Returns
    -------
    SegmentedSolution
        Frozen value bundling per-segment :class:`Solution`\ s.

    Raises
    ------
    KeyError
        If ``spec.by`` is not in ``data.columns``.
    ValueError
        If the segmentation yields no segments after dropping empties.
    """
    # Deferred import to avoid the ``solve/__init__`` import cycle: this
    # module is imported during ``solve/__init__`` evaluation.
    from model_crafter.solve import solve  # noqa: PLC0415

    if not isinstance(spec, SegmentedSpec):
        raise TypeError(
            f"spec must be a SegmentedSpec; got {type(spec).__name__}"
        )
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"data must be a pandas DataFrame; got {type(data).__name__}"
        )
    if spec.by not in data.columns:
        raise KeyError(
            f"segmentation column {spec.by!r} not in data "
            f"(columns: {list(data.columns)})"
        )

    # Resolve a non-string weights= once so we can slice it per segment.
    # String weights= is left as-is and the inner solve() resolves it from
    # the per-segment frame, which is the same column with the same name.
    array_weights: np.ndarray | None = None
    if weights is not None and not isinstance(weights, str):
        array_weights = np.asarray(weights, dtype=float)
        if array_weights.ndim != 1 or array_weights.shape[0] != len(data):
            raise ValueError(
                f"weights array must be 1-D with length {len(data)}; "
                f"got shape {array_weights.shape}"
            )

    # Detect unfitted WoE / binned terms — these need per-segment
    # ``fit_binnings`` before the inner ``solve``. This keeps the "single
    # declarative spec" promise of DESIGN.md §3.4: callers do not have to
    # pre-fit bins themselves when segmenting.
    base_needs_binning = _needs_binning_fit(spec.base)

    segments: dict[str, Solution] = {}
    n_obs_total = 0

    grouped = data.groupby(spec.by, observed=True, sort=False)
    for key, slice_df in grouped:
        if len(slice_df) == 0:
            # Defensive — observed=True usually handles this.
            continue
        seg_key = _stringify_key(key)

        # Per-segment weights: slice the resolved array by the original
        # positional indexer if we have one; otherwise pass the column
        # name through (the inner solve will look it up in slice_df).
        seg_weights: str | np.ndarray | pd.Series | None
        if array_weights is not None:
            # Get the positional indices of the segment in the parent.
            pos = data.index.get_indexer(slice_df.index)
            seg_weights = array_weights[pos]
        else:
            seg_weights = weights

        seg_base = spec.base
        if base_needs_binning:
            # Each segment learns its own bins on its own slice — this is
            # exactly the §3.4 motivation (segmentation as a coarse
            # interaction with the segment column).
            from model_crafter.terms.woe import fit_binnings  # noqa: PLC0415

            seg_base = fit_binnings(spec.base, slice_df)

        sub_sol = solve(
            seg_base,
            slice_df,
            weights=seg_weights,
            method=method,
            on_violation=on_violation,
            suppress=suppress,
            classical_inference=classical_inference,
            stability_splitter=stability_splitter,
        )
        segments[seg_key] = sub_sol
        n_obs_total += sub_sol.n_obs

    if not segments:
        raise ValueError(
            f"segmentation produced zero non-empty segments for "
            f"by={spec.by!r}"
        )

    return SegmentedSolution(
        spec=spec, segments=segments, n_obs=n_obs_total
    )


def predict_segmented(
    sol: SegmentedSolution, new_data: pd.DataFrame
) -> pd.Series:
    r"""Apply a :class:`SegmentedSolution` to ``new_data``.

    Each row of ``new_data`` is routed to the per-segment
    :class:`Solution` whose key matches the row's segment value (after
    stringification). The merged predictions are returned as a
    :class:`pandas.Series` aligned with ``new_data.index``.

    Rows whose segment is **not** present in ``sol.segments`` (unseen at
    fit time) get ``NaN`` and a warning naming the unseen segments and
    the count of affected rows. This matches the spec's "loud failures,
    not silent fallbacks" stance (DESIGN.md §9.8) while keeping predict
    a total function on aligned input.

    Parameters
    ----------
    sol
        A :class:`SegmentedSolution` produced by :func:`solve_segmented`.
    new_data
        Frame whose rows define the prediction points. Must contain the
        segmentation column and every column required by the per-segment
        terms.

    Returns
    -------
    pandas.Series
        Indexed by ``new_data.index``, named after the base target.
    """
    from model_crafter.solve import predict  # noqa: PLC0415

    if not isinstance(sol, SegmentedSolution):
        raise TypeError(
            f"sol must be a SegmentedSolution; got {type(sol).__name__}"
        )
    if not isinstance(new_data, pd.DataFrame):
        raise TypeError(
            f"new_data must be a pandas DataFrame; got "
            f"{type(new_data).__name__}"
        )
    by = sol.spec.by
    if by not in new_data.columns:
        raise KeyError(
            f"segmentation column {by!r} not in new_data "
            f"(columns: {list(new_data.columns)})"
        )

    target_name = sol.spec.base.target
    # Start filled with NaN so unseen segments leave NaN naturally.
    out = pd.Series(
        np.nan, index=new_data.index, dtype=float, name=target_name
    )

    seen_segments = set(sol.segments.keys())
    unseen_counts: dict[str, int] = {}

    grouped = new_data.groupby(by, observed=True, sort=False)
    for key, slice_df in grouped:
        seg_key = _stringify_key(key)
        if seg_key not in seen_segments:
            unseen_counts[seg_key] = unseen_counts.get(seg_key, 0) + len(slice_df)
            continue
        sub_sol = sol.segments[seg_key]
        sub_pred = predict(sub_sol, slice_df)
        out.loc[slice_df.index] = sub_pred.to_numpy(dtype=float)

    if unseen_counts:
        total = sum(unseen_counts.values())
        names = sorted(unseen_counts.keys())
        warnings.warn(
            f"predict_segmented: {total} row(s) belong to segments not "
            f"present at fit time; their predictions are NaN. "
            f"Unseen segments: {names}",
            stacklevel=2,
        )

    return out
