r"""Bootstrap resampling for fitted :class:`Solution` values.

AGENTS.md Task P3.C. DESIGN.md §3.2 (Bootstrap), ESL §7.11 + §8.2.

For models without closed-form standard errors — lasso, segmented models,
splines after selection — the bootstrap is the standard tool for
coefficient uncertainty (ESL §7.11). The :func:`bootstrap` function takes a
:class:`Solution` and returns a :class:`BootstrappedSolution` (see
:mod:`model_crafter.solution`) carrying the empirical coefficient
distribution, per-resample ``fit_state``, and the lasso ``selection_frequency``
diagnostic (ESL §3.4.3).

Three resampling schemes ship in v0:

* **Pairs bootstrap (ESL §8.2.1)** — the default. Each of ``n_boot``
  resamples is drawn by sampling ``n`` rows from ``data`` with replacement.
  When ``stratify`` is supplied the resampling is performed *within* each
  stratum so the marginal frequency of the strata is preserved. This is
  the right choice when :math:`X` is random (the credit-risk case).

  .. math::

      D^{(b)} = \{(x_i, y_i): i \in \mathrm{Multinomial}(n; 1/n, \ldots, 1/n)\}

* **Residual bootstrap (ESL §8.2.2)** — only meaningful for *fixed-X*
  regression with iid errors. Compute residuals
  :math:`e_i = y_i - x_i^\top \hat\beta`. For each resample, draw residuals
  :math:`e^{(b)}_i` with replacement and form
  :math:`y^{(b)}_i = x_i^\top \hat\beta + e^{(b)}_i`, then refit on
  ``(X, y^{(b)})``. Provided for completeness. ``method="pairs"`` is the
  safer choice when :math:`X` is random.

* **Block bootstrap** — when ``splitter=`` is supplied (e.g. a temporal
  splitter from P3.B), each resample concatenates contiguous *blocks*
  defined by the splitter's windows. This preserves the dependence
  structure of time-indexed data (Künsch 1989; López de Prado §7).

Coefficient CIs are returned by
:meth:`BootstrappedSolution.coefficient_ci`. The percentile method is
implemented; the BCa refinement is deferred to a future task and raises
:class:`NotImplementedError` with a docstring pointer (DESIGN.md §3.2).

Notes
-----
* **Failures per resample.** A bootstrap refit can hit
  :class:`~model_crafter.assumptions.AssumptionError` for a degenerate
  resample (e.g. a rank-deficient design when stratified sampling drops a
  column's variance). Such failures are caught and skipped; if the failure
  rate exceeds 5%, :func:`bootstrap` raises :class:`RuntimeError`.
* **Reproducibility.** ``random_state`` seeds a
  :class:`numpy.random.Generator`; identical seeds reproduce identical
  bootstrap draws to floating-point precision.
* **Weights.** ``weights`` is a column name in ``data``; the value is
  passed through to each refit's ``solve(..., weights=...)``. The
  *resampling* itself is uniform-over-rows (or uniform-within-stratum)
  unless ``splitter`` is given.
* **Parallelisation.** Single-threaded for v0; the solver dominates the
  wall clock. A future task can introduce a process-pool driver behind a
  ``n_jobs=`` argument.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from model_crafter.assumptions import AssumptionError
from model_crafter.solution import BootstrappedSolution, Solution
from model_crafter.solve import solve

__all__ = ["bootstrap"]


# Failure-rate threshold above which the bootstrap stops and raises. The
# 5% number is the documented contract in AGENTS.md P3.C.
_MAX_FAILURE_FRACTION = 0.05


def bootstrap(
    sol: Solution,
    data: pd.DataFrame,
    *,
    n_boot: int = 500,
    stratify: str | None = None,
    method: str = "pairs",
    weights: str | None = None,
    splitter: Any = None,
    random_state: int | None = None,
) -> BootstrappedSolution:
    r"""Bootstrap a fitted :class:`Solution` and return a :class:`BootstrappedSolution`.

    Parameters
    ----------
    sol
        The point-estimate :class:`Solution`. ``sol.spec`` is re-solved on
        every resample; the base solution is carried through unchanged on
        the returned :class:`BootstrappedSolution`.
    data
        Training data. Must be the same frame (or compatible) the base
        solution was fit on — the bootstrap re-fits the same spec against
        each resample.
    n_boot
        Number of bootstrap resamples. Conventional defaults: ``500`` is
        adequate for CIs on point coefficients; ``2000+`` is recommended for
        tight lasso selection-frequency estimates (DESIGN.md §11). Must be
        a positive integer.
    stratify
        Column name in ``data``. When set, resampling is performed within
        each stratum so the marginal frequency of strata is preserved
        across resamples (typical use: class label for an imbalanced
        binary classification problem).
    method
        ``"pairs"`` (default; ESL §8.2.1), ``"residual"`` (ESL §8.2.2),
        or — when ``splitter`` is set — automatically becomes ``"block"``
        regardless of this argument. Otherwise an unknown ``method`` is
        rejected with :class:`ValueError`.
    weights
        Column name in ``data`` to pass through as ``solve(..., weights=...)``
        for each refit. The *resampling* itself stays uniform per row;
        weighting the resampling is a different operation (left to a
        future task).
    splitter
        Optional object exposing ``.windows(data) -> Iterable[np.ndarray]``
        of integer block positions. When given, the bootstrap uses
        block resampling: each resample concatenates whole blocks drawn
        with replacement. Imported lazily so this module doesn't depend
        on P3.B being merged. See ESL §8 and López de Prado §7.
    random_state
        Seed for :func:`numpy.random.default_rng`. Same seed → identical
        results.

    Returns
    -------
    :class:`BootstrappedSolution`

    Raises
    ------
    ValueError
        On invalid arguments (``n_boot <= 0``, unknown ``method``,
        ``stratify`` not in ``data``).
    RuntimeError
        When more than 5% of the resamples fail to refit (DESIGN.md §9.8 —
        loud failures, not silent fallbacks).
    """
    if not isinstance(sol, Solution):
        raise TypeError(f"sol must be a Solution; got {type(sol).__name__}")
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"data must be a pandas DataFrame; got {type(data).__name__}"
        )
    if not isinstance(n_boot, int) or n_boot <= 0:
        raise ValueError(f"n_boot must be a positive int; got n_boot={n_boot!r}")
    if stratify is not None and stratify not in data.columns:
        raise KeyError(
            f"stratify column {stratify!r} not in data; "
            f"columns: {list(data.columns)}"
        )
    if weights is not None and not isinstance(weights, str):
        raise TypeError(
            "weights must be a column name (str) for the bootstrap "
            "(arrays would not survive resampling correctly); "
            f"got {type(weights).__name__}"
        )
    if weights is not None and weights not in data.columns:
        raise KeyError(
            f"weights column {weights!r} not in data; "
            f"columns: {list(data.columns)}"
        )

    # Determine method (splitter overrides explicit method).
    if splitter is not None:
        effective_method = "block"
    elif method == "pairs":
        effective_method = "pairs"
    elif method == "residual":
        effective_method = "residual"
    else:
        raise ValueError(
            f"unknown bootstrap method {method!r}; "
            "supported: 'pairs' (default; ESL §8.2.1), "
            "'residual' (ESL §8.2.2; fixed-X), "
            "or pass splitter= for block bootstrap"
        )

    rng = np.random.default_rng(random_state)

    # Resample-generator: yields a fresh DataFrame for each iteration.
    if effective_method == "pairs":
        resamples = _pairs_resamples(data, n_boot, stratify, rng)
    elif effective_method == "residual":
        resamples = _residual_resamples(sol, data, n_boot, rng)
    else:  # "block"
        resamples = _block_resamples(data, splitter, n_boot, rng)

    coefs_rows: list[np.ndarray] = []
    fit_states: list[Mapping[str, Any]] = []
    n_failures = 0
    n_attempted = 0

    base_cols = list(sol.design_columns)
    spec = sol.spec

    for resample_df in resamples:
        n_attempted += 1
        try:
            sol_b = solve(spec, resample_df, weights=weights)
        except AssumptionError:
            n_failures += 1
            # Short-circuit if we exceed the documented threshold.
            if n_failures > max(1, int(_MAX_FAILURE_FRACTION * n_boot)):
                raise RuntimeError(
                    "bootstrap aborted: more than "
                    f"{_MAX_FAILURE_FRACTION:.0%} of resamples failed to "
                    f"refit ({n_failures}/{n_attempted}). The base spec may "
                    "be too close to rank-deficient; consider adding a "
                    "small ridge penalty (ESL §3.4.1) before bootstrapping."
                ) from None
            continue

        if list(sol_b.design_columns) != base_cols:
            # Degenerate: a resample produced a different design shape.
            # Treat as a failure (don't bias the distribution).
            n_failures += 1
            if n_failures > max(1, int(_MAX_FAILURE_FRACTION * n_boot)):
                raise RuntimeError(
                    "bootstrap aborted: resamples produced inconsistent "
                    f"design columns vs the base solution "
                    f"({n_failures}/{n_attempted} failures)."
                )
            continue

        coefs_rows.append(np.asarray(sol_b.coefficients.reindex(base_cols)))
        fit_states.append(sol_b.fit_state)

    n_success = len(coefs_rows)
    if n_success == 0:
        raise RuntimeError(
            "bootstrap aborted: every resample failed to refit. "
            "Check the spec / data — see notes/P3.C.md for the failure-handling policy."
        )

    coefficients_dist = pd.DataFrame(
        np.vstack(coefs_rows),
        columns=pd.Index(base_cols),
    )

    selection_frequency = _compute_selection_frequency(
        coefficients_dist, sol, base_cols
    )

    return BootstrappedSolution(
        base=sol,
        coefficients_dist=coefficients_dist,
        fit_state_dist=tuple(fit_states),
        selection_frequency=selection_frequency,
        n_boot=n_success,
        method=effective_method,
    )


# Resample generators


def _pairs_resamples(
    data: pd.DataFrame,
    n_boot: int,
    stratify: str | None,
    rng: np.random.Generator,
) -> Iterable[pd.DataFrame]:
    r"""Yield ``n_boot`` pairs-bootstrap resamples (ESL §8.2.1).

    Each iterate draws ``n = len(data)`` row positions with replacement; if
    ``stratify`` is set, the draw is performed within each stratum so the
    marginal stratum counts are preserved.
    """
    n = len(data)
    if stratify is None:
        for _ in range(n_boot):
            idx = rng.integers(low=0, high=n, size=n)
            # Use .iloc to drop the original index, then reset to be safe.
            yield data.iloc[idx].reset_index(drop=True)
    else:
        # Pre-compute the per-stratum index arrays once.
        stratum_indices: dict[Any, np.ndarray] = {}
        for key, sub in data.groupby(stratify, sort=False):
            stratum_indices[key] = np.asarray(sub.index, dtype=np.int64)
        # Determine the integer positions per stratum (positions, not labels).
        # We need positions because we'll .iloc back into ``data``.
        pos_of_label = pd.Series(np.arange(n, dtype=np.int64), index=data.index)
        per_stratum_positions: dict[Any, np.ndarray] = {
            key: pos_of_label.reindex(idx).to_numpy(dtype=np.int64)
            for key, idx in stratum_indices.items()
        }
        for _ in range(n_boot):
            chunks: list[np.ndarray] = []
            for positions in per_stratum_positions.values():
                k = positions.shape[0]
                draws = rng.integers(low=0, high=k, size=k)
                chunks.append(positions[draws])
            full = np.concatenate(chunks)
            yield data.iloc[full].reset_index(drop=True)


def _residual_resamples(
    sol: Solution,
    data: pd.DataFrame,
    n_boot: int,
    rng: np.random.Generator,
) -> Iterable[pd.DataFrame]:
    r"""Yield ``n_boot`` residual-bootstrap resamples (ESL §8.2.2).

    Computes residuals once: :math:`e_i = y_i - \hat y_i`. Each iterate
    yields a frame with the same design columns as ``data`` but with the
    target column replaced by :math:`\hat y_i + e^{(b)}_i` where the
    bootstrap residuals are drawn with replacement.
    """
    spec = sol.spec
    target = spec.target

    # Use predict at the base solution to get y_hat. ``predict`` returns a
    # Series aligned with ``data.index``.
    from model_crafter.solve import predict

    y_hat = np.asarray(predict(sol, data), dtype=float)
    y = np.asarray(data[target], dtype=float)
    e = y - y_hat
    # Centre residuals so the bootstrap distribution has the correct mean
    # (Efron & Tibshirani 1993, §9.5 — required for valid residual bootstrap).
    e_centred = e - e.mean()
    n = len(data)
    for _ in range(n_boot):
        draws = rng.integers(low=0, high=n, size=n)
        e_boot = e_centred[draws]
        y_boot = y_hat + e_boot
        frame = data.copy()
        frame[target] = y_boot
        yield frame.reset_index(drop=True)


def _block_resamples(
    data: pd.DataFrame,
    splitter: Any,
    n_boot: int,
    rng: np.random.Generator,
) -> Iterable[pd.DataFrame]:
    r"""Yield ``n_boot`` block-bootstrap resamples from ``splitter``'s windows.

    The splitter contract is loose by design: ``splitter.windows(data)``
    must yield iterables of integer positions defining contiguous blocks.
    This module imports the splitter object only by duck-typing, so the
    bootstrap doesn't depend on P3.B being merged.
    """
    if not hasattr(splitter, "windows"):
        raise TypeError(
            "splitter must expose a .windows(data) method yielding integer "
            f"position arrays; got {type(splitter).__name__}"
        )
    blocks: list[np.ndarray] = [
        np.asarray(w, dtype=np.int64) for w in splitter.windows(data)
    ]
    if not blocks:
        raise ValueError(
            "splitter produced zero windows; block bootstrap requires at "
            "least one window."
        )
    n_blocks = len(blocks)
    for _ in range(n_boot):
        draws = rng.integers(low=0, high=n_blocks, size=n_blocks)
        sampled = np.concatenate([blocks[i] for i in draws])
        yield data.iloc[sampled].reset_index(drop=True)


# Selection frequency


def _compute_selection_frequency(
    coefficients_dist: pd.DataFrame,
    sol: Solution,
    base_cols: list[str],
) -> pd.Series:
    r"""Fraction of resamples where each coefficient was non-zero.

    For lasso fits (ESL §3.4.3) this is the standard diagnostic of
    selection stability under collinearity. For non-lasso fits exact zeros
    are vanishingly rare; in that case ``selection_frequency`` is all 1.0
    by construction. The dispatch sniffs for the presence of an
    :class:`~model_crafter.penalty.L1Penalty` anywhere in the spec's
    penalty (including elastic-net sums).
    """
    from model_crafter.penalty import L1Penalty, PenaltySum

    penalty = sol.spec.penalty
    has_l1 = isinstance(penalty, L1Penalty) or (
        isinstance(penalty, PenaltySum)
        and any(isinstance(p, L1Penalty) for p in penalty.parts)
    )
    if not has_l1:
        return pd.Series(
            np.ones(len(base_cols), dtype=float),
            index=pd.Index(base_cols),
            name="selection_frequency",
        )

    # Exact zero: lasso produces exact zeros by construction. Use absolute
    # comparison to zero — no tolerance — because soft-thresholded
    # coordinate descent emits exact zeros (DESIGN.md §3.2 / ESL §3.4.3).
    arr = coefficients_dist.to_numpy(dtype=float)
    nonzero_frac = (arr != 0.0).mean(axis=0)
    return pd.Series(
        nonzero_frac,
        index=pd.Index(base_cols),
        name="selection_frequency",
    )
