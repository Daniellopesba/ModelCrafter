r"""Hyperparameter tuning over a CV splitter (AGENTS.md Task P3.B,
DESIGN.md §3.2, §7.10).

The package separates **CV for assessment** from **CV for tuning**, per
DESIGN.md §3.2. :func:`~model_crafter.validation.cross_validate.cross_validate`
is the assessment runner; :func:`tune` is the tuning runner. The honest
assessment-after-tuning pattern is :func:`nested_cv` (ESL §7.10.2).

Math
----
For a grid :math:`\{\lambda_g\}_{g=1}^G`, splitter folds
:math:`(T_k, V_k)_{k=1}^K`, and metric :math:`m`, :func:`tune` computes

.. math::

   \mu_g = \frac{1}{K}\sum_{k=1}^K m\bigl(\hat\beta(T_k, \lambda_g), V_k\bigr),
   \qquad
   \sigma_g = \mathrm{SD}_k m\bigl(\hat\beta(T_k, \lambda_g), V_k\bigr).

A selection ``rule`` picks one :math:`\lambda_g` from the curve; the
default is :func:`best_mean` (argmax / argmin of :math:`\mu_g`). ESL
§7.10's :func:`one_se_rule` picks the **simplest** model whose
:math:`\mu_g` is within one standard error of the best.

For regularised linear models in this package, "simpler" means **larger
lambda** — the grid orientation is the convention; see
:func:`one_se_rule` for how to override.
"""

from __future__ import annotations

import html
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from model_crafter.solve import solve
from model_crafter.validation.cross_validate import cross_validate

# Result dataclasses


@dataclass(frozen=True, slots=True)
class TuneResult:
    """The output of :func:`tune` (DESIGN.md §3.2).

    Attributes
    ----------
    best_param
        The grid value chosen by the selection rule.
    cv_curve
        DataFrame indexed by grid value with columns ``metric_mean`` and
        ``metric_sd`` (the per-fold mean and standard deviation of the
        metric).
    solution
        The :class:`~model_crafter.solution.Solution` refit on the full
        data at ``best_param`` (DESIGN.md §3.2: "spec refit on the full
        data at best_param").
    direction
        ``"maximize"`` or ``"minimize"`` — the orientation of the metric.
    """

    best_param: Any
    cv_curve: pd.DataFrame
    solution: Any
    direction: str = "maximize"

    def _repr_html_(self) -> str:
        curve_html = (
            self.cv_curve.to_html(border=0, classes="mc-tune-result")
            if not self.cv_curve.empty
            else "<p>(empty CV curve)</p>"
        )
        return (
            "<div class='mc-tune-result'>"
            "<strong>TuneResult</strong> "
            f"<span>best_param={html.escape(str(self.best_param))}, "
            f"direction={html.escape(self.direction)}</span>"
            "<h4>CV curve</h4>"
            f"{curve_html}"
            "</div>"
        )


@dataclass(frozen=True, slots=True)
class NestedCVResult:
    """The output of :func:`nested_cv` (DESIGN.md §3.2).

    Attributes
    ----------
    outer_metric
        DataFrame with one row per outer fold: the metric value on the
        outer test set, plus the chosen inner param.
    best_params
        Tuple of chosen inner params across outer folds (stability
        diagnostic).
    inner_curves
        Tuple of inner-CV curves (one per outer fold).
    """

    outer_metric: pd.DataFrame
    best_params: tuple
    inner_curves: tuple

    def _repr_html_(self) -> str:
        outer_html = (
            self.outer_metric.to_html(border=0, classes="mc-nested-cv-result")
            if not self.outer_metric.empty
            else "<p>(no outer folds)</p>"
        )
        params_str = ", ".join(html.escape(str(p)) for p in self.best_params)
        return (
            "<div class='mc-nested-cv-result'>"
            "<strong>NestedCVResult</strong> "
            f"<span>n_outer_folds={len(self.outer_metric)}, "
            f"best_params=[{params_str}]</span>"
            "<h4>Outer-fold metrics</h4>"
            f"{outer_html}"
            "</div>"
        )


# Selection rules


def best_mean(curve: pd.DataFrame, direction: str) -> Any:
    r"""Pick the grid value with the best mean metric.

    Parameters
    ----------
    curve
        Indexed by grid value; column ``metric_mean`` is the per-fold
        mean (one row per grid value).
    direction
        ``"maximize"`` or ``"minimize"``.

    Returns
    -------
    The grid value at the best ``metric_mean``.
    """
    if direction not in ("maximize", "minimize"):
        raise ValueError(
            f"direction must be 'maximize' or 'minimize'; got {direction!r}"
        )
    if "metric_mean" not in curve.columns:
        raise KeyError("curve must contain a 'metric_mean' column")
    if direction == "maximize":
        return curve["metric_mean"].idxmax()
    return curve["metric_mean"].idxmin()


def one_se_rule(curve: pd.DataFrame, direction: str) -> Any:
    r"""ESL §7.10's one-standard-error rule.

    Among grid values whose mean is within one standard error of the
    best, pick the **simplest**. The default notion of "simpler" is
    larger grid value (the lambda convention for L1/L2 penalties);
    callers can override by sorting ``curve`` in their preferred
    parsimony order — :func:`one_se_rule` walks the index in its given
    order and picks the **last** eligible value.

    Parameters
    ----------
    curve
        Indexed by grid value; columns ``metric_mean`` and ``metric_sd``.
    direction
        ``"maximize"`` (default for AUC, R^2) or ``"minimize"`` (MSE,
        log-loss).

    Returns
    -------
    The most parsimonious grid value within one SE of the best.
    """
    if direction not in ("maximize", "minimize"):
        raise ValueError(
            f"direction must be 'maximize' or 'minimize'; got {direction!r}"
        )
    for col in ("metric_mean", "metric_sd"):
        if col not in curve.columns:
            raise KeyError(f"curve must contain a {col!r} column")
    if direction == "maximize":
        best_idx = curve["metric_mean"].idxmax()
        best = curve["metric_mean"].max()
        threshold = best - float(curve.loc[best_idx, "metric_sd"])
        eligible = curve[curve["metric_mean"] >= threshold]
    else:
        best_idx = curve["metric_mean"].idxmin()
        best = curve["metric_mean"].min()
        threshold = best + float(curve.loc[best_idx, "metric_sd"])
        eligible = curve[curve["metric_mean"] <= threshold]
    if eligible.empty:
        return best_idx
    # "Simpler" = larger grid value (the lambda convention).
    return eligible.index.max()


# tune


def tune(
    spec_fn: Callable[[Any], Any],
    grid: Iterable,
    data: pd.DataFrame,
    splitter: Any,
    metric: Callable,
    weights: str | np.ndarray | pd.Series | None = None,
    rule: Callable[[pd.DataFrame, str], Any] | None = None,
    direction: str = "maximize",
) -> TuneResult:
    r"""Tune ``spec_fn`` across ``grid`` via CV on ``splitter``.

    For each grid value, builds the spec with ``spec_fn(value)`` and
    runs :func:`cross_validate` across the splitter folds. Aggregates
    per-fold metric values into ``mean ± SD`` per grid value, picks one
    grid value via ``rule``, and refits the spec on the full data at
    the chosen value.

    Parameters
    ----------
    spec_fn
        Callable ``spec_fn(value) -> LinearSpec``.
    grid
        Iterable of grid values (e.g., lambdas from
        :func:`~model_crafter.validation.lambda_path.log_grid`).
    data
        Full input frame.
    splitter
        Any :class:`~model_crafter.validation.splitters.Splitter`.
    metric
        A single metric callable
        ``metric(sol, data, weights=None) -> ResultLike``. Must match
        ``direction``.
    weights
        Sample weights.
    rule
        Selection rule; default :func:`best_mean`. Use :func:`one_se_rule`
        for ESL §7.10's parsimony heuristic.
    direction
        ``"maximize"`` (default, for AUC / R^2) or ``"minimize"``
        (MSE / log-loss).

    Returns
    -------
    TuneResult
        Frozen value with ``best_param``, ``cv_curve``, and a refit
        :class:`~model_crafter.solution.Solution` on the full data.
    """
    if not callable(spec_fn):
        raise TypeError(f"spec_fn must be callable; got {type(spec_fn).__name__}")
    grid_list = list(grid)
    if not grid_list:
        raise ValueError("grid must be a non-empty iterable")
    if rule is None:
        rule = best_mean
    if direction not in ("maximize", "minimize"):
        raise ValueError(
            f"direction must be 'maximize' or 'minimize'; got {direction!r}"
        )

    rows: list[dict] = []
    metric_name = getattr(metric, "name", None) or getattr(
        metric, "__name__", "metric"
    )
    for value in grid_list:
        spec = spec_fn(value)
        cv = cross_validate(
            spec, data, splitter, metrics=[metric], weights=weights
        )
        per_fold = [fold["metrics"][metric_name] for fold in cv.fold_results]
        per_fold_arr = np.asarray(per_fold, dtype=float)
        mean = float(per_fold_arr.mean())
        sd = (
            float(per_fold_arr.std(ddof=1))
            if per_fold_arr.size > 1
            else 0.0
        )
        rows.append(
            {
                "param": value,
                "metric_mean": mean,
                "metric_sd": sd,
            }
        )

    cv_curve = pd.DataFrame(rows).set_index("param")
    best_param = rule(cv_curve, direction)

    # Refit on the full data at best_param.
    refit_spec = spec_fn(best_param)
    refit_solution = solve(refit_spec, data, weights=weights)
    return TuneResult(
        best_param=best_param,
        cv_curve=cv_curve,
        solution=refit_solution,
        direction=direction,
    )


# nested_cv — assessment-after-tuning (ESL §7.10.2)


def nested_cv(
    spec_fn: Callable[[Any], Any],
    grid: Iterable,
    data: pd.DataFrame,
    outer_splitter: Any,
    inner_splitter: Any,
    metric: Callable,
    weights: str | np.ndarray | pd.Series | None = None,
    rule: Callable[[pd.DataFrame, str], Any] | None = None,
    direction: str = "maximize",
) -> NestedCVResult:
    r"""Nested cross-validation (DESIGN.md §3.2, ESL §7.10.2).

    For each outer fold ``(T_outer, V_outer)``:

    1. Run :func:`tune` on ``T_outer`` with ``inner_splitter`` and
       collect the chosen inner param.
    2. Refit the spec at the chosen param on ``T_outer``.
    3. Score the refit on ``V_outer`` — that's the honest held-out
       metric for the outer fold.

    The aggregate of those outer-fold metrics is the *honest assessment*
    of the spec family — it accounts for the optimism that arises when
    the same data are used for both tuning and assessment
    (ESL §7.10.2). Each outer fold's chosen param is recorded for
    stability diagnostics.

    Parameters
    ----------
    spec_fn, grid, metric, weights, rule, direction
        As in :func:`tune`.
    outer_splitter
        Splitter for the outer assessment loop.
    inner_splitter
        Splitter applied to each outer training fold for tuning.

    Returns
    -------
    NestedCVResult
    """
    if not callable(spec_fn):
        raise TypeError("spec_fn must be callable")
    grid_list = list(grid)
    if not grid_list:
        raise ValueError("grid must be a non-empty iterable")
    if rule is None:
        rule = best_mean
    metric_name = getattr(metric, "name", None) or getattr(
        metric, "__name__", "metric"
    )

    outer_rows: list[dict] = []
    chosen_params: list = []
    inner_curves: list[pd.DataFrame] = []
    time_col = getattr(outer_splitter, "time_col", None)

    for k, (train_outer, valid_outer) in enumerate(outer_splitter.split(data)):
        inner_result = tune(
            spec_fn=spec_fn,
            grid=grid_list,
            data=train_outer,
            splitter=inner_splitter,
            metric=metric,
            weights=_subset_weights(weights, data, train_outer),
            rule=rule,
            direction=direction,
        )
        chosen_params.append(inner_result.best_param)
        inner_curves.append(inner_result.cv_curve.copy())

        # Refit at the chosen inner param on the outer train, score on outer test.
        refit_spec = spec_fn(inner_result.best_param)
        refit_solution = solve(
            refit_spec,
            train_outer,
            weights=_subset_weights(weights, data, train_outer),
        )
        valid_weights = _subset_weights(weights, data, valid_outer)
        score = _call_metric(metric, refit_solution, valid_outer, valid_weights)

        if time_col is not None:
            row = {
                "fold": k,
                "train_start": pd.to_datetime(train_outer[time_col]).min(),
                "train_end": pd.to_datetime(train_outer[time_col]).max(),
                "valid_start": pd.to_datetime(valid_outer[time_col]).min(),
                "valid_end": pd.to_datetime(valid_outer[time_col]).max(),
                "best_param": inner_result.best_param,
                metric_name: score,
            }
        else:
            row = {
                "fold": k,
                "best_param": inner_result.best_param,
                metric_name: score,
            }
        outer_rows.append(row)

    outer_metric = pd.DataFrame(outer_rows)
    return NestedCVResult(
        outer_metric=outer_metric,
        best_params=tuple(chosen_params),
        inner_curves=tuple(inner_curves),
    )


# Helpers


def _subset_weights(
    weights: str | np.ndarray | pd.Series | None,
    full_data: pd.DataFrame,
    subset: pd.DataFrame,
) -> str | np.ndarray | None:
    """Slice ``weights`` to match ``subset``.

    A string passes through (each downstream call resolves the column on
    the subset). Arrays/Series are reindexed via ``subset.index``.
    """
    if weights is None:
        return None
    if isinstance(weights, str):
        return weights
    arr = np.asarray(weights, dtype=float)
    if arr.shape[0] != len(full_data):
        raise ValueError(
            f"weights length {arr.shape[0]} != data length {len(full_data)}"
        )
    s = pd.Series(arr, index=full_data.index)
    return s.reindex(subset.index).to_numpy(dtype=float)


def _call_metric(
    fn: Callable,
    sol: Any,
    data: pd.DataFrame,
    weights: str | np.ndarray | None,
) -> float:
    """Same duck-typed metric call as cross_validate."""
    try:
        out = fn(sol, data, weights=weights)
    except TypeError:
        out = fn(sol, data)
    if hasattr(out, "value"):
        return float(out.value)
    return float(out)


# Re-export ``Sequence`` to satisfy unused-import linters in earlier drafts.
_ = Sequence
