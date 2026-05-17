r"""``cross_validate`` — held-out evaluation (DESIGN.md §3.2, AGENTS.md P3.B).

``cross_validate(spec, data, splitter, metrics, weights=None)`` evaluates a
single :class:`~model_crafter.spec.LinearSpec` across a splitter's folds and
returns a :class:`CVResult`. The runner is for *assessment only* — tuning
across multiple specs is :func:`model_crafter.validation.tune.tune` (DESIGN.md
§3.2: "CV for ASSESSMENT vs CV for TUNING").

Math
----
For folds :math:`(T_k, V_k)_{k=1}^K` and metrics
:math:`m_1, \dots, m_M`, the runner computes

.. math::

    \hat\beta_k &= \arg\min_\beta\ \mathcal{L}(y_{T_k},\, X_{T_k}\beta)
                    + R(\beta) \\
    m_{ik}     &= m_i(\hat\beta_k, V_k)

and bundles the table of :math:`m_{ik}` plus the fold solutions on
:class:`CVResult`. Per DESIGN.md §3.2 the runner also enforces
:class:`~model_crafter.assumptions.NoTemporalLeakage` (HARD) — a leaky CV
partition reports performance the model will not achieve in production.

When the spec's ``loss`` declares a ``label_horizon`` attribute and the
splitter's ``gap`` is zero, the runner refuses to start (DESIGN.md §3.2:
a 12-month default label can't be observed until ``t + 365D``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from model_crafter.assumptions import (
    AssumptionError,
    AssumptionReport,
    NoTemporalLeakage,
)
from model_crafter.solution import Solution
from model_crafter.solve import solve

__all__ = ["CVResult", "cross_validate"]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CVResult:
    """The output of :func:`cross_validate` (DESIGN.md §3.2).

    Attributes
    ----------
    fold_results
        One dict per fold with keys:
        ``train_period`` — ``(min_t, max_t)`` of the training window,
        ``valid_period`` — ``(min_t, max_t)`` of the validation window,
        ``metrics`` — dict mapping metric name to its float value on the
        validation window,
        ``solution`` — the :class:`Solution` fitted on the training fold,
        ``train`` / ``valid`` — the actual fold frames (carried so the
        ``NoTemporalLeakage`` check can audit them after the fact).
    solutions
        Convenience tuple of fold solutions (one per fold).
    splitter
        The splitter used to produce the folds.
    assumptions
        :class:`AssumptionReport` containing at minimum the
        :class:`NoTemporalLeakage` check.
    """

    fold_results: tuple[dict, ...]
    solutions: tuple[Solution, ...]
    splitter: Any = None
    assumptions: AssumptionReport | None = None
    # Internal: list of (train, valid) pairs, for the NoTemporalLeakage
    # assumption to walk after the fact (DESIGN.md §3.2).
    folds: tuple[tuple[pd.DataFrame, pd.DataFrame], ...] = field(default=())

    def summary(self) -> pd.DataFrame:
        r"""Return one row per fold with the per-metric values.

        The columns are the metric names; additional columns
        ``train_start``, ``train_end``, ``valid_start``, ``valid_end``
        document each fold's time windows.
        """
        rows = []
        for fr in self.fold_results:
            row: dict[str, Any] = {}
            train_period = fr.get("train_period")
            valid_period = fr.get("valid_period")
            if train_period is not None:
                row["train_start"] = train_period[0]
                row["train_end"] = train_period[1]
            if valid_period is not None:
                row["valid_start"] = valid_period[0]
                row["valid_end"] = valid_period[1]
            for name, value in (fr.get("metrics") or {}).items():
                row[name] = value
            rows.append(row)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------


_ZERO_TD: pd.Timedelta = pd.Timedelta(0)  # type: ignore[assignment]


def _coerce_timedelta(value: Any) -> pd.Timedelta:
    """Coerce ``value`` to a :class:`pd.Timedelta`, treating ``None`` / NaT
    as zero."""
    if value is None:
        return _ZERO_TD
    if isinstance(value, pd.Timedelta):
        return value
    parsed = pd.Timedelta(value)
    if not isinstance(parsed, pd.Timedelta):
        return _ZERO_TD
    return parsed


def cross_validate(
    spec: Any,
    data: pd.DataFrame,
    splitter: Any,
    metrics: Iterable[Callable] | Callable,
    weights: str | np.ndarray | pd.Series | None = None,
) -> CVResult:
    r"""Cross-validated assessment of a single spec on data.

    Parameters
    ----------
    spec
        A :class:`~model_crafter.spec.LinearSpec`. ``cross_validate`` is
        assessment-only — it does NOT iterate over multiple specs.
        :func:`~model_crafter.validation.tune.tune` is the place where
        a grid of specs is evaluated.
    data
        Input frame; must contain ``spec.target`` and the splitter's
        ``time_col``.
    splitter
        Any :class:`~model_crafter.validation.splitters.Splitter`. The
        runner refuses to run when the spec's loss declares a
        ``label_horizon`` attribute and ``splitter.gap`` is zero
        (DESIGN.md §3.2).
    metrics
        A callable or iterable of callables, each with signature
        ``metric(sol, data, weights=None) -> float`` (or a result with a
        ``.value`` attribute — duck-typed so P3.D's richer result types
        slot in cleanly). Per DESIGN.md §9.6 every metric accepts
        ``weights=``.
    weights
        Sample weights — a column name resolved in ``data``, a 1-D
        array, or ``None``.

    Returns
    -------
    CVResult
        Frozen value bundling per-fold metrics and the fold solutions.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be a DataFrame; got {type(data).__name__}")

    metric_callables = _normalise_metrics(metrics)

    # ---------------- label_horizon vs gap check ----------------
    label_horizon = getattr(getattr(spec, "loss", None), "label_horizon", None)
    gap_td = _coerce_timedelta(getattr(splitter, "gap", None))
    if label_horizon is not None and gap_td <= pd.Timedelta(0):
        raise AssumptionError(
            f"cross_validate: spec.loss declares label_horizon="
            f"{label_horizon!s} but splitter.gap is zero. Set "
            "splitter=expanding_window(..., gap=<at least label_horizon>) or "
            "use a splitter that purges around validation windows "
            "(DESIGN.md §3.2)."
        )
    # Optional warning hint: gap shorter than label_horizon is still leaky.
    if label_horizon is not None and gap_td < _coerce_timedelta(label_horizon):
        raise AssumptionError(
            f"cross_validate: spec.loss declares label_horizon="
            f"{label_horizon!s} but splitter.gap={gap_td!s} is smaller. "
            "Increase the splitter's gap to at least label_horizon "
            "(DESIGN.md §3.2)."
        )

    # ---------------- iterate folds ----------------
    full_weights = _resolve_weights_to_array(weights, data)
    fold_pairs: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    fold_results: list[dict] = []
    solutions: list[Solution] = []

    time_col = getattr(splitter, "time_col", None)
    for k, (train, valid) in enumerate(splitter.split(data)):
        if time_col is not None:
            train_period = (
                pd.to_datetime(train[time_col]).min(),
                pd.to_datetime(train[time_col]).max(),
            )
            valid_period = (
                pd.to_datetime(valid[time_col]).min(),
                pd.to_datetime(valid[time_col]).max(),
            )
            # Enforce NoTemporalLeakage per fold *before* solving — fail
            # fast if a custom splitter handed us a leaky partition.
            if valid_period[0] < train_period[1] + gap_td:
                raise AssumptionError(
                    f"NoTemporalLeakage violated at fold {k}: "
                    f"train_end={train_period[1]!s} + gap={gap_td!s} > "
                    f"valid_start={valid_period[0]!s} (DESIGN.md §3.2)."
                )
        else:
            train_period = None
            valid_period = None

        # ---- fit on the training fold ----
        fold_weights = _slice_weights(full_weights, data, train)
        sol = solve(spec, train, weights=fold_weights)
        solutions.append(sol)

        # ---- score on the validation fold ----
        valid_weights = _slice_weights(full_weights, data, valid)
        metric_values: dict[str, float] = {}
        for name, fn in metric_callables:
            metric_values[name] = _call_metric(fn, sol, valid, valid_weights)

        fold_results.append(
            {
                "train_period": train_period,
                "valid_period": valid_period,
                "metrics": metric_values,
                "solution": sol,
                "train": train,
                "valid": valid,
            }
        )
        fold_pairs.append((train, valid))

    # ---------------- run the standalone NoTemporalLeakage audit ----------------
    audit_cv = _StandaloneAuditCV(splitter=splitter, folds=tuple(fold_pairs))
    audit_result = NoTemporalLeakage().check(spec, data, cv=audit_cv)
    report = AssumptionReport(results=(audit_result,))

    return CVResult(
        fold_results=tuple(fold_results),
        solutions=tuple(solutions),
        splitter=splitter,
        assumptions=report,
        folds=tuple(fold_pairs),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _StandaloneAuditCV:
    """Lightweight ``cv`` shape used to call :class:`NoTemporalLeakage`'s
    standalone ``check`` method *after* the folds are materialised. The
    fields are exactly what :class:`NoTemporalLeakage._read_metadata` and
    ``_extract_folds`` consume."""

    splitter: Any
    folds: tuple[tuple[pd.DataFrame, pd.DataFrame], ...]


def _normalise_metrics(
    metrics: Iterable[Callable] | Callable,
) -> tuple[tuple[str, Callable], ...]:
    """Coerce ``metrics`` into a tuple of ``(name, fn)``.

    Bare callables become a singleton; iterables are unpacked. The metric
    name is taken from ``fn.__name__`` (the standard for P3.D's metric
    primitives), or from ``fn.name`` when the callable is itself a value
    object with a ``name`` attribute.
    """
    if callable(metrics) and not isinstance(metrics, Sequence):
        metrics_iter: tuple[Callable, ...] = (metrics,)
    else:
        metrics_iter = tuple(metrics)  # type: ignore[arg-type]
    if not metrics_iter:
        raise ValueError("cross_validate requires at least one metric")
    out: list[tuple[str, Callable]] = []
    for fn in metrics_iter:
        if not callable(fn):
            raise TypeError(f"metric {fn!r} is not callable")
        name = getattr(fn, "name", None) or getattr(fn, "__name__", None) or repr(fn)
        out.append((str(name), fn))
    return tuple(out)


def _resolve_weights_to_array(
    weights: str | np.ndarray | pd.Series | None,
    data: pd.DataFrame,
) -> np.ndarray | None:
    """Resolve ``weights=`` to a numpy array aligned with ``data.index``.

    A string is looked up as a column of ``data``. None passes through.
    """
    if weights is None:
        return None
    if isinstance(weights, str):
        if weights not in data.columns:
            raise KeyError(
                f"weights column {weights!r} not in data (columns: "
                f"{list(data.columns)})"
            )
        return data[weights].to_numpy(dtype=float)
    arr = np.asarray(weights, dtype=float)
    if arr.shape[0] != len(data):
        raise ValueError(
            f"weights length {arr.shape[0]} != data length {len(data)}"
        )
    return arr


def _slice_weights(
    full_weights: np.ndarray | None,
    data: pd.DataFrame,
    fold_df: pd.DataFrame,
) -> np.ndarray | None:
    """Slice resolved weights to match ``fold_df``.

    ``full_weights`` is the array form returned by
    :func:`_resolve_weights_to_array` keyed by ``data.index``. ``None``
    passes through.
    """
    if full_weights is None:
        return None
    s = pd.Series(full_weights, index=data.index)
    return s.reindex(fold_df.index).to_numpy(dtype=float)


def _call_metric(
    fn: Callable,
    sol: Solution,
    data: pd.DataFrame,
    weights: str | np.ndarray | None,
) -> float:
    """Call a metric callable and coerce its return to ``float``.

    Accepts either a bare float or any object with a ``.value`` attribute
    (P3.D's richer result types).
    """
    try:
        out = fn(sol, data, weights=weights)
    except TypeError:
        out = fn(sol, data)
    if hasattr(out, "value"):
        return float(out.value)
    return float(out)
