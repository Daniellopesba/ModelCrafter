"""Tests for ``cross_validate`` and the ``NoTemporalLeakage`` assumption
(AGENTS.md Task P3.B, DESIGN.md §3.2, §4.3).

Acceptance criteria covered here:

* ``NoTemporalLeakage`` fires on a synthetic violation (a CV partition
  whose train and validation windows overlap by more than the splitter's
  gap allows).
* ``cross_validate`` refuses to run when the spec's loss declares a
  ``label_horizon`` attribute and the splitter has ``gap`` unset / zero
  (DESIGN.md §3.2: a 12-month default label can't be observed until
  ``t + 365D``).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from model_crafter.assumptions import (
    AssumptionError,
    NoTemporalLeakage,
)
from model_crafter.loss import squared_error
from model_crafter.spec import linear
from model_crafter.validation.cross_validate import CVResult, cross_validate
from model_crafter.validation.splitters import (
    expanding_window,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _linear_panel(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Synthetic linear-regression panel with a time column."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2020-01-01")
    dates = start + pd.to_timedelta(np.arange(n), unit="D")
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + rng.normal(scale=0.5, size=n)
    return pd.DataFrame(
        {"origination_dt": dates, "x1": x1, "x2": x2, "y": y}
    )


def _ols_spec():
    return linear(
        target="y",
        features=["x1", "x2"],
        loss=squared_error,
    )


def _mse_metric(sol, data, weights=None):
    """Tiny metric callable for the test — P3.D's metric primitives will
    replace this at P3.INTEG."""
    from model_crafter.solve import predict

    yhat = predict(sol, data)
    y = data[sol.spec.target].to_numpy(dtype=float)
    if weights is None:
        return float(np.mean((y - yhat.to_numpy(dtype=float)) ** 2))
    w = np.asarray(weights, dtype=float)
    resid = y - yhat.to_numpy(dtype=float)
    return float(np.sum(w * resid * resid) / np.sum(w))


# ---------------------------------------------------------------------------
# cross_validate — happy path
# ---------------------------------------------------------------------------


def test_cross_validate_runs_and_returns_cvresult():
    df = _linear_panel(n=800)
    splitter = expanding_window(
        time_col="origination_dt",
        n_folds=4,
        horizon="60D",
        gap="0D",
        min_train="120D",
    )
    cv = cross_validate(
        _ols_spec(), df, splitter, metrics=[_mse_metric]
    )
    assert isinstance(cv, CVResult)
    assert len(cv.fold_results) == 4
    assert len(cv.solutions) == 4
    # Each fold dict has the documented keys.
    for fold in cv.fold_results:
        assert "train_period" in fold
        assert "valid_period" in fold
        assert "metrics" in fold
        assert "solution" in fold
        assert "_mse_metric" in fold["metrics"]


def test_cross_validate_summary_is_dataframe_one_row_per_fold():
    df = _linear_panel(n=600)
    splitter = expanding_window(
        time_col="origination_dt",
        n_folds=3,
        horizon="60D",
        gap="0D",
        min_train="120D",
    )
    cv = cross_validate(_ols_spec(), df, splitter, metrics=[_mse_metric])
    summary = cv.summary()
    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == 3
    assert "_mse_metric" in summary.columns


def test_cross_validate_supports_weights():
    df = _linear_panel(n=400)
    df["w"] = 1.0
    splitter = expanding_window(
        time_col="origination_dt",
        n_folds=2,
        horizon="60D",
        gap="0D",
        min_train="120D",
    )
    cv = cross_validate(_ols_spec(), df, splitter, metrics=[_mse_metric],
                        weights="w")
    assert len(cv.fold_results) == 2


# ---------------------------------------------------------------------------
# cross_validate — assumption integration
# ---------------------------------------------------------------------------


def test_cross_validate_refuses_when_label_horizon_set_and_gap_zero():
    """ACCEPTANCE: cross_validate refuses to run when the spec's loss
    declares a ``label_horizon`` attribute and the splitter has ``gap=0``
    (DESIGN.md §3.2)."""
    import contextlib

    df = _linear_panel(n=400)
    spec = _ols_spec()
    loss = spec.loss
    setattr(loss, "label_horizon", pd.Timedelta("365D"))  # noqa: B010
    try:
        splitter = expanding_window(
            time_col="origination_dt",
            n_folds=3,
            horizon="60D",
            gap="0D",
            min_train="120D",
        )
        with pytest.raises(AssumptionError, match="label_horizon"):
            cross_validate(spec, df, splitter, metrics=[_mse_metric])
    finally:
        with contextlib.suppress(AttributeError):
            delattr(loss, "label_horizon")


def test_cross_validate_ok_when_label_horizon_set_and_gap_nonzero():
    """The label_horizon check passes when the splitter's gap is at least
    as large as the loss's declared horizon."""
    import contextlib

    df = _linear_panel(n=1500)
    spec = _ols_spec()
    # Temporarily attach a label_horizon attribute to the (singleton) loss
    # so the spec exposes it. Restore at the end.
    loss = spec.loss
    setattr(loss, "label_horizon", pd.Timedelta("60D"))  # noqa: B010
    try:
        splitter = expanding_window(
            time_col="origination_dt",
            n_folds=3,
            horizon="60D",
            gap="90D",
            min_train="180D",
        )
        cv = cross_validate(spec, df, splitter, metrics=[_mse_metric])
        assert len(cv.fold_results) == 3
    finally:
        with contextlib.suppress(AttributeError):
            delattr(loss, "label_horizon")


# ---------------------------------------------------------------------------
# NoTemporalLeakage — direct check
# ---------------------------------------------------------------------------


def test_no_temporal_leakage_passes_on_clean_partition():
    df = _linear_panel(n=800)
    splitter = expanding_window(
        time_col="origination_dt",
        n_folds=3,
        horizon="60D",
        gap="30D",
        min_train="120D",
    )
    folds = list(splitter.split(df))
    cv = SimpleNamespace(splitter=splitter, folds=folds)
    result = NoTemporalLeakage().check(None, df, cv=cv)
    assert result.passed, result.message


def test_no_temporal_leakage_fires_on_manual_violation():
    """ACCEPTANCE: NoTemporalLeakage fires on a synthetic violation
    (manually-constructed bad fold pair)."""
    df = _linear_panel(n=400)
    # Pick contiguous slices so the validation start is BEFORE the train end.
    train = df.iloc[:200]
    valid = df.iloc[100:300]  # overlaps train
    splitter = SimpleNamespace(
        time_col="origination_dt", gap=pd.Timedelta("0D")
    )
    cv = SimpleNamespace(splitter=splitter, folds=[(train, valid)])
    result = NoTemporalLeakage().check(None, df, cv=cv)
    assert not result.passed
    assert "leakage" in result.message.lower()
    assert "fold 0" in result.message


def test_no_temporal_leakage_fires_on_gap_violation():
    df = _linear_panel(n=800)
    # A splitter ignoring the gap is exactly what we want to catch.
    times = pd.to_datetime(df["origination_dt"])
    cutoff_at = times.iloc[300]
    train = df[times <= cutoff_at]
    valid = df[(times > cutoff_at) & (times <= cutoff_at + pd.Timedelta("60D"))]
    fake_splitter = SimpleNamespace(
        time_col="origination_dt", gap=pd.Timedelta("365D")
    )
    cv = SimpleNamespace(splitter=fake_splitter, folds=[(train, valid)])
    result = NoTemporalLeakage().check(None, df, cv=cv)
    assert not result.passed


def test_no_temporal_leakage_skipped_when_cv_is_none():
    result = NoTemporalLeakage().check(None, _linear_panel(n=10), cv=None)
    assert result.passed
    assert "skipped" in result.message.lower()


def test_cross_validate_attaches_temporal_leakage_check():
    df = _linear_panel(n=800)
    splitter = expanding_window(
        time_col="origination_dt",
        n_folds=3,
        horizon="60D",
        gap="30D",
        min_train="120D",
    )
    cv = cross_validate(_ols_spec(), df, splitter, metrics=[_mse_metric])
    # The CVResult exposes the assumption report on the cv runner.
    assert cv.assumptions is not None
    names = {r.name for r in cv.assumptions.results}
    assert "NoTemporalLeakage" in names
