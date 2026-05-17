"""Tests for ``nested_cv`` (AGENTS.md Task P3.B, DESIGN.md §3.2, §7.10.2).

Acceptance criterion: ``nested_cv`` outer error is statistically larger
(in expectation, on a simulation with a fixed seed) than ``tune`` inner
error on the same data — the ESL §7.10.2 optimism bias is detectable.

The idea (ESL §7.10.2): when the same CV is used to *both* tune
:math:`\\lambda` and report performance, the chosen :math:`\\lambda`
is a function of the held-out folds, so the held-out metric *at the
chosen* :math:`\\lambda` is biased upward. Nested CV holds out a
separate outer fold for assessment, eliminating that bias.

The simulation:

* Generate small noisy panels (n=80, p=8) with a known coefficient
  vector and a high noise level.
* Sweep a coarse lambda grid (8 values).
* Use 3 outer folds and 3 inner folds (small CV → larger optimism).
* Repeat 50 times with different seeds.
* Compare ``tune``'s best inner-CV mean against ``nested_cv``'s outer
  metric mean. The first should be larger on average — a paired t-test
  rejects equal means at p < 0.05.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from model_crafter.loss import squared_error
from model_crafter.penalty import l2
from model_crafter.spec import linear
from model_crafter.validation.splitters import expanding_window
from model_crafter.validation.tune import (
    NestedCVResult,
    nested_cv,
    one_se_rule,
    tune,
)


def _noisy_panel(n: int, p: int, seed: int) -> pd.DataFrame:
    """Synthetic noisy panel with a known linear-regression DGP."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2020-01-01")
    dates = start + pd.to_timedelta(np.arange(n), unit="D")
    X = rng.normal(size=(n, p))
    beta = np.array([1.0, -0.8, 0.5, -0.3, 0.2, -0.15, 0.1, -0.05])[:p]
    noise = rng.normal(scale=2.0, size=n)  # high noise so CV is informative
    y = X @ beta + noise
    df = pd.DataFrame(X, columns=pd.Index([f"x{i}" for i in range(p)]))
    df["origination_dt"] = dates
    df["y"] = y
    return df


def _neg_mse(sol, data, weights=None):
    """Higher is better. Negative MSE so 'maximize' is the right
    direction."""
    from model_crafter.solve import predict

    yhat = predict(sol, data).to_numpy(dtype=float)
    y = data[sol.spec.target].to_numpy(dtype=float)
    if weights is None:
        return -float(np.mean((y - yhat) ** 2))
    w = np.asarray(weights, dtype=float)
    return -float(np.sum(w * (y - yhat) ** 2) / np.sum(w))


def _spec_fn(lam: float):
    return linear(
        target="y",
        features=[f"x{i}" for i in range(8)],
        loss=squared_error,
        penalty=l2(lam),
    )


def test_nested_cv_returns_expected_shape():
    df = _noisy_panel(n=600, p=8, seed=0)
    grid = np.geomspace(1e-3, 100, num=5)
    outer = expanding_window(
        time_col="origination_dt",
        n_folds=3,
        horizon="60D",
        gap="0D",
        min_train="120D",
    )
    inner = expanding_window(
        time_col="origination_dt",
        n_folds=2,
        horizon="20D",
        gap="0D",
        min_train="30D",
    )
    out = nested_cv(
        spec_fn=_spec_fn,
        grid=grid,
        data=df,
        outer_splitter=outer,
        inner_splitter=inner,
        metric=_neg_mse,
    )
    assert isinstance(out, NestedCVResult)
    assert len(out.outer_metric) == 3
    assert len(out.best_params) == 3
    assert len(out.inner_curves) == 3
    for curve in out.inner_curves:
        assert isinstance(curve, pd.DataFrame)
        assert "metric_mean" in curve.columns


def test_nested_cv_optimism_bias_detectable():
    r"""ACCEPTANCE: nested_cv outer error is statistically larger (in
    expectation) than tune inner error on the same data, demonstrating
    ESL §7.10.2's optimism bias.

    We run 50 small-sample simulations. For each, we compute:

    * ``tune.cv_curve.loc[best_param, "metric_mean"]`` — the inner-CV
      mean at the chosen lambda (optimistic).
    * ``nested_cv.outer_metric[metric].mean()`` — the honest assessment.

    A paired t-test on (inner - outer) should reject ``equal means`` in
    favour of ``inner > outer``.
    """
    n_reps = 50
    grid = np.geomspace(1e-3, 100, num=8)
    inner_means: list[float] = []
    outer_means: list[float] = []

    for rep in range(n_reps):
        df = _noisy_panel(n=400, p=8, seed=rep + 1)
        outer = expanding_window(
            time_col="origination_dt",
            n_folds=3,
            horizon="40D",
            gap="0D",
            min_train="120D",
        )
        inner = expanding_window(
            time_col="origination_dt",
            n_folds=3,
            horizon="20D",
            gap="0D",
            min_train="40D",
        )
        # tune on full data — the "optimistic" assessment
        tuned = tune(
            spec_fn=_spec_fn,
            grid=grid,
            data=df,
            splitter=outer,
            metric=_neg_mse,
        )
        optimistic = float(tuned.cv_curve.loc[tuned.best_param, "metric_mean"])

        # nested CV — the "honest" assessment
        nested = nested_cv(
            spec_fn=_spec_fn,
            grid=grid,
            data=df,
            outer_splitter=outer,
            inner_splitter=inner,
            metric=_neg_mse,
        )
        honest = float(nested.outer_metric["_neg_mse"].mean())

        inner_means.append(optimistic)
        outer_means.append(honest)

    diffs = np.asarray(inner_means) - np.asarray(outer_means)
    # Optimism: tune (inner) >= nested (outer) on average.
    assert diffs.mean() > 0, (
        f"expected positive optimism bias, got mean(diff)={diffs.mean():.4f}"
    )
    # Paired t-test (one-sided): mean(diff) > 0.
    result = stats.ttest_1samp(diffs, popmean=0.0, alternative="greater")
    # scipy types ``TtestResult`` opaquely under pyright; use the tuple
    # interface which is documented to be (statistic, pvalue).
    t_stat = float(result[0])  # type: ignore[index]
    p_value = float(result[1])  # type: ignore[index]
    assert p_value < 0.05, (
        f"paired t-test failed to detect optimism: t={t_stat:.3f}, p={p_value:.4f}"
    )


def test_nested_cv_with_one_se_rule():
    df = _noisy_panel(n=600, p=8, seed=42)
    grid = np.geomspace(1e-3, 100, num=5)
    outer = expanding_window(
        time_col="origination_dt",
        n_folds=3,
        horizon="60D",
        gap="0D",
        min_train="120D",
    )
    inner = expanding_window(
        time_col="origination_dt",
        n_folds=2,
        horizon="20D",
        gap="0D",
        min_train="30D",
    )
    out = nested_cv(
        spec_fn=_spec_fn,
        grid=grid,
        data=df,
        outer_splitter=outer,
        inner_splitter=inner,
        metric=_neg_mse,
        rule=one_se_rule,
    )
    # Every chosen inner param must be on the grid.
    for p in out.best_params:
        assert p in grid


def test_nested_cv_rejects_empty_grid():
    with pytest.raises(ValueError, match="grid"):
        nested_cv(
            spec_fn=_spec_fn,
            grid=[],
            data=_noisy_panel(n=200, p=8, seed=0),
            outer_splitter=expanding_window(
                time_col="origination_dt",
                n_folds=2,
                horizon="20D",
                gap="0D",
                min_train="30D",
            ),
            inner_splitter=expanding_window(
                time_col="origination_dt",
                n_folds=2,
                horizon="10D",
                gap="0D",
                min_train="15D",
            ),
            metric=_neg_mse,
        )
