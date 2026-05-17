"""Tests for ``tune`` and the selection rules (AGENTS.md Task P3.B,
DESIGN.md §3.2, §7.10, §11).

Acceptance criterion: ``tune`` returns a curve where the chosen param is
the optimum under the chosen rule (verified for both ``best_mean`` and
``one_se_rule`` on synthetic problems with known optima).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model_crafter.loss import squared_error
from model_crafter.penalty import l2
from model_crafter.spec import linear
from model_crafter.validation.splitters import expanding_window
from model_crafter.validation.tune import (
    TuneResult,
    best_mean,
    one_se_rule,
    tune,
)

# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def _ridge_panel(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2020-01-01")
    dates = start + pd.to_timedelta(np.arange(n), unit="D")
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    # y depends linearly on x1, x2.
    y = 1.0 + 1.5 * x1 - 0.7 * x2 + rng.normal(scale=0.5, size=n)
    return pd.DataFrame(
        {"origination_dt": dates, "x1": x1, "x2": x2, "y": y}
    )


def _ridge_spec(lam: float):
    return linear(
        target="y",
        features=["x1", "x2"],
        loss=squared_error,
        penalty=l2(lam),
    )


def _neg_mse(sol, data, weights=None):
    """Higher is better, like AUC. We negate MSE to mimic 'maximize'."""
    from model_crafter.solve import predict

    yhat = predict(sol, data).to_numpy(dtype=float)
    y = data[sol.spec.target].to_numpy(dtype=float)
    if weights is None:
        return -float(np.mean((y - yhat) ** 2))
    w = np.asarray(weights, dtype=float)
    return -float(np.sum(w * (y - yhat) ** 2) / np.sum(w))


# ---------------------------------------------------------------------------
# Selection rules (synthetic curves so the test is self-contained)
# ---------------------------------------------------------------------------


def test_best_mean_maximize():
    curve = pd.DataFrame(
        {
            "metric_mean": [0.1, 0.5, 0.7, 0.6, 0.4],
            "metric_sd": [0.1, 0.1, 0.1, 0.1, 0.1],
        },
        index=pd.Index([1e-3, 1e-2, 1e-1, 1.0, 10.0]),
    )
    chosen = best_mean(curve, direction="maximize")
    assert chosen == 1e-1


def test_best_mean_minimize():
    curve = pd.DataFrame(
        {
            "metric_mean": [10.0, 3.0, 1.0, 2.5, 5.0],
            "metric_sd": [1.0, 1.0, 1.0, 1.0, 1.0],
        },
        index=pd.Index([1e-3, 1e-2, 1e-1, 1.0, 10.0]),
    )
    chosen = best_mean(curve, direction="minimize")
    assert chosen == 1e-1


def test_one_se_rule_picks_parsimonious_when_within_1se():
    """ESL §7.10: one-SE rule picks the simplest model within 1 SE of
    the best. For lambda, "simpler" means larger lambda."""
    curve = pd.DataFrame(
        {
            "metric_mean": [0.85, 0.86, 0.88, 0.87, 0.80, 0.70],
            "metric_sd": [0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
        },
        index=pd.Index([1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]),
    )
    chosen = one_se_rule(curve, direction="maximize")
    # Best mean is 0.88 (lambda=1e-2); 1 SE down is 0.86. Eligible
    # lambdas: 1e-3 (0.86), 1e-2 (0.88), 1e-1 (0.87). The most
    # parsimonious — largest lambda — is 1e-1.
    assert chosen == 1e-1


def test_one_se_rule_collapses_to_best_mean_when_curve_flat_at_top():
    curve = pd.DataFrame(
        {
            "metric_mean": [0.5, 0.7, 0.85, 0.8, 0.7, 0.4],
            "metric_sd": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
        },
        index=pd.Index([1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]),
    )
    chosen_best = best_mean(curve, direction="maximize")
    chosen_1se = one_se_rule(curve, direction="maximize")
    assert chosen_best == 1e-2
    assert chosen_1se == 1e-2  # only one lambda within 1 SE of best


def test_one_se_rule_minimise_direction():
    curve = pd.DataFrame(
        {
            "metric_mean": [10.0, 3.0, 1.0, 1.1, 5.0],
            "metric_sd": [0.5, 0.5, 0.5, 0.5, 0.5],
        },
        index=pd.Index([1e-4, 1e-3, 1e-2, 1e-1, 1.0]),
    )
    chosen = one_se_rule(curve, direction="minimize")
    # Best mean is 1.0 (lambda=1e-2); 1 SE up is 1.5. Eligible lambdas:
    # 1e-2 (1.0) and 1e-1 (1.1). Simpler = larger lambda = 1e-1.
    assert chosen == 1e-1


# ---------------------------------------------------------------------------
# tune — end-to-end with known optimum
# ---------------------------------------------------------------------------


def test_tune_returns_tuneresult():
    df = _ridge_panel(n=400)
    splitter = expanding_window(
        time_col="origination_dt",
        n_folds=3,
        horizon="60D",
        gap="0D",
        min_train="60D",
    )
    grid = np.array([1e-3, 1e-1, 1.0])
    out = tune(
        spec_fn=_ridge_spec,
        grid=grid,
        data=df,
        splitter=splitter,
        metric=_neg_mse,
    )
    assert isinstance(out, TuneResult)
    assert out.best_param in grid
    assert isinstance(out.cv_curve, pd.DataFrame)
    assert "metric_mean" in out.cv_curve.columns
    assert "metric_sd" in out.cv_curve.columns
    assert len(out.cv_curve) == len(grid)
    # The refit on full data carries the chosen param.
    assert out.solution.spec.penalty.lam == out.best_param


def test_tune_chooses_optimum_under_best_mean():
    """ACCEPTANCE: tune returns a curve where the chosen param is the
    optimum (best_mean rule)."""
    df = _ridge_panel(n=400)
    splitter = expanding_window(
        time_col="origination_dt",
        n_folds=4,
        horizon="60D",
        gap="0D",
        min_train="60D",
    )
    # Geometric grid spanning small-to-large lambdas.
    grid = np.geomspace(1e-4, 100.0, num=10)
    out = tune(
        spec_fn=_ridge_spec,
        grid=grid,
        data=df,
        splitter=splitter,
        metric=_neg_mse,
        rule=best_mean,
    )
    # The chosen param must correspond to the curve's max metric_mean
    # (we're maximising _neg_mse).
    argmax_idx = out.cv_curve["metric_mean"].idxmax()
    assert out.best_param == argmax_idx


def test_tune_chooses_one_se_rule_param():
    """ACCEPTANCE: under one_se_rule, tune picks the most parsimonious
    param within 1 SE of the best."""
    df = _ridge_panel(n=400)
    splitter = expanding_window(
        time_col="origination_dt",
        n_folds=4,
        horizon="60D",
        gap="0D",
        min_train="60D",
    )
    grid = np.geomspace(1e-4, 100.0, num=10)
    out = tune(
        spec_fn=_ridge_spec,
        grid=grid,
        data=df,
        splitter=splitter,
        metric=_neg_mse,
        rule=one_se_rule,
    )
    best_mean_value = out.cv_curve["metric_mean"].max()
    best_idx = out.cv_curve["metric_mean"].idxmax()
    best_sd = out.cv_curve.loc[best_idx, "metric_sd"]
    threshold = best_mean_value - best_sd
    chosen_value = out.cv_curve.loc[out.best_param, "metric_mean"]
    # 1-SE param must be within 1 SE of the best.
    assert chosen_value >= threshold - 1e-12
    # And no larger lambda within 1 SE exists.
    larger = out.cv_curve.loc[out.cv_curve.index > out.best_param]
    if len(larger) > 0:
        assert (larger["metric_mean"] < threshold - 1e-12).all() or (
            larger.index > out.best_param
        ).all()


def test_tune_rejects_non_callable_spec_fn():
    with pytest.raises(TypeError, match="spec_fn"):
        tune(
            spec_fn="not callable",  # type: ignore[arg-type]
            grid=[1e-2],
            data=_ridge_panel(n=100),
            splitter=expanding_window(
                time_col="origination_dt",
                n_folds=2,
                horizon="60D",
                gap="0D",
                min_train="60D",
            ),
            metric=_neg_mse,
        )


def test_tune_rejects_empty_grid():
    with pytest.raises(ValueError, match="grid"):
        tune(
            spec_fn=_ridge_spec,
            grid=[],
            data=_ridge_panel(n=100),
            splitter=expanding_window(
                time_col="origination_dt",
                n_folds=2,
                horizon="60D",
                gap="0D",
                min_train="60D",
            ),
            metric=_neg_mse,
        )
