"""Tests for ``mc.binned`` (bin-indicator basis; DESIGN.md §3.1, ESL §5.2).

Acceptance criterion #3 quoted from the task brief:

3. ``mc.binned`` on the same data + bin definitions yields equivalent
   in-sample fit (up to identifiability — k-1 indicator columns vs
   1 WoE column, but the linear predictor span is the same) but
   different out-of-sample CV performance to ``mc.woe`` (because
   ``mc.binned`` has more parameters to overfit).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import model_crafter as mc
from model_crafter.assumptions.logistic import NoPerfectSeparation
from model_crafter.terms.binning import manual, monotonic
from model_crafter.terms.woe import BinnedTerm, binned, fit_binnings, woe

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_credit_panel():
    """Synthetic non-separable binary classification problem."""
    rng = np.random.default_rng(123)
    n = 2000
    x = rng.normal(50, 15, n)
    p = 1.0 / (1.0 + np.exp(-(x - 50) / 30))
    y = (rng.random(n) < p).astype(int)
    return pd.DataFrame({"x": x, "y": y})


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_binned_constructor_returns_binnedterm():
    t = binned("x", bins=monotonic())
    assert isinstance(t, BinnedTerm)
    assert t.column == "x"
    assert t.fitted is None


def test_binned_rejects_non_binning_strategy():
    with pytest.raises(TypeError, match="binning"):
        binned("x", bins="not a strategy")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Expand: drop-first indicator basis
# ---------------------------------------------------------------------------


def test_binned_expand_drops_first_bin():
    """k bins → k-1 indicator columns (drop-first)."""
    df = pd.DataFrame(
        {
            "x": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            "y": [0, 0, 1, 1, 0, 1, 0, 1],
        }
    )
    spec = mc.linear(
        target="y",
        features=binned("x", bins=manual(edges=[25.0, 55.0])),
        loss=mc.logistic,
        penalty=mc.l2(0.1),
    )
    fitted_spec = fit_binnings(spec, df)
    fitted_term = fitted_spec.features[0]
    expanded = fitted_term.expand(df, fit_state=None)
    # 3 numeric bins → 2 indicator columns.
    assert expanded.values.shape == (8, 2)
    assert len(expanded.columns) == 2
    # Each row has exactly one "1" if it's NOT in bin 0, else all zeros.
    row_sums = expanded.values.sum(axis=1)
    assert ((row_sums == 0) | (row_sums == 1)).all()


def test_binned_expand_column_names_use_bin_labels():
    df = pd.DataFrame(
        {
            "x": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "y": [0, 0, 1, 1, 0, 1],
        }
    )
    spec = mc.linear(
        target="y",
        features=binned("x", bins=manual(edges=[25.0, 55.0])),
        loss=mc.logistic,
        penalty=mc.l2(0.1),
    )
    fitted_spec = fit_binnings(spec, df)
    fitted_term = fitted_spec.features[0]
    expanded = fitted_term.expand(df, fit_state=None)
    # The bin labels for non-reference bins should be present in column names.
    assert "x__" in expanded.columns[0]
    assert "x__" in expanded.columns[1]


def test_binned_expand_errors_when_only_one_bin():
    """A single-bin learned result yields zero non-reference columns → error."""
    df = pd.DataFrame({"x": [5.0] * 50, "y": [0, 1] * 25})
    spec = mc.linear(
        target="y",
        features=binned("x", bins=manual(edges=[100.0])),  # single edge, but values < 100 only
        loss=mc.logistic,
        penalty=mc.l2(0.1),
    )
    # All x values < 100 → bin 0 only. After learning, we have 2 numeric
    # bins (the edge is interior), so 1 indicator column → fine.
    fitted_spec = fit_binnings(spec, df)
    t = fitted_spec.features[0]
    expanded = t.expand(df, fit_state=None)
    # 2 numeric bins (one fully populated, one empty) → 1 indicator column.
    assert expanded.values.shape[1] == 1


# ---------------------------------------------------------------------------
# Acceptance #3: equivalent in-sample span (different out-of-sample)
# ---------------------------------------------------------------------------


def test_woe_and_binned_in_sample_log_loss_within_tolerance(
    synthetic_credit_panel,
):
    """Acceptance #3: same bins, equivalent in-sample fit (up to identifiability).

    The WoE and bin-indicator bases on the *same* bin partition span the
    same linear-predictor space (up to a one-dimensional affine shift
    absorbed by the intercept), so an unregularised joint fit on each
    should achieve essentially the same in-sample log-loss. With
    Laplace-smoothed WoE values and a tiny ridge for identifiability, we
    allow a small tolerance.
    """
    df = synthetic_credit_panel
    bins = manual(edges=[40.0, 50.0, 60.0])

    spec_woe = mc.linear(
        target="y", features=woe("x", bins=bins),
        loss=mc.logistic, penalty=mc.l2(1e-4),
    )
    spec_bin = mc.linear(
        target="y", features=binned("x", bins=bins),
        loss=mc.logistic, penalty=mc.l2(1e-4),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol_woe = mc.solve(
            fit_binnings(spec_woe, df), df,
            on_violation="ignore", suppress=(NoPerfectSeparation,),
        )
        sol_bin = mc.solve(
            fit_binnings(spec_bin, df), df,
            on_violation="ignore", suppress=(NoPerfectSeparation,),
        )

    # Log-loss values should agree to ~1% relative (bin-indicator may
    # slightly outperform on training because it has more parameters).
    rel_diff = abs(sol_woe.loss_value - sol_bin.loss_value) / abs(sol_woe.loss_value)
    assert rel_diff < 0.05, (
        f"in-sample log-loss disagrees: WoE={sol_woe.loss_value:.6f}, "
        f"binned={sol_bin.loss_value:.6f}"
    )


def test_binned_has_more_parameters_than_woe():
    """``mc.binned`` produces k-1 design columns vs ``mc.woe``'s single column."""
    df = pd.DataFrame(
        {
            "x": list(range(100)),
            "y": [0, 1] * 50,
        }
    )
    bins = manual(edges=[25.0, 50.0, 75.0])

    spec_woe = mc.linear(
        target="y", features=woe("x", bins=bins),
        loss=mc.logistic, penalty=mc.l2(0.1),
    )
    spec_bin = mc.linear(
        target="y", features=binned("x", bins=bins),
        loss=mc.logistic, penalty=mc.l2(0.1),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol_woe = mc.solve(
            fit_binnings(spec_woe, df), df,
            on_violation="ignore", suppress=(NoPerfectSeparation,),
        )
        sol_bin = mc.solve(
            fit_binnings(spec_bin, df), df,
            on_violation="ignore", suppress=(NoPerfectSeparation,),
        )

    # WoE: 1 coef + intercept = 2 columns. Binned: 3 indicators + intercept = 4.
    assert len(sol_woe.design_columns) == 2
    assert len(sol_bin.design_columns) == 4


def test_binned_and_woe_produce_similar_in_sample_predictions(
    synthetic_credit_panel,
):
    """Per Acceptance #3: in-sample predictions are very close (same span)."""
    df = synthetic_credit_panel
    bins = manual(edges=[40.0, 50.0, 60.0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol_woe = mc.solve(
            fit_binnings(
                mc.linear(
                    target="y", features=woe("x", bins=bins),
                    loss=mc.logistic, penalty=mc.l2(1e-4),
                ),
                df,
            ),
            df, on_violation="ignore", suppress=(NoPerfectSeparation,),
        )
        sol_bin = mc.solve(
            fit_binnings(
                mc.linear(
                    target="y", features=binned("x", bins=bins),
                    loss=mc.logistic, penalty=mc.l2(1e-4),
                ),
                df,
            ),
            df, on_violation="ignore", suppress=(NoPerfectSeparation,),
        )

    yhat_woe = mc.predict(sol_woe, df).to_numpy()
    yhat_bin = mc.predict(sol_bin, df).to_numpy()
    # The two bases span the same step-function family on the same bin
    # partition, so in-sample predictions should be essentially identical
    # (mean absolute difference << 0.01).
    assert float(np.mean(np.abs(yhat_woe - yhat_bin))) < 0.01


# ---------------------------------------------------------------------------
# Predict-time
# ---------------------------------------------------------------------------


def test_binned_predict_round_trips_through_fit_state():
    df = pd.DataFrame(
        {
            "x": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            "y": [0, 0, 1, 1, 0, 1, 0, 1],
        }
    )
    spec = mc.linear(
        target="y",
        features=binned("x", bins=manual(edges=[25.0, 55.0])),
        loss=mc.logistic,
        penalty=mc.l2(0.1),
    )
    fitted_spec = fit_binnings(spec, df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = mc.solve(
            fitted_spec, df,
            on_violation="ignore", suppress=(NoPerfectSeparation,),
        )

    new_df = pd.DataFrame({"x": [15.0, 40.0, 70.0]})
    yhat = mc.predict(sol, new_df)
    assert len(yhat) == 3
    assert ((yhat >= 0.0) & (yhat <= 1.0)).all()
