"""Tests for WoE-encoded terms (DESIGN.md §3.1, AGENTS.md Task P4.B).

Acceptance criteria quoted from the task brief:

1. ``mc.woe`` with ``mc.monotonic(min_bin_size=0.05)`` on a synthetic
   dataset produces bins where each bin has >= 5% of rows AND event
   rates are monotonically ordered.
2. ``WoEMonotonicityPreserved`` fires (warns) when the joint logistic-
   regression coefficient on a WoE-encoded column is negative.
4. ``binning_table(sol)`` returns a BinningTable with per-bin counts,
   event rates, WoE, IV; IV computed correctly to 1e-12 on a
   hand-derived worked example.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import model_crafter as mc
from model_crafter.assumptions._types import AssumptionError
from model_crafter.assumptions.logistic import NoPerfectSeparation
from model_crafter.assumptions.woe import (
    AtLeastOneEventPerBin,
    MinimumBinSize,
    MonotonicEventRate,
    WoEMonotonicityPreserved,
)
from model_crafter.inspect import binning_table
from model_crafter.terms.binning import BinningResult, manual, monotonic
from model_crafter.terms.woe import (
    BinnedTerm,
    WoETerm,
    binned,
    fit_binnings,
    woe,
)

# ---------------------------------------------------------------------------
# Synthetic data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_logistic_data():
    """Numeric feature with a clear monotone relationship to a binary target."""
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.normal(50, 15, n)
    # Moderate signal — not separable.
    p = 1.0 / (1.0 + np.exp(-(x - 50) / 30))
    y = (rng.random(n) < p).astype(int)
    return pd.DataFrame({"x": x, "y": y})


# ---------------------------------------------------------------------------
# Construction / type checks
# ---------------------------------------------------------------------------


def test_woe_constructor_returns_woeterm():
    t = woe("income", bins=monotonic())
    assert isinstance(t, WoETerm)
    assert t.column == "income"
    assert t.fitted is None


def test_woe_rejects_non_binning_strategy():
    with pytest.raises(TypeError, match="binning"):
        woe("income", bins="something")  # type: ignore[arg-type]


def test_woe_rejects_empty_column():
    with pytest.raises(ValueError, match="non-empty"):
        woe("", bins=monotonic())


def test_woe_declares_required_assumptions():
    t = woe("x", bins=monotonic())
    names = {type(a).__name__ for a in t.assumptions}
    assert {
        "AtLeastOneEventPerBin",
        "MinimumBinSize",
        "MonotonicEventRate",
        "WoEMonotonicityPreserved",
    } <= names


def test_woe_term_addition_is_termsum():
    t1 = woe("a", bins=monotonic())
    t2 = woe("b", bins=monotonic())
    s = t1 + t2
    from model_crafter.terms.base import TermSum

    assert isinstance(s, TermSum)
    assert len(s.terms) == 2


def test_string_plus_woeterm_is_termsum():
    t = "income" + woe("x", bins=monotonic())
    from model_crafter.terms.base import RawTerm, TermSum

    assert isinstance(t, TermSum)
    assert isinstance(t.terms[0], RawTerm)
    assert isinstance(t.terms[1], WoETerm)


# ---------------------------------------------------------------------------
# Acceptance #1: bin size + monotonic event rate
# ---------------------------------------------------------------------------


def test_monotonic_binning_respects_min_bin_size(synthetic_logistic_data):
    """Acceptance #1: every bin has >= min_bin_size * n rows."""
    df = synthetic_logistic_data
    spec = mc.linear(
        target="y", features=woe("x", bins=monotonic(min_bin_size=0.05)),
        loss=mc.logistic,
    )
    fitted_spec = fit_binnings(spec, df)
    result = fitted_spec.features[0].fitted
    n = len(df)
    min_count = int(0.05 * n)
    for label, ev, ne in zip(
        result.bin_labels, result.n_events, result.n_nonevents, strict=True
    ):
        # Skip (Missing) — synthetic has no NaN so this branch won't fire.
        if label == "(Missing)":
            continue
        total = ev + ne
        assert total >= min_count, (
            f"bin {label} has {total} rows; threshold {min_count}"
        )


def test_monotonic_binning_produces_monotone_event_rates(synthetic_logistic_data):
    """Acceptance #1: event rates across ordered bins are monotone."""
    df = synthetic_logistic_data
    spec = mc.linear(
        target="y", features=woe("x", bins=monotonic(min_bin_size=0.05)),
        loss=mc.logistic,
    )
    fitted_spec = fit_binnings(spec, df)
    result = fitted_spec.features[0].fitted
    rates = list(result.event_rate)
    # Allow either direction.
    diffs = np.diff(rates)
    assert np.all(diffs >= -1e-12) or np.all(diffs <= 1e-12), (
        f"event rates not monotone: {rates}"
    )


# ---------------------------------------------------------------------------
# Acceptance #4: hand-derived IV
# ---------------------------------------------------------------------------


def test_iv_matches_hand_derivation_to_1e_12():
    r"""Acceptance #4: IV is computed correctly to 1e-12 on a hand example.

    Worked example
    --------------
    Two bins. Bin 1: 100 rows, 30 events. Bin 2: 100 rows, 50 events.
    Total events: 80, total non-events: 120.

    With Laplace smoothing (+0.5 per bin per side):
      e1 = 30.5, ne1 = 70.5; e2 = 50.5, ne2 = 50.5
      e_tot = 81.0, ne_tot = 121.0
      p_e1 = 30.5/81 = 0.37654..., p_ne1 = 70.5/121 = 0.58264...
      WoE1 = log(0.37654/0.58264) = -0.43740...
      p_e2 = 50.5/81 = 0.62346..., p_ne2 = 50.5/121 = 0.41736...
      WoE2 = log(0.62346/0.41736) = +0.40133...
      IV = (0.37654 - 0.58264) * (-0.43740) + (0.62346 - 0.41736) * (0.40133)
         = 0.09014 + 0.08272 = 0.17286...

    Numerically computed below to 1e-12.
    """
    # Build a dataset that yields exactly bin 1: x < 0 (100/200, 30 events),
    # bin 2: x >= 0 (100/200, 50 events).
    n = 100
    df = pd.DataFrame(
        {
            "x": list(range(-n, 0)) + list(range(n)),
            "y": (
                [1] * 30 + [0] * 70  # bin 1
                + [1] * 50 + [0] * 50  # bin 2
            ),
        }
    )
    spec = mc.linear(
        target="y", features=woe("x", bins=manual(edges=[-0.5])),
        loss=mc.logistic,
    )
    fitted_spec = fit_binnings(spec, df)
    result = fitted_spec.features[0].fitted
    # Hand-derived values.
    expected_woe1 = float(np.log((30.5 / 81.0) / (70.5 / 121.0)))
    expected_woe2 = float(np.log((50.5 / 81.0) / (50.5 / 121.0)))
    expected_iv = (
        (30.5 / 81.0 - 70.5 / 121.0) * expected_woe1
        + (50.5 / 81.0 - 50.5 / 121.0) * expected_woe2
    )
    assert result.woe[0] == pytest.approx(expected_woe1, abs=1e-12)
    assert result.woe[1] == pytest.approx(expected_woe2, abs=1e-12)
    assert result.iv == pytest.approx(expected_iv, abs=1e-12)


def test_binning_table_has_expected_columns(synthetic_logistic_data):
    """Acceptance #4: BinningTable schema."""
    df = synthetic_logistic_data
    spec = mc.linear(
        target="y", features=woe("x", bins=monotonic(min_bin_size=0.10)),
        loss=mc.logistic,
    )
    fitted_spec = fit_binnings(spec, df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = mc.solve(
            fitted_spec, df, on_violation="ignore",
            suppress=(NoPerfectSeparation,),
        )
    bt = binning_table(sol)
    assert "x" in bt.tables
    df_table = bt.tables["x"]
    assert list(df_table.columns) == ["bin", "n", "n_events", "event_rate", "woe", "iv"]
    # IV reported in iv series.
    assert "x" in bt.iv.index
    assert bt.iv["x"] == pytest.approx(fitted_spec.features[0].fitted.iv, abs=1e-12)


def test_binning_table_repr_does_not_crash(synthetic_logistic_data):
    df = synthetic_logistic_data
    spec = mc.linear(
        target="y", features=woe("x", bins=monotonic(min_bin_size=0.10)),
        loss=mc.logistic,
    )
    fitted_spec = fit_binnings(spec, df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = mc.solve(
            fitted_spec, df, on_violation="ignore",
            suppress=(NoPerfectSeparation,),
        )
    s = repr(binning_table(sol))
    assert "BinningTable" in s
    assert "x" in s


# ---------------------------------------------------------------------------
# Acceptance #2: WoEMonotonicityPreserved fires on Simpson's-paradox data
# ---------------------------------------------------------------------------


def test_woe_monotonicity_preserved_passes_when_coef_positive(
    synthetic_logistic_data,
):
    """Acceptance #2: positive joint coefficient → check passes."""
    df = synthetic_logistic_data
    spec = mc.linear(
        target="y", features=woe("x", bins=monotonic(min_bin_size=0.10)),
        loss=mc.logistic,
    )
    fitted_spec = fit_binnings(spec, df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = mc.solve(
            fitted_spec, df, on_violation="ignore",
            suppress=(NoPerfectSeparation,),
        )
    # The joint coefficient on WoE-encoded x should be ~1 (positive).
    assert sol.coefficients["x"] > 0
    # Find the WoEMonotonicityPreserved result.
    matches = [
        r for r in sol.assumptions.results
        if r.name == "WoEMonotonicityPreserved"
    ]
    assert len(matches) == 1
    assert matches[0].passed, matches[0].message


def test_woe_monotonicity_preserved_fires_when_coef_negative():
    """Acceptance #2: Simpson's-paradox-style setup; check warns.

    Construct a dataset where the marginal relationship between x and y is
    positive (used to build the WoE encoding) but a confounding variable z
    flips the joint relationship. We achieve this by adding the *negated*
    raw x as a "confounder" so the joint logistic regression learns a
    negative coefficient on the WoE-encoded x.

    Simpler: directly inject a negative beta by flipping the WoE values
    before solve. We accomplish this through the assumption framework: hand
    it a spec + a fake solution whose coefficient is negative, and verify
    the check fires.
    """
    from types import SimpleNamespace

    # Construct a fitted WoETerm so the check can read .column.
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "x": rng.normal(0, 1, 100),
            "y": rng.integers(0, 2, 100),
        }
    )
    spec = mc.linear(
        target="y", features=woe("x", bins=monotonic(min_bin_size=0.10)),
        loss=mc.logistic,
    )
    fitted_spec = fit_binnings(spec, df)
    fake_sol = SimpleNamespace(
        coefficients=pd.Series({"(Intercept)": 0.0, "x": -0.7}),
        coefficient_se=None,
        design_columns=("(Intercept)", "x"),
    )
    check = WoEMonotonicityPreserved()
    result = check.check(fitted_spec, df, solution=fake_sol)
    assert not result.passed
    assert "negative" in result.message.lower() or "< 0" in result.message
    # Suggestion points to mc.binned.
    assert result.suggestion is not None
    assert "mc.binned" in result.suggestion


# ---------------------------------------------------------------------------
# HARD assumption checks
# ---------------------------------------------------------------------------


def test_at_least_one_event_per_bin_passes_on_normal_data(synthetic_logistic_data):
    df = synthetic_logistic_data
    spec = mc.linear(
        target="y", features=woe("x", bins=monotonic(min_bin_size=0.10)),
        loss=mc.logistic,
    )
    fitted_spec = fit_binnings(spec, df)
    check = AtLeastOneEventPerBin()
    res = check.check(fitted_spec, df)
    assert res.passed


def test_at_least_one_event_per_bin_fires_on_degenerate_bin():
    """A manually-constructed BinningResult with an empty bin triggers HARD."""
    # Build a tiny dataset where bin 1 has zero events.
    df = pd.DataFrame(
        {
            "x": [-1, -1, -1, 1, 1, 1, 1, 1],
            "y": [0, 0, 0, 0, 0, 0, 0, 1],  # bin 1 (x>=0) has 1 event; bin 0 has 0.
        }
    )
    spec = mc.linear(
        target="y", features=woe("x", bins=manual(edges=[0.0])),
        loss=mc.logistic,
    )
    fitted_spec = fit_binnings(spec, df)
    check = AtLeastOneEventPerBin()
    res = check.check(fitted_spec, df)
    assert not res.passed
    assert "zero events" in res.message or "zero non-events" in res.message


def test_minimum_bin_size_passes_when_all_bins_large_enough(synthetic_logistic_data):
    df = synthetic_logistic_data
    spec = mc.linear(
        target="y", features=woe("x", bins=monotonic(min_bin_size=0.10)),
        loss=mc.logistic,
    )
    fitted_spec = fit_binnings(spec, df)
    check = MinimumBinSize(min_fraction=0.05)
    res = check.check(fitted_spec, df)
    assert res.passed


def test_minimum_bin_size_fires_on_tiny_bin():
    df = pd.DataFrame(
        {
            "x": [0.0] * 100 + [10.0] * 100 + [200.0] * 3,  # bin 3 is tiny
            "y": [0] * 50 + [1] * 50 + [0] * 50 + [1] * 50 + [0, 1, 0],
        }
    )
    spec = mc.linear(
        target="y", features=woe("x", bins=manual(edges=[5.0, 50.0])),
        loss=mc.logistic,
    )
    fitted_spec = fit_binnings(spec, df)
    check = MinimumBinSize(min_fraction=0.10)
    res = check.check(fitted_spec, df)
    assert not res.passed
    assert "below" in res.message


def test_monotonic_event_rate_skips_when_strategy_is_not_monotonic():
    df = pd.DataFrame(
        {"x": [0.0] * 100 + [10.0] * 100, "y": [0] * 70 + [1] * 30 + [1] * 60 + [0] * 40}
    )
    spec = mc.linear(
        target="y", features=woe("x", bins=manual(edges=[5.0])),
        loss=mc.logistic,
    )
    fitted_spec = fit_binnings(spec, df)
    check = MonotonicEventRate()
    res = check.check(fitted_spec, df)
    # Manual strategy is not monotonic; check skips with passed=True.
    assert res.passed
    assert "skipped" in res.message.lower()


# ---------------------------------------------------------------------------
# Predict-time behaviour (missing + unseen handling)
# ---------------------------------------------------------------------------


def test_predict_routes_nan_to_missing_bin():
    """NaN at fit time → (Missing) bin; NaN at predict time goes there."""
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, np.nan, np.nan, 5.0, 6.0, 7.0, 8.0],
            "y": [0, 0, 0, 1, 1, 0, 1, 1, 0, 1],
        }
    )
    spec = mc.linear(
        target="y", features=woe("x", bins=manual(edges=[3.5])),
        loss=mc.logistic, penalty=mc.l2(0.1),
    )
    fitted_spec = fit_binnings(spec, df)
    result = fitted_spec.features[0].fitted
    assert result.has_missing_bin
    assert result.bin_labels[-1] == "(Missing)"

    # Predict on NaN -> should use the (Missing) WoE.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = mc.solve(
            fitted_spec, df, on_violation="ignore",
            suppress=(NoPerfectSeparation,),
        )
    new_df = pd.DataFrame({"x": [np.nan, 2.0, 6.0]})
    yhat = mc.predict(sol, new_df)
    assert len(yhat) == 3
    assert not yhat.isna().any()


def test_predict_assigns_zero_woe_to_unseen_categories():
    """Categorical: unseen levels at predict time get WoE = 0."""
    from model_crafter.terms.binning import categorical

    df = pd.DataFrame(
        {
            "region": ["north"] * 50 + ["south"] * 50,
            "y": [0] * 30 + [1] * 20 + [0] * 20 + [1] * 30,
        }
    )
    spec = mc.linear(
        target="y", features=woe("region", bins=categorical(group_rare=0.0)),
        loss=mc.logistic, penalty=mc.l2(0.1),
    )
    fitted_spec = fit_binnings(spec, df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = mc.solve(
            fitted_spec, df, on_violation="ignore",
            suppress=(NoPerfectSeparation,),
        )

    new_df = pd.DataFrame({"region": ["north", "south", "east"]})
    with warnings.catch_warnings(record=True) as warns:
        warnings.simplefilter("always")
        yhat = mc.predict(sol, new_df)
    # Unseen "east" → WoE 0 → η = intercept → check warning was emitted.
    assert any("unseen" in str(w.message).lower() for w in warns)
    assert len(yhat) == 3


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_expand_without_fitted_state_errors_clearly():
    """A WoETerm with no .fitted and no fit_state raises a useful error."""
    t = woe("x", bins=monotonic())
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="fit_binnings"):
        t.expand(df, fit_state=None)


def test_fit_binnings_rejects_non_binary_target():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [0.5, 0.7, 1.2]})
    spec = mc.linear(
        target="y", features=woe("x", bins=monotonic()), loss=mc.logistic,
    )
    with pytest.raises(ValueError, match="binary"):
        fit_binnings(spec, df)


def test_fit_binnings_missing_target_column_errors():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    spec = mc.linear(
        target="y", features=woe("x", bins=monotonic()), loss=mc.logistic,
    )
    with pytest.raises(KeyError, match="target"):
        fit_binnings(spec, df)


def test_binning_result_event_rate_handles_empty_bin():
    """BinningResult.event_rate returns 0.0 for empty bins."""
    r = BinningResult(
        column="x", kind="numeric", edges=(0.0,), categories=(),
        bin_labels=("bin0", "bin1"), has_missing_bin=False,
        n_events=(0, 5), n_nonevents=(0, 10),
        woe=(0.0, 0.0), iv=0.0,
    )
    assert r.event_rate == (0.0, 5.0 / 15.0)


# ---------------------------------------------------------------------------
# End-to-end: WoE pipeline through solve + predict
# ---------------------------------------------------------------------------


def test_woe_end_to_end_pipeline_yields_probabilities(synthetic_logistic_data):
    """A WoE-logistic pipeline returns probabilities in [0,1]."""
    df = synthetic_logistic_data
    spec = mc.linear(
        target="y", features=woe("x", bins=monotonic(min_bin_size=0.10)),
        loss=mc.logistic, penalty=mc.l2(0.1),
    )
    fitted_spec = fit_binnings(spec, df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = mc.solve(
            fitted_spec, df, on_violation="ignore",
            suppress=(NoPerfectSeparation,),
        )
    yhat = mc.predict(sol, df)
    assert ((yhat >= 0.0) & (yhat <= 1.0)).all()


def test_binnedterm_pass_through_with_binned():
    """BinnedTerm constructor wires through correctly."""
    t = binned("income", bins=monotonic())
    assert isinstance(t, BinnedTerm)
    assert t.column == "income"


def test_binnedterm_does_not_declare_woe_monotonicity_preserved():
    """BinnedTerm declares only the HARD bin-based prerequisites."""
    t = binned("x", bins=monotonic())
    names = {type(a).__name__ for a in t.assumptions}
    assert "WoEMonotonicityPreserved" not in names
    assert "AtLeastOneEventPerBin" in names
    assert "MinimumBinSize" in names


def test_assumption_error_imported_from_assumptions_module():
    """Sanity: the AssumptionError class is reachable for tests."""
    from model_crafter.assumptions import AssumptionError as AE
    assert AE is AssumptionError
