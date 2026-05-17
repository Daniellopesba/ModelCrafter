"""Tests for ``mc.performance_over_time`` and ``TemporalPerformanceReport``.

These tests verify Task P5.A acceptance criterion (AGENTS.md / DESIGN.md
§8 Phase 5):

  On a synthetic non-stationary panel where AUC drift is introduced by
  construction, ``performance_over_time`` reproduces the known AUC drift
  to ``1e-4`` across windows.

Synthetic dataset
-----------------

We use a *deterministic interleaving* construction so the AUC of each
validation window is computable in closed form from the data alone — no
random sampling, no Monte-Carlo tolerance. Within each window we place

* ``m`` positives at distinct scores ``[k_i+1, k_i+2, ..., k_i+m]``
* ``m`` negatives at distinct scores ``[1, 2, ..., k_i]`` (the lowest
  ``k_i`` slots) plus ``[k_i+m+1, ..., 2m]`` (the highest ``m-k_i``).

All ``2m`` scores in a window are distinct integers, so there are no
ties. For every positive (score ``> k_i``) the number of negatives below
it is exactly ``k_i``; the number of (pos, neg) pairs with pos > neg is
``m * k_i``; AUC ``= k_i / m``. Pick ``k_i`` decreasing across windows
to induce known AUC drift.

The fitted ``Solution`` is an OLS with a single feature ``score`` and
no intercept on a training frame where ``y == score``; the closed-form
coefficient is exactly ``1.0`` and predictions on any data frame equal
that frame's ``score`` column to machine precision. The validation
windows therefore feed *exactly* our engineered scores into the AUC
primitive, which is the bridge between the analytical construction and
the reported metric value.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

import model_crafter as mc
from model_crafter.performance import PerformanceReport
from model_crafter.performance.over_time import (
    TemporalPerformanceReport,
    performance_over_time,
)

# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

WINDOW_PARAMS = [
    # (k_i, expected AUC = k_i / m for m = 100)
    (90, 0.90),
    (70, 0.70),
    (50, 0.50),
    (30, 0.30),
    (10, 0.10),
]
M_PER_CLASS = 100
WINDOW_SIZE_ROWS = 2 * M_PER_CLASS  # 200 rows per window


def _build_panel() -> tuple[pd.DataFrame, pd.DataFrame, list[float]]:
    """Return ``(panel, train, expected_aucs)``.

    ``panel`` is a chronologically-ordered DataFrame containing five
    consecutive 200-row blocks. Block ``i`` (i = 0..4) has the
    deterministic interleaving pattern described in the module docstring,
    so its AUC under the identity-predictor is ``expected_aucs[i]``
    exactly.

    ``train`` is a small frame used to fit the identity OLS solution
    (target ``= score``, single feature ``score``, no intercept).
    """
    rows: list[dict[str, Any]] = []
    expected: list[float] = []
    # Start dates one block apart; we use a wide spacing (~30 day gap
    # between blocks) so the splitter can isolate them.
    block_start = pd.Timestamp("2024-01-01")
    # pd.Timedelta returns ``Timedelta | NaTType`` in pyright's view —
    # ``cast`` here lets the type-checker see the intended scalar arithmetic.
    block_offset: pd.Timedelta = pd.Timedelta("30D")  # type: ignore[assignment]
    intra_row_delta: pd.Timedelta = pd.Timedelta("1h")  # type: ignore[assignment]
    # Nudge every row 1 hour inside its block so the row times don't
    # land exactly on splitter boundaries.
    row_offset: pd.Timedelta = pd.Timedelta("1h")  # type: ignore[assignment]
    for i, (k_i, target_auc) in enumerate(WINDOW_PARAMS):
        m = M_PER_CLASS
        # Score slots 1..2m. Negatives in [1..k_i] and [k_i+m+1..2m];
        # positives in [k_i+1..k_i+m].
        for slot in range(1, 2 * m + 1):
            score = float(slot)
            label = 1 if k_i + 1 <= slot <= k_i + m else 0
            rows.append(
                {
                    "score": score,
                    "y": label,
                    "t": block_start + i * block_offset + row_offset
                    + (slot - 1) * intra_row_delta,
                    "w": 1.0,
                }
            )
        expected.append(target_auc)
    panel = pd.DataFrame(rows)
    # Training frame for the identity OLS: anything diverse will do. Use
    # a 50-point linspace where y == score; the optimal coefficient is 1.
    train = pd.DataFrame(
        {"score": np.linspace(0.0, 10.0, 50), "y": np.linspace(0.0, 10.0, 50)}
    )
    return panel, train, expected


def _identity_solution() -> Any:
    """Fit an OLS Solution that predicts ``score`` exactly."""
    _, train, _ = _build_panel()
    spec = mc.linear(
        target="y",
        features=["score"],
        loss=mc.squared_error,
        intercept=False,
    )
    return mc.solve(spec, train)


@pytest.fixture
def panel_and_solution():
    panel, _train, expected = _build_panel()
    sol = _identity_solution()
    return sol, panel, expected


def _windowing_splitter():
    """A splitter that yields exactly the 5 engineered validation blocks.

    The five blocks are 200 rows each (total 1000 rows). ``purged_kfold``
    with ``n_folds=5`` slices the time-sorted frame into five equal-row
    buckets, which is the cleanest way to land each engineered block in
    its own validation window — the alternative, ``rolling_window``,
    requires the row timestamps to fall *strictly* inside each train /
    valid half-open interval, which is fiddly when block boundaries align
    with the splitter's step. Using ``purged_kfold`` here keeps the test
    focused on the metric drift rather than the splitter mechanics.

    For this test we only care about the *validation* slices; the model
    is fixed and pre-fit.
    """
    return mc.purged_kfold(time_col="t", n_folds=5, gap="0D")


# ---------------------------------------------------------------------------
# Acceptance criterion: reproduces known AUC drift to 1e-4
# ---------------------------------------------------------------------------


def test_performance_over_time_reproduces_known_AUC_drift(panel_and_solution):
    """Acceptance criterion (AGENTS.md P5.A / DESIGN.md §8 Phase 5):

    "On a synthetic non-stationary panel where AUC drift is introduced
    by construction, ``performance_over_time`` reproduces the known AUC
    drift to ``1e-4`` across windows."
    """
    sol, panel, expected = panel_and_solution
    splitter = _windowing_splitter()
    perf_t = performance_over_time(sol, panel, splitter)

    # The rolling splitter yields one (train, valid) pair per advance of
    # ``step`` after the initial training window — blocks 1..4 are valid
    # windows. We rebuild the expected list to match the splitter output
    # by iterating it directly so the test is splitter-agnostic.
    valid_blocks = [valid for _train, valid in splitter.split(panel)]
    assert len(valid_blocks) == len(perf_t.reports)
    expected_per_window: list[float] = []
    for vb in valid_blocks:
        # Each block contains exactly one engineered period; the lowest
        # positive's score equals ``k_i + 1`` for that block, which
        # uniquely identifies the entry in WINDOW_PARAMS.
        pos_min = int(vb.loc[vb["y"] == 1, "score"].min())
        k_i = pos_min - 1
        match = next(p for p in WINDOW_PARAMS if p[0] == k_i)
        expected_per_window.append(match[1])
    assert expected_per_window == [auc for _, auc in WINDOW_PARAMS]
    _ = expected  # imported for readability in the module docstring

    # Pull reported AUCs straight from the summary DataFrame.
    reported_auc = perf_t.summary["auc"].to_numpy()
    np.testing.assert_allclose(
        reported_auc, np.array(expected_per_window), atol=1e-4
    )

    # Cross-check: the summary's AUC column matches the per-window
    # PerformanceReport.discrimination.auc.value to machine precision —
    # the summary is the *same* numbers, just lifted into a DataFrame.
    for row_auc, rep in zip(reported_auc, perf_t.reports):
        assert np.isclose(row_auc, rep.discrimination.auc.value, atol=1e-12)


# ---------------------------------------------------------------------------
# Interface contract tests (AGENTS.md P5.A pinned signature)
# ---------------------------------------------------------------------------


def test_returns_TemporalPerformanceReport(panel_and_solution):
    sol, panel, _ = panel_and_solution
    perf_t = performance_over_time(sol, panel, _windowing_splitter())
    assert isinstance(perf_t, TemporalPerformanceReport)
    assert isinstance(perf_t.summary, pd.DataFrame)
    assert isinstance(perf_t.reports, tuple)
    assert all(isinstance(r, PerformanceReport) for r in perf_t.reports)


def test_summary_index_is_validation_midpoints(panel_and_solution):
    """The summary DataFrame is indexed by the midpoint of each
    validation window's time column (DESIGN.md §3.3 temporal-performance
    block: ``time-indexed DataFrame: AUC, KS, PSI, Brier per window``).
    """
    sol, panel, _ = panel_and_solution
    splitter = _windowing_splitter()
    perf_t = performance_over_time(sol, panel, splitter)
    midpoints = []
    for _train, valid in splitter.split(panel):
        ts = pd.to_datetime(valid["t"])
        midpoints.append(ts.min() + (ts.max() - ts.min()) / 2)
    assert list(perf_t.summary.index) == midpoints


def test_summary_columns_without_reference(panel_and_solution):
    sol, panel, _ = panel_and_solution
    perf_t = performance_over_time(sol, panel, _windowing_splitter())
    # PSI is absent because we didn't pass a reference.
    expected_cols = {"n_obs", "n_events", "auc", "ks", "brier", "log_loss",
                     "mean_p"}
    assert set(perf_t.summary.columns) == expected_cols


def test_summary_includes_psi_when_reference_provided(panel_and_solution):
    sol, panel, _ = panel_and_solution
    # Use an arbitrary chunk of the panel as the reference.
    reference = panel.iloc[:WINDOW_SIZE_ROWS].copy()
    perf_t = performance_over_time(
        sol, panel, _windowing_splitter(), reference=reference
    )
    assert "psi" in perf_t.summary.columns
    assert bool(perf_t.summary["psi"].notna().all())
    # Reports likewise carry a stability sub-report when reference is set.
    for r in perf_t.reports:
        assert r.stability is not None


def test_weights_column_resolved_per_window(panel_and_solution):
    """The ``weights=`` argument is a column name resolved against each
    validation window (DESIGN.md §3.5: ``weights=`` everywhere).
    """
    sol, panel, _ = panel_and_solution
    # Uniform weights — output should match the unweighted call.
    perf_w = performance_over_time(
        sol, panel, _windowing_splitter(), weights="w"
    )
    perf_u = performance_over_time(sol, panel, _windowing_splitter())
    np.testing.assert_allclose(
        perf_w.summary["auc"].to_numpy(),
        perf_u.summary["auc"].to_numpy(),
        atol=1e-12,
    )


def test_summary_n_obs_and_n_events(panel_and_solution):
    sol, panel, _ = panel_and_solution
    perf_t = performance_over_time(sol, panel, _windowing_splitter())
    # Each window has 200 rows with 100 positives by construction.
    assert (perf_t.summary["n_obs"] == WINDOW_SIZE_ROWS).all()
    assert (perf_t.summary["n_events"] == M_PER_CLASS).all()


def test_repr_includes_summary_layout(panel_and_solution):
    sol, panel, _ = panel_and_solution
    perf_t = performance_over_time(sol, panel, _windowing_splitter())
    s = repr(perf_t)
    assert "TemporalPerformanceReport" in s
    assert "auc" in s
    # One row per window.
    for r in perf_t.reports:
        # Each report contributes (at minimum) its AUC to the repr.
        assert f"{r.discrimination.auc.value:.4f}" in s


def test_frozen_dataclass(panel_and_solution):
    sol, panel, _ = panel_and_solution
    perf_t = performance_over_time(sol, panel, _windowing_splitter())
    with pytest.raises((AttributeError, Exception)):
        perf_t.summary = pd.DataFrame()  # type: ignore[misc]


def test_plot_is_lazy_matplotlib(panel_and_solution):
    """The ``plot()`` helper lazily imports matplotlib; we use the Agg
    backend so the test runs headlessly. We only verify the call returns
    without error and yields a matplotlib Figure (DESIGN.md §3.3 doc:
    ``perf_t.plot()`` is a diagnostic helper, not a tested numerical
    primitive).
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    sol, panel, _ = panel_and_solution
    perf_t = performance_over_time(
        sol, panel, _windowing_splitter(),
        reference=panel.iloc[:WINDOW_SIZE_ROWS].copy(),
    )
    fig = perf_t.plot()
    assert fig is not None


def test_empty_validation_window_skipped():
    """A splitter that produces an empty validation slice yields no
    corresponding entry in the report — degenerate windows are skipped
    silently rather than raising (consistent with ``over_time`` in
    P3.B).
    """
    sol = _identity_solution()
    # A panel whose time span is shorter than the splitter's train_size:
    # the rolling splitter terminates immediately without yielding any
    # window. ``performance_over_time`` returns an empty bundle rather
    # than raising.
    rows = [
        {"score": float(s), "y": int(s % 2 == 0),
         "t": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=s)}
        for s in range(1, 21)
    ]
    panel = pd.DataFrame(rows)
    splitter = mc.rolling_window(
        time_col="t", train_size="365D", horizon="30D",
        step="30D", gap="0D",
    )
    perf_t = performance_over_time(sol, panel, splitter)
    assert len(perf_t.reports) == 0
    assert len(perf_t.summary) == 0
