"""Tests for ``mc.performance_by_segment`` and ``SegmentedPerformanceReport``.

Task P5.B (AGENTS.md). Acceptance criterion (verbatim, Phase 5 §8 #4 in
DESIGN.md and the task-specific phrasing in AGENTS.md):

    On a logistic-regression ``sol`` fit to a dataset with a categorical
    segment column, ``performance_by_segment(sol, test_data,
    by="segment_col")`` produces one ``PerformanceReport`` per segment
    value plus an aggregate report. Verify each segment's ``n_obs`` sums
    to the aggregate ``n_obs`` and each segment's AUC matches a manual
    ``mc.auc(sol, segment_slice)`` to floating-point equality.

The task wires no new dependencies and is pure orchestration over the
existing ``mc.performance``. We segment the held-out evaluation frame —
the ``sol`` is a single fitted ``Solution``; no re-fitting per segment
happens here (that's :class:`mc.segmented` in Phase 6).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import model_crafter as mc
from model_crafter.metrics import auc
from model_crafter.performance import PerformanceReport
from model_crafter.performance.by_segment import (
    SegmentedPerformanceReport,
    performance_by_segment,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _segmented_bernoulli(
    rng: np.random.Generator,
    n_per_segment: dict[str, int],
    *,
    intercept_shift: dict[str, float] | None = None,
) -> pd.DataFrame:
    """A Bernoulli panel with a categorical ``segment`` column.

    Each segment may have a different intercept shift so segment-level
    performance varies in interesting ways.
    """
    intercept_shift = intercept_shift or {}
    frames = []
    for seg, n in n_per_segment.items():
        x1 = rng.normal(0.0, 1.0, n)
        x2 = rng.normal(0.0, 1.0, n)
        shift = intercept_shift.get(seg, 0.0)
        p = 1.0 / (1.0 + np.exp(-(shift + 0.4 + 0.7 * x1 + 0.3 * x2)))
        y = rng.binomial(1, p).astype(float)
        frames.append(
            pd.DataFrame({"x1": x1, "x2": x2, "y": y, "segment": seg})
        )
    df = pd.concat(frames, ignore_index=True)
    # Shuffle so segments are interleaved (mirrors real held-out data).
    return df.sample(frac=1.0, random_state=0).reset_index(drop=True)


@pytest.fixture
def fitted_logistic_segmented():
    """Logistic regression fit on the *aggregate* training set; three segments.

    Per the task contract, segmentation happens at *evaluation* time on
    the held-out frame — not at fit time.
    """
    rng = np.random.default_rng(0)
    train = _segmented_bernoulli(
        rng,
        n_per_segment={"A": 400, "B": 350, "C": 250},
        intercept_shift={"A": -0.3, "B": 0.0, "C": 0.5},
    )
    test = _segmented_bernoulli(
        rng,
        n_per_segment={"A": 300, "B": 250, "C": 200},
        intercept_shift={"A": -0.3, "B": 0.0, "C": 0.5},
    )
    spec = mc.linear(
        target="y",
        features=["x1", "x2"],
        loss=mc.logistic,
    )
    sol = mc.solve(spec, train)
    return sol, train, test


# ---------------------------------------------------------------------------
# Type structure
# ---------------------------------------------------------------------------


def test_returns_SegmentedPerformanceReport(fitted_logistic_segmented):
    sol, _train, test = fitted_logistic_segmented
    rep = performance_by_segment(sol, test, by="segment")
    assert isinstance(rep, SegmentedPerformanceReport)
    assert isinstance(rep.aggregate, PerformanceReport)
    assert isinstance(rep.segments, dict)
    # One entry per unique segment value.
    assert set(rep.segments.keys()) == {"A", "B", "C"}
    for v in rep.segments.values():
        assert isinstance(v, PerformanceReport)


# ---------------------------------------------------------------------------
# Acceptance: per-segment ``n_obs`` sum to aggregate ``n_obs``
# ---------------------------------------------------------------------------


def test_segment_n_obs_sum_to_aggregate(fitted_logistic_segmented):
    """Acceptance (P5.B): each segment's ``n_obs`` sums to the aggregate
    ``n_obs`` and each segment's AUC matches ``mc.auc(sol, segment_slice)``
    to floating-point equality.
    """
    sol, _train, test = fitted_logistic_segmented
    rep = performance_by_segment(sol, test, by="segment")
    total = sum(p.n_obs for p in rep.segments.values())
    assert total == rep.aggregate.n_obs == len(test)


def test_segment_auc_matches_manual_slice(fitted_logistic_segmented):
    """Acceptance (P5.B): each segment AUC equals ``mc.auc`` on the slice."""
    sol, _train, test = fitted_logistic_segmented
    rep = performance_by_segment(sol, test, by="segment")
    for seg_key, seg_report in rep.segments.items():
        slice_ = test[test["segment"] == seg_key]
        manual = auc(sol, slice_).value
        assert math.isclose(
            seg_report.discrimination.auc.value, manual, abs_tol=0.0, rel_tol=0.0
        ), f"AUC mismatch for segment {seg_key!r}: {seg_report.discrimination.auc.value} vs {manual}"


def test_aggregate_matches_full_performance(fitted_logistic_segmented):
    """The aggregate field is exactly ``mc.performance(sol, data, ...)``."""
    sol, _train, test = fitted_logistic_segmented
    rep = performance_by_segment(sol, test, by="segment")
    full = mc.performance(sol, test)
    assert rep.aggregate.n_obs == full.n_obs
    assert rep.aggregate.n_events == full.n_events
    assert math.isclose(
        rep.aggregate.discrimination.auc.value,
        full.discrimination.auc.value,
        abs_tol=0.0,
        rel_tol=0.0,
    )
    assert math.isclose(
        rep.aggregate.calibration.brier.value,
        full.calibration.brier.value,
        abs_tol=0.0,
        rel_tol=0.0,
    )


# ---------------------------------------------------------------------------
# Numeric segment keys are coerced to strings
# ---------------------------------------------------------------------------


def test_numeric_segment_keys_coerced_to_strings(fitted_logistic_segmented):
    sol, _train, test = fitted_logistic_segmented
    test = test.copy()
    # Encode segment as integer codes.
    mapping = {"A": 0, "B": 1, "C": 2}
    test["segment_int"] = test["segment"].map(mapping).astype(int)
    rep = performance_by_segment(sol, test, by="segment_int")
    # All dict keys must be strings.
    for k in rep.segments:
        assert isinstance(k, str)
    assert set(rep.segments.keys()) == {"0", "1", "2"}


# ---------------------------------------------------------------------------
# Weights pass-through
# ---------------------------------------------------------------------------


def test_weights_pass_through(fitted_logistic_segmented):
    sol, _train, test = fitted_logistic_segmented
    test = test.copy()
    test["w"] = 1.0
    rep_w = performance_by_segment(sol, test, by="segment", weights="w")
    rep_u = performance_by_segment(sol, test.drop(columns=["w"]), by="segment")
    # Uniform weights => identical per-segment AUCs.
    for k in rep_u.segments:
        assert math.isclose(
            rep_w.segments[k].discrimination.auc.value,
            rep_u.segments[k].discrimination.auc.value,
        )


# ---------------------------------------------------------------------------
# Reference pass-through (PSI stability lit up by an external reference frame)
# ---------------------------------------------------------------------------


def test_reference_pass_through_enables_stability(fitted_logistic_segmented):
    sol, train, test = fitted_logistic_segmented
    rep = performance_by_segment(sol, test, by="segment", reference=train)
    assert rep.aggregate.stability is not None
    for seg_report in rep.segments.values():
        assert seg_report.stability is not None


# ---------------------------------------------------------------------------
# Empty segments
# ---------------------------------------------------------------------------


def test_empty_segments_dropped(fitted_logistic_segmented):
    """If a segment column carries a value with zero observations after
    filtering, it must not appear as an empty entry. (We pick "drop" over
    "report" — see notes/P5.B.md for the rationale.)
    """
    sol, _train, test = fitted_logistic_segmented
    test = test.copy()
    # Add a category that exists in dtype but has zero rows (pandas
    # ``observed=True`` semantics).
    test["segment"] = pd.Categorical(
        test["segment"], categories=["A", "B", "C", "D"]
    )
    rep = performance_by_segment(sol, test, by="segment")
    assert "D" not in rep.segments
    assert set(rep.segments.keys()) == {"A", "B", "C"}


# ---------------------------------------------------------------------------
# Repr layout
# ---------------------------------------------------------------------------


def test_repr_shows_aggregate_then_segments(fitted_logistic_segmented):
    sol, _train, test = fitted_logistic_segmented
    rep = performance_by_segment(sol, test, by="segment")
    s = repr(rep)
    assert s.startswith("SegmentedPerformanceReport")
    # Aggregate header tag.
    assert "aggregate" in s.lower()
    # Per-segment lines for every segment key.
    for k in rep.segments:
        assert k in s
    # Compact summary should mention the per-segment quantities.
    assert "AUC" in s
    assert "KS" in s
    assert "Brier" in s


# ---------------------------------------------------------------------------
# Frozen dataclass + slots
# ---------------------------------------------------------------------------


def test_segmented_report_is_frozen_with_slots(fitted_logistic_segmented):
    sol, _train, test = fitted_logistic_segmented
    rep = performance_by_segment(sol, test, by="segment")
    with pytest.raises((AttributeError, TypeError)):
        rep.aggregate = None  # type: ignore[misc]
    # ``__slots__`` is the marker for slotted dataclasses.
    assert hasattr(type(rep), "__slots__")


# ---------------------------------------------------------------------------
# Unknown ``by`` column raises with a clear message
# ---------------------------------------------------------------------------


def test_unknown_by_column_raises(fitted_logistic_segmented):
    sol, _train, test = fitted_logistic_segmented
    with pytest.raises(KeyError, match="not_a_column"):
        performance_by_segment(sol, test, by="not_a_column")
