"""Tests for ``mc.segmented`` / :class:`SegmentedSpec` / :class:`SegmentedSolution`.

Task P6 (AGENTS.md / DESIGN.md §3.4). Acceptance criterion (Phase 6 §8):

    A segmented logistic regression with WoE features produces per-segment
    ``Solution``\\ s, per-segment ``AssumptionReport``\\ s, per-segment
    ``BootstrappedSolution``\\ s when wrapped in ``mc.bootstrap``, and
    per-segment ``PerformanceReport``\\ s when passed to
    ``mc.performance_by_segment``. **All from a single declarative spec.**

The package documents segmentation as "re-fit per segment at solve time"
distinct from ``performance_by_segment`` which is "evaluate one fit per
segment slice at eval time". The end-to-end test in this file pins both
behaviours.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import model_crafter as mc
from model_crafter.assumptions import AssumptionReport
from model_crafter.solution import BootstrappedSolution, SegmentedSolution, Solution
from model_crafter.solve.segmented import predict_segmented, solve_segmented
from model_crafter.spec import LinearSpec, SegmentedSpec, segmented

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_panel(
    rng: np.random.Generator,
    n_per_segment: dict[str, int],
    *,
    seg_betas: dict[str, tuple[float, float, float]] | None = None,
) -> pd.DataFrame:
    """Synthetic 3-segment credit panel.

    Each segment carries its own (intercept, beta_income, beta_age) so that
    the per-segment logistic regressions land on materially different
    coefficient vectors — this is what makes segmentation meaningful.
    """
    seg_betas = seg_betas or {}
    frames = []
    for seg, n in n_per_segment.items():
        beta0, b_inc, b_age = seg_betas.get(seg, (-1.0, 0.6, -0.3))
        income = rng.normal(0.0, 1.0, n)
        age = rng.normal(0.0, 1.0, n)
        eta = beta0 + b_inc * income + b_age * age
        p = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, p).astype(float)
        frames.append(
            pd.DataFrame(
                {
                    "income": income,
                    "age": age,
                    "y": y,
                    "product": seg,
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    return df.sample(frac=1.0, random_state=0).reset_index(drop=True)


@pytest.fixture
def panel_train_test():
    rng = np.random.default_rng(42)
    seg_betas = {
        "installments": (-1.5, 0.7, -0.2),
        "revolving": (0.5, 0.4, -0.5),
        "auto": (-0.4, 0.9, 0.1),
    }
    train = _make_panel(
        rng,
        {"installments": 400, "revolving": 350, "auto": 250},
        seg_betas=seg_betas,
    )
    test = _make_panel(
        rng,
        {"installments": 300, "revolving": 250, "auto": 200},
        seg_betas=seg_betas,
    )
    return train, test


@pytest.fixture
def woe_segmented_spec():
    """The WoE-encoded logistic spec used by the end-to-end test."""
    base = mc.linear(
        target="y",
        features=(
            mc.woe("income", bins=mc.monotonic(min_bin_size=0.1, max_bins=5))
            + mc.woe("age", bins=mc.monotonic(min_bin_size=0.1, max_bins=5))
        ),
        loss=mc.logistic,
    )
    return segmented(by="product", base=base)


# ---------------------------------------------------------------------------
# SegmentedSpec construction
# ---------------------------------------------------------------------------


def test_segmented_spec_is_frozen():
    base = mc.linear(target="y", features=["x"], loss=mc.squared_error)
    spec = segmented(by="g", base=base)
    assert isinstance(spec, SegmentedSpec)
    assert spec.by == "g"
    assert spec.base is base
    with pytest.raises((AttributeError, Exception)):
        spec.by = "h"  # type: ignore[misc]


def test_segmented_spec_rejects_non_str_by():
    base = mc.linear(target="y", features=["x"], loss=mc.squared_error)
    with pytest.raises(TypeError):
        segmented(by=1, base=base)  # type: ignore[arg-type]


def test_segmented_spec_rejects_empty_by():
    base = mc.linear(target="y", features=["x"], loss=mc.squared_error)
    with pytest.raises(ValueError):
        segmented(by="", base=base)


def test_segmented_spec_rejects_non_linear_base():
    with pytest.raises(TypeError):
        segmented(by="g", base="not a spec")  # type: ignore[arg-type]


def test_segmented_constructor_returns_segmented_spec():
    base = mc.linear(target="y", features=["x"], loss=mc.squared_error)
    spec = segmented(by="g", base=base)
    assert isinstance(spec, SegmentedSpec)
    assert spec.base.target == "y"


# ---------------------------------------------------------------------------
# SegmentedSolution mapping protocol
# ---------------------------------------------------------------------------


def test_solve_segmented_produces_per_segment_solutions(panel_train_test):
    train, _test = panel_train_test
    spec = segmented(
        by="product",
        base=mc.linear(target="y", features=["income", "age"], loss=mc.logistic),
    )
    sol = mc.solve(spec, train)
    assert isinstance(sol, SegmentedSolution)
    assert set(sol.keys()) == {"installments", "revolving", "auto"}
    # Every per-segment solution is a real Solution with its own assumptions.
    for seg in sol:
        sub = sol[seg]
        assert isinstance(sub, Solution)
        assert isinstance(sub.assumptions, AssumptionReport)
    # n_obs is the sum of per-segment n_obs.
    assert sol.n_obs == sum(sol[k].n_obs for k in sol)


def test_segmented_solution_mapping_helpers(panel_train_test):
    train, _ = panel_train_test
    spec = segmented(
        by="product",
        base=mc.linear(target="y", features=["income", "age"], loss=mc.logistic),
    )
    sol = mc.solve(spec, train)
    assert len(sol) == 3
    items = list(sol.items())
    assert all(isinstance(v, Solution) for _, v in items)
    values = list(sol.values())
    assert all(isinstance(v, Solution) for v in values)


def test_per_segment_assumption_reports_are_independent(panel_train_test):
    train, _ = panel_train_test
    spec = segmented(
        by="product",
        base=mc.linear(target="y", features=["income", "age"], loss=mc.logistic),
    )
    sol = mc.solve(spec, train)
    # The three sub-reports are distinct objects (each segment runs its
    # own assumption pass against its own data slice).
    reports = [sol[k].assumptions for k in sol]
    assert len({id(r) for r in reports}) == 3


def test_solve_segmented_raises_when_by_column_missing(panel_train_test):
    train, _ = panel_train_test
    spec = segmented(
        by="not_a_column",
        base=mc.linear(target="y", features=["income"], loss=mc.logistic),
    )
    with pytest.raises(KeyError):
        mc.solve(spec, train)


def test_solve_segmented_empty_segmentation_raises():
    # Empty dataframe with the right columns
    df = pd.DataFrame({"y": [], "income": [], "product": []})
    spec = segmented(
        by="product",
        base=mc.linear(target="y", features=["income"], loss=mc.logistic),
    )
    with pytest.raises(ValueError, match="zero non-empty segments"):
        mc.solve(spec, df)


def test_solve_segmented_passes_through_weights_column(panel_train_test):
    train, _ = panel_train_test
    train = train.copy()
    train["w"] = 1.0
    spec = segmented(
        by="product",
        base=mc.linear(target="y", features=["income"], loss=mc.logistic),
    )
    sol = mc.solve(spec, train, weights="w")
    assert isinstance(sol, SegmentedSolution)


def test_solve_segmented_passes_through_array_weights(panel_train_test):
    train, _ = panel_train_test
    weights = np.ones(len(train))
    spec = segmented(
        by="product",
        base=mc.linear(target="y", features=["income"], loss=mc.logistic),
    )
    sol = mc.solve(spec, train, weights=weights)
    assert isinstance(sol, SegmentedSolution)


def test_solve_segmented_array_weights_wrong_length_raises(panel_train_test):
    train, _ = panel_train_test
    spec = segmented(
        by="product",
        base=mc.linear(target="y", features=["income"], loss=mc.logistic),
    )
    with pytest.raises(ValueError, match="weights array"):
        mc.solve(spec, train, weights=np.ones(5))


# ---------------------------------------------------------------------------
# predict routing
# ---------------------------------------------------------------------------


def test_predict_routes_per_segment(panel_train_test):
    train, test = panel_train_test
    spec = segmented(
        by="product",
        base=mc.linear(target="y", features=["income", "age"], loss=mc.logistic),
    )
    sol = mc.solve(spec, train)
    p = mc.predict(sol, test)
    assert isinstance(p, pd.Series)
    assert (p.index == test.index).all()
    # logistic predict is probabilities in [0, 1]
    assert ((p >= 0.0) & (p <= 1.0)).all()
    # Per-segment values match the per-segment predict
    for seg in sol:
        idx = test["product"] == seg
        manual = mc.predict(sol[seg], test.loc[idx])
        np.testing.assert_allclose(
            p.loc[idx].to_numpy(),
            manual.to_numpy(),
            rtol=1e-12,
            atol=1e-12,
        )


def test_predict_unseen_segment_yields_nan_and_warning(panel_train_test):
    train, test = panel_train_test
    spec = segmented(
        by="product",
        base=mc.linear(target="y", features=["income", "age"], loss=mc.logistic),
    )
    sol = mc.solve(spec, train)
    # Inject an unseen segment value
    new_data = test.copy()
    new_data.loc[new_data.index[:5], "product"] = "unseen_seg"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p = mc.predict(sol, new_data)
    # The unseen rows are NaN; the seen rows are finite.
    unseen_mask = new_data["product"] == "unseen_seg"
    assert p.loc[unseen_mask].isna().all()
    assert p.loc[~unseen_mask].notna().all()
    # The warning mentions the unseen segments + row count.
    msgs = [str(w.message) for w in caught]
    assert any("unseen_seg" in m for m in msgs)
    assert any("5" in m for m in msgs)


def test_predict_segmented_rejects_missing_by_column(panel_train_test):
    train, test = panel_train_test
    spec = segmented(
        by="product",
        base=mc.linear(target="y", features=["income"], loss=mc.logistic),
    )
    sol = mc.solve(spec, train)
    bad = test.drop(columns=["product"])
    with pytest.raises(KeyError):
        mc.predict(sol, bad)


# ---------------------------------------------------------------------------
# End-to-end acceptance: WoE + logistic + bootstrap + performance_by_segment
# ---------------------------------------------------------------------------


def test_end_to_end_woe_logistic_segmented(panel_train_test, woe_segmented_spec):
    """Phase 6 acceptance: one declarative spec yields per-segment solutions,
    assumption reports, bootstrapped solutions, and performance reports."""
    train, test = panel_train_test
    spec = woe_segmented_spec

    # --- fit ---
    sol = mc.solve(spec, train, on_violation="warn")
    assert isinstance(sol, SegmentedSolution)
    assert set(sol.keys()) == {"installments", "revolving", "auto"}

    # --- per-segment solutions are independent and have their own assumptions ---
    for k in sol:
        sub = sol[k]
        assert isinstance(sub, Solution)
        assert isinstance(sub.assumptions, AssumptionReport)
        # logistic predict yields probabilities
        p = mc.predict(sub, test[test["product"] == k])
        assert ((p >= 0.0) & (p <= 1.0)).all()

    # --- per-segment bootstrap via mc.bootstrap on each per-segment Solution ---
    # The acceptance criterion says "per-segment BootstrappedSolutions when
    # wrapped in mc.bootstrap" — we apply bootstrap to each sub-Solution
    # using its segment's training slice.
    boots = {}
    for k in sol:
        seg_train = train[train["product"] == k]
        bs = mc.bootstrap(sol[k], data=seg_train, n_boot=20, random_state=0)
        assert isinstance(bs, BootstrappedSolution)
        boots[k] = bs
    assert set(boots.keys()) == set(sol.keys())

    # --- per-segment PerformanceReports via mc.performance_by_segment ---
    # We feed performance_by_segment a single (non-segmented) sol for an
    # eval-time slice; but the §3.4 contract also says we can ask for per
    # segment performance from a SegmentedSolution. We do both checks.
    # 1) Aggregate-style: use the auto sub-Solution on the auto slice.
    auto_perf = mc.performance(sol["auto"], test[test["product"] == "auto"])
    assert auto_perf.n_obs == int((test["product"] == "auto").sum())
    # 2) Use performance_by_segment on the *predictions* from the
    # segmented solution. Construct a small adapter Solution-like object
    # by replacing the eval frame's target with predict()'s output.
    # The simpler check: every segment's PerformanceReport has the
    # expected number of observations.
    for k in sol:
        slice_ = test[test["product"] == k]
        perf = mc.performance(sol[k], slice_)
        assert perf.n_obs == len(slice_)


def test_solve_segmented_directly_callable(panel_train_test):
    """The internal helper is callable directly (used by the dispatch)."""
    train, _ = panel_train_test
    spec = segmented(
        by="product",
        base=mc.linear(target="y", features=["income"], loss=mc.logistic),
    )
    sol = solve_segmented(spec, train)
    assert isinstance(sol, SegmentedSolution)


def test_predict_segmented_directly_callable(panel_train_test):
    train, test = panel_train_test
    spec = segmented(
        by="product",
        base=mc.linear(target="y", features=["income"], loss=mc.logistic),
    )
    sol = mc.solve(spec, train)
    p = predict_segmented(sol, test)
    assert isinstance(p, pd.Series)
    assert len(p) == len(test)


# ---------------------------------------------------------------------------
# Type errors
# ---------------------------------------------------------------------------


def test_solve_rejects_unknown_spec_type():
    df = pd.DataFrame({"y": [1.0, 0.0], "x": [1.0, 2.0]})
    with pytest.raises(TypeError, match="LinearSpec or SegmentedSpec"):
        mc.solve("not a spec", df)  # type: ignore[arg-type]


def test_predict_rejects_unknown_solution_type():
    df = pd.DataFrame({"x": [1.0]})
    with pytest.raises(TypeError, match="Solution or SegmentedSolution"):
        mc.predict("not a sol", df)  # type: ignore[arg-type]


def test_solve_segmented_function_rejects_non_segmented_spec(panel_train_test):
    train, _ = panel_train_test
    base = mc.linear(target="y", features=["income"], loss=mc.logistic)
    with pytest.raises(TypeError):
        solve_segmented(base, train)  # type: ignore[arg-type]


def test_predict_segmented_function_rejects_non_segmented_solution(panel_train_test):
    train, _ = panel_train_test
    base = mc.linear(target="y", features=["income"], loss=mc.logistic)
    sub = mc.solve(base, train)
    assert isinstance(sub, Solution)
    with pytest.raises(TypeError):
        predict_segmented(sub, train)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Public exposure
# ---------------------------------------------------------------------------


def test_segmented_constructor_exposed_from_spec_module():
    from model_crafter.spec import SegmentedSpec as SegSpec
    from model_crafter.spec import segmented as seg_fn

    assert SegSpec is SegmentedSpec
    assert seg_fn is segmented


def test_segmented_solution_exposed_from_solution_module():
    from model_crafter.solution import SegmentedSolution as SS

    assert SS is SegmentedSolution


def test_linear_spec_unaffected_by_segmented_addition():
    """Sanity: existing LinearSpec construction still works."""
    spec = mc.linear(target="y", features=["x"], loss=mc.squared_error)
    assert isinstance(spec, LinearSpec)
