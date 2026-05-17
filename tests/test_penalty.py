"""Tests for the Penalty protocol, NoPenalty (P1.A) and l1/l2/PenaltySum (P2.A).

Pins the public-interface contract from AGENTS.md Task P1.A for the
protocol + ``NoPenalty``, and Task P2.A for the ``l1`` / ``l2`` /
``PenaltySum`` extensions.

Acceptance criteria quoted in test docstrings (AGENTS.md §Phase 2, P2.A):

1. ``l1(0.5) + l2(0.5)`` produces a ``PenaltySum`` with two parts;
   iterating yields them in order; ``len(...)`` is 2.
2. ``ComparableFeatureScales`` (declared via P1.B's framework on the
   L1/L2 penalties' ``assumptions`` tuple) fires on a synthetic dataset
   with feature std ratio > 100.
3. ``value()`` and ``prox()`` match hand-derived references to 1e-12 for
   L1, L2, and an elastic-net combination across random
   ``(beta, step, lam)`` triples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model_crafter.penalty import (
    L1Penalty,
    L2Penalty,
    NoPenalty,
    Penalty,
    PenaltySum,
    l1,
    l2,
)

# ---------------------------------------------------------------------------
# P1.A surface (kept green by P2.A — do not regress)
# ---------------------------------------------------------------------------


def test_no_penalty_is_a_penalty() -> None:
    """NoPenalty() satisfies the Penalty protocol."""
    p = NoPenalty()
    assert isinstance(p, Penalty)


def test_no_penalty_has_empty_assumptions() -> None:
    """NoPenalty has no assumptions to declare."""
    p = NoPenalty()
    assert hasattr(p, "assumptions")
    assert p.assumptions == ()


def test_no_penalty_value_is_zero() -> None:
    """NoPenalty.value(beta) == 0 for any beta."""
    p = NoPenalty()
    beta = np.array([1.0, -2.0, 3.0])
    assert p.value(beta) == 0.0
    assert p.value(np.zeros(0)) == 0.0


def test_no_penalty_addition_with_term_raises_typeerror() -> None:
    """A Penalty + Term must be a TypeError pointing at the right argument."""
    from model_crafter.terms.base import RawTerm

    with pytest.raises(TypeError, match="features=.*penalty="):
        _ = NoPenalty() + RawTerm("x")  # type: ignore[operator]


def test_no_penalty_is_frozen() -> None:
    """NoPenalty is immutable."""
    p = NoPenalty()
    with pytest.raises((AttributeError, Exception)):
        p.foo = 1  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------


def test_l1_constructor_returns_l1_penalty() -> None:
    """`l1(lam)` returns an L1Penalty with the right lambda."""
    p = l1(0.25)
    assert isinstance(p, L1Penalty)
    assert p.lam == 0.25
    assert isinstance(p, Penalty)


def test_l2_constructor_returns_l2_penalty() -> None:
    """`l2(lam)` returns an L2Penalty with the right lambda."""
    p = l2(0.5)
    assert isinstance(p, L2Penalty)
    assert p.lam == 0.5
    assert isinstance(p, Penalty)


def test_l1_rejects_negative_lambda() -> None:
    """Negative lambda is a programming error (ESL §3.4 lambda >= 0)."""
    with pytest.raises(ValueError, match="lam.*>=.*0"):
        l1(-0.01)


def test_l2_rejects_negative_lambda() -> None:
    """Negative lambda is a programming error (ESL §3.4 lambda >= 0)."""
    with pytest.raises(ValueError, match="lam.*>=.*0"):
        l2(-0.01)


def test_l1_rejects_non_finite_lambda() -> None:
    """NaN/Inf lambdas are not valid penalties."""
    with pytest.raises(ValueError):
        l1(float("nan"))
    with pytest.raises(ValueError):
        l1(float("inf"))


def test_l2_rejects_non_finite_lambda() -> None:
    """NaN/Inf lambdas are not valid penalties."""
    with pytest.raises(ValueError):
        l2(float("nan"))
    with pytest.raises(ValueError):
        l2(float("inf"))


# ---------------------------------------------------------------------------
# L1: value + prox math
# ---------------------------------------------------------------------------


def test_l1_value_matches_hand_derived() -> None:
    """L1: P(beta) = lam * sum(|beta_j|), matched to 1e-12."""
    p = l1(0.3)
    beta = np.array([1.0, -2.5, 0.0, 4.25])
    expected = 0.3 * (1.0 + 2.5 + 0.0 + 4.25)
    assert p.value(beta) == pytest.approx(expected, abs=1e-12)


def test_l1_value_zero_when_lambda_zero() -> None:
    """At lam=0, the L1 penalty is identically zero."""
    p = l1(0.0)
    assert p.value(np.array([5.0, -3.0, 2.0])) == 0.0


def test_l1_prox_soft_thresholds() -> None:
    """L1 prox is element-wise soft-thresholding:
    sign(beta_j) * max(|beta_j| - lam*step, 0)."""
    p = l1(0.4)
    step = 0.5
    beta = np.array([1.0, -1.0, 0.1, -0.1, 0.2, -0.2, 5.0, -5.0])
    # threshold = lam * step = 0.4 * 0.5 = 0.2
    # |0.1| < 0.2 → 0; |0.2| == 0.2 → 0; |1.0| - 0.2 = 0.8 → sign-preserving
    expected = np.array([0.8, -0.8, 0.0, 0.0, 0.0, 0.0, 4.8, -4.8])
    got = p.prox(beta, step)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_l1_prox_zero_lambda_is_identity() -> None:
    """At lam=0 the L1 prox is the identity map."""
    p = l1(0.0)
    beta = np.array([1.0, -2.5, 0.0, 4.25])
    np.testing.assert_allclose(p.prox(beta, 0.7), beta, atol=1e-12)


def test_l1_prox_zero_step_is_identity() -> None:
    """At step=0 every prox is the identity (no proximal motion)."""
    p = l1(0.4)
    beta = np.array([1.0, -2.5, 0.0, 4.25])
    np.testing.assert_allclose(p.prox(beta, 0.0), beta, atol=1e-12)


# ---------------------------------------------------------------------------
# L2: value + prox math
# ---------------------------------------------------------------------------


def test_l2_value_matches_hand_derived() -> None:
    """L2: P(beta) = (lam / 2) * sum(beta_j^2). The half-norm is the
    glmnet parameterisation (ESL §3.4.1)."""
    p = l2(0.6)
    beta = np.array([1.0, -2.0, 3.0])
    expected = 0.5 * 0.6 * (1.0**2 + 2.0**2 + 3.0**2)
    assert p.value(beta) == pytest.approx(expected, abs=1e-12)


def test_l2_value_zero_when_lambda_zero() -> None:
    """At lam=0, the L2 penalty is identically zero."""
    p = l2(0.0)
    assert p.value(np.array([5.0, -3.0, 2.0])) == 0.0


def test_l2_prox_shrinks() -> None:
    """L2 prox is element-wise shrinkage: beta_j / (1 + lam*step)."""
    p = l2(2.0)
    step = 0.25
    beta = np.array([4.0, -2.0, 0.0, 0.5])
    # 1 + 2.0 * 0.25 = 1.5
    expected = beta / 1.5
    got = p.prox(beta, step)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_l2_prox_zero_lambda_is_identity() -> None:
    """At lam=0 the L2 prox is the identity map."""
    p = l2(0.0)
    beta = np.array([1.0, -2.5, 0.0, 4.25])
    np.testing.assert_allclose(p.prox(beta, 0.7), beta, atol=1e-12)


def test_l2_prox_zero_step_is_identity() -> None:
    """At step=0 every prox is the identity."""
    p = l2(0.7)
    beta = np.array([1.0, -2.5, 0.0, 4.25])
    np.testing.assert_allclose(p.prox(beta, 0.0), beta, atol=1e-12)


# ---------------------------------------------------------------------------
# PenaltySum: composition algebra
# ---------------------------------------------------------------------------


def test_l1_plus_l2_is_penalty_sum_with_two_parts() -> None:
    """Acceptance #1: l1(0.5) + l2(0.5) produces a PenaltySum with two
    parts; iterating yields them in order; len(...) is 2."""
    p = l1(0.5) + l2(0.5)
    assert isinstance(p, PenaltySum)
    parts = list(p)
    assert len(p) == 2
    assert len(parts) == 2
    assert isinstance(parts[0], L1Penalty)
    assert isinstance(parts[1], L2Penalty)
    assert parts[0].lam == 0.5
    assert parts[1].lam == 0.5


def test_l2_plus_l1_preserves_order() -> None:
    """`+` is *not* commutative in terms of ordering of parts —
    l2 + l1 produces (l2, l1) in that order."""
    p = l2(0.1) + l1(0.2)
    assert isinstance(p, PenaltySum)
    parts = list(p)
    assert isinstance(parts[0], L2Penalty)
    assert isinstance(parts[1], L1Penalty)


def test_penalty_sum_flattens_nested_additions() -> None:
    """`(l1 + l2) + l1` flattens to a 3-part PenaltySum, not a nested one."""
    p = (l1(0.1) + l2(0.2)) + l1(0.3)
    assert isinstance(p, PenaltySum)
    assert len(p) == 3
    parts = list(p)
    assert [type(x) for x in parts] == [L1Penalty, L2Penalty, L1Penalty]
    assert [x.lam for x in parts] == [0.1, 0.2, 0.3]
    # No nesting: every part is a non-Sum atomic penalty.
    assert not any(isinstance(x, PenaltySum) for x in parts)


def test_penalty_sum_plus_penalty_sum_flattens() -> None:
    """Sum + Sum is also flat."""
    p = (l1(0.1) + l2(0.2)) + (l1(0.3) + l2(0.4))
    assert isinstance(p, PenaltySum)
    assert len(p) == 4
    assert [type(x) for x in p] == [L1Penalty, L2Penalty, L1Penalty, L2Penalty]


def test_penalty_sum_value_is_sum_of_parts() -> None:
    """PenaltySum.value = sum of part values."""
    p = l1(0.3) + l2(0.6)
    beta = np.array([1.0, -2.0, 3.0])
    expected = (
        0.3 * (abs(1.0) + abs(-2.0) + abs(3.0))
        + 0.5 * 0.6 * (1.0**2 + 2.0**2 + 3.0**2)
    )
    assert p.value(beta) == pytest.approx(expected, abs=1e-12)


def test_penalty_sum_assumptions_are_deduplicated() -> None:
    """L1 and L2 both declare ComparableFeatureScales — PenaltySum dedupes
    to a single instance (so we don't run the same check twice)."""
    p = l1(0.1) + l2(0.2)
    # Both atomic penalties declare ComparableFeatureScales(), which is a
    # frozen dataclass — equal instances should collapse to one.
    assert len(p.assumptions) == 1
    from model_crafter.assumptions import ComparableFeatureScales

    assert isinstance(p.assumptions[0], ComparableFeatureScales)


def test_penalty_sum_addition_with_term_raises() -> None:
    """PenaltySum + Term raises TypeError pointing at features=/penalty=."""
    from model_crafter.terms.base import RawTerm

    p = l1(0.1) + l2(0.2)
    with pytest.raises(TypeError, match="features=.*penalty="):
        _ = p + RawTerm("x")  # type: ignore[operator]


def test_l1_addition_with_term_raises() -> None:
    """L1 + Term raises TypeError pointing at features=/penalty=."""
    from model_crafter.terms.base import RawTerm

    with pytest.raises(TypeError, match="features=.*penalty="):
        _ = l1(0.1) + RawTerm("x")  # type: ignore[operator]


def test_l2_addition_with_term_raises() -> None:
    """L2 + Term raises TypeError pointing at features=/penalty=."""
    from model_crafter.terms.base import RawTerm

    with pytest.raises(TypeError, match="features=.*penalty="):
        _ = l2(0.1) + RawTerm("x")  # type: ignore[operator]


def test_l1_plus_no_penalty_returns_l1() -> None:
    """NoPenalty is the additive identity: L1 + NoPenalty == L1."""
    p = l1(0.3)
    np_ = NoPenalty()
    result = p + np_
    # L1+NoPenalty stays atomic L1 (no PenaltySum wrapping for the identity).
    assert isinstance(result, L1Penalty)
    assert result.lam == 0.3


def test_no_penalty_plus_l1_returns_l1() -> None:
    """NoPenalty + L1 == L1 (NoPenalty's __add__ already handles this)."""
    result = NoPenalty() + l1(0.3)
    assert isinstance(result, L1Penalty)
    assert result.lam == 0.3


def test_l2_plus_no_penalty_returns_l2() -> None:
    """NoPenalty is the additive identity: L2 + NoPenalty == L2."""
    p = l2(0.4)
    result = p + NoPenalty()
    assert isinstance(result, L2Penalty)
    assert result.lam == 0.4


def test_penalty_sum_plus_no_penalty_is_identity() -> None:
    """NoPenalty is the additive identity: PenaltySum + NoPenalty == PenaltySum."""
    p = l1(0.1) + l2(0.2)
    result = p + NoPenalty()
    assert isinstance(result, PenaltySum)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Elastic-net (L1 + L2) prox — composition / closed form
# ---------------------------------------------------------------------------


def _soft_threshold(x: np.ndarray, t: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def test_elastic_net_prox_matches_closed_form() -> None:
    """For elastic net P = l1 * |.| + (l2/2) * .^2, the proximal operator
    of P at step ``s`` is
        prox_P(beta, s) = soft_threshold(beta / (1 + l2 * s), l1 * s)
    (see e.g., Parikh & Boyd §6.5.2). This is the form coordinate descent
    needs in Phase 2 (P2.C)."""
    p = l1(0.3) + l2(0.6)
    beta = np.array([2.0, -0.1, 0.0, -3.0, 0.5])
    step = 0.4
    expected = _soft_threshold(beta / (1.0 + 0.6 * step), 0.3 * step)
    got = p.prox(beta, step)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_elastic_net_prox_order_l2_then_l1_gives_same_result() -> None:
    """L2 + L1 (reversed order in the sum) yields the same elastic-net prox
    because L1 ∘ L2 and L2 ∘ L1 happen to coincide here. The point of the
    test is to lock that property: PenaltySum.prox is order-independent
    for the L1/L2 case."""
    p_a = l1(0.3) + l2(0.6)
    p_b = l2(0.6) + l1(0.3)
    beta = np.array([2.0, -0.1, 0.0, -3.0, 0.5])
    step = 0.4
    np.testing.assert_allclose(
        p_a.prox(beta, step), p_b.prox(beta, step), atol=1e-12
    )


def test_penalty_sum_prox_two_l1_is_one_l1_with_combined_lambda() -> None:
    """For two L1 penalties summed, prox-of-sum equals prox of a single L1
    with combined lambda (composition of soft-thresholds with thresholds
    t1, t2 equals soft-threshold with t1+t2). 1e-12 tolerance."""
    p = l1(0.2) + l1(0.5)
    beta = np.array([2.0, -0.1, 0.0, -3.0, 0.5])
    step = 0.4
    expected = _soft_threshold(beta, (0.2 + 0.5) * step)
    np.testing.assert_allclose(p.prox(beta, step), expected, atol=1e-12)


def test_penalty_sum_prox_two_l2_is_one_l2_with_combined_lambda() -> None:
    """For two L2 penalties summed, prox-of-sum equals prox of a single L2
    with combined lambda (composition of shrinkages with shrink factors
    1/(1+l1*s) and 1/(1+l2*s) does *not* equal 1/(1+(l1+l2)*s); however
    the *value*-derived prox of the sum *is* the latter because the sum
    of two L2 penalties is itself an L2 penalty with lambda = l1+l2).
    1e-12 tolerance."""
    p = l2(0.4) + l2(0.7)
    beta = np.array([2.0, -0.1, 0.0, -3.0, 0.5])
    step = 0.4
    expected = beta / (1.0 + (0.4 + 0.7) * step)
    np.testing.assert_allclose(p.prox(beta, step), expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Property tests: value + prox match hand-derived references across random
# (beta, step, lam) triples to 1e-12.
# ---------------------------------------------------------------------------


def test_l1_value_property_random() -> None:
    """Acceptance #3: random (beta, lam) — L1 value matches hand-derived
    to 1e-12."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        lam = float(rng.uniform(0.0, 5.0))
        beta = rng.standard_normal(rng.integers(1, 30))
        got = l1(lam).value(beta)
        expected = lam * np.sum(np.abs(beta))
        assert got == pytest.approx(expected, abs=1e-12, rel=0)


def test_l2_value_property_random() -> None:
    """Acceptance #3: random (beta, lam) — L2 value matches hand-derived
    (lam/2 * ||beta||_2^2) to 1e-12."""
    rng = np.random.default_rng(1)
    for _ in range(50):
        lam = float(rng.uniform(0.0, 5.0))
        beta = rng.standard_normal(rng.integers(1, 30))
        got = l2(lam).value(beta)
        expected = 0.5 * lam * float(beta @ beta)
        assert got == pytest.approx(expected, abs=1e-12, rel=0)


def test_l1_prox_property_random() -> None:
    """Acceptance #3: random (beta, step, lam) — L1 prox matches
    soft-thresholding to 1e-12."""
    rng = np.random.default_rng(2)
    for _ in range(50):
        lam = float(rng.uniform(0.0, 5.0))
        step = float(rng.uniform(1e-3, 2.0))
        beta = rng.standard_normal(rng.integers(1, 30))
        got = l1(lam).prox(beta, step)
        expected = _soft_threshold(beta, lam * step)
        np.testing.assert_allclose(got, expected, atol=1e-12)


def test_l2_prox_property_random() -> None:
    """Acceptance #3: random (beta, step, lam) — L2 prox matches shrinkage
    to 1e-12."""
    rng = np.random.default_rng(3)
    for _ in range(50):
        lam = float(rng.uniform(0.0, 5.0))
        step = float(rng.uniform(1e-3, 2.0))
        beta = rng.standard_normal(rng.integers(1, 30))
        got = l2(lam).prox(beta, step)
        expected = beta / (1.0 + lam * step)
        np.testing.assert_allclose(got, expected, atol=1e-12)


def test_elastic_net_prox_property_random() -> None:
    """Acceptance #3: random (beta, step, l1_lam, l2_lam) — elastic-net
    prox matches the closed form to 1e-12."""
    rng = np.random.default_rng(4)
    for _ in range(50):
        l1_lam = float(rng.uniform(0.0, 3.0))
        l2_lam = float(rng.uniform(0.0, 3.0))
        step = float(rng.uniform(1e-3, 2.0))
        beta = rng.standard_normal(rng.integers(1, 30))
        p = l1(l1_lam) + l2(l2_lam)
        got = p.prox(beta, step)
        expected = _soft_threshold(beta / (1.0 + l2_lam * step), l1_lam * step)
        np.testing.assert_allclose(got, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# ComparableFeatureScales declared via P1.B's framework
# ---------------------------------------------------------------------------


def test_l1_declares_comparable_feature_scales() -> None:
    """L1 declares ComparableFeatureScales (SOFT) in its assumptions."""
    from model_crafter.assumptions import ComparableFeatureScales

    p = l1(0.1)
    assert any(isinstance(a, ComparableFeatureScales) for a in p.assumptions)


def test_l2_declares_comparable_feature_scales() -> None:
    """L2 declares ComparableFeatureScales (SOFT) in its assumptions."""
    from model_crafter.assumptions import ComparableFeatureScales

    p = l2(0.1)
    assert any(isinstance(a, ComparableFeatureScales) for a in p.assumptions)


def test_comparable_feature_scales_is_soft_and_does_not_require_solution() -> None:
    """ComparableFeatureScales is SOFT severity, doesn't require a solution
    or CV (it's a property of the data + spec). DESIGN.md §4.3."""
    from model_crafter.assumptions import (
        ComparableFeatureScales,
        Severity,
    )

    cfs = ComparableFeatureScales()
    assert cfs.severity is Severity.SOFT
    assert cfs.requires_solution is False
    assert cfs.requires_cv is False


def test_comparable_feature_scales_passes_when_scales_balanced() -> None:
    """ComparableFeatureScales passes (no warning) when feature std ratio
    is below the threshold."""
    from model_crafter.assumptions import ComparableFeatureScales
    from model_crafter.loss import squared_error
    from model_crafter.spec import linear

    rng = np.random.default_rng(7)
    n = 200
    # Two features with comparable scales (std ratio ~ 2).
    df = pd.DataFrame({
        "x1": rng.standard_normal(n),
        "x2": 2.0 * rng.standard_normal(n),
        "y": rng.standard_normal(n),
    })
    spec = linear(target="y", features=["x1", "x2"],
                  loss=squared_error, penalty=l1(0.1))
    cfs = ComparableFeatureScales(std_ratio_max=100.0)
    result = cfs.check(spec, df)
    assert result.passed, result.message


def test_comparable_feature_scales_fires_when_scales_wildly_different() -> None:
    """Acceptance #2: ComparableFeatureScales (declared via P1.B's framework)
    fires on a synthetic dataset with feature std ratio > 100."""
    from model_crafter.assumptions import ComparableFeatureScales
    from model_crafter.loss import squared_error
    from model_crafter.spec import linear

    rng = np.random.default_rng(9)
    n = 200
    df = pd.DataFrame({
        "x1": rng.standard_normal(n),                  # std ~ 1
        "x2": 500.0 * rng.standard_normal(n),          # std ~ 500
        "y": rng.standard_normal(n),
    })
    spec = linear(target="y", features=["x1", "x2"],
                  loss=squared_error, penalty=l2(0.1))
    cfs = ComparableFeatureScales(std_ratio_max=100.0)
    result = cfs.check(spec, df)
    assert not result.passed
    assert result.statistic is not None and result.statistic > 100.0
    # Suggestion points at standardisation (DESIGN.md §4.3).
    assert "standardise" in (result.suggestion or "").lower()
    assert "feature std ratio" in result.message


def test_comparable_feature_scales_fires_via_run_assumptions() -> None:
    """Acceptance #2 (framework-level variant): wide-scale features + L2
    penalty cause ``run_assumptions`` to record a failing SOFT
    ComparableFeatureScales result *and* to emit a warning (SOFT failures
    warn under the default ``on_violation='raise'`` path — only HARD
    failures raise).

    We use ``run_assumptions`` directly rather than ``mc.solve`` because
    the L2 solver path is owned by Task P2.B and isn't integrated yet.
    Once P2.B lands the assumption-framework hookup in ``solve`` will
    deliver the same warning end-to-end.
    """
    import warnings

    from model_crafter.assumptions import (
        ComparableFeatureScales,
        Severity,
        run_assumptions,
    )
    from model_crafter.loss import squared_error
    from model_crafter.spec import linear

    rng = np.random.default_rng(11)
    n = 200
    df = pd.DataFrame({
        "x1": rng.standard_normal(n),
        "x2": 1000.0 * rng.standard_normal(n),
        "y": rng.standard_normal(n),
    })
    spec = linear(target="y", features=["x1", "x2"],
                  loss=squared_error, penalty=l2(0.1))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = run_assumptions(spec, df, on_violation="warn")

    # The penalty's ComparableFeatureScales check fires.
    cfs_results = [
        r for r in report.results
        if r.name == "ComparableFeatureScales"
    ]
    assert len(cfs_results) == 1, [r.name for r in report.results]
    cfs = cfs_results[0]
    assert cfs.severity is Severity.SOFT
    assert not cfs.passed
    # And the framework emits a warning at SOFT severity.
    messages = [str(w.message) for w in caught]
    assert any("ComparableFeatureScales" in m for m in messages), messages
    # Sanity: the assumption is actually declared by the L2 penalty (the
    # framework's collector reads `spec.penalty.assumptions`).
    assert any(
        isinstance(a, ComparableFeatureScales)
        for a in spec.penalty.assumptions
    )


def test_comparable_feature_scales_skips_intercept_only() -> None:
    """With a single feature column, the std-ratio is degenerate (one
    column). The check returns a pass (no scaling concern with one
    feature)."""
    from model_crafter.assumptions import ComparableFeatureScales
    from model_crafter.loss import squared_error
    from model_crafter.spec import linear

    rng = np.random.default_rng(13)
    df = pd.DataFrame({
        "x1": rng.standard_normal(100),
        "y": rng.standard_normal(100),
    })
    spec = linear(target="y", features=["x1"],
                  loss=squared_error, penalty=l1(0.1))
    result = ComparableFeatureScales().check(spec, df)
    assert result.passed


def test_comparable_feature_scales_describe_is_descriptive() -> None:
    """describe() returns a one-line, non-empty description."""
    from model_crafter.assumptions import ComparableFeatureScales

    desc = ComparableFeatureScales(std_ratio_max=100.0).describe()
    assert isinstance(desc, str)
    assert "scale" in desc.lower() or "std" in desc.lower()


# ---------------------------------------------------------------------------
# Frozen / immutability
# ---------------------------------------------------------------------------


def test_l1_is_frozen() -> None:
    """L1Penalty is immutable per DESIGN.md §9.1."""
    p = l1(0.3)
    with pytest.raises((AttributeError, Exception)):
        p.lam = 1.0  # type: ignore[misc]


def test_l2_is_frozen() -> None:
    """L2Penalty is immutable per DESIGN.md §9.1."""
    p = l2(0.3)
    with pytest.raises((AttributeError, Exception)):
        p.lam = 1.0  # type: ignore[misc]


def test_penalty_sum_is_frozen() -> None:
    """PenaltySum is immutable per DESIGN.md §9.1."""
    p = l1(0.1) + l2(0.2)
    with pytest.raises((AttributeError, Exception)):
        p.parts = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Penalty protocol membership
# ---------------------------------------------------------------------------


def test_l1_satisfies_penalty_protocol() -> None:
    """L1Penalty is structurally a Penalty (value + __add__ + assumptions)."""
    assert isinstance(l1(0.1), Penalty)


def test_l2_satisfies_penalty_protocol() -> None:
    """L2Penalty is structurally a Penalty."""
    assert isinstance(l2(0.1), Penalty)


def test_penalty_sum_satisfies_penalty_protocol() -> None:
    """PenaltySum is structurally a Penalty."""
    assert isinstance(l1(0.1) + l2(0.2), Penalty)
