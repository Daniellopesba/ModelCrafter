"""Tests for the lambda-path helpers (AGENTS.md Task P2.C).

The contract:

* ``log_grid(low, high, n)`` returns a descending log-spaced array.
* ``lambda_path(spec, data, n, ratio)`` returns a descending grid
  ``[lambda_max, ..., ratio * lambda_max]`` of length ``n`` where
  ``lambda_max = max_j |X^T y|_j / (n * alpha)`` on the standardised
  design.

P2.A's ``L1Penalty`` / ``L2Penalty`` / ``PenaltySum`` aren't merged at the
time this test is written, so we use minimal local stubs whose **type
names** match the production types (``_extract_alpha`` and
``_split_enet_parts`` both duck-type on ``type(p).__name__`` for that
reason — when P2.A merges the same stubs work without change because the
production classes have the same names).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from model_crafter.loss import squared_error
from model_crafter.spec import LinearSpec
from model_crafter.terms.base import _normalize_features
from model_crafter.validation.lambda_path import (
    compute_lambda_max,
    lambda_path,
    log_grid,
)

# ---------------------------------------------------------------------------
# Penalty stubs (mimic P2.A's contract by class name + .lam / .parts).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class L1Penalty:
    lam: float

    @property
    def assumptions(self) -> tuple:
        return ()

    def value(self, beta: np.ndarray) -> float:
        return float(self.lam * np.sum(np.abs(beta)))

    def __add__(self, other: object):
        return PenaltySum(parts=(self, other))


@dataclass(frozen=True)
class L2Penalty:
    lam: float

    @property
    def assumptions(self) -> tuple:
        return ()

    def value(self, beta: np.ndarray) -> float:
        return float(0.5 * self.lam * np.sum(beta * beta))

    def __add__(self, other: object):
        return PenaltySum(parts=(self, other))


@dataclass(frozen=True)
class PenaltySum:
    parts: tuple

    @property
    def assumptions(self) -> tuple:
        return ()

    def value(self, beta: np.ndarray) -> float:
        return float(sum(p.value(beta) for p in self.parts))

    def __add__(self, other: object):
        return PenaltySum(parts=(*self.parts, other))


def _spec(penalty, target="y", features=("x1", "x2")) -> LinearSpec:
    return LinearSpec(
        target=target,
        features=_normalize_features(list(features)),
        loss=squared_error,
        penalty=penalty,
        intercept=True,
    )


# ---------------------------------------------------------------------------
# log_grid
# ---------------------------------------------------------------------------


def test_log_grid_returns_descending_array() -> None:
    g = log_grid(low=1e-3, high=1.0, n=10)
    assert g.shape == (10,)
    assert g[0] == pytest.approx(1.0, abs=1e-12)
    assert g[-1] == pytest.approx(1e-3, abs=1e-12)
    # Strictly descending
    assert np.all(np.diff(g) < 0)


def test_log_grid_log_spaced() -> None:
    """log_grid spacing is uniform in log space."""
    g = log_grid(low=1e-4, high=1e2, n=7)
    log_diffs = np.diff(np.log(g))
    np.testing.assert_allclose(log_diffs, log_diffs[0], atol=1e-12)


def test_log_grid_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        log_grid(low=0.0, high=1.0, n=5)
    with pytest.raises(ValueError):
        log_grid(low=1.0, high=1.0, n=5)
    with pytest.raises(ValueError):
        log_grid(low=1e-3, high=1.0, n=1)


# ---------------------------------------------------------------------------
# compute_lambda_max
# ---------------------------------------------------------------------------


def test_compute_lambda_max_matches_manual_formula() -> None:
    """λ_max = (1/n) max_j |x_j^T y_centred| on standardised X (α=1)."""
    rng = np.random.default_rng(0)
    n, p = 100, 5
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)

    # Manual reference: standardise X, centre y, max |x^T y| / n.
    col_mean = X.mean(axis=0)
    Xc = X - col_mean
    col_sd = np.sqrt((Xc * Xc).mean(axis=0))  # population SD
    Xs = Xc / col_sd
    yc = y - y.mean()
    inner = Xs.T @ yc
    expected = float(np.max(np.abs(inner)) / n)

    got = compute_lambda_max(X, y, alpha=1.0, intercept=True, standardise=True)
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_compute_lambda_max_alpha_scales_inverse() -> None:
    """For α=0.5, λ_max is twice the α=1 value (KKT for elastic net)."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 3))
    y = rng.normal(size=50)
    lam_1 = compute_lambda_max(X, y, alpha=1.0)
    lam_h = compute_lambda_max(X, y, alpha=0.5)
    assert lam_h == pytest.approx(2.0 * lam_1, rel=1e-12)


def test_compute_lambda_max_rejects_alpha_zero() -> None:
    with pytest.raises(ValueError, match="alpha"):
        compute_lambda_max(np.eye(3), np.ones(3), alpha=0.0)


# ---------------------------------------------------------------------------
# lambda_path
# ---------------------------------------------------------------------------


def _make_data(n: int = 100, p: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = {f"x{i}": rng.normal(size=n) for i in range(1, p + 1)}
    cols["y"] = rng.normal(size=n)
    return pd.DataFrame(cols)


def test_lambda_path_descending_with_correct_endpoints() -> None:
    """lambda_path returns a descending grid spanning [ratio*λ_max, λ_max]."""
    df = _make_data(n=200, p=3, seed=42)
    spec = _spec(L1Penalty(lam=1.0), features=("x1", "x2", "x3"))
    grid = lambda_path(spec, df, n=50, ratio=1e-3)

    assert grid.shape == (50,)
    assert grid[0] > grid[-1]  # descending
    # First element equals lambda_max
    X = df[["x1", "x2", "x3"]].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    lam_max = compute_lambda_max(X, y, alpha=1.0)
    assert grid[0] == pytest.approx(lam_max, rel=1e-12)
    # Last element equals ratio * lambda_max
    assert grid[-1] == pytest.approx(1e-3 * lam_max, rel=1e-12)


def test_lambda_path_elastic_net_uses_alpha() -> None:
    """For an L1+L2 sum the path's λ_max divides by alpha (= lam_l1 / total)."""
    df = _make_data(n=80, p=2, seed=3)
    spec_l1 = _spec(L1Penalty(lam=1.0), features=("x1", "x2"))
    grid_l1 = lambda_path(spec_l1, df, n=10, ratio=1e-2)

    # α = lam_l1 / (lam_l1 + lam_l2) = 0.5 → λ_max doubles.
    spec_enet = _spec(L1Penalty(lam=1.0) + L2Penalty(lam=1.0), features=("x1", "x2"))
    grid_enet = lambda_path(spec_enet, df, n=10, ratio=1e-2)
    assert grid_enet[0] == pytest.approx(2.0 * grid_l1[0], rel=1e-12)


def test_lambda_path_pure_ridge_raises() -> None:
    """A pure-L2 spec doesn't define a lasso path; lambda_path must say so."""
    df = _make_data(seed=7)
    spec = _spec(L2Penalty(lam=1.0), features=("x1", "x2", "x3"))
    with pytest.raises(ValueError, match="pure ridge"):
        lambda_path(spec, df, n=20)
