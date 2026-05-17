"""Tests for the Loss protocol and squared_error.

Covers AGENTS.md Task P1.A's Loss interface contract: ``value``, ``gradient``,
``hessian`` all accept ``weights=``; squared_error declares its assumptions.
"""

from __future__ import annotations

import numpy as np
import pytest

from model_crafter.loss import Loss, squared_error


def test_squared_error_is_a_loss_instance() -> None:
    """squared_error is an object that satisfies the Loss protocol."""
    assert isinstance(squared_error, Loss)


def test_squared_error_declares_assumptions() -> None:
    """Per DESIGN.md §9.7 every loss declares an ``assumptions`` tuple."""
    assert hasattr(squared_error, "assumptions")
    assert isinstance(squared_error.assumptions, tuple)
    # Must include the FullRankDesign HARD prerequisite (DESIGN.md §4.3).
    names = [type(a).__name__ for a in squared_error.assumptions]
    assert "FullRankDesign" in names


def test_squared_error_value_matches_formula_unweighted() -> None:
    r"""L(y, eta) = 0.5 * sum (y - eta)^2 / n by convention."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    eta = np.array([1.0, 2.0, 2.0, 3.0])
    expected = 0.5 * np.mean((y - eta) ** 2)
    assert squared_error.value(y, eta, weights=None) == pytest.approx(expected, abs=1e-15)


def test_squared_error_value_matches_formula_weighted() -> None:
    r"""Weighted: L(y, eta) = 0.5 * sum(w * (y-eta)^2) / sum(w)."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    eta = np.array([1.0, 2.0, 2.0, 3.0])
    w = np.array([1.0, 2.0, 3.0, 4.0])
    expected = 0.5 * np.sum(w * (y - eta) ** 2) / np.sum(w)
    assert squared_error.value(y, eta, weights=w) == pytest.approx(expected, abs=1e-15)


def test_squared_error_gradient_shape() -> None:
    """gradient(y, eta) has the same shape as eta."""
    y = np.zeros(5)
    eta = np.ones(5)
    g = squared_error.gradient(y, eta, weights=None)
    assert g.shape == eta.shape


def test_squared_error_hessian_shape() -> None:
    """hessian is a diagonal weight vector (numpy 1-D array) over observations."""
    y = np.zeros(5)
    eta = np.ones(5)
    h = squared_error.hessian(y, eta, weights=None)
    assert isinstance(h, np.ndarray)
    assert h.shape == (5,)
