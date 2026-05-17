"""Validation utilities.

Phase 2 ships the lambda-path helpers only (:func:`lambda_path`,
:func:`log_grid`) — they're used by the coordinate-descent solver in
:mod:`model_crafter.solve.coordinate` and (in later phases) by
:func:`mc.tune` / :func:`mc.nested_cv`.

The rest of the validation/ subpackage (splitters, ``cross_validate``,
``tune``, ``bootstrap``) lands in Phase 3 (AGENTS.md Task P3.B / P3.C).
"""

from __future__ import annotations

from model_crafter.validation.lambda_path import lambda_path, log_grid

__all__ = ["lambda_path", "log_grid"]
