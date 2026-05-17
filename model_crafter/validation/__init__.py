"""Validation utilities.

Phase 2 ships the lambda-path helpers (:func:`lambda_path`,
:func:`log_grid`). Phase 3 adds the bootstrap (:func:`bootstrap`,
AGENTS.md Task P3.C). Splitters / cross_validate / tune (P3.B) are wired
in by the integration agent.
"""

from __future__ import annotations

from model_crafter.validation.bootstrap import bootstrap
from model_crafter.validation.lambda_path import lambda_path, log_grid

__all__ = ["bootstrap", "lambda_path", "log_grid"]
