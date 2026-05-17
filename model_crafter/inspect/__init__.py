"""Solution inspection (DESIGN.md §5).

Three independent topics live underneath:

* :mod:`.coefficients` — the named coefficient table with SEs, z/t, p.
* :mod:`.binning_table` — per-bin WoE summary for WoE / binned terms.
* :mod:`.diagnostics` — residuals, leverage, Cook's distance, hat matrix,
  influence. Closed-form-only; for lasso / elastic-net / logistic use
  :func:`mc.bootstrap` (ESL §7.11).
"""

from __future__ import annotations

from model_crafter.inspect.binning_table import BinningTable, binning_table
from model_crafter.inspect.coefficients import coefficients
from model_crafter.inspect.diagnostics import (
    Diagnostics,
    Influence,
    diagnostics,
    hat_matrix,
    influence,
)

__all__ = [
    "BinningTable",
    "Diagnostics",
    "Influence",
    "binning_table",
    "coefficients",
    "diagnostics",
    "hat_matrix",
    "influence",
]
