"""model_crafter — a Python package for credit risk modeling that feels like pen and paper.

See DESIGN.md for the architectural contract and AGENTS.md for the build plan.

Phase 1 public surface — re-exports per AGENTS.md Task P1.INTEG. Limited to
what the DESIGN.md §10 example uses for Phase 1 scope: linear, solve,
predict, squared_error, NoPenalty, check_assumptions.
"""

from model_crafter.assumptions import check_assumptions
from model_crafter.loss import squared_error
from model_crafter.penalty import NoPenalty
from model_crafter.solve import predict, solve
from model_crafter.spec import linear

__version__ = "0.0.0"

__all__ = [
    "NoPenalty",
    "__version__",
    "check_assumptions",
    "linear",
    "predict",
    "solve",
    "squared_error",
]
