"""Term primitives.

A ``Term`` is a value that knows how to expand into one or more design-matrix
columns. The base layer (:mod:`model_crafter.terms.base`) defines the protocol,
the trivial column-reference term :class:`RawTerm`, and the additive container
:class:`TermSum`. Phase 4 adds basis expansions (:mod:`~model_crafter.terms.basis`),
WoE / bin-indicator terms (:mod:`~model_crafter.terms.woe`), and interactions
(:mod:`~model_crafter.terms.interact`).

The public symbols are re-exported here so that ``mc.linear`` can accept
``features`` in any of these forms:

>>> import model_crafter as mc  # doctest: +SKIP
>>> features = "income"                                # single string
>>> features = ["income", "age", "tenure"]             # list form
>>> features = mc.raw("income") + "age"                # sum form
>>> features = mc.ns("age", df=5) + mc.woe("region", bins=mc.categorical())
"""

from model_crafter.terms.base import (
    ExpandedTerm,
    RawTerm,
    Term,
    TermSum,
    _normalize_features,
    _promote,
)
from model_crafter.terms.basis import bs, hinge, ns, poly, smooth, step
from model_crafter.terms.interact import cross, interact
from model_crafter.terms.woe import (
    binned,
    categorical,
    manual,
    monotonic,
    tree_bins,
    woe,
)

__all__ = [
    "ExpandedTerm",
    "RawTerm",
    "Term",
    "TermSum",
    "_normalize_features",
    "_promote",
    "binned",
    "bs",
    "categorical",
    "cross",
    "hinge",
    "interact",
    "manual",
    "monotonic",
    "ns",
    "poly",
    "smooth",
    "step",
    "tree_bins",
    "woe",
]
