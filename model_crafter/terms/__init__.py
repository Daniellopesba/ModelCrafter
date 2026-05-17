"""Term primitives.

A ``Term`` is a value that knows how to expand into one or more design-matrix
columns. The base layer (:mod:`model_crafter.terms.base`) defines the protocol,
the trivial column-reference term :class:`RawTerm`, and the additive container
:class:`TermSum`. Later phases add basis expansions, WoE, and interactions.

The public symbols are re-exported here so that ``mc.linear`` can accept
``features`` in any of these forms:

>>> import model_crafter as mc  # doctest: +SKIP
>>> features = "income"                                # single string
>>> features = ["income", "age", "tenure"]             # list form
>>> features = mc.raw("income") + "age"                # sum form
"""

from model_crafter.terms.base import (
    ExpandedTerm,
    RawTerm,
    Term,
    TermSum,
    _normalize_features,
    _promote,
)

__all__ = [
    "ExpandedTerm",
    "RawTerm",
    "Term",
    "TermSum",
    "_normalize_features",
    "_promote",
]
