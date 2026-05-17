"""LinearSpec and the public ``linear`` constructor.

A :class:`LinearSpec` is the declarative description of a linear-predictor
model. It is a frozen, hashable, picklable value: per DESIGN.md §2.1 there is
no ``fit`` method and no learned state on the spec. State lives on the
:class:`~model_crafter.solution.Solution` returned by
:func:`~model_crafter.solve.solve`.

Construction validates eagerly (DESIGN.md §9.8): an empty feature list, a
malformed target, or a loss/penalty that does not satisfy its protocol is
caught at spec construction time so failures point at the spec, not the
solver.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from model_crafter.loss import Loss
from model_crafter.penalty import NoPenalty, Penalty
from model_crafter.terms.base import Term, _normalize_features

__all__ = ["LinearSpec", "linear"]


@dataclass(frozen=True, slots=True)
class LinearSpec:
    """Declarative spec for a linear predictor :math:`\\eta = X\\beta`.

    See DESIGN.md §6.2 for the type signature. ``features`` is always a
    flattened tuple of :class:`Term` (strings have been promoted by
    :func:`linear`). ``loss`` and ``penalty`` are values, not strings —
    string identifiers for losses are explicitly rejected by DESIGN.md §9.3.
    """

    target: str
    features: tuple[Term, ...]
    loss: Loss
    penalty: Penalty = field(default_factory=NoPenalty)
    intercept: bool = True

    def __post_init__(self) -> None:
        # Target validation
        if not isinstance(self.target, str):
            raise TypeError(
                f"target must be a column name (str); got {type(self.target).__name__}"
            )
        if not self.target:
            raise ValueError("target must be a non-empty column name")
        # Features validation: must already be a tuple of Terms.
        if not isinstance(self.features, tuple) or not all(
            isinstance(t, Term) for t in self.features
        ):
            raise TypeError(
                "features must be a tuple of Term; use mc.linear(...) to construct LinearSpec"
            )
        if not self.features:
            raise ValueError("features must be non-empty")
        # Loss / penalty protocol checks: structural, since they are Protocols.
        if not isinstance(self.loss, Loss):
            raise TypeError(
                f"loss must satisfy the Loss protocol; got {type(self.loss).__name__}"
            )
        if not isinstance(self.penalty, Penalty):
            raise TypeError(
                f"penalty must satisfy the Penalty protocol; got {type(self.penalty).__name__}"
            )
        if not isinstance(self.intercept, bool):
            raise TypeError("intercept must be a bool")


def linear(
    *,
    target: str,
    features: str | Term | Iterable[str | Term],
    loss: Loss,
    penalty: Penalty | None = None,
    intercept: bool = True,
) -> LinearSpec:
    """Construct a :class:`LinearSpec` for a linear-predictor model.

    Parameters
    ----------
    target:
        The name of the target column in the data frame passed to ``solve``.
    features:
        Either a single column name (str), a single :class:`Term` (including
        :class:`~model_crafter.terms.base.TermSum`), or an iterable of
        strings/Terms. All forms normalize to a flat ``tuple[Term, ...]``.
    loss:
        A :class:`~model_crafter.loss.Loss` instance — for Phase 1, only
        ``mc.squared_error``.
    penalty:
        Defaults to :class:`~model_crafter.penalty.NoPenalty`. Other
        penalties arrive in Phase 2.
    intercept:
        Whether to include an unpenalized intercept column. Default ``True``.

    Returns
    -------
    LinearSpec
        A frozen, hashable value.
    """
    if penalty is None:
        penalty = NoPenalty()
    return LinearSpec(
        target=target,
        features=_normalize_features(features),
        loss=loss,
        penalty=penalty,
        intercept=intercept,
    )


# These are referenced by `Any` import keep-alive in some toolchains; silence unused.
_ = Any
