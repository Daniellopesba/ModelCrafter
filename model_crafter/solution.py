"""Solution and BootstrappedSolution dataclasses.

A :class:`Solution` is the immutable artefact of fitting. Per DESIGN.md §2.1
there is no fitted/unfitted duality: a ``Solution`` is what falls out when
:func:`~model_crafter.solve.solve` is applied to a
:class:`~model_crafter.spec.LinearSpec` and data.

The ``assumptions`` field carries an :class:`AssumptionReport` (defined by
P1.B) summarising the prerequisite and stability checks that ran at solve
time. P1.A treats this type as opaque — it imports the type from
``model_crafter.assumptions`` and never reaches into the assumption
framework's internals.

The :class:`BootstrappedSolution` value type (AGENTS.md Task P3.C, DESIGN.md
§3.2) bundles the empirical coefficient distribution obtained from
:func:`~model_crafter.validation.bootstrap.bootstrap`. It exposes percentile
and (stubbed) BCa coefficient CIs, a prediction CI helper, and a
``selection_frequency`` Series that is the standard lasso-stability
diagnostic (ESL §3.4.3).
"""

from __future__ import annotations

import html
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from model_crafter.spec import LinearSpec, SegmentedSpec

if TYPE_CHECKING:
    from model_crafter.assumptions import AssumptionReport

__all__ = ["BootstrappedSolution", "SegmentedSolution", "Solution"]


@dataclass(frozen=True, slots=True)
class Solution:
    """Immutable result of solving a :class:`LinearSpec` against data.

    See AGENTS.md Task P1.A for the field contract.
    """

    spec: LinearSpec
    coefficients: pd.Series
    coefficient_se: pd.Series | None
    fit_state: Mapping[str, Any]
    design_columns: tuple[str, ...]
    loss_value: float
    penalty_value: float
    n_obs: int
    converged: bool
    solver_info: Mapping[str, Any]
    assumptions: AssumptionReport

    def __post_init__(self) -> None:
        # Defensive validation. Failure here usually means a solver
        # constructed a Solution with mismatched coefficient names; the
        # error message names the disagreement.
        if not isinstance(self.spec, LinearSpec):
            raise TypeError(
                f"spec must be a LinearSpec; got {type(self.spec).__name__}"
            )
        if not isinstance(self.coefficients, pd.Series):
            raise TypeError("coefficients must be a pandas Series")
        if self.coefficient_se is not None and not isinstance(
            self.coefficient_se, pd.Series
        ):
            raise TypeError("coefficient_se must be a pandas Series or None")
        if list(self.coefficients.index) != list(self.design_columns):
            raise ValueError(
                "coefficients.index must equal design_columns; "
                f"got coefficients={list(self.coefficients.index)} vs "
                f"design_columns={list(self.design_columns)}"
            )
        if self.coefficient_se is not None and list(
            self.coefficient_se.index
        ) != list(self.design_columns):
            raise ValueError(
                "coefficient_se.index must equal design_columns; "
                f"got coefficient_se={list(self.coefficient_se.index)} vs "
                f"design_columns={list(self.design_columns)}"
            )
        if self.n_obs < 0:
            raise ValueError(f"n_obs must be non-negative; got {self.n_obs}")

    def _repr_html_(self) -> str:
        target = html.escape(str(self.spec.target))
        loss_name = html.escape(type(self.spec.loss).__name__)
        coef_df = pd.DataFrame({"estimate": self.coefficients})
        if self.coefficient_se is not None:
            coef_df["std_error"] = self.coefficient_se
        coef_table = coef_df.to_html(border=0, classes="mc-coef")
        return (
            f"<div class='mc-solution'>"
            f"<strong>Solution</strong> "
            f"<span>target={target}, loss={loss_name}, "
            f"n_obs={self.n_obs:,}, "
            f"converged={self.converged}</span>"
            f"{coef_table}"
            f"</div>"
        )


@dataclass(frozen=True, slots=True)
class BootstrappedSolution:
    r"""Empirical bootstrap distribution of a fitted :class:`Solution`.

    See AGENTS.md Task P3.C for the field contract and DESIGN.md §3.2 for
    the motivation (ESL §7.11 / §8.2). A ``BootstrappedSolution`` is the
    value returned by :func:`model_crafter.validation.bootstrap.bootstrap`:
    a single fixed base solution together with ``n_boot`` re-fit
    coefficient vectors and per-resample ``fit_state`` mappings for
    term-level diagnostics.

    Fields
    ------
    base
        The point-estimate :class:`Solution` whose distribution is being
        described. ``coefficients_dist`` columns equal ``base.design_columns``.
    coefficients_dist
        ``n_boot × n_coef`` :class:`pandas.DataFrame`; row ``b`` is the
        coefficient vector from the ``b``-th bootstrap refit.
    fit_state_dist
        Tuple of per-resample ``fit_state`` mappings, one per refit. Each
        is a :class:`Mapping` from term name to learned state (e.g. WoE
        bin edges). Used by term-level diagnostics that look at the
        stability of *learned* state, not just coefficients.
    selection_frequency
        :class:`pandas.Series` indexed by ``design_columns``: the fraction
        of resamples where each coefficient was non-zero. For lasso fits
        (ESL §3.4.3), the standard diagnostic for selection stability under
        collinearity. For unpenalised / ridge fits exact zeros are
        vanishingly rare; in that case the series is all 1.0 by
        construction.
    n_boot
        Number of *successful* bootstrap refits. Failing refits (e.g. a
        rank-deficient resample that raises :class:`AssumptionError`) are
        skipped silently up to a small fraction; if too many fail, the
        bootstrap raises a :class:`RuntimeError` (see ``bootstrap``).
    method
        ``"pairs"`` (default; ESL §8.2.1), ``"residual"`` (ESL §8.2.2; only
        meaningful for fixed-X regression with iid errors), or ``"block"``
        when a splitter is provided for time-dependent data.
    """

    base: Solution
    coefficients_dist: pd.DataFrame
    fit_state_dist: tuple[Mapping[str, Any], ...]
    selection_frequency: pd.Series
    n_boot: int
    method: str

    def __post_init__(self) -> None:
        if not isinstance(self.base, Solution):
            raise TypeError(
                f"base must be a Solution; got {type(self.base).__name__}"
            )
        if not isinstance(self.coefficients_dist, pd.DataFrame):
            raise TypeError("coefficients_dist must be a pandas DataFrame")
        if list(self.coefficients_dist.columns) != list(self.base.design_columns):
            raise ValueError(
                "coefficients_dist.columns must equal base.design_columns; "
                f"got coefficients_dist={list(self.coefficients_dist.columns)} "
                f"vs base.design_columns={list(self.base.design_columns)}"
            )
        if not isinstance(self.selection_frequency, pd.Series):
            raise TypeError("selection_frequency must be a pandas Series")
        if list(self.selection_frequency.index) != list(self.base.design_columns):
            raise ValueError(
                "selection_frequency.index must equal base.design_columns; "
                f"got {list(self.selection_frequency.index)} vs "
                f"{list(self.base.design_columns)}"
            )
        if self.n_boot != self.coefficients_dist.shape[0]:
            raise ValueError(
                f"n_boot={self.n_boot} disagrees with coefficients_dist.shape[0]"
                f"={self.coefficients_dist.shape[0]}"
            )
        if self.n_boot != len(self.fit_state_dist):
            raise ValueError(
                f"n_boot={self.n_boot} disagrees with len(fit_state_dist)"
                f"={len(self.fit_state_dist)}"
            )
        if not isinstance(self.method, str):
            raise TypeError("method must be a string")

    # ------------------------------------------------------------------
    # CIs
    # ------------------------------------------------------------------

    def coefficient_ci(
        self, level: float = 0.95, method: str = "percentile"
    ) -> pd.DataFrame:
        r"""Coefficient confidence intervals from the bootstrap distribution.

        Percentile CI (Efron & Tibshirani 1993, §13; ESL §7.11):

        .. math::

            \mathrm{CI}_{1-\alpha}(\beta_j) = \bigl(
                F_j^{-1}(\alpha/2),\ F_j^{-1}(1 - \alpha/2)
            \bigr)

        where :math:`F_j` is the empirical CDF of the ``b``-th coefficient
        across bootstrap resamples and :math:`\alpha = 1 - \text{level}`.

        Parameters
        ----------
        level
            Coverage probability, in ``(0, 1)``. Default ``0.95``.
        method
            ``"percentile"`` (default) or ``"bca"`` (bias-corrected
            accelerated). BCa is documented in DESIGN.md §3.2 but is
            **not implemented in v0** — calling with ``method="bca"``
            raises :class:`NotImplementedError`. See ``notes/P3.C.md`` for
            the deferral rationale.

        Returns
        -------
        :class:`pandas.DataFrame`
            Indexed by ``base.design_columns``, columns
            ``["lower", "upper"]``.
        """
        if not (0.0 < level < 1.0):
            raise ValueError(
                f"level must be in (0, 1); got level={level}"
            )
        if method == "bca":
            raise NotImplementedError(
                "BCa (bias-corrected accelerated) CIs are not implemented in v0. "
                "DESIGN.md §3.2 lists BCa as the standard refinement of the "
                "percentile interval (Efron & Tibshirani 1993, §14); for v0 "
                "use method='percentile'. See notes/P3.C.md."
            )
        if method != "percentile":
            raise ValueError(
                f"unknown CI method {method!r}; "
                "supported: 'percentile' (default), 'bca' (NotImplementedError)"
            )
        alpha = 1.0 - level
        lower_q = alpha / 2.0
        upper_q = 1.0 - alpha / 2.0
        # np.quantile on a DataFrame returns row-wise by default; transpose
        # the dataframe so each column (each coefficient) becomes its own
        # quantile target.
        arr = self.coefficients_dist.to_numpy(dtype=float)
        lower = np.quantile(arr, lower_q, axis=0)
        upper = np.quantile(arr, upper_q, axis=0)
        return pd.DataFrame(
            {"lower": lower, "upper": upper},
            index=pd.Index(self.coefficients_dist.columns, name=None),
        )

    def prediction_ci(
        self, new_data: pd.DataFrame, level: float = 0.95
    ) -> pd.DataFrame:
        r"""Percentile CI for :math:`\hat y(x)` over the bootstrap refits.

        For each row of ``new_data``, builds the empirical distribution of
        :math:`\hat y_i = X_i^\top \beta^{(b)}` across the ``n_boot`` refits
        (using each refit's ``fit_state`` so any learned design state — e.g.
        WoE bin edges in later phases — is honoured), then returns the
        percentile interval per row.

        Parameters
        ----------
        new_data
            Frame whose rows define the prediction points. The frame must
            contain every column the spec's terms require.
        level
            Coverage probability, in ``(0, 1)``. Default ``0.95``.

        Returns
        -------
        :class:`pandas.DataFrame`
            Indexed by ``new_data.index``, columns ``["lower", "upper"]``.
        """
        if not isinstance(new_data, pd.DataFrame):
            raise TypeError(
                f"new_data must be a pandas DataFrame; got {type(new_data).__name__}"
            )
        if not (0.0 < level < 1.0):
            raise ValueError(f"level must be in (0, 1); got level={level}")

        # Defer the design-matrix import to call time so this module doesn't
        # pull in solve at import time (solve depends on solution.py at
        # import).
        from model_crafter._internal.design import build_design

        n_pred = len(new_data)
        if n_pred == 0:
            return pd.DataFrame(
                {"lower": [], "upper": []}, index=new_data.index
            )

        # For each successful resample, build the design matrix with the
        # resample's learned fit_state and compute X @ beta_b.
        yhat_matrix = np.empty((self.n_boot, n_pred), dtype=float)
        col_order = list(self.base.design_columns)
        coef_arr = self.coefficients_dist.to_numpy(dtype=float)
        spec = self.base.spec
        for b in range(self.n_boot):
            fs = self.fit_state_dist[b]
            design_b = build_design(spec, new_data, fit_state=fs)
            if list(design_b.columns) != col_order:
                # Defensive: if a resample produced a different column shape
                # we cannot mix predictions; raise.
                raise ValueError(
                    "predict-time design columns disagree with the base "
                    "solution; this should not happen for fixed feature "
                    f"sets — got {list(design_b.columns)} vs {col_order}"
                )
            yhat_matrix[b, :] = design_b.values @ coef_arr[b, :]

        alpha = 1.0 - level
        lower = np.quantile(yhat_matrix, alpha / 2.0, axis=0)
        upper = np.quantile(yhat_matrix, 1.0 - alpha / 2.0, axis=0)
        return pd.DataFrame(
            {"lower": lower, "upper": upper}, index=new_data.index
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"BootstrappedSolution(method={self.method!r}, "
            f"n_boot={self.n_boot}, "
            f"n_coef={len(self.base.design_columns)})"
        )

    def _repr_html_(self) -> str:
        method = html.escape(self.method)
        try:
            ci_df = self.coefficient_ci()
        except Exception:  # pragma: no cover - defensive
            ci_df = pd.DataFrame()
        ci_table = ci_df.to_html(border=0, classes="mc-bootstrap-ci") if not ci_df.empty else "<p>(empty CI)</p>"
        return (
            f"<div class='mc-bootstrapped-solution'>"
            f"<strong>BootstrappedSolution</strong> "
            f"<span>method={method}, n_boot={self.n_boot}, "
            f"n_coef={len(self.base.design_columns)}</span>"
            f"<h4>95% percentile CIs</h4>"
            f"{ci_table}"
            f"</div>"
        )


# ---------------------------------------------------------------------------
# SegmentedSolution (Phase 6, DESIGN.md §3.4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SegmentedSolution:
    """Per-segment fitted :class:`Solution` bundle (DESIGN.md §3.4).

    A :class:`SegmentedSolution` is the immutable artefact of solving a
    :class:`~model_crafter.spec.SegmentedSpec`: one independent
    :class:`Solution` per unique value of the segmentation column.

    Attributes
    ----------
    spec
        The :class:`~model_crafter.spec.SegmentedSpec` that produced this
        solution.
    segments
        ``dict`` mapping (stringified) segment key to its fitted
        :class:`Solution`. Each per-segment ``Solution`` carries its own
        ``assumptions`` :class:`AssumptionReport`.
    n_obs
        Total number of observations across all per-segment fits — the
        sum of ``segments[k].n_obs`` for each ``k``.

    Notes
    -----
    Mapping-style accessors are provided for ergonomics: ``sol["A"]`` is
    sugar for ``sol.segments["A"]``; ``for key in sol`` iterates over keys
    in declaration order; ``sol.keys()`` mirrors ``dict.keys``.
    """

    spec: SegmentedSpec
    segments: dict[str, Solution]
    n_obs: int

    def __post_init__(self) -> None:
        if not isinstance(self.spec, SegmentedSpec):
            raise TypeError(
                f"spec must be a SegmentedSpec; got {type(self.spec).__name__}"
            )
        if not isinstance(self.segments, dict):
            raise TypeError("segments must be a dict[str, Solution]")
        for k, v in self.segments.items():
            if not isinstance(k, str):
                raise TypeError(
                    f"segment key {k!r} must be a string; got "
                    f"{type(k).__name__}"
                )
            if not isinstance(v, Solution):
                raise TypeError(
                    f"segments[{k!r}] must be a Solution; got "
                    f"{type(v).__name__}"
                )
        if self.n_obs < 0:
            raise ValueError(f"n_obs must be non-negative; got {self.n_obs}")

    # ------------------------------------------------------------------
    # Mapping ergonomics
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> Solution:
        return self.segments[key]

    def __iter__(self):
        return iter(self.segments)

    def __len__(self) -> int:
        return len(self.segments)

    def keys(self):
        """Return the segment keys (same order as ``segments``)."""
        return self.segments.keys()

    def values(self):
        """Return the per-segment :class:`Solution`\\ s."""
        return self.segments.values()

    def items(self):
        """Return ``(key, Solution)`` pairs."""
        return self.segments.items()

    def __repr__(self) -> str:
        keys = list(self.segments.keys())
        n_seg = len(keys)
        return (
            f"SegmentedSolution(by={self.spec.by!r}, "
            f"n_segments={n_seg}, n_obs={self.n_obs:,}, "
            f"segments={keys!r})"
        )

    def _repr_html_(self) -> str:
        by_safe = html.escape(self.spec.by)
        rows = []
        for k, sub in self.segments.items():
            n_coef = len(sub.design_columns)
            rows.append(
                f"<tr><td>{html.escape(k)}</td>"
                f"<td>{sub.n_obs:,}</td>"
                f"<td>{n_coef}</td>"
                f"<td>{sub.converged}</td></tr>"
            )
        body = (
            "<table class='mc-segmented'>"
            "<thead><tr><th>Segment</th><th>n_obs</th>"
            "<th>n_coef</th><th>converged</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            "</table>"
        )
        return (
            f"<div class='mc-segmented-solution'>"
            f"<strong>SegmentedSolution</strong> "
            f"<span>by={by_safe}, n_segments={len(self.segments)}, "
            f"n_obs={self.n_obs:,}</span>"
            f"{body}"
            f"</div>"
        )
