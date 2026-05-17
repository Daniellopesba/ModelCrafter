r"""Solution-inspection helpers (DESIGN.md §5).

Phase 4 adds :func:`binning_table`, which extracts the per-term WoE
binning state from a fitted :class:`~model_crafter.solution.Solution` and
returns a :class:`BinningTable` value carrying one DataFrame per
WoE/binned column (counts, event rate, WoE, IV) plus an aggregate
Information Value Series.

Phase 6 adds :func:`coefficients`, :func:`diagnostics`, :func:`hat_matrix`,
:func:`influence` per DESIGN.md §5. The diagnostic primitives are
**closed-form-only**: leverage, Cook's distance, the hat matrix and
DFBETAS are well-defined for OLS and closed-form ridge, but not for lasso,
elastic net, or logistic regression. The non-closed-form solvers raise
:class:`NotImplementedError` with a pointer to :func:`mc.bootstrap` (ESL
§7.11) as the alternative uncertainty path.
"""

from __future__ import annotations

import html
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

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


@dataclass(frozen=True, slots=True)
class BinningTable:
    r"""The binning summary for a fitted solution.

    Attributes
    ----------
    tables
        ``dict`` mapping each WoE/binned column name to a
        :class:`pandas.DataFrame` with the per-bin breakdown. Columns:
        ``["bin", "n", "n_events", "event_rate", "woe", "iv"]`` (where
        ``iv`` is the per-bin contribution to total IV, useful for
        spotting which bins dominate the feature's signal).
    iv
        :class:`pandas.Series` of total IV per column (the row-sum of each
        table's ``iv`` column).
    """

    tables: Mapping[str, pd.DataFrame]
    iv: pd.Series

    def __post_init__(self) -> None:
        for col, df in self.tables.items():
            expected = ["bin", "n", "n_events", "event_rate", "woe", "iv"]
            if list(df.columns) != expected:
                raise ValueError(
                    f"binning_table['{col}'] must have columns {expected}; "
                    f"got {list(df.columns)}"
                )

    def __repr__(self) -> str:
        if not self.tables:
            return "BinningTable(empty)"
        lines = ["BinningTable"]
        for col, df in self.tables.items():
            iv = float(self.iv[col]) if col in self.iv.index else 0.0
            lines.append(f"  {col}  (IV = {iv:.4f})")
            lines.append("  " + str(df).replace("\n", "\n  "))
            lines.append("")
        return "\n".join(lines).rstrip()

    def _repr_html_(self) -> str:
        if not self.tables:
            return "<div class='mc-binning-table'>BinningTable (empty)</div>"
        sections = []
        for col, df in self.tables.items():
            iv_val = float(self.iv[col]) if col in self.iv.index else 0.0
            sections.append(
                f"<h4>{html.escape(str(col))} "
                f"<span>(IV = {iv_val:.4f})</span></h4>"
                + df.to_html(border=0, classes="mc-binning-table")
            )
        return (
            "<div class='mc-binning-table'>"
            "<strong>BinningTable</strong>"
            + "".join(sections)
            + "</div>"
        )


def binning_table(sol: Any) -> BinningTable:
    r"""Return the :class:`BinningTable` for a fitted solution.

    Walks ``sol.spec.features``, picks out WoE / bin-indicator terms, and
    materialises a per-bin DataFrame from each term's stored
    :class:`~model_crafter.terms.woe.BinningResult`. The per-bin IV
    contribution :math:`(p^{(1)}_b - p^{(0)}_b) \cdot \mathrm{WoE}_b` is
    computed alongside the totals.

    Parameters
    ----------
    sol
        A fitted :class:`~model_crafter.solution.Solution`. Specifically,
        ``sol.spec.features`` must be iterable and either expose
        :class:`~model_crafter.terms.woe.WoETerm` or
        :class:`~model_crafter.terms.woe.BinnedTerm` values whose
        ``.fitted`` :class:`BinningResult` is populated (as it is after
        going through :func:`~model_crafter.terms.woe.fit_binnings`).

    Notes
    -----
    Industry IV rules of thumb (Siddiqi 2006, Anderson 2007): < 0.02
    useless, 0.02-0.1 weak, 0.1-0.3 medium, 0.3-0.5 strong, > 0.5
    suspiciously high (likely target leakage). The package reports the
    value but does not enforce these thresholds.
    """
    from model_crafter.terms.woe import BinnedTerm, WoETerm

    spec = getattr(sol, "spec", None)
    if spec is None:
        raise TypeError(
            f"binning_table expected a Solution-like object with a `.spec`; got "
            f"{type(sol).__name__}"
        )

    tables: dict[str, pd.DataFrame] = {}
    iv_per_col: dict[str, float] = {}

    for term in getattr(spec, "features", ()) or ():
        if not isinstance(term, (WoETerm, BinnedTerm)):
            continue
        result = getattr(term, "fitted", None)
        if result is None:
            # Solution may have stashed it under fit_state[term.name].
            fit_state = getattr(sol, "fit_state", None) or {}
            cand = fit_state.get(term.column)
            if isinstance(cand, Mapping):
                from model_crafter.terms.woe import BinningResult as _BR
                result = cand.get("result") if isinstance(cand.get("result"), _BR) else None
        if result is None:
            continue

        n_events = list(result.n_events)
        n_nonevents = list(result.n_nonevents)
        bin_labels = list(result.bin_labels)
        n_totals = [e + ne for e, ne in zip(n_events, n_nonevents, strict=True)]
        event_rates = [(e / nt) if nt > 0 else 0.0 for e, nt in zip(n_events, n_totals, strict=True)]

        # Per-bin IV contribution: (p_event_b - p_nonevent_b) * WoE_b, using
        # the same Laplace-smoothed proportions the term used.
        from model_crafter.terms.woe import _SMOOTHING

        e_smooth = [e + _SMOOTHING for e in n_events]
        ne_smooth = [ne + _SMOOTHING for ne in n_nonevents]
        e_tot = sum(e_smooth)
        ne_tot = sum(ne_smooth)
        per_bin_iv = [
            (es / e_tot - nes / ne_tot) * w
            for es, nes, w in zip(e_smooth, ne_smooth, result.woe, strict=True)
        ]

        df = pd.DataFrame(
            {
                "bin": bin_labels,
                "n": n_totals,
                "n_events": n_events,
                "event_rate": event_rates,
                "woe": list(result.woe),
                "iv": per_bin_iv,
            }
        )
        tables[term.column] = df
        iv_per_col[term.column] = float(result.iv)

    iv_series = pd.Series(iv_per_col, name="iv", dtype=float)
    return BinningTable(tables=tables, iv=iv_series)


# ---------------------------------------------------------------------------
# Phase 6: coefficients / diagnostics / hat_matrix / influence (DESIGN.md §5)
# ---------------------------------------------------------------------------


def _solution_is_ols(sol: Any) -> bool:
    """True iff the spec is squared-error + NoPenalty (closed-form OLS)."""
    from model_crafter.loss import _SquaredErrorLoss
    from model_crafter.penalty import NoPenalty

    spec = getattr(sol, "spec", None)
    if spec is None:
        return False
    return isinstance(spec.loss, _SquaredErrorLoss) and isinstance(
        spec.penalty, NoPenalty
    )


def _solution_is_ridge(sol: Any) -> bool:
    """True iff the spec is squared-error + L2 only (closed-form ridge)."""
    try:
        from model_crafter.loss import _SquaredErrorLoss
        from model_crafter.penalty import L2Penalty
    except ImportError:  # pragma: no cover
        return False

    spec = getattr(sol, "spec", None)
    if spec is None:
        return False
    return isinstance(spec.loss, _SquaredErrorLoss) and isinstance(
        spec.penalty, L2Penalty
    )


def _solution_supports_closed_form_hat(sol: Any) -> bool:
    return _solution_is_ols(sol) or _solution_is_ridge(sol)


def _solution_is_logistic(sol: Any) -> bool:
    """True iff the spec uses :class:`LogisticLoss`."""
    from model_crafter.loss import LogisticLoss

    spec = getattr(sol, "spec", None)
    if spec is None:
        return False
    return isinstance(spec.loss, LogisticLoss)


def _coef_se_from_solution(sol: Any) -> pd.Series | None:
    """Return the SE Series stored on the solution, if any."""
    se = getattr(sol, "coefficient_se", None)
    if se is None:
        return None
    return se


# ---------------------------------------------------------------------------
# coefficients
# ---------------------------------------------------------------------------


def coefficients(sol: Any) -> pd.DataFrame:
    r"""Return a coefficient table with estimates, SEs, and Wald statistics.

    The returned :class:`pandas.DataFrame` is indexed by design column
    name and has columns ``["estimate", "std_error", "z", "p_value"]``.

    Math
    ----
    For OLS (closed-form normal equations):

    .. math::

        \widehat{\mathrm{Var}}(\hat\beta) = \hat\sigma^2 (X^\top W X)^{-1},
        \qquad \hat\sigma^2 = \frac{\mathrm{RSS}}{n - p}

    The ``z`` column is the **t-statistic** :math:`\hat\beta_j / \mathrm{SE}_j`
    and the ``p_value`` is the two-sided tail probability of the
    :math:`t_{n-p}` distribution.

    For logistic IRLS (when ``coefficient_se`` is available), the ``z`` is
    the **Wald statistic** :math:`\hat\beta_j / \mathrm{SE}_j` and the
    ``p_value`` is two-sided standard-normal — exactly the convention
    R's :func:`glm` reports.

    For ridge / lasso / elastic-net fits, ``coefficient_se`` is ``None``
    by deliberate design (DESIGN.md §3.2 — the closed-form penalised SE
    is biased and uninformative); in that case ``std_error`` is ``None``
    and the ``z`` / ``p_value`` columns are ``NaN``. The docstring on
    :func:`mc.bootstrap` is the alternative uncertainty path (ESL §7.11).

    Parameters
    ----------
    sol
        A fitted :class:`~model_crafter.solution.Solution`.

    Returns
    -------
    pandas.DataFrame
        Indexed by design column name. Columns:
        ``estimate``, ``std_error``, ``z``, ``p_value``.

    References
    ----------
    ESL §3.2 (OLS standard errors), §4.4.4 (logistic Wald), §3.4 (penalised
    regression — no closed-form SE).
    """
    from scipy.stats import norm, t  # noqa: PLC0415

    coef = sol.coefficients
    if not isinstance(coef, pd.Series):
        raise TypeError("sol.coefficients must be a pandas Series")

    se = _coef_se_from_solution(sol)
    n_cols = len(coef)

    if se is None:
        std_arr = np.full(n_cols, np.nan, dtype=float)
        z_arr = np.full(n_cols, np.nan, dtype=float)
        p_arr = np.full(n_cols, np.nan, dtype=float)
    else:
        std_arr = se.reindex(coef.index).to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            z_arr = coef.to_numpy(dtype=float) / std_arr

        if _solution_is_ols(sol):
            # t-distribution with n - p df.
            df_resid = int(
                sol.solver_info.get("df_resid", max(sol.n_obs - n_cols, 1))
            )
            p_arr = 2.0 * (1.0 - t.cdf(np.abs(z_arr), df=df_resid))
        else:
            # Wald: standard-normal tail probability.
            p_arr = 2.0 * (1.0 - norm.cdf(np.abs(z_arr)))

    return pd.DataFrame(
        {
            "estimate": coef.to_numpy(dtype=float),
            "std_error": std_arr,
            "z": z_arr,
            "p_value": p_arr,
        },
        index=coef.index,
    )


# ---------------------------------------------------------------------------
# diagnostics / hat_matrix / influence — closed-form only
# ---------------------------------------------------------------------------


def _bootstrap_pointer(what: str) -> str:
    return (
        f"{what} is only defined for closed-form linear models "
        "(OLS or closed-form ridge). For lasso, elastic net, or logistic "
        "regression use `mc.bootstrap(sol, data)` (ESL §7.11) — the "
        "bootstrap is the recommended uncertainty / influence diagnostic "
        "for non-closed-form fits."
    )


def _build_design_for_diagnostics(sol: Any) -> tuple[np.ndarray, np.ndarray]:
    r"""Rebuild ``(X, y)`` from ``sol.spec`` against the *training* data.

    Diagnostics need the design matrix the model was fit on; we don't
    persist it on the Solution (DESIGN.md §6.2 keeps the value compact).
    The caller must supply ``sol`` via :func:`diagnostics` /
    :func:`hat_matrix` / :func:`influence` having stashed the training
    frame on ``sol.solver_info`` or passed it through — but for the v0
    contract we accept the more pragmatic interface: ``sol`` carries
    enough to rebuild ``X`` from ``sol.fit_state`` if a training data
    frame is provided to the diagnostic call. Since we cannot mandate
    that, the v0 diagnostics functions raise a clear error directing
    callers to use the explicit ``hat_matrix(sol, data)`` form when
    needed.
    """
    raise RuntimeError(  # pragma: no cover — never called directly
        "internal: _build_design_for_diagnostics is a sentinel"
    )


@dataclass(frozen=True, slots=True)
class Diagnostics:
    r"""Residual / leverage / Cook's-distance diagnostics for a fitted model.

    Closed-form linear-model diagnostics only (DESIGN.md §5; ESL §3.3).

    Attributes
    ----------
    residuals
        :math:`e_i = y_i - \hat y_i`.
    leverage
        :math:`h_{ii}`, the diagonal of the hat matrix
        :math:`H = X(X^\top X)^{-1} X^\top` (or :math:`H_\lambda` for
        ridge).
    cooks_distance
        :math:`D_i = (e_i^2 / (p \hat\sigma^2)) (h_{ii} / (1 - h_{ii})^2)`
        — the leave-one-out influence on the fitted values.
    studentized_residuals
        :math:`r_i = e_i / (\hat\sigma \sqrt{1 - h_{ii}})` — the
        internally-studentized residual.
    sigma2
        :math:`\hat\sigma^2 = \mathrm{RSS} / (n - p)`.
    """

    residuals: pd.Series
    leverage: pd.Series
    cooks_distance: pd.Series
    studentized_residuals: pd.Series
    sigma2: float

    def __repr__(self) -> str:
        n = len(self.residuals)
        lev = self.leverage
        high_lev = int((lev > 2.0 * lev.mean()).sum()) if n > 0 else 0
        cook = self.cooks_distance
        # Cook's distance "large" rule of thumb is > 4/n (Bollen-Jackman).
        cook_threshold = 4.0 / max(n, 1)
        high_cook = int((cook > cook_threshold).sum()) if n > 0 else 0
        lines = [
            "Diagnostics",
            f"  n={n}  sigma^2={self.sigma2:.4g}",
            f"  high leverage (h_ii > 2*mean): {high_lev}",
            f"  influential rows (D > 4/n={cook_threshold:.3g}): {high_cook}",
        ]
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class Influence:
    r"""Influence diagnostics (DFBETAS + Cook's distance + leverage).

    Closed-form linear models only. ESL §3.3 is the underlying reference.

    Attributes
    ----------
    dfbetas
        :class:`pandas.DataFrame` indexed by row, columns indexed by
        design column. Cell ``(i, j)`` is

        .. math::

            \mathrm{DFBETAS}_{ij} =
                \frac{\hat\beta_j - \hat\beta_j^{(-i)}}
                     {\mathrm{SE}^{(-i)}(\hat\beta_j)}

        — the standardised change in :math:`\hat\beta_j` from deleting
        row :math:`i`. ESL §3.3 / Belsley-Kuh-Welsch (1980).
    cooks_distance
        Same series as in :class:`Diagnostics`; cross-linked for
        convenience.
    leverage
        :math:`h_{ii}` per row.
    """

    dfbetas: pd.DataFrame
    cooks_distance: pd.Series
    leverage: pd.Series

    def __repr__(self) -> str:
        n_rows, n_cols = self.dfbetas.shape
        # "Large DFBETAS" rule of thumb |dfbeta| > 2/sqrt(n).
        n = max(n_rows, 1)
        thresh = 2.0 / np.sqrt(n)
        flagged = int((self.dfbetas.abs() > thresh).any(axis=1).sum())
        return (
            f"Influence(n_rows={n_rows}, n_coef={n_cols}, "
            f"rows_with_|dfbetas|>2/sqrt(n)={flagged})"
        )


def _rebuild_design_and_y(sol: Any, data: pd.DataFrame) -> tuple[
    pd.DataFrame, np.ndarray
]:
    """Build the design matrix and y vector from a fitted solution + the
    training frame.

    Diagnostics require ``data`` to recompute residuals / leverages, since
    a :class:`Solution` does not persist :math:`X` or :math:`y`. The
    caller passes the same frame ``sol`` was fit on.
    """
    from model_crafter._internal.design import build_design  # noqa: PLC0415

    design = build_design(sol.spec, data, fit_state=sol.fit_state)
    if list(design.columns) != list(sol.design_columns):
        raise ValueError(
            "data does not reproduce the training design columns; "
            "diagnostics(sol, data) expects the same frame the solution "
            "was fit on. "
            f"Training columns: {list(sol.design_columns)}; "
            f"got: {list(design.columns)}"
        )
    target = sol.spec.target
    if target not in data.columns:
        raise KeyError(
            f"target column {target!r} not in data; "
            f"diagnostics(sol, data) needs the original training frame"
        )
    y_series = pd.to_numeric(data[target], errors="raise")
    y = np.asarray(y_series, dtype=float)
    return pd.DataFrame(design.values, columns=pd.Index(design.columns)), y


def hat_matrix(sol: Any, data: pd.DataFrame | None = None) -> np.ndarray:
    r"""Return the :math:`n \times n` hat matrix for a closed-form fit.

    .. math::

        H = X (X^\top X)^{-1} X^\top \qquad (\mathrm{OLS})

        H_\lambda = X (X^\top X + n \lambda I')^{-1} X^\top \qquad (\mathrm{ridge})

    where :math:`I'` zeroes the intercept row/column when the spec
    has an unpenalised intercept (DESIGN.md §3.4 ridge convention).

    Parameters
    ----------
    sol
        A fitted :class:`~model_crafter.solution.Solution`.
    data
        The training frame (the one ``sol`` was fit on). Required because
        :class:`Solution` does not persist :math:`X`.

    Returns
    -------
    numpy.ndarray
        Shape ``(n, n)``. **Warning** — :math:`O(n^2)` memory. For large
        :math:`n`, prefer :func:`diagnostics`\\ ``(sol, data).leverage``,
        which only materialises the diagonal.

    Raises
    ------
    NotImplementedError
        For lasso, elastic-net, or logistic fits — see
        :func:`mc.bootstrap` (ESL §7.11).
    """
    if not _solution_supports_closed_form_hat(sol):
        if _solution_is_logistic(sol):
            raise NotImplementedError(_bootstrap_pointer("hat_matrix"))
        raise NotImplementedError(_bootstrap_pointer("hat_matrix"))
    if data is None:
        raise TypeError(
            "hat_matrix(sol, data) requires the training data frame "
            "(Solution does not persist X)"
        )

    X, _y = _rebuild_design_and_y(sol, data)
    X_arr = X.to_numpy(dtype=float)
    n, _p = X_arr.shape

    if n > 5000:
        warnings.warn(
            f"hat_matrix returning an (n={n}) x (n={n}) dense array — "
            f"this is O(n^2) memory. For large n, prefer "
            f"`diagnostics(sol, data).leverage` which materialises only "
            f"the diagonal.",
            stacklevel=2,
        )

    if _solution_is_ols(sol):
        # H = X (X^T X)^-1 X^T
        XtX = X_arr.T @ X_arr
        # Solve X^T X · A = X^T → A = (X^T X)^-1 X^T; then H = X · A.
        A = np.linalg.solve(XtX, X_arr.T)
        H = X_arr @ A
    else:
        # Ridge: H_lambda = X (X^T X + n*lambda*I')^-1 X^T.
        from model_crafter._internal.design import INTERCEPT_NAME  # noqa: PLC0415

        lam = float(getattr(sol.spec.penalty, "lam", 0.0))
        XtX = X_arr.T @ X_arr
        # I' has 0 on the intercept row/column (unpenalised intercept).
        Iprime = np.eye(X_arr.shape[1], dtype=float)
        cols = list(sol.design_columns)
        if INTERCEPT_NAME in cols:
            idx = cols.index(INTERCEPT_NAME)
            Iprime[idx, idx] = 0.0
        A = np.linalg.solve(XtX + n * lam * Iprime, X_arr.T)
        H = X_arr @ A
    return np.asarray(H, dtype=float)


def diagnostics(sol: Any, data: pd.DataFrame | None = None) -> Diagnostics:
    r"""Residual / leverage / Cook's-distance diagnostics (closed-form only).

    Math
    ----
    .. math::

        e_i &= y_i - \hat y_i \\
        h_{ii} &= \big[ X (X^\top X)^{-1} X^\top \big]_{ii} \\
        \hat\sigma^2 &= \mathrm{RSS} / (n - p) \\
        D_i &= \frac{e_i^2}{p \hat\sigma^2} \cdot
                 \frac{h_{ii}}{(1 - h_{ii})^2} \\
        r_i &= \frac{e_i}{\hat\sigma \sqrt{1 - h_{ii}}}

    ESL §3.3.

    Parameters
    ----------
    sol
        A fitted :class:`Solution`.
    data
        Training data the solution was fit on.

    Raises
    ------
    NotImplementedError
        For lasso / elastic-net / logistic fits — use :func:`mc.bootstrap`
        instead (ESL §7.11).
    """
    if not _solution_supports_closed_form_hat(sol):
        raise NotImplementedError(_bootstrap_pointer("diagnostics"))
    if data is None:
        raise TypeError(
            "diagnostics(sol, data) requires the training data frame "
            "(Solution does not persist X)"
        )

    X, y = _rebuild_design_and_y(sol, data)
    X_arr = X.to_numpy(dtype=float)
    n, p = X_arr.shape
    cols = list(sol.design_columns)

    beta = sol.coefficients.reindex(cols).to_numpy(dtype=float)
    yhat = X_arr @ beta
    resid = y - yhat
    rss = float(np.sum(resid * resid))
    df_resid = max(n - p, 1)
    sigma2 = rss / df_resid

    H = hat_matrix(sol, data)
    leverage = np.diag(H)
    # Avoid division-by-zero when h_ii = 1 (perfect leverage).
    one_minus_h = np.where(leverage >= 1.0, np.nan, 1.0 - leverage)

    with np.errstate(divide="ignore", invalid="ignore"):
        cooks = (resid * resid / (p * sigma2)) * (
            leverage / (one_minus_h * one_minus_h)
        )
        studentized = resid / (np.sqrt(sigma2) * np.sqrt(one_minus_h))

    index = data.index
    return Diagnostics(
        residuals=pd.Series(resid, index=index, name="residual"),
        leverage=pd.Series(leverage, index=index, name="leverage"),
        cooks_distance=pd.Series(cooks, index=index, name="cooks_distance"),
        studentized_residuals=pd.Series(
            studentized, index=index, name="studentized_residual"
        ),
        sigma2=float(sigma2),
    )


def influence(sol: Any, data: pd.DataFrame | None = None) -> Influence:
    r"""Per-row DFBETAS + Cook's distance + leverage.

    Math
    ----
    For OLS, the leave-one-out coefficient change has the closed form

    .. math::

        \hat\beta - \hat\beta^{(-i)} =
            \frac{(X^\top X)^{-1} x_i e_i}{1 - h_{ii}}

    and DFBETAS standardises by the leave-one-out SE
    :math:`\mathrm{SE}^{(-i)}(\hat\beta_j) =
    \sqrt{\hat\sigma^2_{(-i)} \cdot [(X^\top X)^{-1}]_{jj}}` where

    .. math::

        \hat\sigma^2_{(-i)} = \hat\sigma^2 \cdot
            \frac{n - p - r_i^{*2}}{n - p - 1}

    and :math:`r_i^*` is the studentized residual. ESL §3.3.

    Parameters
    ----------
    sol
        A fitted :class:`Solution`.
    data
        Training frame.

    Raises
    ------
    NotImplementedError
        For non-closed-form fits — use :func:`mc.bootstrap` instead.
    """
    if not _solution_supports_closed_form_hat(sol):
        raise NotImplementedError(_bootstrap_pointer("influence"))
    if data is None:
        raise TypeError(
            "influence(sol, data) requires the training data frame "
            "(Solution does not persist X)"
        )

    X, y = _rebuild_design_and_y(sol, data)
    X_arr = X.to_numpy(dtype=float)
    n, p = X_arr.shape
    cols = list(sol.design_columns)

    beta = sol.coefficients.reindex(cols).to_numpy(dtype=float)
    yhat = X_arr @ beta
    resid = y - yhat
    rss = float(np.sum(resid * resid))
    df_resid = max(n - p, 1)
    sigma2 = rss / df_resid

    XtX = X_arr.T @ X_arr
    XtX_inv = np.linalg.inv(XtX)
    H = X_arr @ XtX_inv @ X_arr.T
    leverage = np.diag(H)

    # Leave-one-out coefficient change: ΔB_i = (X^T X)^-1 x_i e_i / (1 - h_ii)
    one_minus_h = np.where(leverage >= 1.0, np.nan, 1.0 - leverage)
    # Build (p, n) matrix of ΔB column-vectors. Vectorised:
    # A = (X^T X)^-1 X^T  → shape (p, n); scaled by e_i/(1-h_ii) per col.
    A = XtX_inv @ X_arr.T  # (p, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = resid / one_minus_h  # (n,)
    delta_beta = A * scale[np.newaxis, :]  # (p, n)

    # Studentized residual r_i for leave-one-out sigma2.
    with np.errstate(divide="ignore", invalid="ignore"):
        studentized = resid / (np.sqrt(sigma2) * np.sqrt(one_minus_h))
    # Externally studentized sigma2_(-i)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = df_resid - 1
        # Use the internally-studentized r_i to derive sigma2_(-i):
        # sigma2_(-i) = sigma2 * (n - p - r_i^2) / (n - p - 1)
        sigma2_minus_i = sigma2 * (df_resid - studentized * studentized) / max(denom, 1)
        # Guard against tiny negative values from numerical noise
        sigma2_minus_i = np.where(sigma2_minus_i <= 0, np.nan, sigma2_minus_i)

    # SE^{(-i)}(beta_j) = sqrt( sigma2_(-i) * [(X^T X)^-1]_{jj} )
    diag_inv = np.diag(XtX_inv)  # (p,)
    # Build SE matrix (n, p): row i, col j → sqrt(sigma2_(-i) * diag_inv_j)
    se_minus_i = np.sqrt(np.outer(sigma2_minus_i, diag_inv))  # (n, p)

    # DFBETAS_{ij} = delta_beta_{j,i} / se_minus_i_{i,j}
    with np.errstate(divide="ignore", invalid="ignore"):
        dfbetas = delta_beta.T / se_minus_i  # (n, p)

    # Cook's distance via the diagnostics formula (cross-linked).
    with np.errstate(divide="ignore", invalid="ignore"):
        cooks = (resid * resid / (p * sigma2)) * (
            leverage / (one_minus_h * one_minus_h)
        )

    index = data.index
    return Influence(
        dfbetas=pd.DataFrame(dfbetas, index=index, columns=pd.Index(cols)),
        cooks_distance=pd.Series(cooks, index=index, name="cooks_distance"),
        leverage=pd.Series(leverage, index=index, name="leverage"),
    )
