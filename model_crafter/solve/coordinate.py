r"""Coordinate descent solver for lasso and elastic-net least squares.

References
----------
* Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*
  (2nd ed.), §3.4.2 (lasso), §3.4.3 (elastic net).
* Friedman, Hastie & Tibshirani 2010, *Regularization Paths for Generalized
  Linear Models via Coordinate Descent*, JStatSoft 33(1) — referred to
  below as **FHT 2010**. The coordinate-update equation is FHT eq. 5.

Objectives
----------
For the **lasso** (ESL §3.4.2):

.. math::

   \tfrac{1}{2n} \lVert y - X\beta\rVert_2^2
       \;+\; \lambda \sum_j \lvert\beta_j\rvert ,

with the intercept :math:`\beta_0` unpenalised. For the **elastic net**
(ESL §3.4.3, FHT 2010 §2.1):

.. math::

   \tfrac{1}{2n} \lVert y - X\beta\rVert_2^2
       \;+\; \lambda_1 \sum_j \lvert\beta_j\rvert
       \;+\; \tfrac{\lambda_2}{2} \sum_j \beta_j^2 .

In the ``glmnet`` parameterisation, :math:`\lambda_1 = \lambda \alpha` and
:math:`\lambda_2 = \lambda(1 - \alpha)`. This module accepts the
:math:`(\lambda_1, \lambda_2)` form directly — the per-penalty
:math:`\lambda` values flow through from the spec's ``L1Penalty.lam`` and
``L2Penalty.lam``.

Coordinate update
-----------------
With ``X`` standardised so each column has (weighted) sample SD 1 and ``y``
(weighted-) centred, the per-coordinate elastic-net update is FHT 2010
eq. 5:

.. math::

   \beta_j \;\leftarrow\;
       \frac{\operatorname{S}\!\bigl(\tfrac{1}{n}\,x_j^\top r_{(-j)}, \;\lambda_1\bigr)}
            {\tfrac{1}{n}\lVert x_j\rVert_2^2 \;+\; \lambda_2}

where :math:`r_{(-j)} = y - X\beta + x_j\beta_j` is the partial residual
*excluding* coordinate :math:`j`, and :math:`S(z, \gamma) =
\mathrm{sign}(z)\,(|z| - \gamma)_+` is soft-thresholding. The
denominator's :math:`\lVert x_j\rVert_2^2/n` equals 1 for standardised
columns, so it simplifies to :math:`1 + \lambda_2` — but this code keeps
the explicit form because rounding makes the simplification mildly lossy.

Sample weights
--------------
Weights enter every inner product. We use the **unscaled** weighted loss
:math:`\tfrac{1}{2n}\sum_i w_i (y_i - x_i^\top\beta)^2` with weights
internally normalised so they sum to :math:`n` (matching ``glmnet``'s
``weights=`` convention — see FHT 2010 §3 paragraph 2). The convergence
criterion and the lambda path are unchanged by this normalisation.

Standardisation and de-standardisation (``glmnet`` convention)
--------------------------------------------------------------
Coefficients on standardised :math:`X` differ from coefficients on the
original :math:`X` only by per-column rescaling. We fit on the
standardised matrix and convert back:

.. math::

   \beta_j^{\mathrm{orig}} \;=\; \beta_j^{\mathrm{std}} / \mathrm{sd}_j,
   \qquad
   \beta_0^{\mathrm{orig}} \;=\; \bar y - \sum_j \mathrm{mean}_j \;\beta_j^{\mathrm{orig}} .

This is the same convention ``glmnet`` (and most lasso implementations)
use. **Important:** when users compare coefficients to ``glmnet`` output,
they should compare to the de-standardised coefficients — which is what
this solver returns.

Warm starts
-----------
The coordinate updates converge much faster when seeded near the optimum.
On a descending λ path, the previous fit is an excellent warm start for
the next. At :math:`\lambda \geq \lambda_{\max}` the all-zero coefficient
vector is exactly optimal (KKT), so the first point on the path is fit
trivially.

Naive vs covariance updates
---------------------------
FHT 2010 §2.2 contrasts two implementations: **naive** updates recompute
:math:`x_j^\top r` from scratch each sweep (O(np) per sweep), while
**covariance** updates cache :math:`X^\top X` and :math:`X^\top y` so each
sweep is O(p) per coordinate (O(p²) per sweep). Naive wins when many
coefficients change per sweep or when :math:`n \approx p`; covariance
wins for large :math:`n \gg p` with few active coefficients. **This
solver uses naive updates** — they are simpler, numerically robust, and
fast enough for the credit-risk problem sizes (p up to a few hundred)
that ``model_crafter`` targets. Switching to covariance updates is
straightforward and slated for Phase 6 if profiling demands.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from model_crafter.loss import _SquaredErrorLoss
from model_crafter.solve._registry import (
    SolverInputs,
    SolverOutputs,
    register,
)

# Inner numerical kernel


@dataclass(frozen=True, slots=True)
class CoordinateDescentResult:
    """Output of a single coordinate-descent fit (standardised scale)."""

    beta_std: np.ndarray         # shape (p,) — on the standardised X scale
    intercept_std: float          # intercept on the standardised X scale (= 0 when y centred)
    n_iter: int
    converged: bool
    max_change: float


def _normalize_weights(weights: np.ndarray | None, n: int) -> np.ndarray:
    r"""Return weights scaled to sum to ``n`` (``glmnet`` convention).

    With this normalisation the weighted CD updates differ from the OLS
    closed form only by the penalty term, so :math:`\lambda` has the same
    meaning across the weighted and unweighted cases.
    """
    if weights is None:
        return np.ones(n, dtype=float)
    w = np.asarray(weights, dtype=float)
    if w.shape != (n,):
        raise ValueError(f"weights shape {w.shape} != ({n},)")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    s = float(np.sum(w))
    if s <= 0:
        raise ValueError("weights must contain at least one positive value")
    return w * (n / s)


def _standardise(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    *,
    intercept: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    r"""Centre and scale X (and centre y) using **weighted** moments.

    Returns
    -------
    Xs:
        Standardised :math:`X`: each column has weighted mean 0 and
        weighted (population) SD 1.
    yc:
        Weighted-centred :math:`y` (or :math:`y` itself when
        ``intercept=False``).
    col_mean, col_sd:
        Per-column mean and SD used to undo the standardisation.
    y_mean:
        Weighted mean of :math:`y` (0 when ``intercept=False``).
    nonzero_sd:
        Boolean mask of columns with non-zero SD — constant columns are
        kept in :math:`X_s` as identically-zero columns; they cannot enter
        the model under L1 because their gradient is zero.
    """
    n = X.shape[0]
    wsum = float(np.sum(w))  # equals n after _normalize_weights
    col_mean = (w @ X) / wsum
    if intercept:
        Xc = X - col_mean
    else:
        # No intercept — do not centre X (the user is asking for a through-the-origin fit).
        Xc = X.copy()
        col_mean = np.zeros(X.shape[1], dtype=float)
    col_var = (w @ (Xc * Xc)) / wsum
    col_sd = np.sqrt(col_var)
    nonzero = col_sd > 0
    safe_sd = np.where(nonzero, col_sd, 1.0)
    Xs = Xc / safe_sd
    # Zero out constant columns so they don't perturb sweeps.
    if np.any(~nonzero):
        Xs[:, ~nonzero] = 0.0
    if intercept:
        y_mean = float(np.sum(w * y) / wsum)
        yc = y - y_mean
    else:
        y_mean = 0.0
        yc = y
    _ = n  # silence unused in some toolchains
    return Xs, yc, col_mean, col_sd, y_mean, nonzero


def _soft_threshold(z: float, gamma: float) -> float:
    """Soft-thresholding operator :math:`S(z, \\gamma)`."""
    if z > gamma:
        return z - gamma
    if z < -gamma:
        return z + gamma
    return 0.0


def coordinate_descent_path(
    X: np.ndarray,
    y: np.ndarray,
    lambdas_l1: Sequence[float] | np.ndarray,
    *,
    lam_l2: float = 0.0,
    weights: np.ndarray | None = None,
    intercept: bool = True,
    tol: float = 1e-7,
    max_iter: int = 1000,
    warm_start: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[CoordinateDescentResult]]:
    r"""Fit elastic-net coordinate descent at each :math:`\lambda_1` in ``lambdas_l1``.

    Each entry of ``lambdas_l1`` is the L1 penalty strength :math:`\lambda_1`.
    ``lam_l2`` is fixed across the path — this matches the way
    :func:`lambda_path` parameterises the elastic-net family.

    Returns
    -------
    betas:
        Shape ``(n_lambda, p)`` — coefficients on the **original** X
        scale, with the intercept *separate* (returned alongside).
    intercepts:
        Shape ``(n_lambda,)`` — intercept on the original scale.
    infos:
        Per-λ :class:`CoordinateDescentResult` (the standardised-scale
        coefficients and convergence info).
    """
    X = np.ascontiguousarray(X, dtype=float)
    y = np.ascontiguousarray(y, dtype=float)
    n, p = X.shape
    if y.shape != (n,):
        raise ValueError(f"y shape {y.shape} != ({n},)")
    w = _normalize_weights(weights, n)

    Xs, yc, col_mean, col_sd, y_mean, nonzero_sd = _standardise(
        X, y, w, intercept=intercept
    )

    # Pre-compute column squared norms (weighted, divided by n).
    # With weights summing to n and X standardised, this equals 1 for
    # non-constant columns. Compute it explicitly so the inner loop is correct
    # for constant columns too (which sit at 0).
    n_f = float(n)
    col_xx = (w @ (Xs * Xs)) / n_f
    # Numerical safety: clip extremely small column norms.
    col_xx = np.where(nonzero_sd, col_xx, 0.0)

    lambdas_l1 = np.asarray(lambdas_l1, dtype=float)
    n_lam = lambdas_l1.shape[0]
    betas_std = np.zeros((n_lam, p), dtype=float)
    betas_orig = np.zeros((n_lam, p), dtype=float)
    intercepts_orig = np.zeros(n_lam, dtype=float)
    infos: list[CoordinateDescentResult] = []

    beta = np.zeros(p, dtype=float)
    # Initial residual = y_c - X_s beta = y_c (because beta = 0).
    residual = yc.copy()

    for k, lam1 in enumerate(lambdas_l1):
        lam1_f = float(lam1)
        if not warm_start:
            beta[:] = 0.0
            residual = yc.copy()
        # CD sweeps
        n_iter, converged, max_change = _cd_loop(
            Xs, w, residual, beta, col_xx, lam1_f, lam_l2, tol, max_iter, nonzero_sd
        )

        # Capture standardised-scale result
        betas_std[k] = beta
        infos.append(
            CoordinateDescentResult(
                beta_std=beta.copy(),
                intercept_std=0.0,
                n_iter=n_iter,
                converged=converged,
                max_change=max_change,
            )
        )
        # De-standardise: beta_orig[j] = beta_std[j] / sd_j for non-constant j.
        beta_orig = np.zeros(p, dtype=float)
        beta_orig[nonzero_sd] = beta[nonzero_sd] / col_sd[nonzero_sd]
        intercept_orig = (
            y_mean - float(col_mean @ beta_orig) if intercept else 0.0
        )
        betas_orig[k] = beta_orig
        intercepts_orig[k] = intercept_orig

    return betas_orig, intercepts_orig, infos


def _cd_loop(
    Xs: np.ndarray,
    w: np.ndarray,
    residual: np.ndarray,
    beta: np.ndarray,
    col_xx: np.ndarray,
    lam1: float,
    lam2: float,
    tol: float,
    max_iter: int,
    nonzero_sd: np.ndarray,
) -> tuple[int, bool, float]:
    r"""Run cyclic coordinate-descent sweeps until convergence.

    Mutates ``residual`` and ``beta`` in place. Returns
    ``(n_iter, converged, max_change_on_last_sweep)``.

    Convergence criterion: maximum absolute coefficient change across one
    full sweep is less than ``tol``.
    """
    n = Xs.shape[0]
    p = Xs.shape[1]
    n_f = float(n)
    converged = False
    last_max_change = float("inf")
    for it in range(1, max_iter + 1):
        max_change = 0.0
        for j in range(p):
            if not nonzero_sd[j]:
                continue
            xj = Xs[:, j]
            bj_old = beta[j]
            # Partial residual r_(-j) = r + xj * bj_old.  Compute z_j = (1/n) * xj^T w (r + xj*bj_old).
            #   = (1/n) (xj^T (w * r)) + (col_xx[j]) * bj_old
            z = float(xj @ (w * residual)) / n_f + col_xx[j] * bj_old
            denom = col_xx[j] + lam2
            if denom <= 0:
                # Defensive — shouldn't happen for non-constant columns.
                continue
            bj_new = _soft_threshold(z, lam1) / denom
            if bj_new != bj_old:
                # Update residual: r <- r - xj * (bj_new - bj_old)
                residual -= xj * (bj_new - bj_old)
                beta[j] = bj_new
                ch = abs(bj_new - bj_old)
                if ch > max_change:
                    max_change = ch
        last_max_change = max_change
        if max_change < tol:
            converged = True
            return it, converged, last_max_change
    return max_iter, converged, last_max_change


# Penalty-parameter extraction (works with P2.A's L1Penalty / L2Penalty /
# PenaltySum or any duck-typed equivalent — penalty types are not yet in
# model_crafter.penalty at the time this module is written; integration
# wires them up).


def _is_l1_like(p: Any) -> bool:
    """True for an L1Penalty-shaped object (has a ``lam`` and class named L1Penalty)."""
    return type(p).__name__ == "L1Penalty" and hasattr(p, "lam")


def _is_l2_like(p: Any) -> bool:
    return type(p).__name__ == "L2Penalty" and hasattr(p, "lam")


def _split_enet_parts(penalty: Any) -> tuple[float, float]:
    r"""Return ``(lam_l1, lam_l2)`` from a penalty.

    Accepts a plain L1Penalty, L2Penalty, or a PenaltySum of L1 + L2 (in
    either order). Raises a clear ``TypeError`` for anything else
    (PenaltySum containing two L1s, an unknown penalty, etc.). The
    parameter is typed ``Any`` because the penalty types are duck-typed —
    P2.A's ``L1Penalty`` and ``L2Penalty`` aren't depended on at
    type-check time (they may not be importable in early Phase-2 commits;
    see the registration block at the bottom of this module).
    """
    if _is_l1_like(penalty):
        return float(penalty.lam), 0.0
    if _is_l2_like(penalty):
        return 0.0, float(penalty.lam)
    parts: Iterable[Any] | None = getattr(penalty, "parts", None)
    if parts is None:
        raise TypeError(
            f"coordinate-descent solver expects L1 / L2 / L1+L2 penalty; "
            f"got {type(penalty).__name__}"
        )
    lam_l1 = 0.0
    lam_l2 = 0.0
    seen_l1 = False
    seen_l2 = False
    for part in parts:
        if _is_l1_like(part):
            if seen_l1:
                raise TypeError(
                    "coordinate-descent solver: PenaltySum has multiple L1 parts; "
                    "combine them into a single L1Penalty before solving"
                )
            lam_l1 += float(part.lam)
            seen_l1 = True
        elif _is_l2_like(part):
            if seen_l2:
                raise TypeError(
                    "coordinate-descent solver: PenaltySum has multiple L2 parts; "
                    "combine them into a single L2Penalty before solving"
                )
            lam_l2 += float(part.lam)
            seen_l2 = True
        else:
            raise TypeError(
                f"coordinate-descent solver: unsupported PenaltySum part "
                f"{type(part).__name__}; expected L1Penalty or L2Penalty"
            )
    return lam_l1, lam_l2


# Solver entry points (registered with the dispatch)


def _separate_intercept_column(
    X: np.ndarray, columns: Sequence[str], intercept: bool
) -> tuple[np.ndarray, list[str], bool]:
    """If ``columns[0] == '(Intercept)'``, peel it off the design.

    Returns ``(X_no_int, names_no_int, had_intercept_column)``. CD handles
    the intercept itself (always unpenalised, fit via centring); a leading
    intercept column would be a redundancy that the CD update sweeps would
    drive to a singular soft-threshold. We strip it and add the intercept
    back to the coefficient vector after the fit.
    """
    if intercept and len(columns) > 0 and columns[0] == "(Intercept)":
        return X[:, 1:], list(columns[1:]), True
    return X, list(columns), False


def _run_single_lambda(
    inputs: SolverInputs,
    *,
    lam_l1: float,
    lam_l2: float,
    tol: float = 1e-7,
    max_iter: int = 1000,
) -> SolverOutputs:
    r"""Fit a single (λ₁, λ₂) point and wrap the result as ``SolverOutputs``.

    Coefficients are returned on the **original** ``X`` scale, with the
    intercept named ``"(Intercept)"`` to match the design's column
    convention.
    """
    X = inputs.design.values
    y = inputs.y
    w = inputs.weights
    spec = inputs.spec

    X_noint, names_noint, had_intercept = _separate_intercept_column(
        X, inputs.design.columns, spec.intercept
    )

    betas, intercepts, infos = coordinate_descent_path(
        X_noint,
        y,
        lambdas_l1=[lam_l1],
        lam_l2=lam_l2,
        weights=w,
        intercept=had_intercept,
        tol=tol,
        max_iter=max_iter,
        warm_start=True,
    )
    beta_orig = betas[0]
    intercept_val = float(intercepts[0])
    info = infos[0]

    # Build coefficients pd.Series in the same column order as the design.
    coef_values = []
    coef_names = list(inputs.design.columns)
    if had_intercept:
        coef_values.append(intercept_val)
    coef_values.extend(beta_orig.tolist())
    coefficients = pd.Series(coef_values, index=coef_names, name="coefficient")

    # Loss + penalty values (on the original scale).
    eta = X @ coefficients.to_numpy(dtype=float)
    loss_value = spec.loss.value(y, eta, weights=w)
    penalty_value = float(spec.penalty.value(coefficients.to_numpy(dtype=float)))

    solver_info = MappingProxyType(
        {
            "solver": "coordinate_descent",
            "lam_l1": float(lam_l1),
            "lam_l2": float(lam_l2),
            "n_iter": int(info.n_iter),
            "max_iter": int(max_iter),
            "tol": float(tol),
            "max_coef_change": float(info.max_change),
            "standardised": True,
            "update_scheme": "naive",
        }
    )

    return SolverOutputs(
        coefficients=coefficients,
        coefficient_se=None,  # SEs for penalised fits need the bootstrap (Phase 3).
        fit_state=MappingProxyType({}),
        loss_value=float(loss_value),
        penalty_value=penalty_value,
        n_obs=int(X.shape[0]),
        converged=bool(info.converged),
        solver_info=solver_info,
    )


def solve_lasso_cd(inputs: SolverInputs) -> SolverOutputs:
    r"""Solver entry point for ``(squared_error, L1Penalty)``.

    The single-:math:`\lambda` fit follows the coordinate-descent recipe
    documented at the top of this module. For a path of :math:`\lambda`
    values, call :func:`coordinate_descent_path` directly — that's the
    warm-started entry point used by ``mc.tune`` in Phase 3.
    """
    lam_l1, lam_l2 = _split_enet_parts(inputs.spec.penalty)
    if lam_l2 != 0.0:
        raise ValueError(
            "solve_lasso_cd dispatched on a penalty with non-zero L2 weight; "
            "route this case through solve_enet_cd (the dispatch should have "
            "selected PenaltySum, not L1Penalty)."
        )
    return _run_single_lambda(inputs, lam_l1=lam_l1, lam_l2=0.0)


def solve_enet_cd(inputs: SolverInputs) -> SolverOutputs:
    r"""Solver entry point for ``(squared_error, PenaltySum)`` (elastic net).

    The penalty is expected to be a sum of one ``L1Penalty`` and one
    ``L2Penalty`` (either order). A pure-L2 ``PenaltySum`` is rejected
    because the closed-form ridge solver (Task P2.B) handles that case
    far more efficiently. A pure-L1 ``PenaltySum`` is accepted and routes
    through the elastic-net update with ``lam_l2 = 0``.
    """
    lam_l1, lam_l2 = _split_enet_parts(inputs.spec.penalty)
    if lam_l1 == 0.0 and lam_l2 > 0.0:
        raise ValueError(
            "solve_enet_cd dispatched on a PenaltySum with no L1 component; "
            "route this case to the ridge solver instead."
        )
    return _run_single_lambda(inputs, lam_l1=lam_l1, lam_l2=lam_l2)


# Self-registration (best-effort: works when P2.A's L1Penalty / PenaltySum
# are present in model_crafter.penalty; falls back to a deferred-registration
# pattern when they aren't yet, so this module remains importable on the
# pre-P2.A branch and an explicit ``register_solvers()`` call from the
# integration agent wires it up).


_REGISTERED = False


def register_solvers() -> None:
    """Register the CD solvers with the dispatch.

    Idempotent: a second call is a no-op (avoids the registry's
    duplicate-key guard).
    """
    global _REGISTERED
    if _REGISTERED:
        return
    try:
        # P2.A's L1Penalty / PenaltySum live in model_crafter.penalty; at the
        # time this module was committed those symbols may not yet exist on
        # the integration branch, so we resolve them via getattr after a
        # module-level import to keep both runtime and type-checker happy.
        import model_crafter.penalty as _pen

        L1Penalty = _pen.L1Penalty  # pyright: ignore[reportAttributeAccessIssue]
        PenaltySum = _pen.PenaltySum  # pyright: ignore[reportAttributeAccessIssue]
    except (ImportError, AttributeError):
        # P2.A hasn't merged yet. The integration agent calls this function
        # after merging P2.A.
        return
    register((_SquaredErrorLoss, L1Penalty), solve_lasso_cd)
    register((_SquaredErrorLoss, PenaltySum), solve_enet_cd)
    _REGISTERED = True


# Attempt registration on import. If P2.A's L1Penalty / PenaltySum aren't
# present yet this is a no-op and the integration agent wires it manually.
register_solvers()
