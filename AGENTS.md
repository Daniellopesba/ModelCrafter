# AGENTS.md — Parallel Execution Playbook

This document partitions the `model_crafter` build into parallelizable task
units suitable for Superpowers parallel agents. For each task, it specifies:

- **Workspace**: files the agent owns (writable)
- **Reads**: files the agent reads but does not write
- **Depends on**: tasks that must complete before this one starts
- **Public interface**: types/functions other agents may import from this
  workspace, with their signatures pinned
- **Acceptance**: the verifiable contract for "done"

The handoff model is: every task ends with a green test suite for its
workspace, a short note in `notes/<task-id>.md`, and an unchanged set of
files outside the workspace. An integration agent (or the human) merges.

---

## Conventions

- Task IDs follow `<phase>.<letter>` — `P1.A`, `P3.D`, etc.
- "Owns" means the agent is the *only* writer for those paths during the task.
- "Reads" is informational; the file system doesn't enforce it, but agents
  must not modify read-only paths.
- Tests for owned modules live under `tests/<owned-module-name>.py`.
- Tests for cross-cutting integration live under `tests/integration/` and
  are written by the integration agent, not parallel agents.
- Every task has a corresponding GitHub issue / Linear ticket with the same
  ID; the agent quotes the task ID in its PR title.

---

## Dependency graph (high level)

```
            P0
            │
            P1.A  ──┐
            P1.B  ──┼──> P1.INTEG
                    │
            P2.A  ──┐
            P2.B  ──┼──> P2.INTEG
            P2.C  ──┘
                    │
            P3.A  ──┐
            P3.B  ──┼──> P3.INTEG
            P3.C  ──┘
            P3.D  ──┘
                    │
            P4.A  ──┐
            P4.B  ──┼──> P4.INTEG
            P4.C  ──┘
                    │
            P5.A  ──┐
            P5.B  ──┼──> P5.INTEG
            P5.C  ──┘
                    │
            P6  (integration, serial)
```

Phases run sequentially. Within a phase, lettered tasks run in parallel.
`INTEG` tasks are serial integration steps owned by a designated integration
agent.

---

## Phase 0 — Scaffolding (serial, single agent)

### Task P0 — Project scaffolding

**Workspace**:
- `pyproject.toml`
- `.github/workflows/ci.yml`
- `ruff.toml`, `pyrightconfig.json`
- `model_crafter/__init__.py` (empty stub)
- `tests/__init__.py`
- `notes/.gitkeep`
- `README.md` (one paragraph + link to DESIGN.md)

**Reads**: `DESIGN.md`, `CLAUDE.md`

**Depends on**: nothing

**Public interface**: none (no code yet)

**Acceptance**:
- `pip install -e ".[dev]"` works on a clean venv.
- `pytest` runs (empty suite, exits 0).
- `ruff check .` and `pyright` pass with no errors.
- CI workflow runs all three on push.
- Python 3.11+ required; dependencies are numpy, scipy, pandas only (plus
  pytest, ruff, pyright, statsmodels, formulaic as dev deps).

---

## Phase 1 — Terms, OLS, assumptions framework

Two agents in parallel. P1.A builds the math; P1.B builds the assumptions
framework. They merge at P1.INTEG.

### Task P1.A — Terms, LinearSpec, OLS solver, Solution

**Workspace**:
- `model_crafter/spec.py`
- `model_crafter/solution.py`
- `model_crafter/terms/__init__.py`
- `model_crafter/terms/base.py`
- `model_crafter/loss.py` (just `squared_error` and `Loss` protocol)
- `model_crafter/penalty.py` (just `NoPenalty` and `Penalty` protocol)
- `model_crafter/solve/__init__.py`
- `model_crafter/solve/ols.py`
- `model_crafter/_internal/design.py`
- `model_crafter/_internal/linalg.py`
- `tests/test_terms.py`
- `tests/test_ols.py`

**Reads**: `DESIGN.md` (§2, §6, §8 Phase 1)

**Depends on**: P0

**Public interface** (other agents may import these):

```python
# model_crafter.terms.base
class Term(Protocol):
    name: str
    def expand(self, data: pd.DataFrame, *, fit_state: Mapping | None) -> ExpandedTerm: ...
    def __add__(self, other) -> "TermSum": ...

class RawTerm(Term): ...
class TermSum(Term):
    terms: tuple[Term, ...]

def _promote(x: str | Term) -> Term: ...
def _normalize_features(f: str | Term | Iterable) -> tuple[Term, ...]: ...

# model_crafter.loss
class Loss(Protocol):
    assumptions: tuple
    def value(self, y, eta, weights) -> float: ...
    def gradient(self, y, eta, weights) -> np.ndarray: ...
    def hessian(self, y, eta, weights) -> np.ndarray: ...

squared_error: Loss

# model_crafter.penalty
class Penalty(Protocol):
    assumptions: tuple
    def value(self, beta) -> float: ...
    def __add__(self, other) -> "PenaltySum": ...

NoPenalty: Penalty

# model_crafter.spec
@dataclass(frozen=True)
class LinearSpec:
    target: str
    features: tuple[Term, ...]
    loss: Loss
    penalty: Penalty
    intercept: bool

def linear(target, features, loss, penalty=NoPenalty(), intercept=True) -> LinearSpec: ...

# model_crafter.solution
@dataclass(frozen=True)
class Solution:
    spec: LinearSpec
    coefficients: pd.Series
    coefficient_se: pd.Series | None
    fit_state: Mapping
    design_columns: tuple[str, ...]
    loss_value: float
    penalty_value: float
    n_obs: int
    converged: bool
    solver_info: Mapping
    assumptions: "AssumptionReport"   # type comes from P1.B

# model_crafter.solve
def solve(spec, data, *, weights=None, method=None,
          on_violation="warn", suppress=(), classical_inference=False,
          stability_splitter=None) -> Solution: ...

def predict(sol, new_data) -> pd.Series: ...  # always probabilities for classifiers
```

The `assumptions` field on `Solution` is filled in by `solve` after calling
into the assumption framework from P1.B. P1.A imports `AssumptionReport` and
`run_assumptions(...)` from P1.B's public interface and treats them as
opaque values.

**Acceptance**:
- Reproduce ESL §3.2 prostate cancer OLS coefficients to 1e-8 against R's
  `lm()`. Coefficients, SEs, residual standard error, R² all match.
- Test fixture `tests/data/prostate.csv` is committed alongside the test.
- A rank-deficient design matrix raises `AssumptionError` from
  `FullRankDesign` (HARD assumption from P1.B), with a message naming the
  offending columns.
- `mc.predict(sol, new_data)` returns a `pd.Series` aligned with `new_data.index`.

### Task P1.B — Assumption framework + OLS-specific assumptions

**Workspace**:
- `model_crafter/assumptions/__init__.py`
- `model_crafter/assumptions/prerequisites.py`
- `model_crafter/assumptions/stability.py`
- `model_crafter/assumptions/classical.py`
- `tests/test_assumptions.py`
- `tests/test_assumptions_classical.py`

**Reads**: `DESIGN.md` (§4 in full, §8 Phase 1), `model_crafter/spec.py` interface
(import contract — depend only on the public types listed above)

**Depends on**: P0; coordinates with P1.A on `Solution` and `LinearSpec` shapes
(both work against the interface contract documented here)

**Public interface**:

```python
# model_crafter.assumptions
class Severity(Enum):
    HARD = "hard"
    SOFT = "soft"
    INFO = "info"

@dataclass(frozen=True)
class CheckResult:
    name: str
    severity: Severity
    passed: bool
    message: str
    statistic: float | None
    threshold: float | None
    suggestion: str | None

class Assumption(Protocol):
    name: str
    severity: Severity
    requires_solution: bool
    requires_cv: bool
    def describe(self) -> str: ...
    def check(self, spec, data, *, solution=None, cv=None) -> CheckResult: ...

@dataclass(frozen=True)
class AssumptionReport:
    results: tuple[CheckResult, ...]
    def by_severity(self) -> dict[Severity, tuple[CheckResult, ...]]: ...
    def passed(self) -> bool: ...
    def __repr__(self) -> str: ...

class AssumptionError(Exception): ...

def run_assumptions(
    spec, data, *,
    solution=None,
    cv=None,
    on_violation: str = "warn",
    suppress: tuple = (),
    classical_inference: bool = False,
) -> AssumptionReport: ...

def check_assumptions(
    spec, data, *,
    solution=None,
    classical_inference: bool = False,
) -> AssumptionReport: ...

# Concrete assumptions (HARD prerequisites)
FullRankDesign: Assumption

# Concrete assumptions (SOFT stability) — implementations may be stubbed
# in P1 if they need CV which arrives in P3; declare them but allow them
# to skip if cv= is None and requires_cv is True. The orchestration runs
# them when CV is available.
CoefficientStability: Assumption
PredictiveStability: Assumption

# Concrete assumptions (INFO classical, opt-in)
ResidualNormality: Assumption
Homoscedasticity: Assumption
Independence: Assumption
LowVIF: Assumption    # message includes ESL §3.4.1 regularize-don't-prune hint
```

**Acceptance**:
- Unit tests for each concrete assumption: synthetic data triggers PASS,
  synthetic data triggers FAIL, with statistic values verified against a
  reference (scipy or hand-derived).
- `run_assumptions` raises `AssumptionError` on HARD failure when
  `on_violation="raise"` (the default). Warns on SOFT failure. Silently
  records INFO results.
- `LowVIF` is INFO severity and its `suggestion` field reads (verbatim):
  `"High collinearity detected (max VIF = {x:.1f}). ESL §3.4.1 recommends
  ridge or lasso regularization rather than feature pruning. Consider
  penalty=mc.l2(...) or penalty=mc.l1(...)."`
- `classical_inference=False` (the default) suppresses INFO-level checks
  entirely; `classical_inference=True` runs them and includes them in the
  report at INFO severity.

### Task P1.INTEG — Phase 1 integration

**Workspace**: `tests/integration/test_phase1.py`, `model_crafter/__init__.py`
(re-exports only)

**Reads**: P1.A + P1.B workspaces

**Depends on**: P1.A, P1.B both green

**Acceptance**:
- All P1.A and P1.B acceptance criteria still pass.
- `mc.solve(spec, data)` end-to-end on the prostate dataset produces a
  `Solution` with an `assumptions` field that includes a passing
  `FullRankDesign` result.
- `mc.solve(spec, data, classical_inference=True)` on a heteroscedastic
  synthetic dataset produces an `AssumptionReport` containing a
  `Homoscedasticity` INFO-level result with the Breusch-Pagan statistic.
  Without `classical_inference=True`, classical checks are absent.
- Public API surface in `model_crafter/__init__.py` matches what's
  imported in the reference example in `DESIGN.md` §10 *for Phase 1 scope*
  (i.e., `mc.linear`, `mc.solve`, `mc.predict`, `mc.squared_error`,
  `mc.check_assumptions`, `mc.NoPenalty` — nothing else yet).

---

## Phase 2 — Ridge, lasso, elastic net (three parallel)

### Task P2.A — Penalty primitives

**Workspace**:
- `model_crafter/penalty.py` (extend — add `l1`, `l2`, `PenaltySum`)
- `tests/test_penalty.py`

**Reads**: P1.A's `Penalty` protocol

**Depends on**: P1.INTEG

**Public interface**:

```python
def l1(lam: float) -> Penalty: ...
def l2(lam: float) -> Penalty: ...

@dataclass(frozen=True)
class PenaltySum(Penalty):
    parts: tuple[Penalty, ...]
```

Each penalty declares `assumptions = (ComparableFeatureScales(...),)`
(SOFT). `Penalty.__add__` returns a flattened `PenaltySum`. `Penalty + Term`
raises `TypeError` with a message pointing at `features=` vs `penalty=`.

**Acceptance**:
- `l1(0.5) + l2(0.5)` produces a `PenaltySum` with two parts; iterating
  yields them in order.
- `ComparableFeatureScales` (declared via P1.B's framework) fires on a
  synthetic dataset with feature std ratio > 100.
- `value()` and `prox()` (for proximal methods) match hand-derived
  references to 1e-12.

### Task P2.B — Closed-form ridge solver

**Workspace**:
- `model_crafter/solve/ridge.py`
- additions to `model_crafter/solve/__init__.py` dispatch (coordinate
  carefully with P2.C — see "Dispatch ownership" below)
- `tests/test_ridge.py`

**Reads**: P1.A's `Solution`, `LinearSpec`, `_internal/linalg.py`; P2.A's
`l2` (against the interface contract; mock if P2.A not yet merged)

**Depends on**: P1.INTEG

**Public interface**: registers a solver for `(SquaredErrorLoss, L2Penalty)`
via the dispatch in `solve/__init__.py`. Does not export new symbols.

**Acceptance**:
- Closed-form ridge: $\hat\beta = (X^TX + \lambda I)^{-1} X^T y$, with
  intercept handled correctly (not penalized).
- Match `glmnet(alpha=0)` coefficients on the prostate dataset across a
  lambda path to 1e-6.

### Task P2.C — Coordinate descent for lasso and elastic net

**Workspace**:
- `model_crafter/solve/coordinate.py`
- additions to `model_crafter/solve/__init__.py` dispatch (coordinate
  with P2.B — see below)
- `model_crafter/validation/lambda_path.py` (just `lambda_path` + grid helpers)
- `tests/test_coordinate.py`
- `tests/test_lambda_path.py`

**Reads**: P1.A's `Solution`, `LinearSpec`; P2.A's `l1`, `l2`, `PenaltySum`

**Depends on**: P1.INTEG

**Public interface**:

```python
def lambda_path(spec: LinearSpec, data, n: int = 100,
                ratio: float = 1e-3) -> np.ndarray: ...

def log_grid(low: float, high: float, n: int) -> np.ndarray: ...
```

Registers solvers for `(SquaredErrorLoss, L1Penalty)`,
`(SquaredErrorLoss, PenaltySum[L1+L2])`.

**Acceptance**:
- Match `glmnet` coefficients across a lambda path for lasso and elastic
  net, on the prostate dataset, to 1e-6.
- Warm-started path fit (from large to small lambda) is measurably faster
  than from-scratch fits at each lambda. Benchmark in the test.

### Dispatch ownership (P2.B + P2.C coordination)

Both P2.B and P2.C add solver registrations to `solve/__init__.py`. To avoid
conflicts, the dispatch table lives in a dedicated registry module
`model_crafter/solve/_registry.py` (which P1.A creates as part of `solve/__init__.py`).
Each solver module **registers itself on import** via:

```python
# in solve/ridge.py
from model_crafter.solve._registry import register
register((SquaredErrorLoss, L2Penalty), solve_ridge_closed_form)
```

P2.B and P2.C only ever *append* to the registry. They never edit
`solve/__init__.py` directly. The integration agent verifies no
duplicate registrations.

### Task P2.INTEG — Phase 2 integration

**Workspace**: `tests/integration/test_phase2.py`, public API re-exports

**Acceptance**:
- All P2.{A,B,C} acceptance criteria pass.
- `mc.linear(..., loss=squared_error, penalty=l1(0.1)+l2(0.1))` solves
  end-to-end.
- `ComparableFeatureScales` warning fires on unscaled features by default.

---

## Phase 3 — Logistic, temporal CV, bootstrap, performance (four parallel)

### Task P3.A — Logistic loss, IRLS, logistic assumptions

**Workspace**:
- additions to `model_crafter/loss.py` (`logistic` + `LogisticLoss` class)
- `model_crafter/solve/irls.py`
- `model_crafter/assumptions/logistic.py` (extends P1.B's assumptions)
- `tests/test_logistic.py`
- `tests/test_irls.py`

**Reads**: P1.A's `Solution`/`Loss`/`Penalty`; P1.B's assumption framework;
P2.A's `l2`

**Depends on**: P2.INTEG

**Public interface**:

```python
logistic: Loss

# Re-exported from assumptions package
BinaryOrProportionTarget: Assumption
NoPerfectSeparation: Assumption
ClassBalance: Assumption
LinkAdequacy: Assumption  # INFO, opt-in
```

`logistic.assumptions` includes the four above (with `LinkAdequacy` at INFO
severity). Registers solvers for `(LogisticLoss, NoPenalty)` and
`(LogisticLoss, L2Penalty)`. Lasso-logistic goes to P2.C's coordinate
descent; the dispatch registration for `(LogisticLoss, L1Penalty)` is
P3.A's responsibility (calls into P2.C's solver with the right
pseudo-residual transform).

**Acceptance**:
- Reproduce statsmodels logistic regression coefficients to 1e-6 on a
  public credit dataset.
- Perfectly separable synthetic data raises `AssumptionError` from
  `NoPerfectSeparation` with a message recommending `penalty=mc.l2(...)`.
- `mc.predict(sol, data)` for logistic always returns probabilities in
  `[0, 1]` (verified by property test).

### Task P3.B — Temporal validation: splitters, CV, tune, nested CV

**Workspace**:
- `model_crafter/validation/__init__.py`
- `model_crafter/validation/splitters.py`
- `model_crafter/validation/cross_validate.py`
- `model_crafter/validation/tune.py`
- `model_crafter/validation/over_time.py`
- `model_crafter/assumptions/temporal.py` (extends P1.B's framework)
- `tests/test_splitters.py`
- `tests/test_cross_validate.py`
- `tests/test_tune.py`
- `tests/test_nested_cv.py`

**Reads**: P1.A's `Solution`/`LinearSpec`; P1.B's framework

**Depends on**: P2.INTEG (does NOT depend on P3.A; can be developed against
the squared-error loss as the test case)

**Public interface**:

```python
# Splitters
def time_split(df, time_col, ratios) -> tuple[DataFrame, ...]: ...
def expanding_window(time_col, n_folds, horizon, gap, min_train=None) -> Splitter: ...
def rolling_window(time_col, train_size, horizon, step, gap) -> Splitter: ...
def purged_kfold(time_col, n_folds, gap) -> Splitter: ...

# CV runners
@dataclass(frozen=True)
class CVResult:
    fold_results: tuple[dict, ...]
    solutions: tuple[Solution, ...]
    def summary(self) -> pd.DataFrame: ...

def cross_validate(spec, data, splitter, metrics, weights=None) -> CVResult: ...

@dataclass(frozen=True)
class TuneResult:
    best_param: Any
    cv_curve: pd.DataFrame
    solution: Solution

def tune(spec_fn, grid, data, splitter, metric, weights=None,
         rule=None) -> TuneResult: ...

@dataclass(frozen=True)
class NestedCVResult:
    outer_metric: pd.DataFrame
    best_params: tuple
    inner_curves: tuple

def nested_cv(spec_fn, grid, data, outer_splitter, inner_splitter,
              metric, weights=None) -> NestedCVResult: ...

# Selection rules
def one_se_rule(curve: pd.DataFrame, direction: str) -> Any: ...
def best_mean(curve: pd.DataFrame, direction: str) -> Any: ...

# Time-indexed metric runner (used by performance_over_time later)
def over_time(metric, sol, data, splitter) -> pd.Series: ...

# Temporal assumption
NoTemporalLeakage: Assumption
```

**Acceptance**:
- `expanding_window(time_col=..., gap="365D")` produces folds where every
  validation window starts at least 365 days after its training window ends.
- `NoTemporalLeakage` fires on a synthetic violation.
- `cross_validate` refuses to run when a temporal splitter is required
  (loss declares `label_horizon`) but `gap` is unset.
- `tune` returns a curve where the chosen param is the optimum under the
  chosen rule (best mean or 1-SE).
- `nested_cv` outer error is statistically larger (in expectation, verified
  on a simulation with a fixed seed) than `tune` inner error on the same
  data — the ESL §7.10.2 optimism bias is detectable.

### Task P3.C — Bootstrap

**Workspace**:
- `model_crafter/validation/bootstrap.py`
- additions to `model_crafter/solution.py` (`BootstrappedSolution` dataclass)
- `tests/test_bootstrap.py`

**Reads**: P1.A's `Solution`/`solve`; P3.B's splitters (for block bootstrap)

**Depends on**: P2.INTEG (does NOT depend on P3.A or P3.B for the core
implementation; the block-bootstrap variant depends on P3.B's splitters
but can ship with that variant stubbed if P3.B isn't merged yet)

**Public interface**:

```python
@dataclass(frozen=True)
class BootstrappedSolution:
    base: Solution
    coefficients_dist: pd.DataFrame    # n_boot × n_coef
    fit_state_dist: tuple              # per-resample fit state
    selection_frequency: pd.Series     # for lasso; fraction of resamples nonzero
    n_boot: int
    method: str

    def coefficient_ci(self, level: float = 0.95,
                       method: str = "percentile") -> pd.DataFrame: ...
    def prediction_ci(self, new_data, level: float = 0.95) -> pd.DataFrame: ...

def bootstrap(sol, data, n_boot=500, stratify=None,
              method="pairs", weights=None,
              splitter=None) -> BootstrappedSolution: ...
```

**Acceptance**:
- `bootstrap` with `method="pairs"` produces 95% percentile CIs that
  contain the OLS closed-form CIs to within 5% across a synthetic linear
  regression (n=500, n_boot=2000).
- `method="residual"` matches the pairs bootstrap to within 10% on the
  same problem.
- `selection_frequency` on a lasso fit with a known sparse ground truth
  identifies the true nonzeros with frequency > 0.9 and the true zeros
  with frequency < 0.3 (n=1000, sparsity=0.2, n_boot=500).

### Task P3.D — Metric primitives and PerformanceReport

**Workspace**:
- `model_crafter/metrics/__init__.py`
- `model_crafter/metrics/classification.py`
- `model_crafter/metrics/calibration.py`
- `model_crafter/metrics/stability.py`
- `model_crafter/metrics/rank.py`
- `model_crafter/performance/__init__.py`
- `model_crafter/performance/report.py`
- `tests/test_metrics.py`
- `tests/test_performance.py`

**Reads**: P1.A's `Solution`/`predict`

**Depends on**: P2.INTEG

**Public interface**:

```python
# Discrimination
def auc(sol, data, weights=None) -> AUCResult: ...
def gini(sol, data, weights=None) -> GiniResult: ...
def ks(sol, data, weights=None) -> KSResult: ...
def cohens_d(sol, data, weights=None) -> CohensDResult: ...
def delong_test(sol_a, sol_b, data, weights=None) -> DeLongResult: ...

# Calibration
def calibration_curve(sol, data, n_bins=10, weights=None) -> CalibrationCurve: ...
def brier_score(sol, data, weights=None) -> BrierResult: ...
def ece(sol, data, n_bins=10, weights=None) -> ECEResult: ...
def log_loss(sol, data, weights=None) -> LogLossResult: ...
def calibration_slope_intercept(sol, data, weights=None) -> CalibrationFit: ...

# Stability
def psi(reference, current, bins=10) -> PSIResult: ...

# Rank
def lift_table(sol, data, n_deciles=10, weights=None) -> LiftTable: ...
def cumulative_gains(sol, data, weights=None) -> GainsCurve: ...

# The big bundle
@dataclass(frozen=True)
class PerformanceReport:
    discrimination: DiscriminationReport
    calibration: CalibrationReport
    stability: StabilityReport | None
    distribution: DistributionReport
    lift_table: LiftTable
    cumulative_gains: GainsCurve
    n_obs: int
    n_events: int

def performance(sol, data, *, weights=None, reference=None) -> PerformanceReport: ...
```

Every result type has a rich `__repr__`. The `PerformanceReport.__repr__`
matches the layout shown in `DESIGN.md` §3.3.

**Acceptance**:
- `auc` matches scipy/scikit reference to 1e-9 on synthetic data.
- `ks` matches scipy `ks_2samp` to 1e-9.
- `psi` matches the hand-derived formula on a worked example to 1e-12.
- `delong_test` matches `pROC::roc.test` (R) to 1e-6 on a fixed seed.
- `calibration_slope_intercept` recovers slope=1, intercept=0 on
  synthetic perfectly calibrated data to 1e-6.
- `performance(sol, data)` produces a `PerformanceReport` whose
  sub-reports contain values matching individual primitive calls.
- All metrics with `weights=` support weighted variants matching a
  reference (statsmodels survey weights or hand-derived).

### Task P3.INTEG — Phase 3 integration

**Workspace**: `tests/integration/test_phase3.py`, public API re-exports

**Depends on**: P3.A, P3.B, P3.C, P3.D all green

**Acceptance**:
- End-to-end logistic regression with `mc.tune` + `mc.bootstrap` +
  `mc.performance` runs on the public credit dataset.
- `mc.solve(..., classical_inference=True)` produces an `AssumptionReport`
  including `LinkAdequacy` for logistic regression.
- Stability assumptions (`CoefficientStability`, `PredictiveStability`)
  from P1.B's framework now run automatically because CV is available;
  verify they fire on an unstable synthetic dataset.

---

## Phase 4 — Basis expansions, WoE, mc.binned (three parallel)

### Task P4.A — Standard basis expansions

**Workspace**:
- `model_crafter/terms/basis.py`
- `tests/test_basis.py`

**Reads**: P1.A's `Term`/`TermSum`

**Depends on**: P3.INTEG

**Public interface**:

```python
def ns(col: str, df: int, *, knots=None) -> Term: ...        # natural cubic spline
def bs(col: str, df: int, *, degree=3, knots=None) -> Term:... # B-spline
def poly(col: str, degree: int) -> Term: ...
def step(col: str, breakpoints: Sequence[float]) -> Term: ...
def smooth(col: str, df: int) -> Term: ...                   # alias for ns with smoothness penalty
def hinge(col: str, knot: float, direction: str) -> Term: ...# MARS-style
```

Each term declares `SupportContainsPredictData()` as a SOFT assumption that
fires at predict time when test data extends beyond training knot range.

**Acceptance**:
- `ns` matches R's `splines::ns()` to 1e-10.
- `bs` matches R's `splines::bs()` to 1e-10.
- `SupportContainsPredictData` warning names the fraction of extrapolated
  rows when test data exceeds training support.

### Task P4.B — WoE and binned terms + binning strategies

**Workspace**:
- `model_crafter/terms/woe.py`
- `model_crafter/assumptions/woe.py`
- `model_crafter/inspect.py` (extend — add `binning_table`)
- `tests/test_woe.py`
- `tests/test_binned.py`
- `tests/test_binning_strategies.py`

**Reads**: P1.A's `Term`; P1.B's assumption framework

**Depends on**: P3.INTEG

**Public interface**:

```python
def woe(col: str, bins) -> Term: ...        # WoE-encoded basis
def binned(col: str, bins) -> Term: ...     # bin-indicator basis (one coef per bin)

# Binning strategies (values, not classes — call sites use mc.monotonic())
def monotonic(min_bin_size: float = 0.05, max_bins: int = 20) -> Binning: ...
def tree_bins(max_leaves: int = 10, min_samples_leaf: float = 0.05) -> Binning: ...
def categorical(group_rare: float = 0.01) -> Binning: ...
def manual(edges) -> Binning: ...

def binning_table(sol) -> BinningTable: ...

# Assumptions
AtLeastOneEventPerBin: Assumption
MinimumBinSize: Assumption
MonotonicEventRate: Assumption
WoEMonotonicityPreserved: Assumption       # post-fit, SOFT
```

**Acceptance**:
- `mc.woe` with `mc.monotonic(min_bin_size=0.05)` produces bins where each
  bin has ≥ 5% of observations and event rates are monotonically
  ordered.
- `WoEMonotonicityPreserved` fires (warns) when the joint logistic
  regression coefficient on a WoE-encoded column is negative.
- `mc.binned` on the same data and bin definitions yields equivalent
  in-sample fit (up to identifiability) but different out-of-sample
  performance to `mc.woe` — verified on a CV comparison.

### Task P4.C — Interactions

**Workspace**:
- `model_crafter/terms/interact.py`
- `tests/test_interact.py`

**Reads**: P1.A's `Term`

**Depends on**: P3.INTEG

**Public interface**:

```python
def interact(*cols: str | Term) -> Term: ...   # main effects + interaction
def cross(*cols: str | Term) -> Term: ...      # interaction only
```

**Acceptance**:
- `interact(a, b)` expands to the columns of `a`, the columns of `b`, and
  their pairwise products.
- `cross(a, b)` expands to pairwise products only.
- Works with categorical (one-hot) and continuous terms; works with WoE
  terms (interaction of WoE-encoded values).

### Task P4.INTEG — Phase 4 integration

**Workspace**: `tests/integration/test_phase4.py`

**Acceptance**: the §10 north-star example in `DESIGN.md` (the part that
fits inside Phase 4 — WoE + logistic + walk-forward CV + bootstrap +
performance) runs end-to-end on a public dataset.

---

## Phase 5 — Advanced performance analysis (three parallel)

### Task P5.A — Temporal performance

**Workspace**:
- `model_crafter/performance/over_time.py`
- `tests/test_performance_over_time.py`

**Public interface**:

```python
@dataclass(frozen=True)
class TemporalPerformanceReport:
    summary: pd.DataFrame
    reports: tuple[PerformanceReport, ...]
    def plot(self): ...

def performance_over_time(sol, data, splitter, *,
                          reference=None, weights=None) -> TemporalPerformanceReport: ...
```

**Acceptance**: on a synthetic non-stationary panel, reproduces known AUC
drift across windows to 1e-4.

### Task P5.B — Segmented performance

**Workspace**:
- `model_crafter/performance/by_segment.py`
- `tests/test_performance_by_segment.py`

**Public interface**:

```python
@dataclass(frozen=True)
class SegmentedPerformanceReport:
    segments: dict[str, PerformanceReport]
    aggregate: PerformanceReport
    def __repr__(self) -> str: ...

def performance_by_segment(sol, data, by: str, *,
                           weights=None, reference=None) -> SegmentedPerformanceReport: ...
```

**Acceptance**: on a segmented logistic regression, produces one
`PerformanceReport` per segment plus an aggregate report.

### Task P5.C — Model comparison

**Workspace**:
- `model_crafter/performance/compare.py`
- `tests/test_compare.py`

**Public interface**:

```python
@dataclass(frozen=True)
class Comparison:
    reports: dict[str, PerformanceReport]
    delong_pvalues: pd.DataFrame
    def __repr__(self) -> str: ...

def compare(solutions: dict[str, Solution], data, *,
            weights=None) -> Comparison: ...
```

**Acceptance**: DeLong p-values match P3.D's `mc.delong_test` for
pairwise comparisons.

### Task P5.INTEG — Phase 5 integration

**Workspace**: `tests/integration/test_phase5.py`

**Acceptance**: the full §10 north-star example runs end-to-end.

---

## Phase 6 — Segmentation, diagnostics, inspection (serial integration)

### Task P6 — Segmentation, diagnostics, repr polish

Single agent. Implements `mc.segmented`, `mc.diagnostics`, `mc.hat_matrix`,
`mc.influence`, and `_repr_html_` on every value type for Jupyter rendering.

**Workspace**: `model_crafter/spec.py` (extend with `SegmentedSpec`),
`model_crafter/solve/segmented.py`, `model_crafter/inspect.py` (extend),
`tests/test_segmented.py`, `tests/test_diagnostics.py`,
`tests/test_repr_html.py`

**Acceptance**: see `DESIGN.md` §8 Phase 6.

---

## Coordination notes

1. **Interface contracts are pinned by this document, not by code.** If an
   agent needs to change a type signature listed here, they stop and flag
   the change for the human. Do not silently widen or narrow a signature
   other agents are coding against.

2. **The dispatch registry pattern** (P2.B + P2.C coordination) is the
   general approach for any "many writers, one table" situation. Each
   solver, term type, and assumption registers itself on import. The
   `__init__.py` of each subpackage triggers the registrations by
   importing the modules.

3. **When in doubt, mock.** If your task depends on something not yet
   merged, write your code against the interface contract and use mocks
   in tests. The integration agent verifies the mocks line up with the
   real implementations.

4. **Notes go in `notes/<task-id>.md`.** Two paragraphs maximum. What you
   built, what's tested, what's open. The human reads these to track
   progress; future agents read them to understand decisions.

5. **Do not "improve" the design.** If you see something that looks
   wrong in `DESIGN.md`, write a note in `notes/<task-id>.md` flagging it.
   Do not fix it in your task. Drift between agents kills the package.