# `model_crafter` — Design Document (v0)

> A Python package for credit risk modeling that feels like pen and paper.
> Models are values. Fitting is a function. The API reads like the equation.

---

## 1. Motivation

`scikit-learn` was built around a contract — `fit/predict/transform` on stateful
estimator objects, composed via `Pipeline` — that solves real engineering
problems (grid search, persistence, ecosystem interop) at the cost of an API
that does not look like math. For credit risk modeling, where the model
specification is the artifact you defend to regulators, the ceremony gets in
the way of clarity.

`model_crafter` is a craft project. The goal is an API where:

1. A model is a value, not a stateful object. Its output is always a
   probability (for classification) or a predicted value (for regression).
2. The arguments you pass *are* the equation: target, features, loss, penalty.
3. Composition uses `+` only where the math is genuinely additive (terms in a
   linear predictor; penalty terms in a regularized loss).
4. The credit risk vocabulary — WoE, KS, PSI, sample weights, segmentation,
   walk-forward validation — is first-class, not bolted on.
5. Every model carries its mathematical assumptions explicitly, and those
   assumptions are tested against the data at solve time.
6. Performance analysis is a first-class value (`PerformanceReport`), not
   a scatter of metric calls — discrimination, calibration, stability,
   and distribution diagnostics in one bundled output.

Non-goals: sklearn API compatibility, exhaustive ESL coverage, deep learning,
distributed training, AutoML, scorecard / points-table generation (this is
a presentation layer downstream of the model).

---

## 2. Core abstractions

### 2.1 The three verbs

The entire package is built around three verbs applied to values:

```python
spec = mc.linear(...)              # build a model specification (a value)
sol  = mc.solve(spec, data=df)     # produce a solution (also a value)
yhat = mc.predict(sol, new_df)     # apply a solution to new data
```

`spec` and `sol` are **immutable, hashable, picklable values**. There is no
`fit` method, no fitted/unfitted duality, no underscore-suffixed learned
attributes. A `Spec` describes a model in the mathematical sense; a `Solution`
is what falls out when data is applied.

### 2.2 Terms and the `features=` argument

Features are `Term` objects. Strings auto-promote to `RawTerm`. A list of
features and a `+`-composed sum of terms are interchangeable inputs:

```python
features = ["income", "age", "tenure"]                           # list form
features = mc.ns("age", df=5) + mc.poly("tenure", 3) + "income"  # sum form
features = mc.raw("income")                                       # single term
```

All three normalize to `tuple[Term, ...]` internally. The `+` operator is
provided on `Term` for the case where the terms are *interesting* (basis
expansions, interactions, WoE encodings); the list form is preferred for plain
column references.

### 2.3 Loss and penalty

`loss=` and `penalty=` are separate keyword arguments. Penalties compose with
`+` because a penalty mathematically is a sum of penalty terms:

```python
penalty = mc.l1(0.05) + mc.l2(0.05)   # elastic net
```

`+` between a `Term` and a `Penalty` raises with a clear error message. `+`
between two `Penalty` objects produces a `PenaltySum`. There is no operator
overloading anywhere else in the public API.

### 2.4 What `solve` does

`solve(spec, data, *, weights=None, method=None)` dispatches on
`(type(spec), type(spec.loss), type(spec.penalty))` and selects the
mathematically appropriate solver:

| Loss              | Penalty       | Solver                        |
| ----------------- | ------------- | ----------------------------- |
| `squared_error`   | none / `l2`   | Normal equations (closed form)|
| `squared_error`   | `l1` / enet   | Coordinate descent            |
| `logistic`        | none / `l2`   | IRLS                          |
| `logistic`        | `l1` / enet   | Proximal Newton / CD          |

`method=` is the escape hatch for advanced users; default is `None` (let the
math pick). Sample weights are supported on every solver.

### 2.5 What `mc.linear` covers

The `mc.linear` constructor is broader than it looks. ESL Ch 5 and Ch 9.1
frame Generalized Additive Models as
$f(X) = \alpha + \sum_j f_j(X_j)$ where each $f_j$ is a smooth function of
a single variable. That's a linear predictor on a basis expansion — which
is exactly `mc.linear` with smoother basis terms (`mc.ns`, `mc.bs`,
`mc.smooth`). Similarly, MARS (ESL §9.4) is a linear predictor on a
hinge-function basis (`mc.hinge`).

Concretely, `mc.linear` covers:

- Ordinary least squares, ridge, lasso, elastic net (Ch 3)
- Logistic regression and its regularized variants (Ch 4)
- GAMs via `mc.ns`, `mc.bs`, `mc.smooth` (Ch 5, Ch 9.1)
- WoE-encoded and bin-indicator credit risk models
- MARS-style hinge models via `mc.hinge` (Ch 9.4)

Genuinely separate model families — trees (Ch 9.2), boosting (Ch 10),
random forests (Ch 15) — get their own constructors (`mc.tree`,
`mc.boost`, `mc.forest`), because they aren't linear predictors on a
basis. The line is precisely the one ESL draws.

---

## 3. Credit risk specifics

### 3.1 Weight of Evidence (and the ESL-honest alternative)

WoE binning is structurally a step-function basis expansion. It belongs in the
`features=` list, not as a preprocessing step:

```python
features = (
    mc.woe("income",  bins=mc.monotonic(min_bin_size=0.05))
    + mc.woe("age",   bins=mc.tree_bins(max_leaves=8))
    + mc.woe("region", bins=mc.categorical(group_rare=0.01))
    + "tenure"
)
```

`mc.woe(col, bins=...)` returns a `WoETerm`. At solve time it learns bin
edges and WoE values from `data` and stores them in the solution. The
`Solution` exposes the binning table for inspection and report generation.

**A note on what `mc.woe` is doing.** WoE-encode-then-regress is a
*constrained* approach: it fixes each bin's contribution to the univariate
log-odds and lets the joint logistic regression learn one global coefficient
that scales them. ESL §5.2's framing of bin-indicator basis expansions is
more general — it would put one coefficient per bin in the joint fit, letting
the multivariate effect differ from the marginal one. The package supports
both:

```python
mc.woe("income",  bins=mc.monotonic())     # WoE-encoded, one coefficient
mc.binned("income", bins=mc.monotonic())   # bin-indicator basis, one coefficient per bin
```

`mc.woe` is the credit-industry default for good reasons (stability,
interpretability, regulator legibility) and gets a stability assumption
(`WoEMonotonicityPreserved`, §4.3) that fires when the joint fit fights the
univariate encoding. `mc.binned` is the ESL-honest alternative for when you
want the joint model to discover the bin coefficients itself, typically
paired with `penalty=mc.l2(...)` to handle the resulting high-dimensional
design matrix.

Binning strategies are themselves values: `mc.monotonic(...)`,
`mc.tree_bins(...)`, `mc.categorical(...)`, `mc.manual(edges=[...])`.

### 3.2 Temporal validation and the bootstrap

Credit risk data is non-stationary and labels mature over time, so random
k-fold cross-validation is almost always wrong. `model_crafter` ships with
temporal splitters and a CV runner that work over a time column. ESL §7.10
is the reference for what these tools do and don't do.

```python
# Single chronological split (preserves order)
train, valid, test = mc.time_split(df, time_col="origination_dt",
                                   ratios=(0.7, 0.15, 0.15))

# Expanding-window walk-forward CV
splitter = mc.expanding_window(
    time_col   = "origination_dt",
    n_folds    = 5,
    horizon    = "90D",         # length of each validation window
    gap        = "365D",        # buffer for label maturation
    min_train  = "730D",
)

# Rolling fixed-size window (for non-stationary portfolios)
splitter = mc.rolling_window(
    time_col   = "origination_dt",
    train_size = "365D",
    horizon    = "90D",
    step       = "90D",
    gap        = "365D",
)

# Purged k-fold with temporal gaps between folds (López de Prado)
splitter = mc.purged_kfold(time_col="origination_dt", n_folds=5, gap="30D")
```

#### CV for assessment vs CV for tuning (ESL §7.10.2–3)

ESL is emphatic that using the same CV both to tune hyperparameters and to
report performance is optimistic — the chosen $\lambda$ is itself a function
of the held-out folds, so the held-out error underestimates true error.
`model_crafter` makes this distinction in the API rather than leaving it as
a docstring footnote:

```python
# CV for ASSESSMENT — single spec, no tuning. Report-ready performance.
cv = mc.cross_validate(
    spec,                                # fully specified, no tuning
    data     = df,
    splitter = splitter,
    metrics  = [mc.auc, mc.ks, mc.psi],
)

# CV for TUNING — pick a hyperparameter, return the chosen value and a refit.
tuned = mc.tune(
    spec_fn  = lambda lam: spec.with_(penalty=mc.l2(lam)),
    grid     = mc.log_grid(1e-4, 1e2, n=30),
    data     = df,
    splitter = splitter,
    metric   = mc.auc,            # maximize
)
tuned.best_param         # the chosen lambda
tuned.cv_curve           # metric vs hyperparameter, with CV SD
tuned.solution           # spec refit on the full data at best_param

# CV for tuning AND assessment — nested CV. ESL §7.10.2's recommendation.
nested = mc.nested_cv(
    spec_fn        = lambda lam: spec.with_(penalty=mc.l2(lam)),
    grid           = mc.log_grid(1e-4, 1e2, n=30),
    data           = df,
    outer_splitter = splitter,
    inner_splitter = mc.expanding_window(time_col="origination_dt", n_folds=3,
                                         horizon="90D", gap="365D"),
    metric         = mc.auc,
)
nested.outer_metric      # honest held-out performance
nested.best_params       # one chosen lambda per outer fold (for stability)
```

The `gap` argument matters: a loan originated on day `t` does not have a
default-at-12-months label until day `t + 365`, so a fold whose training
data ends at `t` must not include validation observations originating
before `t + 365`. Splitters enforce this; the CV runner refuses to run if
`gap` is unset and the spec's loss declares a `label_horizon` attribute.

Splitters are also reusable for stability diagnostics (PSI over time, KS
over time): `mc.over_time(metric, sol, data, splitter)` runs a metric
across the splitter's windows and returns a time-indexed series.

#### The bootstrap (ESL §7.11, §8.2)

For models without closed-form standard errors — lasso, segmented models,
splines after selection — the bootstrap is the recommended tool. It's a
first-class operation that takes a solution and returns a resampled
solution with empirical distributions on every coefficient:

```python
sol_bs = mc.bootstrap(
    sol,
    data    = df,
    n_boot  = 500,
    stratify = "default_12m",     # preserves class balance per resample
    method  = "pairs",            # or "residual" for OLS
    weights = "sample_weight",
)

sol_bs.coefficient_ci(level=0.95)         # percentile or BCa CIs
sol_bs.coefficient_distribution           # full empirical distribution
sol_bs.selection_frequency                # for lasso: fraction of resamples
                                          # where each coefficient was nonzero
sol_bs.prediction_ci(new_data, level=0.95)
```

`mc.bootstrap` respects temporal structure when the spec carries a time
column: pass `splitter=` to use block bootstrap or stationary bootstrap
for time-dependent data.

`selection_frequency` matters specifically for lasso in credit risk:
ESL §3.4.3 notes lasso's selections are unstable under collinearity, and
bootstrap selection frequency is the standard diagnostic. A coefficient
selected in 95% of resamples is meaningfully different from one selected
in 55%, even if both have nonzero point estimates.

### 3.3 Performance analysis

A fitted model is a `Solution`; a model's performance on a dataset is a
`PerformanceReport`. The package treats performance with the same
structural status as assumptions: a named operation returns a value with
a rich `__repr__`, and individual metrics are primitives accessible
underneath. ESL §7 in spirit — predictive performance on held-out data
is the central object, not an afterthought.

```python
perf = mc.performance(sol, data=test, weights="sample_weight")
print(perf)
# PerformanceReport
# n=42,193  events=1,847 (4.4%)
#
# Discrimination
#   AUC                0.8123   (95% CI: 0.8050 – 0.8197, DeLong)
#   Gini               0.6246
#   KS                 0.4781   (at score 0.0612)
#   Cohen's d          1.243
#
# Calibration
#   Brier              0.0392
#   ECE  (10 bins)     0.0041
#   Log-loss           0.1564
#   Slope / Intercept  0.987 / 0.012  (logit regression of y on linear pred)
#
# Stability
#   PSI vs reference   0.024   (low; reference: train set)
#
# Distribution
#   Mean / Median p̂   0.0441 / 0.0218
#   Score range        [0.0003, 0.8721]

perf.discrimination     # AUC, Gini, KS, Cohen's d as a sub-report
perf.calibration        # calibration curve, Brier, ECE, log-loss, slope/intercept
perf.stability          # PSI vs reference, if a reference was provided
perf.distribution       # score histogram, mean/median, range
perf.lift_table         # decile lift, captured response, KS by decile
perf.cumulative_gains   # cumulative gains curve points

perf.calibration.curve  # the (predicted, observed) tuples for plotting
perf.calibration.plot() # matplotlib helper, lazy import
```

The output is **always a probability**: every metric assumes `sol.predict()`
returns calibrated-or-not probabilities in $[0, 1]$, and the
`PerformanceReport` makes calibration quality visible so the modeler can
see whether the probability interpretation is supported.

#### Temporal performance

```python
perf_t = mc.performance_over_time(
    sol,
    data     = panel,
    splitter = mc.rolling_window(time_col="origination_dt",
                                 train_size="365D", horizon="90D",
                                 step="30D", gap="365D"),
    reference = train,           # for PSI drift
)
perf_t.summary          # time-indexed DataFrame: AUC, KS, PSI, Brier per window
perf_t.plot()
```

This is the deployment-monitoring view: how does discrimination, calibration,
and score-distribution stability evolve as the population shifts? PSI drift
above the conventional 0.25 threshold is flagged in the report.

#### Model comparison

```python
cmp = mc.compare(
    {"baseline": sol_v1, "challenger": sol_v2},
    data    = test,
    weights = "sample_weight",
)
print(cmp)
# Comparison on test (n=42,193)
#
#                       baseline    challenger    Δ          p-value
#   AUC                 0.8123      0.8217        +0.0094    0.003 (DeLong)
#   KS                  0.4781      0.4892        +0.0111    —
#   Brier               0.0392      0.0388        −0.0004    —
#   Log-loss            0.1564      0.1552        −0.0012    —
#   PSI vs train        0.024       0.031         +0.007     —

cmp.delong              # AUC comparison p-values, pairwise
cmp.calibration         # calibration curves overlaid
```

DeLong's test is the standard tool for comparing AUC between two models on
the same held-out set; it's a held-out predictive comparison, which fits the
ESL §7 framing rather than the older inferential-on-coefficients tradition.

#### Individual primitives

Every component of a `PerformanceReport` is also callable directly as a
primitive — useful for custom diagnostics and for the assumption
framework's stability checks:

- `mc.auc`, `mc.gini`, `mc.ks`, `mc.cohens_d`, `mc.delong_test`
- `mc.calibration_curve`, `mc.brier_score`, `mc.ece`, `mc.log_loss`
- `mc.psi(reference, current, bins=...)`
- `mc.lift_table`, `mc.cumulative_gains`

All primitives accept `weights=` and return rich-`__repr__` value objects,
not bare floats. The package does not provide post-hoc calibration
transforms — see §4 and §3.1; calibration is observed, not corrected.

### 3.4 Segmentation

Segmented models are a `SegmentedSpec` that takes a base spec and a segment
column. Solving fits one solution per segment; predict routes by segment:

```python
spec = mc.segmented(
    by    = "product",
    base  = mc.linear(target="default_90d", features=[...], loss=mc.logistic),
)
sol = mc.solve(spec, data=df)
sol.segments               # dict[str, Solution]
sol["installments"]        # the per-segment solution
mc.performance(sol, data=test)   # aggregate report
mc.performance_by_segment(sol, data=test)   # per-segment reports
```

### 3.5 Sample weights and reject inference

`weights=` is accepted by every `solve`, every metric, every performance
call. A `ReweightedSpec` wraps a base spec with a weighting strategy for
reject inference (Heckman, parcelling, fuzzy augmentation) — out of scope
for v0 but the abstraction (`weights=` everywhere) keeps the door open.

---

## 4. Assumptions: explicit, declared, tested

Every model in `model_crafter` has mathematical assumptions, but the package
takes an **ESL-aligned view** of what those assumptions are and how to test
them. ESL §7 argues that held-out predictive performance, cross-validated
stability, and the bootstrap are the primary diagnostics for model adequacy
— not classical in-sample tests like Shapiro-Wilk or Breusch-Pagan, which
target inference on coefficients rather than predictive validity.

Accordingly, the assumption framework has three tiers:

1. **Prerequisites (HARD).** Conditions the math literally requires. Rank
   non-deficiency, no perfect separation, non-empty WoE bins, no temporal
   leakage. Violations raise by default; the model is invalid otherwise.
2. **Stability diagnostics (SOFT, default).** Held-out and resampling-based
   checks: coefficient stability across CV folds, predictive performance
   variance, calibration drift, score-distribution stability (PSI between
   folds), extrapolation rate at predict time. These are ESL §7's tools
   applied as automatic diagnostics.
3. **Classical inference (OPT-IN).** Shapiro-Wilk, Breusch-Pagan, Durbin-
   Watson, VIF, and friends. Used when coefficient inference matters —
   typically for regulatory model documentation — but not run by default,
   because they reflect a different question than "does this model
   predict well on held-out data."

### 4.1 The `Assumption` protocol

```python
class Severity(Enum):
    HARD = "hard"    # prerequisite for the math; raise on violation
    SOFT = "soft"    # stability diagnostic; warn on violation
    INFO = "info"    # classical inference; report only, never warn

class CheckResult:
    severity: Severity
    passed: bool
    message: str
    statistic: float | None
    threshold: float | None
    suggestion: str | None      # e.g., "consider penalty=mc.l2(...)"

class Assumption(Protocol):
    name: str
    severity: Severity
    requires_solution: bool     # post-fit checks set this True
    requires_cv: bool           # stability checks set this True
    def describe(self) -> str: ...
    def check(self, spec, data, *, solution=None, cv=None) -> CheckResult: ...
```

Each `Loss`, `Penalty`, `Term`, and `Solver` declares an `assumptions` tuple
of prerequisites and stability diagnostics. Classical inferential tests are
**not** declared by losses — they're requested explicitly:

```python
class SquaredErrorLoss:
    assumptions = (
        FullRankDesign(),                    # HARD prerequisite
        CoefficientStability(cv_cv_max=0.2), # SOFT, needs CV
        PredictiveStability(metric_cv_max=0.1),
    )

class LogisticLoss:
    assumptions = (
        BinaryOrProportionTarget(),          # HARD
        FullRankDesign(),                    # HARD
        NoPerfectSeparation(),               # HARD, post-fit
        ClassBalance(min_minority=0.01),     # SOFT (data sanity)
        CoefficientStability(cv_cv_max=0.2),
        PredictiveStability(metric_cv_max=0.1),
    )

class WoETerm:
    assumptions = (
        AtLeastOneEventPerBin(),             # HARD
        MinimumBinSize(),                    # HARD
        MonotonicEventRate(when="monotonic=True"),  # HARD
        WoEMonotonicityPreserved(),          # SOFT, post-fit
    )

class L2Penalty:
    assumptions = (
        ComparableFeatureScales(std_ratio_max=100),  # SOFT
    )

class NaturalSpline:
    assumptions = (
        SupportContainsPredictData(),        # SOFT, at predict time
    )
```

`WoEMonotonicityPreserved` is the post-fit check that the joint-fitted
coefficient on a WoE-encoded column has the expected sign (positive — the
multivariate relationship matches the marginal one used to construct the
WoE values). When it fails, it signals that the WoE encoding's univariate
assumption is being violated by the joint model.

### 4.2 How assumptions run

```python
sol = mc.solve(spec, data=df)
# Internally:
#   1. Collect HARD + SOFT assumptions from spec.loss, spec.penalty, every
#      term, and the chosen solver. INFO-level checks are skipped unless
#      explicitly requested.
#   2. Run prerequisite checks (HARD) against data and spec before solving.
#   3. Solve.
#   4. Run post-fit HARD checks (separation diagnostics). Raise on failure.
#   5. Run SOFT stability checks. The ones that require CV are run via an
#      internal small-k CV (default k=5). Warn on failure.
#   6. All results attach to sol.assumptions.
```

Per-call configuration:

```python
sol = mc.solve(spec, data=df, on_violation="raise"|"warn"|"ignore")
sol = mc.solve(spec, data=df, suppress=[mc.assumptions.PredictiveStability])

# Stability checks need a splitter; default is a small internal k-fold.
# For credit risk, pass a temporal splitter so stability is measured the
# way the model will actually be used:
sol = mc.solve(spec, data=df,
               stability_splitter=mc.expanding_window(
                   time_col="origination_dt", n_folds=5, gap="365D"))

# Classical inference tests are opt-in:
sol = mc.solve(spec, data=df, classical_inference=True)
# Now sol.assumptions includes Normality, Homoscedasticity, DurbinWatson,
# VIF, etc. — all at INFO severity. Output goes to the AssumptionReport
# but does not warn or raise.

# Standalone:
report = mc.check_assumptions(spec, data, classical_inference=True)
```

`sol.assumptions` is an `AssumptionReport` value: a list of `CheckResult`s
grouped by severity, with a rich `__repr__` showing pass/fail status, the
test statistic, and any suggestion attached (e.g., "high collinearity
detected; consider `penalty=mc.l2(...)` — see ESL §3.4.1").

### 4.3 The catalogue (v0)

**Prerequisites (HARD).** Run by default; raise on violation.

| Assumption | Declared by | Check |
| --- | --- | --- |
| `FullRankDesign` | every linear loss | rank of design matrix equals column count |
| `BinaryOrProportionTarget` | `LogisticLoss` | dtype + value range |
| `NoPerfectSeparation` | `LogisticLoss` (post-fit) | coefficient magnitude + convergence |
| `AtLeastOneEventPerBin` | `WoETerm` | event count per bin |
| `MinimumBinSize` | `WoETerm` | min(bin_count) / n |
| `MonotonicEventRate` | `WoETerm` (when `monotonic=True`) | sign-consistent ER across ordered bins |
| `NoTemporalLeakage` | `cross_validate` | every fold's train_end + gap ≤ valid_start |
| `OverlappingSupport` | `PSI` | non-empty intersection of reference and current bins |

**Stability diagnostics (SOFT).** Run by default; warn on violation. ESL §7
in spirit.

| Assumption | Declared by | Check |
| --- | --- | --- |
| `CoefficientStability` | every linear loss | max(SD/|mean|) across CV folds ≤ threshold |
| `PredictiveStability` | every linear loss | CV metric SD relative to mean ≤ threshold |
| `ClassBalance` | `LogisticLoss` | minority fraction ≥ threshold |
| `WoEMonotonicityPreserved` | `WoETerm` (post-fit) | joint coefficient on WoE column is positive |
| `ComparableFeatureScales` | `L1Penalty`, `L2Penalty` | ratio of max to min column std ≤ threshold |
| `SupportContainsPredictData` | basis terms (at predict) | fraction of test points outside training knot range |

**Classical inference (INFO).** Opt-in via `classical_inference=True`.
Reported, never warns or raises. For regulatory model documentation.

| Assumption | Declared by | Check |
| --- | --- | --- |
| `ResidualNormality` | `SquaredErrorLoss` | Shapiro-Wilk (n < 5000) / Anderson-Darling |
| `Homoscedasticity` | `SquaredErrorLoss` | Breusch-Pagan |
| `Independence` | `SquaredErrorLoss` | Durbin-Watson |
| `LowVIF` | every linear loss | max VIF across features (with `penalty=...` hint) |
| `LinkAdequacy` | `LogisticLoss` | Hosmer-Lemeshow / link test |

Importantly, `LowVIF` is INFO not SOFT. ESL §3.4.1 is explicit that ridge,
lasso, or elastic net is the right response to multicollinearity. When VIF
is reported and high, the `suggestion` field on the result reads:
*"High collinearity detected (max VIF = X). ESL §3.4.1 recommends ridge or
lasso regularization rather than feature pruning. Consider
`penalty=mc.l2(...)` or `penalty=mc.l1(...)`."*

### 4.4 Why this framing matters

The split between prerequisites, stability diagnostics, and classical
inference is not cosmetic — it reflects ESL's central methodological claim
that **predictive validity on held-out data is the gold standard, and
in-sample distributional tests are a different (often older) question.**

Three reasons this lands in the design doc:

1. **The package's identity rests on it.** "Pen and paper modeling" in
   2026 means ESL's notebook, not Cox and Hinkley's. The diagnostics ship
   accordingly.
2. **It changes the implementation contract.** PRs that add a model
   component without declaring its prerequisites *and* its stability
   diagnostics are incomplete by design. Classical-inference checks are
   opt-in additions, not requirements.
3. **It's a credit-risk-specific advantage.** Regulators ask "what did
   you check?". A built-in `AssumptionReport` that reports prerequisites,
   held-out stability, *and* classical inference (when requested for
   model documentation) is a real artifact. The three-tier structure
   matches how model risk committees actually reason: math validity,
   predictive validity, statistical-inference validity.

---

## 5. Solution inspection

`Solution.__repr__` prints a regression-table-style summary: equation, fitted
coefficients with standard errors (where computable), loss value, penalty
value, and key metrics on training data.

Inspection functions, all of which take a `Solution`:

```python
mc.coefficients(sol)       # named series with SE, z, p where available
mc.diagnostics(sol)        # residuals, leverage, Cook's distance
mc.hat_matrix(sol)         # closed-form linear models only
mc.influence(sol)
mc.binning_table(sol)      # WoE bins, counts, event rates, IV
```

The `Solution` always outputs probabilities via `mc.predict(sol, data)`.
The package does not provide scorecard / points-table conversion in v0 —
that's a presentation layer downstream of the model, and conflating the
two is the same category mistake as bundling calibration as a transform.
A scorecard can be built from a fitted `Solution` and a `BinningTable`
in a handful of lines outside the package when needed.

---

## 6. Type system

### 6.1 Core protocols

```python
class Term(Protocol):
    name: str
    def expand(self, data: pd.DataFrame, *, fit_state: Mapping | None) -> ExpandedTerm: ...
    def __add__(self, other) -> "TermSum": ...

class Loss(Protocol):
    def value(self, y, eta, weights) -> float: ...
    def gradient(self, y, eta, weights) -> np.ndarray: ...
    def hessian(self, y, eta, weights) -> np.ndarray | LinearOperator: ...

class Penalty(Protocol):
    def value(self, beta) -> float: ...
    def prox(self, beta, step) -> np.ndarray: ...   # for proximal methods
    def __add__(self, other) -> "PenaltySum": ...
```

### 6.2 Core dataclasses

```python
@dataclass(frozen=True)
class LinearSpec:
    target: str
    features: tuple[Term, ...]
    loss: Loss
    penalty: Penalty = NoPenalty()
    intercept: bool = True

@dataclass(frozen=True)
class Solution:
    spec: LinearSpec
    coefficients: pd.Series       # indexed by expanded column name
    coefficient_se: pd.Series | None
    fit_state: Mapping            # per-term learned state (e.g., WoE bins)
    design_columns: tuple[str, ...]
    loss_value: float
    penalty_value: float
    n_obs: int
    converged: bool
    solver_info: Mapping
```

All dataclasses are `frozen=True`, `slots=True` where supported. No mutation
after construction.

---

## 7. File layout

```
model_crafter/
├── __init__.py              # public API re-exports
├── spec.py                  # LinearSpec, SegmentedSpec, dataclasses
├── solution.py              # Solution, BootstrappedSolution
├── terms/
│   ├── __init__.py
│   ├── base.py              # Term, TermSum, RawTerm, _promote
│   ├── basis.py             # ns, bs, poly, step, smooth, hinge
│   ├── woe.py               # WoETerm, BinnedTerm, binning strategies
│   └── interact.py          # interact, cross
├── loss.py                  # squared_error, logistic, huber
├── penalty.py               # l1, l2, NoPenalty, PenaltySum
├── solve/
│   ├── __init__.py          # dispatch entry point
│   ├── ols.py               # normal equations
│   ├── ridge.py             # closed-form ridge
│   ├── coordinate.py        # lasso / elastic net CD
│   └── irls.py              # logistic IRLS
├── assumptions/
│   ├── __init__.py          # Assumption protocol, CheckResult, AssumptionReport
│   ├── prerequisites.py     # FullRankDesign, NoPerfectSeparation, etc. (HARD)
│   ├── stability.py         # CoefficientStability, PredictiveStability (SOFT)
│   ├── woe.py               # WoE-specific checks, including monotonicity-preserved
│   ├── temporal.py          # NoTemporalLeakage, OverlappingSupport
│   └── classical.py         # opt-in: Normality, Homoscedasticity, VIF, etc. (INFO)
├── validation/
│   ├── __init__.py
│   ├── splitters.py         # time_split, expanding_window, rolling_window, purged_kfold
│   ├── cross_validate.py    # cross_validate (assessment-only)
│   ├── tune.py              # tune, nested_cv
│   ├── bootstrap.py         # mc.bootstrap, BootstrappedSolution
│   └── over_time.py         # over_time helper for stability diagnostics
├── metrics/
│   ├── __init__.py
│   ├── classification.py    # auc, gini, ks, cohens_d, delong_test
│   ├── calibration.py       # brier, ece, calibration_curve, log_loss
│   ├── stability.py         # psi
│   └── rank.py              # lift_table, cumulative_gains
├── performance/
│   ├── __init__.py          # PerformanceReport, mc.performance, mc.compare
│   ├── report.py            # PerformanceReport dataclass, __repr__
│   ├── over_time.py         # mc.performance_over_time
│   ├── by_segment.py        # mc.performance_by_segment
│   └── compare.py           # mc.compare, Comparison value, DeLong
├── inspect.py               # coefficients, diagnostics, hat_matrix, binning_table
└── _internal/
    ├── design.py            # design matrix construction from terms
    ├── linalg.py            # numerical helpers, conditioning checks
    └── validation.py        # input validation, schema checks
```

Public API is whatever is re-exported from `model_crafter/__init__.py`. The
`_internal` package is private and subject to change.

---

## 8. Phased build plan

### Phase 0 — Scaffolding

- Project structure with `pyproject.toml`, `ruff`, `pyright`, `pytest`.
- Python 3.11+, numpy, scipy, pandas as required deps. No sklearn dependency.
- CI: lint, type-check, test, coverage on push.
- `DESIGN.md` (this document) committed at the root.

**Acceptance:** `pip install -e .` works; `pytest` runs an empty test suite
green; `ruff check` and `pyright` pass on an empty package.

### Phase 1 — Terms, specs, OLS, assumptions framework

- `Term`, `TermSum`, `RawTerm`, `_promote`, `_normalize_features`.
- `LinearSpec` dataclass; `mc.linear(...)` constructor.
- `squared_error` loss, `NoPenalty`.
- `solve` dispatch with the OLS (normal equations) path.
- `Solution` dataclass with coefficients + SE for OLS.
- `mc.predict(sol, new_data)`.
- **Assumption framework**: `Assumption` protocol with HARD / SOFT / INFO
  severity, `CheckResult`, `AssumptionReport`, `on_violation` plumbing,
  `classical_inference=` flag, `suppress=` list, `mc.check_assumptions`.
- OLS prerequisites: `FullRankDesign` (HARD).
- OLS classical-inference checks (opt-in): `ResidualNormality`,
  `Homoscedasticity`, `Independence`, `LowVIF` (with the ESL §3.4.1
  regularization suggestion in the message).

**Acceptance:**
1. Reproduce the ESL §3.2 prostate cancer OLS coefficients to 1e-8 against
   R's `lm()`. Coefficients, SEs, residual standard error, R² all match.
2. A rank-deficient design matrix raises `AssumptionError` from
   `FullRankDesign`. The error message names the offending columns.
3. `mc.solve(spec, data, classical_inference=True)` on a heteroscedastic
   synthetic dataset produces an `AssumptionReport` containing a
   `Homoscedasticity` INFO-level result with the Breusch-Pagan statistic.
   Without `classical_inference=True`, the report does not include
   classical checks.

### Phase 2 — Ridge, lasso, elastic net

- `l1`, `l2`, `PenaltySum`.
- Closed-form ridge solver.
- Coordinate descent solver for lasso and elastic net, with covariance
  updates and warm starts.
- Lambda path utilities: `mc.lambda_path(spec, data, n=100)`.
- `ComparableFeatureScales` (SOFT) declared by `l1` and `l2`.

**Acceptance:** Match `glmnet` coefficients on the prostate dataset across a
lambda path, both lasso and elastic net, to 1e-6. Warm-started path is at
least as fast as a from-scratch fit at each lambda. Unscaled features trigger
the `ComparableFeatureScales` warning by default.

### Phase 3 — Logistic regression, temporal CV, bootstrap, performance report

- `logistic` loss; IRLS solver; ridge-logistic via penalized IRLS.
- Logistic prerequisites: `BinaryOrProportionTarget` (HARD),
  `NoPerfectSeparation` (HARD, post-fit), `ClassBalance` (SOFT).
- Logistic classical-inference (opt-in): `LinkAdequacy` (Hosmer-Lemeshow).
- Metric primitives: `mc.auc`, `mc.gini`, `mc.ks`, `mc.cohens_d`,
  `mc.brier_score`, `mc.calibration_curve`, `mc.ece`, `mc.log_loss`,
  `mc.psi`. All accept `weights=`.
- **`mc.performance(sol, data, weights=, reference=)`** returns a
  `PerformanceReport` value bundling discrimination, calibration,
  stability, and distribution sub-reports with a rich `__repr__`.
- `Solution.__repr__` prints the regression table including the
  `AssumptionReport` summary.
- **Temporal validation**: `mc.time_split`, `mc.expanding_window`,
  `mc.rolling_window`, `mc.purged_kfold`, `mc.cross_validate`,
  `mc.over_time`.
- **Tuning vs assessment split**: `mc.tune`, `mc.nested_cv`. `cross_validate`
  is for assessment only and rejects multiple hyperparameter values.
- **Bootstrap**: `mc.bootstrap` for pairs and residual bootstrap, with
  `stratify=`, `weights=`, and percentile + BCa CIs.
- `NoTemporalLeakage` (HARD) enforced by every CV runner.
- Stability diagnostics operational: `CoefficientStability`,
  `PredictiveStability` are run by default using an internal small-k CV,
  with override via `stability_splitter=`.

**Acceptance:**
1. On a held-out credit dataset (Lending Club or similar public),
   reproduce statsmodels logistic regression coefficients to 1e-6. KS and
   AUC match scipy/reference implementations to 1e-9.
2. A perfectly separable synthetic dataset raises `AssumptionError` from
   `NoPerfectSeparation` with a message pointing the user at adding `l2`.
3. `mc.expanding_window(time_col=..., gap="365D")` produces folds where
   every validation window starts at least 365 days after its training
   window ends; `NoTemporalLeakage` fires on a synthetic violation.
4. `mc.tune` returns a curve where the chosen lambda corresponds to the
   maximum mean CV metric (optionally with 1-SE rule).
5. `mc.nested_cv` outer error is statistically larger (in expectation,
   verified on simulation) than `mc.tune` inner error on the same data,
   demonstrating the optimism bias ESL §7.10.2 warns about.
6. `mc.bootstrap` produces 95% percentile CIs that contain the OLS
   closed-form CIs to within 5% across a synthetic linear regression.
7. `mc.performance(sol, data)` returns a `PerformanceReport` whose
   `discrimination`, `calibration`, `stability`, and `distribution`
   sub-reports each contain the expected primitive metric values; the
   `__repr__` produces a single block matching the format in §3.3.

### Phase 4 — Basis expansions, WoE, and `mc.binned`

- `mc.ns`, `mc.bs`, `mc.poly`, `mc.step`, `mc.smooth`, `mc.hinge` — basis
  expansion terms covering GAMs and MARS-style features.
- `mc.woe` (WoE-encoded) and `mc.binned` (bin-indicator basis) terms,
  both with `mc.monotonic`, `mc.tree_bins`, `mc.categorical`, `mc.manual`
  binning strategies.
- `mc.binning_table(sol)`, including Information Value per term.
- `mc.interact`, `mc.cross` for interactions.
- WoE prerequisites: `AtLeastOneEventPerBin`, `MinimumBinSize`,
  `MonotonicEventRate` (when `monotonic=True`).
- WoE stability: `WoEMonotonicityPreserved` (post-fit, SOFT).
- Basis-term stability: `SupportContainsPredictData` (predict-time, SOFT).

**Acceptance:**
1. A full credit-modeling workflow runs end-to-end on a public dataset:
   WoE binning → logistic regression → walk-forward CV → bootstrap CIs
   → `PerformanceReport`. Each step is a single function call. The
   model output is always a probability via `mc.predict(sol, data)`.
2. The monotonicity constraint is verifiable from the output binning
   table.
3. `mc.binned` on the same dataset produces equivalent in-sample fit to
   a saturated WoE model but typically different out-of-sample
   performance, demonstrated on a CV comparison.
4. Predicting on data outside the training spline knot range emits a
   `SupportContainsPredictData` warning naming the fraction of
   extrapolated rows.

### Phase 5 — Performance analysis (temporal, segmented, comparative)

- `mc.performance_over_time(sol, data, splitter, reference=)` returns a
  time-indexed performance report.
- `mc.performance_by_segment(sol, data, by=)` returns a per-segment
  performance report.
- `mc.compare({"name": sol, ...}, data=, weights=)` returns a
  `Comparison` value with side-by-side metrics.
- `mc.delong_test(sol_a, sol_b, data, weights=)` for paired AUC
  comparisons on a single held-out set.
- Lazy matplotlib helpers: `.plot()` methods on `PerformanceReport`,
  `Comparison`, and their sub-reports.

**Acceptance:**
1. `mc.performance_over_time` on a synthetic non-stationary panel
   reproduces known AUC drift to 1e-4 across windows.
2. `mc.delong_test` matches the `pROC::roc.test` R reference
   implementation to 1e-6 on a fixed seed.
3. `mc.compare` on two models trained with different lambdas on the same
   data shows AUC differences consistent with the underlying
   model_a − model_b on the same test set, including a DeLong p-value
   that matches `mc.delong_test` directly.
4. `mc.performance_by_segment` on a segmented logistic regression
   produces one `PerformanceReport` per segment plus an aggregate.

### Phase 6 — Segmentation, diagnostics, inspection

- `mc.segmented(by=, base=)` and segmented solve/predict.
- `mc.diagnostics`, `mc.hat_matrix`, `mc.influence`.
- Rich `__repr__` and `_repr_html_` on `Solution`, `BootstrappedSolution`,
  `BinningTable`, `AssumptionReport`, `CVResult`, `TuneResult`,
  `NestedCVResult`, `PerformanceReport`, `Comparison`.

**Acceptance:** A segmented logistic regression with WoE features
produces per-segment `Solution`s, `AssumptionReport`s,
`BootstrappedSolution`s, and `PerformanceReport`s, all from a single
declarative spec.

---

## 9. Coding standards for Claude Code agents

1. **No mutation of specs or solutions after construction.** All public
   dataclasses are `frozen=True`. If you find yourself wanting to mutate,
   add a `.with_(...)` method that returns a new value.

2. **No inheritance in the public API except `Term` and `TermSum`.** Loss,
   Penalty, Solution, Spec are all concrete dataclasses or protocols. New
   model families are new constructors, not subclasses.

3. **No keyword-arguments with magic strings where a value would do.**
   `loss=mc.logistic` is right; `loss="logistic"` is wrong. Strings are
   for data column references only.

4. **Every solver path must be testable in isolation.** The dispatch in
   `solve/__init__.py` is the only place that picks a solver. Each solver
   module exposes a function with an explicit signature.

5. **Numerical correctness is non-negotiable.** Match a reference
   implementation (R's `lm`, `glmnet`, `statsmodels`, or a hand-derived
   closed form) to documented tolerance for every Acceptance test. When in
   doubt, derive the math in a docstring before writing the code.

6. **Sample weights everywhere or nowhere.** A solver, loss, or metric
   that doesn't accept `weights=` is broken. Default is `None` (uniform).

7. **Every new loss, penalty, term, or solver declares its assumptions.**
   A PR that adds a model component without an `assumptions = (...)`
   tuple — and at least one test that fabricates a violation and confirms
   the check fires — is incomplete and must not be merged. See §4 for the
   protocol.

8. **No silent broadcasting, no silent NaN handling.** Validate inputs at
   spec construction time where possible, at solve time otherwise. Errors
   should name the offending term or column.

9. **Docstrings are mathematical.** Every loss, penalty, basis term, and
   solver docstring includes the equation it implements (LaTeX in the
   docstring is fine) and a reference to ESL or another canonical source.

10. **Avoid `sklearn` as a dependency.** Wheels and import time matter for
   a craft project. Numpy, scipy, pandas only for v0. `formulaic`,
   `statsmodels` are acceptable as test-only dependencies for cross-checks.

11. **Tests live next to the math they test.** `tests/test_ols.py`,
    `tests/test_woe.py`, `tests/test_logistic.py`. Each acceptance criterion
    in §7 has a corresponding test file with a top-level docstring quoting
    the criterion.

---

## 10. Reference end-to-end example

```python
import model_crafter as mc
import pandas as pd

df = pd.read_parquet("credit_panel.parquet")

# ------------------------------------------------------------------
# Spec: WoE-encoded logistic regression. Lambda left as a placeholder
# because it will be tuned.
# ------------------------------------------------------------------
def build_spec(lam: float) -> mc.LinearSpec:
    return mc.linear(
        target   = "default_12m",
        features = (
            mc.woe("income",  bins=mc.monotonic(min_bin_size=0.05))
            + mc.woe("age",    bins=mc.tree_bins(max_leaves=8))
            + mc.woe("tenure", bins=mc.monotonic())
            + mc.woe("region", bins=mc.categorical(group_rare=0.01))
        ),
        loss    = mc.logistic,
        penalty = mc.l2(lam),
    )

splitter = mc.expanding_window(
    time_col  = "origination_dt",
    n_folds   = 5,
    horizon   = "90D",
    gap       = "365D",         # 12-month default label
    min_train = "730D",
)

# ------------------------------------------------------------------
# Honest assessment via nested CV (ESL §7.10.2).
# Outer folds report performance; inner folds tune lambda.
# ------------------------------------------------------------------
assessment = mc.nested_cv(
    spec_fn        = build_spec,
    grid           = mc.log_grid(1e-4, 1e2, n=30),
    data           = df,
    outer_splitter = splitter,
    inner_splitter = mc.expanding_window(time_col="origination_dt",
                                         n_folds=3, horizon="90D", gap="365D"),
    metric         = mc.auc,
    weights        = "sample_weight",
)
print(assessment.summary())          # honest held-out AUC, KS, PSI per fold
print(assessment.best_params)        # selected lambdas across outer folds

# ------------------------------------------------------------------
# Pick the final lambda using all data (a separate operation from
# assessment — ESL §7.10.3) and fit the production model.
# ------------------------------------------------------------------
tuned = mc.tune(
    spec_fn  = build_spec,
    grid     = mc.log_grid(1e-4, 1e2, n=30),
    data     = df,
    splitter = splitter,
    metric   = mc.auc,
    weights  = "sample_weight",
    rule     = mc.one_se_rule,       # ESL §7.10's parsimony heuristic
)
sol = tuned.solution                 # refit on full data at tuned.best_param

print(sol)                           # regression table + bins + assumption summary
print(sol.assumptions)               # prerequisites + stability diagnostics

# ------------------------------------------------------------------
# Bootstrap for honest coefficient CIs (ESL §7.11).
# Especially important here because logistic-ridge SEs aren't closed-form.
# ------------------------------------------------------------------
sol_bs = mc.bootstrap(sol, data=df, n_boot=500,
                      stratify="default_12m",
                      weights="sample_weight")
print(sol_bs.coefficient_ci(level=0.95))

# ------------------------------------------------------------------
# Performance analysis — the single first-class report on a fitted model.
# Output is always a probability; the report shows whether that
# probability interpretation is supported by the data.
# ------------------------------------------------------------------
perf = mc.performance(
    sol,
    data      = df,
    reference = df,           # for PSI; use a holdout in practice
    weights   = "sample_weight",
)
print(perf)                   # discrimination + calibration + stability + distribution
perf.calibration.plot()       # diagnostic plot, no transform applied

# Deployment-monitoring view: how does performance evolve over time?
perf_t = mc.performance_over_time(sol, data=df, splitter=splitter)
print(perf_t.summary)

# ------------------------------------------------------------------
# Regulatory documentation pass: opt into classical inference checks.
# These are reported, never warned or raised — they're for the model
# risk committee, not for the modeler.
# ------------------------------------------------------------------
regulator_report = mc.check_assumptions(
    sol.spec, data=df,
    solution=sol,
    classical_inference=True,
)
print(regulator_report)       # Normality, VIF (with reg suggestion), etc.

# ------------------------------------------------------------------
# The model output is a probability. Apply it to new data:
# ------------------------------------------------------------------
new_loans   = pd.read_parquet("new_applications.parquet")
p_default   = mc.predict(sol, new_loans)
```

This snippet is the v0 north star. If it reads like the spec of a credit
risk model — which it should — the package is doing its job.

Note what the example does *and doesn't* do, in ESL spirit:

- **The model output is always a probability.** `mc.predict(sol, data)`
  returns $\hat p(x) \in [0, 1]$. There is no scorecard, no points table,
  no SQL CASE WHEN generation in v0 — those are presentation layers
  downstream of the model, and conflating them with the model itself is
  the same category mistake as treating calibration as a transform.
- **`PerformanceReport` is the central output of fitting**, with the same
  structural status as `AssumptionReport`. Assumptions tell you the model
  is *valid*; the performance report tells you it's *good*. Both are
  values returned by named operations.
- **CV for assessment and tuning are separated.** `nested_cv` reports
  honest held-out performance. `tune` selects the production lambda. The
  two operations are not coupled, because conflating them gives optimistic
  error estimates (ESL §7.10.2).
- **The bootstrap is the coefficient-CI tool**, because lasso/ridge SEs
  aren't closed-form (ESL §7.11).
- **No calibration transform** is applied. The calibration curve and Brier
  score are diagnostics; if calibration is poor, the response is to refit,
  not to layer post-hoc corrections.
- **Classical-inference checks are opt-in**, because they're a different
  question than "does this model predict well." They're useful for
  regulatory documentation; they're not the diagnostic that tells you the
  model works.

---

## 11. Open questions (decide before Phase 4)

- **Categorical encoding for non-WoE terms.** Drop-first dummy? Effect coding?
  Frequency encoding? Pick one default and one alternative.
- **Missing value policy.** WoE has a natural "missing bin"; other terms do
  not. Default to a `MissingBin` strategy per term, or a global policy on
  `LinearSpec`? Lean: per-term, with a sensible default per term type.
- **Predict on unseen levels.** For categorical WoE, what's the WoE of an
  unseen category at predict time? Default to 0 (neutral evidence)? Raise?
  Configurable per term.
- **Numerical stability in IRLS for separable data.** Detect separation and
  raise with a clear error, or fall back to a regularized solve with a
  warning? Lean: detect and raise; nudge the user toward `penalty=mc.l2(...)`.
- **Stability-check thresholds.** Default thresholds for
  `CoefficientStability` (CV-of-CV ≤ 0.2) and `PredictiveStability`
  (metric-SD / metric-mean ≤ 0.1) are reasonable but arbitrary. Fixed in
  the package with `suppress=[...]` as the only escape hatch is the lean
  — moving thresholds silently degrades the principle.
- **Internal CV for stability checks.** Stability assumptions need a
  splitter. Default to internal k=5 random splits, or refuse to run
  unless the user passes `stability_splitter=`? Lean: internal default
  for non-temporal data, refuse for data with a time column unless the
  user passes a temporal splitter (silently using random CV on temporal
  data is exactly the mistake the package is trying to prevent).
- **1-SE rule as the default tuning rule.** ESL §7.10 recommends the
  one-standard-error rule for parsimony in lambda selection. Make it the
  default in `mc.tune` and `mc.nested_cv`, or require the user to opt in
  via `rule=mc.one_se_rule`? Lean: opt-in for transparency, but document
  prominently. The "best mean metric" default is more familiar; 1-SE is
  the principled choice.
- **Bootstrap defaults.** `n_boot=500` is conventional but low for tight
  CIs on lasso selection frequencies; `n_boot=2000` is the literature
  default. Pick one and document the tradeoff.

These do not need to be answered before Phase 1, but they should be answered
before the first WoE+logistic acceptance test in Phase 4.