# model_crafter

Credit risk modelling that reads like pen-and-paper math.

A model is a *value*. Fitting is a *function*. The arguments you pass *are*
the equation.

```python
import model_crafter as mc

spec = mc.linear(
    target   = "default_12m",
    features = (
        mc.woe("income",  bins=mc.monotonic(min_bin_size=0.05))
        + mc.woe("age",   bins=mc.tree_bins(max_leaves=8))
        + mc.woe("region", bins=mc.categorical(group_rare=0.01))
    ),
    loss    = mc.logistic,
    penalty = mc.l2(0.1),
)

sol  = mc.solve(spec, data=df)        # → Solution (frozen, picklable)
yhat = mc.predict(sol, new_df)        # → probabilities

print(sol.assumptions)                # is the model valid?
print(mc.performance(sol, data=test)) # is the model good?
```

That is the whole API. `spec` and `sol` are immutable values; there is no
`fit`/`predict` pair on a stateful estimator, no underscore-suffixed
learned attributes, no sklearn compatibility shim. Composition uses `+`
exactly where the math is additive (terms in a linear predictor; penalty
terms in a regularised loss) and nowhere else.

## What's in v0

- **Linear predictors**: OLS, ridge, lasso, elastic net, logistic
  regression and its regularised variants. Numerical correctness is
  pinned against R's `lm` / `glmnet` and against `statsmodels` to
  documented tolerances.
- **Credit-risk first-class**: WoE-encoded terms and ESL-honest
  bin-indicator terms, both with monotonic / tree / categorical /
  manual binning strategies; PSI, KS, lift / cumulative gains; sample
  weights everywhere.
- **Temporal validation**: time-based splits, expanding and rolling
  windows, purged k-fold, walk-forward CV — with a `gap=` knob that
  enforces label-maturation buffers and a runner that refuses to mix
  CV-for-assessment with CV-for-tuning.
- **Two report values**: every `solve()` returns a `Solution` carrying
  an `AssumptionReport` (prerequisites + stability + opt-in classical
  inference); `mc.performance(...)` returns a `PerformanceReport`
  bundling discrimination, calibration, stability and distribution
  diagnostics with a rich `__repr__`. Both are first-class values.
- **Bootstrap**: `mc.bootstrap(sol, data)` for honest CIs on lasso /
  ridge / logistic where closed-form SEs aren't defined (ESL §7.11).
- **Segmentation**: `mc.segmented(by=..., base=...)` fits one solution
  per segment from a single declarative spec.

## What's *not* in v0

No scorecard / points-table generation, no post-hoc calibration
transform, no sklearn API, no deep learning, no distributed training.
Calibration is *observed*, not corrected — the response to a bad
calibration curve is to refit, not to layer a transform.

## Install

```bash
pip install -e ".[dev]"   # numpy, scipy, pandas + pytest, ruff, pyright
```

Python 3.11+. The only runtime dependencies are numpy, scipy, and
pandas.

## More

- [`DESIGN.md`](DESIGN.md) — the architectural contract. ESL §X
  citations throughout; the §10 end-to-end example is the v0 north
  star.
- [`AGENTS.md`](AGENTS.md) — the parallel-build playbook.
- [`CLAUDE.md`](CLAUDE.md) — house rules for agents editing this
  codebase.
