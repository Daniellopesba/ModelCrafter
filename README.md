# model_crafter

Credit risk modelling in Python. Linear and generalised-linear models, basis expansions, weight-of-evidence encoding, temporal cross-validation, bootstrap inference, and held-out performance reporting, organised around an immutable spec / solve / predict API.

## Installation

```bash
pip install -e ".[dev]"
```

`model_crafter` requires Python 3.11 or later. Runtime dependencies are numpy, scipy, and pandas.

## Quickstart

```python
import model_crafter as mc

spec = mc.linear(
    target   = "default_12m",
    features = (
        mc.woe("income",  bins=mc.monotonic(min_bin_size=0.05))
        + mc.woe("age",    bins=mc.tree_bins(max_leaves=8))
        + mc.woe("region", bins=mc.categorical(group_rare=0.01))
    ),
    loss    = mc.logistic,
    penalty = mc.l2(0.1),
)

sol  = mc.solve(spec, data=df)
yhat = mc.predict(sol, new_df)

print(sol.assumptions)
print(mc.performance(sol, data=test))
```

## Subpackages

| Subpackage                  | Description |
| --------------------------- | --- |
| `model_crafter.spec`        | Model specifications as frozen dataclasses; `linear`, `segmented`. |
| `model_crafter.terms`       | Linear-predictor terms: raw columns, polynomial and spline bases, WoE and bin-indicator encodings, interactions. |
| `model_crafter.loss`        | Loss functions: `squared_error`, `logistic`. Each declares its mathematical assumptions. |
| `model_crafter.penalty`     | Regularisation: `l1`, `l2`, and additive compositions. |
| `model_crafter.solve`       | Solvers dispatched by `(loss, penalty)`: closed-form OLS and ridge, coordinate descent for lasso and elastic net, IRLS for logistic. |
| `model_crafter.assumptions` | Prerequisite, stability, and opt-in classical-inference checks attached to every solution. |
| `model_crafter.validation`  | Temporal splitters (`time_split`, `expanding_window`, `rolling_window`, `purged_kfold`), `cross_validate`, `tune`, `nested_cv`, `bootstrap`. |
| `model_crafter.metrics`     | Discrimination (AUC, Gini, KS, Cohen's d), calibration (Brier, ECE, calibration curve, log-loss), stability (PSI), rank-based (lift, cumulative gains). |
| `model_crafter.performance` | `performance`, `performance_over_time`, `performance_by_segment`, `compare`. |
| `model_crafter.inspect`     | Coefficients, diagnostics, hat matrix, influence, binning tables. |

## Documentation

The architectural reference is `DESIGN.md`. The mathematical reference is *The Elements of Statistical Learning* (Hastie, Tibshirani, and Friedman, 2nd ed.); section citations appear throughout the source.

## Testing

```bash
pytest
ruff check
pyright
```

Numerical correctness for each solver is verified against a reference implementation — R's `lm` and `glmnet`, `statsmodels`, or a hand-derived closed form — to a tolerance documented in the corresponding test.

## License

MIT. See `LICENSE`.