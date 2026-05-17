# Maintainability pass

## Before

The codebase was in better shape than the prompt's worst-case
description. The docstrings already carried equations, ESL citations,
and concrete numerical-correctness notes rather than restating function
names. The dominant AI-uniformity tells were the structural-aesthetic
ones: every module used a `# ----` divider line above a single comment
heading; every internal module declared an `__all__`; the top-level
`__init__.py` carried per-phase shepherding comments left over from the
parallel build. A handful of docstrings hedged with words like
"essentially" or "comprehensive"; nothing systemic. The largest
modules (`terms/woe.py` at 1424 lines, `metrics/classification.py` at
636, `terms/basis.py` at 839, `inspect.py` at 743, `solve/irls.py` at
623) had drifted to cover multiple mathematical frames, which made
function discovery harder than it needed to be.

## Splits

Five modules fired the Step A predicates; all were split.

- **`metrics/classification.py`** (636 lines): three frames
  (rank-based discrimination, effect size, paired inference), disjoint
  helpers (`_midrank` used only by DeLong), a `scipy.stats` import for
  one call, ~120 lines of dataclasses up top. Split into
  `metrics/discrimination.py` (auc/gini/ks), `metrics/effect_size.py`
  (cohens_d), and `performance/_delong.py` (the midrank /
  structural-component algebra). `delong_test` moved to
  `performance/compare.py` next to `mc.compare`, its primary consumer;
  `mc.delong_test` now re-exports from `performance` instead of
  `metrics`.
- **`terms/woe.py`** (1424 lines): every predicate fired. The binning
  strategies and their per-strategy fit algorithms moved to
  `terms/binning.py` (`BinningResult`, the four strategy classes, the
  factories, and the full `_fit_*` zoo). `terms/woe.py` is now
  `WoETerm` + `BinnedTerm` + `fit_binnings` + predict-time assignment.
- **`terms/basis.py`** (839 lines): five basis families
  (`bs`/`ns`/`smooth`, `poly`, `step`, `hinge`) with disjoint helpers.
  Split into `terms/spline.py` (B-splines and natural splines + shared
  knot logic), `terms/nonlinear.py` (poly/step/hinge), and
  `terms/_basis_common.py` (the `_BasisExpandedTerm`, `_freeze_state`,
  `_x_series` plumbing).
- **`inspect.py`** (743 lines): three frames — WoE binning inspection,
  coefficient tables, closed-form diagnostics. Converted to a package:
  `inspect/binning_table.py`, `inspect/coefficients.py`,
  `inspect/diagnostics.py`, plus `inspect/_common.py` for the
  spec-kind predicates and bootstrap-pointer message both
  coefficients and diagnostics need.
- **`solve/irls.py`** (623 lines): two solver families (Newton IRLS for
  ridge/no-penalty; proximal-Newton CD for L1/elastic net). Split into
  `solve/irls.py`, `solve/prox_cd.py`, and `solve/_irls_core.py` for
  the working-response / convergence helpers both paths share.

The remaining ~250+ line modules (`solution.py`, `penalty.py`,
`performance/report.py`, `assumptions/stability.py`, etc.) each cover
one mathematical frame; the predicates fired at most once for any of
them, so they stayed.

## Other changes

The largest non-split pass was stripping every `# -----` /
`# Heading` / `# -----` three-line block to a single comment heading
(28 files, ~185 lines), and stripping `__all__` from every internal
module (47 files, ~280 more lines) since the package's public surface
is defined by `model_crafter/__init__.py`. The top-level
`__init__.py` lost its per-phase narration comments. `spec.py` lost
the dead `_ = Any` keep-alive hack. A handful of "essentially" /
"please note"-style hedges were tightened to direct claims; the
incidence was already low.

I did not rewrite test bodies. The only test-file edits are import
updates from the splits (`tests/test_metrics.py`,
`tests/test_compare.py`, `tests/test_basis.py`,
`tests/test_binning_strategies.py`, `tests/test_binned.py`,
`tests/test_woe.py`).

Net line count: 16,661 → 15,889 (-772, -4.6%). The codebase shrunk
less than the prompt's 10% target because the starting material was
denser than the prompt anticipated; further shrinkage would have
required deleting math-bearing docstrings the prompt explicitly says
to keep. `ruff check`, `pyright`, and `pytest` (719 tests) all green.
