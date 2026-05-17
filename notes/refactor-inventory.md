# Refactor inventory

Baseline at start of pass: `ruff check` clean, `pyright` 0 errors, `pytest -x` 719 passed.

## Modules by line count

| LOC | Module |
| ---:|---|
| 1424 | terms/woe.py |
|  839 | terms/basis.py |
|  743 | inspect.py |
|  636 | metrics/classification.py |
|  623 | solve/irls.py |
|  613 | solve/coordinate.py |
|  567 | performance/report.py |
|  539 | assumptions/logistic.py |
|  536 | penalty.py |
|  534 | assumptions/stability.py |
|  497 | validation/splitters.py |
|  474 | validation/tune.py |
|  468 | solution.py |
|  447 | metrics/calibration.py |
|  428 | assumptions/classical.py |
|  420 | validation/bootstrap.py |
|  419 | terms/interact.py |
|  399 | assumptions/woe.py |
|  377 | validation/cross_validate.py |
|  369 | loss.py |
|  367 | performance/over_time.py |
|  360 | assumptions/__init__.py |
|  345 | solve/__init__.py |
|  308 | performance/compare.py |
|  298 | validation/lambda_path.py |
|  293 | solve/ridge.py |
|  286 | solve/segmented.py |
|  269 | metrics/stability.py |
|  267 | metrics/rank.py |
|  261 | terms/base.py |
|  235 | performance/by_segment.py |
|  215 | assumptions/temporal.py |
|  214 | _internal/linalg.py |
|  196 | spec.py |
|  179 | metrics/_common.py |
|  156 | __init__.py |
|  148 | _internal/design.py |
|  129 | validation/over_time.py |
|  127 | assumptions/_types.py |
|  120 | assumptions/_common.py |
|  118 | assumptions/prerequisites.py |
|  101 | solve/_registry.py |
|   78 | metrics/__init__.py |
|   77 | solve/ols.py |
|   60 | terms/__init__.py |
|   51 | validation/__init__.py |
|   44 | performance/__init__.py |
|    7 | _internal/__init__.py |

## Split candidates (Step A predicates fire)

**`terms/woe.py` (1424 lines).** Five predicates fire: covers binning strategies *and* Term semantics (two frames); `_fit_tree`/`_fit_categorical`/`_fit_manual` are disjoint per-strategy fitters; the four binning dataclasses plus `BinningResult` consume ~200 lines before any computation; file is well over 400 lines; a reader hunting for `woe()` scrolls past the entire binning algorithm zoo. Split into:

- `terms/binning.py` — `BinningResult`, the four binning strategy dataclasses, their factories (`monotonic`, `tree_bins`, `categorical`, `manual`), `fit_binnings` dispatcher, and the per-strategy fit algorithms.
- `terms/woe.py` — `WoETerm`, `BinnedTerm`, `woe()`, `binned()` factories and their predict-time assignment helpers.

**`metrics/classification.py` (636 lines).** This is the worked example in Step A. Three frames (rank-based discrimination, effect size, paired-model inference); `_midrank` used by `auc` and DeLong but not `cohens_d`; `scipy.stats.norm.cdf` imported for DeLong only; ~120 lines of dataclasses before any code; ~640 lines total. Split into:

- `metrics/_midrank.py` — the midrank algorithm (used by discrimination and by DeLong).
- `metrics/discrimination.py` — `auc`, `gini`, `ks` and their result types.
- `metrics/effect_size.py` — `cohens_d` and its result.
- `performance/compare.py` — `delong_test` moves here next to `mc.compare`, the only consumer.

**`terms/basis.py` (839 lines).** Five basis families (B-splines, natural splines, polynomial, step, hinge) each with their own dataclass and helpers. Predicates: multiple frames; `_bspline_basis_matrix`/`_ns_projection` disjoint from poly/step/hinge; >800 lines. Split into:

- `terms/_basis_common.py` — `_BasisExpandedTerm`, `_freeze_state`, `_x_series`.
- `terms/spline.py` — `bs`, `ns`, `smooth` (shared knot logic).
- `terms/nonlinear.py` — `poly`, `step`, `hinge`.

**`inspect.py` (743 lines).** Three frames: WoE binning inspection, coefficient tables, linear-model diagnostics. Helpers like `_solution_is_ols`/`_rebuild_design_and_y` only touch diagnostics; `binning_table` is independent. Convert to package:

- `inspect/__init__.py` re-exports.
- `inspect/coefficients.py` — `coefficients` + SE extraction.
- `inspect/binning_table.py` — `BinningTable`, `binning_table`.
- `inspect/diagnostics.py` — `Diagnostics`, `Influence`, `hat_matrix`, `diagnostics`, `influence` + design helpers.

**`solve/irls.py` (623 lines).** Two solver families: Newton IRLS (logistic + ridge-logistic) and proximal-Newton CD (L1 / elastic-net logistic). The prox-CD path has its own `_split_l1_l2` / `_solve_weighted_enet_subproblem` helpers used nowhere else. Split into:

- `solve/irls.py` — IRLS and ridge-IRLS only.
- `solve/prox_cd.py` — `solve_logistic_prox_cd` and prox-CD-specific helpers.

## Left alone (single predicate fires or coherent)

- `performance/report.py` (567): result dataclasses are coupled to `performance()`; splitting would scatter.
- `assumptions/stability.py` (534): splitting `ComparableFeatureScales` or `SupportContainsPredictData` out yields one-class modules.
- `solve/coordinate.py` (613): single coherent topic (CD for lasso/enet).
- `penalty.py` (536), `loss.py` (369), `solution.py` (468), `solve/__init__.py` (345): one frame each.
- Everything ≤ ~300 lines: not flagged unless predicates fire, and they don't.

## Merge candidates

None. The only under-30-line module (`_internal/__init__.py` at 7 lines) is an intentionally empty package init.
