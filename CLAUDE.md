# CLAUDE.md

You are working on `model_crafter`, a Python package for credit risk modeling
built around the principle that models should read like pen-and-paper
mathematics. This file is your entry point.

## Before you write a single line of code

Read these in order. Every time. No exceptions.

1. `DESIGN.md` — the architectural contract for the package. Section 8 (Phased
   build plan) defines the acceptance criteria for whatever phase you are
   working on. **The acceptance criteria are the contract.** A change passes
   when its acceptance criteria pass; it does not pass otherwise.
2. `AGENTS.md` — the parallel-execution playbook. It tells you which files
   you own, which files you can read but not write, and what your specific
   task is. **Find your assigned task in AGENTS.md before doing anything.**
3. Section 9 of `DESIGN.md` — coding standards. Non-negotiable.

If your task is not explicitly listed in `AGENTS.md`, stop and ask. Do not
invent a task. Do not work outside your assigned workspace.

## The package's identity (one paragraph)

`model_crafter` collapses model fitting into three verbs applied to values:
`spec = mc.linear(...)`, `sol = mc.solve(spec, data)`, `yhat = mc.predict(sol, new_data)`.
Specs and solutions are immutable, picklable dataclasses. There is no
`fit/predict/transform`, no inheritance hierarchy of estimators, no `_`-suffixed
learned attributes. Composition uses `+` only for terms in a linear predictor
and for penalty sums. The package's two first-class outputs of fitting are
`AssumptionReport` (is the model valid?) and `PerformanceReport` (is the
model good?). Model output is always a probability. There is no scorecard
generation, no calibration transform, no sklearn compatibility layer.

When in doubt about API shape, the reference end-to-end example in §10 of
`DESIGN.md` is the north star.

## Rules of the road

1. **ESL is the canonical reference.** When `DESIGN.md` cites ESL §X, that
   section governs. When the design and ESL disagree, the design wins (this
   is rare and called out explicitly — e.g., calibration as transform is
   deliberately omitted against ESL §9.7).
2. **Numerical correctness is verified against a reference implementation.**
   Every solver matches R's `lm`, `glmnet`, or `statsmodels` to a documented
   tolerance. If your code doesn't match, your code is wrong — not the
   reference.
3. **No sklearn dependency.** Period. Numpy, scipy, pandas only. `formulaic`
   and `statsmodels` are acceptable as test-only dependencies for cross-checks.
4. **Frozen dataclasses everywhere.** No mutation of specs or solutions after
   construction. If you find yourself wanting to mutate, add a `.with_(...)`
   method that returns a new value.
5. **Sample weights everywhere or nowhere.** A solver, loss, or metric that
   doesn't accept `weights=` is broken.
6. **Every new loss, penalty, term, or solver declares its assumptions.** No
   exceptions. See §4 of `DESIGN.md` for the protocol.
7. **Tests live next to the math they test.** `tests/test_ols.py`,
   `tests/test_woe.py`, etc. Each test file's docstring quotes the acceptance
   criterion it verifies.
8. **Lint, type-check, and test before considering work done.**
   `ruff check`, `pyright`, `pytest -x` all pass. CI runs all three.

## Working with other agents in parallel

`AGENTS.md` partitions work into independent task units with explicit
workspace ownership. **Do not edit files outside your workspace.** If your
task needs a type or function from another agent's workspace, import it via
the public interface defined in `AGENTS.md` for that workspace. If that
interface doesn't exist yet (because the other agent hasn't finished),
either:

- (a) write your code against the interface contract as documented and
  mock the dependency in your tests, or
- (b) stop and report the blocking dependency.

Do not work around missing interfaces by importing from another agent's
private modules. The interface boundary exists so we can integrate cleanly.

## When you finish

1. All your tests pass.
2. `ruff check` and `pyright` pass on your workspace.
3. You've written a short summary of what you built, what's tested, and any
   open questions. Put it in `notes/<task-id>.md`.

That's it. Don't merge yourself; the integration agent (or the human) does that.