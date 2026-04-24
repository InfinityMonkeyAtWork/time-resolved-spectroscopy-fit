---
orphan: true
---

# Planning Note: JAX Backend, Jacobians, and Optimizer

## Summary

The current NumPy GIR backend is ready to be treated as complete for the
current lowered-evaluator scope. A JAX port should proceed as a new track,
not as unfinished work that must land before GIR / eval / scheduler can be
called "ready".

That conclusion rests on the current architecture:

- [`evaluate_2d(plan, theta)`](../../src/trspecfit/eval_2d.py) and
  [`evaluate_1d(plan, theta)`](../../src/trspecfit/eval_1d.py) already expose
  theta-level evaluator entry points.
- [`ScheduledPlan2D`](../../src/trspecfit/graph_ir.py) and
  `ScheduledPlan1D` are already mostly packed-array execution plans rather than
  live model objects.
- [`fit_model_gir`](../../src/trspecfit/spectra.py) already extracts the
  optimizer-visible `theta` vector from the full parameter list and routes into
  the compiled evaluator.
- Parity coverage already exists for static 1D/2D models, profiles,
  convolution/IRF, and subcycle-aware cases in
  [`tests/test_evaluate_1d.py`](../../tests/test_evaluate_1d.py),
  [`tests/test_evaluate_2d.py`](../../tests/test_evaluate_2d.py), and
  [`tests/test_gir_integration.py`](../../tests/test_gir_integration.py).

What remains is real backend work, not "finish the GIR architecture" work.


## Current readiness

The current implementation already has several properties that make a JAX
track feasible:

- The 2D evaluator is already organized as a pure-looking function of
  `(plan, theta) -> spectrum`, even though the current implementation uses
  NumPy scratch buffers and mutation internally.
- The hot loops are plan-driven, not theta-driven. In practice this means the
  branch structure is fixed once `schedule_2d(graph)` has run, which is a good
  fit for a compiled backend.
- The scheduler has already done the hard semantic work: expression binding,
  dynamics grouping, convolution lowering, profile lowering, and subcycle
  lowering all happen before the evaluator runs.
- The current lmfit boundary is narrow. `residual_fun` calls a fit function,
  and `fit_model_gir` already converts the full parameter vector into the
  compact optimizer vector expected by the evaluator.

This is enough to start a JAX branch immediately.

It is not necessary to keep the current GIR branch open for additional
"JAX readiness" cleanup before declaring the NumPy backend complete.


## What still blocks a full JAX path

### 1. Backend-agnostic prep

These items are not blockers for calling the NumPy GIR work done, but they are
useful cleanup before or during a JAX port:

- **Flatten expression storage.** `ScheduledPlan2D.expr_programs` and
  `ScheduledPlan2D.profile_expr_programs` are still Python lists of
  `ExprProgram` objects. They should move to fully packed arrays
  (for example CSR-style instruction storage plus per-program offsets).
- **Keep schedule data fully array-native.** The main plan is already close to
  this ideal; the remaining goal is to avoid Python-wrapped structures in the
  hot path altogether.
- **Preserve a stable theta contract.** The evaluator boundary
  `evaluate(plan, theta)` is already the right API. New work should protect
  that contract rather than pushing JAX concerns back into model objects or
  fit-time parsing.

Of these, expression flattening is the one prep item most worth doing even if
the eventual backend choice changes again.

### 2. JAX evaluator port

The main technical work is in the evaluator itself:

- **Mutation-heavy NumPy code must become functional JAX code.** The current
  2D evaluator uses `.copy()`, slice assignment, and in-place accumulation over
  scratch arrays. A jitted JAX version will need explicit functional updates or
  loop-carried state.
- **Python callable dispatch must become JAX-native dispatch.** The current
  evaluator relies on Python dispatch tables (`OP_DISPATCH`,
  `DYNAMICS_DISPATCH`, `PROFILE_DISPATCH`). A jitted backend will need either
  explicit JAX control flow or trace-time unrolling over the scheduled ops.
- **Host-side checks must stay outside the jitted region.** Shape checks and
  Python exceptions are fine at the outer boundary, but should not live inside
  the compiled path.
- **SciPy-dependent kernels need JAX-compatible replacements.** The current
  code still depends on SciPy for Voigt/Faddeeva (`wofz`) and for convolution
  utilities used by the lowered convolution path. A JAX backend needs
  compatible implementations for those pieces.

### 3. Jacobian and optimizer work

Even after a working JAX evaluator exists, Jacobian / optimizer work is still
its own layer:

- [`fitlib.py`](../../src/trspecfit/fitlib.py) currently constructs
  `lmfit.Minimizer` without a Jacobian hook. If we want analytic Jacobians
  while keeping lmfit, we need explicit `Dfun` plumbing.
- A fully custom JAX optimizer is a larger decision than "use JAX for
  derivatives". It also means deciding how to replace or re-scope:
  - existing result objects and reporting,
  - two-stage fitting workflow,
  - confidence-interval tooling,
  - MCMC integration,
  - parameter bounds / transformations,
  - writeback into the current model/result surface.

Because of that, "JAX evaluator + analytic Jacobian" and "replace lmfit" should
be treated as separate milestones, not one first step.


## Recommendation on optimizer strategy

The recommended first optimizer milestone is:

1. build a JAX evaluator,
2. derive a Jacobian from it,
3. keep lmfit as the outer optimizer at first.

Reasons:

- It isolates the performance question. We learn whether JAX + Jacobian is
  actually worthwhile before committing to a larger optimizer rewrite.
- It preserves the existing user-facing fit workflow while the new backend
  proves itself.
- It avoids bundling backend correctness risk with optimizer-behavior risk in
  the same first implementation.

A custom JAX optimizer may still be the right long-term direction, but it
should come after we have:

- a parity-validated JAX evaluator,
- a working analytic Jacobian path,
- and concrete evidence that lmfit itself is the next real bottleneck.


## Recommended implementation order

### Phase A: backend-agnostic prep

- Flatten `expr_programs` and `profile_expr_programs` into packed arrays.
- Keep the evaluator API unchanged: `evaluate(plan, theta) -> ndarray`.
- Add any small helper refactors that make the evaluator easier to port without
  changing behavior.

### Phase B: first JAX evaluator slice

Target file-level 2D fits first; that is where the payoff is.

Initial scope:

- static component ops,
- dynamics groups,
- arithmetic expressions,
- no profiles,
- no convolution,
- no Voigt.

The goal of this phase is to validate the JAX evaluator structure, not to land
full coverage immediately.

### Phase C: widen JAX coverage to the existing GIR surface

Add the remaining lowered features incrementally:

- profile-varying parameters,
- subcycle-aware dynamics,
- resolved-trace convolution,
- Voigt / special-function support.

Each widening step should ship with direct parity tests against the existing
NumPy GIR evaluator, not only against MCP.

### Phase D: analytic Jacobian

- Define a residual function at the theta level.
- Differentiate that residual or model output with JAX.
- Wire the resulting Jacobian into the current fit flow while keeping the rest
  of the API stable.

At this point we can benchmark fit-level speedup rather than only per-call
evaluator speedup.

### Phase E: optional JAX-native optimizer

Only after Phase D should we decide whether replacing lmfit is justified.

This phase is explicitly optional. It should be driven by measured bottlenecks,
not by the assumption that "using JAX" automatically implies "replace the whole
optimizer stack".


## What does not belong in the closing GIR branch

The following should be treated as next-track work, not as unfinished business
for the current lowered-evaluator branch:

- JAX backend porting work
- Jacobian plumbing
- optimizer replacement
- project-level fit lowering
- mixed-backend execution

Those are legitimate follow-ons, but they are not prerequisites for calling the
current NumPy GIR evaluator complete.


## Success criteria for the JAX track

A JAX branch should be considered successful when all of the following are
true:

- A JAX evaluator matches the NumPy GIR evaluator to practical floating-point
  tolerance on the supported subset.
- The widened JAX path covers the same intended feature surface as the current
  NumPy GIR backend, or clearly documents any temporary exclusions.
- Fit-level benchmarks show a meaningful win from JAX alone or from
  JAX + analytic Jacobian.
- The existing NumPy GIR path remains available as the conservative reference
  implementation until the JAX path has earned default status.


## Bottom line

The current scheduler / evaluator design is already good enough to start a JAX
project now.

There is still meaningful implementation work ahead, but it belongs on a new
backend branch. It should not be framed as "we cannot call GIR ready until we
do this." The right framing is: the NumPy GIR path is the completed reference
backend, and JAX is the next backend experiment built on top of it.
