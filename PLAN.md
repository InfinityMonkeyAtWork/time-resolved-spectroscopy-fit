# Active Plan: JAX backend track

Roadmap and rationale: [docs/design/jax-planning.md](docs/design/jax-planning.md)
(scope, sequencing, open technical constraints). This file tracks execution.

Working branch: `jax-backend`.

## Guiding decisions

- The NumPy GIR evaluator stays the reference backend throughout; JAX must
  earn default status via parity + fit-level benchmarks.
- **Backend capability gating**: mirror the `can_lower_2d` pattern with a
  JAX-specific capability check (own lowerable-function set and node-kind
  set, per the "backend-specific sets" carve-out in `graph_ir.py`). `Voigt`
  stays out of the JAX set until a JAX-native `wofz` (Humlíček-style) lands.
- **Fallback chain is JAX → NumPy GIR → MCP**: a model that fails the JAX
  gate but passes `can_lower_2d` runs on the compiled NumPy path — no
  user-visible regression from partial JAX coverage.
- "JAX evaluator + analytic Jacobian (keep lmfit)" and "replace lmfit" are
  separate milestones; the latter is optional and evidence-driven (Phase E).

## Phase A: backend-agnostic prep (no JAX dependency; can land alone)

- [x] Flatten `ScheduledPlan2D.expr_programs` into packed arrays
      (`expr_instructions` + `expr_indptr`, CSR-style); `ExprProgram`
      dataclass removed, plans are now fully array-native.
- [x] Same for `profile_expr_programs` (both 1D and 2D plans).
- [x] Keep the evaluator API unchanged: `evaluate(plan, theta) -> ndarray`;
      GIR/evaluator parity tests pass (272) and full suite passes (937).
- [ ] Small helper refactors that ease the port without behavior changes
      (only if needed once Phase B starts).

## Phase B: first JAX evaluator slice (file-level 2D fits)

- [x] Packaging: optional `[jax]` extra; `eval_jax.py` guards the import
      and raises a helpful ImportError; tests skip without jax.
      Importing `eval_jax` enables `jax_enable_x64` (float64 parity).
- [x] JAX capability gate: `can_lower_jax_2d` composes with
      `can_lower_2d`, so JAX-rejected graphs land on the compiled NumPy
      path by construction. Voigt / profiles / convolution / subcycles
      excluded.
- [x] Functional JAX 2D evaluator (`make_evaluator_2d_jax`): static
      component ops, dynamics groups, arithmetic expressions; jnp kernel
      mirrors of the functions/ bodies (LinBack drops its host-side
      ordering ValueError — untraceable; fit bounds must keep order).
- [x] Dispatch strategy: trace-time unrolling — one jitted XLA program
      per plan, no dispatch inside the compiled path.
- [x] Host-side checks outside jit (theta shape; plan-level feature
      check in `_check_plan_supported`).
- [x] Parity tests vs. the NumPy GIR evaluator (tests/test_evaluate_jax.py,
      33 tests, rtol/atol 1e-12): all 9 static ops, all 7 dynamics kinds,
      multi-substep + expression-valued dynamics, gate/rejection cases.
- Timing sanity (212x1131 grid, glp + 2 dynamics): JAX jit 0.78 ms/call
  vs NumPy GIR 7.6 ms/call (~9.8x); compile ~230 ms once per plan;
  max |diff| 1.4e-13.

## Phase C: widen JAX coverage to the existing GIR surface

- [ ] Profile-varying parameters.
- [ ] Subcycle-aware dynamics.
- [ ] Resolved-trace convolution (kernel-matrix form; shapes are already
      theta-independent).
- [ ] Voigt via a JAX-implementable `wofz` approximation, with parity
      tests against the SciPy `wofz` path.
- [ ] Each widening step ships with direct NumPy-GIR parity tests.

## Phase D: analytic Jacobian (keep lmfit)

- [ ] Theta-level residual function.
- [ ] JAX-derived Jacobian wired into `lmfit.Minimizer` (`Dfun` plumbing
      in `fitlib.py`).
- [ ] Fit-level benchmarks (not just per-call evaluator speed).

## Phase E (optional, decide after D): JAX-native optimizer

Only if measurements show lmfit itself is the bottleneck. Not scoped here.

## Success criteria (from the design note)

- JAX matches NumPy GIR to practical float tolerance on the supported subset.
- Coverage matches the NumPy GIR surface or documents temporary exclusions.
- Fit-level benchmarks show a meaningful win (JAX alone or JAX + Jacobian).
- NumPy GIR path remains available as the conservative reference.
