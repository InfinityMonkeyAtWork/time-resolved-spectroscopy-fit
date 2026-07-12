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

- [x] Profile-varying parameters: sample groups, per-sample profile
      expressions (broadcast virtual rows), profiled ops vectorized
      over aux + averaged (the NumPy path loops per aux point; under
      XLA the fused form is simpler and still 2.6x faster at n_aux=50).
- [x] Subcycle-aware dynamics: free after scheduling — subcycle info is
      pure data (`dyn_sub_time_axes`/`dyn_sub_masks`); gate carve-out
      plus parity tests only.
- [x] Resolved-trace convolution: kernel-matrix apply in jnp (gather,
      quadrature weights, matmul, analytic edge masses via
      `jax.scipy.special.erfc`). The NumPy path's runtime value checks
      (kernel positivity, row sums) are untraceable and omitted.
- [x] Voigt via Weideman (1994) rational `wofz` approximation, 64
      terms, coefficients precomputed at import (host-side FFT);
      accuracy vs scipy `wofz` < 1e-12 rel over the physical domain
      (tested). ~69x faster than the NumPy/scipy path at 212x1131.
- [x] Parity tests vs NumPy GIR for every widening step (44 tests
      total): all conv kernels + chained double IRF, 2- and 3-subcycle
      models with cross-subcycle expressions, profiled amplitude /
      position / Shirley / mixed profile-dynamics expressions, Voigt.
- `can_lower_jax_2d` now spans the full `can_lower_2d` surface (the
  separate sets remain so future NumPy widening doesn't silently imply
  JAX support).
- Test coverage (2026-07-11): every evaluator-vs-interpreter comparison
  in `test_evaluate_2d.py` also asserts JAX parity when jax is
  importable (`_assert_jax_parity`), so the full 2D matrix including
  regression fixtures runs three-way; `test_evaluate_jax.py` keeps only
  JAX-specific tests (gate, Jacobian, wofz, chained conv, e2e fit).
  The main CI job installs `.[dev,jax]`; the min-versions job stays
  jax-free (jax needs numpy>=2.0, incompatible with the numpy==1.26
  floor world) and JAX tests skip there.
- Timing (per call): profiled x0 n_aux=50 175x280: 10.6 ms vs 27.2 ms;
  gaussCONV 212x1131: 1.0 ms vs 5.0 ms; Voigt+dynamics 212x1131:
  0.63 ms vs 44.0 ms.

## Phase D: analytic Jacobian (keep lmfit)

- [x] `make_jacobian_2d_jax(plan)`: `jax.jit(jax.jacfwd(...))` over the
      shared traced evaluator -> `(n_time, n_energy, n_opt)`. Forward
      mode (theta is short, output grid large). Caveats documented:
      boxCONV width derivative is 0 a.e.; step-like `where` switches
      give subgradient zeros at the exact switching point.
- [x] `spec_fun_str = "fit_model_jax"`: 2D fits run residuals on the
      jitted evaluator (`spectra.fit_model_jax`); JAX-rejected graphs
      fall back to the compiled NumPy plan; 1D fits lower to the NumPy
      plan (JAX backend is 2D-only). Evaluator/jacobian are per-plan
      closures — `lmfit.emcee` with `workers > 1` will not pickle them.
- [x] `Dfun` plumbing: `fitlib.jacobian_fun` mirrors `residual_fun`'s
      signature (lmfit calls Dfun with the same fcn_args), negates the
      windowed model Jacobian, and reorders columns to lmfit's varying-
      parameter order; `fit_wrapper(jac_fun=...)` forwards it as
      `Dfun`/`col_deriv=0` for leastsq stages only (auto-set by
      `File.fit_2d` on the JAX path; pass `jac_fun=None` to disable).
- [x] Jacobian validated against central finite differences of
      `evaluate_2d` (the NumPy backend — cross-backend check) for
      dynamics/expressions, convolution, subcycles, profiles, Voigt;
      plus an internal-space check against lmfit's own FD at a real
      fit's start point (exact match), and a slow e2e fit test.
- [x] Fit-level benchmarks: leastsq stage nfev 22 -> 4 (simple GLP
      model, same optimum); Voigt 2D fit (212x1131, stages=2, 1%
      noise): 25.3 s -> 7.1 s wall (~3.6x, JAX time includes compiles),
      identical redchi, leastsq nfev 8 -> 2.
- Note: from aggressively perturbed starts, plain leastsq lands in
  different local minima with analytic vs FD Jacobians (nonconvex
  objective, not a defect) — the default two-stage workflow (Nelder
  first) is the intended guard, unchanged.

## Phase E (optional, decide after D): JAX-native optimizer

Deferred (2026-07-11). The Phase D benchmarks show the evaluator and
Jacobian — not lmfit — dominate fit wall time at current model sizes,
so replacing the optimizer stack is not justified by evidence. Revisit
only if profiling a real workload shows lmfit overhead dominating.

## Success criteria (from the design note)

- JAX matches NumPy GIR to practical float tolerance on the supported subset.
- Coverage matches the NumPy GIR surface or documents temporary exclusions.
- Fit-level benchmarks show a meaningful win (JAX alone or JAX + Jacobian).
- NumPy GIR path remains available as the conservative reference.
