---
orphan: true
---

# JAX Backend Track (Phases A–D)

> **Status: implemented** (2026-07-11, `jax-backend` branch, v0.12.0).
> Execution record of the JAX track planned in
> [../jax-planning.md](../jax-planning.md). The backend lives in
> `eval_jax.py`, gated by `graph_ir.can_lower_jax_2d`, selected via
> `Project.spec_fun_str = "fit_model_jax"`. Phase E (optimizer
> replacement) was deferred. Follow-on design notes:
> [../ui.md](../ui.md), [../project-level-fits.md](../project-level-fits.md).

## Guiding decisions

- The NumPy GIR evaluator stays the reference backend throughout; JAX
  must earn default status via parity + fit-level benchmarks.
- **Backend capability gating**: mirrors the `can_lower_2d` pattern
  with a JAX-specific capability check (own lowerable-function set and
  node-kind set, per the "backend-specific sets" carve-out in
  `graph_ir.py`).
- **Fallback chain is JAX → NumPy GIR → MCP**: a model that fails the
  JAX gate but passes `can_lower_2d` runs on the compiled NumPy path —
  no user-visible regression from partial JAX coverage. Enforced by
  construction: `can_lower_jax_2d` composes with `can_lower_2d`.
- "JAX evaluator + analytic Jacobian (keep lmfit)" and "replace lmfit"
  are separate milestones; the latter is optional and evidence-driven.

## Phase A: backend-agnostic prep (no JAX dependency; landed alone)

- Flattened `ScheduledPlan2D.expr_programs` into packed arrays
  (`expr_instructions` + `expr_indptr`, CSR-style); `ExprProgram`
  dataclass removed, plans are fully array-native.
- Same for `profile_expr_programs` (both 1D and 2D plans).
- Evaluator API unchanged: `evaluate(plan, theta) -> ndarray`.

## Phase B: first JAX evaluator slice (file-level 2D fits)

- Packaging: optional `[jax]` extra; `eval_jax.py` guards the import
  and raises a helpful ImportError; tests skip without jax. Importing
  `eval_jax` enables `jax_enable_x64` globally (float64 parity).
- Functional JAX 2D evaluator (`make_evaluator_2d_jax`): jnp kernel
  mirrors of the `functions/` bodies. LinBack drops its host-side
  ordering ValueError — untraceable; fit bounds must keep order.
- Dispatch strategy: trace-time unrolling — one jitted XLA program per
  plan, no dispatch inside the compiled path; only theta is traced.
- Host-side checks outside jit (theta shape; plan-level feature check
  in `_check_plan_supported`).
- Timing sanity (212x1131 grid, glp + 2 dynamics): JAX jit 0.78
  ms/call vs NumPy GIR 7.6 ms/call (~9.8x); compile ~230 ms once per
  plan; max |diff| 1.4e-13.

## Phase C: widened to the full lowered 2D surface

- Profile-varying parameters: sample groups, per-sample profile
  expressions (broadcast virtual rows), profiled ops vectorized over
  aux + averaged (the NumPy path loops per aux point to avoid
  temporaries; under XLA the fused form is simpler and still 2.6x
  faster at n_aux=50).
- Subcycle-aware dynamics: free after scheduling — subcycle info is
  pure data (`dyn_sub_time_axes`/`dyn_sub_masks`); gate carve-out plus
  parity tests only.
- Resolved-trace convolution: kernel-matrix apply in jnp (gather,
  quadrature weights, matmul, analytic edge masses via
  `jax.scipy.special.erfc`). The NumPy path's runtime value checks
  (kernel positivity, row sums) are untraceable and omitted.
- Voigt via Weideman (1994) rational `wofz` approximation, 64 terms,
  coefficients precomputed at import (host-side FFT); accuracy vs
  scipy `wofz` < 1e-12 rel over the physical domain (tested). ~69x
  faster than the NumPy/scipy path at 212x1131.
- `can_lower_jax_2d` spans the full `can_lower_2d` surface; the
  separate sets remain so future NumPy widening doesn't silently imply
  JAX support.
- Timing (per call): profiled x0 n_aux=50 175x280: 10.6 ms vs 27.2 ms;
  gaussCONV 212x1131: 1.0 ms vs 5.0 ms; Voigt+dynamics 212x1131:
  0.63 ms vs 44.0 ms.

## Phase D: analytic Jacobian (keeping lmfit)

- `make_jacobian_2d_jax(plan)`: `jax.jit(jax.jacfwd(...))` over the
  shared traced evaluator -> `(n_time, n_energy, n_opt)`. Forward mode
  (theta is short, output grid large). Caveats: boxCONV width
  derivative is 0 a.e.; step-like `where` switches give subgradient
  zeros at the exact switching point.
- `spec_fun_str = "fit_model_jax"`: 2D fits run residuals on the
  jitted evaluator (`spectra.fit_model_jax`); JAX-rejected graphs fall
  back to the compiled NumPy plan; 1D fits lower to the NumPy plan
  (JAX backend is 2D-only). Evaluator/jacobian are per-plan closures —
  `lmfit.emcee` with `workers > 1` will not pickle them.
- `Dfun` plumbing: `fitlib.jacobian_fun` mirrors `residual_fun`'s
  signature (lmfit calls Dfun with the same fcn_args), negates the
  windowed model Jacobian, and reorders columns to lmfit's varying-
  parameter order; `fit_wrapper(jac_fun=...)` forwards it as
  `Dfun`/`col_deriv=0` for leastsq stages only (auto-set by
  `File.fit_2d` on the JAX path; pass `jac_fun=None` to disable).
- Jacobian validated three ways: central finite differences of
  `evaluate_2d` (cross-backend) for dynamics/expressions, convolution,
  subcycles, profiles, Voigt; lmfit's internal-variable FD at a real
  fit's start point (exact match through the bounds transformation);
  slow e2e `File.fit_2d` truth-recovery test.
- Fit-level benchmarks: leastsq stage nfev 22 -> 4 (simple GLP model,
  same optimum); Voigt 2D fit (212x1131, stages=2, 1% noise): 25.3 s
  -> 7.1 s wall (~3.6x, JAX time includes compiles), identical redchi,
  leastsq nfev 8 -> 2. Benchmark-skill fit mode (example 2): JAX+Dfun
  2.10 s vs GIR 4.86 s vs interpreter 17.0 s.
- Investigated: from aggressively perturbed starts, plain single-stage
  leastsq lands in different local minima with analytic vs FD
  Jacobians (nonconvex objective, not a defect) — the default
  two-stage workflow (Nelder first) is the intended guard, unchanged.

## Test coverage decision

Every evaluator-vs-interpreter comparison in `test_evaluate_2d.py`
also asserts JAX parity when jax is importable (`_assert_jax_parity`),
so the full 2D matrix including regression fixtures runs three-way;
`test_evaluate_jax.py` keeps only JAX-specific tests (gate, Jacobian,
wofz, chained conv, e2e fit). The main CI job installs `.[dev,jax]`;
the min-versions job stays jax-free (jax needs numpy>=2.0,
incompatible with the numpy==1.26 floor world) and JAX tests skip
there.

## Phase E: JAX-native optimizer — deferred

Deferred (2026-07-11). The Phase D benchmarks show the evaluator and
Jacobian — not lmfit — dominate fit wall time at current model sizes.
Revisit only if profiling a real workload shows lmfit overhead
dominating; the most plausible pilot is a vmap-batched slice-by-slice
solver (see [../ui.md](../ui.md)).
