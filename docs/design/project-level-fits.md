---
orphan: true
---

# Design Note: Project-Level Shared Fits on a Compiled Backend

Forward-looking note (2026-07-11). Captures the design direction for
the "Project-level fit backend" TODO item, written just after the JAX
backend landed (see [jax-planning.md](jax-planning.md) and the
`eval_jax.py` section of [repo_architecture.md](repo_architecture.md)).
Nothing here is implemented.

## Current state

`Project.fit_2d()` already supports `Project`/`File`/`Static` vary
levels, but evaluates through `fit_project_mcp()`: per residual call it
distributes the combined parameter vector to each file's model via
name/dict lookups, runs the full **interpreter** (`create_value_2d`)
per file, applies per-file `e_lim`/`t_lim` windows, and concatenates.
This is the slowest evaluation path left in the codebase — none of the
lowered-evaluator work applies to it yet.

The original TODO question was: lower the multi-file residual to GIR,
or keep project-managed per-file loops?

## Recommended direction: plans + JAX + joint analytic Jacobian

These are not alternatives — the JAX evaluator consumes
`ScheduledPlan`s, so per-file lowering (graphs, plans, and packing the
combined-theta -> per-file mapping into index arrays instead of name
dicts) is the prerequisite step either way. The question is only what
evaluates the plans, and the shared-fit workload is where JAX's
*relative* edge over a NumPy per-file loop is largest:

1. **The Jacobian argument scales with file count.** A shared fit's
   combined theta is large (~n_files x per-file params, minus shared).
   Numeric differencing costs one full multi-file evaluation per theta
   entry per iteration — 10 files x ~6 params is 40+ multi-file
   evaluations per Jacobian estimate. `jax.jacfwd` shares the forward
   computation across all columns inside one XLA program, and the
   shared parameters' columns (which cut across every file) come out
   exactly rather than via noisy differencing. The leastsq nfev
   collapse measured on single files (8 -> 2, 22 -> 4) multiplies by
   the per-call cost of evaluating all files.
2. **The multi-file residual is one fusable program.** Concatenating N
   independent per-file evaluations is exactly what XLA fuses and
   internally parallelizes — no Python loop between files. The common
   shared-fit scenario (same physical model across a measurement
   series) is the ideal `vmap` case: identical plan structure, stacked
   data, one batched call. Unlike batched slice-by-slice fitting
   ([ui.md](ui.md)), none of this needs an optimizer replacement — it
   is one joint lmfit `leastsq` with a fused residual and an analytic
   `Dfun`, i.e. Phase-D machinery, not Phase E.
3. **The baseline is the interpreter, not GIR.** Headroom per call is
   interpreter-vs-JAX (10-70x measured on single files), on top of the
   Jacobian effect.

## Implementation sketch

- Per file: `build_graph` + `schedule_2d` (unchanged machinery).
- Pack the vary-level mapping once at fit setup: combined theta ->
  per-plan theta scatter as index arrays (replacing the per-call
  name/dict distribution in `fit_project_mcp`).
- A JAX factory over the list of plans: evaluate each plan's traced
  function, apply the per-file windows (static slices — they trace
  cleanly), flatten and concatenate; `jax.jit` the whole residual and
  `jax.jacfwd` it for the joint Jacobian.
- Wire into the existing flow like the single-file path: a project
  variant of `fit_model_jax` plus `fitlib.jacobian_fun`-style `Dfun`
  column reordering against the combined lmfit parameter set.

## Caveats and open questions

- **Compile time** grows with total op count across files; a 10-file
  program may take a few seconds to compile — amortized over a joint
  fit, but worth measuring.
- **Heterogeneous files** (different grids or model structures) forgo
  `vmap` batching; unrolled fusion in one program still applies.
- **Mixed lowerability**: one non-JAX-lowerable file currently implies
  falling back for the whole project. Mixed-backend execution is
  explicitly next-track in [jax-planning.md](jax-planning.md); the
  first implementation should fall back whole-project (JAX -> NumPy
  plans -> MCP) rather than mix.
- Same constraints as the single-file JAX path: closures do not pickle
  (no parallel-worker MCMC), and the untraceable runtime value checks
  are absent.
- Weighted residuals (`sigma_type` expansion, tracked in `TODO.md`)
  should be designed in from the start if it lands first.
