---
orphan: true
---

# Archived Plan: Lowered Evaluator Implementation

> Archived on 2026-04-20 after Phase 6 closed out the lowered-evaluator
> implementation track. Keep [../lowered_evaluator.md](../lowered_evaluator.md)
> as the long-lived design/spec. This file preserves the phase-by-phase
> implementation history and the constraints/success criteria that drove it.

This file tracks implementation status and active follow-up work for the
lowered evaluator.

## Canonical references

- Long-lived design/spec: [../lowered_evaluator.md](../lowered_evaluator.md)
- Supported-model ground truth: [../supported_models.md](../supported_models.md)
- Closed backend decision / benchmarks: [numba_vs_jax.md](numba_vs_jax.md)

## Final status

- Phases 1-5 are complete: GraphIR, scheduling, the lowered evaluator,
  integration, compare mode, benchmark-driven backend validation, static
  caching, and Voigt lowering all landed.
- Phase 6 is complete: `ENERGY_1D`, 1D/2D profile lowering, 2D time-domain
  IRF/convolution lowering, and subcycle-aware lowering are all implemented.
  `fit_model_gir` now covers the full MCP-supported model surface for
  the bundled examples.

## Completed implementation history

- Phase 1: built the GraphIR layer, domain classification, and the initial
  lowering gates.
- Phase 2: added the scheduler, expression compilation, and structural plan
  validation.
- Phase 3: implemented `evaluate_2d(plan, theta)` and numerical parity
  validation against MCP.
- Phase 4: integrated `fit_model_gir`, compare mode, fit-time plan building,
  and post-fit parameter writeback.
- Phase 5: benchmarked the lowered path, added static component caching,
  added Voigt support, and closed the Numba-vs-JAX decision in favor of
  staying on the NumPy backend for this branch.
- Phase 6.1: lowered `ENERGY_1D` fits via `can_lower_1d`, `schedule_1d`,
  and `evaluate_1d`, then wired the fast path into `fit_baseline()` and
  `fit_spectrum()`.
- Phase 6.2a: lowered profile-varying parameters for 1D fits, including
  profile-dependent expressions and parity/dispatch coverage.
- Phase 6.2b: lowered profile-varying parameters in `fit_2d()`, including
  profile expression programs, profiled-component evaluation, and parity
  coverage.
- Phase 6.3: lowered 2D time-domain IRF/convolution nodes for single-cycle
  resolved-trace cases. The graph remains the sole source of truth; unsupported
  convolution shapes still fall back cleanly to MCP.
- Phase 6.4: lowered subcycle dynamics. `SUBCYCLE_REMAP` / `SUBCYCLE_MASK`
  graph nodes are compiled away in `schedule_2d` into per-substep
  `dyn_sub_time_axes` / `dyn_sub_masks` arrays -- no new OpKinds, no
  runtime graph traversal. The evaluator hot loop becomes
  `func(axis[s], *pars) * mask[s]`. Resolved-trace (`subcycle=0`)
  convolution coexists with subcycle-aware dynamics in the same model;
  only `subcycle>0` substeps carry remap/mask data. Parity verified
  against MCP on two-/three-subcycle fixtures (including cross-subcycle
  expressions) and on a mixed IRF + subcycle model; example 3
  (multi_cycle) benchmarks at ~3.8x over MCP at 7e-15 parity, with no
  regression on non-subcycle example 2 (~3.4x, 1e-13 parity).

## Non-goals for this plan

- Project-level fit lowering remained a separate track.
- Jacobian plumbing and optimizer changes remained a separate track.
- JAX backend work remained a separate track.
- Dynamic kernel-support resizing during fitting was out of scope;
  kernel support stayed fixed for a built model / scheduled plan.
- Mixed-backend execution inside one fit was out of scope.

## Constraints

- Prefer flat, packed-array schedule state so the door stays open for a future
  JAX port if the roadmap ever justifies it.
- Coverage parity comes first; widening lowering coverage should not regress
  the previously lowerable subset by more than about 5% per call on existing
  benchmarks.
- Land the remaining work incrementally and keep the default GIR path numerically
  stable.

## Success criteria

- Default `fit_model_gir` dispatch no longer falls back to MCP for the
  remaining supported subcycle-aware cases in the bundled examples.
- Newly lowerable `SUBCYCLE_*` coverage ships with direct parity tests and
  compare-mode coverage.
- Existing lowerable 1D/2D GIR paths remain numerically stable and
  performance-neutral within the regression budget above.
