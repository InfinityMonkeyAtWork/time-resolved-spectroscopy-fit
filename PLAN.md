# Active Plan: Lowered Evaluator Follow-ups

This file tracks implementation status and active follow-up work for the
lowered evaluator.

## Canonical references

- Long-lived design/spec: [docs/design/lowered_evaluator.md](docs/design/lowered_evaluator.md)
- Supported-model ground truth: [docs/design/supported_models.md](docs/design/supported_models.md)
- Closed backend decision / benchmarks: [docs/design/archive/numba_vs_jax.md](docs/design/archive/numba_vs_jax.md)

## Current status

- Phases 1-5 are complete: GraphIR, scheduling, the lowered evaluator,
  integration, compare mode, benchmark-driven backend validation, static
  caching, and Voigt lowering all landed.
- Phase 6 is partially complete: `ENERGY_1D`, 1D/2D profile lowering, and
  2D time-domain IRF/convolution lowering are implemented.
- The current active step is subcycle-aware lowering.

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

## Active next step: Phase 6.4

Lower subcycle dynamics.

- Compile `SUBCYCLE_REMAP` and `SUBCYCLE_MASK`.
- Extend the profile-capable / convolution-capable lowering path to
  subcycle-aware models once the single-cycle path is solid.
- Keep GraphIR as the sole semantic source of truth instead of re-deriving
  support from MCP-side helpers.
- Ship direct parity tests against MCP plus compare-mode integration coverage
  before widening the lowering gate.

## Non-goals for this plan

- Project-level fit lowering remains a separate track.
- Jacobian plumbing and optimizer changes remain a separate track.
- JAX backend work remains a separate track.
- Dynamic kernel-support resizing during fitting remains out of scope;
  kernel support stays fixed for a built model / scheduled plan.
- Mixed-backend execution inside one fit remains out of scope.

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
