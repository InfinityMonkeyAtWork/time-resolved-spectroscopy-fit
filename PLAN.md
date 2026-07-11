# Active Plan: Exact conv edge handling via kernel edge masses + kernel cleanup

Branch: `conv-kernel-matrix` (continues the uncommitted kernel-matrix
change set; everything commits together). Design note:
[docs/design/archive/kernel-matrix-convolution.md](docs/design/archive/kernel-matrix-convolution.md)
(edge-handling section to be corrected by this work). TODO.md item:
"Exact conv edge handling (kernel CDFs)" `[ACTIVE]`.

## Problem

The ghost-point edge scheme silently truncates broad kernels: coverage is
only `n_t x boundary_step` per side, any tail mass beyond it is discarded,
and row normalization hides the loss (semantics degrade from edge-value
padding to truncate-and-renormalize). Worst exactly on fine-boundary axes.

## Decisions adopted

- **Drop `voigtCONV` and `lorentzCONV`** (2026-07-11, user decision) on
  physics grounds: no basis as *time-domain* IRFs (Voigt/Lorentz are
  energy-domain lineshape stories, already covered by the energy layer);
  unused by every example; pre-1.0 is the time to remove names. Voigt's
  missing closed-form CDF is what blocked the exact-edge design; Lorentz
  has an elementary CDF (`(W/2)(atan(2x/W) + pi/2)` for our body), so its
  removal is purely the physics/API call, not a technical necessity.
  Dropping Voigt retires `wofz` from `functions/time.py` (closes the JAX
  conv-kernel gap).
- **Exact analytic exterior masses via per-kernel edge-mass companions**
  replace the ghost-point extension entirely. Each companion returns both
  mass vectors directly: `_<name>CONV_edge_mass(dt_left, dt_right,
  **params) -> (M_L, M_R)` where `M_L(i) = integral of the body over
  (-inf, t_0] = S(dt_left[i])` (upper tail) and `M_R(i) = integral over
  [t_end, inf) = lower tail at dt_right[i]`. Returning masses (not CDFs)
  lets each kernel use its natural well-conditioned form — `erfc` for
  gauss, direct `exp` for the exponentials, `clip` for box — and removes
  any `G(inf)` / `np.inf`-evaluation convention. (Note: plain
  `G(inf)-G(x)` subtraction would already be fine in *absolute* terms —
  masses join O(1) row sums — but the direct forms are cleaner and free.)
  No truncation for any theta; no coverage limit; no guard needed.
- **Companions must integrate the body with its exact normalization**
  (gauss body is peak-1: `M = SD*sqrt(pi/2)*erfc(x/(sqrt(2)*SD))` form;
  expDecay/expRise: `tau`-scaled one-sided exps, one of the two masses
  identically 0 on most rows; expSym piecewise; box: `clip` form).
- **Operator shrinks**: dt matrix `(n_t, n_t)` (was `(n_t, 3*n_t)`);
  `n_ghost` concept deleted. `ConvOperator` fields: `dt_unique`,
  `gather_idx`, `quad_weights` (true trapezoid: half-cells at both ends —
  NOT `np.gradient`, whose endpoint weights are full cells), `dt_left`
  (`t - t[0]`, >= 0), `dt_right` (`t - t[-1]`, <= 0).
- **Layering preserved**: `arrays.py` stays kernel-agnostic —
  `conv_matrix_apply(operator, kernel_values, edge_mass_left,
  edge_mass_right, y)`; callers compute the masses via the registry.
- **Companions are private** (`_gaussCONV_edge_mass`, ...) in
  `functions/time.py`, exposed through one public registry dict
  `CONV_EDGE_MASS: dict[str, Callable]` — single source for both paths;
  private names keep them out of registry introspection (no
  INTERNAL_SUFFIXES resurrection).
- **No callables on the plan**: `ScheduledPlan2D` keeps numeric
  kernel-kind ids exactly as today; the evaluator resolves companions via
  a `CONV_EDGE_MASS_DISPATCH` keyed by the same `ConvKernelKind` enum
  (mirror of `CONV_KERNEL_DISPATCH`). `schedule_2d` only *validates* the
  entry exists. Keeps the plan serializable / JAX-friendly.
- **Missing companion is a loud validation error** at `add_components` /
  `schedule_2d` time, not a silent fallback.
- **Validation split per the two-layer design**: authoring layer — conv
  kernel parameters (all widths/timescales) must have strictly positive
  lower bounds, and *varying* parameters must carry an explicit lower
  bound (the unbound `[value, True]` form would give lmfit `min=-inf`),
  enforced at model load with a clear error; edge-mass companions
  validate their parameters (strictly positive, finite) on every
  evaluation as the backstop for expression-driven kernel parameters
  (boxCONV width = 0 would otherwise silently become the identity
  operator); hot path — cheap O(n) finite + nonnegative checks on
  `kernel_values`, both mass vectors, and all input shapes (incl. `y`)
  in `conv_matrix_apply` (row-sum check alone lets signed errors
  cancel). These are host-side checks, outside any future jit boundary.
- **Release as 0.11.0, not 0.10.3** (public names removed: two kernels,
  `conv_kernel_support`, width helpers; conv semantics changed).
  pyproject currently says 0.10.3 — user call to rebump.

## Task list

1. [x] Remove `voigtCONV` + `lorentzCONV`: bodies + `wofz` import
   (`functions/time.py`), `ConvKernelKind` entries + name map
   (`graph_ir.py`), `CONV_KERNEL_DISPATCH` (`eval_2d.py`), their classes in
   `tests/test_functions_convolution.py`, entries in
   `tests/test_config_functions.py`, `MonoExpPosLorentzIRF` /
   `MonoExpPosVoigtIRF` parity pairs in `tests/test_evaluate_2d.py`
   (~L890) + their models in `tests/models/file_time.yaml`, `voigtCONV`
   use in `tests/test_mcp_library.py` (~L342, switch to `gaussCONV`),
   kernel list in `docs/quickstart.md`.
2. [x] `utils/arrays.py`: rework `ConvOperator` + `conv_matrix_operator`
3. [x] `functions/time.py`: private edge-mass companions for gauss,
   expSym, expDecay, expRise, box + `CONV_EDGE_MASS` registry dict.
4. [x] mcp path: `Component.convolve` computes edge masses via
   `CONV_EDGE_MASS`; `add_components` validates companion exists;
   model-load validation of strictly positive kernel-parameter bounds.
5. [x] GIR: `schedule_2d` validates the companion exists (by kernel
   kind); evaluator-side `CONV_EDGE_MASS_DISPATCH` (enum-keyed, mirror of
   `CONV_KERNEL_DISPATCH`); `resolve_param_traces` kind==2 computes
   masses per theta and calls the new apply signature. No callables
   stored on `ScheduledPlan2D`.
6. [x] Tests: broad-Gaussian fine-start regression, asymmetric swap
   detectors, companion-vs-quad consistency, registry completeness,
   retained parity/constant/my_conv/dense-reference checks.
7. [x] Examples: example 21 data regenerated through the edge-mass
   operator (2026-07-11).
8. [x] Benchmark example 01: 2.95 ms/call GIR (was 8.4 ms/call ghost
   scheme; 3x smaller interior matrix).
9. [x] Docs: changelog, archived design note, repo_architecture,
   jax-planning, quickstart, add-function skill updated.
10. [x] `docs/ai/add-function.md`: edge-mass companion checklist items.

## Status log

- 2026-07-11: plan created after edge-truncation defect found in the
  ghost-point scheme (user-reported); kernel drop + CDF design agreed.
- 2026-07-11: implementation complete. Interior-only `(n_t, n_t)` operator
  with analytic edge masses via `CONV_EDGE_MASS` / `CONV_EDGE_MASS_DISPATCH`;
  `voigtCONV`/`lorentzCONV` removed. Full suite green (913 passed); example
  01 benchmark 2.95 ms/call GIR (6.5x vs interpreter). Example 21 data
  regenerated. Release target: 0.11.0 (user).
- 2026-07-11: review feedback round: varying kernel params now require an
  explicit positive lower bound at model load; companions validate
  positivity/finiteness at evaluation (called before the kernel body on
  both paths); `conv_matrix_apply` validates the `y` shape; function
  discovery in `config/functions.py` restricted to module-defined
  functions (imported `erf`/`erfc`/`wofz`/`Callable` no longer leak into
  the registry); truth YAMLs of examples 01/03/04 given bounded conv
  params; changelog SciPy wording corrected (`erfc` remains).
