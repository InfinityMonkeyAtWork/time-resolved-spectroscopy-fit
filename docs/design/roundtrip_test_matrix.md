# Roundtrip Test Matrix

This document tracks the intended roundtrip-test surface for single-file fits.
It is meant to answer three questions at a glance:

1. Which workflow API is under test?
2. Which backend expectation is under test?
3. Which supported model family is under test?

Project-level fitting and MCMC/parallel execution are important too, but they
are tracked as secondary dimensions so the main matrix stays readable.

## Axes

### Workflow axis

- `B`: `File.fit_baseline()`
- `Sp`: `File.fit_spectrum()`
- `SbS`: `File.fit_slice_by_slice()`
- `2D`: `File.fit_2d()`

### Scope axis

- `SF`: single-file workflows
- `P`: project-level workflows via `Project.fit_2d()`

### Backend axis

- `M`: force MCP with `project.spec_fun_str = "fit_model_mcp"` and recover truth
- `G`: run default compiled dispatch and recover truth on the GIR path
- `C`: assert `delta(MCP, GIR) = 0`

For `C`, prefer `fit_model_compare` when the workflow supports it. Otherwise use
direct parity checks on residuals or evaluated outputs.

### Cell notation

- `M/G/C`: all three test types should exist for that workflow/model-family pair
- `-`: not applicable for that pair

### Execution-mode qualifiers

These are not a full extra matrix axis; they are required focused variants on
top of the main matrix:

- `Opt`: normal optimizer-based fit, no MCMC
- `MC1`: MCMC enabled with `workers=1`
- `MC2`: MCMC enabled with `workers=2`
- `W1`: explicit serial execution for a workflow with worker support
- `W2`: parallel execution with `workers=2`

Use `2` as the standard parallel test setting. We usually care about
`1` versus `>1`, not about many-worker scaling in CI.

## Canonical model families

Each row below should have one canonical fixture. Closely related variants can
be parameterized under the same row instead of getting separate rows.

| ID | Model family | Representative fixture(s) |
| --- | --- | --- |
| `F1` | Plain energy model | `single_glp`, `glp_only` |
| `F2` | Static expressions in energy model: direct refs, fan-out, forward refs | `two_glp_expr_amplitude`, `expression_fan_out`, `energy_expression_forward_reference`, `glp_expression` |
| `F3` | Top-level standard dynamics | `single_glp` + `MonoExpPos` |
| `F4` | Top-level dynamics with IRF / convolution | `single_glp` + `MonoExpPosIRF` and other lowerable IRF kernels |
| `F5` | Top-level subcycle / multi-cycle dynamics | `single_glp` + `["ModelNone", "MonoExpNeg", "MonoExpPosExpr"]`, `frequency=10` |
| `F6` | Top-level profile only | `single_gauss` + `roundtrip_pLinear_x0` / `roundtrip_pExpDecay_A` |
| `F7` | Top-level profile plus separate top-level dynamics on another parameter | `single_gauss` + profile on `Gauss_01_A` + dynamics on `Gauss_01_x0` |
| `F8` | Profile-internal dynamics | `single_gauss` + profile on `Gauss_01_A` + dynamics on `Gauss_01_A_pExpDecay_01_A` |
| `F9` | Expression parameter referencing a top-level time-dependent base parameter | `two_glp_expr_amplitude` + dynamics on `GLP_01_A` |
| `F10` | Expression parameter referencing a top-level profiled base parameter | `two_glp_expr_amplitude` + profile on `GLP_01_A` |
| `F11` | Expression parameter referencing a profiled parameter whose profile internals are time-dependent | `two_glp_expr_amplitude` + profile on `GLP_01_A` + dynamics on `GLP_01_A_pExpDecay_01_A` |
| `F12` | Mixed expression referencing both profiled and time-dependent base parameters | `two_glp_mixed_profile_dynamics` with profile on `GLP_01_A` and dynamics on `GLP_01_x0` |

## Target matrix

Scope: `SF` (single-file)

| Family | B | Sp | SbS | 2D | Notes |
| --- | --- | --- | --- | --- | --- |
| `F1` Plain energy | `M/G/C` | `M/G/C` | `M/G/C` | `-` | Core 1D workflow coverage |
| `F2` Static expressions | `M/G/C` | `M/G/C` | `M/G/C` | `-` | Include at least one direct-ref case and one fan-out or forward-ref case |
| `F3` Standard dynamics | `-` | `-` | `-` | `M/G/C` | Core 2D dynamic family |
| `F4` IRF dynamics | `-` | `-` | `-` | `M/G/C` | One canonical roundtrip plus parametrized parity across kernels |
| `F5` Subcycle dynamics | `-` | `-` | `-` | `M/G/C` | Important for multi-cycle indexing and expression prefixing |
| `F6` Profile only | `M/G/C` | `M/G/C` | `M/G/C` | `-` | Covers aux-axis plumbing in 1D APIs |
| `F7` Profile + separate dynamics | `-` | `-` | `-` | `M/G/C` | Top-level mixed feature case |
| `F8` Profile-internal dynamics | `-` | `-` | `-` | `M/G/C` | Single-cycle only |
| `F9` Expr -> time-dependent base par | `-` | `-` | `-` | `M/G/C` | High-value bug class for update ordering and pickling |
| `F10` Expr -> profiled base par | `M/G/C` | `M/G/C` | `M/G/C` | `-` | Expression namespace must see profiled values |
| `F11` Expr -> profiled base par with profile-internal dynamics | `-` | `-` | `-` | `M/G/C` | Single-cycle only |
| `F12` Mixed expr(profile + dynamics refs) | `-` | `-` | `-` | `M/G/C` | Stress case for combined namespace resolution |

## Project-level matrix

Scope: `P` (`Project.fit_2d()`)

Project-level fitting should be tracked separately because it is currently
wired through `fit_project_mcp`, so the single-file GIR/MCP expectations do not
apply cleanly yet.

| Family | 2D | Notes |
| --- | --- | --- |
| `PF1` Shared plain dynamics across files | `M` | Current core project roundtrip surface |
| `PF2` Project-level expressions | `M` | Includes file/project prefix rewriting and shared refs |
| `PF3` Shared dynamics with IRF | `M` | Covered with `BiExpProject` + `gaussCONV` |
| `PF4` Shared subcycle dynamics | `M` | Add once project fixtures exist |

Future:

- if project-level GIR lands, upgrade applicable cells from `M` to `M/G/C`
- until then, do not force fake GIR coverage into the project matrix

## MCMC and worker policy

Yes: MCMC should be tracked.

Yes: worker mode should be tracked anywhere the code can execute differently
between serial and parallel paths.

But neither should multiply every cell in the main matrix. Instead use focused
requirements:

### MCMC requirements

MCMC is a second-layer contract on top of the clean optimizer roundtrips.

Minimum MCMC set:

- `MC1`: one canonical `B` test on `F1`
- `MC2`: one canonical `B` test on `F1`
- `MC2`: one expression-sensitive case, preferably `F9` or `F10`
- `MC2`: one 2D varying case, preferably `F3` or `F8`

Rationale:

- `MC1` checks that MCMC itself still works
- `MC2` checks pickling / process-boundary behavior
- expression-heavy and nested-model cases are the highest-value bug surfaces

### SbS worker requirements

Yes, `SbS` should distinguish `W1` and `W2` because `n_workers=1` uses the
serial path and `n_workers>1` crosses a process boundary.

Current status:

- the main `SbS` matrix runs with `n_workers=1`
- focused `W2` tests cover `F1` and profile-bearing `F6` with `n_workers=2`

Future requirement if worker-specific risk grows:

- add a more expression-heavy `W2` `SbS` case, likely `F2`, if process-boundary
  risk shows up beyond the existing plain/profile cases

### Project worker requirements

For project-level fits, add worker variants only when project execution gains a
parallel path that is semantically different from serial execution.

## Practical rule

Use this rule to decide whether a new dimension deserves explicit tracking:

- add it as a full matrix axis only if it changes almost every cell
- otherwise add it as a focused secondary requirement

By that rule:

- project-level fits: yes, but separate matrix
- MCMC: yes, as focused secondary coverage
- `workers=1` vs `workers=2`: yes, but only for APIs that actually expose
  worker-dependent behavior

## Minimum test shape per cell

For each required cell, the minimum useful test is:

- simulate noiseless data from a truth model
- fit through the target workflow API
- assert recovered non-expression parameters match truth
- for `C`, assert MCP and GIR agree exactly or within a tight tolerance

For MCMC-focused cells, the minimum useful test is:

- run the fit with `mc_settings.use_mc=1`
- assert no crash for `MC1`
- assert no crash and no pickling/serialization failure for `MC2`
- when runtime allows, also assert basic parameter recovery or constraint
  preservation

Noisy roundtrip tests are still valuable, but should be a second layer. The
clean matrix above is the baseline contract.

## Current snapshot

This is the current high-level state of the suite, not a substitute for the
table above.

- Covered today:
  - the full single-file clean matrix above for `M/G/C`
  - `F2` variants for direct, fan-out, and forward-reference expressions
  - noisy second-layer checks for `F3`, `F6`, and `F8` on the GIR path
  - focused MCMC checks for `MC1`, `MC2`, expression-sensitive `MC2`, and 2D `MC2`
  - focused `W2` coverage for plain and profile-bearing `fit_slice_by_slice()`
  - project-level `M` roundtrips for `PF1`, `PF2`, and `PF3`

- Thin or missing today:
  - project-level `PF4` shared subcycle dynamics
  - project-level `G/C` coverage, because project fitting is still MCP-only
  - expression-heavy `W2` coverage for `fit_slice_by_slice()`
  - MCMC assertions beyond no-crash / process-boundary coverage
  - exhaustive noisy coverage, intentionally kept out of the main matrix

## Suggested implementation order

The original single-file matrix is implemented. Highest-value next steps:

1. Add `PF4` once a shared project-subcycle fixture exists.
2. Add a focused expression-heavy `W2` `SbS` test if process-boundary risk shows
   up beyond the existing plain/profile cases.
3. Add lightweight recovery or constraint-preservation assertions to focused
   MCMC tests when runtime allows.
4. Upgrade project-level cells from `M` to `M/G/C` if project-level GIR lands.

## Non-goals

This matrix does not try to track:

- invalid / explicitly unsupported model combinations
- low-level evaluator unit tests
- plotting-only behavior

Those should stay in their existing focused tests.
