# Repo Architecture

Orientation guide for anyone (human or LLM) modifying the trspecfit codebase.
Start here before writing new code — the goal is to avoid reinventing wheels
and to put new code in the right layer.

For what model combinations are actually supported, see
[supported_models.md](supported_models.md).
For how the compiled 2D evaluator was designed, see
[lowered_evaluator.md](lowered_evaluator.md).

## Two-layer design

trspecfit has two distinct layers, and **the distinction matters for every
change you make**:

1. **Authoring / user-facing layer** (`mcp.py`, `trspecfit.py`, `simulator.py`,
   YAML parsing in `utils/parsing.py`). Optimized for **readability, clear
   error messages, and interactive exploration**. Performance is secondary —
   these modules run once per model, not once per optimizer iteration.
   Users poke at objects here; tracebacks land here.

2. **Compiled hot-path layer** (`graph_ir.py`, `eval_1d.py`, `eval_2d.py`,
   and the numeric bodies in `functions/`). Optimized for **performance**.
   These run inside the residual loop, possibly millions of times per fit.
   Array-oriented, no Python dicts/strings in the inner loops, no object
   attribute lookups where a packed array will do. See
   [lowered_evaluator.md](lowered_evaluator.md) for the motivation.

`spectra.py` is the bridge — it hands the current parameter vector from
`fitlib` to the compiled evaluator.

New features are usually added first in the authoring layer (mcp), then
lowered into the compiled layer once the semantics are stable.

## Top-level modules (`src/trspecfit/`)

### `trspecfit.py` — user entry point

Defines `Project` and `File`. This is the API most users see in notebooks.
`Project` holds project-wide config (plot defaults, file I/O formats, fit
settings). `File` wraps a single dataset (1D or 2D), its axes, and any
number of named `Model`s for comparison. Methods: `load_model`,
`add_time_dependence`, `add_par_profile`, `set_fit_limits`, `fit_baseline`,
`fit_slice_by_slice`, `fit_2d`. **Always use these public methods in tests
and examples** — they carry out validation and axis propagation that
direct `Model` construction skips. Authoring-layer; keep readable.

### `mcp.py` — Model / Component / Par system

Hierarchical model construction: `Par` → `Component` → `Model`. Plus two
`Model` subclasses: `Dynamics` (time-dependent behavior, multi-cycle,
convolution kernels) and `Profile` (auxiliary-axis parameter variation).
Handles parameter naming (`{model}_{component}_{param}`), expressions via
`asteval`, and a slow reference evaluator used by the simulator and for
cross-checking. **Keep this file human-readable with no regard for
performance** — hot-path evaluation lives in the compiled layer. User
tracebacks and interactive debugging go through here.

### `graph_ir.py` — compiled intermediate representation

Lowers an mcp `Model` into a typed DAG (`NodeKind`, `EdgeKind`, domain
classification) and then into a packed `ScheduledPlan1D` / `ScheduledPlan2D`:
flat numpy arrays of instructions, parameter indices, and RPN expression
programs. No Python objects, strings, or dicts in the execution data — the
whole plan is array-oriented so the evaluator can run in a tight loop.
Also contains compile-time gates (`can_lower_2d`, etc.) that decide
whether a given model is representable in the v1 fast path.

### `eval_1d.py`, `eval_2d.py` — hot-path evaluators

Pure functions of the form `evaluate_Nd(plan, theta) -> spectrum`. Consume
the `ScheduledPlan` produced by `graph_ir` and the current parameter
vector. `eval_2d` broadcasts peak functions with `(n_time, 1)` parameters
against `(1, n_energy)` energy. Dynamics, convolution, and profile
dispatch tables live here (`DYNAMICS_DISPATCH`, `CONV_KERNEL_DISPATCH`).
**Performance-critical — prefer array operations, avoid Python-level
branching on model structure (the plan already captured it).**

### `spectra.py` — evaluator bridge

Thin module that the fitting engine calls on every residual evaluation.
`fit_model_gir` (the default) dispatches to the compiled evaluator
(`evaluate_1d` / `evaluate_2d`) when a `ScheduledPlan` is present, and
falls back to `fit_model_mcp` — the mcp reference evaluator — when the
model is not lowerable or when 1D component-wise spectra are requested
for plotting. Users can swap in a custom spectrum function via
`Project.spec_fun_str`.

### `fitlib.py` — lmfit wrappers, CI, MCMC, plotting

The fitting machinery: residual function, `fit_wrapper` (global + local
solvers), confidence intervals via `lmfit.conf_interval`, MCMC via
`lmfit.emcee`, and the 1D/2D fit-result plotting (`plt_fit_res_1d`,
`plt_fit_res_2d`). Internal module — method docstrings stay minimal,
module-level doc carries the weight.

### `simulator.py` — synthetic data generation

User-facing `Simulator` class. Generates 1D/2D spectra from a `Model`
with noise (Poisson, Gaussian, none) and detector type (analog, photon
counting). Supports `simulate_n` (replicates), `ParameterSweep`
integration for ML training-data generation, and HDF5 export. Use here
for testing, fit-pipeline validation, identifiability studies, and
training-data synthesis.

## `config/` — runtime configuration

### `config/functions.py`

Introspects `functions/{energy,time,profile}.py` to discover which function
names are available. Provides `all_functions`, `background_functions`,
`convolution_functions`, `energy_functions`, `numbering_exceptions`,
`get_function_parameters`. The YAML parser and mcp use this to decide
which components can be numbered (`GLP_01`, `GLP_02`, ...) and which are
singletons (backgrounds, convolutions). **If you add a new background
function, register it here.**

### `config/plot.py`

`PlotConfig` dataclass. The single source of truth for plot appearance
(axis labels/limits/direction, colormaps, DPI, etc.). Inheritance chain:
Project defaults → File overrides → Model inherits → per-call overrides.
Use `PlotConfig` whenever you add a plotting function — do not invent new
keyword arguments for styling.

## `functions/` — the function registry

Three flat modules of numeric functions. **Function names and parameter
names deliberately use CamelCase / PascalCase** (not snake_case) because
`_` is the component-ID delimiter (`{model}_{component}_{param}`). See
`CLAUDE.md` at the repo root for the full naming rule. These
functions are called directly from the compiled evaluators — they should
be fast, numpy-only, and free of Python-level allocation where possible.

### `functions/energy.py`

Peak and background shapes used as spectral components. Examples: `GLP`,
`Gauss`, `Voigt`, `DoniachSunjic`, `Offset`, `LinBack`, `Shirley`. Peak
functions have signature `func(x, par1, par2, ...)`; background functions
have signature `func(x, par, spectrum=None)` so they can depend on the
current peak sum (e.g. Shirley). Add new peak or background shapes here.

### `functions/time.py`

Dynamics and convolution kernels. Dynamics functions (e.g. `expFun`,
`sinFun`, `linFun`, `erfFun`, `sqrtFun`) share signature
`func(t, par1, ..., t0, y0)` with the invariant `f(t < t0) = 0`.
Convolution kernels are named `funcCONV` (e.g. `gaussCONV`) with a
companion `funcCONV_kernel_width()` returning the kernel-width multiplier.
Add new time-domain behavior or IRF kernels here.

### `functions/profile.py`

Auxiliary-axis profile functions, required to start with a `p` prefix
(e.g. `pExpDecay`, `pLinear`, `pGauss`). Signature `func(x, par1, ...)`
where `x` is the auxiliary axis (depth, position, fluence, ...). Attached
to a parameter via `File.add_par_profile`; evaluation samples the
profiled parameter across the aux axis and averages. Add new profile
shapes here.

## `utils/` — reusable helpers (check here before inventing)

### `utils/arrays.py`

Array/numeric helpers. Notably `my_conv` (used by the 2D evaluator for
IRF convolution), `format_float_scientific` (fixed-width scientific
notation), `oom` (order-of-magnitude), running averages, sign-change
detection, angular normalization. Use `my_conv` rather than rolling your
own `scipy.signal.convolve` wrapper.

### `utils/hdf5.py`

Typed HDF5 helpers. `require_group`, `require_dataset`, `json_loads_attr`.
All HDF5 I/O in the repo should go through these rather than raw
`h5py` calls — they normalize attribute types across numpy/bytes/str.

### `utils/lmfit.py`

lmfit-parameter plumbing. Parameter construction, extraction, conversion
to pandas DataFrames, MCMC config helpers, and the `VARY_LEVELS` /
`_vary_to_bool` / `vary_to_level` machinery for the
`static`/`file`/`project` vary hierarchy. Any new lmfit interop belongs
here rather than in `trspecfit.py` or `fitlib.py`.

### `utils/parsing.py`

YAML model parsing. `ModelValidationError`, the custom
`_ComponentNumberingConstructor` that auto-numbers duplicate YAML keys
(`GLP` → `GLP_01`, `GLP_02`), and the validation that enforces which
function names and parameters are legal. Extend here — not in mcp — when
adding new YAML syntax.

### `utils/plot.py`

Generic matplotlib helpers used by the library and user notebooks:
1D/2D data plotting, image loading (`load_plot`, `load_plot_grid`) for
embedding saved figures in reports, axis formatting utilities. All
plotting functions take a `PlotConfig`. Specialized plotting (e.g. fit
residuals) lives in `fitlib.py`, not here.

### `utils/sweep.py`

`ParameterSweep` (grid / random / uniform / normal sampling) and
`SweepDataset` (inspection of generated HDF5 datasets). Used by
`Simulator` for ML training-data generation. Grows as new sampling
strategies are needed.

## Typical execution flow

For a 2D fit via `File.fit_2d`:

1. `File.fit_2d` gathers the target `Model`, axes, fit limits.
2. `Model` is lowered to a `ScheduledPlan2D` via `graph_ir.schedule_2d`
   once, up front.
3. `fitlib.fit_wrapper` runs lmfit; each residual call goes
   `residual_fun` → `spectra.fit_model_gir` → `eval_2d.evaluate_2d(plan, theta)`.
4. `evaluate_2d` produces the model spectrum using only the plan arrays
   and the parameter vector — no mcp objects touched in the hot path.
5. After the fit: confidence intervals / MCMC / plotting run in
   `fitlib`, and results are exported via `Project` paths.

Models outside the current compiled support set (see
[supported_models.md](supported_models.md)) fall back to the mcp reference
evaluator. New features are generally prototyped on that slow path first.

## Where to put new code — quick guide

- **New energy / time / profile function** → implement it in `functions/{energy,time,profile}.py`; for the full checklist (tests, registration, and GIR follow-up when needed), use [../ai/add-function.md](../ai/add-function.md).
- **New YAML keyword / syntax** → `utils/parsing.py` + validation.
- **New user-facing method on a file** → `File` in `trspecfit.py`.
- **New model composition rule** → mcp first; update `supported_models.md`; lower into `graph_ir` once stable.
- **New plot style / axis logic** → `utils/plot.py`, driven by `PlotConfig`.
- **New fit-result post-processing (CI, exports, plots)** → `fitlib.py`.
- **New simulator feature / sampling strategy** → `simulator.py` / `utils/sweep.py`.
- **New HDF5 I/O** → go through `utils/hdf5.py` helpers.
- **Performance optimization of an existing feature** → lower into `graph_ir` / `eval_*`. Do **not** optimize mcp.
