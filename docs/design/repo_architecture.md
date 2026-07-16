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

### `eval_jax.py` — experimental JAX backend (optional `[jax]` extra)

Jitted mirror of the 2D evaluator plus an analytic Jacobian
(`make_evaluator_2d_jax`, `make_jacobian_2d_jax`), compiled per plan by
trace-time unrolling of the schedule arrays. Gated by
`graph_ir.can_lower_jax_2d` (at most as wide as `can_lower_2d`, so
rejected graphs fall back to the compiled NumPy path, never straight to
the interpreter). Selected via `Project.spec_fun_str = "fit_model_jax"`;
the Jacobian reaches lmfit's leastsq through `fitlib.jacobian_fun`
(`Dfun`). Voigt uses a Weideman rational `wofz` approximation instead
of SciPy's. For project-level shared fits, `make_project_evaluator_2d_jax`
/ `make_project_jacobian_2d_jax` fuse all per-file plans into one jitted
program (windowed, flattened, concatenated; `jacfwd` for the joint
Jacobian) — see
[project-level-fits.md](project-level-fits.md).

### `spectra.py` — evaluator bridge

Thin module that the fitting engine calls on every residual evaluation.
`fit_model_gir` (the default) dispatches to the compiled evaluator
(`evaluate_1d` / `evaluate_2d`) when a `ScheduledPlan` is present, and
falls back to `fit_model_mcp` — the mcp reference evaluator — when the
model is not lowerable or when 1D component-wise spectra are requested
for plotting. Users can swap in a custom spectrum function via
`Project.spec_fun_str`. Project-level shared fits evaluate through
`fit_project_mcp` (interpreter, name-based parameter distribution) or
`fit_project_jax` (fused jitted evaluator; `pack_project_theta`
converts the combined-parameter mapping into gather index arrays at
fit setup).

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

### `fit_results.py` — completed-fit inspection / comparison

User-facing `FitResults` class — the immutable view over a list of
`SavedFitSlot`. Two construction paths: `FitResults.load(path)` for
loaded archives and the `Project.results` property for in-session work.
A `FitResults` is frozen at construction (the underlying slot list is
copied), so `r1 = p.results; <run another fit>; r2 = p.results` gives
two distinct snapshots — `r1` does not see the new slot. Query API:
`find` / `get` / `files` / `models` / iteration. Comparison:
`compare_models` (returns a metrics DataFrame; refuses to compare
slots whose `observed_sha256` differs on the same `(file, fit_type)`)
and `plot_residuals` (smoke-test-grade panels, no energy/time labels —
slots don't carry parent-file axes). The save/export side lives in
`utils/fit_io.py`; this module is read-only on top of those slots.

## Fit results: save / export / load architecture

The fit-output persistence layer is **slot-driven**, not model-walking.
Once a fit completes, the result is captured eagerly into a
`SavedFitSlot` (one per `(file, model, fit_type, selection)`); everything
downstream — save, export, in-session comparison, archive load — reads
slots, never live `Model.result`.

```
fit_baseline / fit_spectrum /                ┌──────────────────────────┐
fit_slice_by_slice / fit_2d  ────► result ───►  _slot_from_<fit_type>   │
                                              │   (eager extraction in  │
                                              │    utils/fit_io.py)     │
                                              └────────────┬────────────┘
                                                           │
                                                           ▼
                                            Project._fit_history (append-only log)
                                                           │
                                       ┌───────────────────┼──────────────────────┐
                                       ▼                   ▼                      ▼
                       Project.results (wrapper)  Project.save_fits        Project.export_fits
                                                  (filter + snapshot       (filter + CSV/PNG
                                                   collapse → HDF5)         tree)

HDF5 archive ────► reader ────► FitResults (FitResults.load / Project.load_fits)
                                Independent of _fit_history; never merged in.
```

**Two different I/O directions, two different surfaces:**

- **Save / load** (round-trippable): `Project.save_fits(path)` →
  HDF5 archive; `FitResults.load(path)` (or the equivalent
  `Project.load_fits(path)` convenience) deserializes back. Schema in
  [fit_archive_schema.md](fit_archive_schema.md). Append-mode by default;
  slot-scoped overwrite.
- **Export** (one-way): `Project.export_fits(path, format="csv")` →
  directory of human-readable CSVs and PNGs. No `load` counterpart —
  round-tripping fits is HDF5's job.

`File.save_fit` / `File.export_fit` / `File.compare_models` are
one-line delegates to the corresponding `Project.*` / `FitResults.*`
methods. There is no `File.load_fit`: load is path-scoped, not file-scoped.

Auto-export inside `fit_slice_by_slice` / `fit_2d` / `Project.fit_2d`
routes through the same slot exporter as explicit `export_fits` calls,
writing the grouped `{path_results}/{file}/{model}__{fit_type}/` tree
(unless disabled via `Project.auto_export = False`, default `True`).
Interactive display (`show_output >= 1`) renders inline from the
just-captured `SavedFitSlot` via the `File._display_*` helpers — the
figure a user sees is built from the same arrays the export saves.
Fit-time diagnostics (per-stage parameter CSVs from `fitlib.fit_wrapper`,
SbS per-slice CSVs/PNGs) are separate from the results export and land
under `File.model_path()` (`{path_results}/{file}/{fit_type}/{model}/`).
The pre-0.14 legacy savers (`save_sbs_fit` / `save_2d_fit` and their
`_save_*_fit_legacy` impls, which wrote a flat layout) were removed.

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
`sinFun`, `linFun`, `erfFun`, `sqrtFun`, `stepFun`) share signature
`func(t, par1, ..., t0)` with the invariant `f(t < t0) = 0`; constant
offsets are their own additive component (`stepFun`, or `erfFun` for a
broadened onset) rather than a `y0` parameter on every function.
Convolution kernels are named `funcCONV` (e.g. `gaussCONV`) and must be
elementwise in their first argument: the kernel-matrix convolution
evaluates them on a 2D matrix of time differences
(`utils.arrays.conv_matrix_operator`). Every kernel additionally
registers a private edge-mass companion in `CONV_EDGE_MASS` (exact
analytic exterior masses of the kernel body, for edge-value padding);
kernels without one are rejected at model validation and scheduling.
Add new time-domain behavior or IRF kernels here (see
`docs/ai/add-function.md` for the full checklist).

### `functions/profile.py`

Auxiliary-axis profile functions, required to start with a `p` prefix
(e.g. `pExpDecay`, `pLinear`, `pGauss`). Signature `func(x, par1, ...)`
where `x` is the auxiliary axis (depth, position, fluence, ...). Attached
to a parameter via `File.add_par_profile`; evaluation samples the
profiled parameter across the aux axis and averages. Add new profile
shapes here.

## `utils/` — reusable helpers (check here before inventing)

### `utils/arrays.py`

Array/numeric helpers. Notably the kernel-matrix convolution pair
`conv_matrix_operator` / `conv_matrix_apply` (used by both the mcp and
GIR evaluators for IRF convolution — exact on non-uniform time axes,
static shapes per theta, exterior kernel mass handled analytically via
per-kernel edge-mass companions), `my_conv` (padded 1D convolution, now only
backing `running_mean`), `format_float_scientific` (fixed-width
scientific notation), `oom` (order-of-magnitude), running averages,
sign-change detection, angular normalization. Use these rather than
rolling your own `scipy.signal.convolve` wrapper.

### `utils/hdf5.py`

Typed HDF5 helpers. `require_group`, `require_dataset`, `json_loads_attr`.
All HDF5 *reads* should go through these rather than raw subscripting —
`require_*` narrow `Group | Dataset` lookups with a clear error naming the
archive path, and `json_loads_attr` normalizes JSON attributes across
numpy/bytes/str. Write-side calls (`h5py.File(...)`, `create_group`,
`create_dataset`, attribute assignment) have no wrapper and use `h5py`
directly.

### `utils/fit_io.py`

Fit-results persistence. Owns the `SavedProject` / `SavedFile` /
`SavedFitSlot` dataclasses (the on-disk data model), the four
per-fit-type slot extractors (`_slot_from_baseline`,
`_slot_from_spectrum`, `_slot_from_sbs`, `_slot_from_2d` — all called
once at fit completion with copied snapshot args, never live `Model`
references), the identity helpers (`compute_file_fingerprint`,
`compute_history_key`, `compute_archive_slot_key`,
`build_selection_json`, `compute_observed_sha256`), the
snapshot-collapse helper (`collapse_history_to_snapshot`), and the
HDF5 reader/writer (`read_archive`, `write_archive`) plus the CSV/PNG
exporter (`write_csv_export`). The `SavedFitSlot` is the **single
source of truth for completed-fit state** — neither `Model` nor `File`
carries observed/fit/metrics. New persistence work lands here, not in
`fitlib` or `trspecfit.py`. See `docs/design/fit_archive_schema.md`
for the on-disk schema.

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
   `fitlib`. The completed result is then captured eagerly into a
   `SavedFitSlot` via `utils/fit_io.py` and appended to
   `Project._fit_history`; that slot is what `Project.results`,
   `Project.save_fits`, and `Project.export_fits` operate on. Live
   `Model.result` is never re-read by these paths — see
   "Fit results: save / export / load architecture" above.

Models outside the current compiled support set (see
[supported_models.md](supported_models.md)) fall back to the mcp reference
evaluator. New features are generally prototyped on that slow path first.

## Where to put new code — quick guide

- **New energy / time / profile function** → implement it in `functions/{energy,time,profile}.py`; for the full checklist (tests, registration, and GIR follow-up when needed), use [../ai/add-function.md](../ai/add-function.md).
- **New YAML keyword / syntax** → `utils/parsing.py` + validation.
- **New user-facing method on a file** → `File` in `trspecfit.py`.
- **New model composition rule** → mcp first; update `supported_models.md`; lower into `graph_ir` once stable.
- **New plot style / axis logic** → `utils/plot.py`, driven by `PlotConfig`.
- **New fit-result post-processing (CI, MCMC, in-fit plots)** → `fitlib.py`.
- **New fit-archive field, exporter format, or comparison metric** → `utils/fit_io.py` (data model + writer/reader + CSV exporter) and `fit_results.py` (query / `compare_models`). Slot extraction stays in `utils/fit_io.py`; the four `_append_<fit_type>_slot` call sites in `trspecfit.py` should not be replicated elsewhere.
- **New simulator feature / sampling strategy** → `simulator.py` / `utils/sweep.py`.
- **New HDF5 reads** → go through `utils/hdf5.py` helpers (writes use `h5py` directly).
- **Performance optimization of an existing feature** → lower into `graph_ir` / `eval_*`. Do **not** optimize mcp.
