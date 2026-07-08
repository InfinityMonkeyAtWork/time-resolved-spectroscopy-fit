# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com),
and this project adheres to [Semantic Versioning](https://semver.org/).
This file is maintained using the shared changelog workflow in
[`docs/ai/changelog.md`](docs/ai/changelog.md).

## [0.10.0] - 2026-07-08

### Added

- **`stepFun(t, A, t0)` dynamics primitive**: a causal step (0 before `t0`, `A` after). Since dynamics components combine by addition, `stepFun` is the dedicated way to model baselines and plateaus (e.g. `expFun` + `stepFun` sharing `t0` gives a decay to a nonzero plateau); `erfFun` is its Gaussian-broadened counterpart. Registered on both the mcp and compiled (GIR) evaluation paths.

### Changed

- **Breaking: `y0` removed from all dynamics functions** (`linFun`, `expFun`, `sinFun`, `sinDivX`, `erfFun`, `sqrtFun`). The per-function offset broke the causality convention `f(t < t0) = 0` inconsistently (`sqrtFun`/`erfFun` leaked `y0` before `t0`; the others clamped to 0) and duplicated what an additive component expresses directly. Migration for model YAMLs: delete `y0:` lines that were `0` (the common case); replace a nonzero `y0` with a `stepFun` component sharing the function's `t0`. Stale `y0:` entries now fail model validation with a parameter-count error. The energy-domain `Offset(x, y0)` background is unaffected.
- **Breaking: `Voigt` amplitude is now grid-independent.** The profile was normalized by its maximum over the sampled energy window, so the value at a fixed energy depended on the window/grid, and a peak center pushed outside the fit window rescaled the in-window tail up to amplitude `A`. Normalization now uses the analytic peak value `wofz(i·W/(2·SD·√2))` at `dx = 0`; `A` is the true peak height regardless of the sampled grid. Fitted `A` values may shift slightly relative to earlier releases (the two normalizations agree only when the sampled grid contains the exact peak). `voigtCONV` is unchanged (its kernel axis is symmetric around the peak, where max-over-grid is exact).

### Fixed

- **`expRiseCONV` docstring claimed the kernel is causal.** It is deliberately anti-causal — the mirror of `expDecayCONV`, nonzero only for `x ≤ 0`, so the convolved response rises before the excitation and saturates at `t0`. The docstring now says so.
- **`boxCONV` docstring claimed "smooth edges"** — the kernel is a hard `|x| ≤ width/2` threshold; docstring corrected.
- **`GLS`/`GLP` docstrings** now state the valid mixing range `m ∈ [0, 1]` (for `GLP`, `m < 0` makes the denominator `1 + 4·m·u²` cross zero, producing NaN/inf).
- **`DS` docstring** now documents that the asymmetric tail extends toward `x < x0` (correct for a kinetic-energy axis; mirrored on a binding-energy axis).

## [0.9.3] - 2026-06-29

### Added

- **Live uncertainty accessors.** New `File.get_correlations()`, `File.get_conf_intervals()`, and `File.get_mcmc()` read uncertainty results directly off `model.result`, so workflows no longer index raw `result[1..4]` tuples. `get_mcmc()` returns a `ulmfit.MCMCResult` dataclass exposing `.table` (best-fit + quantile CIs), `.flatchain` (for `corner.corner`), and `.acceptance_fraction`. These are live-only views of the most recent fit; persisting `correl` / `acceptance_fraction` into saved slots is deferred.
- **MCMC noise-model controls.** `ulmfit.MC()` gains `sigma_ini` / `sigma_min` / `sigma_max` knobs that seed and bound the `__lnsigma` nuisance parameter sampled by `lmfit.emcee`, letting the noise level be estimated alongside the model parameters.

### Changed

- **Examples reorganized around user-workflow tracks.** The suite now progresses from basic fitting through model comparison, save/load/export, uncertainty analysis, and multi-file workflows, with clearer numbering and filenames. See `examples/fitting_workflows/README.md` for the legend and `docs/design/examples_upgrade.md` for the design.
- **Basic and post-fit workflows split more cleanly.** `01_basic_fitting` now focuses on first-fit results (`get_fit_results` + plots); comparison, persistence/export, and uncertainty analysis live in dedicated follow-up notebooks (`10`–`12`).
- **Multi-file independent fitting example added.** `20_multi_file_independent_fit` bridges single-file fitting and shared-parameter project fitting (`21_multi_file_shared_fit`).
- Benchmark harness (`docs/ai/benchmark.md`, `.claude/skills/benchmark/benchmark_gir.py`) updated for the new example numbering: batch mode (`--example 0`) iterates examples 1–4; project-level fitting now lives at `21_multi_file_shared_fit` and remains outside the harness.
- **MCMC quantile table / `mcmc.ci` no longer include fixed parameters.** `File.get_mcmc().table` and the persisted `mcmc.ci` previously emitted a `-1` sentinel row for every fixed parameter, which rendered as if it were a real quantile. The emcee CI build now iterates only sampled parameters (varying params + `__lnsigma`); fixed parameters get no row, mirroring `lmfit.conf_interval`. Old archives still load.
- **Convolution-placement validation relaxed for single-component models.** The "convolution cannot be the last component" rule now applies only to multi-component models, so a conv-only global dynamics element (e.g. `['IRF', ...]` in `add_time_dependence`) is accepted as documented in `Dynamics.set_frequency`.
- Plot-title time values are now formatted with a shared helper for consistent precision/units across fit plots; vertical axis labels and the Slice-by-Slice / 2D fit-result display were cleaned up.

### Fixed

- **`__lnsigma` leaked into the stored leastsq result after an MCMC fit.** `fit_wrapper` added the `__lnsigma` nuisance parameter to the live `par_fin.params` in place, so it persisted into `result[1]` (the leastsq result) after every MCMC run. The parameter set is now deep-copied before `__lnsigma` is added — emcee gets the copy, `result[1]` stays model-only. Was display-only (a spurious `__lnsigma` row in `display(result[1].params)` and `get_fit_results("sbs")`); metrics and save/export paths were already filtered.
- **Slice-by-Slice parallelism (`n_workers > 1`) failed under `%run notebook.ipynb`** with a `BrokenProcessPool` — spawn workers re-ran the notebook JSON as `__main__` and died. `utils.sbs.sanitized_spawn_main()` now hides a non-`.py` `__main__.__file__` for the pool's lifetime so workers start from the installed model only.
- **Non-numeric parameter values now raise a clear error.** Model-load value validation silently skipped its numeric/bounds checks when the first entry of a parameter info list was not an int/float, so a stray non-numeric scalar slipped past and failed cryptically downstream. It now raises `ModelValidationError` (hinting at the single-element `["expr"]` form for the expression case). Expressions are unaffected — they are exclusively the 1-element string form.

## [0.9.0] - 2026-05-27

### Added

- **Fit-results archive — save, reload, and compare fits.** Every completed `File.fit_baseline` / `fit_spectrum` / `fit_slice_by_slice` / `fit_2d` is now captured as a slot in an in-memory fit history, surfaced as a read-only `FitResults` view through the new `Project.results` property (so refits no longer clobber earlier results the way `File.model_*.result` did). `Project.save_fits(path)` persists the current snapshot to a self-contained HDF5 archive (default `./fit_results/<project>.fit.h5`; stores raw data, per-slot `observed` + `fit` on the actual fit grid, metrics, and full fit-view identity), and `FitResults.load(path)` / `Project.load_fits(path)` reload it — the former with no live `Project` needed. `FitResults` (now a top-level export alongside `File`, `Project`, `Simulator`, `PlotConfig`) provides `.find()`, `.get()`, `.files()`, `.models()`, `.compare_models()`, and `.plot_residuals()`; `File.compare_models()` is kept as per-file sugar. `Project.export_fits()` / `File.export_fit()` give a one-way CSV + PNG dump of the current fits. v1 archives fit *outputs* and metrics, not a rehydratable model graph; project-scoped joint-fit slots are deferred.
- **σ-calibrated fit metrics and `File.set_sigma()`.** Each fit now computes and stores `chi2_raw`, `chi2_red_raw`, `chi2`, `chi2_red`, `r2`, `aic`, and `bic` (per-slice for Slice-by-Slice; a single value otherwise). Raw lmfit-unweighted diagnostics are always named `chi2_raw` / `chi2_red_raw`; the σ-calibrated values (`≈ 1` for a fit at the noise floor) are always named `chi2` / `chi2_red` and are `NaN` unless a noise sigma was set. `File.set_sigma(...)` records forward-looking file noise state (also inheritable from flat `project.yaml` defaults) that materializes into each saved slot; requesting calibrated metrics with no sigma set raises with a pointer to `set_sigma` / the raw metric name.
- **`Project.auto_export` toggle**: gates the automatic CSV/PNG side effects of `File.fit_baseline` / `fit_spectrum` / `fit_slice_by_slice` / `fit_2d` and `Project.fit_2d`. Default `True` preserves existing behavior; set `project.auto_export = False` (or `auto_export: false` in `project.yaml`) to suppress fit-completion writes for parameter sweeps, ML training-data generation, and real-time-fitting workflows. Explicit `File.export_fit` / `Project.export_fits` / `Project.save_fits` always write regardless of the flag. The legacy `save_baseline_fit` / `save_spectrum_fit` / `_save_sbs_fit_legacy` / `_save_2d_fit_legacy` auto-calls and the `fit_wrapper(save_output=...)` CSV dumps are gated by `auto_export`. The in-fit `plt_fit_res_1d` plot calls are skipped entirely (not just suppressed) when neither saving nor showing is wanted — important for SbS where rendering each per-slice figure is non-trivial work. A small pure helper `utils.plot._save_img_flag(save=..., show=...)` maps already-decided booleans onto the legacy `save_img` int, keeping the save/show decisions explicit at each call site rather than hidden inside a Project method.
- **Slice-by-Slice parallelism**: `File.fit_slice_by_slice()` accepts an `n_workers` keyword argument that dispatches per-slice fits across a `ProcessPoolExecutor` using the `spawn` start method (only portable option — Windows lacks `fork`). Default is `os.cpu_count() - 1`, capped at the number of slices. Set `n_workers=1` to keep the original serial path as a debug escape hatch. Workers reuse one pickled model installed at startup, render plots with the non-interactive Agg backend, and report progress via `tqdm`. On Linux/macOS spawn startup is a few hundred ms per worker; on Windows ~1-2s per worker, so very small fits (~< 20 slices) usually want `n_workers=1`. SbS seeding is now explicit too: `seed_source` chooses the shared template (`"model"`, `"baseline"`, or `"explicit"`), and `seed_adapt` controls the optional per-slice x0 tweak (`None` or `"argmax_shift"`).
- `Model`, `Component`, and `Par` are now pickleable (and therefore deep-copyable) via `__getstate__` / `__setstate__`. This enables `copy.deepcopy(model)` and lets live models cross process boundaries, which unblocks future multiprocessing workflows and fixes latent MCMC parallelism (see `Fixed`). Pickled instances are for short-lived transfer, not persistence — parent back-references (`parent_file`, `parent_model`) and transient fit state (`const`, `args`) are nulled.
- Example workflow `10_model_comparison/` walks the full save / load / compare loop end to end: fit two competing models at baseline, Slice-by-Slice, and 2D levels, rank them with `File.compare_models()` (including `sbs_aggregation` modes), persist with `File.save_fit()`, and reload + re-compare via `FitResults.load()` with no live `Project`.

### Changed

- Slice-by-Slice multiprocessing plumbing (worker globals, `_sbs_worker_init`, `_sbs_fit_one_slice`, and the seed-handling helpers) moved out of `trspecfit.py` into `trspecfit.utils.sbs`. `trspecfit.py` now opens directly on `class Project` instead of ~190 lines of worker plumbing. No public API changes.
- **Breaking (internal):** removed `Project.spec_lib` attribute and `Project.spec_fun` property. Fitting functions always live in `trspecfit.spectra`, so the indirection is hardcoded. Only affects code reaching into `project.spec_lib` / `project.spec_fun`; no public fit-workflow API changes.
- `fitlib.residual_fun()` and `fitlib.plt_fit_res_1d()` no longer accept a `package` argument. The constant tuple passed to `fit_wrapper()` drops its third entry (`package`) and now has shape `(x, data, function_str, unpack, e_lim, t_lim)`.

### Removed

- **Breaking:** `Project.skip_first_n_spec` and `Project.first_n_spec_only` debug controls (and the `first_n_spec_only` / `skip_first_n_spec` parameters on `fitlib.results_to_df()`, plus the now-unused `fitlib.results_select()` helper). With Slice-by-Slice parallelism a 200-slice fit takes seconds, so the "fit only the first N slices" debug shortcut no longer earns its keep. Users who want to fit a sub-range can slice the input array directly: `file.data = file.data[start:stop]; file.time = file.time[start:stop]`.

### Fixed

- **MCMC `workers > 1`**: `lmfit.emcee(workers=N)` via `ulmfit.MC(workers=N)` previously failed with `TypeError: cannot pickle 'module' object` because the residual closure carried a live module reference. The pickleable-model work plus the `spec_lib` removal close both sources of the error; MCMC parallel sampling now works end-to-end.
- **Cross-component expressions across the pickle boundary**: `Model.__getstate__` nulled `parent_model` on every `Par`, which broke `Par._evaluate_dynamic_expression` because it resolves expression references through `parent_model.get_all_parameters()`. Any model whose expression on one Par references a `t_vary` or `p_vary` Par on a different component (e.g. roundtrip family F12) raised `NameError` after unpickling. `Model.__setstate__` now rewires the intra-Model `parent_model` back-refs (Components, Pars, and any attached `Par.t_model` / `Par.p_model` sub-Models) from `self`, so `lmfit.emcee(workers > 1)` and `fit_slice_by_slice(n_workers > 1)` work on those models too.
- **Profiled parameters with convolution (IRF) dynamics on the compiled fast path**: a profiled parameter whose time-dynamics included a convolution (e.g. `MonoExpPosIRF`) failed to lower through the GIR `schedule_2d` backend. The convolution-chain walk now resolves to the underlying parameter node correctly, so profile-param IRF dynamics fit through the compiled evaluator instead of falling back.

## [0.8.0] - 2026-04-20

### Added

- **Compiled fit fast path**: 1D and 2D fits now run through a graph-IR (GIR) backend that lowers the model into a flat scheduled plan executed by a compiled evaluator. All supported functions and model compositions (see [`docs/design/supported_models.md`](docs/design/supported_models.md)) are lowerable. Enabled by default (`spec_fun_str="fit_model_gir"`), with automatic fallback to the interpreter (`fit_model_mcp`) for any model that cannot be lowered. A `fit_model_compare` mode runs both paths and asserts parity for validation.
- **`Model.visualize()`**: render the model graph as inline SVG (`rendering="graphviz"`) or raw DOT (`rendering="string"`). Per-sample profile nodes collapse into single representatives by default (`collapse_profiles=True`). Falls back gracefully if Graphviz is not installed.
- Static component caching: time-independent components are evaluated once and reused across the fit hot loop, reducing redundant work.

### Changed

- **Breaking:** convolution functions outside the Dynamics/time layer are now rejected at model-load time with a clear error. Previously this only emitted a warning. IRF-style convolutions belong in the dynamics model.
- Documentation restructured: shared agent workflows and code-review checklists live under `docs/ai/`, a new `docs/design/repo_architecture.md` documents the module map, and `docs/design/supported_models.md` is the canonical reference for supported model combinations and expressions.

## [0.7.5] - 2026-04-08

### Changed

- `Project` now exposes every `PlotConfig` field as an attribute, so `project.yaml` can fully configure plotting (limits, colors, linestyles, waterfall, etc.) without touching code. `PlotConfig.from_project()` iterates dataclass fields automatically, so new fields are picked up without changes.
- `Model.plot_1d`, `Model.plot_2d`, `Component.plot`, and `Simulator.plot_comparison` accept a `config=` argument to replace the inherited `PlotConfig` for a single call, plus per-call keyword overrides that are merged on top of the active config.

## [0.7.4] - 2026-04-02

### Added

- **Individual spectrum fitting**: `File.fit_spectrum()` fits a 1D energy model to a single spectrum extracted at a specific time point or time range, without running a full Slice-by-Slice or 2D fit.
- **Data corrections pipeline**: `File.subtract_dark()` and `File.calibrate_data()` apply dark subtraction and sensitivity calibration to raw data. `File.reset_dark()` and `File.reset_calibration()` revert to uncorrected data. Corrections stack and automatically recompute the baseline if one is defined.
- `File.describe()` auto-waterfall display: small 2D datasets (≤ 12 spectra) now render as waterfall plots instead of 2D maps. Override via the new `waterfall` parameter (`None` for auto, `0` for 2D map, or a `float` for a fixed offset).
- Per-trace opacity in `plot_1d`: traces outside the active time fit window are dimmed (`alpha=0.35`). Configurable via `alphas` parameter and `PlotConfig`.

### Changed

- Shared time validation helper `File._resolve_time_selection()` now backs `define_baseline()`, `set_fit_limits()`, and `fit_spectrum()`. Out-of-range or empty time slices raise `ValueError` instead of silently producing zero-length results.

### Fixed

- 2D models are now rejected in 1D fit contexts with a clear error instead of producing incorrect results.

## [0.7.0] - 2026-04-02

### Added

- **Project-level fitting**: fit shared models across multiple files in a single workflow. New `Project` methods: `load_models()`, `set_fit_limits()`, `define_baselines()`, `fit_baselines()`, `add_time_dependences()`, and project-level `fit_2d()` with per-file result plots and grid summaries.
- File access by name or index: `project["file_name"]` and `project[index]`.
- `File(name=...)` parameter for explicit file identifiers (defaults to path stem); duplicate names are rejected.
- `CITATION.cff` with DOI for citing trspecfit in research.
- Example workflow `05_project_level_fitting/` demonstrating multi-file project-level fitting.

### Changed

- `model_info` now accepts `str` for single models (e.g. `model_info="GLP"`) in addition to `list[str]`; `list[str]` is still required for multi-cycle submodels.
- Shared "project" parameters are validated to have matching bounds across individual file-level models.
- `Project.describe(detail=1)` now shows energy/time/z ranges per file and plots a 2D data grid with auto-column layout.

### Fixed

- Residual multiplier (`res_mult`) now read from `PlotConfig` / `project.yaml` instead of being hardcoded.

## [0.6.1] - 2026-03-26

### Added

- `Project.describe()` for inspecting project configuration, attached files, and model summaries at multiple detail levels.
- `File.get_fit_results()` to retrieve fit results as a DataFrame (`fit_type="baseline"`, `"sbs"`, or `"2d"`).
- `PlotConfig` propagation throughout the codebase: `Component.plot()`, `Simulator.plot_comparison()`, and all fit plotting now respect project-level settings (axis direction, labels, log scales, colormaps).
- Log color scale support for 2D plots via `z_type="log"` in `PlotConfig`.
- Round-trip fitting tests (simulate, fit, recover) for both basic and profile workflows.
- Parameter profile feature: `File.add_par_profile()` attaches depth/auxiliary-axis variation to any energy model parameter. Profile functions (`pExpDecay`, `pLinear`, `pGauss`) describe how a parameter varies over the auxiliary dimension.
- Single-subcycle multi-cycle models: `model_info=["none", "MonoExp"]` is now valid for a single repeating model with a set frequency.
- Python 3.13 and 3.14 support (tested in CI).

### Changed

- **Breaking:** All public API methods renamed to snake_case. Key renames:
  - `fit_SliceBySlice` -> `fit_slice_by_slice`
  - `fit_2Dmodel` -> `fit_2d`
  - `save_SliceBySlice_fit` / `save_2Dmodel_fit` -> `save_sbs_fit` / `save_2d_fit`
  - `model_SbS` / `model_2D` -> `model_sbs` / `model_2d`
  - `simulate_2D` / `simulate_1D` / `simulate_N` -> `simulate_2d` / `simulate_1d` / `simulate_n`
  - `plot_2D` / `plot_1D` -> `plot_2d` / `plot_1d`
  - `show_info` -> `show_output` (simplified to binary 0/1)
  - `par_names` -> `parameter_names` on `Model`
  - `fit` parameter -> `stages` in all fit methods
- **Breaking:** `pShirley` removed hidden 1e-6 multiplier. YAML values must be scaled accordingly (e.g. `400` -> `4E-4`).
- **Breaking:** `Par.value()` returns a scalar `float` instead of a one-element list.
- `load_model()` now always returns the loaded `Model` (previously returned `None` for energy models). `model_info` accepts both `str` and `list[str]`.
- `add_time_dependence()` and `add_par_profile()` now take explicit `target_model` and `target_parameter` as first arguments.
- Fit methods raise `ValueError` instead of silently warning when preconditions aren't met.
- Error messages throughout the API now suggest available options and corrective actions.
- `define_baseline()` time bounds are now inclusive on both ends, matching `set_fit_limits()` semantics.

### Fixed

- `PlotConfig.from_project()` regression where project YAML settings (colormap, axis labels, etc.) silently failed to load due to a global YAML constructor conflict.
- Expression parameters referencing profiled parameters now correctly resolve the profile value.
- Transitive expression chains through dynamic parameters (e.g. `A3=A2*0.5` where `A2` references a time-varying `A1`) now raise at analysis time instead of producing incorrect results.
- `GLS` formula: Lorentzian term was missing the amplitude multiplier.
- `lorentzCONV`: `W` parameter now correctly represents FWHM as documented (was FWHM/4); kernel width adjusted.
- Fit limit slicing off-by-one: `e_lim` and `t_lim` now store proper `[start, stop)` slice indices.
- `Simulator.simulate_n()` rejects `n <= 0` with a clear error.
- `Simulator` sweep metadata now records actual `dim` used and correct seed-zero handling.

### Removed

- `fit_type=0` ("show initial guess only") path from `fit_wrapper`. Use `model.describe(detail=1)` to inspect initial guesses.
- Fractional and debug verbosity levels (1.5, 3) from `show_output`.

## [0.5.2] - 2026-03-17

### Added

- Parameter profile feature: depth-dependent parameter variation over an auxiliary axis.
- Profile functions: `pExpDecay`, `pLinear`, `pGauss` in `functions/profile.py`.
- Example workflow `04_par_profiles/` demonstrating profile fitting.
- Model validation guardrails: background-last and convolution placement checks.

### Fixed

- `SinDivX` time function division-by-zero (replaced `np.sin` with `np.sinc`).
- Various Pylance type errors across the codebase.

## [0.4.3] - 2026-02-24

### Added

- Initial public release on PyPI.
- Sphinx documentation on Read the Docs.
- Core fitting workflow: `Project`, `File`, `load_model`, `fit_baseline`, `fit_slice_by_slice`, `fit_2d`.
- Energy functions: `GLP`, `Gauss`, `Lorentz`, `GLS`, `DS`, `Shirley`, `LinBack`, `Offset`.
- Time dynamics: `expFun`, `expRiseFun`, `SinFun`, `SinDivX`, `none`, with `gaussCONV`/`lorentzCONV`/`expDecayCONV`/`expRiseCONV` kernels.
- `Simulator` for generating synthetic spectroscopy data.
- Parameter sweeps via `ParameterSweep` for ML training data generation.
- Multi-cycle dynamics with subcycle frequency support.
- Expression-based parameter dependencies in YAML models.
