# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com),
and this project adheres to [Semantic Versioning](https://semver.org/).
This file is maintained using the shared changelog workflow in
[`docs/ai/changelog.md`](docs/ai/changelog.md).

## [Unreleased]

### Added

- `Model`, `Component`, and `Par` are now pickleable (and therefore deep-copyable) via `__getstate__` / `__setstate__`. This enables `copy.deepcopy(model)` and lets live models cross process boundaries, which unblocks future multiprocessing workflows and fixes latent MCMC parallelism (see `Fixed`). Pickled instances are for short-lived transfer, not persistence — parent back-references (`parent_file`, `parent_model`) and transient fit state (`const`, `args`) are nulled.

### Changed

- **Breaking (internal):** removed `Project.spec_lib` attribute and `Project.spec_fun` property. Fitting functions always live in `trspecfit.spectra`, so the indirection is hardcoded. Only affects code reaching into `project.spec_lib` / `project.spec_fun`; no public fit-workflow API changes.
- `fitlib.residual_fun()` and `fitlib.plt_fit_res_1d()` no longer accept a `package` argument. The constant tuple passed to `fit_wrapper()` drops its third entry (`package`) and now has shape `(x, data, function_str, unpack, e_lim, t_lim)`.

### Fixed

- **MCMC `workers > 1`**: `lmfit.emcee(workers=N)` via `ulmfit.MC(workers=N)` previously failed with `TypeError: cannot pickle 'module' object` because the residual closure carried a live module reference. The pickleable-model work plus the `spec_lib` removal close both sources of the error; MCMC parallel sampling now works end-to-end.

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
