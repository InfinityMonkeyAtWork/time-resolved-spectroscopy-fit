# Project TODOs

## Test coverage: round-trip fits

- [ ] Basic 1D energy fit — known peaks, add noise, fit, assert recovered params
- [ ] Multi-peak 2D — two or more peaks with independent dynamics
- [ ] With profiles — depth-varying spectrum, add noise, fit, assert
- [ ] Time-dependent profiles — spectral diffusion, add noise, fit, assert

## Fitting

- [ ] **ML-based peak finder**: replace or augment current peak finder with ML approach for better initial guesses.
- [ ] **Project-level fits**: allow parameters to vary at Project, File, or Static scope (multi-file fitting).
- [ ] **Configurable `__lnsigma` for emcee**: `fitlib.py` hardcodes `__lnsigma` value/min/max for MCMC sampling. Allow users to pass these via `mc_settings`.
- [ ] **pShirley `1e-6` scaling factor**: hardcoded scale factor in `energy.py` silently affects fit results. Needed to prevent numerical instability, but should be derived from data or made an explicit parameter.

## Fit results: save & load (for model comparison)

- [ ] **HDF5 save**: add HDF5 output option for fit results. Keep existing file-based saving as default for backwards compatibility; let user choose format.
- [ ] **HDF5 load**: implement `File.load_fit()` to restore fit results from HDF5. Only HDF5 format needs load support — file-based results are write-only legacy.
- [ ] **Model comparison**: `File.compare_models()` uses loaded fit results to compare different models. Reuses the load infrastructure.

## UX improvements

- [ ] **ROI input for SNR**: `Simulator.snr()` uses full data array for signal power. Allow user to specify a region of interest for the calculation.
- [x] **Consistent PlotConfig usage**: audit all plot functions across the codebase to ensure they respect `PlotConfig` settings (e.g. `Simulator.plot_comparison()` ignores `x_type`/`y_type`).
- [x] **API ergonomics audit**: rationalize `show_output` verbosity levels, replace 0/1 flags with enums or bools, fix boolean-trap positional args (FBT001/FBT002). One pass to make the API more self-documenting.
- [ ] **AI-assisted YAML model creation**: use AI to help users build and validate YAML model files — smart autocorrect, suggesting parameter names, and potentially inspecting the data to propose initial guesses or flag mismatches.

## Performance & architecture

Don't touch until feature set is stable.

- [ ] **Dependency-aware evaluation order**: component evaluation order is currently reverse list position (last→first, so backgrounds see peak sums). Profile refresh happens inside each component's `_value_profile_instances`, and cross-component expression references read cached profile state via `Par.value()`. This means correctness depends on evaluation order coincidentally matching data flow. When an expression-only component evaluates before the component that owns the referenced profile, `Par.value()` must eagerly refresh the profile (current bandaid — causes double profile evaluation per `t_ind`). **Proper fix**: build a DAG at model construction time (nodes = components + profile models, edges = expression refs + profile ownership + background deps). Topological sort determines eval order. Profile models get a dedicated pre-pass keyed to `t_ind`, then components read from already-fresh profiles. Benefits: correctness by construction, no stale reads, transitive chain detection for free (cycle detection), clearer path to component-level caching.
- [ ] **Freeze non-varying pars**: pars without time-dependence (or profile dependence) are re-evaluated at every aux-axis point; could evaluate once and reuse.
- [ ] **Parallelization / vectorization**: explore numba, GPU acceleration, or parallel evaluation across aux-axis points.

## Build & release

- [ ] **Automate tagging and pushing**: automate `git tag v1.2.3` + `git push v1.2.3` as part of the release workflow.
- [ ] **Remove legacy/backwards-compat code**: before v1.0.0 release, audit codebase for legacy fallbacks and backwards compatibility shims and consider removing.
