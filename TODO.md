# Project TODOs

## Test coverage: round-trip fits

- [ ] Basic 1D energy fit — known peaks, add noise, fit, assert recovered params
- [ ] 2D with dynamics — known decay, add noise, fit, assert
- [ ] With profiles — depth-varying spectrum, add noise, fit, assert
- [ ] Time-dependent profiles — spectral diffusion, add noise, fit, assert
- [ ] REGRESSION: add test for each bug as discovered (ongoing)

## Profile feature gaps

- [x] **Expression + profile inheritance**: direct refs work; transitive chains raise `ValueError` by design (2026-03-18).
- [x] **Par.value() return type cleanup**: `Par.value()` returns scalar float now. HOTFIX workarounds removed (2026-03-19).
- [ ] **Weighted aux-axis integration**: aux-axis averaging uses uniform `sum/N`. User-configurable weights would cover non-uniform spacing (trapezoidal), importance weighting, etc.
- [ ] **Smart autocorrect for profile names**: fuzzy match suggestions when YAML key doesn't match an energy parameter. Low priority.
- [ ] **Performance: freeze non-profile pars**: non-varying pars re-evaluated at every aux-axis point; could evaluate once and reuse.

## Simulator

- [ ] **ROI input for SNR**: `Simulator.snr()` uses full data array for signal power. Allow user to specify a region of interest for the calculation.
- [ ] **Simulator.plot_comparison should use PlotConfig**: `plot_comparison()` doesn't respect `x_type`/`y_type` (lin vs log) from `PlotConfig`. Wire up axis scale settings.

## Fitting

- [ ] **ML-based peak finder**: replace or augment current peak finder with ML approach for better initial guesses.
- [ ] **Project-level fits**: allow parameters to vary at Project, File, or Static scope (multi-file fitting).

## Fit results: save & load (for model comparison)

- [ ] **HDF5 save**: add HDF5 output option for fit results. Keep existing file-based saving as default for backwards compatibility; let user choose format.
- [ ] **HDF5 load**: implement `File.load_fit()` to restore fit results from HDF5. Only HDF5 format needs load support — file-based results are write-only legacy.
- [ ] **Model comparison**: `File.compare_models()` uses loaded fit results to compare different models. Reuses the load infrastructure.

## Project.describe() enhancements

- [x] **File inventory (detail=1)**: implemented (2026-03-19). Lists all Files (path, dim, shape, aux_axis, models, active model). Previous detail=1 (plot/IO settings) shifted to detail=2.

## Cleanup

- [ ] **Configurable `__lnsigma` for emcee**: `fitlib.py` hardcodes `__lnsigma` value/min/max for MCMC sampling. Allow users to pass these via `MCsettings`.
- [x] **Remove deprecated `fit_type == 0` path**: removed (2026-03-19).

## Code quality

- [x] **Attribute inheritance vs references**: resolved (2026-03-19). Axis assignments (`comp.energy = self.energy`) are Python reference-sharing, not copies — no duplication problem. Properties would break conv components (kernel time axis). Decision: keep reference-sharing, document the contract, add `Project.files` for downward traversal. Fixed `Component.plot(plot_ind=True)` aux_axis bug for expression-only profile dependence.
- [ ] **Replace 0/1 flags with enums**: codebase uses `0`/`1` where `True`/`False` or enums would be more readable and self-documenting. Audit and convert.
- [ ] **Fix FBT001/FBT002 (boolean trap)**: functions like `debug` accept booleans as positional args, making call sites unclear. Consider keyword-only or enum alternatives.
- [ ] **Optional dependency extras**: move `ipython` and `matplotlib` out of core into optional extras (e.g. `pip install trspecfit[lab]` or `trspecfit[all]` for notebook/plotting functionality). Keep base install light for programmatic use.

## Build & release

- [ ] **Automate tagging and pushing**: automate `git tag v1.2.3` + `git push v1.2.3` as part of the release workflow.
