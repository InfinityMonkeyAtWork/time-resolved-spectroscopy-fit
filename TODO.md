# Project TODOs

## Fitting

- [ ] **Project-level fits**: allow parameters to vary at Project, File, or Static scope (multi-file fitting).
- [ ] **Fit results save/load**: HDF5 output for fit results, `File.load_fit()` to restore, `File.compare_models()` for model comparison. Keep project/global-fit outputs separate from true file-level fits so users do not assume identical per-file fields/statistics.
- [ ] **Mismatched initial guesses**: round-trip tests — one each for basic, profile, profile+dynamics.

Note: `fitlib.py` hardcodes `__lnsigma` value/min/max for MCMC sampling — make configurable via `mc_settings` if users need it.

## UX improvements

- [ ] **ROI input for SNR**: `Simulator.snr()` uses full data array for signal power. Allow user to specify a region of interest.

## Performance & architecture

Don't touch until feature set is stable.

- [ ] **Evaluation order correctness**: component eval order depends on coincidental list position; make it explicit. One option: build a directed acyclic graph (DAG) at model construction and topological-sort.
- [ ] **Freeze non-varying pars**: pars without time-dependence (or profile dependence) are re-evaluated at every aux-axis point; could evaluate once and reuse.

## Testing

- [ ] **Pylance/Pyright `| None` noise in tests**: ~280 Pyright errors across tests, all from accessing `File`/`Model` attributes typed as `ndarray | None`. Current `# type guard` asserts are inconsistent and don't propagate through helper methods. Find a cleaner pattern (e.g. `TypeGuard`, narrowing wrapper, or Pyright config) and apply consistently.

## Build & release

- [ ] **Automate tagging and pushing**: automate `git tag v1.2.3` + `git push v1.2.3` as part of the release workflow.
- [ ] **Remove legacy/backwards-compat code**: before v1.0.0 release, audit codebase for legacy fallbacks and backwards compatibility shims and consider removing.
