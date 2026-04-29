# Project TODOs

## Fitting

- [ ] **Fit results save/load**: HDF5 output for fit results, `File.load_fit()` to restore, `File.compare_models()` for model comparison. Keep project/global-fit outputs separate from true file-level fits so users do not assume identical per-file fields/statistics.
- [ ] **Mismatched initial guesses**: round-trip tests — one each for basic, profile, profile+dynamics.

Note: `fitlib.py` hardcodes `__lnsigma` value/min/max for MCMC sampling — make configurable via `mc_settings` if users need it.

## Performance & architecture

- [ ] **Project-level fit backend**: `Project.fit_2d()` already supports `Project`/`File`/`Static` vary levels, but it currently evaluates through `fit_project_mcp()` and `Model.create_value_2d()` rather than the GIR scheduler/evaluator path. Decide whether to lower the multi-file residual to GIR or explicitly prefer project-managed per-file loops when we want maximum graph-IR speedups.
- [ ] **JAX backend / Jacobian follow-on**: if we revisit a JAX evaluator, analytic Jacobians, or optimizer replacement, use [docs/design/jax-planning.md](docs/design/jax-planning.md) as the roadmap for scope, sequencing, and open technical constraints.
- [ ] **MCMC multiprocessing context**: `lmfit.emcee(workers=N)` currently inherits Python's default multiprocessing start method, which triggers a Python 3.12 `fork()` deprecation warning in multithreaded test runs. Investigate whether we can supply a `spawn`-backed worker pool or otherwise steer emcee/lmfit away from raw `fork`.

## Testing

- [ ] **Pylance/Pyright `| None` noise in tests**: ~280 Pyright errors across tests, all from accessing `File`/`Model` attributes typed as `ndarray | None`. Current `# type guard` asserts are inconsistent and don't propagate through helper methods. Find a cleaner pattern (e.g. `TypeGuard`, narrowing wrapper, or Pyright config) and apply consistently.

## Build & release

- [ ] **Automate tagging and pushing**: automate `git tag v1.2.3` + `git push v1.2.3` as part of the release workflow.
- [ ] **Remove legacy/backwards-compat code**: before v1.0.0 release, audit codebase for legacy fallbacks and backwards compatibility shims and consider removing.
