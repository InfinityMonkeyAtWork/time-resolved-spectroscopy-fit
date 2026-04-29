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

## User and AI ergonomics

- [ ] **Curate the public API surface before v1.0.0**: audit user-facing classes (`File`, `Project`, `Simulator`, `Model`, etc.) and decide which methods/attributes should be discoverable in notebooks and docs. Add curated `__dir__()` output for autocomplete, keep `__all__`/API docs aligned, and gradually rename or deprecate internal helper methods that should not look like primary user workflows. This should improve both human notebook ergonomics and AI/LLM efficiency by making the intended workflow surface smaller, clearer, and easier to infer.
- [ ] **Document API tiers**: add a short guide that separates stable user API (`Project`, `File`, `Simulator`, `PlotConfig`), advanced public API (`mcp.Model`, `Component`, `Par`, `ParameterSweep`, `MC`), and internal implementation modules (`graph_ir`, `eval_1d`, `eval_2d`, low-level parsing/HDF5 helpers). Use this as the source of truth for docs, tests, examples, and AI-agent guidance.
- [ ] **Add tool-neutral agent orientation**: add `AGENTS.md` or `docs/ai/agent-orientation.md` pointing agents to `CLAUDE.md`, `TODO.md`, `PLAN.md`, `docs/design/repo_architecture.md`, supported-model docs, common commands, and API-change guardrails. Keep it concise so any LLM can quickly find the intended workflow and repo boundaries.
- [ ] **Add more AI-friendly task recipes**: extend `docs/ai/` with checklists for common repo changes, such as adding YAML syntax, adding plotting options, changing fitting workflows, modifying GIR/evaluator behavior, extending save/load fields, and preparing a release.
- [ ] **Add minimal runnable workflow examples**: supplement notebooks with small script-like examples or docs snippets for the canonical public workflows: load data, load a model, set limits, fit baseline, fit 2D, inspect results, simulate data, and run a parameter sweep.
- [ ] **Improve public validation errors**: make user-facing errors state what failed, where it failed (file/model/component/parameter when applicable), and what the user or agent should change next. Prioritize YAML parsing, model loading, fit setup, and unsupported-model fallback paths.
- [ ] **Tighten public type hints and aliases**: reduce ambiguous `Any` on public APIs, document key aliases such as `ModelRef`, and keep return types crisp for IDEs, Pyright, generated docs, and LLM code navigation.
- [ ] **Document fast verification slices**: add focused pytest commands for common edits (public workflow/API, YAML parser, functions, GIR/evaluator, plotting) so contributors and agents can validate changes quickly before running the full suite.

## Build & release

- [ ] **Automate tagging and pushing**: automate `git tag v1.2.3` + `git push v1.2.3` as part of the release workflow.
- [ ] **Remove legacy/backwards-compat code**: before v1.0.0 release, audit codebase for legacy fallbacks and backwards compatibility shims and consider removing.
