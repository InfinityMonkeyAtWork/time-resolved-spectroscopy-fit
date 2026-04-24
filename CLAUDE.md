# General behavior

- **Confidence Rule:** Do not make changes until you have 95% confidence. Understand the relevant files before editing and ask follow-up questions until you reach this threshold or when tradeoffs are non-obvious.
- **Context Discipline:** Monitor context usage. At 60% usage (or if it starts getting tight), summarize progress and prompt me to `/compact` or `/clear`.
- **Token Efficiency:** Be concise. Reference file paths and line numbers rather than quoting large code blocks.
- **Subagent Protocol:** Use subagents for repo-wide scans, parallel research, or scanning large directories. Instruct them to return only concise summaries to keep the main context window lean.
- **Guardrails:** Use this file for coding guardrails; heavier review checklists live in `docs/ai/code-review.md`.
- **Renaming / API changes:** grep the entire repo—notebooks, YAML, tests, and docs all reference the public API.


# Persistent State Management

- **TODO.md:** Keep for high-level, long-term project goals. Do not add granular task items here. Mark the actively worked on feature with an `[ACTIVE]` tag.
- **PLAN.md:** Maintain in the root for active, multi-step feature work. Read at session start and keep updated in real-time as tasks are completed. Small one-off fixes do not need `PLAN.md`.
- **The Archive:** Once a feature is 100% complete, ask user if (A) the contents of `PLAN.md` warrant being moved into a new renamed file in `docs/design/archive/` or (B) the changelog is enough documentation. In any case clear the root `PLAN.md` and update `TODO.md` (including removing the `[ACTIVE]` tag).


# Architecture guardrails

- **Two-Layer Design:**
  - **Authoring / User-facing layer** (`mcp.py`, `trspecfit.py`, `simulator.py`, and YAML parsing in `utils/parsing.py`): Optimize for readability, human-usability, validation, and clear errors. Performance is not prioritized here.
  - **Compiled Hot-Path layer** (`graph_ir.py`, `eval_1d.py`, `eval_2d.py`, and numeric bodies in `functions/`): Performance-critical and array-oriented. Avoid Python objects and model-structure branching in inner loops.
- **Bridge & Logic:** `spectra.py` bridges fitting to the compiled evaluator. `fitlib.py` drives `lmfit`, CI, MCMC, and fit-result plotting.
- **Registries:** Check `config/`, `functions/`, and `utils/` for shared registries and helpers before adding new ones.
- **Source of Truth:** Treat `docs/design/supported_models.md` as the source of truth for supported model combinations, expressions, and compositions.
- **Module Map:** Full reference at `docs/design/repo_architecture.md`.


# Code style

- **Linting:** Ruff is the formatter/linter. Line length 88.
- **Spacing:** Add an empty line after each docstring. Add a comment line containing only `#` before function/method definitions and two such lines before class definitions.
- **Naming:** Use `snake_case` by default. **Exception:** Function registry names (`GLP`, `pExpDecay`, etc.) and their parameters (`A`, `x0`, etc.) in `src/trspecfit/functions/` use CamelCase/PascalCase because `_` is the component ID delimiter (`{model}_{component}_{param}`).
- **Signatures:** Prefer keyword arguments for all parameters except the primary data object. Use `*` to enforce keyword-only arguments for any parameter that isn't the primary data "subject."
- **Signature Exception:** Registry functions in `src/trspecfit/functions/` keep positional signatures because parsing/introspection depends on ordered parameter lists.


# Testing

- **Pattern:** Use plain pytest. Avoid `unittest.TestCase` and fixtures; prefer explicit helper builders named by intent.
- **API Usage:** Use the public API (`Project`, `File.load_model`, etc.) in tests to avoid masking bugs by skipping validation or setup. Use internals only for pure-math unit tests or explicit invariant checks.
- **Execution:** Run `pytest -q`. Keep YAML test assets in `tests/models/`.
- **Plots:** Always suppress plot display in tests: pass `show_plot=False` where available, or `save_img=-2`.
- **Type Guards:** When `assert x is not None` narrows an `X | None` type, add a `# type guard` comment.
- **Variable Naming:** For variables derived from registry parameters or components, keep original casing (e.g., `SD = 2.0`, `c_Shirley = Component("Shirley")`). Name derived variables as `{par}_{qualifier}` (e.g., `A_early`, `mean_A`).


# Documentation

- Keep `README.md` minimal (overview and quick-start only). Detailed docs belong in `docs/`.
- Use NumPy-style docstrings. User-facing API and `functions/` get extensive docstrings; internal modules keep method docstrings minimal.
- Past, large, impactful design decisions are documented here: `docs/design/archive/`.
