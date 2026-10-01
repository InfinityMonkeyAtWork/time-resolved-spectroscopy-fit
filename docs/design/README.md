---
orphan: true
---

# Design documents

Index of `docs/design/`, one line per document. Live documents state the
current contract; the archived plans and decisions under `archive/` record why
it is that way and what was rejected. Add a line here whenever a document is
added or a plan is archived (see "The Archive" in `CLAUDE.md`).

## Live

- [repo_architecture.md](repo_architecture.md) — module map and orientation guide: the two-layer design, ownership boundaries, where each concern lives.
- [api_ownership_contract.md](api_ownership_contract.md) — settled 2026-09-24: what counts as public, who owns each piece of live state (`File` inputs, corrections, baselines, model definition vs parameter state, completed results), which mutations are supported, and what inspection may read; the YAML is the source of truth, a model belongs to its file, parameter vocabulary by tier.
- [supported_models.md](supported_models.md) — source of truth for supported model combinations, expressions and compositions on the 1D and 2D evaluators.
- [fit_archive_principles.md](fit_archive_principles.md) — fit identity, ownership and comparability rules behind the archive, including which inputs are keyed (the declared noise model among them).
- [fit_archive_schema.md](fit_archive_schema.md) — on-disk HDF5 layout of the fit archive (schema 7) and how the reader maps it back to objects.
- [lowered_evaluator.md](lowered_evaluator.md) — the Graph IR (GIR) lowered evaluator: design and spec of the compiled hot path.
- [examples_architecture.md](examples_architecture.md) — how the example tree is organized and the decisions behind it.
- [roundtrip_test_matrix.md](roundtrip_test_matrix.md) — the intended roundtrip-test surface for single-file fits.
- [project-level-fits.md](project-level-fits.md) — forward-looking note (2026-07): project-level shared fits on a compiled backend.
- [ui.md](ui.md) — forward-looking note (2026-07): backend requirements for an interactive UI.

## Archived plans and decisions

- [archive/noise_model_weighting_plan.md](archive/noise_model_weighting_plan.md) — v0.17.0: `File.set_noise`, deviance and Gaussian weighting, noise in identity and metrics, Cramér-Rao verification, the example teaching split; frozen data-based weights rejected.
- [archive/fit_archive_schema_plan.md](archive/fit_archive_schema_plan.md) — conversion plan to fit-archive schema 7: what changed on disk and in memory.
- [archive/fit_results_save_load_plan.md](archive/fit_results_save_load_plan.md) — the first fit-results archive: per-slot `observed`, two identity keys, HDF5 over pickle, the in-memory history layer.
- [archive/joint_fit_result_plan.md](archive/joint_fit_result_plan.md) — first-class joint fit results from `Project.fit_2d`: semantics and public-API decisions.
- [archive/results_ownership_plotting_plan.md](archive/results_ownership_plotting_plan.md) — v0.14.0: the results ownership boundary and disentangling plotting from saving.
- [archive/lowered_evaluator_implementation_plan.md](archive/lowered_evaluator_implementation_plan.md) — implementation record of the lowered (GIR) evaluator and its follow-ups.
- [archive/numba_vs_jax_decision.md](archive/numba_vs_jax_decision.md) — decision: Numba versus JAX for the lowered evaluator, with benchmarks (`numba_vs_jax_benchmarks.svg`).
- [archive/jax_backend_note.md](archive/jax_backend_note.md) — planning note for the JAX backend, analytic Jacobians and optimizer replacement (Phase E deferred).
- [archive/jax_backend_plan.md](archive/jax_backend_plan.md) — v0.12.0 execution record of the JAX backend track, Phases A–D.
- [archive/kernel_matrix_convolution_plan.md](archive/kernel_matrix_convolution_plan.md) — the kernel-matrix convolution operator on the mcp and GIR paths, and why two kernels were dropped.
- [archive/code_review_2026_07.md](archive/code_review_2026_07.md) — full-repo code review of July 2026: every finding fixed, declined with rationale, or moved to `TODO.md`.
- [archive/review_2026_06_triage.md](archive/review_2026_06_triage.md) — the June 2026 external review triaged against v0.19.0: each weakness retired with evidence or carried to its `TODO.md` item; the reachable-trigger rule.
