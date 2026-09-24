---
orphan: true
---

# Results Ownership Boundary & Plotting Disentanglement

> **Status: implemented** (2026-07-17, `model-vs-fitresult` branch, v0.14.0).
> Execution record for two coupled TODO items: "Define the results-data
> ownership boundary" and "Disentangle plotting from saving/conversion in
> the fit pipeline". The living contract is summarized in
> [../repo_architecture.md](../repo_architecture.md); the wire format in
> [../fit_archive_schema.md](../fit_archive_schema.md) (schema 3).

## The ownership contract

- **`Model`/`File` (live layer)** own inputs and fit execution.
  `model.result` (originally the raw `[par_ini, par_fin, conf_ci,
  emcee_fin, emcee_ci]` list; a typed `FitOutput` since Phase 8 below)
  is a transient internal of the fit run; nothing user-facing reads it.
- **`SavedFitSlot`** is the single authoritative record of a completed
  fit — everything a user can ask about a fit must be in (or derivable
  from) the slot.
- **`FitResults`** is the single read/query/plot surface over slots —
  live history (`Project.results`) and loaded archives
  (`FitResults.load`) alike. `File.get_*` / `File.plot_*` /
  `File.compare_models` are thin sugar delegating to it.

## Load-bearing decisions (settled with user, 2026-07-14/16)

1. **Full auto-export layout break.** Auto-export inside
   `fit_slice_by_slice` / `fit_2d` / `Project.fit_2d` routes through the
   slot exporter (`export_fit(path_results, ..., overwrite=True)`),
   producing the grouped `{path_results}/{file}/{model}__{fit_type}/`
   tree. The legacy flat layout, the deprecated `save_sbs_fit` /
   `save_2d_fit`, and the `_save_*_fit_legacy` impls were deleted.
   Fit-time diagnostics (per-stage CSVs from `fit_wrapper`, SbS
   per-slice artifacts) intentionally stay under `File.model_path()` —
   they are the fit audit trail, not the results export, and the export
   slot-dir name is snapshot-dependent (hash suffix) so it cannot be
   computed mid-fit.
2. **Slot-backed accessors, latest-match semantics.**
   `get_fit_results` / `get_correlations` / `get_conf_intervals` /
   `get_mcmc` read the latest matching slot (`find()` is history-ordered),
   mirroring the old convention where each `fit_*` call overwrote the
   live result of its type. Accessors return copies. Consequences
   accepted: covariance-less fits raise from `get_correlations` instead
   of fabricating an identity matrix (`correl` is captured only when
   `result.covar` exists — Nelder without numdifftools and project joint
   fits store `None`, mirroring absent stderr/conf_ci); fits on Files
   without `data` record no slot (nothing to fingerprint) and so the
   accessors raise.
3. **Plot API on `FitResults` with `File` sugar** (the `compare_models`
   precedent): `plot_fit`, `plot_param_evolution`, upgraded
   `plot_residuals`. `FitResults` carries fingerprint-matched axes
   providers (live `File`s / `SavedFile`s) so plots get real energy/time
   axes on both construction paths, with array-index fallback. The fit
   methods' inline display goes through the same API — the figure shown
   at fit time is the one the API reproduces later. Config resolution:
   explicit `config=` > live file's `plot_config` > `PlotConfig()`;
   styling is deliberately not persisted in archives.

## Schema-3 additions (all additive; v2 archives load with `None`)

- `correl` — varying-parameter correlation matrix (slice 0 for SbS).
- mcmc `acceptance_fraction` — emcee's per-walker array.
- `params_meta` (SbS) — the slice-invariant `[name, vary, min, max,
  expr]` metadata. Vary is uniform across slices by construction (one
  model, no mid-loop hook; serial ≡ parallel dispatch pinned by test).
- `params_stderr` (SbS) — per-slice standard errors, previously
  discarded entirely; the future weights for 1D trace fitting.
- `fit_settings` — optimizer provenance on every fit type: `stages`,
  `fit_alg_1/2`, `try_ci`, SbS seeding recipe (`seed_source` /
  `seed_adapt` / `seed_values`; `None` kept as meaningful), and the MC
  sampling settings when MCMC ran. Excluded from `history_key` (a refit
  with different settings is still a refit).

## Considered and rejected

- **YAML-derived parameter capture** (store the model.yaml as JSON in
  the slot): the runtime state routinely diverges from the YAML — the
  default SbS workflow seeds from the *baseline fit result*, users
  mutate models between load and fit, and composed models span multiple
  YAMLs. Slots snapshot the model *as fit*. Raw-YAML provenance /
  model rehydration stays a separate, deferred feature.
- **Slice-0 long-form params sidecar** for SbS: would have carried
  per-slice fields (`value`, `stderr`, `init_value` — the latter
  diverges under `seed_adapt`) misleadingly presented as representative.
  Split instead into the honest `params_meta` + `params_stderr`.
- **`n_workers` in `fit_settings`**: serial and parallel SbS dispatch
  are result-identical by design (same seed template per slice, no
  cross-slice warm start) and pinned by a parity test — recording the
  worker count would imply it can influence results.
- **Live-with-slot-fallback accessors**: two code paths to test for
  zero benefit once slots are captured on every fit.
- **Interim 2D-only hoist** of the legacy plotting (pre-branch): would
  have left three coexisting conventions; done as one deliberate pass
  instead.

## Deferred

- Typed result object replacing the internal raw `result[1..4]` list —
  completed in Phase 8 (below) rather than deferred after all.
- Model rehydration from archives (raw YAML text provenance).
- The in-place-mutation guard stance for user-facing arrays (separate
  TODO item; slots store arrays by reference).

## Phases 7–8 continuation (same branch, 2026-07-17)

Two follow-on phases from the post-completion review completed the same
principle in the same release (0.14.0).

### Phase 7 — remove auto-export; fits never write

`Project.auto_export`, `Project.path_results`, and `File.model_path`
removed: fit methods compute, display (per `show_output`), and capture
slots — they never touch disk. Persistence is the explicit `save_fits`
(HDF5) / `export_fits` (CSV/PNG tree) pair, both slot-fed, with the
single default output root `./fit_results/<Project.name>/`
(`Project.name` default `"test"` → the placeholder `"my_project"`).
A `project.yaml` still setting a removed key fails loudly with
migration guidance. Everything auto-export used to write is
reproducible:

- MCMC walker-acceptance and corner PNGs → `FitResults.plot_mcmc`
  (renders from the persisted slot payload — live sessions and loaded
  archives alike; `File.plot_mcmc` sugar).
- SbS per-slice PNGs → `File.plot_sbs_slices` (live `results_sbs` only;
  shows the per-slice seeded initial guess and component decomposition
  the old PNGs lacked). Deliberately NOT part of `export_fits` — export
  stays archive-reproducible, per-slice panels need live inputs.
- Component-decomposed `fit_1d.csv` → `save_baseline_fit` /
  `save_spectrum_fit` (kept as explicit calls, no longer auto-invoked).
- Accepted losses: per-slice `par_ini` CSVs (re-derivable from the
  persisted `fit_settings` seeding recipe) and `lmfit.fit_report` text
  dumps (all contents persisted in the slot).

`fitlib.fit_wrapper` lost `save_output` / `save_path` / `num_fmt` /
`delim` and its CSV/TXT dump block; emcee diagnostics figures are
display-only. The SbS worker payload shed its IO-only arguments
(`auto_export`, `path_slice`, `plot_config`).

### Phase 8 — typed `FitOutput`

The raw 5-list became a frozen dataclass — named `FitOutput`, in
`utils/lmfit.py` because `fitlib` imports `spectra` → `mcp` and `mcp`
needs the annotation (so the class must live below all three). Fields
`par_ini` / `par_fin` / `conf_ci` / `emcee_fin` / `emcee_ci`;
`Model.result` is `FitOutput | None` (was `[]` when unfitted),
`File.results_sbs` holds one per slice, and `Project.fit_2d`'s per-file
stand-in is a real minimal `MinimizerResult` (params/method/nvarys)
instead of a `SimpleNamespace`. lmfit sets result attributes
dynamically (invisible to pyright), so a TYPE_CHECKING-only
`TypedMinimizerResult` shim declares the attributes trspecfit reads,
with casts at the two construction choke points. No list-index
back-compat; `SavedFitSlot` and all `FitResults` accessors unchanged.
Naming consolidation in the same pass: `plt_fit_res_1d(par_init=)` →
`par_ini` (matching the `par_ini`/`par_fin` pair); `show_init` (a verb
phrase) and lmfit's own `init_value` deliberately kept.
