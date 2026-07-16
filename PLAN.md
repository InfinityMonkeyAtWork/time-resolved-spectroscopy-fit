# Active Plan: results ownership boundary + plotting/saving disentanglement

Branch: `model-vs-fitresult`. Covers two TODO items (both `[ACTIVE]`):
"Define the results-data ownership boundary" and "Disentangle plotting from
saving/conversion in the fit pipeline". Scoping session 2026-07-13 and design
decisions 2026-07-14.

## Decisions (settled with user, 2026-07-14)

1. **Full layout break**: auto-export for SbS/2D routes through the slot-based
   exporter (`fit_io._export_slot`); the legacy flat layout is dropped.
   `_save_sbs_fit_legacy`, `_save_2d_fit_legacy`, and the deprecated public
   `save_sbs_fit` / `save_2d_fit` are deleted in this branch. Partially
   unblocks v1.0.0 checklist item 4 (legacy-shim removal).
2. **Plot API home**: explicit plot methods live on `FitResults` (reading
   slots), with thin `File.plot_*` sugar — same pattern as `compare_models`.
3. **Data source**: relocated accessors read **persisted slots** (latest
   matching slot in `_fit_history`). Live `model.result[...]` becomes an
   internal detail. Requires persisting `correl` + `acceptance_fraction`
   (Phase 1). Raw `result[1..4]` access and the deeper unified-results-object
   question stay deferred (per TODO).

## Ownership boundary (the contract)

- **`Model`/`File` (live layer)**: own inputs and fit execution. `model.result`
  is a transient internal of the fit run; nothing user-facing reads it.
- **`SavedFitSlot`**: the single authoritative record of a completed fit —
  everything a user can ask about a fit must be in (or derivable from) the slot.
- **`FitResults`**: the single read/query/plot surface over slots (live history
  and loaded archives alike). `File.get_*` / `File.plot_*` / `File.compare_models`
  are thin sugar delegating to `project.results` filtered to that file.

## Phase 1 — Persist `correl` + `acceptance_fraction` (schema v3) — DONE

- [x] Add `correl: pd.DataFrame | None` field to `SavedFitSlot` (correlation
      matrix over varied params, mirroring `get_correlations()` output).
      Stored like `conf_ci` via `_encode_dataframe` (all-float64 square
      matrix; index restored from `columns` on read). Matrix builder is
      `ulmfit.correl_to_df`; `File.get_correlations` now delegates to it.
      Captured in the `_append_*_slot` methods, gated on
      `result_fin.covar is not None` — so `correl` is `None` for
      covariance-less optimizers and project joint fits (mirrors
      stderr/conf_ci absence) instead of a misleading identity matrix.
      SbS captures slice 0 (the conf_ci/mcmc convention).
- [x] Add `acceptance_fraction` (per-walker float array) to the slot `mcmc`
      payload (`_mcmc_payload`, `_write_mcmc_group`, `_read_mcmc_group`).
- [x] Bump `SCHEMA_VERSION` "2" → "3" with `SUPPORTED_READ_VERSIONS = ("2",
      "3")`. Reader accepts v2 archives (new fields → `None`); append across
      versions stays refused (existing policy). `fit_archive_schema.md`
      updated (also fixed the stale metrics attr list there).
- [x] Round-trip tests: `test_correl_roundtrip` (leastsq pins the
      deterministic covar path; Nelder covar depends on numdifftools),
      conf_ci/correl/mcmc comparison added to `_assert_slot_round_tripped`,
      acceptance_fraction round-trip in `TestMcmcPayload` (slow), v2
      read-tolerance + unknown-version rejection tests.

## Phase 2 — Relocate accessors to `FitResults`, slot-backed

- [ ] `FitResults.get_fit_results / get_correlations / get_conf_intervals /
      get_mcmc(file=..., model=..., fit_type=...)` reading the latest matching
      slot (define "latest" = last in history order; document).
- [ ] `get_mcmc` builds `ulmfit.MCMCResult` from the slot payload
      (flatchain, ci table, lnsigma, acceptance_fraction).
- [ ] `File.get_*` become thin delegates to `self.p.results` (signatures
      unchanged → notebook 12 keeps working). Delete `File._result_model`.
- [ ] SbS gains accessor coverage for free (slots exist; `_result_model`
      used to raise for sbs).
- [ ] Re-run / spot-check notebook 12 (`12_uncertainty_mcmc`) — its calls are
      the compatibility contract.

## Phase 3 — Purify the conversion layer (fitlib)

- [ ] `results_to_df`: strip CSV writing and plotting → pure
      results-list → DataFrame conversion (drop `save_df`/`save_path`/plot
      args). Its only caller today is `_save_sbs_fit_legacy` (dies in Phase 4).
- [ ] `results_to_fit_2d`: strip `save_2d` CSV writing → pure reconstruction.
      (Slot already stores the `fit` array; exporter doesn't need this.)
- [ ] `plt_fit_res_1d` / `plt_fit_res_2d` / `plt_fit_res_pars` remain the
      pure renderers (already flag-driven via `_save_img_flag`).

## Phase 4 — Route auto-export through slots; delete legacy path

- [ ] `fit_slice_by_slice` / `File.fit_2d` / `Project.fit_2d`: replace the
      `_save_*_fit_legacy(save_files=...)` calls with the baseline template —
      display (`show_output>=1`) via direct renderer call on slot data with
      `_save_img_flag(save=..., show=...)`; export (`auto_export`) via the
      slot exporter. Keep the skip-entirely guard when neither is set
      (hot-path invariant pinned by `TestPlotHelperSkipped`).
- [ ] Per-slice PNGs inside the SbS loop (trspecfit.py ~3345) stay gated by
      `auto_export` (unchanged behavior, new destination layout).
- [ ] Delete `_save_sbs_fit_legacy`, `_save_2d_fit_legacy`, `save_sbs_fit`,
      `save_2d_fit`. **Grep whole repo** (notebooks, YAML, docs, llms.txt,
      AGENTS.md) for callers/mentions.
- [ ] Update guardrail tests (`TestVerboseDisplayWithoutExport`,
      `TestPlotHelperSkipped`, auto-export write/no-write classes) to the new
      call targets; semantics (display-without-write, silent-skip) unchanged.
- [ ] Changelog entry flagging the auto-export layout change (breaking).

## Phase 5 — Explicit plotting API

- [ ] **Prerequisite**: `FitResults` must retain slot → axes context. Today it
      flattens `SavedProject.files[*].slots` and discards `SavedFile`
      (why `plot_residuals` plots vs. array index). Keep a per-slot reference
      to its `SavedFile` (or a fingerprint-keyed axes lookup); `Project.results`
      builds the equivalent from live `File` axes. `_slot_axes`
      (fit_io.py ~1707) already implements the slicing.
- [ ] `FitResults.plot_fit(file=..., model=..., fit_type=...)` — 1D/2D
      observed/fit/residual from slot, real energy/time axes, delegating to
      the fitlib renderers; `show_plot`/save args via `_save_img_flag`.
- [ ] `FitResults.plot_param_evolution(...)` — SbS per-parameter-vs-time
      (successor of the `results_to_df` → `plt_fit_res_pars` chain; varied
      params by default).
- [ ] Upgrade `plot_residuals` to use real axes now that they're available.
- [ ] `File.plot_fit` / `File.plot_param_evolution` sugar.
- [ ] PlotConfig: renderers already accept `config`; thread the owning file's
      `plot_config` through (keep `figsize`-style overrides minimal).

## Phase 6 — Docs, tests, release hygiene

- [ ] Update `docs/design/repo_architecture.md` (ownership contract),
      `docs/design/ui.md` cross-refs, export-related docstrings.
- [ ] Reconcile `llms.txt` / `AGENTS.md` guidance (auto-export layout,
      accessor story).
- [ ] Notebooks: 12 (accessors), 11/20 (saving/export) — check for legacy
      layout or `save_*_fit` mentions.
- [ ] TODO.md: drop both items on completion, remove `[ACTIVE]`; note partial
      progress on v1.0.0 item 4 (remaining shims: sweep.py legacy branch,
      plot.py int-mapping helper).
- [ ] Version bump: breaking layout change + schema bump → `0.14.0`.
- [ ] Archive decision: this file likely warrants `docs/design/archive/`
      (ownership contract is durable) — ask at completion per protocol.

## Open implementation points (resolve while working, not user-blocking)

- Exact "latest slot" tie-break when the same file/model/fit_type was fit
  multiple times in one session (history order; consider a `history_key` sort).
- Whether `Project.fit_2d`'s forced `show_output=0` block survives the Phase 4
  rewrite or collapses into the uniform template.
- `MCMCResult` construction from slot: ci table column fidelity after HDF5
  round-trip (dtype/index).
