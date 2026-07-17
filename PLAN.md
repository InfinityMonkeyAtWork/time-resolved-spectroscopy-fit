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

## Phase 2 — Relocate accessors to `FitResults`, slot-backed — DONE

- [x] `FitResults.get_fit_results / get_correlations / get_conf_intervals /
      get_mcmc(file=..., model=..., fit_type=...)` reading the latest matching
      slot via `_latest_slot` (find() is history-ordered; last match wins,
      mirroring the live-model overwrite convention). Accessors return
      copies so callers can't mutate slot state. "Not fit yet" raises
      ValueError with the legacy "Run fit_x() first" message shape.
- [x] `get_mcmc` builds `ulmfit.MCMCResult` from the slot payload;
      `MCMCResult.acceptance_fraction` widened to `np.ndarray | None`
      (None for slots loaded from schema-2 archives).
- [x] `File.get_*` are thin delegates to `self.p.results` (fit_type kwarg
      unchanged; optional `model=` filter added). `File._result_model`
      deleted. Behavior changes: (a) covariance-less fits now raise from
      `get_correlations` instead of returning an identity-with-zeros
      matrix; (b) fits on Files without `data` (fingerprint source) record
      no slot, so accessors raise — data_base-only test fixtures updated
      to set `file.data`.
- [x] SbS accessor coverage: get_fit_results serves the wide per-slice
      frame; correlations/conf_intervals/mcmc serve slice 0 (documented).
- [x] Notebook 12 compatibility: call signatures unchanged (fit_type
      kwarg); stages=2 default → leastsq covar → correl present. Full
      notebook re-run deferred to the Phase 6 docs/examples pass.

## Phase 3 — Purify the conversion layer (fitlib) — DONE

- [x] `results_to_df`: stripped CSV writing and plotting → pure
      results-list → DataFrame conversion (dropped `save_df`/`save_path`/
      `num_fmt`/`delim`; kept `config` only for the `y_label` column name).
- [x] `results_to_fit_2d`: stripped `save_2d` CSV writing → pure
      reconstruction (slot already stores the `fit` array).
- [x] `_save_sbs_fit_legacy` compensates inline (CSV writes + the
      varied-only `plt_fit_res_pars` flag logic) — byte-identical output,
      pinned by the slow export-parity tests; the whole block dies in
      Phase 4.
- [x] `plt_fit_res_1d` / `plt_fit_res_2d` / `plt_fit_res_pars` remain the
      pure renderers (flag-driven via `_save_img_flag`).

## Phase 4 — Route auto-export through slots; delete legacy path — DONE

- [x] `fit_slice_by_slice` / `File.fit_2d` / `Project.fit_2d`: display
      (`show_output>=1`) renders inline from the just-captured slot via new
      `File._display_fit_2d_maps` / `_display_sbs_fit` helpers (save_img=0);
      export (`auto_export`) calls `export_fit(self.p.path_results, ...,
      overwrite=True)` — the slot exporter, rooted at `path_results` so the
      configured results dir (and test redirection) is respected. Skip-
      entirely guard preserved (`TestPlotHelperSkipped` green). The
      `_append_*_slot` methods now return the slot (None for mocked/
      no-data fits, which then skip display/export gracefully).
- [x] Fit-time diagnostics unchanged in place: per-slice CSVs/PNGs and
      `fit_wrapper`'s per-stage CSVs stay under `model_path()`
      (`{path_results}/{file}/{fit_type}/{model}/`). Deviation from the
      original "new destination" note: the export slot-dir name is
      snapshot-dependent (hash suffix), so it can't be computed mid-fit,
      and these are fit diagnostics, not results.
- [x] Deleted `_save_sbs_fit_legacy`, `_save_2d_fit_legacy`, `save_sbs_fit`,
      `save_2d_fit`. Repo grep clean; `Project.fit_2d`'s PNG-grid display
      replaced by per-file inline slot maps (works without auto_export now).
- [x] Tests: `TestVerboseDisplayWithoutExport` semantics unchanged;
      `test_2d_legacy_saver_creates_its_directory` → slot-tree + refit-
      overwrite tests; `test_export_fits_parity.py` repurposed to
      auto-export ≡ explicit-export tree parity; test_file legacy-saver
      validation tests → absence test; project-fit lifecycle test uses
      `export_fit`.
- [x] CHANGELOG `[Unreleased]` section written (breaking layout, breaking
      get_correlations, removals, schema 3, accessor relocation);
      `repo_architecture.md` save/export section updated.

## Phase 5 — Explicit plotting API

Design settled 2026-07-16 (session notes): three pieces of information the
SbS plotting path used are unavailable from slots — vary flags (lost),
axes (persisted but discarded at the FitResults layer), and PlotConfig
(never persisted, by choice). Vary is slice-invariant by construction
(one model, one vary set, no mid-loop hook; serial ≡ parallel pinned by
test_gir_integration). YAML-derived capture was considered and rejected:
the runtime state diverges from the YAML (default SbS seeds from the
baseline *fit*; users mutate models between load and fit), so slots
snapshot the model *as fit*. Model rehydration stays deferred.

### 5a — schema-3 additions (still unreleased; no extra bump) — DONE

- [x] SbS **shared param metadata** frame `[name, vary, min, max, expr]`
      (`SavedFitSlot.params_meta`, sbs-only; captured from slice-0 result
      params, column-aligned with the wide frame). `_display_sbs_fit` now
      reads vary from it (fully slot-driven, live-result dependency gone).
- [x] SbS **per-slice stderr** wide frame (`SavedFitSlot.params_stderr`;
      NaN where absent; `ulmfit.list_of_par_stderr_to_df`).
- [x] **`fit_settings` provenance dict** on all fit types
      (`fit_io.build_fit_settings`, JSON attr, not in `history_key`).
      Full scope incl. MC settings (gotcha: `MC` stores `use_mc` as
      `.use_emcee`); worker counts deliberately excluded.
- [x] Round-trip + capture + v2-tolerance tests; schema doc updated
      (params_meta / params_stderr / fit_settings sections); changelog.

### 5b — axes retention + plot methods — DONE

- [x] `FitResults(slots=..., files=...)`: fingerprint-keyed provider lookup
      (`_files_by_fp`); `load` passes the archive's `SavedFile`s,
      `Project.results` the live `File`s (duck-typed `.energy`/`.time`;
      live Files also give `.plot_config`). Files that can't fingerprint
      (no data) are skipped — they produced no slots. Missing lookup →
      index-based fallback. `_axes_for` mirrors `fit_io._slot_axes`
      cropping but tolerates missing providers/axes.
- [x] `FitResults.plot_fit`: 2d/sbs → `fitlib.plt_fit_res_2d` on slot
      arrays + real axes (fitlib imported lazily — the package `__init__`
      imports fit_results, so a top-level import would cycle); baseline/
      spectrum → direct observed/fit + residual panels vs energy.
- [x] `FitResults.plot_param_evolution`: varied-only default via
      `params_meta` (all params for schema-2 archives), explicit `params=`
      with KeyError on unknown names, silent no-op when nothing varied.
- [x] `plot_residuals` upgraded to real axes (1D energy x-axis, 2D
      imshow extent); index fallback preserved.
- [x] `File.plot_fit` / `File.plot_param_evolution` sugar. The fit-time
      `_display_*` helpers were deleted — fit methods display through the
      plot API, so the fit-time figure equals what the API reproduces.
- [x] Config resolution: explicit `config=` > live `plot_config` >
      `PlotConfig()`. Changelog updated.

## Phase 6 — Docs, tests, release hygiene — DONE

- [x] `docs/design/repo_architecture.md`: fit_results.py section rewritten
      around the ownership contract (accessors, plot API, axes providers);
      save/export section was updated in Phase 4. `docs/design/ui.md` had
      no affected content.
- [x] `llms.txt`: plot_fit added to the workflow snippet + a line stating
      the get_*/plot_* accessors read persisted records and work on loaded
      archives. AGENTS.md had no affected content.
- [x] Notebooks greped for legacy layout / removed APIs: only notebook 11
      mentions export artifacts, and its description matches the slot
      exporter (which is unchanged). Accessor signatures unchanged →
      notebooks 12/01/03/04/20/21 compatible. Full notebook re-runs left
      to the release flow (`docs/ai/check-example`).
- [x] API docs: `docs/api/trspecfit.rst` gained a "Results, Plotting, and
      Persistence" section (dropping the deleted save_sbs_fit/save_2d_fit
      entries that broke autodoc); new `docs/api/fit_results.rst` documents
      `FitResults` + `MCMCResult`. `make -C docs html` warning-free.
- [x] TODO.md: both `[ACTIVE]` items removed — replaced by a slim deferred
      item (typed result object / raw `result[1..4]` internal cleanup);
      v1.0.0 item 4 updated (legacy savers gone; sweep.py + plot.py shims
      remain); trace-fitting item notes params_stderr now persisted.
- [x] Version bump: 0.13.1 → 0.14.0 (breaking auto-export layout +
      schema 3). CHANGELOG `[Unreleased]` section already complete.
- [ ] Archive decision: ask user — move this plan into
      `docs/design/archive/` (ownership contract is durable) or let the
      changelog stand; then clear PLAN.md per protocol.

## Open implementation points (resolve while working, not user-blocking)

- Exact "latest slot" tie-break when the same file/model/fit_type was fit
  multiple times in one session (history order; consider a `history_key` sort).
- Whether `Project.fit_2d`'s forced `show_output=0` block survives the Phase 4
  rewrite or collapses into the uniform template.
- `MCMCResult` construction from slot: ci table column fidelity after HDF5
  round-trip (dtype/index).
