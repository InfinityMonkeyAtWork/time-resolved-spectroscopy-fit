# Active Plan

## Archive-backed plotting and renderer consolidation

> **Superseded in part.** The `PlotConfig` design below — file-level, captured
> with the first completed slot, one config per `SavedFile`, non-retroactive —
> is superseded by `docs/design/fit_archive_principles.md`, which specifies a
> single **project-owned** config resolved at render time. **The live-session
> half of that design landed 2026-08-03** (branch `fit-archive-schema-7`):
> `Project.plot_config` is the one real config, `File.plot_config` and
> `PlotConfig.from_project` are deleted, and `FitResults` receives the
> resolving config at construction. What remains of the config work is the
> schema-7 on-disk payload (serialization + decode on load). Steps 1, 2, and
> 7 below still need rewriting against that reality before being worked (step
> 1's frozen-config regression tests and step 7's "first completed fit wins"
> rule no longer apply). The renderer-consolidation work — steps 4, 5, 6, 8 —
> is unaffected and remains valid. Do not implement the config sections as
> written.

### Goal

Bring production plotting to three explicit boundaries:

1. All direct Matplotlib and `corner` rendering logic lives in
   `src/trspecfit/utils/plot.py`.
2. Every completed-fit plot gets its scientific content and default
   presentation config from persisted result records: `SavedFitSlot` for
   fit-specific arrays/metadata plus an immutable file-level `SavedFile`
   snapshot for full data, axes, and `PlotConfig`. No completed-fit plot reads
   a live `File`, `Model`, `Model.result`, `results_sbs`, or mutable provider
   after fit completion.
3. Pre-fit inspection and setup methods may evaluate live state (`describe`,
   `describe_model`, `Model.plot_*`, `Component.plot`, `set_fit_limits`,
   `define_baseline`, simulator inspection), but they pass plain arrays and a
   `PlotConfig` into the shared renderers rather than owning Matplotlib code.

`FitResults` remains the query/orchestration layer: select slots, assemble the
array views and titles, resolve explicit presentation options, and delegate
rendering. `File.plot_*` methods remain thin convenience wrappers.

### Scope decision: slot plus file-level archive payload

Do not duplicate raw data and axes into every `SavedFitSlot`. The archive
schema already has the correct normalized ownership:

- `SavedFitSlot`: observed fit view, final fit, initial fit, components,
  parameters, metrics, MCMC, selection, and fit metadata.
- `SavedFile`: full raw data, energy/time/aux axes, file identity, one
  file-level default `PlotConfig`, and the slots belonging to that file.

For this plan, "slot-backed plotting" means the persisted `SavedFitSlot` plus
its persisted file-level `SavedFile`, never a live authoring object. This keeps
archives self-contained without multiplying large file arrays by the number of
fits. Capture the file's `PlotConfig` once, with the first completed slot for
that file identity. Later mutation of the live `File.plot_config` does not
change existing result defaults; callers may deliberately restyle any result
with an explicit `config=` override.

Persist the config as a small JSON-safe file-group payload. This is an additive
HDF5 schema 7 change; schema 2 through 6 archives have no saved config and
therefore resolve to `PlotConfig()`. Keeping one config per `SavedFile`, rather
than a config history per slot, makes the default stable across in-memory and
loaded results while avoiding presentation duplication in every fit slot.

### Steps

- [ ] **1. Add regression tests for the audited failures before changing the
      implementation.**
      - Hold `results = project.results`, mutate or replace the live
        `File.data`/`energy`/`time`, and prove every plot still uses the
        original result snapshot (or cleanly uses index axes when an archived
        axis is genuinely unavailable). Cover `full_range=True`, where the
        current code re-derives observed data from the live provider.
      - Create two named Files with byte-identical data/axes and different
        plot configs; prove each slot resolves to its own file payload and
        presentation config rather than the last fingerprint collision.
      - Give a fitted File a non-default config and prove
        `project.results.plot_fit(file="A")`, `file.plot_fit(...)`, a loaded
        archive, and `export_fits` use the same saved default. Mutate the live
        config after fitting and prove none of those defaults changes; prove
        an explicit `config=` still overrides the saved value.
      - Remove/reload/modify the fitted live Model after slot capture and prove
        `plot_fit`, `plot_sbs_slices`, `plot_param_evolution`, `plot_mcmc`, and
        `plot_residuals` do not consult it.
      - Add a source-boundary test (AST or a focused repository scan) that
        rejects production imports/calls of `matplotlib`, `pyplot`, and
        `corner.corner` outside `utils/plot.py`. Exclude tests, docs, and
        example notebooks, which legitimately inspect or demonstrate plots.

- [ ] **2. Capture and persist one immutable `SavedFile` payload per fitted
      file identity.**
      - Add a Project-owned mapping keyed by `(file_fingerprint, file_name)`.
        Capture it at the same moment the first slot for that identity is
        appended, not later when `Project.results`, `save_fits`, or
        `export_fits` happens.
      - Copy `data`, `energy`, `time`, optional `aux_axis`, and the complete
        `PlotConfig` once per file identity. Make cached arrays read-only and
        normalize the config to JSON-safe primitive values. Copy fit-limit
        metadata and fingerprint as values. Reuse `SavedFile` with `slots=()`
        as the cached payload, then use `dataclasses.replace` (or an equally
        small helper) to attach selected slots when building a `SavedProject`.
      - Route all four slot appenders through one Project helper so appending a
        slot and ensuring its file snapshot cannot drift apart.
      - Make `Project.results`, `save_fits`, and `export_fits` consume these
        cached payloads. They must continue to work if the live File is later
        mutated, renamed, removed from `Project.files`, or has its active model
        replaced. `export_fits` must also get styling from the cached
        `SavedFile`, never `_find_file_for_slot` or another live lookup.
      - Document the memory tradeoff: one raw-data snapshot and one default
        config per fitted file identity, not one copy per fit slot.

- [ ] **3. Preserve exact slot-to-file association in `FitResults`.**
      - Replace `_files_by_fp` lookup with an association that cannot collapse
        byte-identical named files. At minimum use `(fingerprint, file_name)`
        for Project-owned records.
      - On archive load, retain the `SavedFile` encountered while flattening
        each slot rather than trying to reconstruct the association from a
        fingerprint-only dictionary. This must also handle archive groups
        whose fingerprint and display name match but whose `original_path`
        differs.
      - Treat the associated `SavedFile` as the source of full axes/data and
        the default plot config, so selection cannot resolve content from one
        file and styling from another.
      - Keep `FitResults(slots=[...])` usable for synthetic/tests/orphan slots:
        missing file payload means index-axis and cropped-slot fallback, not a
        live lookup.
      - Remove all support for live `File` objects as axes/data providers.
        Update constructor types, docstrings, and internal names so the
        invariant is visible rather than conventional.

- [ ] **4. Move every production rendering primitive into `utils/plot.py`.**
      Add array/dataframe-in, figure-out utilities for:
      - 1D fit panels (existing `plot_fit_panel_1d`, expanded as needed).
      - 2D data/fit/residual panels (move Matplotlib body of
        `fitlib.plt_fit_res_2d`).
      - MCMC walker acceptance and corner diagnostics.
      - Side-by-side 1D residual comparisons.
      - Side-by-side 2D residual heatmaps.
      - Parameter-evolution plots, either as a small dedicated utility or a
        clearly documented adapter over `plot_1d`.

      Each renderer must own figure creation, axes/artists, labels, layout,
      saving, showing/closing, and return value. Prefer finalization against
      the explicit Figure rather than pyplot's implicit current figure so
      multi-figure MCMC and batch rendering cannot save or close the wrong
      object.

- [ ] **5. Remove Matplotlib ownership from orchestration and fitting modules.**
      - `fit_results.py`: keep selection, slot/file array assembly, validation,
        titles, and config resolution; replace all pyplot/corner calls with
        `utils.plot` calls.
      - `fitlib.py`: remove the pyplot import and move both fit-result renderer
        bodies. If `plt_fit_res_1d`, `plt_fit_res_2d`, or
        `plt_fit_res_pars` must remain for API compatibility, make them thin,
        Matplotlib-free adapters and document their status; otherwise grep the
        entire repo and remove/rename them coherently.
      - `utils/fit_io.py`: export PNGs by calling `utils.plot` directly rather
        than importing plotting functions from `fitlib`. Resolve its default
        config from each persisted `SavedFile`; remove the live
        `_find_file_for_slot` styling path from `Project.export_fits`.
      - `utils/sbs.py`: remove the now-unused worker-side Matplotlib backend
        switch. Workers run fitting only and must not import plotting for
        defensive side effects.
      - Confirm `rg` finds no production `matplotlib`/`pyplot` import outside
        `utils/plot.py`.

- [ ] **6. Adapt the allowed live/pre-fit entry points to shared renderers.**
      - Keep live evaluation and validation in `mcp.py`, `trspecfit.py`, and
        `simulator.py`; pass their evaluated arrays to `utils.plot`.
      - Replace `File.describe_model(detail=1)` calls through fitlib renderers
        with direct shared-renderer calls while preserving its initial-guess,
        component, fit-limit, and residual display.
      - Verify `Project.describe`, `File.describe`, `define_baseline`,
        `set_fit_limits`, `Model.plot_1d/plot_2d`, `Component.plot`, Dynamics
        inspection, and `Simulator.plot_comparison` remain live-state
        exceptions and never get routed through `FitResults`.
      - Do not add model evaluation to a renderer; utilities receive arrays,
        never `File`, `Model`, lmfit result, or spectrum-function objects.

- [ ] **7. Persist and resolve `PlotConfig` consistently across result
      renderers.**
      - Add canonical `PlotConfig` serialization helpers that emit and accept
        JSON-safe primitive values, normalize tuple-valued fields on load, and
        reject unsupported values clearly. Store that payload once on the
        schema-7 file group alongside the full data and axes.
      - Resolve config with one precedence rule everywhere: explicit
        `config=` override, then a fresh `PlotConfig` reconstructed from the
        associated `SavedFile`, then `PlotConfig()` for legacy/orphan slots.
        Never fall back to a live `File.plot_config`.
      - Make `File.plot_*` default to the archived config just like direct
        `Project.results` and loaded `FitResults` calls. Do not implicitly
        forward the File's current config; a caller who wants current or new
        styling passes it explicitly. Apply the same saved default to
        `export_fits`.
      - Define and test the supported mapping for axis labels, scales,
        directions, limits, display/save DPI, colormaps, color limits,
        reference-line style, tick size, and line/component styles.
      - Expand `plot_fit_panel_1d` so `config=` is a real styling override,
        rather than honoring only `x_dir` and `dpi_save` while hard-coding the
        rest. Bring the 2D fit renderer onto the same rules, including
        `dpi_plot`, `dpi_save`, `z_lim`, residual colormap/limits, and colorbar
        label.
      - Add `config=` to comparison/MCMC APIs only where it controls meaningful
        presentation. Keep statistical data and titles slot-derived.
      - Document the file-level "first completed fit wins" rule: changing the
        live config later does not rewrite prior or subsequent slots for that
        archived file identity. Per-call explicit overrides are the supported
        restyling mechanism.
      - Record intentional visual changes and avoid accidental changes with
        axes/artist assertions; use image comparisons only where artist-level
        checks cannot express the contract robustly.

- [ ] **8. Consolidate figure lifecycle and compatibility behavior.**
      - Preserve `show_plot=False` / `save_img=-2` behavior: build only when an
        API call needs a figure, save before close, and leave no pyplot figures
        registered after suppressed batch calls.
      - Preserve returned Figure objects where tests/users inspect them after
        close; specify return types for multi-figure MCMC output.
      - Decide compatibility for public-looking `fitlib.plt_fit_res_*` names
        after a whole-repo grep, including API docs and notebooks. Prefer thin
        forwarding wrappers for one release if removal would be a needless
        break; wrappers contain no Matplotlib logic.
      - Keep `File.plot_*` signatures and default visuals stable unless an
        additive `config=` argument is required. Document any deliberate API
        or visual change.

- [ ] **9. Update tests across live, in-memory, and loaded-archive paths.**
      - For every completed-fit plot, exercise both `Project.results` and
        `FitResults.load` and compare selected arrays, axes, labels, and config
        behavior.
      - Cover baseline, spectrum, SbS, and 2D; cropped and full-range views;
        missing optional schema-2-through-6 payloads; schema-7 config
        round-tripping; identical-file identities; saved and explicit config;
        export styling; save-only; show-only; and close-only lifecycle.
      - Keep tests on the public API (`File.plot_*`, `Project.results`,
        `FitResults.load`) except focused renderer unit tests.
      - Retain the existing no-model-mutation and archive-portable SbS/MCMC
        regression tests.

- [ ] **10. Align current documentation and guardrails.**
      - Update `CLAUDE.md`: `fitlib` owns fitting/CI/MCMC computation;
        `utils/plot.py` owns all rendering; `FitResults` owns completed-result
        plot orchestration.
      - Update `docs/design/repo_architecture.md`, `llms.txt`, API docstrings,
        and `docs/api/fitlib.rst` to match the final compatibility decision.
      - Update `docs/design/fit_archive_schema.md` for schema 7, the in-memory
        file snapshot, file-level `PlotConfig` payload, legacy fallback, and
        the `SavedFitSlot + SavedFile` result-plot contract. Do not edit
        archived design records.
      - Add a CHANGELOG entry covering immutable in-session result plots,
        renderer consolidation, config corrections, compatibility wrappers or
        removals, and any visible style changes.

- [ ] **11. Verify and close the feature.**
      - Run focused plotting/archive tests, then `pytest -q`.
      - Run Ruff check/format, mypy, and pyright using the repo environment.
      - Run the production-source Matplotlib boundary scan and a whole-repo
        grep for stale `plt_fit_res_*`, live-provider, live-`plot_config`, and
        live-result plotting references.
      - Re-run representative example plotting mechanically where practical.
      - When all steps are complete, ask whether this plan warrants an archived
        design record or whether the changelog is sufficient; then clear
        `PLAN.md` and remove the `[ACTIVE]` marker from `TODO.md`.

### Non-goals

- Persisting a separate `PlotConfig` history per slot. The one file-level
  archived default is stable; callers may pass presentation overrides.
- Adding 2D per-component decomposition; the evaluator/archive does not
  currently produce it.
- Moving direct Matplotlib use out of tests or example notebooks. The ownership
  rule applies to production package code under `src/trspecfit/`.
- Solving the broader policy for arbitrary in-place mutation of every public
  array. This plan isolates completed-fit plotting with immutable snapshots;
  the systemic mutation policy remains a separate TODO.
