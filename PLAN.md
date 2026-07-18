# Active Plan: remove auto-export; typed fit-result object

Branch: `model-vs-fitresult` (continuation). Phases 1–6 (results ownership
boundary, plotting disentanglement, schema 3, plot API) are complete and
archived in
[docs/design/archive/results-ownership-and-plotting.md](docs/design/archive/results-ownership-and-plotting.md).
Phases 7–8 emerged from the post-completion review discussion (2026-07-17)
and complete the same principle, in the same breaking release (0.14.0).

## Decisions (settled with user, 2026-07-17)

1. **Remove `Project.auto_export` entirely.** Fits compute, display (per
   `show_output`), and capture slots — they never write to disk. The
   original justification for auto-export (results existed only as files)
   died with the slot architecture; persistence is the explicit
   `save_fit(s)` (HDF5) / `export_fit(s)` (CSV/PNG tree) pair. No new
   method names — the save-vs-export split already covers it.
2. **Everything auto-export used to write must be reproducible** from
   what the slot persists, or via a documented explicit call (the Phase 7
   reproducibility checklist below).
3. **Keeps**: `save_baseline_fit` / `save_spectrum_fit` stay as explicit
   API (their component-decomposed `fit_1d.csv` is the one artifact slots
   can't reproduce — only the total fit curve is persisted). They take
   explicit paths, as do all persistence calls.
4. **One output root** (settled 2026-07-17): the `./fit_results/<name>/`
   family already used by explicit `save_fits` / `export_fits` defaults.
   `Project.path_results` and `File.model_path` are removed — after the
   de-wiring nothing in the package writes there, so keeping them would
   preserve a dead second convention. `Project.name` default changes
   from `"test"` to `"my_project"` — a clear placeholder that stops
   `fit_results/test/` / `test.fit.h5` from colliding with test-suite
   naming (pytest traversal, glob confusion). The per-example naming
   convention stays with the "Revisit default Project.name" TODO item.
5. **`plot_*` saving convention**: `save_path=None` means display-only
   (consistent across the plot API); saving always requires an explicit
   path. No default save locations for diagnostics.
4. **Phase 8 scope is internal-only**: replace the raw
   `[par_ini, par_fin, conf_ci, emcee_fin, emcee_ci]` list with a typed
   class. No archive-schema change, no public-API change — this branch
   removed every user-facing reader of `model.result`, which is what
   makes the cleanup safe now.

## Phase 7 — remove auto-export; fits never write

### Knob removal

- [x] Delete `Project.auto_export` (attribute, `_set_defaults`,
      `project.yaml` default). Check how YAML parsing treats the removed
      key — an existing `project.yaml` with `auto_export:` must fail or
      warn clearly, not be silently ignored (grep example/test yamls).
- [x] Delete `Project.path_results` and `File.model_path` (dead once the
      fit methods stop writing): update `Project.describe`, the tests
      that redirect `path_results`, and any `project.yaml` key handling.
      Single default output root remains `./fit_results/<name>/`.
- [x] `Project.name` default `"test"` → `"my_project"` (placeholder that
      can't be mistaken for test-suite artifacts). Grep tests/docs/
      examples for reliance on the old default.
- [x] Fit methods drop all export wiring:
      - `fit_baseline` / `fit_spectrum`: the `save_baseline_fit` /
        `save_spectrum_fit` auto-calls and `fit_wrapper(save_output=...)`.
      - `fit_slice_by_slice`: the post-fit `export_fit` call, the serial
        per-slice PNG block, per-slice `save_output`, and the worker args
        that exist only for mid-loop IO (`auto_export`, `path_slice`,
        `plot_config`) — leaner spawn payload, faster hot path.
      - `File.fit_2d` / `Project.fit_2d`: the post-fit `export_fit(s)`
        calls and `save_output` wiring.
      - Display gating (`show_output`) is untouched.
- [x] `fitlib.fit_wrapper`: remove `save_output` / `save_path` /
      `num_fmt` / `delim` params, the CSV/txt dump block, and the emcee
      figure *save* path (walker/corner figures become display-only via
      `show_output`). Fit methods drop their `num_fmt`/`delim`
      setdefaults for fit_wrapper.

### Reproducibility checklist (what auto-export wrote → where it lives now)

- [x] `*_par_ini.csv` / `*_par_fin.csv` → slot `params`
      (`init_value`/`value`/`stderr`/bounds/`vary`/`expr`); exported as
      `params.csv`. ✓ nothing to add.
- [x] `*_conf_ci.csv`, `*_emcee_flatchain.csv`, `*_emcee_ci.csv` → slot
      `conf_ci` / `mcmc` payload; exported under the slot dir. ✓
- [x] `*_emcee_walker_acceptance_ratio.png`, `*_emcee_corner_plot.png` →
      add `FitResults.plot_mcmc(file=..., model=..., fit_type=...,
      show_plot=...)` rendering the corner plot (from persisted
      `flatchain`) and per-walker acceptance (from persisted
      `acceptance_fraction`), with `File.plot_mcmc` sugar — turnkey
      reproduction from live sessions *and* archives.
- [x] SbS per-slice PNGs → new `File.plot_sbs_slices(model=...,
      slices=None, show_init=True, save_path=None, show_plot=...)`,
      logic in `utils/sbs.py`; uses live `results_sbs` (per-slice
      `par_ini` + component decomposition via `plt_fit_res_1d`), so it
      can do *more* than the old auto-PNGs. Live-session only —
      document. `save_path=None` → display-only; deliberately NOT part
      of `export_fits` (export stays slot-fed and archive-reproducible;
      per-slice diagnostics need live inputs and would flood the tree).
- [x] SbS per-slice `*_par_ini.csv` → accepted loss: re-derivable from
      the persisted `fit_settings` seeding recipe; document in changelog.
- [x] `lmfit.fit_report` text dumps → accepted loss: contents (params,
      stderr, correlations, metrics) all persisted; document.
- [x] Baseline/spectrum component-decomposed `fit_1d.csv` → explicit
      `save_baseline_fit` / `save_spectrum_fit` (kept, no longer
      auto-called).
- [x] 2D/SbS result trees → explicit `export_fit(s)` (unchanged).

### Tests / docs

- [x] `tests/test_auto_export.py` reshaped: "fits write nothing" becomes
      the unconditional default; explicit save/export tests remain; the
      display/silent guardrail matrix (`TestPlotHelperSkipped`,
      `TestVerboseDisplayWithoutExport`) survives with `auto_export`
      references removed. `make_project(auto_export=...)` helper param
      goes; export-parity tests re-anchor on two explicit exports.
- [x] New tests: `plot_mcmc` (slot-backed, incl. loaded archive),
      `plot_sbs_slices` (live; raises helpfully without `results_sbs`).
- [x] Docs: llms.txt headless section shrinks (`show_output=0` is the
      only knob — no-write is default); repo_architecture.md auto-export
      paragraph rewritten (fits never write; explicit save/export;
      diagnostics on demand); changelog breaking entry; grep notebooks +
      example `project.yaml`s for `auto_export`.

## Phase 8 — typed fit-result object (internal)

- [x] Introduce a small class (named `FitOutput` at implementation; in
      `utils/lmfit.py` — `fitlib` imports `spectra`→`mcp`, so the class
      lives below both) with named fields `par_ini`, `par_fin`,
      `conf_ci`, `emcee_fin`, `emcee_ci` replacing the raw 5-list.
      `fit_wrapper` returns it. Frozen dataclass; `par_fin` is annotated
      via a TYPE_CHECKING-only `TypedMinimizerResult` shim (lmfit sets
      result attributes dynamically, invisible to pyright).
- [x] Update all internal consumers: the four fit methods,
      `_append_*_slot` capture, `results_sbs` per-slice entries, the
      MCMC-payload builder, and `Project.fit_2d`'s `SimpleNamespace`
      stand-in (now a real `FitOutput` wrapping a minimal
      `MinimizerResult`).
- [x] No list-index back-compat: verified by grep that nothing outside
      the package (notebooks, docs) indexes `model.result[...]` or
      `results_sbs[i][...]`; mocked-result test (`test_file.py`) now
      builds a placeholder `FitOutput`.
- [x] Closes the "Unified results object / raw `result[1..4]` cleanup"
      TODO item.

## Completion

- [ ] Changelog entries for both phases; docs build; full + slow suites.
- [ ] Extend the archive doc (results-ownership-and-plotting.md) with a
      Phases 7–8 section; clear PLAN.md; un-tag TODO.
