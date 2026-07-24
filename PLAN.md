# Active Plan

## SbS per-slice plotting: consolidate live and archive paths

### Background

`FitResults.plot_sbs_slices` (added this session, `fit_results.py`) renders
per-slice SbS panels from the persisted `SavedFitSlot`
(`observed`/`fit`/`fit_ini`/`components`, schema 6+) — no live `Model`/`File`
needed. `File.plot_sbs_slices` / `utils/sbs.py:plot_sbs_slices` still exist as
a **second, independent implementation** that re-evaluates live through
`model_sbs` (snapshot/restore dance on `model_sbs.lmfit_pars`,
`fitlib.plt_fit_res_1d` evaluation via `par_ini`/`par_fin`/`args`). That's the
one-implementation-per-fit-type pattern already achieved for MCMC (commit
`f898778`) and for `plot_fit`/`plot_param_evolution` (thin `File.*` sugar
over `FitResults.*`) — SbS per-slice plotting is the last holdout with two
copies that can silently drift apart.

Key fact making consolidation possible: `File.fit_slice_by_slice` **always**
appends the SbS slot via `_append_sbs_slot` (`trspecfit.py:~3250`) whenever
`stages >= 1`, regardless of `show_output`. By the time a user calls
`plot_sbs_slices()` afterward, the archived slot already holds everything
the live path re-derives. The live implementation is therefore provably
redundant except for one feature: `save_path` PNG export.

### Steps

- [ ] **1. Extract the rendering primitive.** Pull the matplotlib
      figure-building core of `FitResults._plot_fit_1d` (observed + fit +
      components + residual, ROI dashed lines, gold dotted init overlay)
      out of `fit_results.py` into `utils/plot.py`, as a plain
      array-in/figure-out function (naming consistent with that module's
      existing `plot_1d`/`plot_2d`/`plot_grid`). `FitResults._plot_fit_1d`
      and the new `FitResults.plot_sbs_slices` become thin callers: resolve
      slot → extract/slice arrays → resolve config → call the utility.
      Rationale: `fit_results.py` is the inspection/query layer per its own
      module docstring; heavy rendering code belongs in `utils/plot.py`.
      - **Figure lifecycle, precise.** End the utility with
        `_finalize_plot(save_img, save_path, dpi_save)` (`utils/plot.py:850`
        — save-then-show/close, on the implicit *current* figure, exactly
        like `plot_1d`/`plot_2d` already do), not the current hand-rolled
        `if show_plot: plt.show() else: plt.close(fig)`. This is what makes
        `save_path` (step 3) come out correctly ordered instead of risking
        a save-after-close or save-of-the-wrong-figure.
      - **`save_img=-2` needs an explicit branch, not `_save_img_flag`.**
        `_save_img_flag(save=False, show=False)` falls through to `return
        0`, and `_finalize_plot` treats `0` as "show" — but its own
        docstring says it "never returns -2" because it assumes callers
        skip the plot helper entirely when both flags are false. That
        assumption doesn't hold here: `save_path=None, show_plot=False` is
        a real, common call (every suppressed-display test/batch fit), and
        today's `_plot_fit_1d` closes without showing in that case. Naively
        computing `save_img = _save_img_flag(save=save_path is not None,
        show=show_plot)` would silently turn that into `plt.show()`
        instead, leaking open figures across batch loops. Fix: `save_img =
        -2 if save_path is None and not show_plot else
        _save_img_flag(save=save_path is not None, show=show_plot)` —
        `-2` is exactly the value CLAUDE.md's own testing convention
        already uses for "no save, no show, just close."
      - **But it must still return the built `Figure`** — unlike
        `plot_1d`/`plot_2d` (`-> None`). `tests/test_fit_history.py` has
        several call sites doing `fig = FitResults._plot_fit_1d(..., show_plot=False);
        ax_fit = fig.axes[0]; ...` — they inspect the object directly after
        it's already closed, which works today because `plt.close(fig)`
        only detaches a figure from pyplot's registry, it doesn't destroy
        the object or its artists. Verified `_finalize_plot`'s implicit
        `plt.close()` has the same non-destructive behavior, so this is
        safe to preserve, but the new utility's signature can't mirror
        `plot_1d`/`plot_2d` exactly — it needs to return `fig`.

- [ ] **2. Extract SbS-specific helper logic into `utils/sbs.py`.** Slice-index
      resolution/validation (`slices=None` → all, bounds-check otherwise)
      and per-slice title formatting (`{base_title} — slice {i} (t = ...)`)
      are SbS business logic, not generic rendering or FitResults
      orchestration — they belong next to the other SbS-specific helpers
      already in that module (`extract_sbs_seed_template`,
      `prepare_sbs_model_for_slice`). Both `FitResults.plot_sbs_slices` and
      (until step 5) the live implementation import them from there, so the
      two paths can't silently diverge on validation/formatting in the
      interim.
      - **Import lazily.** Traced the import graph: `utils.sbs → fitlib →
        spectra → mcp`, none of which import `trspecfit.trspecfit` or
        `fit_results` at module top level, so no actual cycle. But
        `fit_results.py` deliberately imports `fitlib` *locally inside*
        `plot_fit`/`plot_mcmc` (not at module top) to keep a bare
        `FitResults()` construction cheap — `utils/sbs.py` top-level-imports
        `fitlib`, which pulls in matplotlib/IPython/multiprocessing. Import
        the new slice/title helpers the same way: locally inside
        `plot_sbs_slices`, not at `fit_results.py`'s module top, to preserve
        that existing discipline rather than undercut it.

- [ ] **3. Close the `save_path` gap.** Add PNG-export support to
      `FitResults.plot_sbs_slices` using the existing `utils.plot.img_save`
      / `_save_img_flag` primitives (same mechanism `utils/sbs.py`'s live
      version already uses). Filenames use a fixed `f"{s_i:06d}.png"`
      convention — no configurable format parameter (see "Remove
      `Project.da_slices_fmt`" below for why). This makes archive-loaded
      SbS results just as PNG-exportable as a live session — closing the
      asymmetry, not just documenting it.
      - Explicitly **not** in scope here: adding `save_path` to
        `plot_fit`/`plot_mcmc`/`plot_param_evolution`/`plot_residuals`. None
        of those currently support it; extending them is a separate,
        broader API-consistency decision, not a prerequisite for this one.

- [ ] **4. Make `File.plot_sbs_slices` pure sugar.** Replace its body with
      `self.p.results.plot_sbs_slices(file=self, model=model,
      slices=slices, show_init=show_init, save_path=save_path,
      show_plot=show_plot)` — same pattern as
      `File.plot_fit`/`plot_mcmc`/`plot_param_evolution`. No filename-format
      parameter to thread through (see below).

- [ ] **4b. Remove `Project.ext`, `da_fmt`, and `da_slices_fmt`** — all
      three are dead in the same way: set in `_set_defaults`, printed in
      the settings dump, and otherwise unconsumed (`da_slices_fmt`'s one
      real consumer, `utils/sbs.py:275`, is deleted by step 5 anyway).
      Checked: no `project.yaml` anywhere in the repo (tests, examples) sets
      any of the three, so no fixture updates needed. Confirmed still-live
      siblings in the same docstring group, kept as-is: `num_fmt`/`delim`
      (threaded into CSV export at `trspecfit.py:466-467`, `:2735-2736`).
      All three are YAML-configurable like every other `_set_defaults`
      attribute (`_load_config`'s generic `setattr`, `trspecfit.py:829`),
      so removal is a real breaking change:
      - Remove `self.ext = ".dat"`, `self.da_fmt = "%04d"`, and
        `self.da_slices_fmt = "%06d"` (`_set_defaults`, ~lines 271/274/275),
        their docstring mentions (~line 198), and their print lines
        (~772/775/776).
      - Add all three to `_load_config`'s `_removed_keys` dict (~line 810),
        matching the existing `auto_export`/`path_results` precedent — fail
        loudly on a stale config instead of silently warning-and-ignoring.

- [ ] **5. Retire the live-evaluation implementation.** Delete
      `plot_sbs_slices` from `utils/sbs.py` (the `model_sbs` re-evaluation,
      the `lmfit_pars` snapshot/restore dance — both become unnecessary,
      since nothing evaluates the live model for this anymore). Leave the
      unrelated multiprocessing helpers in that file untouched
      (`sbs_worker_init`, `sbs_fit_one_slice`, `prepare_sbs_model_for_slice`,
      `extract_sbs_seed_template`).

- [ ] **6. Update tests** (`tests/test_fit_side_effects.py`,
      `TestPlotSbsSlices`):
      - `test_raises_on_model_mismatch` currently asserts `match="most
        recent"` — the live error text said "Only the most recent
        fit_slice_by_slice() run is available". Once this routes through
        `FitResults._latest_slot`'s generic filter-miss error ("No sbs fit
        results (model=...). Run fit_slice_by_slice() first."), that
        substring is gone — update the assertion (still a clear
        `ValueError`, just reworded).
      - `test_raises_without_live_results` (`match="fit_slice_by_slice"`)
        and `test_raises_on_out_of_range_slice` (`match="out of range"`)
        should keep passing unchanged — verify.
      - `test_does_not_mutate_model_params` becomes trivially true (no model
        evaluation happens at all) — keep it as a regression guard anyway.
      - Add a new archive-path test mirroring
        `test_full_range_plot.py`'s pattern: save an SbS fit archive, load
        with `FitResults.load` (no live `File`), call
        `loaded.plot_sbs_slices(save_path=...)`, assert the PNGs land with
        the same naming convention as the live path.
      - `test_save_path_writes_one_png_per_slice` currently builds expected
        filenames via `project.da_slices_fmt % s` — update to assert the
        fixed names directly (`000000.png`, `000002.png`).
      - **Expand `TestRemovedConfigKeys`** (`tests/test_fit_side_effects.py:117`,
        currently `@pytest.mark.parametrize("key", ["auto_export",
        "path_results"])`) to also cover `"ext"`, `"da_fmt"`, and
        `"da_slices_fmt"` — same `test_removed_key_raises` mechanism
        (writes each key to a scratch `project.yaml`, asserts
        `ValueError` matching `"'{key}' was removed"`), just widening the
        existing parametrization rather than a new test.
      - **New test: older SbS fit retrievable by model name.** Fit
        `fit_slice_by_slice` twice on the same file with two different
        model names, then call `file.plot_sbs_slices(model=<older name>)`
        and confirm it succeeds and renders the older fit — today's live
        path can only ever see the single most recent `results_sbs`/
        `model_sbs` and raises on any other name;
        `FitResults._latest_slot` scans the full `_fit_history` instead.
        Deliberate improvement, not a side effect — needs a test asserting
        it, not just a changelog line.

- [ ] **7. Update docstrings and current design docs** (not archived ones —
      `docs/design/archive/*` is a preserved historical record per past
      guidance and stays untouched):
      - `File.plot_sbs_slices` (`trspecfit.py:~4412`) still says it reads
        live `results_sbs` and that archive plotting lacks `save_path` —
        both wrong post-refactor. This one matters beyond source hygiene:
        `docs/api/trspecfit.rst:75` explicitly `automethod::`-documents it,
        so a stale docstring ships straight into the built docs.
      - `FitResults.plot_sbs_slices` (`fit_results.py:970`) needs its
        docstring updated for the new `save_path` parameter.
        `docs/api/fit_results.rst:11-12` uses `autoclass:: FitResults
        :members:`, so this is picked up automatically — no `.rst` edit
        needed, just accurate docstring content.
      - `docs/design/repo_architecture.md` (~line 205) currently describes
        `File.plot_sbs_slices` as rendering "from the live `results_sbs`
        state (live-session only)" — update to describe the
        sugar-over-`FitResults` pattern, matching how `plot_mcmc`/`plot_fit`
        are already described there.

- [ ] **8. CHANGELOG entry** once implemented, explicitly covering:
      - The capability addition (archive-portable SbS per-slice viewer,
        now with `save_path` support).
      - **The visual style change**: `File.plot_sbs_slices` switches from
        the live `fitlib.plt_fit_res_1d` look (full x-range, one combined
        panel, "5×residual" overlay, dashed ROI lines) to the archive
        style (cropped x-range, separate residual subplot) — a real,
        user-visible change to existing behavior, not just an internal
        refactor (see the visual-regression section below).
      - The error-message wording change on model-mismatch/no-results.
      - The older-SbS-fit-by-model-name retrieval improvement.
      - The `Project.ext`/`da_fmt`/`da_slices_fmt` removal (breaking,
        `_removed_keys`-guarded).

### Visual-regression baseline (captured pre-implementation)

Ran example 01's workflow (`fit_baseline` + `fit_slice_by_slice`, serial),
saved PNGs for slices [0, 12, 24] to the git-ignored
`fit_results/_sbs_plot_regression_baseline/{live,archive}/` (see
`capture_sbs_baseline.py` in scratch — rerun-able, not committed).

**Finding: the two paths render in genuinely different styles today** — not
just missing `save_path`. Live (`utils/sbs.py`, via `fitlib.plt_fit_res_1d`):
full x-range, one combined panel, residual overlaid as "5×residual", dashed
ROI lines. Archive (`FitResults.plot_sbs_slices`, via `_plot_fit_1d`):
cropped to the fit window, separate residual subplot, no ROI lines. Preferred
style: the archive one (legend styling to be revisited later, separately).

**What "matching" means post-implementation** — there is no "live path" left
to compare after step 4: `File.plot_sbs_slices` becomes pure sugar with zero
live evaluation, so calling it vs. calling `FitResults.plot_sbs_slices`
directly runs the identical function, not two implementations that happen to
agree. The only meaningful regression check is **archive-before vs. after**
(same renderer, relocated to `utils/plot.py` + given `save_path`) — expected
to match pixel-for-pixel. The old live-style PNGs are a discarded look, not
a reproduction target.

### Open questions / risks flagged during planning

- **No blast radius beyond this repo's own tests/docs.** Grepped the whole
  tree for `plot_sbs_slices`/`usbs.`/`utils.sbs`; only hits are
  `fit_results.py`, `trspecfit.py`, `utils/sbs.py`,
  `tests/test_fit_side_effects.py`, and the two design docs above — no
  notebooks or examples call it directly, so no example/notebook updates
  needed.
- **2D fits' missing component decomposition is a separate, harder
  problem** (schema note: components are "never present for 2d" — `eval_2d`
  doesn't retain per-component curves at all, live or archived). Out of
  scope for this plan; tracked separately as the declined
  "Compiled-plan/GIR persistence" idea.
