# Active Plan

## `full_range` toggle for `FitResults.plot_fit`

Full design/rationale: `/home/yoyo/.claude/plans/hm-i-guess-the-eventual-simon.md`
(session-local; contents mirrored here for repo persistence).

**Goal**: close the live-vs-archive display-range gap honestly. The
archive already persists the full, uncropped `data`/`energy`/`time` once
per file (`SavedFile`). Give `FitResults.plot_fit` a `full_range=True`
mode that shows the real full data, with fit/residual/components drawn
only inside the fit window (`NaN` outside — never a fabricated padding
value) plus dashed ROI boundary lines, matching `describe_model`'s
existing visual language. `fit_baseline`/`fit_spectrum`/`fit_2d`/
`fit_slice_by_slice`'s own post-fit live display switch to this as the
new default.

- [x] **`utils/arrays.py`**: extracted `File._resolve_time_selection`'s
      body into a standalone `resolve_time_selection` function (time
      array as a plain arg); `File._resolve_time_selection` is now a
      3-line wrapper. Unit-tested in `test_arrays.py`.
- [x] **`FitResults`** (`src/trspecfit/fit_results.py`):
      `_axes_for` gained `full_range: bool = False`; added
      `_provider_for`, `_full_observed_for` (baseline: `np.mean` over
      persisted `base_t_ind`; spectrum: resolves raw time selection via
      `resolve_time_selection`; sbs/2d: `provider.data` directly), and
      `_pad_axis` (axis-parametrized `NaN`-padding helper reused for
      `fit` and `components` across all 4 fit types). `plot_fit` gained
      `full_range: bool = False` — reconstructs when possible, falls
      back to the cropped view otherwise (never raises); 2d/sbs reuse
      `fitlib.plt_fit_res_2d`'s existing `x_lim`/`y_lim` support
      unchanged. `_plot_fit_1d` stays a staticmethod, now accepts
      precomputed `observed`/`fit`/`components`/`roi` overrides (default
      path unchanged); draws dashed ROI boundary lines when `roi` is
      given; title enhanced via new `_slot_title` (file name, yaml stem,
      spectrum's time selection).
- [x] **`fitlib.plt_fit_res_2d`**: "Fit" panel title + `range_dat_fit`
      scale calc switched to `np.nanmin`/`np.nanmax` (needed for
      `full_range` mode's `NaN`-padded `fit`; identical result when no
      `NaN` present).
- [x] **`trspecfit.py`**: `fit_baseline`/`fit_spectrum` — deleted the dead
      `initial_guess` extraction + bespoke title, replaced the direct
      `fitlib.plt_fit_res_1d` call with `self.plot_fit(...)`.
      `fit_2d`/`fit_slice_by_slice`/`Project.fit_2d`/`Project.fit_baselines`
      all call `plot_fit` too (the batch `fit_baselines` call was missed
      in an earlier pass and fixed). `File.plot_fit` sugar forwards
      `full_range`.
- [x] **`full_range` promoted to a real `PlotConfig` member** (raised as a
      design question before committing: project.yaml `full_range: ...`
      was silently ignored — `_load_config` only applies known `Project`
      attributes, and neither `PlotConfig` nor `Project._set_defaults`
      had one). `config/plot.py` gained `full_range: bool = True`
      (existing precedent: `data_slice`/`x_lim`/`y_lim` are already
      "what to show" fields here, not just style);
      `Project._set_defaults` gained `self.full_range = True`.
      `FitResults.plot_fit`/`File.plot_fit`'s `full_range` param changed
      `bool = False` → `bool | None = None` (`None` = "use config"),
      resolved once via `cfg.full_range` right after `_config_for`. All 6
      live call sites now call `plot_fit`/`self.plot_fit` with **no**
      `full_range` kwarg at all — they inherit the resolved default
      instead of hardcoding `True`, so a project-wide override (or a
      per-call explicit `full_range=`) actually takes effect everywhere.
      Field-existence coverage comes free from `test_plotting.py`'s
      existing generic `test_every_field_settable_via_project` (iterates
      `dataclasses.fields(PlotConfig)`).
- [x] **Tests**: `test_fit_history.py` — 2 new `_plot_fit_1d` direct-call
      tests (NaN-gap + roi boundary lines; default-path parity); retargeted
      `test_baseline_plots_when_verbose`/`_skips_plot_when_silent` to
      `FitResults._plot_fit_1d`. New `tests/test_full_range_plot.py` —
      real-fit integration tests for all 4 fit types (reload with no live
      `File`, reconstructed-observed correctness, `NaN` outside ROI,
      values match `slot.fit` inside), plus graceful-fallback-without-
      provider. `test_arrays.py` — `resolve_time_selection` unit tests.
      `test_fitlib.py` — `plt_fit_res_2d` NaN-aware rendering test.
      `test_fit_side_effects.py::TestFullRangeConfigResolution` (replaces
      an earlier, now-obsolete systemic mock-and-assert-kwargs test class
      that checked all 6 call sites hardcoded `full_range=True` — moot
      once nothing hardcodes it): `Project.full_range` defaults `True`;
      a live baseline fit's post-fit display shows the full axis when
      config is `True` and the cropped window when set to `False`; an
      explicit per-call `full_range=False` overrides config. Verified
      the `fit_baselines` gap and the config-resolution logic each fail
      without their respective fix (temporary-revert round-trips). Full
      suite (1200 tests incl. slow), mypy, pyright, ruff all clean
      (whole-tree sweep).
- [x] **Docs**: `docs/design/fit_archive_schema.md` — noted
      `full_range=True` as a concrete consumer of the already-documented
      full-data-duplication design decision. `config/plot.py`'s
      `PlotConfig` docstring gained a `full_range` entry. No schema/
      version change (reuses already-persisted fields). Verified with a
      `sphinx -W` build.

**Status**: implementation complete, verified 2026-07-21.

**Out of scope**: no schema/wire-format change; no model rehydration /
extrapolated-fit-curve reconstruction outside the ROI (declined already
for schema-4); no reintroduction of `show_init`.

## Fix: persisted `init_value` is wrong for two-stage fits

Full design/rationale: `/home/yoyo/.claude/plans/hm-i-guess-the-eventual-simon.md`
(session-local; overwritten with this fix's plan, contents mirrored here).

**Goal**: while discussing whether the initial guess should be treated as
fit-result provenance (a different seed can land the same algorithm on a
different local minimum), found that the already-persisted
`SavedFitSlot.params` `init_value` column (baseline/spectrum/2d) is
silently wrong for `stages=2` fits — lmfit's `Minimizer.prepare_fit()`
unconditionally resets `Parameter.init_value = Parameter.value` at the
start of every stage, and stage 2 starts from stage 1's *output*, so a
two-stage result's `init_value` reflects stage 1's output, not the true
original seed. Verified directly against the installed `lmfit` source.
`stages=1` fits were already correct.

- [x] **`utils/lmfit.py`**: added `restore_true_init_values(result_params,
      par_ini)` — corrects `result_params`' `init_value` in place from
      the true seed.
- [x] **`fitlib.fit_wrapper`** (not the slot extractors — moved after
      review): calls `restore_true_init_values(par_fin_params, par_ini)`
      right after stage 2's `mini.minimize()`, before `par_fin_params` is
      printed via `lmfit.report_fit` or returned in `FitOutput`.
      `_result_params` returns `result.params` by reference (no copy), so
      this is the same object that later becomes `FitOutput.par_fin.params`
      — fixing it once at the source means the live printed report *and*
      every direct consumer of `FitOutput.par_fin` (not just code that
      passes through `_append_baseline_slot`/`_append_spectrum_slot`/
      `_append_2d_slot`) sees the true seed consistently. The three
      downstream calls in `trspecfit.py` from the first pass were removed
      as redundant.
- [x] **Tests**: `tests/test_lmfit_utils.py` (new) — direct unit tests
      for `restore_true_init_values`. `test_fit_history.py` — two
      regression tests via the persisted slot (`stages=2` seed correctly
      persisted, not stage 1's output; `stages=1` companion confirming no
      change), plus a new direct test asserting both
      `FitOutput.par_fin.params[name].init_value` and `report_fit`'s
      printed `"(init = ...)"` (captured via `capsys`, matching lmfit's
      own `.7g` format) show the true seed for the local-optimization
      stage — verified each regression test actually fails without its
      fix via temporary-revert round-trips. Full suite (1200+ tests incl.
      slow), mypy, pyright, ruff all clean.
- [x] **Docs**: `docs/design/fit_archive_schema.md` — clarified the
      `init_value` column description, pointing at `fitlib.fit_wrapper`
      as the fix location. No schema/wire-format change (same column,
      same shape, corrected values) — no version bump. Verified with a
      `sphinx -W` build.

**Status**: implementation complete, verified 2026-07-21.

**Out of scope (deliberately deferred)**: SbS initial-guess persistence
(schema-new, not a correctness fix — candidate: slice-0's true
`init_value` in `params_meta`, mirroring how `correl`/`mcmc` are already
slice-0-only); the project-level joint-fit `par_ini=None` case; any new
schema field.

**Next**: plan the seed-as-provenance schema extension — persist the true
initial guess for baseline/spectrum/2d (e.g. a `fit_ini`/`components_ini`
evaluated at `par_ini`, mirroring the schema-4 `components` pattern, so
`plot_fit` can render an initial-guess overlay archive-side) and decide
whether to fold in SbS slice-0 `init_value` at the same time.

## Persist the true initial guess (`fit_ini`) for all 4 fit types (schema 5 → 6)

Full design/rationale: `/home/yoyo/.claude/plans/hm-i-guess-the-eventual-simon.md`
(session-local; contents mirrored here for repo persistence).

**Goal**: the direct sequel to the two sections above — persist an
*evaluated* initial-guess curve so `FitResults.plot_fit` can render the
archive-side overlay that live `plt_fit_res_1d`/`plot_sbs_slices` already
show (`show_init=True`, dotted-gold line). Same shape of change as the
schema-4 `components` work: persist the data now, richer viewers later.

- [x] **`utils/lmfit.py`**: added `list_of_par_ini_to_df(results)` —
      per-slice true initial-guess values (rows=fits, columns=parameters),
      mirrors `list_of_par_stderr_to_df` but reads `result.par_ini`
      directly (unaffected by the stage-2 `init_value` fix above).
- [x] **`utils/fit_io.py`**: schema `"5"` → `"6"` (additive). `SavedFitSlot`
      gained `fit_ini: np.ndarray | None` (all 4 fit types; `None` on the
      project-level joint-fit path, same case `components` already
      handles) and `params_init: pd.DataFrame | None` (sbs only, mirrors
      `params_stderr`'s shape — every slice, not slice-0-representative,
      since `results_sbs[i].par_ini` is already available at no extra
      plumbing cost). `_write_slot`/`_read_slot` conditional write/read;
      `_slot_from_*` helpers and `_build_slot` thread the new kwargs
      through.
- [x] **`trspecfit.py`**: `_append_baseline_slot`/`_append_spectrum_slot`/
      `_append_2d_slot` evaluate a second `fitlib.residual_fun(...,
      par=fit_out.par_ini, res_type="fit")` alongside the existing
      final-params evaluation, cropped identically to `fit_arr`.
      `_append_sbs_slot` does the same per-slice inside the existing
      per-slice loop (using `self.results_sbs[s_i].par_ini`), plus
      `params_init = ulmfit.list_of_par_ini_to_df(self.results_sbs)`
      after the loop.
- [x] **`config/plot.py` / `Project._set_defaults`**: `PlotConfig` gained
      `show_init: bool = True` (docstring entry, placed near
      `full_range`); `Project._set_defaults` gained `self.show_init = True`.
- [x] **`fit_results.py`**: `FitResults.plot_fit`/`File.plot_fit` gained
      `show_init: bool | None = None`, resolved via `cfg.show_init` the
      same `None` = "use config" way `full_range` was done.
      `_plot_fit_1d` gained a `fit_ini` override param (mirrors `fit`/
      `components`/`roi`); renders the dotted-gold "initial guess" line
      (`color="#FFD700", linestyle=":"`, matching `plt_fit_res_1d`'s live
      style) when `show_init` resolves `True` and `fit_ini` is not
      `None`. `full_range` mode `NaN`-pads `fit_ini` via the existing
      `_pad_axis` helper, same honesty principle as `fit`/`components`.
      Rendering is 1D-only (baseline/spectrum); 2D persists `fit_ini` for
      completeness/symmetry but doesn't render it (no live precedent for
      a 2D init overlay); SbS's archive view still routes through the
      heatmap-style `plt_fit_res_2d` (no per-slice viewer yet — deferred,
      but `fit_ini`/`params_init` are now available for it).
- [x] **Tests**: `test_fit_archive_roundtrip.py` — `_assert_slot_round_tripped`
      extended with `fit_ini`/`params_init` checks across the F1/F6/F8
      family matrix; sbs cross-checks `params_init` against each slice's
      true seed straight from the live `results_sbs` (joint validation
      with the `fitlib.fit_wrapper` fix); new `_downgrade_archive_to_v5` +
      `test_reader_accepts_schema_v5_archive`. `test_lmfit_utils.py` — new
      `TestListOfParIniToDf` unit tests. `test_fit_history.py` — 4 new
      `_plot_fit_1d` direct-call tests (renders when present+shown; omitted
      when `show_init=False`; omitted when absent; `NaN`-padded in
      `full_range` mode). `test_fit_side_effects.py` — new
      `TestShowInitConfigResolution` class mirroring
      `TestFullRangeConfigResolution` exactly (project default `True`; a
      live baseline fit's display honors `config.show_init` in both
      directions; per-call override wins). Full suite (1041 + 171 slow),
      mypy, pyright, ruff all clean (whole-tree sweep; the 5 pre-existing
      mypy errors in `test_full_range_plot.py`/`test_fit_archive_roundtrip.py`
      were confirmed present on the base branch, unrelated to this work).
- [x] **Docs**: `docs/design/fit_archive_schema.md` — bumped documented
      `schema_version` to `"6"`, added the 5→6 version-history entry, new
      `fit_ini`/`params_init` dataset sections, updated the slot-group
      layout diagram, reader→object-model mapping table, and per-fit-type
      cheat sheet. `config/plot.py`'s `PlotConfig` docstring gained a
      `show_init` entry. Verified with a `sphinx -W` build.
- [x] **Manual verification**: a `stages=2` baseline fit with
      `show_output=1` shows stage-2's printed `init_value` matching the
      true seed; `save_fits` → reload with no live `Model` →
      `FitResults.plot_fit(..., full_range=True)` renders the "initial
      guess" line, `NaN`-masked outside the fit window.

**Status**: implementation complete, verified 2026-07-21.

**Out of scope (deliberately deferred)**: no SbS per-slice archive viewer
yet (`fit_ini`/`params_init` make it buildable later without another
schema bump); no 2D visual initial-guess overlay; no model rehydration
for an initial-guess curve extrapolated beyond the fit window.
