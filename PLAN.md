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
