# Active Plan

## Persist per-component 1D fit data in SavedFitSlot (schema 4)

Full design/rationale: `/home/yoyo/.claude/plans/hm-i-guess-the-eventual-simon.md`
(session-local; contents mirrored here for repo persistence).

**Goal**: close the live-vs-archive gap for 1D fit component visibility.
`FitResults.plot_fit` currently shows only observed/fit/residual for
baseline/spectrum/SbS slots; component decomposition is live-only
(`describe_model(detail=1)`, `File.plot_sbs_slices`). Persist components
directly in the slot instead.

**Decisions**: persist for baseline, spectrum, and every SbS slice
(symmetric); 2D untouched (no component concept there); store explicit
`component_names` labels (not parsed from `params_df` — breaks for
static `par_profile`-attached models); unconditional computation, no
perf opt-out; persistence-only in this pass, no new SbS per-slice
archive viewer yet.

- [x] **Schema** (`src/trspecfit/utils/fit_io.py`): added `components`/
      `component_names` fields to `SavedFitSlot`; bumped `SCHEMA_VERSION`
      3→4, extended `SUPPORTED_READ_VERSIONS`; `_write_slot`/`_read_slot`
      optional-field read/write; threaded through `_slot_from_baseline`/
      `_slot_from_spectrum`/`_slot_from_sbs`.
- [x] **Slot construction** (`src/trspecfit/trspecfit.py`):
      `_append_baseline_slot`/`_append_spectrum_slot` evaluate + crop
      components alongside the existing fit-curve eval;
      `_append_sbs_slot` does the same per-slice inside its existing
      parent-process finalization loop.
- [x] **Plotting** (`src/trspecfit/fit_results.py`): `_plot_fit_1d`
      renders components when present, falls back to lean sum-only
      when `None` (old-schema archives).
- [x] **Tests**: extended `_assert_slot_round_tripped` (schema round-trip
      across F1/F6/F8 families, including the F6 static-profile edge
      case) with components/component_names + a sum-reconstructs-fit
      invariant; added `test_reader_accepts_schema_v3_archive` (schema-3
      backward compat, `components=None`, `plot_fit` still renders); added
      `_downgrade_archive_to_v3`; extended `_downgrade_archive_to_v2` to
      also strip schema-4 fields; added
      `test_plot_fit_1d_renders_components_when_present` /
      `test_plot_fit_1d_falls_back_to_lean_when_components_none` in
      `test_fit_history.py`. Full suite (1005 tests), mypy, pyright, ruff
      all clean.

**Status**: implementation complete, verified 2026-07-20. Not yet
committed — awaiting user review/approval before commit.

**Next**: revisit the plot-range-vs-fit-limits inconsistency
(`describe_model`/fit-time displays show full data range,
`FitResults.plot_fit` shows only the fit window) — plan already drafted,
deferred at user's request until this work landed.
