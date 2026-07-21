# Active Plan

## Persist `aux_axis` at the per-file archive level (schema 4 → 5)

Full design/rationale: `/home/yoyo/.claude/plans/hm-i-guess-the-eventual-simon.md`
(session-local; contents mirrored here for repo persistence).

**Goal**: while investigating the live-vs-archive display-range question,
found that the archive already persists the full, uncropped `data`/
`energy`/`time` once per file (`SavedFile`), but not `aux_axis`
(`File.aux_axis` — the auxiliary physical axis used by `par_profile`
models). Reloading an archive with no live `File` loses it entirely.
Added it, following the exact `data`/`energy`/`time` pattern.

- [x] **Schema** (`src/trspecfit/utils/fit_io.py`): added
      `aux_axis: np.ndarray | None = None` to `SavedFile`; bumped
      `SCHEMA_VERSION` 4→5, extended `SUPPORTED_READ_VERSIONS`;
      `_write_file_payload`/`_read_file` optional-field write/read
      (conditional write, `.get()` + `None` fallback — not the
      unconditional pattern used for `data`/`energy`/`time`).
- [x] **Writer call site** (`src/trspecfit/trspecfit.py`): threaded
      `aux_axis=live.aux_axis` into the `fit_io.SavedFile(...)`
      construction used by `save_fits`.
- [x] **Tests**: extended `test_baseline_roundtrip` (F1/F6/F8) with an
      `aux_axis` round-trip assertion (non-`None` + array-equal for
      F6/F8, `None` for F1) via the loaded `FitResults._files_by_fp`
      provider; added `_downgrade_archive_to_v4` +
      `test_reader_accepts_schema_v4_archive` (pre-5 archives load with
      `aux_axis=None`). Full suite (1006 tests), mypy, pyright, ruff all
      clean.
- [x] **Docs**: `docs/design/fit_archive_schema.md` — bumped documented
      `schema_version` to `"5"`, added the 4→5 version-history entry, the
      `aux_axis` line in the file-group layout diagram, and a notes-
      section callout on the omit-if-`None` write rule and the
      fingerprint exclusion. Verified with a `sphinx -W` build.

**Status**: implementation complete, verified 2026-07-20. Not yet
committed — awaiting user review/approval before commit.

**Out of scope (deliberately deferred)**: exposing `aux_axis` through
`FitResults._axes_for` or any plotting method (no current consumer needs
it yet); adding `aux_axis` to `file_fingerprint` identity hashing.

**Next**: return to the live-vs-archive 1D post-fit display-range
inconsistency. Investigation this session found `fit_2d`/
`fit_slice_by_slice` already route their post-fit live display through
`self.plot_fit(...)` (so 2D/SbS are already consistent live vs. archive);
only `fit_baseline`/`fit_spectrum` still call `fitlib.plt_fit_res_1d`
directly with the full, uncropped data. Chosen fix (confirmed with user):
route those two through `self.plot_fit(...)` too, matching `fit_2d`/
`fit_slice_by_slice` — this drops the `show_init` initial-guess overlay
from post-fit display (confirmed acceptable: init guess isn't a
completed-fit artifact) and deletes a chunk of now-dead code
(`initial_guess` extraction, bespoke title construction) in both methods.
Also needs a small title-informativeness enhancement in
`FitResults._plot_fit_1d` (file name, yaml stem, spectrum's time
selection) since it becomes the sole 1D post-fit display path. Needs a
fresh plan file before implementing (the plan file was overwritten for
this aux_axis detour).
