# Active Plan — Fit results save/load

**v1 scope: a fit-results archive, not a model-rehydration archive.**

The immediate user value is: "I fitted this yesterday; now I want to save the
result, reload summaries, compare models, inspect residuals, and export
plots/tables." That does **not** require reconstructing the live `Model` graph
(profiles, dynamics, programmatic mutations). v1 stores final fit *outputs*
plus metrics; full model rehydration / warm restore is deferred until users
demand it.

## Decisions (locked)

- **One HDF5 per Project.** Default `./fit_results/<project_name>.fit.h5`. `overwrite` is **slot-scoped** (per file × model × fit_type × selection): re-running an existing slot errors unless `overwrite=True`. To start fresh, pass a new path. Mirrors `Simulator.save_data` ergonomics.
- **Object model first; HDF5 is the serialization.** The data model below is the source of truth; the on-disk schema mirrors it 1:1.
- **In-memory fit history is the canonical in-session store.** `Project._fit_history: list[SavedFitSlot]` is append-only; each `fit_baseline` / `fit_spectrum` / `fit_slice_by_slice` / `fit_2d` call materializes a slot at completion (via `_slot_from_<fit_type>` helpers) and appends. Solves the "File only keeps the last fit per fit_type" problem without disk side effects, path management, or read-only-fs failure modes. Memory cost is small (one extra `observed` + `fit` array per completed fit; for typical sessions, MB not GB). If memory ever bites in long-running sbs/2d-heavy sessions, add a config knob to drop slots later — not v1.
- **History is the log; archive is (by default) a snapshot.** `_fit_history` records every completed fit, including refits with the same canonical key. `save_fits` collapses to **latest-per-`history_key`** by default (snapshot semantics — one slot per `archive_slot_key` in the archive holds as an invariant). A `keep_history=True` flag for full-log save is a follow-on (deferred — needs schema work to disambiguate same-key slots via timestamp/sequence number). Rationale: most users want "save the current state," not "save every iteration." `compare_models` reads from `_fit_history` and *can* compare multiple takes on the same model in-session; the snapshot is for sharing/persistence.
- **Eager extraction, not lazy walking.** Slots are built once at fit completion (when the result is fresh and `File.model_base` etc. haven't been overwritten by a subsequent fit). `Project.results` is then a cheap wrapper around `_fit_history`, not a per-access walk over `File.model_*.result`. This dodges the race-with-self problem (fit modelB → modelA's result was on `model_base.result` and is now gone) entirely.
- **`FitResults` is a first-class results-browser class; `Project` is the fitting workspace.** Loaded archives do not live inside `Project` — they're a different concern (immutable inspection vs. mutable fitting). Architecture split:
  - `Project.save_fits(path)` — fitting workspace owns what to save; saves stay on Project.
  - `Project.load_fits(path) -> FitResults` — convenience entry point; **returns a fresh `FitResults`, does not mutate Project state**.
  - `Project.results` (property) — returns a `FitResults` view wrapping `Project._fit_history` (the in-memory log of completed fit slots, populated eagerly at fit completion). Project-level fits (`File._project_fit_result`) are not appended to history in v1 — see "Out of scope." Bridges in-memory fits into the same comparison API as loaded archives.
  - `Project.export_fits(format="csv")` — CSV + PNG dump of current in-memory fits, **one-way export**. CSV default; `format` kwarg reserved.
  - `FitResults.load(path) -> FitResults` — canonical entry point for inspecting an archive without a Project.
  - `FitResults.compare_models(...)`, `.find(...)`, `.get(...)`, `.files()`, `.models(...)`, `.plot_residuals(...)`, iteration — all browsing/comparison happens here.
  - `File.save_fit()` / `File.export_fit()` are 1-line delegates to Project (save/export are still fitting-workspace concerns).
  - **`File.compare_models()` is kept as a delegate** to `self.p.results.compare_models(file=self, ...)`. UX rationale: per-file comparison is the dominant pattern during model development, and File is the natural scope. The delegate is sugar — implementation lives entirely in `FitResults.compare_models`.
  - **`File.load_fit()` is dropped.** Loading is fundamentally an archive operation; the path argument dominates and `FitResults.load(path)` is the canonical entry. A `File.load_fit(path)` delegate would be no shorter than `project.load_fits(path, file=f)` and rarely what the user wants (usually they load the whole archive, then query).
  - `File.save_sbs_fit` / `File.save_2d_fit` become deprecated aliases (`DeprecationWarning`); removal scheduled before v1.0.0.
- **Self-contained archive.** Each file's group stores raw `data`, `energy`, `time` plus identity attrs. Required for archive portability (the original raw data file may not be present at load time) and as the canonical reference for what the file contained.
- **Per-slot `observed` array (not just `fit`).** Each slot stores both `fit` and `observed` on the **fit grid**, so residuals are unconditionally `observed - fit` with no recipe replay. Rationale: baseline fits operate on `data_base = np.mean(data[base_t_ind, :], axis=0)` ([trspecfit.py:1856](src/trspecfit/trspecfit.py#L1856)), spectrum fits on `data_spec` (a slice or mean over a time range, [trspecfit.py:2173](src/trspecfit/trspecfit.py#L2173)), and sbs/2d fits on data cropped by `e_lim` / `t_lim`. None of those equal raw `file.data`, so `file.data - slot.fit` has wrong shape or wrong grid. Storing `observed` per slot avoids encoding "how was this data view built" in the loader.
- **Stable, HDF5-safe group keys; identity lives in attrs.** All group-path components (files, slots) use **zero-padded positional keys** (`000000`, `000001`, ...). Human-meaningful identifiers (`File.name`, `original_path`, `model_name`, `fit_type`, `selection`) live as attrs on the group's `metadata`. Rationale: HDF5 path components forbid `/`, and user-facing names (especially `model_name` from YAML) can contain anything. Positional keys sidestep that entirely and match the `Simulator.save_data` precedent.
- **Canonical slot identity (mechanically defined; two-key form).** Each slot has explicit identity attrs:
  - `model_name` — user string, stored as attr only.
  - `fit_type` — `"baseline" | "spectrum" | "sbs" | "2d"`.
  - `selection_json` — JSON-serialized dict capturing the fit-view identity (see below).
  - **In memory**: `history_key = sha256(file_fingerprint | model_name | fit_type | selection_json)` — used by `_collapse_history_to_snapshot`, in-session dedup, comparison.
  - **In archive**: `archive_slot_key = sha256(file_ref | model_name | fit_type | selection_json)` — written to slot `metadata.attrs`, used for slot-scoped overwrite detection on save. Computed at save time only, after fingerprint → `file_ref` mapping.

  Both keys serve the same logical purpose ("uniquely identify this slot"); they just use different file-identity tokens because in-memory and on-disk identity primitives differ (fingerprint vs archive-local positional path). Two functions in `utils/fit_io.py`: `compute_history_key` and `compute_archive_slot_key`.

- **`selection_json` includes the full fit-view identity** so refits with different windows/limits don't collide on the canonical key. **All `*_lim` fields are index slices `[start, stop)` (matching `File.e_lim` / `File.t_lim` semantics — [trspecfit.py:1155](src/trspecfit/trspecfit.py#L1155)), not absolute physical values.** The `_abs` parallels (`e_lim_abs`, `t_lim_abs`) are user-meaningful absolutes but are *not* used in the canonical key — the index form determines the actual fit grid, and `observed_sha256` catches any drift the indices happen to miss.
  - **baseline**: `{"base_t_ind": [start, stop), "e_lim": [start, stop) | None}` — `base_t_ind` is the time-window index slice averaged for `data_base`.
  - **spectrum**: `{"time_point": float | None, "time_range": [lo, hi] | None, "time_type": "abs" | "ind", "e_lim": [start, stop) | None}`.
  - **sbs**: `{"e_lim": [start, stop) | None, "t_lim": [start, stop) | None}`.
  - **2d**: `{"e_lim": [start, stop) | None, "t_lim": [start, stop) | None}`.

  Empty `{}` is no longer the default for any fit type; every slot carries the relevant view identity.

- **`observed_sha256` as a belt-and-suspenders cross-check.** Each slot stores `sha256(observed.tobytes())` as an attr. `compare_models` refuses to compare slots whose `observed_sha256` doesn't match (when comparing same fit_type on same file) — guards against silent drift if `selection_json` ever fails to capture a relevant view detail.
- **Two-tier identity (within-archive vs across-archive).** Use the right tool for the right job:
  - **Within an archive** (e.g. a slot pointing at its file): use the archive-local positional path `files/000000` — unambiguous and stable for the lifetime of that archive file.
  - **Across archive ↔ live Project** (matching a loaded `Project.files[*]` to an archive file on `load_fits`, or aligning two archives): use **content fingerprints**, not positional index. File matching uses `(data_sha256, energy_sha256, time_sha256, shape)` with `name` / `original_path` as tie-break metadata. Multiple shas + shape avoid the "identical replicate files share `data_sha256`" ambiguity that bare-data-hash matching would have.
- **Fit types covered (4):** `baseline`, `spectrum`, `sbs`, `2d`. **Project-level / global fits are deferred from v1** — the pipeline itself is flagged as architecturally unfinished (TODO line 12), so locking in archive schema for it now is premature. Adding project-level slots later is a strict additive change. Add `spectrum` to `File.get_fit_results(fit_type=...)` while we're here (currently missing — [trspecfit.py:3022](src/trspecfit/trspecfit.py#L3022)).
- **Fit-quality metrics computed and stored:** `chi2`, `chi2_red`, `r2`, `aic`, `bic`. Per-slice for SbS; single value for baseline / spectrum / 2d. Small helper in `fitlib`: `(observed, fit, n_free_pars) → dict` — takes the observed data view actually fit against, not raw `file.data`.

## Object model

The unit of persistence is the **fit slot**: one completed fit result for a given (project, file, model, fit_type, selection).

```
SavedProject
├── name, timestamp, trspecfit_version
└── files: list[SavedFile]
    ├── identity:   name, original_path, dim, shape, data_sha256
    ├── arrays:     data, energy, time            # time empty for 1D files
    ├── plot ctx:   e_lim, t_lim
    └── slots: list[SavedFitSlot]
        ├── identity:    model_name, fit_type, selection
        │                  • baseline:  {base_t_ind, e_lim}              # all are index slices [start, stop)
        │                  • spectrum:  {time_point, time_range, time_type, e_lim}
        │                  • sbs:       {e_lim, t_lim}                   (one slot covers all slices)
        │                  • 2d:        {e_lim, t_lim}
        │                  + observed_sha256                              (defensive cross-check)
        ├── provenance:  fit_alg, yaml_filename (human breadcrumb only), timestamp
        ├── params:      DataFrame[name, value, init_value, stderr, min, max, vary, expr]
        ├── metrics:     {chi2, chi2_red, r2, aic, bic}     # per-slice arrays for sbs
        ├── observed:    ndarray  (the data view that was actually fit — data_base / data_spec / cropped data)
        ├── fit:         ndarray  (model evaluated at final params; same shape as `observed`)
        ├── conf_ci:     DataFrame | None
        └── mcmc:        {flatchain, ci, lnsigma} | None

Invariant: `observed.shape == fit.shape`. Residuals = `observed - fit`, always, for any fit_type.
```

Explicit non-goals for the slot:

- No serialized Model graph, no restorable model snapshot. `yaml_filename` is recorded for human reference only; we do not promise to deserialize a Model from it in v1.
- No "warm-start" payload. The archive cannot be used to continue/resume a fit.
- No live link between a SavedFitSlot and the live fit state on `File` after extraction. Slots in `_fit_history` capture *snapshots* of `model_base.result` / `model_spec.result` / `results_sbs` + `model_sbs` / `model_2d.result` taken at fit completion; subsequent overwrites of those File attrs do not affect already-captured slots. Loaded results from disk are similarly independent — they never merge into `_fit_history`.

## Public API shape

```python
# === Project (fitting workspace) ===

Project.save_fits(
    filepath: PathLike | None = None,
    *,
    file: int | str | File | list | None = None,
    model: str | list[str] | None = None,
    fit_type: Literal["baseline","spectrum","sbs","2d"] | list | None = None,
    overwrite: bool = False,                 # slot-scoped
    show_output: int = 1,
) -> None
# Filters Project._fit_history by (file, model, fit_type), then collapses to
# latest-per-history_key (snapshot semantics — one slot per `archive_slot_key`
# in the archive). Filter args operate on slot identity (not on live File.model_*),
# so they work cleanly even after refits overwrote the live attr.
# `keep_history=True` for full-log save: deferred to v2 (needs schema work).

Project.load_fits(
    filepath: PathLike,
    *,
    file: int | str | list | None = None,
    model: str | list[str] | None = None,
    fit_type: ... | None = None,
    show_output: int = 1,
) -> FitResults
# Returns a fresh FitResults. Does NOT mutate Project state.
# Equivalent to FitResults.load(path, ...); provided as a convenience entry point
# for users who already have a Project in hand.

Project.results -> FitResults       # property
# Returns a FitResults wrapper around Project._fit_history — the append-only log of
# slots materialized at fit completion. Cheap (no copy of underlying slot arrays).
# Same API as a loaded FitResults; lets users compare in-session fits without saving.
# Project-level fits are not appended to history in v1.

Project.export_fits(
    filepath: PathLike | None = None,
    *,
    format: Literal["csv"] = "csv",
    file=..., model=..., fit_type=...,
    overwrite: bool = False,
    show_output: int = 1,
) -> None

# === FitResults (inspection / comparison artifact) ===

FitResults.load(
    filepath: PathLike,
    *,
    file=..., model=..., fit_type=...,       # optional load-time filters
) -> FitResults
# Standalone: works without a Project.

FitResults.compare_models(
    file: str | SavedFile | None = None,
    *,
    models: list[str] | None = None,
    fit_type: ... | None = None,
    metrics: list[str] | None = None,        # default: ["chi2_red", "r2", "aic", "bic"]
    sbs_aggregation: Literal["median", "mean", "sum", "long"] = "median",
    plot_residuals: bool = False,
) -> pd.DataFrame
# For SbS slots, per-slice metrics are aggregated to a scalar via sbs_aggregation
# before being placed in the comparison DataFrame.
#   "median"  — robust; default. One row per slot, value = np.median(per_slice).
#   "mean"    — average across slices.
#   "sum"     — sum across slices (statistically meaningful for chi2/aic/bic if
#               slices are independent; less natural for chi2_red/r2).
#   "long"    — return per-slice rows; comparison DataFrame gets a slice_index column.
# Refuses to compare slots whose observed_sha256 differs when both are same
# (file, fit_type) — silent grid drift would invalidate the comparison.

FitResults.find(*, file=..., model=..., fit_type=..., selection=...) -> list[SavedFitSlot]
FitResults.get(*, file, model, fit_type, selection=None) -> SavedFitSlot   # raises if 0 or >1
FitResults.files() -> list[SavedFile]
FitResults.models(file=None) -> list[str]
FitResults.plot_residuals(*, file, models, ...) -> None
for slot in fit_results: ...

# === File (per-file convenience delegates) ===

File.save_fit(**kw)              # → self.p.save_fits(file=self, **kw)
File.export_fit(**kw)            # → self.p.export_fits(file=self, **kw)
File.compare_models(*models, **kw)
# → self.p.results.compare_models(file=self, models=list(models) or None, **kw)
# Sugar; implementation lives in FitResults.compare_models.

# DROPPED: File.load_fit — load is path-scoped, not File-scoped.
# Use FitResults.load(path) or project.load_fits(path).

# === Deprecated aliases (DeprecationWarning; remove before v1.0.0) ===

File.save_sbs_fit(save_path)   # → File.export_fit(fit_type="sbs", filepath=save_path)
File.save_2d_fit(save_path)    # → File.export_fit(fit_type="2d",  filepath=save_path)
```

## FitResults class

`FitResults` is the **only comparison engine** — there is no parallel comparison API on `Project` or `File`. Two construction paths:

1. **Loaded from disk** (`FitResults.load(path)`, or equivalently `Project.load_fits(path)`) — a snapshot of an archive on disk. Independent of any live Project.
2. **In-memory view** (`Project.results` property) — a `FitResults` wrapper around `Project._fit_history`, the append-only log of slots materialized at fit completion. **Not a loaded archive** — no file involved, no persistence, no fingerprint validation against an archive. Just the in-session fit log, exposed through the same query/comparison surface.

The distinction matters because users will naturally compare during model development ("I just ran two models, which is better?") *before* they think about saving. Forcing save-before-compare would invert the natural workflow. The history mechanism makes comparison immediately available without a save round-trip — and crucially, it preserves *all* completed fits, including refits with the same canonical key, since `File.model_base` etc. only hold the latest by themselves.

### Convergent pipeline

The fit-completion path produces `SavedFitSlot` objects exactly once; everything downstream reads slots, never live `Model.result`. There is one metrics implementation, one residual implementation, one comparison engine:

```
fit_baseline / fit_spectrum /                ┌──────────────────────────┐
fit_slice_by_slice / fit_2d  ────► result ───►  _slot_from_<fit_type>   │
                                              │   (eager extraction)    │
                                              └────────────┬────────────┘
                                                           │
                                                           ▼
                                            Project._fit_history (append-only log)
                                                           │
                                       ┌───────────────────┼──────────────────────┐
                                       ▼                   ▼                      ▼
                       Project.results (wrapper)  Project.save_fits        Project.export_fits
                                                  (filter + snapshot       (filter + CSV/PNG)
                                                   collapse → HDF5)

HDF5 archive ────► reader ────► FitResults (FitResults.load / Project.load_fits)
                                Independent of _fit_history; never merged in.
```

The `_slot_from_<fit_type>` extractors live in `utils/fit_io.py` and are called *once* at fit completion. Everything downstream (history wrapper, save, export) operates on already-built slots. CSV export reads slot fields, the writer serializes them, the FitResults wrapper exposes them — the slot is the single center of gravity for everything downstream of a completed fit.

**Internal identity** (consistent with the rest of the schema): each slot is keyed by `(file_fingerprint, model_name, fit_type, selection_json)`, where `file_fingerprint` is the multi-sha tuple from the "two-tier identity" decision (`data_sha256` + `energy_sha256` + `time_sha256` + `shape`). `file_name` is **display metadata only** — used for printing and as a query input that resolves to fingerprint at lookup time. This:

- Survives file renames between save/load.
- Avoids same-name-different-content collisions when users hold multiple `FitResults` instances side-by-side.
- Keeps identity aligned with `history_key` / `archive_slot_key` and `file_ref` decisions elsewhere — names live in attrs, never in keys.

**Holds, but does not own** (snapshot semantics):

- A `FitResults` is **immutable after construction**. Its slot list is frozen at the moment of construction.
- **`Project.results` returns a fresh snapshot per access**: `FitResults(slots=list(self._fit_history))` copies the current history list at call time. Slot *objects* inside are shared (the underlying `observed` / `fit` / `params` arrays are not duplicated), but the *list* is a snapshot — subsequent fits append to `_fit_history` and do **not** affect previously-returned `FitResults`. Users see updated history by calling `p.results` again. Object identity is unstable: `p.results is p.results` is False; the contents at a given access are fixed.
- `Project.load_fits()` returns a fresh `FitResults` and **never** appends loaded slots to `_fit_history`. `_fit_history` is reserved for fits that happened in this session; loaded archives are held by user-named variables: `loaded = project.load_fits(...)` or `loaded = FitResults.load(...)`. This keeps the "current session log" semantics clean.

**Module placement**: `trspecfit/fit_results.py` (new module). Exported from `trspecfit/__init__.py` as `FitResults` for the standalone `FitResults.load(...)` entry point.

## Current state (observed)

- `File.save_sbs_fit` ([trspecfit.py:2559](src/trspecfit/trspecfit.py#L2559)) — wide CSV + PNGs via `fitlib.results_to_df` / `results_to_fit_2d` / `plt_fit_res_2d`. Logic moves to `Project.export_fits` (CSV path); method becomes deprecated alias.
- `File.save_2d_fit` ([trspecfit.py:2979](src/trspecfit/trspecfit.py#L2979)) — only plots data/fit/residual maps. Parameter CSVs *are* written, but earlier in `fit_2d()` itself via `fit_wrapper(..., save_output=1)` at [trspecfit.py:2951](src/trspecfit/trspecfit.py#L2951), not by `save_2d_fit`. So the persistence path is split across two methods today. Same fate as `save_sbs_fit`: the CSV-writing logic moves into `Project.export_fits` (CSV path), `save_2d_fit` becomes a deprecated alias.
- Baseline fit save path in `fitlib` writes per-table CSVs ([fitlib.py:743+](src/trspecfit/fitlib.py#L743)) — logic moves to `export_fits` CSV path.
- `File.load_fit` ([trspecfit.py:2265](src/trspecfit/trspecfit.py#L2265)) is a stub. **Removed in v1** — replaced by `FitResults.load(path)` and `Project.load_fits(path)`.
- `File.compare_models` ([trspecfit.py:3077](src/trspecfit/trspecfit.py#L3077)) is a stub. Becomes a thin delegate to `self.p.results.compare_models(file=self, ...)`.
- `File.fit_spectrum` ([trspecfit.py:2085](src/trspecfit/trspecfit.py#L2085)) — 1D fit at a `time_point` / `time_range`. Slot identity must include those.
- `File.get_fit_results(fit_type=...)` ([trspecfit.py:3019](src/trspecfit/trspecfit.py#L3019)) returns DataFrames for `baseline` / `sbs` / `2d`; **`spectrum` missing** — fix as part of this work.
- `File` always has a parent Project ([trspecfit.py:1113](src/trspecfit/trspecfit.py#L1113)) — `self.p` is never None, so File-level delegates rely on it unconditionally.
- `Simulator.save_data` ([simulator.py:1386](src/trspecfit/simulator.py#L1386)) is the structural template we follow.

## HDF5 schema (sketch — mirrors object model 1:1)

```
<project_name>.fit.h5
├── metadata/                                 # attrs: trspecfit_version, timestamp, project_name
├── files/
│   ├── 000000/                               # zero-padded
│   │   ├── metadata                          # attrs: name, original_path, dim, shape,
│   │   │                                     #        data_sha256, energy_sha256, time_sha256,
│   │   │                                     #        e_lim, t_lim
│   │   ├── energy                            # dataset
│   │   ├── time                              # dataset (empty if 1D)
│   │   ├── data                              # dataset
│   │   └── slots/
│   │       ├── 000000/                       # zero-padded positional; identity in attrs
│   │       │   ├── metadata                  # canonical key attrs:
│   │       │   │                             #   file_ref ("files/000000"),
│   │       │   │                             #   model_name, fit_type, selection_json,
│   │       │   │                             #   archive_slot_key, observed_sha256
│   │       │   │                             # provenance attrs:
│   │       │   │                             #   fit_alg, yaml_filename, timestamp
│   │       │   │                             # metrics attrs:
│   │       │   │                             #   chi2, chi2_red, r2, aic, bic
│   │       │   ├── params                    # dataset: structured (name, value, init_value, stderr, min, max, vary, expr)
│   │       │   ├── observed                  # dataset: data view that was fit (data_base / data_spec / cropped); same shape as `fit`
│   │       │   ├── fit                       # dataset: model evaluated at final params (1D or 2D)
│   │       │   ├── metrics_per_slice         # dataset: 2D (slices × {chi2, chi2_red, r2, ...}) — sbs only
│   │       │   ├── conf_ci                   # dataset (optional)
│   │       │   └── mcmc/                     # group (optional): flatchain, ci, lnsigma
│   │       └── 000001/...
│   └── 000001/...
# project-level / global fits: NOT in v1. See "Out of scope" below.
```

Notes:

- **No raw user names in path components.** All group keys are positional; `model_name` / `fit_type` / `selection` live in attrs.
- **`fit_type` is an attr, not a path segment.** The string `"2d"` only appears in `metadata.attrs["fit_type"]`, never as a group name.
- **Within-archive cross-reference uses `file_ref`** (e.g. `"files/000000"`). Resolves the earlier "positional vs sha lookup" open question: archive-internal links use archive-local paths, which are stable for the lifetime of the archive. Cross-archive / archive ↔ live Project matching uses the multi-sha fingerprint. (Currently used only by within-file slot→file references; the use case will expand if/when project-level fits land in v2.)

## Tasks

### Precursors

- [x] Confirm scope + answers to open questions.
- [x] Add `spectrum` to `File.get_fit_results(fit_type=...)`.
- [x] Add `fitlib.compute_fit_metrics(observed, fit, n_free_pars) -> dict` returning `{chi2, chi2_red, r2, aic, bic}`. Takes **`observed`** (the actual data view fit against), not raw `file.data`. Use lmfit's `MinimizerResult.chisqr` / `.redchi` / `.aic` / `.bic` where available; compute R² locally.

**Note on the observed/fit/metrics capture:** the original precursor wording
("wire metric computation … so the values exist on `Model.result`") is
**intentionally dropped**. `Model` should not carry archive/history concerns —
that creates two sources of truth. Instead, `SavedFitSlot` is the first owner
of `observed`, `fit`, `metrics`, `observed_sha256`, `selection_json`, and
`history_key`. The fit-path → snapshot args → `_slot_from_<fit_type>` →
`_fit_history` pipeline captures and computes everything in one shot at fit
completion. See "Object model + I/O" below.

### Object model + I/O

- [ ] Define `SavedProject` / `SavedFile` / `SavedFitSlot` dataclasses (probably in `utils/fit_io.py`).
- [ ] Define `FitResults` class in new module `trspecfit/fit_results.py`, exported as `trspecfit.FitResults`. Includes `load` classmethod, `find` / `get` / `files` / `models` / `__iter__` query API, and `compare_models` / `plot_residuals`. Internal key is `(file_fingerprint, model_name, fit_type, selection_json)`; name-based queries resolve to fingerprint internally. Constructor accepts a list of `SavedFitSlot` (used by both `load` and the `Project.results` wrapper path).
- [ ] Add `Project._fit_history: list[SavedFitSlot]` attr (initialized to `[]` in `Project.__init__`).
- [ ] Implement per-fit-type extraction helpers in `utils/fit_io.py`. **Each helper takes already-copied snapshot args** (not live `File.model_*` references) so call-site ordering is irrelevant — the helper cannot be broken by post-fit cleanup like the seed-template restoration at [trspecfit.py:2551](src/trspecfit/trspecfit.py#L2551). Signatures:
  - `_slot_from_baseline(*, file_fingerprint, fit_alg, yaml_filename, params_df, observed, fit, base_t_ind, e_lim, n_free_pars) -> SavedFitSlot`
  - `_slot_from_spectrum(*, file_fingerprint, fit_alg, yaml_filename, params_df, observed, fit, time_point, time_range, time_type, e_lim, n_free_pars) -> SavedFitSlot`
  - `_slot_from_sbs(*, file_fingerprint, fit_alg, yaml_filename, parameter_names, params_per_slice, observed, fit, e_lim, t_lim, n_free_pars) -> SavedFitSlot` — caller passes already-extracted per-slice params (from a copy of `results_sbs`) plus the parameter_names captured from `model_sbs` *before any restoration*.
  - `_slot_from_2d(*, file_fingerprint, fit_alg, yaml_filename, params_df, observed, fit, e_lim, t_lim, n_free_pars) -> SavedFitSlot`

  Each helper computes `metrics` (via `compute_fit_metrics`), `observed_sha256`, `selection_json`, and `history_key`. Project-level (`_project_fit_result`) is explicitly skipped in v1.
- [ ] Wire eager extraction into the four fit code paths. Call site is responsible for capturing snapshot args **at the moment results are valid**:
  - `fit_baseline`: extract immediately after fit completes, before any further mutation.
  - `fit_spectrum`: same; capture `time_point` / `time_range` / `time_type` from fit args.
  - `fit_slice_by_slice`: extract **before** [trspecfit.py:2551](src/trspecfit/trspecfit.py#L2551) (the seed-template restoration that would otherwise blow away `model_sbs.parameter_names`/result state). Snapshot the relevant fields into local copies, then call the helper.
  - `fit_2d`: extract immediately after fit completes.

  All four append the resulting slot to `self.p._fit_history`.
- [ ] Implement `Project.results` property: returns `FitResults(slots=list(self._fit_history))`. Cheap: no array copies, just a list snapshot.
- [ ] Finalize HDF5 schema (structured-array dtypes, attr keys, MCMC layout) and document in `docs/design/`. **All group-path components are positional zero-padded keys; user-facing names live only in attrs.**
- [ ] Add identity-key helpers in `utils/fit_io.py`:
  - `compute_history_key(file_fingerprint, model_name, fit_type, selection_json) -> str` — sha256, used in-memory.
  - `compute_archive_slot_key(file_ref, model_name, fit_type, selection_json) -> str` — sha256, used at save time once `file_ref` is known.
  - `compute_file_fingerprint(data, energy, time) -> dict[str, str]` — multi-sha (`data_sha256`, `energy_sha256`, `time_sha256`, `shape`).
  - `compute_observed_sha256(observed) -> str` — for the slot's defensive cross-check.
  - `build_selection_json(fit_type, **fields) -> str` — deterministic JSON serialization (sorted keys) so equivalent selections produce identical hashes.
- [ ] Add `_find_slot_by_archive_key(file_group, archive_slot_key) -> Group | None` and `_find_file_by_fingerprint(archive, fingerprint) -> Group | None` helpers — used by overwrite detection (save) and project-matching (load).
- [ ] Add `_collapse_history_to_snapshot(slots: list[SavedFitSlot]) -> list[SavedFitSlot]` helper: keeps the latest slot per `history_key` (snapshot semantics for default `save_fits`).
- [ ] Implement writer in `utils/fit_io.py`: takes a list of slots (already filtered + collapsed), serializes to HDF5. The writer is *slot-driven*; it does not walk `Project` or live `File.model_*` — that walking is done at fit-completion time by the extraction helpers, with the result accumulating in `_fit_history`.
- [ ] Implement reader in `utils/fit_io.py`: deserializes HDF5 into a list of `SavedFitSlot` (plus `SavedFile` records for raw arrays). **Does not touch live `File.models` or `_fit_history`.**

### Project-level API

- [ ] `Project.save_fits()` — filter `_fit_history` by `(file, model, fit_type)`, collapse to snapshot via `_collapse_history_to_snapshot` (using `history_key`), then for each slot: resolve `file_fingerprint → file_ref` (look up or create the file group in the archive), compute `archive_slot_key`, check for existing slot, write or error per `overwrite=True/False`.
- [ ] `Project.load_fits()` — thin wrapper that returns `FitResults.load(path, ...)`. Does not mutate Project state.
- [ ] `Project.export_fits(format="csv")` — same filter pipeline as `save_fits`, but emits CSV+PNGs instead of HDF5. Absorbs CSV+PNG logic from current `File.save_sbs_fit` / `save_2d_fit` + baseline-CSV path in `fitlib`.

### File-level delegates + deprecation

- [ ] `File.save_fit()` / `export_fit()` / `compare_models()` as 1-line delegates. `save_fit` / `export_fit` route to `self.p.save_fits` / `self.p.export_fits`; `compare_models` routes to `self.p.results.compare_models(file=self, ...)`. **Do not add `File.load_fit`** — load is path-scoped (use `FitResults.load(path)` or `Project.load_fits(path)`).
- [ ] Convert `File.save_sbs_fit` / `save_2d_fit` to deprecated aliases (`DeprecationWarning`); add removal-before-v1.0.0 marker in code.
- [ ] Track v1.0.0 removal in TODO.md under "Build & release → Remove legacy/backwards-compat code."

### Tests + docs

- [ ] Round-trip tests: save → load → compare metrics / param tables / fit / observed arrays match. Cover basic / profile / profile+dynamics models, all four fit types where applicable. Verify `observed - fit` reproduces residuals for each fit_type without reading `file.data`.
- [ ] `_fit_history` tests: fit modelA-baseline, fit modelB-baseline, verify history has both slots and `Project.results` exposes both. Refit modelA-baseline, verify history has *all three* slots. Save with default snapshot semantics, verify archive has only two slots (one per `history_key`, latest wins).
- [ ] **Selection-identity tests**: refit baseline with different `base_t_ind`, refit sbs/2d with different `e_lim`/`t_lim`, refit spectrum with different `time_point` — verify `history_key` differs in each case, snapshot collapse keeps both, archive stores both as distinct slots.
- [ ] **Snapshot semantics tests**: capture `r1 = p.results`, run another fit, verify `r1` does not see the new slot (frozen list), `r2 = p.results` does.
- [ ] **SbS extraction-timing test**: simulate the seed-template restoration at [trspecfit.py:2551](src/trspecfit/trspecfit.py#L2551); verify the extracted slot still has correct `params_per_slice` / `parameter_names` / metrics (helper used copied snapshot args, not live state).
- [ ] **`observed_sha256` cross-check test**: construct two slots with same canonical key but mutated observed array; `compare_models` raises (or warns clearly) on grid mismatch.
- [ ] `compare_models` tests: two models on same file, returns expected metrics ordering; residual plot smoke test. Multi-version compare on same canonical key (multiple takes on modelA-baseline) — verify default behavior picks latest, `find` exposes all. SbS aggregation: test all four `sbs_aggregation` modes on a multi-slice fit.
- [ ] `export_fits` parity tests: same column shapes as old `save_sbs_fit` / `save_2d_fit` outputs.
- [ ] DeprecationWarning tests for the old aliases.
- [ ] Update example notebooks to demo `Project.save_fits` / `Project.load_fits` / `Project.export_fits` / `compare_models`.
- [ ] Update `docs/design/repo_architecture.md` with the new `utils/fit_io.py` module and the save/export split.

## Out of scope (deferred to v2 if users demand it)

- **Auto-save (implicit persistence on every fit).** v1 keeps fit history in memory only (`Project._fit_history`); persistence is explicit (`Project.save_fits()`). Auto-save is orthogonal to the in-memory history mechanism — it can be added later as an opt-in `Project(auto_save_path=...)` init kwarg, where each `_fit_history.append` also serializes incrementally to the archive. Deferred until we see how much friction explicit save actually causes in real workflows; standard scientific-Python idiom is explicit persistence (pandas, lmfit, NumPy all require explicit `save`/`to_csv`/`pickle`).
- **`auto_export` opt-out toggle for fit-completion side effects.** Every `fit_baseline` / `fit_spectrum` / `fit_slice_by_slice` / `fit_2d` unconditionally triggers `fit_wrapper(save_output=1)` plus an auto-`save_*_fit(...)` call, writing ~6 CSVs + several PNGs per fit. Quick benchmark on the 30×24 test dataset (May 2026): baseline/spectrum saves are imperceptible (<1 ms each), but SbS adds ~0.9 s of per-parameter PNG generation and 2D's three-panel matplotlib plot is ~0.5 s — saves *exceed* the GIR fit time itself for 2D on small data. Add `Project.auto_export: bool = True` gating all four `fit_wrapper(save_output=...)` calls and all four `save_*_fit` call-sites in `fit_*` methods. Default True preserves current behavior; explicit `File.save_*_fit(path)` / `Project.export_fits(...)` calls always run. Real-time fitting (the long-term dream for this repo), parameter sweeps, and ML training-data generation are the driving use cases — defer implementation until that work begins.
- **MCMC decoupled from `fit_wrapper`.** Today `fit_wrapper` bundles optimization, confidence intervals, MCMC, and export; `mc_settings.use_emcee=2` will silently kick off a potentially very expensive MCMC run when CI fails. Cleaner shape: `fit_*()` runs optimization only and appends a normal `SavedFitSlot`; users inspect via `Project.results`, then explicitly call something like `project.run_mcmc(slot=...)` / `file.run_mcmc(fit_type=..., model=...)` on the fits worth interrogating. CI can fail and say so without secretly upgrading the call. **Schema wrinkle to resolve when this lands**: `SavedFitSlot.mcmc` exists in the v1 schema as an optional sub-record; with append-only history a separate run shouldn't mutate an existing slot in place. Either produce an enriched-copy slot keyed by the same `history_key`, or introduce a sibling `SavedMCMCResult` keyed by `history_key` — both keep in-session history append-only. Drops `mc_settings` / `use_emcee=2` fallback from `fit_wrapper` at the same time. Deferred — not blocking v1 of the save/load work, but worth doing before MCMC sees real use, since changing the call surface later breaks more callers than now.
- **`keep_history=True` for full-log save.** `save_fits` currently collapses to latest-per-canonical-key (snapshot). Saving the full append-only log of refits would let archives preserve every iteration but requires schema work to disambiguate same-key slots (timestamp or sequence number in the canonical key). Deferred — most users want snapshot semantics; revisit if the in-session multi-version-compare workflow grows into a "preserve every refit" need.
- **Memory cap / history pruning.** v1's `_fit_history` is unbounded. For typical sessions (10s of fits, MB-size data) this is fine. If long sbs/2d-heavy sessions show memory growth, add a config knob (`Project(history_max_slots=N)` or similar) or a `Project.clear_history()` method. Defer until measured.
- **Project-level / global fit save.** `File._project_fit_result` is *not* persisted in v1, and project-level fits are not appended to `_fit_history`. Reasoning: the project-level fit pipeline itself is flagged as architecturally unfinished (TODO line 12 — open question on lowering multi-file residual to GIR), so locking in archive schema for it now would be premature. v1 covers `baseline` / `spectrum` / `sbs` / `2d` (the file-level fits, which is the 95% case). Adding project-level slots later is a strict additive change to the schema and the SavedProject hierarchy.
- **Model rehydration / warm restore.** Reconstructing live `Model` objects from the archive (with profiles, dynamics, programmatic mutations intact) so users can resume fitting or call `model.create_value_2d()` on a loaded fit. v1 stores the *output* fit array instead, which covers inspection and comparison without the fragility.
- **YAML round-trip from archive.** v1 stores `yaml_filename` as a breadcrumb only; not promised to deserialize back into a Model.
- **Non-CSV export formats** (parquet, mat, json) — `format` kwarg reserved.
- **Resumable partial writes** (interrupting `save_fits` mid-write).
- **A `save_outputs` / `save_type` setting in `project.yaml`** — saves stay method-driven for now.
