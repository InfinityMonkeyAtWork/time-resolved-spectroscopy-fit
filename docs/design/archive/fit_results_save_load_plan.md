---
orphan: true
---

# Archived Plan: Fit Results Save/Load

> Archived on 2026-05-27 after the fit-results save/load feature shipped.
> Keep [../fit_archive_schema.md](../fit_archive_schema.md) as the long-lived
> wire-format reference; this file is the historical design rationale (why
> per-slot `observed`, why two identity keys, why HDF5 instead of pickle, the
> in-memory history layer, and the deferred project-scoped joint-fit slot).
> Links into the source tree below point at line numbers as they were at archival.

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
  - `Project.results` (property) — returns a `FitResults` view wrapping `Project._fit_history` (the in-memory log of completed fit slots, populated eagerly at fit completion). `Project.fit_2d()` emits ordinary per-file `fit_type="2d"` slots into `_fit_history`, so its results are visible via `Project.results` like any other fit; only a *project-scoped joint-result* slot (one record owning the shared parameters without per-file duplication) is deferred to v2 — see "Out of scope." Bridges in-memory fits into the same comparison API as loaded archives.
  - `Project.export_fits(format="csv")` — CSV + PNG dump of current in-memory fits, **one-way export**. CSV default; `format` kwarg reserved.
  - `FitResults.load(path) -> FitResults` — canonical entry point for inspecting an archive without a Project.
  - `FitResults.compare_models(...)`, `.find(...)`, `.get(...)`, `.files()`, `.models(...)`, `.plot_residuals(...)`, iteration — all browsing/comparison happens here.
  - `File.save_fit()` / `File.export_fit()` are 1-line delegates to Project (save/export are still fitting-workspace concerns).
  - **`File.compare_models()` is kept as a delegate** to `self.p.results.compare_models(file=self, ...)`. UX rationale: per-file comparison is the dominant pattern during model development, and File is the natural scope. The delegate is sugar — implementation lives entirely in `FitResults.compare_models`.
  - **`File.load_fit()` is dropped.** Loading is fundamentally an archive operation; the path argument dominates and `FitResults.load(path)` is the canonical entry. A `File.load_fit(path)` delegate would be no shorter than `project.load_fits(path, file=f)` and rarely what the user wants (usually they load the whole archive, then query).
  - `File.save_sbs_fit` / `File.save_2d_fit` become deprecated aliases (`DeprecationWarning`); removal scheduled before v1.0.0.
- **Self-contained archive.** Each file's group stores raw `data`, `energy`, `time` plus identity attrs. Required for archive portability (the original raw data file may not be present at load time) and as the canonical reference for what the file contained.
- **Per-slot `observed` array (not just `fit`).** Each slot stores both `fit` and `observed` on the **fit grid**, so residuals are unconditionally `observed - fit` with no recipe replay. Rationale: baseline fits operate on `data_base = np.mean(data[base_t_ind, :], axis=0)` ([trspecfit.py:1856](../../../src/trspecfit/trspecfit.py#L1856)), spectrum fits on `data_spec` (a slice or mean over a time range, [trspecfit.py:2173](../../../src/trspecfit/trspecfit.py#L2173)), and sbs/2d fits on data cropped by `e_lim` / `t_lim`. None of those equal raw `file.data`, so `file.data - slot.fit` has wrong shape or wrong grid. Storing `observed` per slot avoids encoding "how was this data view built" in the loader.
- **Stable, HDF5-safe group keys; identity lives in attrs.** All group-path components (files, slots) use **zero-padded positional keys** (`000000`, `000001`, ...). Human-meaningful identifiers (`File.name`, `original_path`, `model_name`, `fit_type`, `selection`) live as attrs on the group's `metadata`. Rationale: HDF5 path components forbid `/`, and user-facing names (especially `model_name` from YAML) can contain anything. Positional keys sidestep that entirely and match the `Simulator.save_data` precedent.
- **Canonical slot identity (mechanically defined; two-key form).** Each slot has explicit identity attrs:
  - `model_name` — user string, stored as attr only.
  - `fit_type` — `"baseline" | "spectrum" | "sbs" | "2d"`.
  - `selection_json` — JSON-serialized dict capturing the fit-view identity (see below).
  - **In memory**: `history_key = sha256(file_fingerprint | file_name | model_name | fit_type | selection_json)` — used by `_collapse_history_to_snapshot`, in-session dedup, comparison. `file_name` is included so two distinct `Project.files` with byte-identical raw arrays (same fingerprint, different names) don't collapse into a single slot. Project enforces unique `File.name` in-session, so name suffices to break the fingerprint tie; the archive's full identity is the `(fingerprint, name, original_path)` triple.
  - **In archive**: `archive_slot_key = sha256(file_ref | model_name | fit_type | selection_json)` — written to slot `metadata.attrs`, used for slot-scoped overwrite detection on save. Computed at save time only, after fingerprint → `file_ref` mapping.

  Both keys serve the same logical purpose ("uniquely identify this slot"); they just use different file-identity tokens because in-memory and on-disk identity primitives differ (fingerprint vs archive-local positional path). Two functions in `utils/fit_io.py`: `compute_history_key` and `compute_archive_slot_key`.

- **`selection_json` includes the full fit-view identity** so refits with different windows/limits don't collide on the canonical key. **All `*_lim` fields are index slices `[start, stop)` (matching `File.e_lim` / `File.t_lim` semantics — [trspecfit.py:1155](../../../src/trspecfit/trspecfit.py#L1155)), not absolute physical values.** The `_abs` parallels (`e_lim_abs`, `t_lim_abs`) are user-meaningful absolutes but are *not* used in the canonical key — the index form determines the actual fit grid, and `observed_sha256` catches any drift the indices happen to miss.
  - **baseline**: `{"base_t_ind": [start, stop), "e_lim": [start, stop) | None}` — `base_t_ind` is the time-window index slice averaged for `data_base`.
  - **spectrum**: `{"time_point": float | None, "time_range": [lo, hi] | None, "time_type": "abs" | "ind", "e_lim": [start, stop) | None}`.
  - **sbs**: `{"e_lim": [start, stop) | None, "t_lim": [start, stop) | None}`.
  - **2d**: `{"e_lim": [start, stop) | None, "t_lim": [start, stop) | None}`.

  Empty `{}` is no longer the default for any fit type; every slot carries the relevant view identity.

- **`observed_sha256` as a belt-and-suspenders cross-check.** Each slot stores `sha256(observed.tobytes())` as an attr. `compare_models` refuses to compare slots whose `observed_sha256` doesn't match (when comparing same fit_type on same file) — guards against silent drift if `selection_json` ever fails to capture a relevant view detail.
- **Two-tier identity (within-archive vs across-archive).** Use the right tool for the right job:
  - **Within an archive** (e.g. a slot pointing at its file): use the archive-local positional path `files/000000` — unambiguous and stable for the lifetime of that archive file.
  - **Across archive ↔ live Project** (matching a loaded `Project.files[*]` to an archive file on `load_fits`, or aligning two archives): use **content fingerprints**, not positional index. File matching uses `(data_sha256, energy_sha256, time_sha256, shape)` with `name` / `original_path` as tie-break metadata. Multiple shas + shape avoid the "identical replicate files share `data_sha256`" ambiguity that bare-data-hash matching would have.
- **Fit types covered (4):** `baseline`, `spectrum`, `sbs`, `2d`. `Project.fit_2d()` participates in v1 as ordinary per-file `fit_type="2d"` slots (one per file). **What's deferred is a project-scoped joint-result slot** that would own the shared parameter values without per-file duplication — the underlying joint pipeline is flagged as architecturally unfinished (an open item in [TODO.md](https://github.com/InfinityMonkeyAtWork/time-resolved-spectroscopy-fit/blob/main/TODO.md) as of archival), so locking in archive schema for that construct now is premature. Adding a joint slot later is a strict additive change. Add `spectrum` to `File.get_fit_results(fit_type=...)` while we're here (currently missing — [trspecfit.py:3022](../../../src/trspecfit/trspecfit.py#L3022)).
- **Stable chi-square / sigma semantics.** Raw objective diagnostics are always named `chi2_raw` / `chi2_red_raw`; σ-calibrated values are always named `chi2` / `chi2_red` and are `NaN` when no sigma was set. Sigma is file state (`File.set_sigma(...)`), inherited from flat project-YAML defaults when present, and materialized into each `SavedFitSlot` at fit completion as `noise_type`, `sigma_source`, `sigma_type`, `sigma_data`, and fit-view-specific `sigma_eff`. `File.set_sigma()` is forward-looking and does not rewrite existing `_fit_history` slots or archives. `compare_models()` has no `sigma=` kwarg; it only reads slot state. If calibrated metrics are explicitly requested when no matched slot has sigma, it raises with a pointer to `file.set_sigma(...)` / the raw metric name.
- **Fit-quality metrics computed and stored:** `chi2_raw`, `chi2_red_raw`, `chi2`, `chi2_red`, `r2`, `aic`, `bic`. Per-slice for SbS; single value for baseline / spectrum / 2d. Small helper in `fitlib`: `(observed, fit, n_free_pars, sigma_eff=None) → dict` — takes the observed data view actually fit against, not raw `file.data`.

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
        ├── metrics:     {chi2_raw, chi2_red_raw, chi2, chi2_red, r2, aic, bic}
        │                  # per-slice arrays for sbs; calibrated fields NaN when no sigma
        ├── noise:       noise_type, sigma_source, sigma_type, sigma_data, sigma_eff
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
# Project.fit_2d() appends per-file 2d slots like any other fit; a project-scoped
# joint-result slot is deferred to v2.

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
    metrics: list[str] | None = None,        # dynamic default; see below
    sbs_aggregation: Literal["median", "mean", "sum", "long"] = "median",
    plot_residuals: bool = False,
) -> pd.DataFrame
# For SbS slots, per-slice metrics are aggregated to a scalar via sbs_aggregation
# before being placed in the comparison DataFrame.
#   "median"  — robust; default. One row per slot, value = np.median(per_slice).
#   "mean"    — average across slices.
#   "sum"     — sum for additive metrics (chi2_raw/chi2/aic/bic);
#               chi2_red_raw and chi2_red aggregate as Σnumerator / ΣDoF.
#   "long"    — return per-slice rows; comparison DataFrame gets a slice_index column.
# Default columns:
#   no sigma:   ["chi2_red_raw", "r2", "aic", "bic"]
#   with sigma: ["chi2_red_raw", "sigma_eff", "chi2_red", "r2", "aic", "bic"]
# There is intentionally no compare_models(sigma=...) view; sigma enters via File.set_sigma().
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
File.set_sigma(sigma, *, noise_type=None, sigma_source="user_supplied", sigma_type="constant")
# Sets per-file sigma for future fits only. Existing slots keep their materialized sigma snapshot.
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

- `File.save_sbs_fit` ([trspecfit.py:2559](../../../src/trspecfit/trspecfit.py#L2559)) — wide CSV + PNGs via `fitlib.results_to_df` / `results_to_fit_2d` / `plt_fit_res_2d`. Logic moves to `Project.export_fits` (CSV path); method becomes deprecated alias.
- `File.save_2d_fit` ([trspecfit.py:2979](../../../src/trspecfit/trspecfit.py#L2979)) — only plots data/fit/residual maps. Parameter CSVs *are* written, but earlier in `fit_2d()` itself via `fit_wrapper(..., save_output=1)` at [trspecfit.py:2951](../../../src/trspecfit/trspecfit.py#L2951), not by `save_2d_fit`. So the persistence path is split across two methods today. Same fate as `save_sbs_fit`: the CSV-writing logic moves into `Project.export_fits` (CSV path), `save_2d_fit` becomes a deprecated alias.
- Baseline fit save path in `fitlib` writes per-table CSVs ([fitlib.py:743+](../../../src/trspecfit/fitlib.py#L743)) — logic moves to `export_fits` CSV path.
- `File.load_fit` ([trspecfit.py:2265](../../../src/trspecfit/trspecfit.py#L2265)) is a stub. **Removed in v1** — replaced by `FitResults.load(path)` and `Project.load_fits(path)`.
- `File.compare_models` ([trspecfit.py:3077](../../../src/trspecfit/trspecfit.py#L3077)) is a stub. Becomes a thin delegate to `self.p.results.compare_models(file=self, ...)`.
- `File.fit_spectrum` ([trspecfit.py:2085](../../../src/trspecfit/trspecfit.py#L2085)) — 1D fit at a `time_point` / `time_range`. Slot identity must include those.
- `File.get_fit_results(fit_type=...)` ([trspecfit.py:3019](../../../src/trspecfit/trspecfit.py#L3019)) returns DataFrames for `baseline` / `sbs` / `2d`; **`spectrum` missing** — fix as part of this work.
- `File` always has a parent Project ([trspecfit.py:1113](../../../src/trspecfit/trspecfit.py#L1113)) — `self.p` is never None, so File-level delegates rely on it unconditionally.
- `Simulator.save_data` ([simulator.py:1386](../../../src/trspecfit/simulator.py#L1386)) is the structural template we follow.

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
│   │       │   │                             # noise attrs:
│   │       │   │                             #   noise_type, sigma_source, sigma_type,
│   │       │   │                             #   sigma_data, sigma_eff
│   │       │   │                             # metrics attrs:
│   │       │   │                             #   chi2_raw, chi2_red_raw, chi2,
│   │       │   │                             #   chi2_red, r2, aic, bic
│   │       │   ├── params                    # dataset: structured (name, value, init_value, stderr, min, max, vary, expr)
│   │       │   ├── observed                  # dataset: data view that was fit (data_base / data_spec / cropped); same shape as `fit`
│   │       │   ├── fit                       # dataset: model evaluated at final params (1D or 2D)
│   │       │   ├── metrics_per_slice         # dataset: 2D (slices × {chi2_raw, chi2_red_raw, chi2, chi2_red, r2, ...}) — sbs only
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
- [x] Add `fitlib.compute_fit_metrics(observed, fit, n_free_pars, sigma_eff=None) -> dict` returning `{chi2_raw, chi2_red_raw, chi2, chi2_red, r2, aic, bic}`. Takes **`observed`** (the actual data view fit against), not raw `file.data`. Raw fields match the unweighted objective diagnostics; calibrated fields are populated only when `sigma_eff` is finite.

**Note on the observed/fit/metrics capture:** the original precursor wording
("wire metric computation … so the values exist on `Model.result`") is
**intentionally dropped**. `Model` should not carry archive/history concerns —
that creates two sources of truth. Instead, `SavedFitSlot` is the first owner
of `observed`, `fit`, `metrics`, `observed_sha256`, `selection_json`, and
`history_key`. The fit-path → snapshot args → `_slot_from_<fit_type>` →
`_fit_history` pipeline captures and computes everything in one shot at fit
completion. See "Object model + I/O" below.

### Object model + I/O

- [x] Define `SavedProject` / `SavedFile` / `SavedFitSlot` dataclasses (probably in `utils/fit_io.py`). **All three done; `SavedFitSlot` at [utils/fit_io.py:42](../../../src/trspecfit/utils/fit_io.py#L42), `SavedFile` and `SavedProject` at [utils/fit_io.py:120-200](../../../src/trspecfit/utils/fit_io.py#L120-L200) (frozen dataclasses; tuple-of-slots / tuple-of-files for immutability).**
- [x] Define `FitResults` class in new module `trspecfit/fit_results.py`, exported as `trspecfit.FitResults`. Includes `load` classmethod, `find` / `get` / `files` / `models` / `__iter__` query API, and `compare_models` / `plot_residuals`. Internal key is `(file_fingerprint, model_name, fit_type, selection_json)`; name-based queries resolve to fingerprint internally. Constructor accepts a list of `SavedFitSlot` (used by both `load` and the `Project.results` wrapper path). **Done. Skeleton + query API + `load` at [fit_results.py:46](../../../src/trspecfit/fit_results.py#L46). `compare_models` at [fit_results.py:212](../../../src/trspecfit/fit_results.py#L212) — filters on `(file, models, fit_type)`, defends against silent grid drift via the `observed_sha256` cross-check (raises if two slots in the same `(file_fingerprint, fit_type)` group disagree), and aggregates SbS per-slice metrics with `sbs_aggregation` ∈ `{"median", "mean", "sum", "long"}`; `"long"` emits one row per slice. `file=` accepts `str | SavedFile | trspecfit.File` (anything with `.name`). `plot_residuals` at [fit_results.py:330](../../../src/trspecfit/fit_results.py#L330) — smoke-test-grade side-by-side panels for 1D fits and residual heatmaps for SbS / 2D; uses index axes since slots do not carry parent-file energy/time arrays. Both methods covered by tests in `tests/test_fit_history.py::TestFitResultsCompareModels` (13 cases) and `TestFitResultsPlotResiduals` (5 cases).**
- [x] Add `Project._fit_history: list[SavedFitSlot]` attr (initialized to `[]` in `Project.__init__`). [trspecfit.py:179](../../../src/trspecfit/trspecfit.py#L179)
- [x] Implement per-fit-type extraction helpers in `utils/fit_io.py`. **Each helper takes already-copied snapshot args** (not live `File.model_*` references) so call-site ordering is irrelevant — the helper cannot be broken by post-fit cleanup like the seed-template restoration at [trspecfit.py:2551](../../../src/trspecfit/trspecfit.py#L2551). Signatures (omit `conf_ci` / `mcmc` kwargs and identity args `file_name` / `model_name` for brevity; all four take them):
  - `_slot_from_baseline(*, file_fingerprint, ..., params_df, observed, fit, base_t_ind, e_lim, n_free_pars, noise_type, sigma_source, sigma_type, sigma_data) -> SavedFitSlot`
  - `_slot_from_spectrum(*, file_fingerprint, ..., params_df, observed, fit, time_point, time_range, time_type, e_lim, n_free_pars, noise_type, sigma_source, sigma_type, sigma_data) -> SavedFitSlot`
  - `_slot_from_sbs(*, file_fingerprint, ..., params_df, observed, fit, e_lim, t_lim, n_free_pars, noise_type, sigma_source, sigma_type, sigma_data) -> SavedFitSlot` — caller passes the already-built per-slice DataFrame (from a copy of `results_sbs`) before any seed-template restoration.
  - `_slot_from_2d(*, file_fingerprint, ..., params_df, observed, fit, e_lim, t_lim, n_free_pars, noise_type, sigma_source, sigma_type, sigma_data) -> SavedFitSlot`

  Each helper computes `metrics` (via `compute_fit_metrics`, threading `sigma_eff` derived from `sigma_data` + selection — `σ / √N_avg` for baseline, σ verbatim elsewhere), `observed_sha256`, `selection_json`, `history_key`, and materializes the 5 noise fields onto the slot. The bare `File._project_fit_result` 5-tuple from a joint `Project.fit_2d()` is not separately extracted in v1; the per-file slots produced inside `Project.fit_2d` go through `_slot_from_2d` like any other 2d fit. **Done at [utils/fit_io.py:247-431](../../../src/trspecfit/utils/fit_io.py#L247-L431).**
- [x] Wire eager extraction into the four fit code paths. Call site is responsible for capturing snapshot args **at the moment results are valid**:
  - `fit_baseline`: extract immediately after fit completes, before any further mutation.
  - `fit_spectrum`: same; capture `time_point` / `time_range` / `time_type` from fit args.
  - `fit_slice_by_slice`: extract **before** [trspecfit.py:2551](../../../src/trspecfit/trspecfit.py#L2551) (the seed-template restoration that would otherwise blow away `model_sbs.parameter_names`/result state). Snapshot the relevant fields into local copies, then call the helper.
  - `fit_2d`: extract immediately after fit completes.

  All four append the resulting slot to `self.p._fit_history`. **Done via `_append_baseline_slot` / `_append_spectrum_slot` / `_append_sbs_slot` / `_append_2d_slot` ([trspecfit.py:2795-3053](../../../src/trspecfit/trspecfit.py#L2795-L3053)), called from [fit_baseline](../../../src/trspecfit/trspecfit.py#L2122), [fit_spectrum](../../../src/trspecfit/trspecfit.py#L2348), [fit_slice_by_slice](../../../src/trspecfit/trspecfit.py#L2703), [fit_2d](../../../src/trspecfit/trspecfit.py#L3412), and [Project.fit_2d](../../../src/trspecfit/trspecfit.py#L1009).**
- [x] Implement `Project.results` property: returns `FitResults(slots=list(self._fit_history))`. Cheap: no array copies, just a list snapshot. [trspecfit.py:239](../../../src/trspecfit/trspecfit.py#L239)
- [x] Finalize HDF5 schema (structured-array dtypes, attr keys, MCMC layout) and document in `docs/design/`. **All group-path components are positional zero-padded keys; user-facing names live only in attrs.** Documented at [docs/design/fit_archive_schema.md](../fit_archive_schema.md).
- [x] Add identity-key helpers in `utils/fit_io.py`:
  - `compute_history_key(file_fingerprint, file_name, model_name, fit_type, selection_json) -> str` — sha256, used in-memory. `file_name` was added so two distinct `Project.files` with byte-identical raw arrays don't collapse into a single slot during snapshot save; Project enforces unique `File.name` in-session, so name suffices to break the fingerprint tie.
  - `compute_archive_slot_key(file_ref, model_name, fit_type, selection_json) -> str` — sha256, used at save time once `file_ref` is known. **Done at [utils/fit_io.py:209](../../../src/trspecfit/utils/fit_io.py#L209).**
  - `compute_file_fingerprint(data, energy, time) -> dict[str, str]` — multi-sha (`data_sha256`, `energy_sha256`, `time_sha256`, `shape`).
  - `compute_observed_sha256(observed) -> str` — for the slot's defensive cross-check.
  - `build_selection_json(fit_type, **fields) -> str` — deterministic JSON serialization (sorted keys) so equivalent selections produce identical hashes.
- [x] Add `_find_slot_by_archive_key(file_group, archive_slot_key) -> Group | None` and `_find_file_by_fingerprint(archive, fingerprint) -> Group | None` helpers — used by overwrite detection (save) and project-matching (load). **Done at [utils/fit_io.py:567](../../../src/trspecfit/utils/fit_io.py#L567) and [utils/fit_io.py:614](../../../src/trspecfit/utils/fit_io.py#L614). `_find_file_by_fingerprint` accepts optional `name` / `original_path` tie-break args (required-when-passed, per the write-side identity rule); read-side callers omit them for fingerprint-only matching.**
- [x] Add `_collapse_history_to_snapshot(slots: list[SavedFitSlot]) -> list[SavedFitSlot]` helper: keeps the latest slot per `history_key` (snapshot semantics for default `save_fits`). **Implemented as `collapse_history_to_snapshot` (no leading underscore — module-public) at [utils/fit_io.py:525](../../../src/trspecfit/utils/fit_io.py#L525).**
- [x] Implement writer in `utils/fit_io.py`: takes a list of slots (already filtered + collapsed), serializes to HDF5. The writer is *slot-driven*; it does not walk `Project` or live `File.model_*` — that walking is done at fit-completion time by the extraction helpers, with the result accumulating in `_fit_history`. **Done. Entry point `write_archive(filepath, *, project: SavedProject, overwrite=False)` at [utils/fit_io.py:776](../../../src/trspecfit/utils/fit_io.py#L776). Append-mode default: existing archives are augmented in place; `timestamp_created` is preserved, `timestamp_updated` is rewritten on every save. Slot collisions are pre-checked across all files before any mutation, so a single conflicting slot never leaves a half-written payload (`_precheck_slot_collisions`). Helpers below it: `_validate_archive_compatibility` (rejects schema-version mismatch on append), `_write_top_metadata`, `_write_file_payload`, `_write_slot`, `_write_slot_metadata`, `_write_slot_params` (per-fit-type type-tag dispatch), `_write_metrics_per_slice` (sbs), `_write_mcmc_group`. DataFrame encoding uses the unified `_encode_dataframe` helper (homogeneous → 2D float64 + `columns` attr; heterogeneous → structured `c000000`-fields + `columns`/`dtypes` attrs). The complementary input-builder for `Project.save_fits` (slots → `SavedProject`) is part of step 16, not the writer.**
- [x] Implement reader in `utils/fit_io.py`: deserializes HDF5 into a list of `SavedFitSlot` (plus `SavedFile` records for raw arrays). **Does not touch live `File.models` or `_fit_history`.** **Done. Entry point `read_archive(filepath) -> SavedProject` at [utils/fit_io.py:1305](../../../src/trspecfit/utils/fit_io.py#L1305) (line numbers approximate); inverse of `write_archive`. Per-section helpers: `_decode_dataframe` (inverse of `_encode_dataframe`, handles both all-numeric and heterogeneous forms), `_read_metrics_per_slice`, `_read_mcmc_group` (NaN→None for `lnsigma`), `_read_slot` (recomputes `history_key` from fingerprint + identity attrs per schema; on-disk value is debug-only), `_read_file`. Strict `schema_version` check on entry. Source dtype preserved through `[...]`-read of arrays. `FitResults.load(path)` at [fit_results.py:46](../../../src/trspecfit/fit_results.py#L46) wraps it. Round-trip verified for baseline, sbs, conf_ci with awkward sigma labels, mcmc with flatchain + ci, and float32 raw arrays — all fields match incl. dtypes, `history_key`, `observed_sha256`, and per-slice metrics. Pyright clean.**

### Project-level API

- [x] `Project.save_fits()` — filter `_fit_history` by `(file, model, fit_type)`, collapse to snapshot via `_collapse_history_to_snapshot` (using `history_key`), then for each slot: resolve `file_fingerprint → file_ref` (look up or create the file group in the archive), compute `archive_slot_key`, check for existing slot, write or error per `overwrite=True/False`. **Done at [trspecfit.py:296](../../../src/trspecfit/trspecfit.py#L296). Default path `./fit_results/<project_name>.fit.h5`. `file=` accepts `int | str | File | Sequence`; `model` / `fit_type` accept `str | Sequence`. Filter / grouping / live-file lookup all key on the **`(fingerprint, file_name)` tuple**, not fingerprint alone, so two `Project.files` with byte-identical raw arrays but distinct names are kept separate (matches the archive's `(fingerprint, name, original_path)` identity rule). Collapses via `collapse_history_to_snapshot`, then groups by `(fingerprint, file_name)` and looks up the live `Project.files[*]` via `_find_file_for_slot` (requires both name and fingerprint to match). Helpers `_resolve_save_file_filter` (returns `set[(fp_key, name)]`) and `_find_file_for_slot` live on Project; module-level `_fp_key`, `_to_str_set`, `_trspecfit_version` at [trspecfit.py:99](../../../src/trspecfit/trspecfit.py#L99).**
- [x] `Project.load_fits()` — thin wrapper that returns `FitResults.load(path, ...)`. Does not mutate Project state. **Done at [trspecfit.py:411](../../../src/trspecfit/trspecfit.py#L411). Pure delegate; filter args (`file` / `model` / `fit_type`) accept `str | Sequence` and pass through to `FitResults.load`, which now supports load-time filtering on `slot.file_name` / `model_name` / `fit_type`.**
- [x] `Project.export_fits(format="csv")` — same filter pipeline as `save_fits`, but emits CSV+PNGs instead of HDF5. Absorbs CSV+PNG logic from current `File.save_sbs_fit` / `save_2d_fit` + baseline-CSV path in `fitlib`. **Done at [trspecfit.py:308](../../../src/trspecfit/trspecfit.py#L308). Default path `./fit_results/<project_name>/`. Filter / collapse pipeline shared with `save_fits` via the new `Project._build_saved_project_from_history` helper, so both methods see identical slot grouping. Output layout: `<root>/<file_name>/<model_name>__<fit_type>[__<hash>]/...`; the `__<hash>` (first 8 chars of `history_key`) suffix appears only when more than one slot in the snapshot shares the `(file, model, fit_type)` triple. Per-slot artifacts: `params.csv`, `metrics.csv` (or `metrics_per_slice.csv` for sbs), optional `conf_ci.csv` / `mcmc/flatchain.csv` / `mcmc/ci.csv`. Per fit type: 1D fits get `fit_1d.csv` (energy, observed, fit, residual); sbs/2d get `fit_2d.csv` + `observed_2d.csv` + `energy.csv` + `time.csv` + `2D_data_fit_res.png`; sbs additionally gets `fit_pars.csv` (parity with `results_to_df`) plus per-parameter PNGs from `plt_fit_res_pars`. Overwrite is per-slot directory and pre-checked across all slots before any writes (mirrors `_precheck_slot_collisions` in the writer). Slot-driven serialization lives in `fit_io.write_csv_export` ([utils/fit_io.py:1517](../../../src/trspecfit/utils/fit_io.py#L1517)) so the export never reaches into live `Model` state.**

### File-level delegates + deprecation

- [x] `File.save_fit()` / `export_fit()` / `compare_models()` as 1-line delegates. `save_fit` / `export_fit` route to `self.p.save_fits` / `self.p.export_fits`; `compare_models` routes to `self.p.results.compare_models(file=self, ...)`. **Do not add `File.load_fit`** — load is path-scoped (use `FitResults.load(path)` or `Project.load_fits(path)`). **All three done. `File.save_fit` at [trspecfit.py:2777](../../../src/trspecfit/trspecfit.py#L2777); `File.export_fit` at [trspecfit.py:2805](../../../src/trspecfit/trspecfit.py#L2805) (mirrors `Project.export_fits` kwargs); `File.compare_models` at [trspecfit.py:4022](../../../src/trspecfit/trspecfit.py#L4022) — takes positional `*models` per the PLAN spec and forwards to `self.p.results.compare_models`. The pre-existing `File.load_fit` stub was later removed outright (no callers in src/tests/docs/examples; load is path-scoped via `FitResults.load` / `Project.load_fits`).**
- [x] Convert `File.save_sbs_fit` / `save_2d_fit` to deprecated aliases (`DeprecationWarning`); add removal-before-v1.0.0 marker in code. **Done. Renamed the legacy implementations to `_save_sbs_fit_legacy` / `_save_2d_fit_legacy` (private; still used by the auto-export path inside `fit_slice_by_slice` / `fit_2d` / `Project.fit_2d`); replaced the public `save_sbs_fit` / `save_2d_fit` with thin wrappers at [trspecfit.py:2848-2885](../../../src/trspecfit/trspecfit.py#L2848) that emit `DeprecationWarning(stacklevel=2)` pointing at `File.export_fit`, then call the legacy impl. Behavior preserved byte-for-byte for users who haven't migrated. Tests added in `tests/test_file.py::TestFitPreconditions::test_save_sbs_fit_emits_deprecation_warning` / `test_save_2d_fit_emits_deprecation_warning`. The mock-patch in `test_fit_sbs_model_seed_allows_no_baseline_fit` was rerouted to `_save_sbs_fit_legacy` since that is now the call path.**
- [x] Track v1.0.0 removal in TODO.md under "Build & release → Remove legacy/backwards-compat code." **Done. Sub-bullets added to the existing "Remove legacy/backwards-compat code" item naming `File.save_sbs_fit` / `File.save_2d_fit` (and their `_save_*_fit_legacy` impls). The `File.load_fit` stub was initially tracked here too, but was later removed outright (see below), so that tracking entry was dropped.**

### Tests + docs

- [x] Round-trip tests: save → load → compare metrics / param tables / fit / observed arrays match. Cover basic / profile / profile+dynamics models, all four fit types where applicable. Verify `observed - fit` reproduces residuals for each fit_type without reading `file.data`. **Done at [tests/test_fit_archive_roundtrip.py](../../../tests/test_fit_archive_roundtrip.py) — 11 tests covering F1 (basic) baseline/spectrum/sbs, F3 (basic+dynamics) 2d, F6 (profile-only) baseline/spectrum/sbs, F8 (profile+dynamics) baseline/2d, plus a `Project.load_fits` ↔ `FitResults.load` parity test and a multi-slot (baseline+spectrum+sbs in one archive) round-trip. F6 spectrum specifically exercises the profile path through `fit_spectrum` (per-spectrum lmfit params include the profile sub-parameters, and the serialized params DataFrame must round-trip those rows + their min/max/expr metadata). The shared `_assert_slot_round_tripped` helper checks identity (fingerprint, hashes, selection, history_key, observed_sha256), arrays (shape + dtype + bytewise equality), metrics (scalar or per-slice), params (column-by-column to handle `expr` None ↔ "" and `stderr` None ↔ NaN round-trips), provenance, and the PLAN invariant that `observed - fit` reproduces chi2 on the loaded slot alone.**
- [x] `_fit_history` tests: fit modelA-baseline, fit modelB-baseline, verify history has both slots and `Project.results` exposes both. Refit modelA-baseline, verify history has *all three* slots. Save with default snapshot semantics, verify archive has only two slots (one per `history_key`, latest wins). **Done at [tests/test_fit_history.py::TestHistoryAccumulationAndSnapshot](../../../tests/test_fit_history.py) — 3 tests using `single_glp` + `two_glp_expr_amplitude` as the two distinct models on a shared fit file. Verifies (a) `_fit_history` keeps all 3 slots in fit order, (b) `Project.results.find(model="single_glp")` exposes both refits and they share a history_key, (c) snapshot save collapses to 2 distinct slots and the surviving `single_glp` slot's `timestamp` matches the third (latest) fit.**
- [x] **Selection-identity tests**: refit baseline with different `base_t_ind`, refit sbs/2d with different `e_lim`/`t_lim`, refit spectrum with different `time_point` — verify `history_key` differs in each case, snapshot collapse keeps both, archive stores both as distinct slots. **Done at [tests/test_fit_history.py::TestSelectionIdentity](../../../tests/test_fit_history.py) — covers `base_t_ind` (baseline), `e_lim` (sbs), `t_lim` (2d). Spectrum `time_point` was already covered at `TestSpectrumSlot::test_refit_at_different_time_point_creates_distinct_slots`. Each test verifies distinct `history_key` values, the captured `selection` field reflects the right index slice, and the archive holds both slots after a snapshot save.**
- [x] **Snapshot semantics tests**: capture `r1 = p.results`, run another fit, verify `r1` does not see the new slot (frozen list), `r2 = p.results` does. **Already covered at [tests/test_fit_history.py::TestResultsSnapshot](../../../tests/test_fit_history.py) — `test_results_returns_fresh_wrapper` (object-identity per access) and `test_returned_results_is_frozen_against_subsequent_fits` (captured FitResults stays len=1 after a second fit; new access shows len=2).**
- [x] **SbS extraction-timing test**: simulate the seed-template restoration at [trspecfit.py:2551](../../../src/trspecfit/trspecfit.py#L2551); verify the extracted slot still has correct `params_per_slice` / `parameter_names` / metrics (helper used copied snapshot args, not live state). **Already covered at [tests/test_fit_history.py::TestSbSSlot::test_sbs_slot_survives_seed_template_restoration](../../../tests/test_fit_history.py) — runs a real `fit_slice_by_slice` (which ends with `model_sbs.update_value(seed_template)`) and asserts the captured slot still has finite per-slice metrics and a params row per time slice.**
- [x] **`observed_sha256` cross-check test**: construct two slots with same canonical key but mutated observed array; `compare_models` raises (or warns clearly) on grid mismatch. **Already covered at [tests/test_fit_history.py::TestFitResultsCompareModels::test_observed_mismatch_raises](../../../tests/test_fit_history.py) (and three companion tests verifying the cross-check is *not* triggered across different fit_types, different files, or replicate-but-distinct files).**
- [x] `compare_models` tests: two models on same file, returns expected metrics ordering; residual plot smoke test. Multi-version compare on same canonical key (multiple takes on modelA-baseline) — verify default behavior picks latest, `find` exposes all. SbS aggregation: test all four `sbs_aggregation` modes on a multi-slice fit. **Already covered at [tests/test_fit_history.py::TestFitResultsCompareModels](../../../tests/test_fit_history.py) (13 cases incl. `test_sbs_aggregation_modes` for median/mean/sum and `test_sbs_long_mode_emits_per_slice_rows` for "long") and [tests/test_fit_history.py::TestFitResultsPlotResiduals](../../../tests/test_fit_history.py) (5 cases). Multi-version `find()` exposure is now also verified at `TestHistoryAccumulationAndSnapshot::test_results_exposes_all_history_entries`.**
- [x] `export_fits` parity tests: same column shapes as old `save_sbs_fit` / `save_2d_fit` outputs. **Done at [tests/test_export_fits_parity.py](../../../tests/test_export_fits_parity.py) — 3 tests (`test_sbs_export_parity`, `test_2d_export_parity`, `test_2d_export_includes_new_artifacts`). The fit-side project's `path_results` is rerouted into `tmp_path/legacy/` so the auto-export path inside `fit_slice_by_slice` / `fit_2d` lands in the test sandbox; `project.export_fits` writes into a sibling `tmp_path/new/` tree. Parity is asserted on `fit_pars.csv` (legacy emits a redundant pandas auto-index — stripped before comparison; meaningful columns + per-slice values match exactly), `fit_2d.csv` (shape **and values** via `assert_allclose(rtol=0, atol=0)` — both SbS and 2D paths, since asserting shape alone would let a right-sized wrong-matrix bug slip through), `energy.csv`, `time.csv`, and the per-parameter PNG set. The new-artifacts test documents the additive payload (`observed_2d.csv`, `params.csv`, `metrics.csv` with the stable raw/calibrated metric schema) so a future regression that drops one fails loudly.**
- [x] DeprecationWarning tests for the old aliases. **Already covered at [tests/test_file.py::TestFitPreconditions::test_save_sbs_fit_emits_deprecation_warning](../../../tests/test_file.py) and `test_save_2d_fit_emits_deprecation_warning` — both `pytest.warns(DeprecationWarning, match="export_fit")`.**
- [x] **Noise-schema test coverage**: the σ work has dedicated test classes so a future reader can see it was tested intentionally, not by accident. `File.set_sigma` + `normalize_sigma_data` validation (incl. NaN-as-unset and the YAML-omits-`sigma_data` regression) at [tests/test_file.py::TestSetSigma](../../../tests/test_file.py) (12 cases); stable raw/calibrated `compare_models` column set, missing-σ `KeyError`, and SbS sum-mode aggregate-reduced χ² at [tests/test_fit_history.py::TestFitResultsCompareModelsSigmaColumns](../../../tests/test_fit_history.py) (8 cases); slot-side noise-field + 7-key-metric round-trip with NaN-aware comparisons in `_assert_slot_round_tripped`, exercised by every case in [tests/test_fit_archive_roundtrip.py](../../../tests/test_fit_archive_roundtrip.py).
- [x] Update example notebooks to demo `Project.save_fits` / `Project.load_fits` / `Project.export_fits` / `compare_models`. **Done at `examples/fitting_workflows/10_model_comparison/` — a self-contained notebook that generates synthetic data inline (kicked-decay pump-probe with a Gaussian IRF and strongly-Lorentzian peak), fits two competing models at three levels (baseline / SbS / 2D), calls `file.compare_models(...)` on each, persists via `file.save_fit("comparison.fit.h5")`, reloads through `FitResults.load(...)`, and exercises `sbs_aggregation="long"` for per-slice inspection. Also documents the stable σ-calibrated column schema (`chi2_red_raw` / `sigma_eff` / `chi2_red`), shows the `file.set_sigma(NOISE_SIGMA)` one-shot setup, and demonstrates the pandas one-liner for what-if recalibration of loaded archives. Re-executes end-to-end without auto-export side effects (`auto_export: False` in `project.yaml`).**
- [x] Update `docs/design/repo_architecture.md` with the new `utils/fit_io.py` module and the save/export split. **Done. Added a `fit_results.py` entry under top-level modules, a `utils/fit_io.py` entry under utils, a new "Fit results: save / export / load architecture" section with the slot-driven pipeline diagram (eager extraction → `_fit_history` → save/export/results, plus the load → `FitResults` arm), the deprecated-alias status, the save-vs-export distinction, and updated the "Typical execution flow" + "Where to put new code" guides. Also removed the dead `File.load_fit` TODO stub from `trspecfit.py` (no callers in src/tests/docs/examples) and dropped its entry from TODO.md so the doc claim "load is path-scoped" matches the codebase.**

## Out of scope (deferred to v2 if users demand it)

- **Auto-save (implicit persistence on every fit).** v1 keeps fit history in memory only (`Project._fit_history`); persistence is explicit (`Project.save_fits()`). Auto-save is orthogonal to the in-memory history mechanism — it can be added later as an opt-in `Project(auto_save_path=...)` init kwarg, where each `_fit_history.append` also serializes incrementally to the archive. Deferred until we see how much friction explicit save actually causes in real workflows; standard scientific-Python idiom is explicit persistence (pandas, lmfit, NumPy all require explicit `save`/`to_csv`/`pickle`).
- ~~**`auto_export` opt-out toggle for fit-completion side effects.**~~ **Implemented** (2026-05-17, in this branch). `Project.auto_export: bool = True` lives in `Project._set_defaults`, picks up YAML overrides via the existing config loop, and gates all four `fit_wrapper(save_output=...)` calls plus the five auto `save_*_fit` / `_save_*_fit_legacy` call sites in `fit_baseline` / `fit_spectrum` / `fit_slice_by_slice` (serial + parallel worker) / `fit_2d` / `Project.fit_2d`. The in-fit `plt_fit_res_1d` calls are *skipped entirely* when neither saving nor showing is wanted (not just save-suppressed) — critical for SbS where building each per-slice figure is non-trivial work; baseline/spectrum use explicit `save_plot` / `show_plot` booleans gating both the call and the `save_img` int via `utils.plot._save_img_flag(save=..., show=...)`. Explicit `File.export_fit` / `Project.export_fits` / `Project.save_fits` are unaffected. Coverage: `tests/test_auto_export.py` (10 tests: default-true, post-init flip, baseline-no-files, 2D-no-files, baseline-writes-by-default, explicit `export_fits` works under `auto_export=False`, explicit `save_fits` works under `auto_export=False`, plus three monkeypatch tests confirming `plt_fit_res_1d` is not called when silent + no export, IS called when verbose even without export, and per-slice SbS plotting is skipped under `auto_export=False`).
- **MCMC decoupled from `fit_wrapper`.** Today `fit_wrapper` bundles optimization, confidence intervals, MCMC, and export; `mc_settings.use_emcee=2` will silently kick off a potentially very expensive MCMC run when CI fails. Cleaner shape: `fit_*()` runs optimization only and appends a normal `SavedFitSlot`; users inspect via `Project.results`, then explicitly call something like `project.run_mcmc(slot=...)` / `file.run_mcmc(fit_type=..., model=...)` on the fits worth interrogating. CI can fail and say so without secretly upgrading the call. **Schema wrinkle to resolve when this lands**: `SavedFitSlot.mcmc` exists in the v1 schema as an optional sub-record; with append-only history a separate run shouldn't mutate an existing slot in place. Either produce an enriched-copy slot keyed by the same `history_key`, or introduce a sibling `SavedMCMCResult` keyed by `history_key` — both keep in-session history append-only. Drops `mc_settings` / `use_emcee=2` fallback from `fit_wrapper` at the same time. Deferred — not blocking v1 of the save/load work, but worth doing before MCMC sees real use, since changing the call surface later breaks more callers than now.
- **`keep_history=True` for full-log save.** `save_fits` currently collapses to latest-per-canonical-key (snapshot). Saving the full append-only log of refits would let archives preserve every iteration but requires schema work to disambiguate same-key slots (timestamp or sequence number in the canonical key). Deferred — most users want snapshot semantics; revisit if the in-session multi-version-compare workflow grows into a "preserve every refit" need.
- **Memory cap / history pruning.** v1's `_fit_history` is unbounded. For typical sessions (10s of fits, MB-size data) this is fine. If long sbs/2d-heavy sessions show memory growth, add a config knob (`Project(history_max_slots=N)` or similar) or a `Project.clear_history()` method. Defer until measured.
- **Project-scoped joint-result slot.** `Project.fit_2d()` runs a joint multi-file fit but currently emits one ordinary `fit_type="2d"` slot per file (each carrying that file's projection of the joint result), so its results *are* in `_fit_history` and the archive. What's deferred is a separate "joint" archive construct that owns the shared parameter values without per-file duplication; the bare `File._project_fit_result` 5-tuple is also not separately captured. Reasoning: the project-level fit pipeline itself is flagged as architecturally unfinished (an open item in [TODO.md](https://github.com/InfinityMonkeyAtWork/time-resolved-spectroscopy-fit/blob/main/TODO.md) as of archival — lowering the multi-file residual to GIR), so locking in archive schema for the joint construct now would be premature. v1 covers `baseline` / `spectrum` / `sbs` / `2d` (the file-level fits, which is the 95% case); per-file 2d projections from joint fits ride that path. Adding a joint slot later is a strict additive change to the schema and the `SavedProject` hierarchy.
- **Model rehydration / warm restore.** Reconstructing live `Model` objects from the archive (with profiles, dynamics, programmatic mutations intact) so users can resume fitting or call `model.create_value_2d()` on a loaded fit. v1 stores the *output* fit array instead, which covers inspection and comparison without the fragility.
- **YAML round-trip from archive.** v1 stores `yaml_filename` as a breadcrumb only; not promised to deserialize back into a Model.
- **Non-CSV export formats** (parquet, mat, json) — `format` kwarg reserved.
- **Resumable partial writes** (interrupting `save_fits` mid-write).
- **A `save_outputs` / `save_type` setting in `project.yaml`** — saves stay method-driven for now.
