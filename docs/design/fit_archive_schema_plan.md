---
orphan: true
---

# Fit-archive schema 7 — conversion plan

Status: **planned, not implemented.** Branch `fit-archive-schema-7`.

This document is the **conversion plan**: what changes on disk, what changes in
the object model, in what order, and what proves it. It deliberately does not
restate rationale — every "why" lives in
[fit_archive_principles.md](fit_archive_principles.md), which is settled and
authoritative. An earlier draft duplicated the principles and their decision
table here and drifted out of sync within a day; this version keeps one copy of
every claim.

[fit_archive_schema.md](fit_archive_schema.md) documents the current wire format
(schema 6) and stays authoritative until this conversion lands, at which point
it is rewritten as the self-contained schema-7 spec and this plan moves to
`docs/design/archive/`.

## What each principle changes on disk

| Principle | On-disk consequence |
|---|---|
| 0 — fit archive, not project serializer | A file group exists only as the referent of ≥ 1 slot. Unfitted files absent. No empty `slots/`. |
| 1 — identity is a guarded name | File identity is `name`. `original_path` becomes a pure breadcrumb. The three `*_sha256` attrs collapse into one `file_content_hash`. The `archive_slot_key` / `history_key` pair collapses to one stored `handle`. |
| 2 — durability classes | `PlotConfig` gains a project-level home; session ephemera stay out. |
| 3 — a fit is unique by its complete input | Slots gain `handle`, `optimization_hash`, `input_files`, `model_structure`, `fit_settings` additions (`seed`, effective backend, `jac_fun` name). `observed_sha256` → `fit_view_sha256`. |
| 4 — mutable inputs snapshot per fit | `dark` / `calibration` move from the file group to each slot. File-level corrected `data` is dropped and reconstructed. `data_raw` added. Datasets gain compression. |
| 5 — provenance persisted even when not identity | Slots gain the attached model YAML snippets. |
| 6 — variants are viewed, not suppressed | Slots gain an optional user `label`. |

## Target layout

```
<archive>.fit.h5
├── metadata/                        # wire-format identity only
│     attrs:
│       format             : str     # "trspecfit-fit-archive"
│       schema_version     : str     # "7"
│       trspecfit_version  : str     # updated on every write
│       timestamp_created  : str     # ISO 8601 UTC, first write
│       timestamp_updated  : str     # ISO 8601 UTC, most recent write
└── project/                         # exactly one
    │   attrs:
    │     name         : str         # written once; checked on append
    │     plot_config  : str         # JSON; rewritten on every save
    ├── joint/                (opt)  # one child per joint optimization
    │   └── 000000/
    │       ├── metadata/
    │       │     optimization_hash : str      # 64 hex; shared with every projection
    │       │     input_files       : str      # JSON; scope == "project"
    │       │     model_structure   : str      # JSON; N per-file entries
    │       │     projections       : str      # JSON: records; see "projection records"
    │       │     label       (opt) : str      # the bundle's only label
    │       │     fit_alg, fit_settings, timestamp
    │       │     aic, bic, chi2_raw, chi2_red_raw   # whole-objective
    │       ├── params                         # combined; long format, incl. init_value
    │       ├── conf_ci             (opt)
    │       ├── correl              (opt)      # joint correlation matrix
    │       └── mcmc/               (opt)      # joint chain
    └── files/
        ├── 000000/
        │   ├── metadata/
        │   │     name              : str
        │   │     original_path     : str    # breadcrumb; never identity
        │   │     dim               : int64
        │   │     shape             : int64[ndim]
        │   │     file_content_hash : str
        │   ├── data_raw                     # immutable; source dtype
        │   ├── energy                       # source dtype
        │   ├── time                         # length 0 for 1D files
        │   ├── aux_axis            (opt)    # omitted iff File.aux_axis is None
        │   └── slots/
        │       ├── 000000/
        │       └── 000001/...
        └── 000001/...
```

There is deliberately **no** file-level `data`, `dark`, `calibration`, `e_lim`,
or `t_lim`.

### Slot group

```
project/files/000000/slots/000000/
├── metadata/
│     # --- identity ---
│     handle             : str      # 64 hex; sha256(optimization_hash + file_name)
│                                   # STORED, authoritative; abbreviated to 8 for
│                                   # display, prefix-matched on lookup
│     optimization_hash  : str      # 64 hex; shared by all joint siblings
│     input_files        : str      # JSON: [scope, [[name, version_stamp, selection], ...]]
│     model_structure    : str      # JSON: sorted [[file_name, per_file_structure], ...]
│                                   # per_file_structure = [[energy models],
│                                   #   [[target_par, [submodels], frequency], ...]]
│                                   # one entry for file scope, N for joint
│     fit_view_sha256    : str      # replaces observed_sha256
│     fit_type           : str
│     model_name         : str      # display; identity lives in model_structure
│     # --- provenance ---
│     fit_alg            : str
│     fit_settings       : str      # JSON; + seed, effective backend, jac_fun qualname
│     timestamp          : str
│     label              : str (opt) # user-set, post-hoc, mutable
│     joint_ref          : str (opt) # the joint record's optimization_hash, not a path
│                                   # mandatory iff input_files.scope == "project"
│     # --- noise (unchanged from schema 6) ---
│     noise_type, sigma_source, sigma_type, sigma_data, sigma_eff
│     # --- metrics (non-sbs only) ---
│     chi2_raw, chi2, r2                  # always
│     chi2_red_raw, chi2_red, aic, bic    # omitted for joint projections
├── dark                            # correction in force for this fit
├── calibration                     # correction in force for this fit
├── observed, fit                   # unchanged
├── fit_ini                  (opt)
├── components, component_names (opt)
├── params                          # unchanged per fit type
├── params_meta, params_stderr, params_init (opt; sbs)
├── metrics_per_slice        (opt; sbs)
├── conf_ci, correl          (opt)
├── mcmc/                    (opt)
└── model_yaml/              (opt)  # one dataset per snippet; see below
    ├── 000000                      # scalar vlen-utf8 = the snippet text
    │     attrs: role, name, source_file, target_par (opt), sequence_index (opt)
    └── 000001/...
```

## What carries over unchanged

Accepted as-is from [fit_archive_schema.md](fit_archive_schema.md) and **not**
restated here. The conversion must not alter them:

- **Conventions** — positional zero-padded six-digit group keys; identity in
  attrs never in path segments; vlen UTF-8 strings; `None`-handling rules (omit
  attr / `NaN` / `""` / omit group); source-dtype preservation for user arrays;
  int64 for positional attrs; tuple-valued attrs as 1D int64.
- **DataFrame encoding** — the all-numeric vs heterogeneous split, positional
  `c000000` field names, `columns` / `dtypes` attrs.
- **Slot datasets** — `params`, `params_meta`, `params_stderr`, `params_init`,
  `observed`, `fit`, `fit_ini`, `metrics_per_slice`, `conf_ci`, `correl`,
  `mcmc/`, `components`, `component_names`: layouts and dtypes unchanged.
- **Noise / σ fields** — unchanged and still slot-only.
- **Per-fit-type cheat sheet** — observed shapes, params layout, metrics
  location, `t_lim`-not-applied for SbS. **One exception:** the
  degrees-of-freedom metrics are omitted for joint projections (below).

Explicitly **not** carried over: the `archive_slot_key` / `history_key` pair and
its recompute-on-read rule. Both collapse into one stored `handle`.

## What changes

### `metadata/`

Reduced to wire-format identity; `project_name` moves to `project/`. New
`format` attr. `schema_version` = `"7"`; `SUPPORTED_READ_VERSIONS` = `("7",)`,
and every schema-2-through-6 fallback branch is deleted rather than left inert.

### `project/`

- `name` — written once; on append compared against the incoming project name,
  mismatch raises with a "choose a new path" message.
- `plot_config` — JSON encoding every `PlotConfig` field, rewritten on every
  save. Needs canonical serialization helpers that emit JSON-safe primitives,
  normalize the tuple-typed fields (`x_lim`, `y_lim`, `z_lim`, `panel_size`)
  back to tuples on read, round-trip the list-valued style fields and
  `data_slice` natively, and reject unsupported values loudly rather than
  degrade.
- `joint/` — one child per joint optimization, written whenever
  `Project.fit_2d` runs. Reserving the name without populating it would have
  bought nothing (HDF5 does not enforce layout reservations) and would have
  forced a second format break immediately after this one.

  **Schema 7 serializes a first-class joint result; it does not produce one.**
  Promoting the shared result from a transient on
  `Project._project_fit_result` to a captured object that survives the
  `fit_2d()` call is the upstream prerequisite (see "Prerequisites"). The raw
  material is already computed — that transient is a `FitOutput` carrying
  `par_ini`, `par_fin` (hence correlation, covariance, AIC/BIC, ndata),
  `conf_ci`, `emcee_fin`, and `emcee_ci` — but today only the per-file
  projections reach `_fit_history`. The fields below are what the archive
  stores *given* a settled record; the prerequisite branch decides what that
  record contains, and this layout follows it.

  Rules, per Principle 3:

  - **Ownership is two-level.** The joint record owns the combined parameter
    table, joint correlation/CI/MCMC, joint optimizer settings,
    whole-objective AIC/BIC, and the projection list. Projections own their
    per-file arrays and σ. Payloads are disjoint; a projection has no
    `correl` / `conf_ci` /
    `mcmc` to conflict over.
  - **Projection parameter tables are materialized views** of the combined
    result through the sharing map — validated against it, never independently
    overwritten.
  - **Mutation is one transaction over the bundle**: validate the joint record
    and every projection, then apply all or none. This is **logical
    atomicity** — it does not survive process interruption or an HDF5 error
    mid-write, and HDF5 offers no transactions.
  - **Every declared projection reference must resolve, or saving raises an
    integrity error.** Partial bundles are not representable — the
    comparability tuple is computed on read from the projections, so an
    unresolved reference would make it uncomputable.
  - **`joint_ref` is mandatory whenever `input_files.scope == "project"`**,
    including a degenerate one-file project fit. Invariant:
    `scope == "project"` ⟺ `joint_ref` present.
  - **`select=` expands silently** to the whole bundle; **`drop()` raises**,
    naming the joint handle and sibling count.

#### Cross-references are content-addressed

Every reference in schema 7 uses a **content hash, never a positional path**:
`joint_ref` holds the joint record's `optimization_hash`, and a projection
record identifies its slot by `handle`. Schema 6's `file_ref` stored an
archive-local path (`"files/000000"`); that form is retired.

Content addressing means references survive an archive being rewritten,
compacted, or merged, where positional keys would silently point at the wrong
group. The cost is that resolution needs a scan or a load-time index rather than
a direct path lookup — negligible at these sizes, and the writer already
enumerates groups to find insertion points.

#### Projection records

Sorting projections by file name would discard the association to optimizer
order, and the combined parameter names encode *position*, not name:
`proj_name = f"file{file_idx:02d}_{local_name}"`
([trspecfit.py:1202](../../src/trspecfit/trspecfit.py#L1202)), indexed by
`enumerate(self.files)`. Project-level shared parameters stay unprefixed
([:1156](../../src/trspecfit/trspecfit.py#L1156)). So the sharing map is **not**
recoverable from names alone, and without it the materialized-view invariant
cannot be validated.

Each projection is therefore a named record carrying its prefix:

```json
{"file_name": "A", "handle": "…", "parameter_prefix": "file00_"}
```

- A prefixed combined parameter belongs to exactly one projection:
  `file00_Gauss_01_A` → file `A`, local `Gauss_01_A`.
- An **unprefixed** combined parameter is project-shared and maps to that same
  local name in every projection.
- Static and file-varying parameters follow the same prefix rule.

Storing the prefix beats storing `file_index`: readers never reproduce the
`f"file{index:02d}_"` formatting rule, so a future change to it cannot
retroactively misparse existing archives. Records stay sorted by file name for
canonicality — the stored prefix, not the ordering, carries the association.

**Strip the declared prefix exactly once; never pattern-match `fileNN_`.** A
component legitimately named `file00` would otherwise be mangled: the combined
name `file00_file00_Gauss_01_A` must yield local `file00_Gauss_01_A`.

Writer and reader validate that prefixes are unique, that every prefixed
combined parameter matches exactly one declared projection, and that every
projection's local parameters appear in the combined table under its prefix —
the mapping must be total in both directions. That is what makes the
materialized-view invariant checkable rather than asserted.

### File group

| Change | Detail |
|---|---|
| Removed | `data` (reconstructed on demand), `dark`, `calibration` (now per-slot), `e_lim` / `t_lim` (no readers), the three `*_sha256` attrs |
| Added | `data_raw`; `file_content_hash` |
| Unchanged | `name`, `original_path`, `dim`, `shape`, `energy`, `time`, `aux_axis` |

`file_content_hash = sha256(dtype + shape + data_raw + energy + time + aux_axis)`.
When `aux_axis` is absent it contributes a fixed sentinel, so "no auxiliary
axis" is distinguishable from "a zero-length one".

**Write-side identity.** File identity within the archive is `name`. An incoming
file matching an existing group by name but differing in `file_content_hash` is
a different measurement under a reused name — **raise**, never fork a second
group. This catches renames and in-place `data_raw` mutation, which today
duplicate a group invisibly.

**Read-side matching** stays deliberately forgiving for aligning an archive
against a live `Project`: match on `name`, and treat a `file_content_hash`
difference as reportable staleness rather than a lookup failure. `original_path`
never participates.

### Slot group

New **identity** attrs (`handle`, `optimization_hash`, `input_files`,
`model_structure`) per Principle 3. `input_files` and `model_structure` are JSON
attrs in canonical order — attachments sorted, submodel tuples and the parameter
table in model order.

One new **comparability** attr, `fit_view_sha256`, replacing
`observed_sha256`. It is not part of identity — see execution step 4 for why it
is a sibling of `optimization_hash` rather than an input to it.

`fit_settings` gains `seed` (only when supplied), the **effective** evaluator
backend, and the `jac_fun` qualified name (only when an analytic Jacobian was
applied). `fit_alg_2` is recorded always but **hashed only when
`stages == 2`**.

New datasets: `dark` and `calibration` as captured for this fit, and the
`model_yaml/` group below.

#### `model_yaml/` encoding

A fit involves several YAML snippets — one energy model, one per dynamics
attachment (several if the attachment is a multi-cycle sequence), and one per
profile attachment — drawn from **more than one file**. So this is a group with
one child per snippet, following the same positional-key convention as `files/`
and `slots/`: identity in attrs, never in path segments.

Each child is a **scalar vlen-utf8 dataset** holding the snippet text, carrying
its own attrs:

| Attr | Type | Presence | Meaning |
|---|---|---|---|
| `role` | str | always | `"energy"` \| `"dynamics"` \| `"profile"` |
| `name` | str | always | the YAML top-level key |
| `source_file` | str | always | YAML filename this snippet came from |
| `target_par` | str | dynamics, profile | the parameter it attaches to |
| `sequence_index` | int64 | dynamics | position in the multi-cycle sequence; 0 is the global element |

Group-per-snippet rather than parallel arrays: the alternative — one vlen-str
dataset of texts plus five index-aligned attr arrays — has no structural
enforcement of alignment, so writing one and forgetting another produces a
silently mismatched record. Attaching each snippet's metadata to the snippet
makes misalignment unrepresentable.

**Text is ruamel round-trip output for that key's subtree**, not a verbatim byte
slice of the source file. Slicing a top-level key out of a YAML file by byte
range is fiddly; ruamel's round-trip mode preserves comments and ordering, which
is what makes the snippet readable provenance.

**These datasets are not compressed**, unlike the array datasets. HDF5 filters
require chunked storage and scalar datasets cannot be chunked; even for a 1D
vlen layout, compression would apply to the pointer array rather than the string
heap. Snippets are tens of lines, so this costs nothing — but the
"datasets are written with compression" convention has this exception, and
claiming otherwise would send someone chasing a filter that cannot be applied.

`model_yaml/` is omitted entirely for a model built programmatically with no
YAML source, following the omit-when-`None` convention.

**This supersedes `yaml_filename`, which is dropped.** That attr is a single
`str | None`, but `add_time_dependence(..., dynamics_yaml, ...)` takes a
*separate* path from the energy model's YAML
([trspecfit.py:3853](../../src/trspecfit/trspecfit.py#L3853)), so a fit spanning
`models.yaml` and `models_time.yaml` could only ever record one of them —
a pre-existing gap, not one this conversion introduces. Per-snippet
`source_file` records all of them. Consumer note: `_slot_title`
([fit_results.py:117](../../src/trspecfit/fit_results.py#L117)) uses
`yaml_filename` for plot titles and must read the energy snippet's `source_file`
instead.

`label` is a mutable attr — settable on an existing archive without rewriting
the slot.

**Joint projections omit the degrees-of-freedom metrics.** `chi2_red_raw`,
`chi2_red`, `aic`, and `bic` all divide by or penalize a parameter count, and a
joint fit's parameter count does not decompose per file — a shared parameter is
constrained by every participating file's data, so no share of it is
attributable to one. Those four attrs are therefore absent; `chi2_raw`, `chi2`,
and `r2` are always present because they depend only on residuals.

This completes a rule schema 6 already applies rather than adding a new one: the
same non-decomposability already means a joint projection has no `correl`, no
`conf_ci`, no `fit_ini`, and a `NaN` `stderr` column in `params`. The metrics
were the one field where the exception was missed.

The condition is `scope == "project"` in `input_files`, not `fit_type == "2d"` —
today `Project.fit_2d` is the only joint path, so every joint projection happens
to be 2d, but keying on scope means a future project-level SbS needs no change
here.

Consumer consequence: three of the four entries in `DEFAULT_METRICS_NO_SIGMA`
([fit_results.py:52](../../src/trspecfit/fit_results.py#L52)) are in the omitted
set, so `compare_models` must drop those columns when any matched slot lacks
them rather than surfacing a `KeyError` or a column of `None`.

### Conventions

Datasets are written with lossless compression (gzip + shuffle). Dtypes and
every hash stay valid; only file size changes.

## In-memory object model

### `SavedProject`

Gains `plot_config: PlotConfig`. Keeps `name`, `trspecfit_version`,
`schema_version`, `timestamp_created`, `timestamp_updated`, `files`.

### `SavedFile`

Drops `e_lim`, `t_lim`, `data`, and the three-sha `fingerprint` dict. Gains
`data_raw` and `file_content_hash`. Keeps `name`, `original_path`, `dim`,
`shape`, `energy`, `time`, `aux_axis`, `slots`.

All arrays are **copies, read-only** — the ownership boundary that makes this a
snapshot rather than a view (Principle 4).

### `SavedJointFit` (new)

`optimization_hash`, `input_files`, `model_structure`, `projections`
(ordered `(file_name, handle)` pairs), `params` (combined, long format),
`conf_ci`, `correl`, `mcmc`, `fit_alg`, `fit_settings`, `timestamp`, `label`,
and the whole-objective metrics. Frozen, arrays copied like every other record.

`SavedProject` gains `joint: tuple[SavedJointFit, ...]`.

### `SavedFitSlot`

Gains `handle`, `optimization_hash`, `input_files`, `model_structure`,
`fit_view_sha256`, `dark`, `calibration`, `model_yaml`, `label`. Drops
`observed_sha256`, `history_key`, the `file_fingerprint` dict, and
`yaml_filename` (superseded by per-snippet `source_file`). Gains `joint_ref`,
mandatory iff `scope == "project"`. Keeps everything else.

`model_yaml` is a tuple of records — `(role, name, source_file, target_par,
sequence_index, text)` — mirroring the group encoding above.

`dark`, `calibration`, and `observed` are copies, not references or views into
`File.data`.

### `FitResults`

1. **Hold `(SavedFile, SavedFitSlot)` pairs**, not a fingerprint dictionary.
   `_files_by_fp` ([fit_results.py:197](../../src/trspecfit/fit_results.py#L197))
   collapses byte-identical files; the parent link is already correct on disk
   and should be retained rather than re-derived from a hash.
2. **Carry the resolving `PlotConfig`.** `Project.results` passes the live
   `Project.plot_config`; `FitResults.load` passes the config decoded from
   `project/`. One rule: a `FitResults` renders with its project's config. Pass
   a `PlotConfig` and a project name, never a `Project` — `fit_results.py` is
   deliberately a leaf in the import graph
   ([fit_results.py:96-99](../../src/trspecfit/fit_results.py#L96)).
3. **Query layer** per Principle 6: slot handles with prefix matching, the
   variant table with constant columns suppressed, pairwise input/output diffs,
   `select=` on save and export, explicit pruning. Comparability grouping
   becomes `(file_name, fit_type)` + `fit_view_sha256` — `file_fingerprint`
   leaves the key.

   **σ does not enter the grouping key**, but it gates which metrics may be
   compared within a group. Three tiers:

   | Condition | Comparable |
   |---|---|
   | same `fit_view_sha256` | raw metrics — `chi2_raw`, `r2` |
   | same view **and** same `sigma_eff` | additionally `chi2`, `chi2_red` |
   | same view, differing `sigma_eff` | raw only; σ-scaled must be withheld |

   The failure this prevents is reachable precisely because σ is an attachment
   rather than a keyed input: fit M1 at σ=1, call `set_sigma(10)`, fit M2. The
   two are distinct slots sharing one fit view, and ranking them by `chi2`
   (100 vs 1.2) makes M2 look dramatically better when `chi2_raw` (100 vs 120)
   says it is worse. Putting σ *in* the grouping key would be the wrong fix —
   it would split the group and block the raw comparison, which is valid.
4. **Filtering stays projection-based.** `file="A"` matches
   `slot.file_name == "A"`, never a containment test over `input_files`.

### Capture

One path, not two: the immutable `SavedFile` payload is captured when a file
produces its **first** slot. Nothing is read from live state at save time.

## Live-session changes

- **`Project` holds a real `PlotConfig`** instead of ~35 flat attributes
  scraped by `PlotConfig.from_project`. Removes the field-walking, the
  tuple-coercion case, and the `x_label`/`e_label`, `dpi_plot`/`dpi_plt` alias
  map duplicated in
  [config/plot.py:216-220](../../src/trspecfit/config/plot.py#L216) and
  [trspecfit.py:798-802](../../src/trspecfit/trspecfit.py#L798). `project.yaml`
  keys stay unchanged — the YAML is a user artifact.
- **`File.plot_config` is deleted.** The 22 `src` read sites become
  `self.p.plot_config`; `Model.plot_config`
  ([mcp.py:280](../../src/trspecfit/mcp.py#L280)) resolves through
  `parent_file.p`.
- **[trspecfit.py:734](../../src/trspecfit/trspecfit.py#L734)** — the
  project-level 2D plot's `files_2d[0].plot_config` hack is deleted.
- **`export_fits`' per-file `plot_configs` dict**
  ([trspecfit.py:451-458](../../src/trspecfit/trspecfit.py#L451)) and
  `_resolve_plot_config`'s per-file lookup
  ([fit_io.py:2474](../../src/trspecfit/utils/fit_io.py#L2474)) collapse to one
  config.
- **A corrected-data reconstruction helper in `utils/`**, imported by both
  `fit_results` and `trspecfit` rather than owned by either. Needed for
  `full_range=True`, which shows corrected data outside the fit window where
  nothing is stored.
- **`select=` replaces `collapse=` / `which_one=`** on `save_fits` (default
  `"all"`) and `export_fits` (default `"latest"`), accepting `"all"`,
  `"latest"`, `"best"` + `by=`, a handle or prefix, or a label. `"latest"` and
  `"best"` resolve within each `(file, model, fit_type)` group.
- **`frequency` becomes persisted state.** It reaches the archive through
  `model_structure`; today it is stored nowhere.

## Prerequisites

Not part of this conversion, but upstream of it (Principle "Sequencing"):

1. **Guard `File.name`** on mutation, and **model-name uniqueness** within a
   `File` — `trspecfit.py:2111` appends unconditionally today.
2. **Demote the fingerprint to a version stamp**; make slot→file lookup a name
   lookup.

Both are pure live-object-model work with no schema change, so existing
archives keep reading throughout. The `PlotConfig` refactor is independent of
them and can proceed in parallel.

## Deferred / non-goals

- **Model rehydration.** `model_yaml` is provenance; it is not a complete
  record, since ordering, `frequency`, and profile attachment are programmatic.
- **`keep_history=True`.** Superseded — a complete input hash means distinct
  configurations no longer collide, so `select="all"` is the default and
  collapse is opt-in.
- **Recomputing the handle on read.** Reconsider post-1.0, once the hash inputs
  have stabilized.
- **The systemic array-mutation policy.** Independent; the archive's own
  ownership rules do not wait for it.
- **Per-file `PlotConfig` overrides.** Re-addable additively as a sparse
  override if heterogeneous-file projects ever make per-call overrides painful.

## Execution order

1. **Prerequisites** — the identity guards above.
2. **Live-session config refactor** — `Project.plot_config` as a real field;
   delete `File.plot_config` and update all read sites; the four per-file-config
   tests are rewritten against the project.
3. **`PlotConfig` (de)serialization** — canonical JSON helpers, round-trip
   tests first.
4. **Hash construction** — unit-tested independently of any I/O. Two
   independent families, not one chain:

   - **Identity**: `file_content_hash` → `file_version_stamp` → `input_files`,
     plus `model_structure`, the shared parameter metadata, the initial-state
     matrix (9-significant-digit quantization, `-0.0` normalized), and the
     conditional optimizer settings → `optimization_hash` → `handle`.
   - **Comparability**: `fit_view_sha256` over `observed` dtype/shape/values,
     the selected energy coordinates, the selected time coordinates where the
     file has a time axis, and `aux_axis` **whenever the file has one** —
     never conditioned on whether a model consumes it, or a profile and a
     non-profile model on identical observations would fall into different
     comparability groups.

   **The formulas in the principles are notation, not an implementation
   instruction.** `sha256(x + y + z)` means "hash a canonical **tagged**
   encoding of these fields" — named or length-prefixed records — never string
   concatenation, which would make `("ab", "c")` and `("a", "bc")` collide.
   Every tuple gets a stated ordering rule: `input_files` and joint
   `projections` sorted by file name, dynamics attachments sorted by target
   parameter, submodel tuples and the parameter table in model order,
   `model_structure`'s per-file entries sorted by file name.

   `fit_view_sha256` is **not** an input to `optimization_hash`. The view is
   already implied there by `(file_version_stamp, selection)`; the separate hash
   exists to verify that derivation against the arrays actually used, so feeding
   it back in would be circular and would destroy its value as a cross-check.
5. **Object model** — the `SavedProject` / `SavedFile` / `SavedFitSlot` field
   changes, with copy-and-freeze at capture.
6. **Writer** — `project/` group; nested `files/`; project-name check on append;
   same-name/different-content raise; new file and slot datasets; compression;
   the collision rules — four cases, not two:

   | Fitted params | Attachment state | Behavior |
   |---|---|---|
   | agree | absent in archive, present incoming | enrich freely |
   | agree | present in archive, absent incoming | no change — never delete |
   | agree | present in both | requires `overwrite=True` |
   | **differ** | any | hard conflict; requires `overwrite=True` |

   Only **absent → present** is free. "Richer" is not a criterion: comparing a
   5000-step chain against a 1000-step one at a better acceptance fraction has
   no well-defined answer, so any present → different-present is a replacement
   and needs opting in. Attachments merge individually — `conf_ci` may enrich in
   the same write where `mcmc/` is left untouched.
7. **Reader** — mirror of 6; `SUPPORTED_READ_VERSIONS = ("7",)`; delete every
   pre-7 fallback.
8. **Capture** — first-slot `SavedFile` capture; per-slot correction snapshots;
   and serialization of the first-class joint result delivered by the
   prerequisite branch, alongside the N projections `Project.fit_2d` already
   emits. This step *consumes* that record — it does not implement it. If the
   prerequisite lands with a different record shape than the layout above
   assumes, the layout follows the record, not the reverse.
9. **`FitResults` query layer** — handles, variant table, diffs, `select=`,
   pruning, the regrouped comparability check. Two behaviors that are easy to
   omit and produce silently wrong output:

   - **σ-scaled metrics are withheld when `sigma_eff` is inconsistent** across
     the compared group. Follow the existing precedent in
     `_resolve_metric_keys`: for the dynamic defaults, drop the columns; for an
     explicit `metrics=[...]` request naming `chi2` / `chi2_red`, raise with the
     conflicting σ values in the message. For `select="best", by="chi2_red"`,
     **raise** — a bad ranking there picks a winner and discards or omits the
     loser, which is worse than a missing column.
   - **Joint fits diff at the bundle level.** Diff two joint fits by their
     joint records, never by a pair of projections: a projection's parameter
     table is a materialized view of the combined result, so diffing
     projections alone shows a shadow of the real difference. Because the joint
     record stores the combined table, bundle diffs are complete — a difference
     in shared parameter state is visible rather than merely implied by
     differing `optimization_hash` values.
   - **Joint comparability uses a canonical sorted tuple of
     `(file_name, fit_view_sha256)` pairs**, never a set of view hashes. A set
     discards association and multiplicity: two byte-identical files under
     different names hash alike, so a set would collapse them and make a
     two-file joint fit look like a one-file one. Computed on read from the
     projections — no stored field.
10. **Tests** — see below.
11. **Docs** — rewrite [fit_archive_schema.md](fit_archive_schema.md) as the
    schema-7 spec; update [repo_architecture.md](repo_architecture.md),
    `llms.txt`, `AGENTS.md`, `CLAUDE.md`; CHANGELOG covering the break, the
    `File.plot_config` removal, and the `select=` change; move this plan to
    `docs/design/archive/`.
12. **Verify** — `pytest -q`, Ruff, mypy, pyright, and a whole-repo grep for
    `plot_config`, `observed_sha256`, `history_key`, `e_lim`/`t_lim` on
    `SavedFile`, and stale schema-version references.

## Test coverage

Beyond round-tripping every field at schema 7:

- **Identity** — a `vary` flip, a bound change, a fit-limit change, and a
  correction each mint a distinct slot; an identical re-run does not. Renaming
  an energy model or dynamics submodel mints a slot; renaming a profile model
  propagates through the parameter table.
- **Handles** — stored at full 64-hex width; an unambiguous prefix resolves;
  an ambiguous prefix raises rather than picking; a prefix matching nothing
  raises.
- **Hash completeness** — two models differing only in component *order* hash
  apart; two dynamics attachments differing only in `frequency` hash apart; two
  files differing only in `aux_axis` hash apart; `fit_alg_2` differing under
  `stages == 1` does **not** hash apart.
- **Collisions** — one test per row of the writer's collision table, since the
  distinctions are exactly what a two-case implementation would lose:
  - saving a fit, then running MCMC and re-saving, attaches the chain with **no**
    `overwrite=`;
  - re-running a **shorter** MCMC chain over a longer archived one raises
    without `overwrite=True`, and the archived chain is intact afterwards;
  - re-saving a fit that has **no** MCMC does not delete an archived chain;
  - diverging parameters raise, and `overwrite=True` resolves it **without a
    refit** — assert the MCMC attachment survives the resolution, proving
    nothing was recomputed;
  - in-session collapse applies the same rule as the archive boundary.
- **Comparability** — a profile and a non-profile model on identical
  observations share a `fit_view_sha256` and compare; different fit limits do
  not; joint projections expose only the residual-based metrics.
- **σ tiers** — two slots sharing a view but differing in `sigma_eff` still
  compare on `chi2_raw` / `r2`; the σ-scaled columns are dropped from the
  dynamic defaults, an explicit `metrics=["chi2_red"]` raises naming both σ
  values, and `select="best", by="chi2_red"` raises rather than ranking. Build
  the fixture so the σ-scaled ranking is the *reverse* of the raw one, or the
  test passes for the wrong reason.
- **Joint bundles** — a joint fit writes one `joint/` record plus N
  projections, every projection carrying `joint_ref`; the degenerate one-file
  project fit also gets one. `select=` on a single projection expands to the
  bundle; `drop()` on one raises. A joint record whose projection reference
  does not resolve fails the integrity check at save. Joint covariance and a
  joint MCMC chain round-trip. Two joint fits differing only in shared
  parameter state show that difference in a bundle-level diff.
- **Joint comparability** — a two-file joint fit over byte-identical files
  under different names produces a two-entry comparability tuple, not one.
  This is the multiplicity case a set would collapse.
- **Sharing map** — a joint fit's projection records recover local parameter
  names from combined ones; a component legitimately named `file00` round-trips
  (strip the declared prefix once, never pattern-match); the mapping validates
  total in both directions; a projection whose parameter table disagrees with
  the combined result fails validation.
- **Per-file model structure** — a joint fit where A uses `frequency=0.25` and
  B uses `0.5` under one model name hashes differently from one where both use
  `0.25`. This is the case a single global `model_structure` would miss.
- **Hash framing** — `model_structure` for `['IRF', 'MonoExp_Neg']` differs
  from a single model literally named `IRF_MonoExp_Neg`; more generally,
  moving a character across a field boundary must change the hash. Guards the
  tagged-encoding requirement against a regression to concatenation.
- **Project scope** — appending a differently-named project raises; a file with
  no slots is absent from the archive; a same-name/different-content file
  raises.
- **Config** — round-trip; render-time resolution identical between
  `Project.results` and `FitResults.load`; a live restyle changes subsequent
  plots; an explicit `config=` overrides.
- **Ownership** — mutating a live `File` array after capture does not change any
  archived or in-memory slot.
- **Rejection** — a schema-6 archive is refused with a clear message.

## Interaction with the current `PLAN.md`

`PLAN.md` carries a banner marking its `PlotConfig` sections superseded. Full
mapping:

| `PLAN.md` step | Effect |
|---|---|
| 1 (regression tests) | Per-file-config and frozen-config tests are obsolete; the live-mutation and byte-identical-file tests survive |
| 2 (per-file `SavedFile` capture) | Retained; the `PlotConfig` half is dropped, and capture has one path rather than two |
| 3 (slot-to-file association) | Retained, subsumed by the `FitResults` change above |
| 4, 5, 6, 8 (renderer consolidation) | Unaffected — orthogonal to the schema |
| 7 (persist and resolve `PlotConfig`) | Dissolved; replaced by a project-owned config resolved at render time |
| 9, 10, 11 | Retained, with the schema-7 doc and test scope folded in |
