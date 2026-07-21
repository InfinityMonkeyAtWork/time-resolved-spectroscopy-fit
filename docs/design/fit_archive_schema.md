# Fit-archive HDF5 schema (schema_version 5)

On-disk layout for the fit-results archive written by `Project.save_fits()`
and read by `FitResults.load()` / `Project.load_fits()`. The object model
([utils/fit_io.py](../../src/trspecfit/utils/fit_io.py)) is the source of
truth; this document specifies the 1:1 mapping to HDF5 so the writer and
reader agree on dtypes, attr keys, and None-handling.

For the design rationale (why per-slot `observed`, why two identity keys,
why HDF5 instead of pickle, etc.), see the archived design plan,
[Fit Results Save/Load](archive/fit_results_save_load_plan.md).
This file is the wire format.

## Conventions

- **Group-path components are positional, zero-padded six-digit keys**
  (`000000`, `000001`, ...). HDF5 path components forbid `/`, and
  user-meaningful names (`File.name`, `model_name`, etc.) can contain
  arbitrary characters. Identity lives in attrs, never in path segments.
- **Strings in attrs and string-typed dataset fields use
  `h5py.string_dtype(encoding="utf-8")`** (variable-length UTF-8). Fixed-
  length string types are not used; readers must not assume any length.
- **`None` handling**:
  - For optional **strings** (e.g. `yaml_filename`): omit the attr
    entirely. The reader treats absence as `None`.
  - For optional **integer-pair lists** (`e_lim`, `t_lim`): omit the attr
    entirely (do not write a sentinel array).
  - For optional **floats** like `stderr` inside structured arrays: write
    `np.nan`. The reader maps NaN back to `None` only for fields where the
    object model permits `None` (`stderr`); other float fields are kept as
    floats.
  - For optional **strings** inside structured arrays (long-form params
    `expr`): write `""`. The reader maps `""` back to `None` on
    columns where the object model permits `None` (lmfit's `expr=None`).
  - These slot-specific ``↔`` mappings are applied by the slot reader,
    not the generic DataFrame decoder. ``conf_ci``, ``mcmc/flatchain``,
    ``mcmc/ci``, and sbs ``params`` carry no None semantics; their
    literal `""` / `NaN` values are data.
  - For optional **groups/datasets** (`conf_ci`, `mcmc/`): omit the
    group/dataset. The reader treats absence as `None`.
- **Float dtype**: structured-array float fields and metric attrs are
  `float64`. The four user-array datasets — file `data` / `energy` /
  `time` and slot `observed` / `fit` — are written in **the source
  array's native dtype** (typically `float64` or `float32`). This
  preserves byte-for-byte equivalence with the inputs, which the
  fingerprints (`data_sha256`, `energy_sha256`, `time_sha256`) and
  `observed_sha256` rely on. The reader does not re-cast.
- **Integer dtype**: positional/index attrs and shape are `int64`.
- **Bool**: `vary` in params is HDF5 `bool` (numpy `?`).
- **Tuple-valued attrs** (`shape`): stored as 1D `int64` arrays.

### DataFrame encoding

All persisted `pd.DataFrame` payloads (slot `params`, `conf_ci`,
`mcmc/flatchain`, `mcmc/ci`) follow one uniform rule so the writer/reader
has a single code path and column labels never collide with HDF5 field-
name restrictions:

1. **All-numeric DataFrames** (homogeneous `float64` columns, e.g. sbs
   `params`, `flatchain`): 2D `float64` dataset of shape
   `(n_rows, n_cols)`, plus attr `columns` — a 1D vlen-utf8 array of
   length `n_cols` listing the column labels in axis-1 order.

2. **Heterogeneous-dtype DataFrames** (e.g. baseline/spectrum/2d
   `params`, `conf_ci`, `mcmc/ci`): 1D structured dataset of shape
   `(n_rows,)` with fields named positionally `c000000`, `c000001`, ...
   (zero-padded six-digit, matching the group-key convention). Each
   field's dtype is chosen per-column from `{vlen str, float64, bool}`.
   Attr `columns` (1D vlen-utf8 array, length = field count) gives the
   actual column labels in field order. Attr `dtypes` (1D vlen-utf8
   array, same length) gives a short type tag per column from
   `{"str", "float64", "bool"}` so the reader can rebuild the DataFrame
   without inferring dtypes back.

This convention isolates HDF5 from arbitrary user-facing labels (e.g.
sigma columns like `"+1"`, `"best fit"`, or future column-renames in
`par_to_df`) without giving up structured-array benefits for mixed
dtypes.

## Top-level layout

```
<archive>.fit.h5
├── metadata                                # group; identity attrs only
│   attrs:
│     trspecfit_version  : str              # e.g. "0.4.0"; updated on every write
│     project_name       : str              # Project.name; set on first write
│     timestamp_created  : str              # ISO 8601 UTC, first write
│     timestamp_updated  : str              # ISO 8601 UTC, most recent write
│     schema_version     : str              # "5"; bump on incompatible change
└── files/                                  # group; one subgroup per file
    ├── 000000/                             # SavedFile (see "File group")
    └── 000001/...
```

`save_fits` is slot-scoped (a single archive may be written multiple
times as new fits accumulate), so the archive carries both
`timestamp_created` (set once when the file is first opened with mode
`"w"`) and `timestamp_updated` (rewritten on every save). The writer
must not recreate the archive on subsequent saves unless the caller
explicitly asks for that; the canonical way to start fresh is to choose
a new path.

`schema_version` is currently `"5"`. Version history:

- `"1"` → `"2"`: the σ-calibrated chi-square columns and per-slot sigma
  metadata changed the stored fields — a clean break, so schema-1 archives
  can no longer be read.
- `"2"` → `"3"` (2026-07): **additive** — slot `correl` dataset, mcmc
  `acceptance_fraction` dataset, the `fit_settings` provenance attr, and
  the sbs-only `params_meta` / `params_stderr` datasets. The reader
  accepts both `"2"` and `"3"` (`SUPPORTED_READ_VERSIONS` in
  `utils/fit_io.py`); schema-2 archives load with the new fields as
  `None`. The writer still refuses to append to an archive whose version
  differs from its own — re-save to a new path to migrate.
- `"3"` → `"4"` (2026-07): **additive** — slot `components` and
  `component_names` datasets for 1D fit types (baseline, spectrum, sbs).
  Never present for `fit_type == "2d"` — there is no per-component
  concept there. The reader accepts `"2"`, `"3"`, and `"4"`
  (`SUPPORTED_READ_VERSIONS` in `utils/fit_io.py`); schema-2/3 archives
  load with `components` / `component_names` as `None`, and
  `FitResults.plot_fit` falls back to a sum-only rendering in that case.
  The writer still refuses to append to an archive whose version differs
  from its own.
- `"4"` → `"5"` (2026-07): **additive** — per-file (not per-slot) optional
  `aux_axis` dataset (`File.aux_axis`, the auxiliary physical axis used by
  `par_profile`-attached models, e.g. depth). Sits alongside `data` /
  `energy` / `time` in the file group, not under `slots/`. The reader
  accepts `"2"` through `"5"` (`SUPPORTED_READ_VERSIONS` in
  `utils/fit_io.py`); pre-5 archives and files with no auxiliary axis both
  load with `aux_axis` as `None` on `SavedFile`. The writer still refuses
  to append to an archive whose version differs from its own.

Future incompatible changes (e.g. project-scoped joint-result slots or
`keep_history=True` full-log save — both deferred, see "What's *not* in
v1") bump it again. The reader rejects archives with a `schema_version` it
does not recognize. (This wire-format number is independent of the
feature-scope "v1" used elsewhere in this doc.)

## File group

```
files/000000/
├── metadata                                # group, no datasets; carries identity attrs
│   attrs:
│     name           : str                  # File.name
│     original_path  : str                  # absolute path of source file at save time
│     dim            : int64                # 1 or 2
│     shape          : int64[ndim]          # data.shape as 1D array
│     data_sha256    : str                  # 64 hex chars
│     energy_sha256  : str                  # 64 hex chars
│     time_sha256    : str                  # 64 hex chars; "" for 1D files
│     e_lim          : int64[2]   (opt)     # [start, stop) index slice; omit if None
│     t_lim          : int64[2]   (opt)     # [start, stop) index slice; omit if None
├── energy                                  # 1D dataset; preserves source dtype
├── time                                    # 1D dataset; length 0 if 1D file; preserves source dtype
├── data                                    # 1D (1D file) or 2D (n_t, n_e) dataset; preserves source dtype
├── aux_axis                         (opt)  # 1D dataset; preserves source dtype; schema ≥ 5
└── slots/
    ├── 000000/                             # SavedFitSlot (see "Slot group")
    └── 000001/...
```

Notes:

- The full data + axes are duplicated into the archive deliberately
  (decision in the archived design plan — "Self-contained archive"). On load, the reader
  hands these back via `SavedFile`; the live `Project` is not mutated.
- `data_sha256`, `energy_sha256`, `time_sha256` together with `shape`
  form the `file_fingerprint` used to match an archive's file to a
  `Project.files[*]` (or to another archive). See
  `compute_file_fingerprint` in `utils/fit_io.py`. `aux_axis` is not part
  of the fingerprint — file identity stays `data`/`energy`/`time`-based.
- `aux_axis` is omitted entirely when `File.aux_axis is None` (most
  files — only `par_profile`-attached models use it), following the
  same omit-when-`None` rule as the optional slot datasets, not the
  "empty array" convention used for `time` on 1D files.

### Identity collisions

Two distinct rules apply, in two different directions:

- **Archive uniqueness (write side).** A file group's effective identity
  is `(file_fingerprint, name, original_path)`. Two source files with
  byte-identical `data` / `energy` / `time` but different `name` or
  `original_path` are stored in **separate** file groups. Files agreeing
  on all three are treated as the same file (one group, slots merge).
  This means the writer's "find existing file group" lookup
  (`_find_file_by_fingerprint`) must compare `name` / `original_path`
  in addition to fingerprint when more than one candidate matches.

- **Live-Project matching (read side).** When a `FitResults` archive is
  loaded and the caller wants to align archive files with
  `Project.files[*]`, fingerprint is the primary key, and `name` /
  `original_path` are tie-breakers if multiple candidates match. The
  loader does not require an exact `original_path` match — that path is
  baked at save time and may not exist on the loading machine.

The asymmetry is deliberate: at write time we want strict separation of
intentionally-distinct files; at read time we want forgiving matching
that survives copying the archive between machines.

## Slot group

```
files/000000/slots/000000/
├── metadata                                # group; identity + provenance + (non-sbs) metrics in attrs
│   attrs:
│     # --- identity ---
│     file_ref          : str               # "files/000000" (archive-local)
│     model_name        : str               # SavedFitSlot.model_name
│     fit_type          : str               # "baseline" | "spectrum" | "sbs" | "2d"
│     selection_json    : str               # SavedFitSlot.selection_json
│     archive_slot_key  : str               # sha256(file_ref|model_name|fit_type|selection_json)
│     history_key       : str               # in-memory key from save time; non-authoritative
│     observed_sha256   : str               # 64 hex chars
│     # --- provenance ---
│     fit_alg           : str               # e.g. "leastsq", "Nelder"
│     yaml_filename     : str        (opt)  # human breadcrumb; omit if None
│     fit_settings      : str        (opt)  # JSON dict; see "fit_settings attr"; schema ≥ 3
│     timestamp         : str               # ISO 8601 UTC, slot creation time
│     # --- metrics (baseline / spectrum / 2d only) ---
│     chi2_raw          : float64   (cond)
│     chi2_red_raw      : float64   (cond)
│     chi2              : float64   (cond)
│     chi2_red          : float64   (cond)
│     r2                : float64   (cond)
│     aic               : float64   (cond)
│     bic               : float64   (cond)
├── params                                  # see "params dataset" below; layout depends on fit_type
├── params_meta                      (opt)  # heterogeneous-DataFrame dataset; sbs only; schema ≥ 3
├── params_stderr                    (opt)  # all-numeric DataFrame dataset; sbs only; schema ≥ 3
├── observed                                # 1D or 2D dataset; preserves source dtype
├── fit                                     # 1D or 2D dataset; preserves source dtype; observed.shape == fit.shape
├── metrics_per_slice                (opt)  # 1D structured dataset; sbs only
├── conf_ci                          (opt)  # heterogeneous-DataFrame dataset; see "conf_ci dataset"
├── correl                           (opt)  # all-numeric DataFrame dataset; see "correl dataset"
├── mcmc/                            (opt)  # see "mcmc group"
├── components                       (opt)  # 1D fit types only; see "components dataset"; schema ≥ 4
└── component_names                  (opt)  # present iff components is; see "components dataset"; schema ≥ 4
```

`(cond)` = present iff `fit_type != "sbs"`. SbS metrics live in the
`metrics_per_slice` dataset because they are per-slice arrays, not
scalars.

`(opt)` = present iff the corresponding `SavedFitSlot` field is non-`None`
(`conf_ci`, `correl`, `mcmc`, `components`, `component_names`) or
applicable to the fit type (`metrics_per_slice` is sbs-only; `components`
/ `component_names` are never present for `fit_type == "2d"`).

### `archive_slot_key` vs `history_key`

The authoritative on-disk slot key is `archive_slot_key`, computed at
save time once the file's archive position is known:

```
archive_slot_key = sha256(file_ref | model_name | fit_type | selection_json)
```

Both keys exist for the same logical purpose (uniquely identify a slot);
they use different file-identity tokens because in-memory and on-disk
identity primitives differ (multi-sha fingerprint vs archive-local
positional path). `archive_slot_key` is what the writer's slot-scoped
overwrite check (`_find_slot_by_archive_key`) compares against.

`history_key` is also persisted as a non-authoritative attr (a debugging
aid for archive inspection and round-trip tests), but the reader
**recomputes** it from
`(file_fingerprint, model_name, fit_type, selection_json)` and uses the
recomputed value for the `SavedFitSlot`. The on-disk value is ignored
on read; it exists only so an external inspector (e.g. a notebook
poking at the HDF5 directly) can correlate slots to in-session history
without redoing the hash.

## `params` dataset

Two distinct shapes depending on `fit_type`, both following the
DataFrame-encoding rule from "Conventions".

### baseline / spectrum / 2d — long format (one row per parameter)

Heterogeneous-dtype DataFrame:

```
params : 1D structured dataset, shape (n_par,)
  fields (positional, in column order):
    c000000 : vlen str    # column "name"        (parameter name, e.g. "GLP_01_A")
    c000001 : float64     # column "value"
    c000002 : float64     # column "stderr"      (NaN ↔ lmfit returned None)
    c000003 : float64     # column "init_value"
    c000004 : float64     # column "min"         (-inf permitted)
    c000005 : float64     # column "max"         (+inf permitted)
    c000006 : bool        # column "vary"
    c000007 : vlen str    # column "expr"        ("" ↔ None)
  attrs:
    columns : vlen str[8] = ["name","value","stderr","init_value","min","max","vary","expr"]
    dtypes  : vlen str[8] = ["str","float64","float64","float64","float64","float64","bool","str"]
```

Mirrors the DataFrame returned by `par_to_df(..., col_type="min")` in
`utils/lmfit.py`. `stderr` is the only float column that legitimately
holds `NaN`-as-`None` — the others must always have a real value.
`min`/`max` may carry IEEE `-inf`/`+inf` (unbounded parameters); those
are written verbatim.

### sbs — wide format (one row per slice, one column per parameter)

All-numeric DataFrame:

```
params : 2D float64 dataset, shape (n_slices, n_par)
  attrs:
    columns : vlen str[n_par]   # parameter names; axis-1 order
```

Stores optimized values only. Mirrors `list_of_par_to_df(results)` in
`utils/lmfit.py`. The slice-invariant metadata and the per-slice stderr
live in the sibling `params_meta` / `params_stderr` datasets (schema ≥ 3)
— do not redefine `params`.

## `params_meta` dataset (sbs only, optional; schema ≥ 3)

Shared per-parameter metadata — exactly the columns that are
slice-invariant by construction (one model, one vary set for every
slice; only *values* differ per slice):

```
params_meta : 1D structured dataset, shape (n_par,)
  fields (positional, in column order):
    c000000 : vlen str    # column "name"
    c000001 : bool        # column "vary"
    c000002 : float64     # column "min"   (-inf permitted)
    c000003 : float64     # column "max"   (+inf permitted)
    c000004 : vlen str    # column "expr"  ("" ↔ None)
  attrs:
    columns : vlen str[5] = ["name","vary","min","max","expr"]
    dtypes  : vlen str[5] = ["str","bool","float64","float64","str"]
```

Captured from the slice-0 result params; rows are column-aligned with the
wide `params` frame. Deliberately excludes `value` / `stderr` /
`init_value`, which are per-slice (`init_value` diverges under
`seed_adapt`). The runtime state is the source — not the model YAML,
which the fit may have diverged from (e.g. `seed_source="baseline"`).

## `params_stderr` dataset (sbs only, optional; schema ≥ 3)

Per-slice parameter standard errors, mirroring the wide `params` layout:

```
params_stderr : 2D float64 dataset, shape (n_slices, n_par)
  attrs:
    columns : vlen str[n_par]   # parameter names; axis-1 order
```

`NaN` where the optimizer reported no stderr for that slice; the NaN is
data (no None mapping). Mirrors `list_of_par_stderr_to_df(results)` in
`utils/lmfit.py`.

## `fit_settings` attr (optional; schema ≥ 3)

JSON-encoded dict on the slot `metadata` group recording the optimizer
configuration that can influence the result:

- all fit types: `stages`, `fit_alg_1`, `fit_alg_2`, `try_ci`;
- sbs: `seed_source`, `seed_adapt`, `seed_values` (JSON `null` is
  meaningful — "no adaptation" is provenance too);
- when MCMC was enabled: an `mc` sub-dict (`use_mc`, `steps`, `nwalkers`,
  `burn`, `thin`, `ntemps`, `is_weighted`, `sigma_ini/min/max`).

Execution details that cannot change the result (SbS / emcee worker
counts; serial ≡ parallel dispatch is pinned by test) are deliberately
excluded. `fit_settings` is not part of `history_key` — a refit with
different settings is still a refit of the same (file, model, fit_type,
selection). Built by `build_fit_settings` in `utils/fit_io.py`.

## `metrics_per_slice` dataset (sbs only)

```
metrics_per_slice : 1D structured dataset, shape (n_slices,)
  dtype:
    chi2_raw     : float64
    chi2_red_raw : float64
    chi2         : float64
    chi2_red     : float64
    r2           : float64
    aic          : float64
    bic          : float64
```

Row order follows the time-slice order in `observed` axis 0. The reader
reconstructs `SavedFitSlot.metrics` as `{name: column_array}` for sbs.

## `conf_ci` dataset (optional)

Heterogeneous-dtype DataFrame (one string column for the parameter
name, the rest float):

```
conf_ci : 1D structured dataset, shape (n_par,)
  fields (positional, in column order):
    c000000 : vlen str         # column "parameter" (or whatever par_to_df produced)
    c000001 : float64          # first sigma column, e.g. "-3"
    c000002 : float64          # next, e.g. "-2"
    ...
    c00000K : float64          # last, e.g. "+3"
  attrs:
    columns : vlen str[K+1]    # actual column labels (e.g. ["parameter","-3",...,"+3"])
    dtypes  : vlen str[K+1]    # ["str","float64","float64",...,"float64"]
```

Sigma labels come from `conf_interval_to_df` in `utils/lmfit.py`
(typically `["-3", "-2", "-1", "best fit", "+1", "+2", "+3"]`). The
positional fields insulate HDF5 from arbitrary user-facing labels; the
`columns` attr restores them on read. Omitted entirely if
`SavedFitSlot.conf_ci is None`.

## `correl` dataset (optional; schema ≥ 3)

All-numeric DataFrame — the varying-parameter correlation matrix built by
`correl_to_df` in `utils/lmfit.py`:

```
correl : 2D float64 dataset, shape (n_vary, n_vary)
  attrs:
    columns : vlen str[n_vary]   # varying parameter names; axis-1 order
```

The matrix is square with `index == columns`, so only the column labels
are stored; the reader restores the index from the `columns` attr.
Omitted entirely if `SavedFitSlot.correl is None` — which is the case
when the optimizer reported no covariance (e.g. Nelder without
numdifftools) and for project-level joint fits (joint covariance does not
decompose per file). For SbS the matrix is slice 0's, mirroring
`conf_ci` / `mcmc`.

## `mcmc/` group (optional)

```
mcmc/
├── flatchain                               # all-numeric DataFrame
│   2D float64 dataset, shape (n_samples, n_par)
│   attrs:
│     columns : vlen str[n_par]             # parameter labels; axis-1 order
├── ci                                (opt) # heterogeneous-dtype DataFrame
│   1D structured dataset, shape (n_par,)
│   field/attr layout identical to conf_ci above
├── acceptance_fraction               (opt) # 1D float64 dataset, shape (n_walkers,); schema ≥ 3
└── attrs:
      lnsigma : float64                     # __lnsigma point estimate
```

If `SavedFitSlot.mcmc is None`, the entire `mcmc/` group is omitted.
Within the group:

- `flatchain` is required when `mcmc/` is present, but may be empty if
  emcee returned an empty chain.
- `ci` is optional (emcee CI may not have been computed).
- `lnsigma` is required when `mcmc/` is present.
- `acceptance_fraction` is optional (absent in schema-2 archives and when
  emcee did not expose it); the reader maps absence to `None` in the
  payload dict.

## `components` / `component_names` (optional; schema ≥ 4)

Per-component fit curves for 1D fit types (baseline, spectrum, sbs),
evaluated at final params on the same grid as `fit`. Never present for
`fit_type == "2d"` — there is no per-component concept there.

```
components : ndarray (preserves source dtype)
  baseline / spectrum : shape (n_components, n_e_view)
  sbs                 : shape (n_slices, n_components, n_e_view)
component_names : 1D vlen-utf8 dataset, shape (n_components,)
  component labels; order matches components' component axis
```

Both fields are omitted together — `components` is `None` on the object
model iff `component_names` is. `component_names` is captured directly
from `[comp.name for comp in model.components]` at fit time rather than
re-derived from `params.name`: a static (`dim == 1`) attached
`par_profile` splices a nested model's parameters into its host
component's name block without adding a distinct `model.components`
entry, which would break any prefix-based re-derivation from parameter
names. Summing `components` along its component axis reconstructs `fit`
exactly (verified by `_assert_slot_round_tripped` in
`tests/test_fit_archive_roundtrip.py`).

Absent in schema-2/3 archives; the reader maps absence to `None` for
both fields, and `FitResults.plot_fit` falls back to the pre-schema-4
sum-only 1D rendering when `components is None`.

## Reader → object-model mapping

Per slot, the reader produces a `SavedFitSlot` with:

| `SavedFitSlot` field | Source                                                         |
|----------------------|----------------------------------------------------------------|
| `file_fingerprint`   | parent file group's `metadata` attrs                           |
| `file_name`          | parent file group's `metadata.name` attr                       |
| `model_name`         | slot `metadata.model_name` attr                                |
| `fit_type`           | slot `metadata.fit_type` attr                                  |
| `selection`          | `json.loads(metadata.selection_json)`                          |
| `selection_json`     | slot `metadata.selection_json` attr                            |
| `observed_sha256`    | slot `metadata.observed_sha256` attr                           |
| `history_key`        | recomputed from `file_fingerprint + model_name + fit_type + selection_json` |
| `params`             | `params` dataset (+ its `columns` attr) → DataFrame            |
| `params_meta`        | `params_meta` dataset → DataFrame, or `None` if absent         |
| `params_stderr`      | `params_stderr` dataset → DataFrame, or `None` if absent       |
| `fit_settings`       | `metadata.fit_settings` attr (JSON) → dict, or `None` if absent |
| `metrics`            | scalar attrs (non-sbs) or `metrics_per_slice` (sbs) → dict     |
| `observed`           | `observed` dataset                                             |
| `fit`                | `fit` dataset                                                  |
| `fit_alg`            | slot `metadata.fit_alg` attr                                   |
| `yaml_filename`      | slot `metadata.yaml_filename` attr (None if absent)            |
| `timestamp`          | slot `metadata.timestamp` attr                                 |
| `conf_ci`            | `conf_ci` dataset → DataFrame, or `None` if absent             |
| `correl`             | `correl` dataset → DataFrame (index restored from `columns`), or `None` if absent |
| `mcmc`               | `mcmc/` group → dict, or `None` if absent                      |
| `components`         | `components` dataset → ndarray, or `None` if absent            |
| `component_names`    | `component_names` dataset → list of str, or `None` if absent   |

`history_key` is persisted as a non-authoritative attr but recomputed
by the reader (see "`archive_slot_key` vs `history_key`"). The on-disk
value is for debugging and external inspection only; the in-memory key
on the returned `SavedFitSlot` always comes from the live recompute.

## Per-fit-type cheat sheet

| fit_type   | `observed.shape`        | `params` layout                      | metrics location          | `components.shape`                    | sbs-only datasets     | t_lim applied |
|------------|-------------------------|--------------------------------------|---------------------------|----------------------------------------|-----------------------|---------------|
| baseline   | `(n_e_view,)`           | structured (long, named columns)     | scalar attrs              | `(n_components, n_e_view)`             | —                     | n/a           |
| spectrum   | `(n_e_view,)`           | structured (long, named columns)     | scalar attrs              | `(n_components, n_e_view)`             | —                     | n/a           |
| sbs        | `(n_t_full, n_e_view)`  | 2D float64 + `columns` attr (wide)   | `metrics_per_slice`       | `(n_t_full, n_components, n_e_view)`   | `metrics_per_slice`   | **no**        |
| 2d         | `(n_t_view, n_e_view)`  | structured (long, named columns)     | scalar attrs              | always `None`                          | —                     | yes           |

`n_e_view` denotes the energy axis cropped by `e_lim`; `n_t_view`
denotes the time axis cropped by `t_lim`. `n_t_full` is the file's full
time-axis length: `fit_slice_by_slice` iterates every slice in
`File.data` regardless of `t_lim`, so `selection.t_lim` is always
`None` for sbs slots ([trspecfit.py:2987](../../src/trspecfit/trspecfit.py#L2987)).
`spectrum` and `baseline` reduce time via `time_point` / `time_range`
or `base_t_ind`, captured separately in `selection`.

Project-side: `Project.fit_2d()` produces ordinary `fit_type="2d"`
slots, one per file ([trspecfit.py:1004-1009](../../src/trspecfit/trspecfit.py#L1004-L1009)).
The archive does not distinguish them from slots produced by
`File.fit_2d()`.

## What's *not* in v1

- **Project-scoped joint-result slots.** `Project.fit_2d()` runs a joint
  multi-file fit but currently emits one ordinary `fit_type="2d"` slot
  per file (each carrying that file's projection of the joint result).
  There is no archive construct for a single "joint" slot that owns the
  shared parameter values without per-file duplication. The pipeline
  that would justify one is flagged as architecturally unfinished
  ([TODO.md](https://github.com/InfinityMonkeyAtWork/time-resolved-spectroscopy-fit/blob/main/TODO.md)
  — "Project-level fit backend"). Adding a
  joint slot later is a strict additive change: a new top-level group
  (e.g. `project_slots/`) and a schema-version bump; existing per-file
  2d slots stay untouched.
- **`keep_history=True` full-log save.** The default `Project.save_fits`
  collapses to latest-per-`history_key`. Persisting every refit needs a
  timestamp/sequence component in the slot key; deferred to v2.
- **Model rehydration.** `yaml_filename` is a breadcrumb; v1 does not
  promise to deserialize a `Model` from the archive.
- **MCMC trace metadata beyond acceptance fraction** (autocorrelation
  times, etc.) — schema 3 added `acceptance_fraction`; the rest is still
  not persisted. If the decoupled-MCMC follow-on (the archived design
  plan, "Out of scope") lands, that work owns the schema extension.

## Cross-references

- Object model + identity helpers: [src/trspecfit/utils/fit_io.py](../../src/trspecfit/utils/fit_io.py)
- `FitResults` query API: [src/trspecfit/fit_results.py](../../src/trspecfit/fit_results.py)
- Eager extraction call sites: `_append_*_slot` in [src/trspecfit/trspecfit.py](../../src/trspecfit/trspecfit.py)
- DataFrame builders the schema mirrors: `par_to_df`, `list_of_par_to_df`,
  `conf_interval_to_df` in [src/trspecfit/utils/lmfit.py](../../src/trspecfit/utils/lmfit.py)
- Structural precedent for HDF5 layout: `Simulator.save_data` in
  [src/trspecfit/simulator.py](../../src/trspecfit/simulator.py)
