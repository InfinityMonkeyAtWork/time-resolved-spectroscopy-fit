# Fit-archive HDF5 schema (schema_version 7)

On-disk layout for the fit-results archive written by `Project.save_fits()`
and read by `FitResults.load()` / `Project.load_fits()`. The object model
([utils/fit_io.py](../../src/trspecfit/utils/fit_io.py)) is the source of
truth; this document specifies the 1:1 mapping to HDF5 so the writer and
reader agree on dtypes, attr keys, and None-handling.

Every design rationale (why identity is a stored handle, why corrections are
per-slot, why variants coexist instead of overwriting, the collision-rule
decision table) lives in
[fit_archive_principles.md](fit_archive_principles.md), which is settled and
authoritative. This file is the wire format.

## Version history

`schema_version` is `"7"`. Schema 7 is a **clean break**: the reader accepts
only `"7"` (`SUPPORTED_READ_VERSIONS` in `utils/fit_io.py`), and the writer
refuses to append to an archive whose version differs from its own. Pre-7
archives are not readable — re-fit and re-save under schema 7. The
incremental history of schemas 1–6 lives in this document's git history.

What the break bought, in one pass:

- **Stored identity.** Slots gain the `handle` / `optimization_hash` /
  `input_files` / `model_structure` chain and `fit_view_sha256`; the
  schema-6 `archive_slot_key` / `history_key` pair and `observed_sha256`
  are retired.
- **Immutable raw data, per-slot corrections.** The file group stores
  `data_raw` once; each slot carries the `dark` / `calibration` state it
  was fit under. File-level corrected `data`, `dark`, `calibration`,
  `e_lim`, and `t_lim` no longer exist.
- **First-class joint results.** One `project/joint/` record per
  `Project.fit_2d` optimization, cross-referenced by content hash.
- **Project-owned presentation.** `project/` carries the `PlotConfig`
  JSON, rewritten on every save.
- **Provenance.** Slots persist their model YAML snippets and an optional
  mutable `label`; `fit_settings` gains the optimizer `seed`, the
  effective evaluator backend, and the Jacobian's qualified name.
- **Compression.** Array datasets are gzip + shuffle.

## Conventions

- **Group-path components are positional, zero-padded six-digit keys**
  (`000000`, `000001`, ...). HDF5 path components forbid `/`, and
  user-meaningful names (`File.name`, `model_name`, etc.) can contain
  arbitrary characters. Identity lives in attrs, never in path segments.
- **Strings in attrs and string-typed dataset fields use
  `h5py.string_dtype(encoding="utf-8")`** (variable-length UTF-8). Fixed-
  length string types are not used; readers must not assume any length.
- **`None` handling**:
  - For optional **string attrs** (`label`, `joint_ref`, `fit_settings`):
    omit the attr entirely. The reader treats absence as `None`.
  - For optional **floats** like `stderr` inside structured arrays: write
    `np.nan`. The reader maps NaN back to `None` only for fields where the
    object model permits `None` (`stderr`); other float fields are kept as
    floats.
  - For optional **strings** inside structured arrays (long-form params
    `expr`): write `""`. The reader maps `""` back to `None` on columns
    where the object model permits `None` (lmfit's `expr=None`).
  - These slot-specific `↔` mappings are applied by the slot reader, not
    the generic DataFrame decoder. `conf_ci`, `mcmc/flatchain`, `mcmc/ci`,
    and sbs `params` carry no None semantics; their literal `""` / `NaN`
    values are data.
  - For optional **groups/datasets** (`dark`, `calibration`, `conf_ci`,
    `correl`, `mcmc/`, `model_yaml/`, `aux_axis`, ...): omit the
    group/dataset. The reader treats absence as `None`.
  - **Metric attrs omitted by rule** (the DoF metrics of a joint
    projection; a joint record's `r2`) rehydrate as `NaN`, never as
    `None` — a structurally undefined metric is still a float slot in the
    metrics dict.
- **Float dtype**: structured-array float fields and metric attrs are
  `float64`. The user-array datasets — file `data_raw` / `energy` /
  `time` / `aux_axis` and slot `observed` / `fit` — are written in **the
  source array's native dtype**. This preserves byte-for-byte equivalence
  with the inputs, which `file_content_hash` and `fit_view_sha256` cover
  (both hash dtype, shape, and content); casting on write would
  invalidate them. The reader does not re-cast.
- **Integer dtype**: positional/index attrs and `shape` are `int64`.
- **Bool**: `vary` in params is HDF5 `bool` (numpy `?`).
- **Tuple-valued attrs** (`shape`): stored as 1D `int64` arrays.
- **Compression**: every array dataset is created with
  `compression="gzip", shuffle=True`. Two exceptions, both forced by
  HDF5 (filters require chunked storage): zero-size arrays (e.g. the
  empty `time` axis of a 1D file) and the scalar vlen-utf8 `model_yaml`
  snippet datasets are stored uncompressed.
- **Read-only rehydration**: every array the reader returns has its
  write flag cleared (principles, Principle 4 — the records are
  snapshots, not views).

### DataFrame encoding

All persisted `pd.DataFrame` payloads follow one uniform rule so the
writer/reader has a single code path and column labels never collide with
HDF5 field-name restrictions:

1. **All-numeric DataFrames** (homogeneous `float64` columns, e.g. sbs
   `params`, `flatchain`, `correl`): 2D `float64` dataset of shape
   `(n_rows, n_cols)`, plus attr `columns` — a 1D vlen-utf8 array of
   length `n_cols` listing the column labels in axis-1 order.

2. **Heterogeneous-dtype DataFrames** (e.g. long-form `params`,
   `conf_ci`, `mcmc/ci`): 1D structured dataset of shape `(n_rows,)`
   with fields named positionally `c000000`, `c000001`, ... (zero-padded
   six-digit, matching the group-key convention). Each field's dtype is
   chosen per-column from `{vlen str, float64, bool}`. Attr `columns`
   (1D vlen-utf8 array, length = field count) gives the actual column
   labels in field order. Attr `dtypes` (1D vlen-utf8 array, same
   length) gives a short type tag per column from
   `{"str", "float64", "bool"}` so the reader can rebuild the DataFrame
   without inferring dtypes back.

This convention isolates HDF5 from arbitrary user-facing labels (e.g.
sigma columns like `"+1"`, `"best fit"`, or future column renames in
`par_to_df`) without giving up structured-array benefits for mixed
dtypes.

## Top-level layout

```
<archive>.fit.h5
├── metadata/                        # wire-format identity only
│     attrs:
│       format             : str     # "trspecfit-fit-archive"; written once
│       schema_version     : str     # "7"; written once
│       trspecfit_version  : str     # updated on every write
│       timestamp_created  : str     # ISO 8601 UTC, first write
│       timestamp_updated  : str     # ISO 8601 UTC, most recent write
└── project/                         # exactly one
    │   attrs:
    │     name         : str         # written once; checked on append
    │     plot_config  : str         # PlotConfig JSON; rewritten on every save
    ├── joint/                (opt)  # one child per joint optimization
    │   └── 000000/                  # see "Joint record group"
    └── files/
        ├── 000000/                  # see "File group"
        └── 000001/...
```

`save_fits` is append-mode: an existing archive is augmented in place
(`timestamp_created` preserved; `timestamp_updated` and `plot_config`
rewritten), and the canonical way to start fresh is a new path. Three
append-time refusals, all raised **before any mutation**:

- **Schema mismatch** — the writer never appends across schema versions.
- **Foreign HDF5** — a non-empty file without fit-archive `metadata`
  attrs is not silently converted.
- **Different project** — one project per archive; `project/name` is
  written once and checked on every append.

## File group

```
project/files/000000/
├── metadata/
│     attrs:
│       name              : str     # File.name — the identity
│       original_path     : str     # breadcrumb; never identity, never matched
│       dim               : int64   # 1 or 2
│       shape             : int64[ndim]
│       file_content_hash : str     # 64 hex; see below
├── data_raw                        # immutable, uncorrected; source dtype
├── energy                          # 1D; source dtype
├── time                            # 1D; length 0 for 1D files
├── aux_axis                 (opt)  # omitted iff File.aux_axis is None
└── slots/
    ├── 000000/                     # see "Slot group"
    └── 000001/...
```

There is deliberately **no** file-level `data`, `dark`, `calibration`,
`e_lim`, or `t_lim`: corrected data is a *derived* view, reconstructed on
demand from `data_raw` plus a slot's own correction snapshots
(`utils.arrays.apply_corrections`), and fit windows are per-slot
selection state. A file group exists only as the referent of at least one
slot (Principle 0 — this is a fit archive, not a project serializer);
unfitted files are absent.

`file_content_hash` is one sha256 over a tagged record of `data_raw`,
`energy`, `time`, and `aux_axis` — each contributing dtype, shape, and
content; a `None` axis contributes a sentinel distinct from a zero-length
one (`compute_file_content_hash`). It is a **content stamp**, never
identity: file identity is the `name` attr, guarded unique in the live
session (Principle 1).

### Identity and collisions (file level)

- **Write side.** The writer's "find existing file group" lookup matches
  by `name` alone, in positional-key order. An incoming file that matches
  a group by name but differs in `file_content_hash` raises `ValueError`
  in **both** `overwrite` modes, before any mutation — a reused name, or
  a different measurement re-saved to the same path, cannot file new
  slots under another payload's data. `overwrite=` is slot-scoped and
  never authorizes this.
- **Read side.** Each slot keeps the parent association to its own
  `SavedFile` record — the reader hands `FitResults` `(record, slot)`
  pairs, so a slot always resolves to its own group's axes and data.
  `original_path` participates in no lookup.

## Slot group

```
project/files/000000/slots/000000/
├── metadata/
│     attrs:
│       # --- identity ---
│       handle             : str     # 64 hex; sha256(optimization_hash + file_name)
│       optimization_hash  : str     # 64 hex; shared by all joint siblings
│       input_files        : str     # JSON: [scope, [[name, version_stamp, selection], ...]]
│       model_structure    : str     # JSON: canonical per-file structure encoding
│       fit_view_sha256    : str     # comparability hash; see below
│       fit_type           : str     # "baseline" | "spectrum" | "sbs" | "2d"
│       model_name         : str     # display; identity lives in model_structure
│       # --- provenance ---
│       fit_alg            : str     # final-stage optimizer method
│       fit_settings       : str  (opt)  # JSON dict, sorted keys; see below
│       timestamp          : str     # ISO 8601 UTC, slot creation time
│       label              : str  (opt)  # user-set, post-hoc, MUTABLE; see collisions
│       joint_ref          : str  (opt)  # owning joint record's optimization_hash
│       #                              present iff input_files scope == "project"
│       # --- noise (snapshot at fit time) ---
│       noise_type         : str
│       sigma_source       : str
│       sigma_type         : str
│       sigma_data         : float64  # NaN when no σ was set
│       sigma_eff          : float64  # NaN when no σ was set
│       # --- metrics (non-sbs only) ---
│       chi2_raw, chi2, r2                : float64  # always
│       chi2_red_raw, chi2_red, aic, bic  : float64  # omitted for joint projections
├── params                          # layout depends on fit_type; see below
├── params_meta              (opt)  # sbs only
├── params_stderr            (opt)  # sbs only
├── params_init              (opt)  # sbs only
├── observed                        # 1D or 2D; source dtype
├── fit                             # same shape as observed; source dtype
├── fit_ini                  (opt)  # same shape as fit
├── dark                     (opt)  # correction in force for this fit
├── calibration              (opt)  # correction in force for this fit
├── metrics_per_slice        (opt)  # sbs only
├── conf_ci                  (opt)
├── correl                   (opt)
├── mcmc/                    (opt)
├── components               (opt)  # 1D fit types only
├── component_names          (opt)  # present iff components is
└── model_yaml/              (opt)  # one dataset per attached YAML snippet
    ├── 000000                      # scalar vlen-utf8 = verbatim snippet text
    │     attrs: role, name, source_file, target_par (opt), sequence_index (opt)
    └── 000001/...
```

`(opt)` = present iff the corresponding `SavedFitSlot` field is non-`None`
or applicable to the fit type. `metrics_per_slice`, `params_meta`,
`params_stderr`, and `params_init` are sbs-only; `components` /
`component_names` are never present for `fit_type == "2d"`; `fit_ini` and
`components` are `None` on joint projections (no per-file seed or
component decomposition exists for a joint optimization).

**There is no separate `selection` attr.** A slot's data selection
(`e_lim`, `t_lim`, `base_t_ind`, `time_point`, ...) is stored only inside
`input_files` — where it is hashed into identity — as the
`selection_json` of the entry matching the slot's own file. The reader
recovers `SavedFitSlot.selection` from there.

The metric split is keyed on **scope, not fit type**: the residual-only
metrics (`chi2_raw`, `chi2`, `r2`) are defined for every non-sbs slot,
while the DoF metrics divide by (or penalize) a parameter count that does
not decompose per file for a joint fit — omitted on disk for projections
(`input_files` scope `"project"`), rehydrated as `NaN`. The joint record
owns the whole-objective values.

### The identity chain

All stored, none recomputed on read (Principle 3 — a stored handle
survives later changes to the hash inputs; an old slot and a new run of
the same configuration then simply appear as two variants). Exact byte
framing lives in the `compute_*` / `encode_*` functions in
`utils/fit_io.py`; each hash is a sha256 over a tagged JSON record, so
no two inputs can collide by concatenation.

| Field | Function | Folds in |
|---|---|---|
| `file_content_hash` | `compute_file_content_hash` | `data_raw`, `energy`, `time`, `aux_axis` (dtype, shape, content) |
| version stamp | `compute_file_version_stamp` | `file_content_hash` + the `dark` / `calibration` state the fit consumed |
| `input_files` | `encode_input_files` | scope (`"file"` \| `"project"`) + per-file `(name, version_stamp, selection_json)`, sorted by name |
| `model_structure` | `encode_model_structure` | per file: energy-model composition and dynamics attachments incl. `frequency`, in model order |
| optimizer settings | `encode_optimizer_settings` | stage count, per-stage methods (`fit_alg_2` only when `stages == 2`), effective backend, `seed` when supplied, Jacobian qualname when a `leastsq` stage is in force |
| `optimization_hash` | `compute_optimization_hash` | `input_files` + `fit_type` + `model_structure` + parameter metadata `(name, min, max, vary, expr)` in model order + initial state (quantized values) + optimizer settings |
| `handle` | `compute_slot_handle` | `optimization_hash` + `file_name` |

The full 64-hex `handle` is authoritative on disk; display abbreviates to
8 chars and every lookup (the query layer's `handle=` / `select=` /
`diff` / `drop_fits`) prefix-matches, git-style. Joint siblings share the
`optimization_hash` and differ only through the file name.

`fit_view_sha256` (`compute_fit_view_sha256`) is **comparability, not
identity**: a tagged record of the `observed` array plus the selected
energy/time coordinates and the aux axis. Two slots' metrics are directly
comparable only when their views match; σ (an attachment) then gates the
σ-scaled metrics on top (principles §Comparability). It is deliberately
not an input to `optimization_hash` — the view is derived from
`input_files` state, and hashing it twice would add nothing.

### `fit_settings` attr

JSON dict (sorted keys) recording the optimizer configuration that can
influence the result — built by `build_fit_settings`:

- all fit types: `stages`, `fit_alg_1`, `fit_alg_2`, `backend` (the
  **effective** evaluator backend the dispatch site actually ran),
  `try_ci`;
- when supplied: `seed` (the optimizer RNG seed, forwarded to the stage-1
  method), `jac_fun` (the analytic Jacobian's qualified name);
- sbs: `seed_source`, `seed_adapt`, `seed_values` (JSON `null` is
  meaningful — "no seed adaptation" is provenance too; these choose
  initial parameter values and are unrelated to the RNG `seed`);
- when MCMC ran: an `mc` sub-dict with the settings as resolved at fit time
  (`use_mc`, `steps`, `nwalkers`, `burn`, `thin`, `ntemps`, `is_weighted`;
  `sigma_ini/min/max` for unweighted sampling only; `seed` when one was
  set — the sampler seed, distinct from the optimizer `seed` above). Knobs
  the caller left at `None` appear with their derived values. SbS records
  slice 0's, mirroring the slot's slice-0 MCMC payload.

Execution details that cannot change the result (SbS / emcee worker
counts; serial ≡ parallel dispatch is pinned by test) are deliberately
excluded. The identity-keyed subset of this dict is derived by
`optimizer_settings_from_provenance` and folded into
`optimization_hash` — a re-run with a changed setting is a distinct
configuration, a distinct handle, a distinct slot.

### Identity and collisions (slot / joint level)

Collisions key on the stored `handle` (slots) or `optimization_hash`
(joint records). All conflicts are detected **before any mutation**, so a
refused append leaves the archive byte-untouched. The rules (principles
§"One rule, both boundaries" — the same comparator also guards the
in-session collapse at save time):

| stored vs incoming | rule |
|---|---|
| fitted **values** differ beyond tolerance | hard conflict — requires `overwrite=True`, which replaces the record wholesale |
| values equivalent; attachment absent in archive, present incoming | written freely (enrichment — e.g. run MCMC later, re-save) |
| values equivalent; attachment stored, absent incoming | kept — never deleted by omission |
| values equivalent; attachment present on **both** sides | requires `overwrite=True`; replaced individually |

Fitted-value equivalence is per parameter, matched by name, on the
`value` column only: `|a − b| <= atol + rtol·|b|` with `rtol = 1e-6`,
`atol = 1e-12`, `NaN == NaN` (`_fitted_values_equivalent`). `stderr` is
not compared — it legitimately drifts with optimizer internals.
Attachments merge **individually**: `conf_ci` may enrich in the same
write that leaves `mcmc/` untouched. "Richer" is not a criterion —
comparing a longer chain against a shorter one at better acceptance has
no well-defined answer, so any present → different-present replacement
is opt-in. The `fit_settings` keys that describe an attachment travel
with it: writing or replacing `conf_ci` updates `try_ci`, writing or
replacing `mcmc/` updates the `mc` block, from the incoming record's
provenance; every other key is fixed by the handle and stays as stored.

`label` is the one **mutable** field: rewritten whenever the incoming
record carries one, with no `overwrite` required, and never deleted by
an incoming `None`.

## `params` dataset

Two shapes depending on `fit_type`, both following the DataFrame-encoding
rule above.

### baseline / spectrum / 2d — long format (one row per parameter)

Heterogeneous-dtype DataFrame:

```
params : 1D structured dataset, shape (n_par,)
  fields (positional, in column order):
    c000000 : vlen str    # column "name"        (e.g. "GLP_01_A")
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

Mirrors `par_to_df(..., col_type="min")` in `utils/lmfit.py`. `stderr` is
the only float column that legitimately holds NaN-as-`None`; `min`/`max`
may carry IEEE `-inf`/`+inf` verbatim. `init_value` is the true pre-fit
seed regardless of `stages` — `fitlib.fit_wrapper` restores it on the
stage-2 result (`restore_true_init_values`), since lmfit's `prepare_fit`
otherwise resets it to stage 1's output.

### sbs — wide format (one row per slice, one column per parameter)

All-numeric DataFrame:

```
params : 2D float64 dataset, shape (n_slices, n_par)
  attrs:
    columns : vlen str[n_par]   # parameter names; axis-1 order
```

Optimized values only. The slice-invariant metadata and the per-slice
stderr / initial values live in the sibling datasets:

- **`params_meta`** — heterogeneous, shape `(n_par,)`, columns
  `["name","vary","min","max","expr"]` (tags
  `["str","bool","float64","float64","str"]`). Captured from the slice-0
  result; rows are column-aligned with the wide `params`. Deliberately
  excludes `value` / `stderr` / `init_value`, which are per-slice.
- **`params_stderr`** — all-numeric, shape `(n_slices, n_par)`. NaN
  where the optimizer reported no stderr; the NaN is data.
- **`params_init`** — all-numeric, shape `(n_slices, n_par)`: the true
  per-slice seed (diverges across slices under `seed_adapt`). Unlike
  `correl` / `conf_ci` / `mcmc` (slice-0-representative), this covers
  every slice.

## `metrics_per_slice` dataset (sbs only)

```
metrics_per_slice : 1D structured dataset, shape (n_slices,)
  dtype: chi2_raw, chi2_red_raw, chi2, chi2_red, r2, aic, bic — all float64
```

Row order follows the time-slice order in `observed` axis 0. The reader
reconstructs `SavedFitSlot.metrics` as `{name: column_array}` for sbs.

## `conf_ci` dataset (optional)

Heterogeneous-dtype DataFrame (one string column for the parameter name,
the rest float):

```
conf_ci : 1D structured dataset, shape (n_par,)
  fields: c000000 (vlen str), c000001..c00000K (float64)
  attrs:  columns / dtypes per the DataFrame-encoding rule
```

Sigma labels come from `conf_interval_to_df` in `utils/lmfit.py`
(typically `["-3", "-2", "-1", "best fit", "+1", "+2", "+3"]`). Omitted
entirely if `conf_ci is None` (the fit ran with `try_ci=0`, or CI
failed). For SbS the table is slice 0's.

## `correl` dataset (optional)

All-numeric DataFrame — the varying-parameter correlation matrix
(`correl_to_df` in `utils/lmfit.py`):

```
correl : 2D float64 dataset, shape (n_vary, n_vary)
  attrs:
    columns : vlen str[n_vary]   # varying parameter names; axis-1 order
```

Square with `index == columns`, so only the column labels are stored.
Omitted when the optimizer produced no covariance and on joint
projections (joint covariance does not decompose per file — the joint
record owns it). For SbS the matrix is slice 0's.

## `mcmc/` group (optional)

```
mcmc/
├── flatchain                       # all-numeric DataFrame; always present
│     2D float64, shape (n_samples, n_par); attrs: columns
├── ci                       (opt)  # heterogeneous DataFrame, layout as conf_ci
├── acceptance_fraction      (opt)  # 1D float64, shape (n_walkers,)
└── attrs:
      lnsigma : float64             # __lnsigma point estimate; NaN ↔ None
```

Omitted wholesale when the fit had no MCMC step. `flatchain` is always
written when the group exists — a 0-row chain with named columns still
records `(0, n_par)` plus the parameter labels. `lnsigma` is NaN when the
sampling was weighted (no nuisance scale); the reader maps NaN back to
`None`. For SbS the payload is slice 0's.

## `components` / `component_names` (optional)

Per-component fit curves for 1D fit types (baseline, spectrum, sbs),
evaluated at final params on the same grid as `fit`. Never present for
`fit_type == "2d"`.

```
components : ndarray (source dtype)
  baseline / spectrum : shape (n_components, n_e_view)
  sbs                 : shape (n_slices, n_components, n_e_view)
component_names : 1D vlen-utf8 dataset, shape (n_components,)
```

Both fields are omitted together. `component_names` is captured directly
from `[comp.name for comp in model.components]` at fit time rather than
re-derived from parameter names: a static attached `par_profile` splices
a nested model's parameters into its host component's name block, which
would break any prefix-based re-derivation. Summing `components` along
its component axis reconstructs `fit` exactly.

## `fit_ini` dataset (optional)

Model evaluated at the true pre-fit seed (`FitOutput.par_ini`), on the
same grid as `fit`:

```
fit_ini : ndarray (source dtype)
  baseline / spectrum : shape (n_e_view,)
  2d                  : shape (n_t_view, n_e_view)
  sbs                 : shape (n_slices, n_e_view)   # every slice
```

`None` on joint projections (no per-file seed exists). Rendered (1D fit
types) as the dotted-gold initial-guess overlay when
`PlotConfig.show_init` resolves `True`; NaN-padded outside the fit window
in `full_range` mode — never a fabricated extrapolation.

## `model_yaml/` group (optional)

Verbatim provenance of the fitted model's YAML snippets — one scalar
vlen-utf8 dataset per snippet, its metadata riding as attrs on the same
dataset so a mismatched record is structurally unrepresentable:

```
model_yaml/000000
  data  : the snippet text, verbatim
  attrs : role           ("energy" | "dynamics" | "profile" | ...)
          name           (model / submodel name)
          source_file    (the YAML file it came from)
          target_par     (opt; dynamics/profile attachment target)
          sequence_index (opt; int64, position in an attachment sequence)
```

Omitted for models built programmatically with no YAML source.
Provenance, not identity, and not a complete record — component
ordering, `frequency`, and profile attachment are programmatic, so the
archive does not promise model rehydration from it.

## Joint record group

One child of `project/joint/` per `Project.fit_2d` optimization. A
one-file project fit is still a joint record — it came through the
project-scoped path.

```
project/joint/000000/
├── metadata/
│     attrs:
│       optimization_hash : str     # 64 hex; shared with every projection slot
│       input_files       : str     # JSON; scope == "project"
│       model_structure   : str     # JSON; one entry per participating file
│       model_name        : str     # display
│       projections       : str     # JSON records; see below
│       label       (opt) : str     # the bundle's only label (mutable)
│       fit_alg           : str
│       fit_settings      : str     # JSON, sorted keys
│       timestamp         : str
│       chi2_raw, chi2_red_raw, chi2, chi2_red, aic, bic : float64
│       #                           # whole-objective; r2 omitted (undefined),
│       #                           # rehydrated as NaN
├── params                          # combined table; long format incl. init_value
├── conf_ci                  (opt)  # joint profiled CI
├── correl                   (opt)  # joint correlation matrix
└── mcmc/                    (opt)  # joint chain; layout as the slot mcmc/
```

`params` is the authoritative combined parameter table in optimizer
order: per-file names carry a `file{idx:02d}_` position prefix,
project-shared parameters are unprefixed, and `init_value` is the
effective optimizer-entry value (after baseline-result injection,
project-sharing resolution, and expression evaluation). The prefix is a
naming convention for the optimizer, **not** an interface — readers use
the parameter map, below.

The `projections` attr is canonical JSON, records sorted by file name:

```
[{"file_name": ..., "handle": ..., "parameter_map": {combined: local, ...}}, ...]
```

Each `parameter_map` is data mapping combined optimizer names to the
projection's local parameter names — nothing anywhere pattern-matches the
`fileNN_` prefix, so a component legitimately named `file00` round-trips.

**Cross-references are content-addressed, never positional**: a
projection identifies its slot by `handle`, and every projection slot's
`joint_ref` attr holds the record's `optimization_hash`. Content
addressing survives an archive being rewritten or merged, where a
positional path (`"files/000000"`) would silently point at the wrong
group.

Bundle invariants, enforced by the writer before any mutation and
re-validated by the reader:

- `input_files` scope `"project"` ⟺ `joint_ref` present, in both
  directions — a project-scoped slot without a resolvable joint record
  (or the reverse) is a corrupt bundle and raises.
- Every declared projection `handle` must resolve to a stored slot;
  partial bundles are not representable (`save_fits` expands any
  selection touching a bundle member to the whole bundle for exactly
  this reason).
- Every `parameter_map` must be total in both directions **and**
  value-consistent: each mapped projection value must equal the combined
  table's value for its combined name.

The record and its projections carry **disjoint payloads**: the joint
`conf_ci` / `correl` / `mcmc` live only on the record (projections have
none to conflict over), the per-file arrays and σ live only on the
projections. Collision handling for a re-saved joint record follows the
same table as slots, keyed on `optimization_hash`, applied as one
logical transaction over the bundle.

## Reader → object-model mapping

Per slot, the reader produces a `SavedFitSlot` with:

| `SavedFitSlot` field | Source |
|----------------------|--------|
| `handle`, `optimization_hash`, `input_files`, `model_structure`, `fit_view_sha256` | slot `metadata` attrs, verbatim (never recomputed) |
| `file_name` | parent file group's `metadata.name` attr |
| `model_name`, `fit_type`, `fit_alg`, `timestamp` | slot `metadata` attrs |
| `selection` / `selection_json` | the `input_files` entry matching the slot's file name |
| `label`, `joint_ref` | slot `metadata` attrs (`None` if absent) |
| `noise_type`, `sigma_source`, `sigma_type`, `sigma_data`, `sigma_eff` | slot `metadata` attrs |
| `params` (+ `params_meta` / `params_stderr` / `params_init`) | datasets → DataFrames; long-form `""`/NaN ↔ `None` restored |
| `fit_settings` | `metadata.fit_settings` JSON → dict (`None` if absent) |
| `metrics` | scalar attrs (non-sbs; omitted attrs → NaN) or `metrics_per_slice` (sbs) |
| `observed`, `fit`, `fit_ini`, `dark`, `calibration`, `components` | datasets → read-only ndarrays (`None` if absent) |
| `component_names` | dataset → list of str (`None` if absent) |
| `conf_ci`, `correl`, `mcmc` | datasets/group (`None` if absent); `correl` index restored from `columns` |
| `model_yaml` | `model_yaml/` group → `ModelYamlRecord` tuple (`None` if absent) |

Joint records rehydrate with their projection slots resolved by `handle`
to the **same objects** returned under `files/` — `FitResults.load`
serves one object per slot, whether reached directly or through a
bundle. The reader validates the bundle invariants above on every load;
metric attrs omitted on disk come back as `NaN`.

## Per-fit-type cheat sheet

| fit_type | `observed.shape` | `params` layout | metrics location | `components.shape` | `fit_ini.shape` | sbs-only datasets | t_lim applied |
|----------|------------------|-----------------|------------------|--------------------|-----------------|-------------------|---------------|
| baseline | `(n_e_view,)` | structured (long) | scalar attrs | `(n_components, n_e_view)` | `(n_e_view,)` or `None` | — | n/a |
| spectrum | `(n_e_view,)` | structured (long) | scalar attrs | `(n_components, n_e_view)` | `(n_e_view,)` or `None` | — | n/a |
| sbs | `(n_t_full, n_e_view)` | 2D float64 + `columns` (wide) | `metrics_per_slice` | `(n_t_full, n_components, n_e_view)` | `(n_t_full, n_e_view)` | `metrics_per_slice`, `params_meta`, `params_stderr`, `params_init` | **no** |
| 2d | `(n_t_view, n_e_view)` | structured (long) | scalar attrs | always `None` | `(n_t_view, n_e_view)` or `None` | — | yes |

`n_e_view` / `n_t_view` denote the axes cropped by `e_lim` / `t_lim`;
`n_t_full` is the file's full time-axis length — `fit_slice_by_slice`
iterates every slice regardless of `t_lim`, so an sbs slot's selection
carries `t_lim = None`. `baseline` / `spectrum` reduce time via
`base_t_ind` / `time_point` / `time_range`, captured in the selection.

`Project.fit_2d` produces one `fit_type="2d"` projection slot per file
plus the `project/joint/` record; `File.fit_2d` produces a file-scoped
2d slot and no joint record. The two are distinguished by `input_files`
scope and the presence of `joint_ref`, never by a separate fit type.

## Deliberately not stored

- **Model rehydration.** `model_yaml/` is provenance; ordering,
  `frequency`, and profile attachment are programmatic, so no complete
  model record exists on disk (`frequency` does reach the archive inside
  `model_structure`, as identity).
- **A full refit log.** Distinct configurations never collide (identity
  is the complete input), so every variant can be archived side by side;
  exact re-runs dedup to the latest per handle. A timestamped log of
  byte-identical re-runs adds nothing the session history doesn't have.
- **Recomputed identity.** Handles are stored, never recomputed on read;
  see Principle 3 for the self-correcting variant behavior across hash
  revisions.
- **Deletion.** The archive is append-and-merge; pruning happens
  in-session (`Project.drop_fits`) before saving, or by saving a
  selection (`select=`) to a new path.

## Cross-references

- Design rationale and decision tables:
  [fit_archive_principles.md](fit_archive_principles.md)
- Object model, identity helpers, writer/reader:
  [src/trspecfit/utils/fit_io.py](../../src/trspecfit/utils/fit_io.py)
- `FitResults` query API: [src/trspecfit/fit_results.py](../../src/trspecfit/fit_results.py)
- Slot capture call sites: `_append_*_slot` / `capture_saved_file` in
  [src/trspecfit/trspecfit.py](../../src/trspecfit/trspecfit.py)
- DataFrame builders the schema mirrors: `par_to_df`,
  `list_of_par_to_df`, `conf_interval_to_df` in
  [src/trspecfit/utils/lmfit.py](../../src/trspecfit/utils/lmfit.py)
