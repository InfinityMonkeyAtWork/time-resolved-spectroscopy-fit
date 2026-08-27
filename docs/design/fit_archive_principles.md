---
orphan: true
---

# Fit-archive principles

Status: **settled.** Branch `fit-archive-schema-7`. No open forks remain;
[fit_archive_schema_plan.md](fit_archive_schema_plan.md) has been rewritten
against these principles.

This document answers three questions that everything else in the fit archive
depends on:

1. What is the archive **for**?
2. What constitutes a unique **file** in a project?
3. What constitutes a unique **fit**?

It is deliberately about *principles*, not wire format. The on-disk layout is a
consequence: [fit_archive_schema.md](fit_archive_schema.md) documents the
current format (schema 6) and stays authoritative until the conversion lands;
[fit_archive_schema_plan.md](fit_archive_schema_plan.md) is the conversion plan
and has been rewritten against this document, carrying no rationale of its own.
Both are downstream — when this document and either of them disagree, this one
wins and the other is wrong.

The archive is a wire format: it is the hardest thing in the project to change
later. Getting identity wrong here means encoding a broken model into the
artifact users accumulate. Hence settling this first.

## Principle 0 — this is a fit archive, not a project serializer

> A fit archive contains selected fit records and the file-level content
> referenced by those records. It is **not** a serialization of the live
> `Project` session.

Everything below is bounded by this. Unfitted registered files,
loaded-but-unused models, and other session state do not belong in the
archive. If full project serialization is ever wanted, it is a separate
artifact with its own format, not an expansion of this one.

This reverses an earlier working decision (schema plan D9, "the archive is a
project snapshot: every registered file with data gets a group, fitted or
not"). That decision predates the rest of this document and contradicts it;
the earlier schema-plan draft was corrected against it. It is recorded here
rather than silently dropped because it is the most likely way the scope creeps
back.

### What follows from it

- **A file group exists only as the referent of at least one slot.** An empty
  `slots/` group never occurs. A file registered on the `Project` but never
  fit is absent from the archive entirely.
- **Capture has one path, not two.** The immutable `SavedFile` payload is
  captured when a file produces its first slot. There is no second path
  reading live state at save time for unfitted files, and no question about
  what to do with a bare `File()` that has no data.
- **`select=` needs no carve-out.** It filters slots; file groups follow from
  the slots that survive. If filtering removes every slot for a file, that
  file group is not written. Under the project-snapshot rule, filters applied
  to slots while file groups were written unconditionally — an inconsistency
  that now disappears.
- **Archive-vs-session divergence stops being a question.** "What if the
  project has fewer files than last save — drop the group, or keep a union?"
  does not arise, because groups are created by fits, not by registration.
- **The mirroring intent survives.** The structural correspondence
  `Project → File → Fit` (Principle 1) is about *shape*, never about
  completeness. Only the population rule changes.

### Two hashes, two jobs

Scoping the archive to referenced content sharpens the file-level integrity
check. The file group stores only immutable content (`data_raw`, `energy`,
`time`, `aux_axis` — Principle 4), so:

| Hash | Answers |
|---|---|
| `file_content_hash` | is this the same measurement, under this name? |
| `file_version_stamp` | which data version did this fit see? |

Their exact contents are defined once, in Principle 3's formula — the version
stamp nests the content hash rather than overlapping it. This table is about the
two *roles*, deliberately not restating the inputs, since a second copy of them
is what drifts.

Two files sharing a name but differing in `data_raw` are genuinely different
measurements and must raise. Two *fits* on the same file differing in their
version stamp are the normal result of applying a correction and refitting —
not a conflict. The schema plan's fingerprint-based collision rule conflates
these; the earlier schema-plan draft was rewritten against this split.

## Principle 1 — identity is a guarded name; content hashes are version stamps

> A name, unique within its parent and guarded against collision and mutation,
> at every level: `Project → File → Model`. Content hashes are version stamps,
> never identity.

### The problem being fixed

Three competing identity notions exist today and they disagree:

| Notion | Where | Stable? |
|---|---|---|
| `File.name` | enforced unique at construction ([trspecfit.py:1664-1670](../../src/trspecfit/trspecfit.py#L1664)) | no — freely reassignable afterwards |
| multi-sha fingerprint over corrected `data` + `energy` + `time` | [fit_io.py:423-449](../../src/trspecfit/utils/fit_io.py#L423) | no — changes whenever the data is corrected |
| `(fingerprint, name, original_path)` | archive write side | inherits both defects |

The fingerprint is currently treated as primary ("Identity uses fingerprint;
`file_name` is metadata only", [fit_io.py:171](../../src/trspecfit/utils/fit_io.py#L171)).
That is the error: **it hashes mutable content**, so a file's identity changes
under operations that alter its numbers without altering what it *is*.

Verified failure (2026-07-25): applying `subtract_dark()` moves
`data_sha256` from `60ff129e…` to `8be449ca…`. Any slot fit before that point
no longer matches the live file, `_find_file_for_slot` returns `None`, and
`save_fits()` raises `"Slot for file 'A' has no matching Project.files entry"`
— aborting the **entire** save, not just that slot. Correcting your data after
fitting makes your earlier fits unsaveable.

It fails in the other direction too: byte-identical replicate files collide,
which is patched by adding `file_name` to the key — but names are unguarded
(`f.name = "B"` silently succeeds and breaks the project's own lookup), so the
patch shares the defect it patches.

The stated rationale for fingerprint-primary — forgiving cross-machine
matching of an archive to a live project — **has no consumer.** Every
fingerprint use in `src` is archive-internal (`_files_by_fp`, matching slots
to `SavedFile`s from the same archive) or in-session (`_find_file_for_slot`,
`_resolve_save_file_filter`). `FitResults.load(file=...)` filters on name
strings ([fit_results.py:394](../../src/trspecfit/fit_results.py#L394)).

### The rule

A `File` is *one measured dataset*. Its identity is not its current numeric
contents — those legitimately change under corrections. So:

- **Identity is `File.name`**, unique within its `Project` and guarded on
  mutation as well as construction.
- **The content hash is a version stamp**, answering a different question: is
  this result still current with respect to the file's data? Divergence is a
  reportable staleness fact, not a fatal lookup failure.
- **`original_path` is a breadcrumb.** People move files; it must never
  participate in identity.

The same rule applies one level down: **model names must be unique within a
`File` and guarded.** Uniqueness is enforced at load time (`File.load_model`
rejects a name that `select_model` already resolves); the missing half was
the mutation guard, closed together with `File.name`'s.

### What this deletes

- `_files_by_fp` ([fit_results.py:197](../../src/trspecfit/fit_results.py#L197))
  and the byte-identical-file collision it causes. This removes the problem
  class rather than working around it.
- The **dual-key system**. `archive_slot_key` and `history_key` exist only
  because in-memory identity (multi-sha fingerprint) and on-disk identity
  (positional group path) are different primitives. With a name at both ends
  there is one key.
- `original_path` from every identity tuple.

## Principle 2 — durability classes

Where state lives is decided by the live session; whether it is persisted is
decided by what kind of state it is.

| Class | Treatment | Examples |
|---|---|---|
| Content / identity | persist at the owning level | project name; file name, `data_raw`, axes |
| Input to the numbers | persist, frozen per fit | σ, `dark`, `calibration`, parameter state, `fit_settings`, evaluator backend |
| Presentation | persist at the owning level, resolved at render time | `PlotConfig` |
| Session ephemera | never persist | `show_output`, `Project.path`, `_config_file`, live lmfit objects |

The discriminator between rows 2 and 3: **anything that entered the numbers is
frozen into the fit record; anything that only affects rendering lives at its
owning level and is resolved when you plot.**

## Principle 3 — a fit is unique by its complete input

> Fit identity is one hash over everything that fed the optimizer.

### Why not a coarse key

The current key is `(file, model_name, fit_type, selection)` and deliberately
excludes `fit_settings`, σ, and model contents — "a refit with different
settings is still a refit."

That is wrong for the primary use case. The most common thing a user does is
*"how much better does the fit get if I flip this parameter from `vary=False`
to `True`?"* If those two runs are not two fits, the archive cannot do the one
job it exists for. Variant comparison **is** the workload; treating it as
clutter optimizes storage against the point.

The coarse key also has a concrete defect: `model_name` is a bare
user-supplied string (`mcp.py:2105`, built by joining the selected model keys
via `model_list_to_name`) with no relationship to model content, and
duplicates are unguarded. Two different YAML files sharing a top-level key
produce the same name, so the second fit silently replaces the first.

### The primitive: resolved parameter table, not YAML text

The intuitive candidate is to hash the model's YAML entry. That is not
sufficient, because the YAML is not the whole model: `add_time_dependence()`,
`add_par_profile()`, and `Model.update_value()`
([mcp.py:581](../../src/trspecfit/mcp.py#L581)) all mutate a loaded model
without touching any YAML — potentially including the exact `vary` flip that
motivates the whole design.

The **resolved pre-fit parameter table** — `(name, init_value, min, max,
vary, expr)` per parameter — is literally "initial guess, bounds, vary,
constraints". It captures the YAML path and the programmatic path alike, it is
the actual state handed to the optimizer, and it is already persisted: in the
long-form `params` frame for baseline, spectrum, and 2D, and as `params_meta`
plus the per-slice `params_init` matrix for SbS, whose wide `params` frame holds
optimized values only. Those two shapes are unified below into shared metadata
plus an initial-state matrix.

### The rule

```
file_content_hash  = sha256(dtype + shape + data_raw + energy + time + aux_axis)
file_version_stamp = sha256(file_content_hash + dark + calibration)

input_files = (scope, sorted tuple of (file_name, file_version_stamp, selection))
               scope ∈ {"file", "project"}

optimization_hash = sha256(
    input_files                  # which data, which corrections, which windows
  + fit_type
  + model_structure              # per-attachment records, each with its own
                                 # submodel order and frequency; see below
  + shared parameter metadata    # name, bounds, vary, expr — IN MODEL ORDER
  + initial-state matrix         # (n_slices, n_par); n_slices = 1 except SbS
  + optimizer settings           # only what was actually in force; see below
)

slot_handle = sha256(optimization_hash + file_name)
```

The two file hashes nest rather than overlap: content is what the file *is*,
the version stamp adds how it was corrected. `aux_axis` is inside the content
hash — it is immutable file content that `par_profile` models consume
numerically, and omitting it would let two files differing only in their
auxiliary axis produce colliding fits. (The schema-5 decision to exclude it
was made when the hash *was* file identity; under Principle 1 it is a version
stamp feeding a fit's input hash, and that rationale no longer transfers.)

### `model_structure`: a collection of per-attachment records

The parameter table does not fully describe the objective function — model
structure can change the numbers without changing any parameter's name,
initial value, bounds, `vary`, or expression. Dynamics attachments are the
concrete instance.

`add_time_dependence(target_model, target_parameter, dynamics_yaml,
dynamics_model, *, frequency=-1)`
([trspecfit.py:3849](../../src/trspecfit/trspecfit.py#L3849)) attaches **one
dynamics model to one parameter**, and each call carries its own
`dynamics_model` list and its own `frequency`. A model may therefore hold
several attachments running at different frequencies. So `frequency` is **not**
a global scalar — it belongs inside each attachment record:

```
per_file_structure = (
    ordered tuple of energy-model names,
    sorted tuple of dynamics attachments, each:
        (target_parameter, ordered submodel tuple, frequency)
)

model_structure = sorted tuple of (file_name, per_file_structure)
```

**`model_structure` is per file, always** — one entry for a single-file fit, N
for a joint one. Each file owns its *own* model instance: `_build_fit_params`
resolves `f.select_model(model_name)` per file, and while
`Project.add_time_dependences` applies one frequency across all of them,
`File.add_time_dependence` is public and per-file. So a joint fit where A runs
at `frequency=0.25` and B at `0.5` under one model name is reachable through the
public API, and a single global structure would hash those identically —
silently voiding the frequency protection exactly where it is hardest to notice.

Keeping the mapping uniform rather than bare-for-single/mapping-for-joint costs
one nesting level in the common case and removes a conditional shape. Given that
a model-conditional `aux_axis` rule already produced one bug in this design,
the conditional is not worth its brevity.

Attachments are **sorted** for canonicality — two attachments to different
parameters are independent, so the order of the `add_time_dependence()` calls
must not change the hash. Within each record the submodel tuple keeps its
**order**, because that is what assigns subcycles.

`examples/fitting_workflows/03_multi_cycle_dynamics` shows why: it declares
`IRF`, `MonoExpNeg`, and `MonoExpPos` as three unrelated top-level YAML models,
and what makes them a cycle sequence is entirely programmatic —

| Fact | Where it lives |
|---|---|
| which submodel is which subcycle | the *order* of the list passed to `add_time_dependence()` |
| repetition rate | that call's `frequency` kwarg |

`Component.subcycle` ([mcp.py:1292](../../src/trspecfit/mcp.py#L1292)) needs no
separate treatment: it is *derived* from the same list order, so the two move
together. Setting it inconsistently with the list order is not a supported path
— no YAML key declares it. `mask` and `time_norm` are likewise derived from
`frequency` + `subcycle`.

Profile attachments need no record of their own: their topology is already in
the parameter names (`Gauss_01_A_pExpDecay_01_tau`) and they carry no
non-parameter configuration analogous to `frequency`. If they ever gain some,
they get records on the same footing.

`frequency` appears in no YAML and is **persisted nowhere** today — the example
fits at `frequency=0.25` and the archive cannot say so. It therefore needs both
hashing and storing.

**`model_structure` is a canonical ordered tuple, not the joined composite
name.** `model_list_to_name` joins with `_`, which is lossy: model names are
*not* underscore-constrained — the no-underscore guard
(`test_config_functions.py:116-134`) covers only function and parameter names in
the `functions/` registries, and `parsing.py:130-133` explicitly handles
underscores in component names. So `['IRF', 'MonoExp_Neg']` joins to
`'IRF_MonoExp_Neg'`, indistinguishable from a single model of that literal
name. Hashing the tuple removes that collision class. It also makes the
load-bearing property explicit — what matters is the *ordering*, not the label.

One cost worth naming: renaming an **energy model or a dynamics submodel** in
YAML mints a new slot even when the science is identical, since those names are
the tuple's elements. That is the safe direction — it can only split slots,
never merge them — and arguably correct, since the user chose to call it
something different. Profile-model names are deliberately absent from
`model_structure`; a profile rename still propagates, but through the parameter
table, because `mcp.py:696` requires a profile model's name to match its target
parameter exactly.

**No general "model config manifest".** One was considered and rejected as
over-engineering: after accounting for `model_name`, the independent field
count is one. A guard test enumerating model attributes to catch future
additions was rejected with it — distinguishing configuration from cache from
derived state is a per-attribute judgment call, making such a test
high-maintenance and low-confidence, unlike the repo's existing naming-convention
guard. The maintenance obligation is instead documentary: **adding
non-parameter model configuration means extending this hash**, and that belongs
in the "adding YAML syntax / model configuration" task recipe when it is
written.

**The parameter table is hashed in model order, never sorted.** Component
order is load-bearing: `Shirley`
([energy.py:74-87](../../src/trspecfit/functions/energy.py#L74)) and `LinBack`
both take the accumulated `spectrum` as an argument, so moving a background
before or after a peak changes the numbers. Canonicalizing by sorting
parameter names — the obvious thing to reach for — would erase model topology
from the hash silently. Contrast `input_files`, which *is* sorted, because
file order is meaningless.

Any difference in these inputs → a new slot. One rule; no case-by-case
adjudication about which inputs "count".

**`optimizer_settings` is a keyed subset of the stored `fit_settings`, not the
whole dict.** `build_fit_settings`
([fit_io.py:580-625](../../src/trspecfit/utils/fit_io.py#L580)) returns one
dict carrying `try_ci` and an `mc` sub-dict alongside the optimizer fields,
and the slot persists all of it as provenance. Hashing that dict wholesale
would silently key σ, confidence intervals, and MCMC — contradicting the
attachment rule below.

**Key what was in force, not what was passed.** A setting that the run did not
consume must not enter the hash, or it mints a false variant of an identical
computation:

| Keyed | Condition |
|---|---|
| `stages` | always |
| `fit_alg_1` | always |
| `fit_alg_2` | **only when `stages == 2`** — with one stage it is never called ([fitlib.py:799](../../src/trspecfit/fitlib.py#L799)) yet still defaults to `"leastsq"` |
| effective evaluator backend | always |
| `seed` | when supplied (SciPy rejects it on methods that ignore it, so supplied implies consumed) |
| `jac_fun` identity | when an analytic Jacobian was actually applied — i.e. `jac_fun is not None` and the method is `leastsq` ([fitlib.py:789](../../src/trspecfit/fitlib.py#L789)) |

`jac_fun` has to be keyed because a caller can override the backend's
Jacobian: both call sites use
`fit_wrapper_kwargs.setdefault("jac_fun", ...)`
([trspecfit.py:1414](../../src/trspecfit/trspecfit.py#L1414),
[:4077](../../src/trspecfit/trspecfit.py#L4077)), so a `jac_fun` passed through
`fit_2d`'s `**fit_wrapper_kwargs` wins. Recording the effective backend does
not cover this — the override is independent of which backend ran. Since a
`Dfun` replaces finite differences, a different or incorrect Jacobian can
converge elsewhere. Key its **qualified name** (`module.qualname`), or `None`
when no analytic Jacobian was applied.

A qualified name is sufficient for the package's own Jacobians, which is the
supported workflow. It does not identify closures, lambdas, or monkey-patched
functions — see "What the hash cannot cover" for the explicit scope boundary.

The SbS `seed_source` / `seed_adapt` / `seed_values` are keyed **nowhere** —
the initial state they produce is hashed directly instead (see "The initial
state is the identity input").

The backend recorded is the **effective** one, not
the requested one: the JAX gate rejects some models and falls back to the
compiled NumPy path, so `spec_fun_str == "fit_model_jax"` does not by itself
say what ran — and the analytic-Jacobian path follows from whichever backend
actually executed.

### Composite keys must preserve structure, association, and multiplicity

> Hash canonical **tagged records** — never bare values, sets, or unframed byte
> concatenations.

This is stated as a rule because the same mistake has been made five times in
this design, each time in a different disguise — the fifth was caught *by* the
rule, which is the argument for having written it down:

| Failure | What was discarded |
|---|---|
| content-only file keys (`_files_by_fp`) | **association** — which file the content belonged to |
| `observed_sha256` over `tobytes()` | **structure** — dtype and shape |
| a set of view hashes for joint records | **association and multiplicity** — byte-identical files collapse |
| projection lists sorted by file name | **association** — combined parameter names encode `Project.files` position, not name |
| `model_list_to_name`'s `_` join | **framing** — `['IRF','MonoExp_Neg']` and `['IRF_MonoExp','Neg']` are indistinguishable |

The framing case generalizes beyond names: concatenating `a + b` without
delimiters or length prefixes makes `("ab", "c")` and `("a", "bc")` hash
identically. So the hash formulas in this document are notation, **not an
implementation instruction** — `sha256(x + y + z)` means "hash a canonical
tagged encoding of these fields", not string concatenation. Serialize as tagged
records (named fields, or length-prefixed), with tuples ordered canonically and
their ordering rule stated per field.

### What the hash cannot cover

The hash identifies a configuration, and a configuration is expressed in
**data** — names, values, flags, arrays. It cannot cover **code**. If someone
edits the body of `GLP`, of a `jac_fun`, or of the evaluator itself, every
archived fit silently becomes stale and no hash detects it. `trspecfit_version`
is the only proxy, and it is coarse — it moves on releases, not on local edits.

This is a boundary, not a gap awaiting a patch, and it is worth stating once
because a large family of "the hash is still incomplete" observations reduce to
it. Keying `jac_fun` by qualified name distinguishes *which* Jacobian was
applied but not a changed implementation of the same one; keying
`model_structure` distinguishes *which* submodels but not a rewritten function
body. Both stop at the same line, deliberately.

It also bounds what any "model configuration manifest" could have achieved,
which is part of why one was rejected: no enumeration of attributes closes a
hole that is fundamentally about source code.

**Outside the identity guarantee.** The guarantee holds for the **supported
package workflow**: models built through the public API, evaluated by the
package's own backends, using the package's own Jacobians. A qualified name
distinguishes those adequately — `fitlib.jacobian_fun` from
`fitlib.jacobian_fun_project` from none at all. It does not extend to:

- arbitrary caller-supplied callables passed through `**fit_wrapper_kwargs`
- **closures and lambdas**, whose `__qualname__` is shared across instances
  carrying different captured state — `make_jac(1.0)` and `make_jac(2.0)` both
  report `make_jac.<locals>.jac`, and every lambda reports `<lambda>`, so two
  genuinely different functions collide within a single session
- monkey-patched package functions
- forked or locally modified evaluator implementations

None of these are defects awaiting a fix. They are the boundary of what a hash
over data can assert, and stating it explicitly is what keeps "the hash is still
incomplete" from being an inexhaustible objection.

Worth considering as a small honesty measure, on the same principle that makes a
schema-7 joint projection self-describing rather than misrepresented: record a
Jacobian that is not one of the package's own as `custom:<qualname>`, so a slot
that falls outside the guarantee says so in the data rather than only in this
document.

### A fit may have more than one input file

A fit is not inherently owned by one file — project-level joint fits already
demonstrate otherwise. The hash therefore takes a **tuple** of inputs, with one
entry in the normal case and N for a joint fit. This closes the generalization
independently of the joint record's payload, which is specified under "Layout:
joint records are a sidecar".

**Identity is two-level.** All slots produced by one joint optimization carry
the *identical* `input_files` tuple — not just their own entry. That is what
stops a joint fit's projection onto file A from colliding with an independent
fit of A using the same model and settings: same file, same table, same
algorithm, and without the full tuple, the same hash and a silent overwrite.

But identical hash inputs across siblings would also give them identical
handles, so:

| Level | Scope | Role |
|---|---|---|
| `optimization_hash` | shared by all siblings | identity of the optimization; what `joint_ref` points at |
| `slot_handle` | one per slot | `optimization_hash` + the file this slot projects onto |

For a single-file fit the two collapse, so the common case is unchanged.
Sibling identity — which the current archive represents nowhere — falls out
for free: slots sharing an `optimization_hash` are products of one fit, by
construction.

**The combined parameter table is hashed, never the per-file projection.**
The hash describes the optimization, and the optimization is joint. This also
keys the shared-parameter map for free: `_build_fit_params` keeps the local
name for `vary_level: project` parameters
([trspecfit.py:1156-1157](../../src/trspecfit/trspecfit.py#L1156)) and remaps
file-level ones per file, so a parameter shared at project level is *one*
combined parameter while the same parameter at file level is *N*. Different
sharing map → different parameter count and names → different hash. Hashing
the per-file projection instead would let two genuinely different joint fits
collide.

**Why `scope` is in the tuple.** For N > 1 it is redundant. It exists for the
degenerate case: a one-file project whose parameters are all
`vary_level: project` produces a combined table identical to the single-file
one, so `Project.fit_2d` and `File.fit_2d` would otherwise hash the same. The
numbers genuinely are identical there — this is a structural guard, not a
numerical one. A slot carrying a `joint_ref` points at shared covariance that
an independent fit has no equivalent of, and overwriting one with the other
would destroy that relationship. Recorded as a deliberate exception to the
"key only what changes the numbers" rule rather than smuggled in.

### The handle is stored, not recomputed

The computed handle is persisted and authoritative; the reader does not
recompute it from the archive's contents.

This continues existing practice for the authoritative key — `archive_slot_key`
is already stored and is what the writer's collision check compares against.
What disappears is the `history_key` recomputation, which existed only because
the in-memory identity token (multi-sha fingerprint) differed from the on-disk
one (positional group path). Principle 1 collapsed those into one primitive, so
there is no longer a second value to derive.

Recomputation would buy read-time verification of identity against contents. It
would cost three things:

- **Every hash input becomes a permanently required stored field**, including
  ones used only at fit time.
- **The hash algorithm becomes part of the wire format.** The reader would have
  to reproduce bit-for-bit the conditional-inclusion rules (`fit_alg_2` only
  when `stages == 2`, `jac_fun` only when applied, `seed` only when supplied),
  the float quantization precision, every canonical ordering, and the
  model-order-not-sorted rule. Changing any of them — including fixing a bug in
  one — would make every existing archive fail verification.
(A third argument — that joint fits would have to duplicate the combined
parameter table into each projection — applied while the joint sidecar was
deferred. The sidecar is now part of schema 7 and stores that table, so the
argument is void. The two above are sufficient on their own.)

The trade is that identity is asserted rather than derived, so a writer bug
producing wrong handles is not caught at read time. That bug class is
*systematic* rather than per-slot — a hash bug affects every fit — so a
round-trip test asserting "same configuration → same handle" catches it. Far
easier to detect than data-dependent corruption.

The decisive argument is that the algorithm is not stable: its inputs changed
roughly eight times over the course of writing this document. Freezing an
unstable algorithm into the wire format, where every later refinement becomes a
breaking change, is the mistake that produced the schema 2→6 accretion this
rewrite exists to escape. Storing also makes the archive robust to our own
churn — an archive keeps its handles across a hash change, showing old and new
fits of one configuration as two variants rather than failing verification
wholesale.

**Verification belongs in tests, not at read time.** Recomputation can be
reconsidered once the inputs have stabilized — post-1.0 is the natural point —
at which stage it becomes an additive integrity check rather than a constraint
on the format.

### Layout: joint records are a sidecar

Slots stay under their file group. A joint fit writes an ordinary slot into
each participating file's `slots/`, each carrying a `joint_ref` to a small
shared record holding the joint parameters, correlation, MCMC, and settings:

```
project/
├── files/000000/slots/000000/     ← A's projection, joint_ref → joint/000000
├── files/000001/slots/000000/     ← B's projection, joint_ref → joint/000000
└── joint/000000/                  ← shared params, correl, mcmc, settings
```

Rejected alternatives, both of which put per-file records somewhere other than
the file group: nesting them inside a project slot, and moving *all* slots to
a flat top-level collection. Both make `files/A/slots/` stop meaning "A's
fits", which is a quiet trap for anything reading the HDF5 directly, and both
force every file-oriented query — `files()`, `select=`, `compare_models`, the
variant table, export — to traverse two locations forever.

The joint record is **represented in schema 7**, not reserved for a later
version. Schema 7 **consumes** a first-class joint result; it does not create
one. Producing that record was sequenced ahead of the conversion (see
"Sequencing") and landed on branch `joint-fit-result` (2026-07-31, decisions
in [joint_fit_result.md](joint_fit_result.md)): every `Project.fit_2d` now
captures a `JointFitResult` — combined parameter table, per-file parameter
maps, joint `conf_ci`/`correl` (correlation over covariance), joint MCMC, and
whole-objective metrics — into `Project._joint_fit_history`, published
together with the per-file projection slots as one bundle.

Designing the on-disk shape while that in-memory object was still transient was
the wrong order, and an earlier draft of this document did exactly that. The
layout below is what the archive stores *given* a settled joint record; the
prerequisite branch decides what that record contains.

Reserving the group name without populating it would have bought nothing: HDF5
does not enforce layout reservations, so a later addition is no cheaper for
having been named. Implementing it now also *removes* complexity rather than
adding it — the partial-diff caveat, the "combined parameters unavailable"
marker, and the differing-hash-empty-diff invariant all exist only because the
combined table was unstored — and it avoids breaking the format twice in
succession for users who adopt schema 7.

### Ownership is two-level; mutation is one transaction

These answer different questions and both are needed.

**Ownership.** The joint record owns shared state; projections own per-file
state:

| Level | Owns | Attachments it may hold |
|---|---|---|
| Joint record | combined initial/final parameter table, the sharing map (per-projection combined → local parameter maps), joint correlation, joint CI, joint MCMC, joint optimizer settings, the whole-objective metrics, the projection reference list | `conf_ci`, `correl`, `mcmc` |
| Projection | `observed`, `fit`, `fit_ini`, `components`, `dark`, `calibration`, residual metrics | σ / noise metadata (per file) |

`correl` is a **correlation** matrix, not a covariance matrix — that is what
schema 6 stores and what `correl_to_df` produces. The joint-record branch
settled the covariance question
([joint_fit_result.md](joint_fit_result.md)): correlation only — with
`stderr` in the parameter table it recovers covariance as
`correl(i,j) · stderr(i) · stderr(j)`, and storing both would invite
disagreement.

The payloads are disjoint, which is what makes "all or nothing" well defined. A
projection has no `correl`, `conf_ci`, or `mcmc` to conflict over — those are
not merely absent, they are **not theirs to hold**. Without single ownership, saving
A's projection in one session and B's in another could leave two chains both
claiming to be the same posterior, free to diverge.

**Projection parameter tables are materialized views**, not independently owned
state. Each is the joint combined result projected through the sharing map. They
must agree with the joint mapping, are validated against it, and can never be
overwritten independently.

That validation is only possible because the sharing map is stored. Combined
parameter names encode Project.files *position*
(`f"file{file_idx:02d}_{local_name}"`), not file name, so sorting projections
canonically by name would sever the association — the fifth instance of the
composite-key rule above, this time discarding association. Each projection
record therefore carries its own combined → local `parameter_map`; readers
look names up rather than parse the prefix convention
([joint_fit_result.md](joint_fit_result.md)), and a project-shared parameter
appears under the same unprefixed name in every projection's map.

**Mutation.** One transaction over the whole bundle: validate the joint record
and every projection first, then enrich or overwrite all of them or none. A
projection must not make an independent overwrite decision that leaves half a
joint fit updated.

This extends an existing discipline rather than inventing one —
`_precheck_slot_collisions` already detects every collision before any mutation
"so a single conflicting slot does not leave half the payload written."

The guarantee is **logical atomicity**: it protects against conflicts and
validation failures, not against process interruption, disk failure, or an HDF5
exception mid-write. HDF5 offers no transactions. Create-then-relink would
reduce the exposure window but is not full crash atomicity either; that needs a
staging-and-recovery protocol or whole-file replacement, and neither is in scope
here.

### The bundle is atomic for save, remove, and prune

Selecting any projection expands to **every sibling declared by the joint
record**, resolved from captured history rather than live project state.
Removing a `File` from `Project.files` is harmless — the captured `SavedFile`
and projection survive independently (Principle 4).

**Every declared projection reference must resolve, or saving fails an integrity
check.** Partial joint bundles are not representable. This is not merely tidy:
the comparability tuple below is computed on read from the projections, so an
unresolved reference would make it uncomputable. Allowing incomplete bundles
would force the joint record to store the view mapping itself plus an explicit
incomplete-state marker — strictly more machinery for a worse guarantee. A
salvage mode could be added later as a recovery tool; it is not normal behavior.

Save and remove differ in how they expand, and deliberately:

| Operation | On a joint projection | Why |
|---|---|---|
| `save_fits(select=...)` | **expands silently** to the whole bundle | The surprise is benign — a superset, nothing lost |
| `drop()` / prune | **raises**, naming the joint handle and sibling count | Expanding a delete past what was asked is the "silently discard someone's work" failure |

`joint_ref` is **mandatory whenever `input_files.scope == "project"`**,
including a degenerate one-file project fit. That gives a checkable invariant —
`scope == "project"` ⟺ `joint_ref` present — and is what finally gives the
`scope` tag a job beyond disambiguation.

### Comparability for joint records

A joint record spans N views, so its comparability key is a **canonical sorted
tuple of `(file_name, fit_view_sha256)` pairs** — never a set of hashes. A set
discards file association and multiplicity: two byte-identical files under
different names produce equal view hashes, so a set would collapse them and make
a two-file joint fit indistinguishable from a one-file one.

No new stored field is needed. Each projection already stores its own
`fit_view_sha256`, and the joint record's reference list supplies the names, so
the tuple is computed on read. A derived `joint_view_sha256` would be pure
convenience.

- **Co-location enables a real comparison — for the metrics that decompose.**
  An independent fit of A and a joint projection onto A share the same
  `fit_view_sha256`, so they land in the same comparability group
  (Principle 6). But only residual-based metrics are meaningful across them:

  | Comparable | Not comparable |
  |---|---|
  | `chi2_raw`, `chi2`, `r2` | `chi2_red_raw`, `chi2_red`, `aic`, `bic` |

  The right-hand column depends on a parameter count, and a joint fit's
  parameter count does not decompose per file — a shared parameter is
  constrained by every participating file's data, so no share of it can be
  attributed to A. Same reason the joint covariance does not decompose.
  Per-file degrees-of-freedom metrics are therefore **not defined for a joint
  projection and are not stored**; the joint record carries AIC/BIC over the
  whole dataset.

  Worth flagging for implementation: three of the four default comparison
  metrics (`DEFAULT_METRICS_NO_SIGMA`,
  [fit_results.py:52](../../src/trspecfit/fit_results.py#L52)) are in the
  non-comparable column, so the default `compare_models` output would be
  misleading for joint projections unless those columns are omitted for them.

**Labels live on the joint record**, resolved through `joint_ref` for projection
display. A label is mutable, and duplicating a mutable field into N projections
means N places to update and N chances to diverge. It also names a *result*, and
the joint optimization is the result — labelling half a fit differently from the
other half is meaningless.

**Joint MCMC is a producer capability, not a wire-format rule.** `mcmc/` is
optional everywhere; schema 7 persists a joint chain whenever one was produced.
The JAX backend's limitation is specifically on **parallel-worker** MCMC — not
on MCMC as such, and the interpreter path produces a full posterior end to end.
Either way the effective-backend field records which path ran, so the format
needs no special case and no capability statement of its own.

Consequences elsewhere: `fit_view_sha256` is per input file rather than a
single value. Filtering stays **projection-based** — `file="A"` selects slots
projected onto A (`slot.file_name == "A"`), exactly as today. It must *not*
become a containment test over `input_files`, or asking for A would also
return B's projection of any joint fit involving both. A separate
`involves_file=` query can be added later if the containment semantics turn
out to be wanted.

### A slot is a configuration, not an execution

**A slot identifies an optimizer configuration, not an execution event.**
Repeating the same deterministic configuration is not archived as additional
history.

This has to be stated explicitly rather than assumed, because "same inputs →
same result" is not true in general. `fit_alg_1` / `fit_alg_2` are free-form
strings passed straight to `mini.minimize(method=...)`
([fitlib.py:799](../../src/trspecfit/fitlib.py#L799)), with no whitelist, so
stochastic global optimizers (`differential_evolution`, `basinhopping`,
`dual_annealing`, `ampgo`) are all reachable.

**A random seed is a keyed input when the user supplies one.** `seed` is an
optional passthrough: forwarded to the stage-1 optimizer if given, part of
the key if given, absent otherwise. If it is absent the user has not asked
for reproducibility, and the collision rules below handle the consequences.

No capability table of which methods accept a seed is maintained. SciPy
already enforces that, its knowledge is always current with the installed
version, and a hand-written table would go stale on every release. Verified
against lmfit 1.3.4 / SciPy 1.17.0: `leastsq` and `nelder` raise
`TypeError: unexpected keyword argument 'seed'`, `differential_evolution`
accepts it. Letting the library produce that error is correct behavior.

**Landed 2026-08-14**: `fit_wrapper` accepts `seed`, `_method_kws` forwards
it to the `fit_alg_1` stage only — the two-stage contract designates stage 2
as deterministic refinement, though `fit_alg_2` stays free-form: a
stochastic second stage stays unseeded and is surfaced by the collision
rules, never silently — and `build_fit_settings` records a `seed` field only
when supplied. Every fit API inherits the kwarg through its
`**fit_wrapper_kwargs` passthrough (including SbS, whose `seed_source` /
`seed_values` knobs choose initial *parameter values*, a different thing
from the RNG state). When divergence is detected on runs that already carry
a seed, the collision message points at a stochastic `fit_alg_2` instead of
re-recommending the seed the user already supplied.

The unseeded stochastic case is then handled by the collision rules rather
than by infrastructure:

| scenario | input hash | fitted params | outcome |
|---|---|---|---|
| `leastsq`, re-run | same | same | enrich / no-op |
| `differential_evolution` unseeded, re-run | same | differ | conflict, user warned |
| `differential_evolution` `seed=42`, re-run | same | same | enrich / no-op |
| unseeded, then `seed=42` | differ | — | two slots |

**A collision is not automatically an overwrite.**

| input hash | fitted params | attachments | behavior |
|---|---|---|---|
| same | same (within tolerance) | absent in archive, present incoming | enrich in place |
| same | same | present in archive, absent incoming | no change — never delete |
| same | same | present in both | requires `overwrite=True` |
| same | **differ** | any | **hard conflict**; requires `overwrite=True` |
| differ | — | — | new slot |

Only **absent → present** is free. Richness is deliberately not a criterion:
comparing a 5000-step chain against a 1000-step one at a better acceptance
fraction has no well-defined answer, so any present → different-present is a
replacement requiring opt-in. Attachments merge individually, so `conf_ci` can
enrich in the same write that leaves `mcmc/` untouched.

The never-delete row matters because the optimizer result is unchanged: an
archived chain is still a valid posterior for that fit, so re-saving the same
configuration without having re-run MCMC must not drop it. Under the diverging-
parameters row the opposite holds — `overwrite=True` there replaces the whole
slot including its attachments, because a chain computed for a different optimum
is stale.

**Why the parameter comparison exists.** Not as a reproducibility feature —
it is what makes enrichment safe. Attaching MCMC to an existing slot without
demanding `overwrite=True` means letting some collisions through silently,
and that is only safe if the underlying optimizer result is genuinely
unchanged. Otherwise a divergent refit followed by an MCMC attach would
silently replace the fitted parameters under cover of adding a chain. The
alternative — no enrichment, every attach requires `overwrite=True` — is
worse, because `overwrite=True` also authorizes replacing the optimizer
result, so it hands over a broader permission than the user intends.

No new stored field is needed — `params` already carries the `value` column.
Only `params` is compared, never `fit`: the curve is derived from the
parameters by the same evaluator, and the backend is keyed.

**Equivalence is defined, not "loose".** Parameters in one fit span wildly
different scales — an amplitude of 1e5 beside a width of 0.1 — so a
meaningful difference need not be large in relative terms, and a single
hand-waved tolerance is not good enough. Compare **per parameter, matched by
name**, with numpy semantics `|a − b| <= atol + rtol·|b|`. Names and order are
guaranteed identical when the hash matches, but matching by name rather than
position makes that explicit rather than incidental.

Starting values: `rtol=1e-6`, `atol=1e-12`, as **named module constants**, not
literals at the call site. The `atol` term is not optional — a parameter
converging to `+1e-15` in one run and `−1e-15` in another differs by 200%
relatively and by nothing that matters. This is the one number in the design
chosen rather than derived, so it lives in one place and is tunable with
evidence.

**One rule, both boundaries.** Two fits with the same hash can meet in two
places: at in-session collapse, and at the archive. `Project._fit_history` is
append-only, so both sit in the list, and `collapse_history_to_snapshot`
reduces to one per key *before* the writer ever runs. Until v0.14 that
reduction was a bare dict overwrite with no comparison and no warning: an
unseeded stochastic fit run twice and saved once lost the first result
silently, while the same two runs with a save in between hit a conflict —
identical user behavior, surfaced only if they happened to save in the
middle.

Collapse applies the same rule as the archive (**landed 2026-08-14**):
**divergent parameters under one hash raise; `overwrite=True` selects the
latest and reports the replacement.** Since collapse runs inside
`save_fits`, both raises come from the same call — one rule, not two that
happen to agree.

Two things make the raise the right default rather than a burden:

- **Nothing is recomputed.** `overwrite=` is an argument to `save_fits`, not
  to the fit. Both results are already in `_fit_history`, MCMC attachments and
  all, so resolving a conflict is a re-call, never a re-fit.
- **The error must name the cause and the fix.** Divergence under one hash
  means the configuration was non-deterministic, so the message should say
  that and point at pinning a seed — not merely offer `overwrite=True`. A
  deliberate seed-sensitivity study varies the seed, which makes the runs
  distinct configurations that never collide; the raise fires only when
  intent and configuration disagree.

This is a pre-existing silent-loss path rather than something the new design
introduces. The complete input hash fixes most instances of it (different
algorithms now hash apart); the shared rule closes the rest.

The model stays self-consistent under pressure: if someone genuinely wants two
retained runs of the same configuration, the honest way to express that is to
vary the seed — which makes them different configurations.

Consequences:

- **Model names stay part of structural identity, but are no longer the only
  protection against incompatible reuse.** They enter the hash via
  `model_structure`, and two same-named models with different contents now also
  hash apart through the parameter table. Name uniqueness within a `File`
  remains worth guarding for usability (`select_model` by name), but
  correctness no longer rests on it alone.
- **File identity folds in.** The file version stamp sits *inside* the fit
  hash, so correcting `dark` and refitting yields a new slot instead of a
  staleness conflict, and the pre-correction result survives as the record of
  what you had before.
- **Collapse-on-save stops hiding the variants.** The collapse *rule* is
  unchanged — latest-per-key — but the key is finer, so collapse now removes
  only literal re-runs. The variants were always there:
  `Project._fit_history` is append-only in-session, and saving is what
  discarded them. Principle 3 does not create a mess; it stops suppressing
  one. The response therefore belongs in the query layer (Principle 6), not in
  the storage rule.

### Keyed inputs vs post-hoc attachments

The key covers everything that determines the **fitted parameter values**.
Analyses layered on an unchanged optimum are **attachments** to a fit, not
distinct fits.

σ is purely post-hoc: `compute_fit_metrics` divides `chi2_raw` by
`sigma_eff²` ([fitlib.py:132-139](../../src/trspecfit/fitlib.py#L132)) and
nothing passes weights to the minimizer. Setting σ and refitting therefore
produces byte-identical `fit`, `params`, and `components`, differing only by
a scalar divisor — and `chi2_raw` / `chi2_red_raw` are stored
unconditionally, so not keying on σ loses nothing. `try_ci` and MCMC are the
same shape: keying on them would produce sibling slots where one is strictly
a superset of the other, and would fork a slot every time MCMC is re-run with
a different random draw.

| Keyed | Attachment (supplements an existing slot) |
|---|---|
| file version stamp (data + dark + calibration) | σ / noise metadata |
| `fit_type` + selection / fit limits | `try_ci` → `conf_ci` |
| shared parameter metadata + initial-state matrix | MCMC (chain, CI, acceptance) |
| `model_structure` (per-attachment records, incl. each `frequency`) | |
| optimizer settings actually in force | |

SbS `seed_source` / `seed_adapt` / `seed_values` are in **neither** column:
they are provenance in `fit_settings`, because the initial state they produce
is hashed directly instead.

Consequence for the writer: **a slot collision whose fitted parameters agree
is an enrichment path, not an error.** Running a fit, saving, then later
running MCMC on the same optimizer result attaches to the existing slot
rather than raising `FileExistsError` or creating a sibling. A collision whose
fitted parameters *disagree* is a hard conflict — see the collision rules
under Principle 3.

Attachment replacement is the one case that can silently destroy work — re-running
MCMC with fewer steps would overwrite a longer chain. Attachment replacement
should therefore obey the same `overwrite=` guard as slot replacement:
enriching an empty attachment is free, replacing a populated one requires
opting in.

### The initial state is the identity input, not the mechanism that produced it

All fit types are hashed the same way: **shared per-parameter metadata plus an
ordered initial-value matrix.** For baseline, spectrum, and 2D that matrix has
one row; for SbS it has one row per slice (`params_meta` + `params_init`, both
already persisted since schema 6). This is the second place the design uses a
tuple-with-one-entry-in-the-common-case shape, after `input_files`.
(`model_structure` is *not* an instance — its outer shape is
`(energy_models, dynamics_attachments)`, a pair of collections rather than a
one-or-N tuple.)

Consequently `seed_source`, `seed_adapt`, and `seed_values` are **not** hashed.
They remain in `fit_settings` as provenance. Hashing the mechanism rather than
its effect misses two real cases:

- **`seed_source="baseline"`** derives seeds from the baseline fit's output.
  Refit the baseline, re-run SbS with the same `seed_source`, and the seeds
  change while the mechanism does not — mechanism-hashing would call two
  genuinely different fits one configuration.
- **`seed_adapt="argmax_shift"`** depends on `data_base`, hence on
  `base_t_ind`, which is *not* part of the SbS selection (`{e_lim, t_lim}`).
  Redefining the baseline changes every seed invisibly to the mechanism.

The converse case is correct rather than a loss: if two seed mechanisms produce
byte-identical initial states, the optimizer cannot distinguish them, so
treating them as one configuration is right. The only cost is mild provenance
ambiguity — the surviving slot records one mechanism's `fit_settings` (latest
wins) when both collapse together.

There is **no cross-slice warm start**
([trspecfit.py:3000](../../src/trspecfit/trspecfit.py#L3000)); each slice's
seed is a deterministic function of the base seed and that slice's data, so the
hashed initial state is stable across identical re-runs rather than drifting.

**Quantize the initial-state matrix before hashing.** Hashing raw `tobytes()`
is the obvious implementation and the brittle one: these seeds are frequently
*derived* rather than typed (a `seed_source="baseline"` seed is float64
optimizer output; `argmax_shift` adds data-dependent arithmetic), so a last-bit
difference would change the hash and mint a spurious slot instead of yielding a
comparison. Mechanism-hashing had a cushion here that effect-hashing does not,
and quantization restores it.

**Scope: quantize the initial-state matrix, regardless of any individual
entry's origin; hash every other numerical input exactly.** The scope is
structural — one named input — not a per-value judgement about whether the
package computed it. The *reason* the matrix is the one that needs it is that it
is the only hashed input that can be package-computed; but applying it entry by
entry would be both fiddly and pointless, since quantization is a no-op for a
YAML-typed seed like `1.5`.

| Hashed input | Origin | Treatment |
|---|---|---|
| `data_raw`, `energy`, `time`, `aux_axis`, `dark`, `calibration` | user-supplied arrays | exact bytes |
| bounds (`min` / `max`), `frequency`, `time_point` / `time_range` | user-supplied scalars | exact |
| `e_lim` / `t_lim` / `base_t_ind`, `stages`, `seed` | integers | exact |
| algorithm names, `model_structure` names, `jac_fun` qualname | strings | exact |
| **initial-state matrix** | may be computed | **quantized** |

Quantizing the data or correction arrays would not merely be unnecessary but
**wrong** — two genuinely different datasets could collide.

**Method.** Round each value to 9 significant decimal digits — scale-invariant,
so it behaves identically for an amplitude of 1e5 and a width of 1e-3, unlike
fixed decimal places — and hash the resulting canonical decimal *text* rather
than re-parsed float bytes. Normalize `-0.0` to `0.0`, or the two hash apart
despite being equal. The digit count is a named constant in one place.

Because the handle is **stored rather than recomputed** on read, this algorithm
is not wire format. It must be deterministic within a version, not
interoperable across them — and changing the digit count later shows old and new
fits of one configuration as two variants rather than invalidating archives.
This is consistent with the `rtol` / `atol` policy adopted for parameter
comparison, which is the same idea applied to a different question.

## Principle 4 — mutable inputs are snapshotted per fit, immutable content is stored once

This is Principle 2 row 2, applied to the file's arrays. It replaces the idea
of keeping a *correction history* on the file: you do not need a history, you
need each fit to carry the correction that was in force when it ran. That
yields every correction that ever mattered — every one that produced a result
— with no separate mechanism. Corrections applied and then changed without
fitting are not interesting.

| Array | Mutability | Home |
|---|---|---|
| `data_raw`, `energy`, `time`, `aux_axis` | immutable after construction | file, one copy |
| `dark`, `calibration` | mutable, entered the numbers | **slot**, per fit |
| σ (`sigma_data`, `sigma_eff`, noise metadata) | mutable, entered the numbers | **slot**, per fit (already the case) |
| `data` (corrected) | derived *and* mutable | **nowhere** — reconstructed on demand |

Cost is negligible: `dark` and `calibration` are 1D of length `n_energy`
(~16 KB per slot at a 1000-point axis) against `observed` / `fit` /
`components`, which dominate slot size.

`data = (data_raw - dark) / calibration`, so with per-slot corrections the
corrected array is fully reconstructible. Keeping it at file level stores
versioned state at an immutable level — the same category error as the
file-level `e_lim` / `t_lim` that this redesign removes.

### The archive owns its arrays

`SavedFile` and `SavedFitSlot` are documented as immutable snapshot records,
but today they store arrays **by reference** — `frozen=True` blocks
reassignment of the field, not mutation of the object it points at. A record
holding a reference into a live object's array is a *view*, not a snapshot.

So the archive copies at capture. This is an **ownership boundary, not
defensive machinery**, and it is what makes these records snapshots at all:

1. **Copy once when the first slot captures `SavedFile`** — `data_raw`,
   `energy`, `time`, `aux_axis`.
2. **Make the captured arrays read-only.**
3. **Copy the mutable per-fit arrays into each slot** — `dark`,
   `calibration`, `observed`.

`dark` and `calibration` are mutable *by design* (`subtract_dark()` and
`calibrate_data()` exist to change them), so nothing upstream will ever freeze
them; copying them is simply what capturing a changing value means, the same
reason σ is copied. `observed` needs the same treatment if it is currently a
view into `File.data` rather than a materialized array — persisting it as
stored bytes resolves that at capture.

**Why not delegate this to the systemic mutation policy.** That was considered
and rejected. The argument for delegating was "if the arrays are immutable,
sharing by reference is correct" — but `setflags(write=False)` is advisory,
can be flipped back, and does not prevent rebinding, so a read-only numpy
array is not immutable the way a frozen value is. More decisively, the two
choices are not equally reversible: copy-now-relax-later is a safe local
change if the systemic policy ever makes the copies provably redundant, while
share-now-tighten-later is a correctness bug for the whole interim.

The broader "guard against in-place mutation of user-facing arrays" TODO
remains worthwhile on its own merits — it governs what happens when a user
mutates `file.data` and then plots or refits — but the archive does **not**
depend on it, and schema 7 does not wait for it. That TODO lists four
candidate mechanisms and no decision; blocking a wire format on an unresolved
systemic redesign trades one open problem for another.

### Reconstructed vs stored

Reconstructibility is not on its own a reason to drop something. The two
cases resolve in opposite directions:

- **The corrected `data` array is dropped and reconstructed.** It is context,
  it is versioned, and nothing scientific depends on the exact bytes.
- **Per-slot `observed` is kept as stored bytes.** It is the scientific record
  of what was actually fit.

The reason is not size. Dropping `observed` saves only 14–33% of a slot's
array payload, because `fit`, `fit_ini`, and `components` are irreducible
without model rehydration:

| fit type | stored arrays | `observed` share |
|---|---|---|
| baseline / spectrum | `observed`, `fit`, `fit_ini`, `components` (n_comp × n_e) | 1/(3 + n_comp) → 14% at 4 components |
| sbs | same, × n_slices | 14% |
| 2d | `observed`, `fit`, `fit_ini` (no components) | 33% |

The reason is **stability**. If `observed` had to be reconstructed, the
archive's meaning would depend on the package's current view-derivation code
— baseline averaging over `base_t_ind`, `resolve_time_selection` semantics,
correction order — so a later refactor would silently reinterpret every
existing archive. Stored bytes do not have that property. The existing
`observed_sha256` field — replaced by `fit_view_sha256`, Principle 6 — is an
admission of exactly this risk: the schema describes it as guarding "against
silent grid drift *if `selection` ever fails to capture a view detail*". Under
reconstruction that check becomes one that can fail with no recovery, because
the bytes it guarded are gone.
Reconstruction is also hostile to external consumers reading the HDF5 from
MATLAB or Julia.

A reconstruction helper is still required — `full_range=True` rendering has
to show corrected data *outside* the fit window, where nothing is stored. It
belongs in `utils/`, imported by both `fit_results` and `trspecfit` rather
than owned by either. There it reconstructs *context*, never the scientific
record.

### Compression

Datasets are currently written uncompressed. Lossless compression
(gzip + shuffle) typically gets 2–4× on smooth float64 spectra, preserves
dtype exactly, and leaves every content hash and version stamp valid. It is the
cheapest
response to the extra slots Principle 3 admits, and it is orthogonal to the
data model.

## Principle 5 — provenance is persisted even when it is not identity

Identity answers "is this the same thing". Provenance answers "what was this".
They are different, and the second is not subordinate to the first.

Persist the **model YAML snippet(s)** attached to each fit — energy, time,
and profile entries — as slot provenance. It is what makes a slot
human-readable without a live session. It is simply not the identity
primitive.

It is also **not** a rehydration record, which an earlier draft of this
principle overstated. Model construction is partly programmatic:
`add_time_dependence()` supplies the subcycle ordering and `frequency`,
`add_par_profile()` attaches profiles, and `update_value()` can move a seed —
none of which appear in any YAML. For a multi-cycle model the snippets yield
several independent top-level models with no indication that they form a cycle
sequence or at what rate. The hash carries what determines the numbers
(Principle 3); the snippets carry human-readable declaration only.

## Principle 6 — variants are viewed, not suppressed

Principle 3 makes many slots per `(file, model, fit_type)` the normal state.
The gap this opens is **not** that the query layer rejects them — it does
not — but that its output cannot tell them apart: `compare_models` keys rows
on `(file, model, fit_type, selection_json)`, and two variants differing only
in a `vary` flag or a bound share all four. They come back as visually
identical rows with different metrics and no column explaining why. The
response belongs in the query layer.

The framing matters: this is a **view problem, not a pruning problem.** The
urge to prune comes from fear of clutter, and clutter is only frightening
when there is no way to look at it. A lab notebook is indexed, not pruned.
With compression on, ten variants cost almost nothing; what costs the user is
being unable to see what distinguishes them.

### The diff is computable, not declared

Generic experiment trackers log whatever the user declares, so they can only
diff opaque key-value bags. Here the key is a *structured* input tuple — file
version, selection, parameter table, algorithm — so what differs between any
two slots is computable exactly, from data already stored.

Pairwise, that is a diff of inputs against outputs:

```
a3f2 → 91cd
  inputs
    GLP_01_A.vary       False → True
    e_lim               [100, 400] → [80, 420]
  outputs
    chi2_red_raw        2.41 → 1.08
    r2                  0.981 → 0.997
    n_varying           7 → 8
```

**Joint fits diff at the bundle level.** Because the joint record stores the
combined parameter table (Principle 3), diffs across joint fits are complete —
a difference in shared parameter state is visible, not merely implied by
differing `optimization_hash` values. Diff two joint fits by their joint
records, not by a pair of projections: a projection's parameter table is a
materialized view of the combined result, so diffing projections alone would
show a shadow of the real difference.

Across a set, the same primitive gives a variant table with **columns that
are constant across the set suppressed**, so only what actually varies is
shown:

```
        A.vary   B.min   fit_alg    chi2_red_raw    r2      aic
a3f2     False     0.0   leastsq            2.41   0.981   -412
91cd      True     0.0   leastsq            1.08   0.997   -598
7b1e      True    -0.5   leastsq            1.07   0.997   -596
c204      True     0.0   Nelder             1.11   0.996   -591
```

This is the generalization of `compare_models()` — model name is one more
differing column. As the `repr` of a `FitResults`, it makes the variant set
self-explaining on print.

### Slot handles

A slot is addressed by its **input hash**, git-style. Not a UUID: random means
re-running an identical fit yields a different handle, contradicting
Principle 3. Not a sequence number: it depends on insertion order and shifts
under filtering or archive merges, implying a total order that does not exist.
The hash is stable, content-derived, order-independent, needs no bookkeeping,
and reproduces itself on an identical re-run — which is correct, because it is
the same fit.

**Stored in full; abbreviated only for display and lookup.** The handle is the
complete 64-character hex digest. Truncation is a presentation and addressing
concern, never a storage one:

- **Display** abbreviates to 8 characters, matching the existing export-path
  convention (schema 6 already suffixes colliding export directories with the
  first 8 characters of the history key).
- **Lookup** accepts any prefix, and **raises on an ambiguous one** rather than
  picking a match — the same contract as git.

Storing a truncated digest would be irreversible: if abbreviation ever collided
there would be no way to extend it, whereas an 8-character display width can be
widened freely because the full value is on disk.

That reproducibility holds **within a given hash implementation.** Because
handles are stored rather than recomputed (Principle 3), a later change to the
hash inputs or the quantization constant leaves existing archives untouched, so
an old slot and a new run of the same configuration will not share a handle and
will appear as two variants. Untidy, self-correcting, and much preferable to
invalidating archives — but it means "identical configuration → identical
handle" is a statement about one version, not across them.

### Labels

Optional, settable at any time, never required and never prompted for.
Requiring a label at fit time is both a barrier and wrong on the merits — you
rarely know at fit time which run mattered. Interactive prompting at export
is also rejected: it breaks scripts, headless runs, and CI, and
`show_output=0` API mode is first-class in this package. A label is set
post-hoc (`results.label("a3f2", "final")`) and used for display, export
directory names, and selection.

Labels are also what make a **durable** pointer to a chosen fit. An explicit
`accept` marker was considered and rejected: its only advantage over
selection-at-point-of-use is persisting a decision, which a label already
does, and it would add persistent state with its own questions (survival
across save/load and merge, behavior when the accepted slot is dropped).
"Accept" also reads naturally as "discard the rest", which it would not do.

### Comparability

Two slots are directly comparable only if they were fit against the same
fit view, which `fit_view_sha256` answers exactly. Slots with different views
(different fit limits, different corrections) are not comparable, and their
metrics must not be ranked against one another.

**`fit_view_sha256` replaces `observed_sha256`.** The existing field hashes
observed *values* only (`tobytes()` of the array), which is not faithful array
identity — shape is not encoded, so a `(2, 50)` and a `(100,)` array of the
same bytes hash alike — and it says nothing about the axes the model was
evaluated on. The replacement hashes the whole fit view:

- `observed` dtype, shape, and values
- the selected energy coordinates
- the selected time coordinates, where the file has a time axis
- the auxiliary axis, **whenever the file has one**

**The view hash depends on the file and the selection, never on the model.**
The auxiliary axis is included because the *file* has one, not because a
particular model consumes it. Making it conditional on use would give a profile
model and a non-profile model different view hashes for identical observations,
putting them in different comparability groups — breaking precisely the
cross-model comparison this grouping exists to enable. Model differences are
carried by `model_structure` and the parameter table in `optimization_hash`;
they must not leak into a hash whose job is to say "these were fit against the
same data".

This is the opposite of the rule governing optimizer settings, and the
distinction is easy to get backwards:

| Hash | Question | Conditional inclusion |
|---|---|---|
| `optimization_hash` | which computation was this? | **correct** — an unused `fit_alg_2` would mint a false variant |
| `fit_view_sha256` | which data were these fit against? | **wrong** — it splits groups that should compare |

Fit-type dependence is fine and expected: a baseline fit reduces time by
averaging over `base_t_ind` while a spectrum fit selects a point or range, so
their views genuinely differ. That is a property of the selection, not the
model, and `fit_type` is part of the grouping key regardless.

It must hash **the arrays actually used**, not be derived from
`(file_version_stamp, selection)` even though those fully determine it. The
original field existed to catch a mismatch between the derivation and reality
("silent grid drift if `selection` ever fails to capture a view detail"), and a
derived hash cannot do that.

**The guard's logic stays; its grouping key changed.**
`_check_observed_consistency`
([fit_results.py:1287-1322](../../src/trspecfit/fit_results.py#L1287)) already
implements exactly this rule — raise when one group holds more than one view.
It used to group by `(file_fingerprint, file_name, fit_type)`; Principle 1
demoted `file_fingerprint` from identity to a version stamp, so it no longer
belongs in an identity key, and the regrouping to `(file_name, fit_type)`
landed with the identity guards (2026-08-03). What remains for schema 7 is
`observed_sha256` → `fit_view_sha256` doing the data discrimination: group by
identity, discriminate by view. Collapse and prune operations must respect
the same grouping.

**σ does not enter the grouping key.** It gates which *metrics* may be compared
within a group:

| Condition | Comparable |
|---|---|
| same `fit_view_sha256` | raw metrics — `chi2_raw`, `r2` |
| same view **and** same `sigma_eff` | additionally `chi2`, `chi2_red` |
| same view, differing `sigma_eff` | raw only; σ-scaled withheld |

Two distinct slots can share a view and differ in σ precisely because σ is an
attachment rather than a keyed input: fit M1 at σ=1, call `set_sigma(10)`, fit
M2. Ranking those by `chi2` (100 vs 1.2) makes M2 look far better while
`chi2_raw` (100 vs 120) says it is worse. Adding σ to the grouping key would be
the wrong fix — it would split the group and block the raw comparison, which is
perfectly valid.

### Pruning and selection

Explicit only, never automatic: `results.drop([...])` by handle, or a
selection criterion at save/export time. Selection criteria and their
directions:

| Criterion | Direction | Requires σ | Character |
|---|---|---|---|
| `aic`, `bic` | minimize | no | proper model selection |
| `chi2_red` | minimize \|x − 1\| | **yes** | is this fit at the noise floor |
| `chi2_red_raw` | minimize | no | weak complexity penalty, arbitrary scale |

Raw chi-square is not offered: a fit with more free parameters almost always
wins on it while being the worse model. `chi2_red` is offered but **must not
be minimized** — a σ-calibrated fit at the noise floor sits at ≈ 1 and
overfitting drives it *below* 1, so smallest-wins would systematically select
the most overfit variant.

Group semantics: `"latest"` and `"best"` resolve within each
`(file, model, fit_type)` group, never as a single winner across the archive.

### Archive vs export

**The archive is the record; the export is the presentation.** Both take the
same `select=` parameter; the defaults differ:

| | default | rationale |
|---|---|---|
| `save_fits(select=...)` | `"all"` | the archive must never silently discard someone's work |
| `export_fits(select=...)` | `"latest"` | an export is a curated human-facing artifact |

`select=` accepts `"all"`, `"latest"`, `"best"` (with a `by=` criterion), a
slot handle or handle prefix, or a label. The differing defaults are
principled, not inconsistent.

### Rejected: lineage pointers

Parent-slot references (a git-style DAG) were considered and rejected. Fits
are not derived from each other; they are derived from a model state already
captured in the key, and the package cannot observe what the user did between
two fits. Diff-on-demand delivers essentially the same value with no
bookkeeping.

## Sequencing

**The in-memory object model is settled before the wire format, not
alongside it.** Several items that read like schema work are consequences of
the object model — `_files_by_fp` removal, the slot-to-file association fix,
the same-name/different-content integrity rule, and the dual-key collapse.
The joint record is the case that proves the rule: this document initially
tried to design its on-disk shape while the in-memory joint result was still
a transient discarded after `fit_2d()`, which is the wrong order.

1. **Identity in the live object model.** Guard `File.name` on mutation;
   guard model-name uniqueness within a `File`; demote the fingerprint to a
   version stamp; make slot→file lookup a name lookup. No schema change;
   existing archives keep working.
2. **The `PlotConfig` refactor** (project-owned, single config). Independent
   of identity; can proceed in parallel.
3. **First-class joint fit results.** Make the joint result a captured record
   rather than a transient on `Project._project_fit_result`, so the sharing
   map, joint uncertainty, and joint MCMC survive past the `fit_2d()` call —
   including whether that uncertainty is stored as correlation, covariance, or
   both.
   Schema 7 cannot be finalized before this — it would have nothing to
   serialize. **Done** on branch `joint-fit-result` (2026-07-31): the record
   is `JointFitResult` (correlation chosen over covariance), decisions in
   [joint_fit_result.md](joint_fit_result.md).
4. **Schema 7.** With the object model settled, the wire format largely falls
   out. Array ownership (Principle 4) is local to the capture path and ships
   with it — it does **not** wait on the systemic array-mutation TODO, which
   stays an independent quality item.

## Evidence

Empirical checks run against the working tree on 2026-07-25 (branch
`fit-archive-schema-7`, package version 0.14.0):

| Claim | Result |
|---|---|
| A data correction changes the file fingerprint | Confirmed — `subtract_dark()` moved `data_sha256` `60ff129e…` → `8be449ca…` |
| `File.name` is unguarded after construction | Confirmed — `f.name = "B"` succeeds; the project's `"A"` lookup then fails |
| Duplicate model names are unguarded | Confirmed — `trspecfit.py:2111` appends unconditionally |
| Per-file `PlotConfig` is unused | Confirmed — no example or `src` customization; four call sites, all tests of the feature itself |
| `SavedFile.e_lim` / `t_lim` have no readers | Confirmed — written and read back into the dataclass, consumed nowhere |
| Fingerprint is never used to match an archive to a live project | Confirmed — all uses are archive-internal or in-session |
| Parameters are never edited programmatically in examples | Confirmed — the example workflow is entirely YAML-driven |
| σ never reaches the minimizer | Confirmed — `compute_fit_metrics` divides `chi2_raw` by `sigma_eff²`; no weights are passed |
| Optimizer choice is unrestricted | Confirmed — `fit_alg_1` / `fit_alg_2` are free-form strings; stochastic global optimizers are reachable |
| No seed reaches any optimizer | Confirmed — `_method_kws` returns `{}` except `Dfun` on `leastsq`; `build_fit_settings` has no seed |
| SciPy rejects `seed` on methods that do not accept it | Confirmed against lmfit 1.3.4 / SciPy 1.17.0 — `leastsq` and `nelder` raise `TypeError`, `differential_evolution` accepts |
| In-session collapse drops divergent re-runs silently | Confirmed — `collapse_history_to_snapshot` is a bare dict overwrite ([fit_io.py:1036](../../src/trspecfit/utils/fit_io.py#L1036)) |
| `compare_models` does *not* reject multiple variants | Corrected — an earlier draft claimed it did. `_check_observed_consistency` raises only on `len(shas) > 1`, i.e. differing data views; many slots sharing one view hash pass |
| A model-conditional view hash breaks cross-model comparison | Corrected — an earlier draft included `aux_axis` only "where it participates in evaluation", which would split a profile and a non-profile model on identical observations into separate comparability groups |
| `observed_sha256` does not encode shape | Confirmed — `compute_observed_sha256` hashes `tobytes()` only, so it is not faithful array identity and covers no axis information. No live counterexample was constructible within a `(file, fit_type)` group; the defect is latent |
| Project-level `vary_level` params keep their local name | Confirmed — `_build_fit_params` sets `proj_name = local_name` ([trspecfit.py:1156](../../src/trspecfit/trspecfit.py#L1156)); file-level params are remapped per file |
| `aux_axis` is excluded from the file fingerprint | Confirmed — `compute_file_fingerprint` hashes only data/energy/time, though `aux_axis` is stored and consumed numerically by profiles |
| Component order changes the numbers | Confirmed — `Shirley` and `LinBack` take the accumulated `spectrum` as an argument ([energy.py:74-87](../../src/trspecfit/functions/energy.py#L74)) |
| Profile attachment is encoded in parameter names | Confirmed — names are `Gauss_01_A_pExpDecay_01_tau` (`test_evaluate_1d.py:288`); `mcp.py:696` requires the profile name to match its target parameter exactly |
| `seed_adapt` alters per-slice starting state | Confirmed — defaults to `"argmax_shift"` ([trspecfit.py:2992](../../src/trspecfit/trspecfit.py#L2992)) |
| SbS has no cross-slice warm start | Corrected — an earlier draft claimed per-slice seeds derive from the previous slice. [trspecfit.py:3000](../../src/trspecfit/trspecfit.py#L3000) states there is no warm start; each seed is a function of the base seed and that slice's data |
| Multi-cycle structure is programmatic, not YAML | Confirmed — subcycle assignment is the list order passed to `add_time_dependence()`; `frequency` is a kwarg ([trspecfit.py:3856](../../src/trspecfit/trspecfit.py#L3856)). The YAML in `03_multi_cycle_dynamics` declares only unrelated top-level models |
| Dynamics attach per parameter, each with its own frequency | Confirmed — `add_time_dependence(target_model, target_parameter, ..., frequency=-1)` ([trspecfit.py:3849](../../src/trspecfit/trspecfit.py#L3849)) attaches one dynamics model to one parameter per call, so `frequency` cannot be a global scalar |
| `frequency` is persisted nowhere | Confirmed — no occurrence in `fit_io.py`; the shipped example fits at `frequency=0.25` |
| A caller can override the backend's Jacobian | Confirmed — both call sites use `setdefault("jac_fun", ...)` ([trspecfit.py:1414](../../src/trspecfit/trspecfit.py#L1414), [:4077](../../src/trspecfit/trspecfit.py#L4077)), so a `jac_fun` passed through `fit_2d`'s `**fit_wrapper_kwargs` wins over the backend default |
| `fit_alg_2` is unused when `stages == 1` | Confirmed — the one-stage branch calls only `fit_alg_1` ([fitlib.py:799](../../src/trspecfit/fitlib.py#L799)), yet `fit_alg_2` still defaults to `"leastsq"` |
| Model names may contain underscores | Confirmed — the no-underscore guard (`test_config_functions.py:116-134`) covers only `functions/` registry names; `parsing.py:130-133` handles underscored component names. Joining a submodel list is therefore lossy |
