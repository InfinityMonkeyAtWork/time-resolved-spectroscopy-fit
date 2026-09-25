---
orphan: true
---

# API and ownership contract

Status: **settled** (2026-09-24). Branch `api-ownership-contract`, step 1 of
the plan recorded in `TODO.md`. The four forks at the end record what was
decided and why. Enforcement is step 4; the per-name tier inventory and the
renames are step 5.

This document answers three questions that the ownership work, the
reconstruction loader, and the v1.0 stability promise all depend on:

1. What counts as **public**?
2. Who **owns** each piece of live state, and which **mutations are
   supported**?
3. What may **inspection** and **completed-fit** consumers read?

It is about the live session: `Project`, `File`, `Model`, `Simulator`, and
the objects they hand out. The completed-fit side is already settled by
[fit_archive_principles.md](fit_archive_principles.md) (Principles 0 to 4);
this document defers to it wherever an archive question arises and restates
only what the live API has to guarantee for those principles to hold. The
module map and the fit-to-slot capture boundary are in
[repo_architecture.md](repo_architecture.md); the deprecation policy and the
user-tier list are in [../stability.md](../stability.md).

## What counts as public

> Public is what a user can **reach** through the documented workflow: the
> top-level exports, every object a public method returns, and every
> attribute the examples read. The export list alone does not describe the
> API.

`__all__` names six things: `Project`, `File`, `FitResults`, `Simulator`,
`PlotConfig`, `sensitivity`. The examples also read, and therefore teach,
`file.model_active` / `file.model_base` (a `Model`), `file.data`,
`file.energy`, `file.base_t_ind`, `file.e_lim`, `file.noise_type`,
`file.sigma_data`, `project.results` (a `FitResults`), the `SavedFitSlot`
fields `params`, `observed`, `fit`, `metrics`, `handle`, `model_name`,
`JointFitResult.params`, and `Simulator.data_clean` / `data_noisy` / `noise`.
All of that is inside the contract. Two notebooks also edit
`model.lmfit_pars[name].vary` and `.value` directly; that is the one taught
use this contract retires (rule 4).

Three tiers, classified coarsely here and inventoried name by name in step 5:

| Tier | Members | Contract |
|---|---|---|
| User | the `__all__` exports, the YAML model format, and the record objects their methods return (`SavedFitSlot`, `SavedFile`, `JointFitResult`, `NoiseModel`) as read surfaces | every rule below; deprecation cycle from v1.0 |
| Advanced | `mcp.Model`, `Component`, `Par`, `Dynamics`, `Profile`, `ParameterSweep`, `MC`, reached through `file.model_active` and friends | the ownership rules below, as objects reached from a file, not as a construction API; stability promise stated by the step 5 tier guide |
| Internal | `graph_ir`, `eval_1d`, `eval_2d`, `eval_jax`, `spectra`, `fitlib`, `fit_io` internals, parsing and HDF5 helpers | none |

## The contract

| Surface | Owner | Supported change | Not supported |
|---|---|---|---|
| Raw data and axes (`data_raw`, `energy`, `time`, `aux_axis`) | `File`, copied at construction, exposed read-only | none; different data or axes means a new `File` | assignment after construction; in-place edits |
| Corrections (`dark`, `calibration`) and the corrected `data` | `File`, owned copies | `subtract_dark`, `calibrate_data`, `reset_dark`, `reset_calibration`; `data` is recomputed from `data_raw` | assigning `dark`, `calibration`, or `data`; editing them in place |
| Baseline (`data_base`, `base_t_ind`, `base_t_abs`) | `File`, derived | `define_baseline`; recomputed when corrections change | assigning `data_base` |
| Fit window and noise (`e_lim`, `t_lim`, `noise`) | `File`, configuration | `set_fit_limits`, `set_noise`, `set_sigma` | assigning the attributes |
| Model definition (YAML records, attachment targets, order, frequency) | `Model`, declarative; constructed only by `File.load_model` | edit the YAML and reload; `add_time_dependence`, `add_par_profile` | building a model in Python; adding or removing components after load; editing records |
| Model parameter state (`lmfit_pars`: values, bounds, `vary`, `expr`) | `Model`; the latest execution state (authored at load, then seeded and fitted values) | edit the YAML and reload; seeding, fitting, sweeps and predictions through the package | editing `lmfit_pars` in memory; anything that rewrites the definition or a completed result |
| Completed results (`SavedFitSlot`, `SavedFile`, `JointFitResult`) | the record | `label`, through `set_label` | every other field; reads hand out protected views or detached copies |
| Identity (`Project.name`, `File.name`, `Model.name`) | construction | none | assignment (setters already raise) |
| Presentation (`PlotConfig`) | `Project` | field edits, `config=` per call | recording presentation in a result |
| Project settings (`show_output`, noise defaults, ...) | `Project`, configuration | assignment | none; `File` construction snapshots the noise defaults |
| Project registry (`Project.files`) | `Project` | `File` construction registers | external append, removal, reordering |
| Simulator draw record (`sigma_data`, `noise_model`) | `Simulator`, snapshot per draw | a new draw | later `set_noise_*` calls changing what a past draw reports |

The rest of this document states the rule behind each row, what follows from
it, and how far the code is from it today. Line references are as of
2026-09-24 and will drift; the function names will not.

## 1. Inputs are owned at construction

> `data`, `energy`, `time`, and `aux_axis` are supplied to the constructor,
> copied, and exposed read-only. Replacing any of them means constructing a
> new `File`. A `File` has one **primary axis**; today it is energy, and it
> may be time. Guards validate the primary axis, never "energy".

**Why construction-only.** The capture-hash guard already forbids changing
`data_raw` or an axis after the first fit (`_capture_file_identity` raises
at the next fit). Allowing assignment before the first fit would need a
second guard for everything derived from the inputs before any fit exists:
fit limits, the baseline window, and the axes the loaded models hold by
reference. One rule with no window is simpler than two guards with one.

**Why copy and expose read-only.** Today `data`, `energy`, `time`, and
`aux_axis` are stored by reference; only `data_raw` is copied
(`File.__init__`). A caller who edits the array they passed changes
`file.data` without changing `data_raw`, so the version stamp still matches
and a refit mints the same handle for different numbers. `Model.energy` /
`time` / `aux_axis` are references to the `File` arrays; read-only copies
make that sharing safe.

**Primary axis.** Step 13 promotes slice-by-slice traces into a `File` whose
primary axis is time. Nothing in the enforcement work may hard-code energy
as the axis that must exist, that limits apply to, or that a 1D fit runs
along; it addresses the primary axis and lets the `File` say which one that
is.

**Construction validates.** Since 0.17.1 the constructor rejects data that
is not 1D or 2D, 1D data given with a time axis, and axes whose length does
not match the data, and it names the transpose when the two lengths are
merely swapped. Two states are therefore unreachable through the
constructor: data without axes (index axes are synthesized) and a 2D file
without fit limits (full-range limits are set at construction). The tests
that build those states by assignment test dead branches and leave with
those branches in step 4.

**What follows.** Constructing a bare `File()` and assigning axes afterwards
is outside the contract. The test suite did exactly that at 130 sites in 17
files, 42 of them also hand-setting `dim`; the constructor already accepts
`data`, `energy`, `time`, and `aux_axis`. Deleting the `dim` lines showed
that no method needs `dim` on a data-less grid file: every failure was a
builder that had assigned data after construction (commit `e588459`). The
remaining assignment sites migrate to constructor arguments through shared
builders. The capture-hash guard stays as the backstop; it stops being the
only line.

## 2. Corrections are operations on owned arrays

> `subtract_dark`, `calibrate_data`, `reset_dark`, and `reset_calibration`
> are the only way to change a correction. The `File` copies the array it is
> given. The corrected `data` and the baseline are derived and recomputed;
> they are exposed read-only and never assigned.

Recomputation already exists: `_apply_corrections` rebinds `data` from
`data_raw` and re-runs `define_baseline` when a window is set. The gap is
ownership: `subtract_dark` and `calibrate_data` store the caller's array by
reference, so editing it afterwards changes the correction the next fit
records without the `File` knowing.

**Noise restriction preserved.** The four correction methods refuse to run
under a declared weighted noise model because σ describes the uncorrected
view and nothing propagates it. That stays until σ propagation is
implemented (step 13); it is a consequence of this rule, not a temporary
guard.

## 3. Baselines come through the baseline API

> `define_baseline()` is the only source of `data_base`, `base_t_ind`, and
> `base_t_abs`. Assigning `data_base` is outside the contract.

**A 1D file has no baseline.** `define_baseline` and `fit_baseline` refuse
1D files and point at `fit_spectrum`, which fits a single spectrum as is
(0.17.1). Before, a 1D file had no public fit path at all, and the library
tests hand-assigned `data_base` to get one; they now build a grid `File`,
evaluate the model, and construct the data `File`, the pattern of the
example generators. Three 2D sites still hand-assign `data_base`: a rescaled
refit, a missing-baseline error, and the test of the hand-set guard itself.
The first two rewrite through `calibrate_data` and plain construction in
step 4. The guard that refuses a declared weighted noise model on a hand-set
baseline leaves with its test; it exists only because the assignment is
possible.

## 4. A model has a definition and a parameter state

> The YAML is the source of truth, always. A `Model` is its **definition**:
> the YAML records plus the declared attachments (target parameter, sequence
> order, frequency). It changes only by editing the YAML and reloading, or
> through `add_time_dependence` and `add_par_profile`, which are declarative
> and captured as records. Its parameter state in `lmfit_pars` (values,
> bounds, `vary`, `expr`) starts as the authored state and afterwards holds
> the latest **execution state**: the values a fit wrote back, the values
> the baseline injection seeded. The archive, not the live model, is
> authoritative for what any fit used; nothing rewrites the definition or a
> completed result.

**Why the split.** Completed fits capture the parameter state at fit time
and key identity on it (Principle 3). Reconstruction (step 7) rebuilds a
model from the definition records and then applies a chosen parameter
state: authored, captured pre-fit, or fitted. Baseline injection, the
`seed_source` choice of step 2, and an external inverse model's prediction
are all the same operation on that state. Keeping the definition
untouchable is what makes those states comparable.

**No runtime parameter edits.** To fix a parameter, change a seed, move a
bound, or add an expression: edit the YAML, reload, refit. A variant is one
more named entry in the same YAML file, loaded by name. It is a little
clumsy, and it keeps things clear: the loaded model always says what the
file says, the compared models are named things, and the archive persists
the YAML snippet at capture, so editing the file afterwards changes nothing
recorded. Notebooks 10 and 12 currently edit `lmfit_pars` in memory; 10
moves to two named variants of `base_GLP`, and 12 passes the parameters it
holds fixed to the sensitivity calculation as an explicit input instead of
editing the truth model. A UI can make authoring more convenient later; it
will author YAML, not mutate models. Decided 2026-09-24.

**The package writes execution state into the live model, and that is
fine.** After a fit the optimized values are written back, so the next fit
of that model and the baseline injection start from them, and
`seed_source="model"` means the current values. That is a warm start for
free, and it costs nothing in the record: every fit captures its effective
parameter state and keys identity on it, so a second fit of the same model
with a changed input is a distinct slot, and reconstruction (step 7) reads
the captured state, never the YAML text. Making the live model immutable
after load was considered and dropped 2026-09-24 as restrictive without a
correctness gain. The one thing it would have removed is that a fresh YAML
seed after a fit needs a reload, which notebook 10 does before each
variant; that stays.

**Structural edits after load are not supported.** Adding or removing
components, renaming parameters, or editing a YAML record on a loaded model
has no public route and gets none; the YAML is the authoring surface. The
attachment calls are the only post-load structural operations because they
are recorded (`ModelYamlRecord.role`, `target_par`, `sequence_index`,
`model_structure`).

**A model belongs to a file.** A `Model` is constructed only by
`File.load_model`, which gives it the file's axes, its provenance records,
and its place in the identity chain from file to model to slot. The axes
have one owner, so a model's grid cannot drift from, or contradict, the
data it describes. Today the model needs its file for nothing else than
the project-owned `PlotConfig`, and evaluation, `Simulator`, and
`sensitivity` read only the model and its axes; that stays. Detached
evaluation, a pickled worker copy or the step 7 loader building a model
from a saved file record, is a package-internal state, not a public
construction path. Building a model in Python from `Model`, `Component`,
and `add_pars` is the internal representation every YAML model becomes; it
is tested as such and is not a public authoring route. Decided 2026-09-24.

**Replacement and attachment are atomic.**

> Construct and validate the candidate first, then publish it. A failed
> reload or a rejected attachment leaves the previous model usable and
> unchanged.

Three current violations, all reproduced by the September 2026 review and
confirmed in code:

- `File.load_model` removes the existing model of that name before the
  replacement YAML is parsed; a parse, component, or submodel error leaves
  the file with neither model.
- `Model.add_dynamics` and `Model.add_profile` set `parent_model` on the
  attached model before the unknown-parameter, expression-linked, and
  missing-axis rejections run.
- `Par.update` sets `t_vary` and `t_model` before `create_value_1d()`; a
  `NameError` there is re-raised as `ValueError` after the flags are set and
  before `Model.update()` runs, leaving the parameter half attached with a
  stale parameter list.

`Model.result` (a frozen `FitOutput`) is a live convenience value written by
the fit methods, not a record; completed-fit consumers read slots (Principle
0) and this contract makes no snapshot promise for it.

## 5. Completed results are protected snapshots

> The content of a `SavedFitSlot`, `SavedFile`, or `JointFitResult` is
> immutable after capture. `label` is the one sanctioned mutable field, set
> through `set_label`. Every public read returns a protected view or a
> detached copy; editing what a read returned never changes the record or
> its archive.

Where this stands: the numpy fields are frozen copies at capture (schema 7,
Principle 4). `FitResults.get_parameters`, `get_correlations`,
`get_confidence_intervals`, and `get_mcmc` return copies, and `variants`,
`compare_models`, and `diff` build new frames. The gap is the nested
containers: `params`, `conf_ci`, `correl`, `params_meta`, `params_stderr`,
`params_init` (DataFrames), `metrics`, `selection`, `fit_settings`, `mcmc`
(dicts), and `component_names` (list) are shared objects inside the frozen
dataclass, and `find`, `get`, iteration, `find_joint`, and `get_joint` hand
out the stored records. A caller who edits `slot.params` edits the history,
and the next `save_fits` writes it. Whether step 4 closes that with copies at
capture plus copy-on-read, or with read-only proxies, is a mechanism choice;
the contract only requires that neither path reaches the record.

`FitResults` itself is a snapshot of the slot list; `Project.results` builds
a fresh one on each access, so two results objects taken around a fit differ
by exactly that fit. That is documented behaviour and stays.

## 6. Inspection reads current state; completed-fit views read captured state

> `File.describe()`, `Project.describe()`, `describe_model`, model previews,
> and plots of current data describe the live session. Everything about a
> completed fit is read from its record and its captured `SavedFile`
> provider (Principle 0). `PlotConfig` is project-owned and changes
> presentation only (Principle 2); no recorded result depends on it.

This is already how the code behaves. It is stated so that the ownership
work does not "fix" `describe()` into reading captured state, and so that no
completed-fit path acquires a live-state shortcut.

## 7. Simulator output records what was drawn

> `Simulator.sigma_data` and `noise_model` describe the last draw, not the
> current settings (v0.17.0). A saved dataset states the definition and the
> parameter values that produced each array; generated arrays stay associated
> with that state.

`save_data` and the sweep already persist `model_name`, the
`model_parameters` JSON, and per-configuration values; the full definition
record and the noise declaration are step 8. One check for step 4: the draw
snapshot holds `clean_data` by reference and relies on the simulation
methods rebinding rather than mutating `data_clean`. That convention is
inside the package, so it is acceptable, but it belongs in the probe set.

## 8. Identity is construction-only

`Project.name`, `File.name`, and `Model.name` already raise on assignment
with a "pass name= at construction" message, and `File.plot_config` raises to
redirect to the project-owned object. That is the precedent for how the
rules above are enforced: refuse at the point of misuse with a message that
names the supported route. Validating the names as path segments is a step 5
item and does not change this rule.

## Parameter vocabulary

The public surface says `par`, `pars`, `param`, `params`, and
`parameter(s)` for the same concept: `get_parameters`,
`plot_param_evolution(params=...)`, `add_par_profile(target_parameter=...)`,
`lmfit_pars`, `Component.pars`, `find_par_by_name`, `add_pars`,
`create_pars`, and the archive keys `params`, `params_init`, `target_par`.

Decided 2026-09-24, implemented in step 5: one root per tier.

- `parameter` / `parameters` in user-tier names (`File`, `Project`,
  `FitResults`, `Simulator`, the sweep). Today's exceptions,
  `plot_param_evolution(params=...)` and `add_par_profile`, become
  `plot_parameter_evolution(parameters=...)` and `add_parameter_profile`,
  matching the `target_parameter` keyword they already carry.
- `par` / `pars` in the mcp layer (`Par`, `pars`, `lmfit_pars`, `add_pars`,
  `find_par_by_name`). Today's exceptions, `parameter_names` and
  `get_all_parameters`, become `par_names` and `get_all_pars`.
- lmfit's `Parameters` / `params` stay where the object is lmfit's; the
  persisted names (`params`, `params_init`, `params_meta`, `params_stderr`,
  `target_par`, `parameter_map`) stay as they are. Persisted names are
  compatibility decisions taken in step 7, not renamed for consistency.

Rejected: one root everywhere. `parameter` everywhere costs the mcp layer
and about 270 `lmfit_pars` sites for no gain in the archive; `param`
everywhere renames `get_parameters` a second time in two minor releases;
`par` everywhere puts a private abbreviation in the user tier.

## Rejected alternatives

- **General setters for data and axes.** Rejected for the reason in rule 1:
  a second guard window with no user benefit over constructing a new `File`.
- **Copying at the slot only.** Declined 2026-07-10 as papering over one
  symptom; the archive then copied at capture on its own merits (Principle
  4) and explicitly refused to depend on this contract. Nothing here reopens
  that.
- **Read-only flags as the guarantee.** `setflags(write=False)` is advisory
  and reversible, and DataFrames cannot be frozen at all. The contract is
  therefore stated in terms of supported operations and ownership; flags
  and copies are mechanisms for step 4.
- **An immutable live model (no write-back of fitted values).** Considered
  and dropped 2026-09-24: the archive is already authoritative and complete,
  so it would cost step 2 work and the free warm start for a clarity gain
  only. See rule 4.
- **A user-tier parameter-editing method, or direct `lmfit_pars` edits as a
  supported advanced-tier route.** Both rejected 2026-09-24: any runtime edit
  lets the loaded model's state drift from its YAML, and the ambiguity is
  already visible in notebook 10's reload-before-each-variant. A variant is
  one YAML entry away. Revisit only when a UI needs a programmatic authoring
  path, and then as authoring (a written YAML), not as mutation.

## Forks, all settled 2026-09-24

1. **Vocabulary.** Settled 2026-09-24: option D above.
2. **Axes after construction.** Settled 2026-09-24: construction-only. The
   alternative, "assignable until the first model is loaded or the first fit
   runs", is a weaker rule that needs its own guard and invalidation of
   limits, baseline, and model axes.
3. **Direct `lmfit_pars` edits.** Settled 2026-09-24: none, at any tier;
   the YAML is the source of truth and a variant is another named entry
   (rule 4). Whether `lmfit_pars` becomes a read-only view or stays an
   unguarded attribute is a step 4 mechanism question. Usage today: 122
   sites in tests (55 set a value, 9 a `vary` flag, 3 a bound, none an
   expression), 4 in the two notebooks, 15 in src outside mcp, of which the
   writers are the fit write-back, the baseline injection, and the sweep.
4. **A `Model` without a `File`.** Settled 2026-09-24: a model is
   constructed only by its file (rule 4), because axes with two owners can
   drift and contradict the data's provenance. The alternative, a standalone
   model with its own axes-at-construction rule, would also have made
   programmatic authoring public, against fork 3. Of the nineteen test sites
   that set `model.energy` directly, six re-assign an axis the file already
   gave the model and simply go; thirteen test the mcp objects themselves
   and stay as internal unit tests.

## Sequencing

- **Step 2** relies on rule 4: `seed_source="model"` means the live model's
  current values and `"baseline"` the baseline fit's, so neither needs the
  definition to change and neither needs a parent-baseline reference.
- **Step 3** turns each row of the contract table into a behavioural probe
  in `docs/ai/code-review.md`.
- **Step 4** enforces rules 1 to 5; the violations listed above are its
  scope, the probes its acceptance tests.
- **Step 5** inventories the tiers name by name and applies the vocabulary.
- **Step 7** relies on the definition / parameter-state split of rule 4.
- **Step 13** introduces time as a primary axis against rule 1.

## Evidence

- Reproduced by the September 2026 review on the then-current checkout:
  caller-array mutation changing `File.data` without `data_raw`;
  correction-array mutation changing recorded corrections without
  recomputation; mutable DataFrames and dicts reachable through frozen result
  records; `load_model` removing the old model before validating the new
  one; a rejected attachment leaving parameter state changed; simulator
  metadata describing a different noise level than the saved data (fixed in
  v0.17.0 by the draw snapshot).
- Confirmed against `main` at v0.17.0 on 2026-09-24 by code inspection:
  storage by reference in `File.__init__`, `subtract_dark`,
  `calibrate_data`; the `_frozen_copy` calls in the slot builders and
  `capture_saved_file`; the copy-returning `FitResults.get_*` accessors and
  the reference-returning `find` / `get` / iteration; the removal-before-parse
  order in `File.load_model`; the `parent_model` assignment before the
  rejections in `Model.add_dynamics` and `Model.add_profile`.
- Usage counts from the same inspection: 130 test sites in 17 files assign
  `file.energy` / `time` / `data` after construction; 7 assign `data_base`;
  notebooks 10 and 12 edit `lmfit_pars` directly; notebook 04 indexes
  `components[1]`.
- Found on this branch (2026-09-24): a 1D file had no public fit path, since
  `fit_spectrum` refused it and pointed at `fit_baseline`, which needs a
  baseline that only `define_baseline` sets, which refuses 1D data. The
  constructor accepted data of any shape against any axes. Both closed in
  0.17.1 (commit `99986d5`).
