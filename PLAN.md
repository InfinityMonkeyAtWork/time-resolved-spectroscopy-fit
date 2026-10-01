# Active Plan: enforce the ownership contract (TODO item 4)

Target: v0.20.0 (breaking: assigning `File` arrays raises; result reads are
detached copies). Branch `enforce-ownership`, started 2026-09-30. TODO items
covered: both `[4]` items under "API contract & ownership". Contract:
`docs/design/api_ownership_contract.md`, rules 1 to 5 plus the rule 7 check;
acceptance tests: the behavioral probes of check 21 in `docs/ai/code-review.md`.

Goal in one line: rules 1 to 5 land, each at its stated level: owned `File`
attributes that refuse assignment (runtime), atomic replacement and
attachment (runtime), detached result reads (runtime), and the lmfit
objects as private API (a naming boundary, not a guard); the four known
violations are closed, and each probe row becomes a regression test. The
registries row (`Project.files`, `File.models`) stays unenforced by
decision.

Status 2026-09-30: every design call below is settled; implementation has
not started (paused at the user's request after the plan review; a third
review round added the window tuples, the sixth lmfit handle, the full
rollback, the extra probes, the simulator drawing from a `NoiseModel` with
the detection-metadata fix falling out of it, and the removal of signed
Poisson sampling). One idiom
is adopted for the whole branch: an ownership rule is declared on the
attribute where it is defined, by a descriptor, instead of being spread over
setters and accessor methods. The three mechanisms (`frozen_copy`, `owned`,
`detached`) live together in a new `utils/ownership.py`, so the contract can
point at one file.

## Scan results (what the code does today, 2026-09-30)

- `File.__init__` copies only `data_raw`; `data`, `energy`, `time`,
  `aux_axis` are the caller's arrays. Nothing is frozen. All owned attributes
  are plain; the raising-setter precedent is `File.name` / `Project.name`
  ("…cannot be reassigned; pass name= at construction."). No `src` code
  writes into these arrays in place, so freezing them breaks nothing internal.
- `subtract_dark` / `calibrate_data` store the caller's array by reference.
  `_apply_corrections` already allocates a new `data` and re-runs
  `define_baseline` from `base_t_abs`.
- `fit_baseline` carries the hand-set-baseline guard (`len(base_t_ind) != 2`
  under weighted noise); one test hand-assigns `data_base`
  (`tests/test_noise_api.py::test_baseline_set_by_hand_refuses_weighted_noise`).
- `File.load_model` removes the same-named model before parsing the
  replacement. `delete_model` / `reset_models` never clear `model_active`.
- `Model.add_dynamics` sets `parent_model` and applies `set_frequency`
  before the unknown / expression-linked checks; `Par.update` sets `t_vary`,
  `t_model` before `create_value_1d()`; `_analyze_expression_dependencies`
  can reject a transitive chain after the attachment is complete, with no
  rollback. `add_profile` mirrors this. Neither checks for an already
  attached parameter (only the `File` wrappers do).
- Records: numpy fields are frozen copies at capture. Every DataFrame, dict
  and list field is stored by reference and handed out as is by `find`,
  `get`, iteration, `find_joint`, `get_joint`. Two live aliases: `conf_ci`
  is the live `model.result.conf_ci` object; all projection slots of a joint
  fit share one `fit_settings` dict. The writer, reader and exporter use
  explicit attribute access and keyword construction; `set_fit_label` writes
  `label` through `object.__setattr__`; `dataclasses.replace` is used on
  `SavedFile`. The one loop that reads a container field repeatedly is
  `compare_models` (`_slot_metric` inside slices × keys).
- Simulator: `_NoiseSnapshot.clean_data` is a reference and `declaration()`
  recomputes the Poisson σ from it on every call; `simulate_*` return the
  arrays they also hold; `add_noise` is public and snapshots the caller's
  array; `save_data` writes `model_parameters` from the live model at save
  time.
- Tests: the 130 bare-`File` sites are gone (`e588459`, `ec26e71`). Left: 6
  deliberate corrupted-state sites in `tests/test_file.py`, 4 `e_lim`/`t_lim`
  writes (`test_fit_history.py`, `test_project_fit.py`), 1 `data_base` write,
  70 `lmfit_pars` writes in 12 files (value 55, vary 12, bound 3), 15
  `model.energy`/`time` lines on bare mcp objects (stay, fork 4). No test
  mutates a returned record. Attachment-rejection tests assert the exception
  only, never the state afterwards.

## Design

### Rules 1 to 3: owned attributes on `File` (settled by the contract)

One descriptor, `owned(route)`, declared once per owned attribute on
`File`: `data`, `data_raw`, `energy`, `time`, `aux_axis`, `dim`, `dark`,
`calibration`, `data_base`, `base_t_ind`, `base_t_abs`, `e_lim`, `e_lim_abs`,
`t_lim`, `t_lim_abs`, `noise`. Reads return the stored object; assignment
raises `AttributeError("File.<attr> is owned by the file and cannot be
assigned; <route>.")` with the route from the contract table (construct a
new `File` / `subtract_dark()` … / `define_baseline()` / `set_fit_limits()` /
`set_noise()`). Internal writes go to `self._<attr>`; about 30 sites, all in
`trspecfit.py`.

An owned attribute holds an immutable object, so a read cannot be edited in
place either: frozen arrays, tuples for the windows (`e_lim`, `t_lim`,
`e_lim_abs`, `t_lim_abs`, `base_t_ind`, `base_t_abs`, built by
`set_fit_limits` and `define_baseline`), the frozen `NoiseModel`, an int for
`dim`. No `src` code mutates a window in place and no test appends to one;
four test comparisons against list literals change. Slot capture converts
windows to lists where the archive's `selection` shape expects them, so the
JSON and HDF5 layout is unchanged. Probe: item assignment on a returned
window raises.

Arrays the file hands out are frozen copies: `data_raw`, `energy`, `time`,
`aux_axis` at construction, `dark` / `calibration` in the correction methods
(and their identity defaults), `data` and `data_base` when recomputed. One
helper, `frozen_copy`, moved from `fit_io._frozen_copy` to
`utils/ownership.py` and imported by `fit_io` (`NoiseModel._freeze_sigma`
can use it too). The arrays stay readable: they are the user's own
measurement and axis, the examples read them for things the package does
not do (a count-budget scale from the data sum, a static block for a
by-hand noise estimate, an axis slice for the simulator), and no package
method stands in for "give me my data". Private storage, public read-only
read (decided 2026-09-30).
Models keep sharing the file's axes by reference; frozen arrays make that
safe. A numpy write into a frozen array raises numpy's own "read-only" error;
the rule-8 message is for the attribute assignment.

The hand-set-baseline guard in `fit_baseline` and its test leave. The six
corrupted-state tests in `test_file.py` either rebuild the state through the
constructor (a 1D file has no time axis, so `fit_2d` / `fit_sbs` "no time
axis" stays reachable) or go with their dead branch (data without axes;
`energy = None` after a model is loaded).

### Rule 4: atomic replacement and attachment (settled by the contract)

- `File.load_model`: parse, construct, add components and records on the
  candidate; only then remove the old model, append, and `set_active_model`.
  `delete_model` / `reset_models` clear `model_active` (and `model_base`)
  when they remove the object it points to (confirmed 2026-09-30).
- `Model.add_dynamics` / `add_profile`: every check before any write to
  either model (unknown parameter, expression-linked, already attached
  `t_vary` / `p_vary` at the mcp level, missing time or aux axis, invalid
  frequency). Then one transaction: candidate setup (`set_frequency`, the
  aux-axis rebinding), `Par.update` (which calls `create_value_1d()` before
  it sets `t_vary` / `t_model`), `update()`, and the expression analysis,
  with `parent_model` set last. Because the analysis needs the attached
  graph and its first pass rewrites every parameter's `expr_refs*` flags
  before it can reject a transitive chain, the transaction restores on
  failure everything it wrote, on both models: the target parameter's
  fields and `_lmfit_par_list`, the model lists and containers (via
  `update()`), the dependency flags (re-run the analysis on the restored
  graph; verify `Par.analyze_expression_dependencies` resets its flags,
  else reset them explicitly), the candidate's `parent_model`, frequency
  state (`frequency`, `time_norm`, `n_sub`, `n_counter`, per-component
  `time_n_sub` / `time_norm`) and `aux_axis` rebinding; then re-raises.
  Probe: for each rejection, the target model evaluates identically before
  and after (`np.array_equal` on `value_1d` / `value_2d`), and its parameter
  names, vary levels, flags and the candidate's state are what they were.
  Key matching alone is not acceptance.

### Rule 5: result containers — settled 2026-09-30: detached copies, not proxies

Mechanism: one dataclass field descriptor, `detached()`, on every container
field of the record classes. It copies on set (closing the `conf_ci` and
shared `fit_settings` aliases at capture) and copies on every read
(DataFrame `.copy()`, dict / list `deepcopy`). Verified on Python 3.12 +
pandas 3.0.1: works with `frozen=True`, required fields, keyword
construction, `dataclasses.replace`, pickling, and the `object.__setattr__`
label write; `fields()` and `repr` unchanged.

What it is: a descriptor is a class-level object whose `__get__` / `__set__`
Python consults on attribute access; `property` is the built-in one, and a
hand-written descriptor is a property written once and declared on many
attributes. Dataclasses support descriptor fields explicitly: the generated
`__init__` stores through `__set__`, reads go through `__get__`, the field
keeps its name as the `__init__` keyword, and a `__get__` that raises
`AttributeError` for `instance=None` makes the field required. The record
stays frozen because the dataclass `__setattr__` raises before the
descriptor is reached. Users never see the descriptor; they see "a read
gives you your own copy", which `get_parameters()` already promised. The
fields stay public because they are the product: the examples read
`slot.params`, `slot.metrics`, `observed` and `fit` directly, and the
renderers and exporters take the records as plain data at the fit-to-slot
boundary. Accessor methods instead would be the same mechanism plus a
rename of every read for no added protection.

First task of the step, before any field is converted: run mypy and pyright
on the prototype. The descriptor needs a typed `__get__` (overloads for the
class access and the instance access) so a read types as the annotation,
and both checkers must accept a required dataclass field that carries a
descriptor default. Fallback if a checker cannot model it: the same
semantics through copy-on-set in `__post_init__` and a `__getattribute__`
override naming the container fields, which is less explicit at the field
but has no typing surface.

Fields: `SavedFitSlot`
(`selection`, `params`, `metrics`, `conf_ci`, `correl`, `mcmc`,
`params_meta`, `params_stderr`, `fit_settings`, `component_names`,
`params_init`), `JointFitProjection.parameter_map`, `JointFitResult`
(`params`, `metrics`, `fit_settings`, `conf_ci`, `correl`), `MCMCResult`
(`table`, `flatchain`; `acceptance_fraction` becomes a frozen copy).
`find` / `get` / iteration keep returning the record object, so identity
tests (`results.get(...) is entry`) stay valid.

Cost: 12 µs per 300-row frame copy, 4 µs per metrics dict (measured). The
`compare_models` helpers, `_fitted_value_map` and `plot_param_evolution`
hoist their reads to once per slot; `compare_models` on an SbS archive is
timed before and after. The MCMC chain is the one container that is not
small (walkers × steps rows), so `get_mcmc`, `plot_joint_mcmc` and the
export of a slot with a representative chain (the default `mc` settings)
are timed too, and their reads hoisted the same way.

Probes, all on existing surfaces: (a) mutation after a read: `slot.params`,
`metrics`, `component_names`, `fit_settings`, `mcmc["flatchain"]`,
`JointFitResult.params`, `JointFitProjection.parameter_map`,
`MCMCResult.flatchain` → the record is unchanged, and one projection's
returned `fit_settings` edited → the other projections and the joint
record are unchanged (that is copy-on-read too, since each read is a
copy); (b) mutation of the source after capture, the cases that motivated
copy-on-set: through the public lifecycle, edit `model.result.conf_ci`
after a fit → the slot's `conf_ci` is unchanged; the shared joint settings
dict is assembled inside the package (`build_fit_settings`, a local of
`Project.fit_2d`) and no public caller holds it, so its copy-on-set is a
record-level invariant test: build two records from one dict and one
DataFrame (direct construction, as `tests/test_fit_archive_writer.py`
already does), mutate the originals → both records unchanged; (c) after
(a) and (b), `save_fits` then `load_fits` equals the records captured
before the edits.

Why not read-only proxies: pandas has no read-only DataFrame, and a wrapper
is not a DataFrame (`isinstance`, pandas ops, export). `MappingProxyType` is
shallow, cannot be deep-copied, pickled or JSON-dumped, and leaves nested
lists and arrays open. Proxies would need two mechanisms (frames and
mappings) and still copy at capture; detached copies are one rule.

### Rule 4, parameter state — settled 2026-09-30: the lmfit objects go private

"Read-only view or unguarded attribute" had a third answer: nothing outside
the package needs the lmfit objects. Every reader of the container is a
package module (the fit entry points, the project fit, `spectra`, `sbs`,
`simulator`, `sensitivity`); the examples read it once (notebook 12 counts
free parameters) and display goes through `describe`, `describe_model`,
`print_all_pars` and the results tables. The six handles on the same
`lmfit.Parameter` objects:

| Class | list of `lmfit.Parameter` | `lmfit.Parameters` container |
|---|---|---|
| `Par` | `lmfit_par_list` | `lmfit_par`, the parameter's own container (`mcp.py:2209`; read by `graph_ir.py`) |
| `Component` | `lmfit_par_list` | `lmfit_pars`, built for `describe()` only |
| `Model` | `lmfit_par_list` | `lmfit_pars`, handed to lmfit by the fits |

All six get the underscore. The list / container split stays in the names
(`_lmfit_par_list` is always the list, `_lmfit_par` / `_lmfit_pars` always a
container). `Model.update_value` becomes `_update_value`, the package's
write route (residual, write-back, seeding, SbS template, sweep); with the
attribute private it would otherwise be the one public edit route that rule
4 forbids. This is API privacy, not runtime enforcement: the underscore
marks the access unsupported and removes the handle from the public surface
and the API docs; it does not prevent mutation. The contract states that
boundary in fork 3.

Consequences:

- Notebook 12 counts free parameters through `get_vary_levels()` (level
  `"static"` is fixed); verify expression-linked parameters count as before,
  the cell asserts exact numbers.
- Example 21's data generator sets six truth amplitudes through
  `update_value`; it loads six named YAML entries instead (the variant
  policy).
- Docstrings in `simulator.py` (`model.lmfit_pars[...]` examples) and
  `mcp.py` (`print_all_pars`), three lines of `lowered_evaluator.md`, fork 3
  and the vocabulary list of the contract, one changelog line (advanced
  tier, no deprecation cycle pre-1.0).
- Tests: mechanical rename (205 `lmfit_pars`, 34 `update_value` sites). The
  reads that check write-back are invariant checks on internals. The 70
  writes follow one rule: fixture inputs a test does not exercise (truth
  values for simulated data, evaluator theta vectors) may use the internal
  route; a test of a public workflow constructs the state under test through
  the route users have, because a direct `vary` or bound write bypasses the
  loader validation that route carries. That is the 15 `vary` / bound sites,
  which become YAML variants in `tests/models/`, by the rule the directory
  already follows: a state that need not share the model name is another
  entry in the base file; a state that must share it (archive grouping,
  per-file bound mismatch) is a second file `<base>_<par>_<variant>.yaml`
  defining only the differing model, opening with a comment "`<model>` from
  `<base>.yaml` with <what differs>". New files: `file_energy_x0_fixed.yaml`,
  `project_time_tau_bounds.yaml`; new entries: `gauss_SD_fixed`,
  `gauss_all_fixed` in `sensitivity_energy.yaml`. The rule goes into the
  Testing section of `CLAUDE.md` as one sentence.
- `Component.lmfit_pars` exists only for `describe()`; dropping it is a TODO
  item 14 consolidation, not done here.

Footprint: `mcp.py` 19 `lmfit_pars` + 27 `lmfit_par_list` + ~10 `lmfit_par`;
other `src` 53 `lmfit_pars` + 1 `lmfit_par_list` + 5 `lmfit_par`
(`graph_ir.py`) + 14 `update_value`; tests 205 + 0 + 6 + 34; examples 1
(notebook 12 read) + 1 (example 21 generator).

Rejected: a plain public attribute (keeps advertising what the rule says
not to touch); a read-only proxy (partial unless the whole mcp graph is
proxied, heavy); stamp-and-check (enforcement machinery for an attribute
that need not be public; stays the mechanism if enforcement is ever wanted,
in the fit-preparation object of TODO item 14).

### Rule 7 check: simulator — settled split (2026-09-30)

In this branch, as the contract names it: `_NoiseSnapshot` holds a frozen copy
of `clean_data` (or the per-point σ computed at the draw), so a later
mutation of the returned clean array, or of the array a caller passed to
`add_noise`, cannot change `noise_model` / `sigma_data`. Probe:
`simulate()`, mutate the returned clean array in place → `noise_model` and
`sigma_data` are unchanged. (The saved clean array does follow that edit
until outputs are owned; that is the deferred item below, not a failing
probe here.)

Also in this branch (decided 2026-09-30, option A): the simulator draws
from the same `NoiseModel` the fits weight with. Today `add_noise` draws
with the generators' own formulas (`rng.normal(0, noise_amp)`,
`rng.poisson(signal_scaled)`, `rng.poisson(expected_counts)`) and then
`_snapshot_noise` re-derives kind, sigma and scale from the settings with
formulas written to mirror them (its docstring says so). A generator change
would silently misdescribe the data, and that declaration is what notebooks
12 and 13 feed into `set_noise` to weight their fits. New shape: the
settings and the clean array resolve to one `NoiseModel` (`utils/noise.py`,
the class `File.set_noise` builds; the mapping is `photon_counting` →
`poisson` with scale = count budget / reference; `analog` gaussian →
`gaussian` with sigma = noise level × clean maximum; `analog` poisson →
`poisson` with scale = `1 / (noise_level + _POISSON_LEVEL_EPS)`, the
formula as it is today); the draw samples from that model (normal with its
sigma, or Poisson on clean × scale divided back); the snapshot is that
`NoiseModel` plus the settings and seed that produced it (`detection`,
`noise_level`, `noise_type`, `counts_per_delay`, `count_rate`,
`integration_time`, `seed`). One mapping function, no mirror. The settings
vocabulary stays the simulator's because resolving it needs the clean data.

Signed Poisson sampling is abandoned (decided 2026-09-30). Today both
Poisson samplers draw from `abs(signal)` and restore the sign, and the
snapshot declares the result as a per-point gaussian sigma from
`abs(clean)`. A count cannot be negative (notebook 12 already teaches that
for fitting), the fallback is a second noise law with a hole (a zero bin
gives sigma zero, which `set_noise` rejects; the bleach fixture is built to
avoid it), and no shipped example draws a signed signal. New rule: Poisson
sampling, `photon_counting` or analog `poisson`, needs a non-negative
signal; a signal with any negative value raises at the draw and names the
routes (simulate before dark subtraction or add an offset; analog gaussian
for a difference spectrum). A pump-on / pump-off difference-spectrum noise
model is TODO item 13 territory. With that, draw and declaration coincide
exactly and there is no approximation to document. The two signed-output
tests and the `_bleach_model` fixture leave with the feature (that fixture
held the one `.set(min=, value=)` internal write of the inventory), replaced
by a refusal test.

Zero cases. Preserved as today: `noise_type="none"` draws nothing and
declares nothing; photon counting on a signal with zero total adds nothing
and declares nothing, and with a nonzero signal but no count budget raises;
zero bins in a non-negative Poisson signal are fine for the sampler and for
the scale-based declaration; analog poisson keeps its epsilon formula and
needs no special case. Changed (decided 2026-09-30): a gaussian draw whose
sigma would be zero, from a zero noise level or an all-zero signal, raises
at the draw instead of drawing zeros and producing a declaration that
`set_noise` rejects. Nothing extra is written for it: the simulator builds
the `NoiseModel` before it draws, that model already refuses a zero sigma,
and the simulator's message names `noise_type="none"` as the noiseless
route. The snapshot holds `None` where no `NoiseModel` exists.

Acceptance compares `NoiseModel`s by `kind`, `scale` and `sigma` (scalar
equality or `np.array_equal`), because the class is `eq=False`.

What falls out: `_write_detection_metadata` writes the settings from the
snapshot instead of the live attributes, which closes the defect the
contract's Evidence wrongly lists as fixed in v0.17.0 (only `sigma_data`
was); no field is added or renamed. TODO item 8 later persists the
declaration through the archive's `encode_noise_model` rather than a
second encoder.

Constraint: the refactor reproduces today's draws for the same seed on
every non-negative signal, so the rng calls keep their order and
arguments; the committed example CSVs (`examples/fitting_workflows/*/data/`)
are the byte-level check, and the existing simulator tests the second.
Probes (check 21, "change settings between computing and saving"): draw,
change the noise level and type, save → the file describes the draw; draw,
`set_noise(**sim.noise_model)` on a `File` → the file's `NoiseModel`
matches the snapshot's by kind, scale and sigma; a negative signal under
either Poisson mode raises before any draw.

Pushed to TODO item 8, the simulator output rework (recorded there 2026-09-30):

- Read-only output arrays. `simulate_*` return the arrays the simulator
  also holds, so an in-place edit changes `save_data`, `get_snr` and
  `plot_comparison`. Today the attributes double as an input surface:
  example 02 assigns a loaded sweep configuration to `data_clean` /
  `data_noisy` / `noise` to reuse `plot_comparison`. Decided 2026-09-30:
  that pattern goes; the arrays become owned outputs and plotting a loaded
  configuration takes arrays. Executed with the rework, where the plotting
  entry point for loaded datasets is designed once.
- `model_parameters` at save time versus generation time: `save_data` reads
  the live model, so a reload between simulate and save desynchronizes
  them. The field carries the whole specification (value, `vary`, bounds,
  expression), so what must describe generation time is that specification,
  not a `valuesdict()`. TODO item 8 owns it. Step 4 runs the probe and
  records the result.

## Steps

- [x] 0. User calls, all settled 2026-09-30: `owned` attributes with a
  public read-only read (rules 1 to 3); `model_active` clearing and the
  mcp-level already-attached check (rule 4); the lmfit objects private incl.
  `_update_value` (rule 4); `detached` copies (rule 5); the rule 7 split
  (snapshot copy here, owned simulator outputs and `model_parameters` to
  TODO item 8); the YAML-variant rule for test models.
- [ ] 1. Rules 1 to 3 on `File`: `utils/ownership.py` with `frozen_copy`
  and `owned` (`detached` joins in step 3, after its typing gate); the
  `owned` attributes with immutable storage (frozen arrays, window tuples),
  correction copies, baseline guard removed with its test, the 11 test
  sites migrated (6 corrupted-state, 4 limits, 1 baseline) plus the 4
  list-literal window comparisons, simulator docstring examples that assign
  `file.data` rewritten. Probes: constructor array mutated afterwards;
  correction array mutated afterwards; item assignment on a returned window
  raises; assignment of each owned attribute raises with the route.
- [ ] 2. Rule 4 atomicity: `load_model` candidate-then-publish;
  `add_dynamics` / `add_profile` / `Par.update` validate-then-attach with
  restore; already-attached check in mcp; `model_active` cleared on removal.
  Probes: state after a broken-YAML reload, after an unknown-parameter,
  expression-linked, already-attached, invalid-frequency and
  transitive-chain rejection, each with the before/after evaluation check.
- [ ] 3. Rule 5: mypy + pyright on the `detached` prototype first (the
  typing gate), then `detached` into `utils/ownership.py` and onto the
  record classes, `MCMCResult` included; hoisted reads in the
  `compare_models` helpers, `_fitted_value_map`, `plot_param_evolution` and
  the MCMC readers; `compare_models` (SbS archive) and MCMC access / export
  timed before and after; the configured checkers run on the integrated
  change (they are in pre-commit). Probes (a), (b), (c) of the rule 5
  section.
- [ ] 4. Rule 7: the simulator draws from a `NoiseModel` built once from
  the settings and the clean array; the snapshot is that model plus the
  settings and seed, holding a frozen copy of what it needs (`add_noise`
  included); `_write_detection_metadata` reads the snapshot. Signed Poisson
  sampling removed: a negative signal under either Poisson mode raises; the
  two signed-output tests and `_bleach_model` go, a refusal test comes.
  Draws reproduce bit-for-bit for the same seed on non-negative signals
  (example CSVs, simulator tests). Probes: snapshot unchanged after caller
  mutation; settings changed between draw and save;
  `set_noise(**sim.noise_model)` round trip compared by kind, scale and
  sigma; negative-signal refusal. The `model_parameters` probe run and its
  result recorded for TODO item 8, no fix here.
- [ ] 5. Parameter state: underscore the six lmfit handles and
  `update_value` across `src` (including `graph_ir.py`) and tests; notebook
  12 via `get_vary_levels`; example 21's generator via named YAML entries;
  the 15 `vary` / bound test writes via YAML variants; docstrings and
  `lowered_evaluator.md`. Acceptance: `hasattr` is false for `lmfit_pars`,
  `lmfit_par_list`, `lmfit_par` and `update_value` on `Model`, `Component`
  and `Par`; `describe`,
  `describe_model` and `get_vary_levels` work; notebook 12's count matches;
  the example notebooks run. Curated autocomplete is TODO item 5.
- [ ] 6. Docs and release: `api_ownership_contract.md` "where this stands"
  paragraphs and Evidence closed with the mechanisms recorded (fork 3
  outcome, rule 5 mechanism, stamp-and-check as the deferral) and a short
  "Mechanisms" paragraph stating the idiom (an ownership rule is declared on
  the attribute by a descriptor; `utils/ownership.py`); `repo_architecture.md`
  ownership paragraph and module map entry; `CLAUDE.md` Testing sentence on
  YAML variants; record docstrings ("reads are detached copies");
  `CHANGELOG.md` 0.20.0 bullets; version bump; `pytest -q`; `/code-review
  diff` with check 21 on the touched surfaces; `/benchmark --fit` on `main`
  versus the branch (expected: within run-to-run noise, since no
  per-evaluation code changes; the measured post-fit costs are the
  `compare_models` and MCMC timings of step 3).

## Out of scope (stays where it is)

- `Project.files` / `File.models` as read-only registries (contract row, no
  numbered rule; lists stay lists).
- `Model.result` snapshot promises (contract: none).
- The simulator noise declaration fields in saved datasets (TODO item 8).
- σ propagation through corrections (TODO item 13); the correction methods keep
  refusing a declared weighted noise model.
- Parameter vocabulary, tier inventory and curated autocomplete (TODO item
  5).
- Path-segment validation of names (TODO item 5).
