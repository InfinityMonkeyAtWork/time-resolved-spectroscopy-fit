# Active Plan

## Fit-archive schema 7 (milestone)

Authoritative design: `docs/design/fit_archive_principles.md` (settled).
Conversion detail: `docs/design/fit_archive_schema_plan.md` — this plan is the
working checklist; it references that document rather than restating it.

Sequencing status (principles §Sequencing): step 1 (identity guards —
`ccb4da0`, `ef0339a`, `d6b2cf9`), step 2 (project-owned `PlotConfig` —
`7befe82`), and step 3 (first-class joint results — `61bcd61`, `67587e3`) are
**done**. Part A below (review-findings hardening) is also done; what remains
is Part B (schema 7 itself). Part C (renderer consolidation) is queued behind
the milestone and orthogonal to it.

### Part A — integrity hardening (verified review findings)

An external review of the three identity-guard commits surfaced six findings;
all six were verified against the code. Four are silent-wrongness paths worth
closing before schema 7 lands, because each fix is small, wire-compatible with
schema 6, and is a schema-7 rule applied early — none of this work is thrown
away. Findings 1–3 share a root: schema 6 stores one data payload per file, so
it cannot faithfully represent fits made against different correction states
of the same file. Schema 7 dissolves that (immutable `data_raw` + per-slot
`dark`/`calibration`); until then the writer must be honest about the limit
instead of silent.

- [x] **A1. Save-path consistency** (finding 1 — `trspecfit.py:541`,
      `fit_io.py:2158`). `_build_saved_project_from_history` stamps the
      `SavedFile` payload with `slots[0].file_fingerprint` (fit-time) while
      copying `live.data` (current). After fit → `subtract_dark()` → save, the
      stored fingerprint does not hash the stored data. Fix: the payload
      fingerprint is `live.fingerprint()` — the archive hashes what it stores.
      Slots whose fit-time stamp differs from the payload stamp are **stale**:
      schema 6 cannot represent them faithfully, so the writer skips them with
      a warning naming each skipped slot and the reason (decision point 1
      below). Covers `save_fits` and `export_fits` (shared builder). Tests: fit
      → correct → save warns and writes nothing (every slot is stale); fit →
      correct → refit → save produces an archive whose fingerprint verifies
      against its data and contains only the refit; the warning names each
      skipped fit.
- [x] **A2. Version stamp back into fit identity** (findings 3 + 5 —
      `fit_io.py:595`). `compute_history_key` dropped the fingerprint
      entirely, so pre- and post-correction fits of the same view now share a
      key and `collapse_history_to_snapshot` (`fit_io.py:1267`) silently drops
      the earlier one. That contradicts principles §Principle 3 ("File
      identity folds in", :880): the stamp is part of **fit** identity even
      though it is no longer **file** identity. `ef0339a` overshot — name-based
      slot→file lookup was the fix; removing the stamp from the key was a
      regression. Fix: the key hashes a canonical **tagged** encoding (JSON
      list, per principles §Composite keys, :402) of
      `(file_name, version_stamp, model_name, fit_type, selection_json)`,
      which also fixes the unframed-`|` collision (`("a|b","c")` vs
      `("a","b|c")` — constructible via file/model names). File lookup stays
      name-based. `File.fingerprint()`'s docstring (`trspecfit.py:2330`)
      becomes true again. Tests: correction refit retains both slots
      in-session with distinct keys; the exact adversarial pair
      ("a|b","c") vs ("a","b|c") yields distinct keys; a correction
      *reversal* (reset_dark after a corrected refit) archives the raw
      fit rather than nothing — the stamp in the key is also what makes
      A1's stale-filter-after-collapse safe (variants never share a
      key, so collapse cannot drop a current slot).
      Note: `compute_archive_slot_key` keeps its `|` join — its first token is
      the writer-generated `files/NNNNNN` and `fit_type` is a closed literal
      set, so a collision is not constructible through the public API
      (theoretical per the proportionality rule); the key is retired wholesale
      in schema 7.
- [x] **A3. Append integrity** (finding 2 — `fit_io.py:1534`).
      `write_archive` drops incoming slots into the first name-matched file
      group without comparing content, so appending a same-name file with
      different data stores the new slot under the old group's data, axes,
      and fingerprint. Fix: same name + different fingerprint **raises** with
      a "different measurement under a reused name — save to a new path"
      message. This is exactly schema 7's write-side identity rule
      (schema plan §File group) applied early. The check must be
      **unconditional and pre-mutation**: the existing collision precheck
      runs only when ``overwrite=False`` and after ``timestamp_updated``
      is rewritten — the name/content raise fires in both overwrite
      modes, before any metadata or group is touched. Test: append after
      a data correction (or a true name reuse) raises under both
      ``overwrite`` modes; the archive — including its timestamps — is
      untouched.
- [x] **A4. Load-path provider association** (finding 4 —
      `fit_results.py:204`). `_files_by_name` collapses same-name `SavedFile`
      providers, but schemas 2–6 legitimately contain same-name groups
      distinguished by fingerprint/path — earlier slots then resolve to the
      last group's axes and data. Fix: `FitResults.load` already iterates
      `(SavedFile, slot)` pairs — retain that parent association per slot
      instead of rebuilding it through a name dict. Name matching remains
      only for live `File` providers (`Project.results`), where uniqueness is
      now guarded. This is schema plan §FitResults item 1 done early. Test: a
      legacy archive with two same-name groups resolves each slot to its own
      group's axes.
- [x] **A5. Truth-up docs and stale identity claims** (finding 6).
      `fit_archive_schema.md:290` still specifies fingerprint-based
      `history_key` recomputation; the reader (`fit_io.py:2081`) uses the new
      algorithm for all schemas 2–6. Update the schema doc's recompute
      paragraph to the v0.14 rule (name-based + version stamp after A2), note
      it as an unversioned reader-behavior change in the CHANGELOG, and sweep
      docstrings for stale fingerprint-identity claims
      (`fit_results.py:180` "matched to slots by fingerprint", `fit_io.py`
      module header, `find`/`get` filter docs).

Known interim limitation (documented, not fixed): `full_range=True` renders
provider data outside the fit window, so an in-session stale slot (fit before
a correction, plotted after) shows corrected data there. The real fix is
schema 7's corrected-data reconstruction helper (schema plan §Live-session
changes); not worth an interim mechanism.

Decision points (resolved 2026-08-04):

1. **Stale-slot handling at save: warn + skip.** The variant stays in
   `_fit_history` (append-only) and the message says why it was skipped and
   that schema 7 retains it. Raising was rejected: filters cannot express
   "current-stamp slots only", so fit → correct → refit → save would
   dead-end.
2. **Part A lands before Part B.** One commit per item, except A1+A2,
   which land as one commit: each is broken without the other (without the
   stamp in the key, collapse can drop a current slot in favor of a stale
   one on correction reversal; without the stale skip, the stamp makes
   refit saves collide on the on-disk slot key).

### Part B — schema 7 conversion

Follow `docs/design/fit_archive_schema_plan.md` §Execution order; steps 1–3
are done (identity guards, PlotConfig refactor and serialization). The schema
plan was reconciled against the landed `JointFitResult` record on 2026-08-05:
parameter maps supersede the prefix-based projection records, the record
extends in place (no `SavedJointFit`), the joint sidecar stores six
whole-objective metrics plus `model_name`, and `compare_models`' NaN-cell
rendering is landed while the drop-only-when-all-lack column rule stays a B9
item. Checklist mirrors the plan's numbering — scope details live there, not
here.

Landing strategy (decided 2026-08-09): **B5–B8 land as one commit.** Dropping
fields in B5 breaks the schema-6 writer/reader/capture, keep-green shims would
be deleted again by B7, and intermediate commits would fail the pre-commit
type checks (forcing `--no-verify`) and break bisect. The tree stays red
between B5 and B8; B9–B12 land as separate green commits afterward.

- [x] **B3. `PlotConfig` (de)serialization** — done 2026-08-05:
      `PlotConfig.to_json`/`from_json`, deterministic (sorted keys), tuple
      fields validated (two-element numeric) and restored on read, unknown
      keys / NaN-Infinity constants / non-JSON values raise naming the
      field, missing keys keep defaults. B6 stores the payload on
      `project/`; B7 decodes it.
- [x] **B4. Hash construction** — done 2026-08-07 (review fixes 2026-08-09):
      eight pure functions in `fit_io.py` — identity family
      `compute_file_content_hash` → `compute_file_version_stamp` →
      `encode_input_files` (+ `encode_model_structure`,
      `encode_optimizer_settings`) → `compute_optimization_hash` →
      `compute_slot_handle`, and independent `compute_fit_view_sha256`.
      Tagged JSON records with record-type tags; only the initial-state
      matrix quantized (9 significant digits, named constant, `-0.0`
      normalized). `model_structure` carries **top-level YAML model names**
      (never component names — those enter identity via the parameter
      table); dynamics submodels are a flat ordered name tuple, order
      assigning subcycles. No-I/O unit tests in
      `tests/test_fit_identity_hashes.py`, including rename/ordering
      regressions and isolated per-input assertions.
- [x] **B5. Object model** — done 2026-08-09 (in tree, part of the B5–B8
      commit): `SavedProject` + `plot_config`/`joint`; `SavedFile` +
      `data_raw`/`file_content_hash`, − `data`/`e_lim`/`t_lim`/fingerprint;
      `SavedFitSlot` + `handle`/`optimization_hash`/`input_files`/
      `model_structure`/`fit_view_sha256`/`dark`/`calibration`/`model_yaml`/
      `label`/`joint_ref`, − `observed_sha256`/`history_key`/
      `file_fingerprint`/`yaml_filename`; `JointFitResult` +
      `optimization_hash`/`input_files`/`model_structure`/`label` in place;
      new `ModelYamlRecord` NamedTuple for the per-snippet provenance
      records. Copy-and-freeze enforcement is B8's (capture). Pyright
      enumerates 47 broken references — the B6–B8 worklist: slot builders
      + joint capture + in-session dedup (`fit_io.py`), snapshot path
      (`trspecfit.py:568-615`), reader (`fit_io.py:2103-2732`),
      `_slot_title` yaml_filename + `observed_sha256` comparability check
      (`fit_results.py:119,1471`).
- [x] **B6. Writer** — done 2026-08-09 (in tree, part of the B5–B8 commit):
      hierarchy moved under `project/` (attrs `name` written once + checked
      on append, `plot_config` rewritten every save); `metadata/` reduced to
      wire identity + new `format` attr; file groups store
      `data_raw`/`file_content_hash` (content-mismatch raise);
      slot metadata carries the identity block, conditional
      `label`/`joint_ref`, DoF metrics omitted when scope == "project";
      new `dark`/`calibration` datasets and `model_yaml/` snippet group
      (scalar vlen, uncompressed); `joint/` sidecar (projections JSON with
      parameter maps, six metrics, r2 omitted); gzip+shuffle on all
      non-empty array datasets; four-case collision rules on `handle`/
      `optimization_hash` (params-differ = hard conflict, per-attachment
      conf_ci/correl/mcmc merge, label mutable) with pre-mutation
      prechecks; bundle-integrity validation (scope⟺joint_ref, resolution
      both directions, map totality). Retired: `compute_archive_slot_key`,
      `_file_ref`, `_find_slot_by_archive_key` (→ `_find_slot_by_handle`,
      `_find_joint_by_hash`); export dir suffix now `handle[:8]`.
      Verified by a 19-assertion smoke script (scratchpad, B10 skeleton) —
      the pytest suite stays red until B8.
- [x] **B7. Reader** — done 2026-08-14 (in tree, part of the B5–B8 commit):
      `SCHEMA_VERSION = "7"`, `SUPPORTED_READ_VERSIONS = ("7",)`, schema
      2–6 evolution comment and fallbacks gone (retires the A5
      legacy-reader note). Decodes `project/` (name, `PlotConfig.from_json`),
      file groups (`data_raw` + `file_content_hash`), slots (identity
      block, selection recovered from this file's `input_files` entry,
      omitted DoF metrics → NaN, `dark`/`calibration`/`model_yaml`/
      `label`/`joint_ref`), and the `joint/` sidecar → `JointFitResult`
      with projection slots resolved by handle to the same objects under
      `files` (r2 → NaN). Reader validates like the writer: dangling
      handles, `joint_ref` mismatches, and non-total parameter maps raise
      (shared `_assert_parameter_maps_total`). All arrays come back
      read-only (`_read_array`, Principle 4). Smoke script extended to 21
      passing checks incl. full-field slot round-trip and joint
      rehydration.
- [x] **B8. Capture** — done 2026-08-14 (in tree, part of the B5–B8
      commit). Prerequisites landed: `mcp.Model` retains `submodel_names`
      (ordered YAML keys) + `yaml_records` (verbatim top-level-key slices
      via new `uparsing.dump_yaml_subtrees` — ruamel round-trip cannot
      load the duplicate component keys the numbering feature allows;
      schema plan §model_yaml corrected), populated by `File.load_model`;
      new `Model.dynamics_entries()` / `yaml_provenance()` walks (incl.
      profile-nested dynamics); `build_fit_settings` gains required
      `backend` (effective, from dispatch-site `args` shape via
      `_effective_backend`) and records `jac_fun` qualname; new
      `fit_io.optimizer_settings_from_provenance` derives the identity
      subset from the provenance dict (one source). Capture: the four
      slot builders compute the local identity chain (`_slot_identity`)
      from `version_stamp` + `params_identity(par_ini)` (per-slice seed
      matrix for SbS) + fit-view coordinates; joint projections inherit
      `(optimization_hash, input_files)` computed once in
      `Project.fit_2d` and pass the N-file `model_structure`;
      `JointFitResult` gets its identity fields. `File._capture_file_identity`
      registers the frozen `SavedFile` payload on first fit
      (`Project._captured_files`), normalizes identity corrections to
      None, and raises on in-place `data_raw` mutation. Assembler
      (`_build_saved_project_from_history`) reads nothing live: captured
      payload + `dataclasses.replace`, joint bundles expand to whole,
      collapse rekeyed on `handle` (variants kept). Retired:
      `compute_file_fingerprint`, `fingerprint_stamp`,
      `compute_observed_sha256`, `compute_history_key`,
      `File.fingerprint`, `_find_file_for_slot`, and the stale-slot
      warn-skip (per-slot corrections make it moot). Consumers:
      `_slot_title` reads the energy snippet's `source_file`,
      `_check_observed_consistency` discriminates on `fit_view_sha256`,
      `FitResults.load` passes the decoded `plot_config` + `joint`.
      Also landed: `utils.arrays.apply_corrections` (the schema plan's
      corrected-data reconstruction helper), used by `File` and by
      full-range plotting to rebuild corrected data from
      `SavedFile.data_raw` + the slot's `dark`/`calibration`. Test
      reconciliation pulled forward: `test_fit_archive_roundtrip.py`
      asserts the schema-7 identity fields + correction/model_yaml/label
      round-trip and drops the v2–v5 downgrade tests (rejection test
      kept); `test_export_fits_parity.py`, `test_full_range_plot.py`,
      and `test_mcp_library.py` fixtures updated. Ruff/mypy/pyright
      clean; smoke script green; **suite 1033 passed with only
      `tests/test_fit_history.py` excluded** (collection-broken: imports
      retired helpers; B10 rewrites it — the last blocker for the B5–B8
      commit).
- [ ] **B9. `FitResults` query layer** — handles, variant table, diffs,
      `select=`, pruning, regrouped comparability, σ tiers, bundle-level
      joint diffs. (A4 pre-completes the association item.) Also owns the
      matrix rows B10 deferred with these features: handle-prefix
      resolution, σ-tier `select=`/`metrics=` semantics, variant table,
      `select=` bundle expansion, `drop()`, bundle-level diffs.
- [x] **B10. Tests** — done 2026-08-14 (in tree, part of the B5–B8
      commit; unblocks it). `tests/test_fit_history.py` rewritten for
      schema 7: fit-level identity (the handle chain recomputes exactly
      from the persisted slot payload; an exact re-run — seed restored —
      shares a handle and collapses; vary flip / bound change / selection
      change / correction each mint a distinct slot; correction reversal
      restores identical `input_files`), capture ownership (slot arrays
      and captured payload frozen and independent; in-place `data_raw`
      mutation raises on the next fit), correction variants archiving
      side by side with per-slot `dark`, and the query/compare/plot
      classes on schema-7 stubs (comparability discriminates on
      `fit_view_sha256`; stubs use the real hash functions). New
      `tests/test_fit_archive_writer.py` (the smoke script graduated):
      layout/compression/retired attrs, one test per collision-table row
      (incl. archived-chain intact after a refused shorter re-run, and
      the incoming MCMC surviving a params-conflict overwrite), append
      integrity byte-identical after a failed append, joint sidecar +
      DoF omission, joint attachment enrich/collide + mcmc/correl
      round-trip, bundle-integrity raises with the archive untouched,
      reader raise on a dangling projection handle, read arrays frozen.
      `test_fit_archive_roundtrip.py` gains `test_joint_bundle_roundtrip`
      (real two-file `Project.fit_2d` → save under a one-file filter →
      bundle expansion → load: identity fields, combined params, correl,
      parameter maps recovering local names, projections resolving to the
      same slot objects, DoF NaN). Regression surfaced by the slow tier
      and fixed: B8's joint-bundle expansion leaked into `export_fits`,
      so a per-file `File.export_fit` wrote the sibling file's tree and
      the sibling's own export then collided. Expansion is an archive
      representability rule — `_build_saved_project_from_history` now
      takes `expand_joint_bundles` (`save_fits`: True; `export_fits`:
      False, which also stops attaching joint records the CSV writer
      never rendered). Verified: 1144 default + 180 slow, 0 failed;
      ruff/mypy/pyright clean.
- [x] **B10.1 Review fix pack** — done 2026-08-14 (in tree, part of the
      B5–B8 commit). An external review of the unit surfaced six
      findings, all verified. Four fixed here:
      (3) live full-range plotting served the file's *current* corrected
      `data` to slots fitted under a different correction state —
      `_full_observed_for` now reconstructs from `data_raw` + the slot's
      own `dark`/`calibration` whenever `data_raw` exists (completes the
      Part A "known interim limitation"; `.data` remains only a fallback
      for providers without `data_raw`).
      (4) the handle-collision comparator now matches the settled rule
      (principles §"Equivalence is defined, not loose"): fitted **values
      only**, per parameter by name, `_PARAMS_EQUIV_RTOL=1e-6` /
      `_PARAMS_EQUIV_ATOL=1e-12` named constants, NaN==NaN —
      `_fitted_values_equivalent` replaces the exact all-column
      `_frames_equal`, so stderr drift or FP noise no longer forces an
      attachment-destroying `overwrite=True`.
      (5) `_assert_parameter_maps_total` → `_assert_parameter_maps_consistent`:
      besides two-way totality it now checks every mapped projection
      value against the combined table (the schema plan's required
      regression), and the reader gained the slot→joint mirror pass
      (scope⟺joint_ref invariant + dangling `joint_ref` raise — deleting
      `joint/` from an archive no longer loads silently).
      (6) copy-and-freeze holes closed: per-slice SbS metric arrays
      frozen on capture (`_per_slice_metrics`) and on read
      (`_read_metrics_per_slice`); the synthesized empty `time` axis in
      `capture_saved_file` frozen.
      Tests: two comparator rows, projection-value disagreement,
      missing-`joint/` reader raise, live-correction-state full-range
      regression, frozen-array assertions extended (incl. mcmc
      acceptance). Verified: 1149 default + 180 slow, 0 failed;
      ruff/mypy/pyright clean.
- [ ] **B10.2 Collapse applies the archive collision rule** (review
      finding 1; own commit right after the B5–B8 commit). Principles
      §"One rule, both boundaries": divergent fitted parameters under one
      handle **raise** at collapse — the error names the non-determinism
      and points at pinning a seed — and `overwrite=True` selects the
      latest and reports the replacement; same rule for the joint
      history's `latest_joint`. Uses the B10.1 comparator. Implements the
      matrix row "in-session collapse applies the same rule as the
      archive boundary" that B10 deferred. Open sub-decision: the shared
      builder also serves `export_fits`, whose `overwrite=` means
      directory overwrite — decide which flag resolves the collapse
      conflict on the export path.
- [ ] **B10.3 Optimizer seed producer** (review finding 2; paired with
      B10.2 — it is the remedy its error message points at). Fit-API
      `seed=` → `fit_wrapper` → per-method forwarding in
      `fitlib._method_kws` → `fit_settings["seed"]`; the identity slot
      already exists (`encode_optimizer_settings` keys `seed` only when
      supplied), so no schema change. Until it lands, unseeded stochastic
      re-runs are surfaced by B10.2's raise.
- [ ] **B11. Docs** — rewrite `fit_archive_schema.md` as the schema-7 spec;
      archive `fit_archive_schema_plan.md` and `joint_fit_result.md`; update
      `repo_architecture.md`, `llms.txt`, `AGENTS.md`, `CLAUDE.md`, CHANGELOG.
- [ ] **B12. Verify** — full suite, Ruff, mypy, pyright, whole-repo grep for
      stale `observed_sha256` / `history_key` / schema-version references.

### Part C — renderer consolidation (queued behind schema 7)

Surviving steps from the previous plan; the config steps (old 1, 2, 3, 7)
dissolved into the milestone per schema plan §Interaction with the current
PLAN.md. Unchanged in scope:

- [ ] **C1. Move every production rendering primitive into `utils/plot.py`**
      (old step 4): 1D fit panels, 2D data/fit/residual panels (move the
      Matplotlib body of `fitlib.plt_fit_res_2d`), MCMC walker/corner
      diagnostics, side-by-side 1D residuals and 2D heatmaps,
      parameter-evolution plots. Each renderer owns figure creation, layout,
      saving, show/close, and return value; finalize against the explicit
      Figure, never pyplot's implicit current one.
- [ ] **C2. Remove Matplotlib ownership from orchestration/fitting modules**
      (old step 5): `fit_results.py` keeps selection/assembly/titles and
      delegates rendering; `fitlib.py` loses its pyplot import (thin adapters
      or coherent removal for `plt_fit_res_*` after a whole-repo grep);
      `fit_io.py` exports PNGs via `utils.plot` directly; `utils/sbs.py`
      drops the worker-side backend switch. End state: no production
      `matplotlib`/`pyplot` import outside `utils/plot.py`, enforced by a
      source-boundary test.
- [ ] **C3. Adapt live/pre-fit entry points to shared renderers** (old
      step 6): `describe_model`, `Model.plot_*`, `Component.plot`,
      `define_baseline`, `set_fit_limits`, simulator plots stay live-state
      exceptions but pass plain arrays + `PlotConfig` into the shared
      renderers; renderers never receive `File`/`Model`/lmfit objects.
- [ ] **C4. Figure lifecycle and compatibility** (old step 8): preserve
      `show_plot=False` / `save_img=-2` semantics, returned-Figure contracts,
      and decide `fitlib.plt_fit_res_*` compatibility after a repo grep.
- [ ] **C5. Tests, docs, close-out** (old steps 9–11, renderer-scoped):
      exercise `Project.results` and `FitResults.load` per plot; update
      `CLAUDE.md` module ownership, `repo_architecture.md`, API docs,
      CHANGELOG; then run the archive-or-changelog close-out question for
      this plan.

### Non-goals (carried over)

- 2D per-component decomposition (evaluator does not produce it).
- Matplotlib in tests/examples — the ownership rule is for `src/trspecfit/`.
- The systemic array-mutation policy (independent TODO; archive ownership
  rules do not wait for it — principles §Sequencing item 4).
