# Active Plan

## Fit-archive schema 7 (milestone)

Authoritative design: `docs/design/fit_archive_principles.md` (settled).
Conversion detail: `docs/design/archive/fit_archive_schema_plan.md` — this plan is the
working checklist; it references that document rather than restating it.

Sequencing status (principles §Sequencing): step 1 (identity guards —
`ccb4da0`, `ef0339a`, `d6b2cf9`), step 2 (project-owned `PlotConfig` —
`7befe82`), and step 3 (first-class joint results — `61bcd61`, `67587e3`) are
**done**. Part A below (review-findings hardening), Part B (schema 7
itself), and Part C (renderer consolidation) are all done — the
milestone is complete.

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

Known interim limitation (since resolved): `full_range=True` used to render
provider data outside the fit window, so an in-session stale slot (fit before
a correction, plotted after) showed corrected data there. Schema 7's per-slot
correction reconstruction fixed the data, and C.7 (captured `SavedFile`
providers on `Project.results`) removed the last live dependency.

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

Follow `docs/design/archive/fit_archive_schema_plan.md` §Execution order; steps 1–3
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
- [x] **B9. `FitResults` query layer** — done 2026-08-26 (in tree).
      UX decisions settled with the user: the variant table is a
      **separate method** (`variants()`) — `compare_models` spans
      different models (Gauss vs GLP) and compares *outputs*, the
      variant table compares *configurations* of one model (m fixed vs
      varying); to keep the two joinable by eye, `compare_models` gains
      a short `handle` column. Handle pinning enters the accessors as a
      `handle=` kwarg (prefix-only, never labels), mutually exclusive
      with the `file`/`model`/`fit_type` trio (`fit_type` defaults are
      now `None` → `"baseline"` so the exclusion is checkable);
      `diff`/`drop_fits`/`select=` take references directly.
      Landed: one shared resolver (`fit_io.resolve_fit_reference` —
      git-style prefixes over handles and joint hashes plus exact
      labels; ambiguous/none raise; same-handle re-runs resolve to the
      latest). `variants()`: one `(file, model, fit_type)` group,
      constant input columns suppressed, multi-group raises.
      `diff(a, b)`: identity/input/result sections at the archive's
      equivalence tolerances (SbS results as per-slice medians);
      projection refs escalate — joint fits diff at the **bundle
      level** via the combined table, `metric` rows gated by the joint
      comparability key (`fit_io.joint_comparability`: sorted
      `(file_name, fit_view_sha256)` *pairs*, multiplicity preserved).
      `FitResults.set_label(ref, label)` per principles §Labels
      (`fit_io.set_fit_label`, the sanctioned mutator; reserved
      `select=` words rejected; projection refs escalate to the joint
      record; persists via `save_fits`). σ tiers in `compare_models`:
      a group mixing finite `sigma_eff` drops σ-scaled columns from
      dynamic defaults (`sigma_eff` stays visible), explicit
      `metrics=["chi2_red"]` raises naming both σ; all-NaN default
      metric columns dropped (explicit requests always render).
      `select=`/`by=` on `save_fits` (default `"all"`) and
      `export_fits` (default `"latest"` — behavior change for the CSV
      tree) via `fit_io.select_snapshot_slots` (per-group post-collapse;
      `by=` is the principles §"Pruning and selection" table exactly —
      `aic`/`bic`/`chi2_red_raw` minimize, `chi2_red` ranks by |x − 1|,
      raw χ²/r² not offered; SbS by per-slice median; a group spanning
      multiple fit views refuses to rank, σ-mixed `chi2_red` raises,
      all-NaN group raises); reference-style `select` is mutually
      exclusive with the filter trio; bundle expansion runs **after**
      selection so the bundle invariant wins.
      `Project.drop_fits(ref)` prunes all runs of a handle or a whole
      bundle by joint hash; a projection ref raises naming the bundle.
      Tests: stub rows in `test_fit_history.py` (+24: resolver
      semantics, `select_snapshot_slots` incl. the reversed-σ-ranking
      fixture, σ-tier compare rows, all-NaN column drop, comparability
      multiplicity); end-to-end `tests/test_fit_query.py` (+17:
      variants/diff/label/handle=/select=/drop on real fits; slow joint
      test covers label escalation, projection-select bundle expansion,
      projection-drop raise, whole-bundle drop, and a bundle diff
      showing a shared-τ bound clamp as input + result + metric rows).
      Deferred, not implemented: label-based export directory naming
      (export suffixes stay handle-based; principles §Labels mentions
      it) and the notebook example updates — `10_model_comparison`
      demos handles → variants → diff → drop, `11_save_load_export`
      demos `select=`; fold into the examples work, discuss first if a
      larger restructure is needed.
      Review addendum (2026-08-26, five findings — four fixed, one
      refuted with a direct repro): (1) `select="best"` now refuses a
      group spanning multiple fit views (principles §Comparability —
      metrics of different views must never be ranked); (2) the `by=`
      policy restored to the settled principles table — first
      implementation offered raw χ²/`chi2`/`r2` and minimized
      `chi2_red` directly, which would systematically select the most
      overfit variant; (3) refuted: a cross-model diff with disjoint
      parameter sets does not crash — missing sides render NA (the
      claimed `float(pd.NA)` path does not exist; pinned as a
      regression test); (4) variant/diff correction cells are 8-hex
      content digests, never booleans (value-only dark changes now
      show), and bundle diffs decode `input_files` into per-file
      version-stamp/selection fields; (5) `File.save_fit` gained
      `select=`/`by=` and every `File.get_*`/`plot_*` wrapper gained
      `handle=` with an ownership guard (a handle naming another file's
      slot raises).
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
- [x] **B10.2 Collapse applies the archive collision rule** — done
      2026-08-14 (review finding 1; own commit after `735762c`).
      Principles §"One rule, both boundaries":
      `collapse_history_to_snapshot` and new
      `collapse_joint_history_to_snapshot` compare fitted values with the
      B10.1 comparator on a handle/hash collision — divergence raises
      `FileExistsError` (the archive's collision type, one `except`
      covers both boundaries) naming the non-determinism and pointing at
      pinning a seed; `overwrite=True` keeps the latest and warns.
      `_build_saved_project_from_history` threads `overwrite` from both
      callers; the joint collapse runs only over the bundles the save
      touches. Sub-decision resolved: `export_fits`' `overwrite=` also
      resolves the collapse conflict — one flag, one meaning ("replace
      what conflicts"), matching its existing authority to clobber
      output directories. Implements the deferred matrix row
      ("in-session collapse applies the same rule as the archive
      boundary"); principles §"One rule, both boundaries" truth-up
      (bare-dict-overwrite description → landed). Tests: unit rule
      (raise / overwrite-keeps-latest-and-warns / within-tolerance
      silent, slots and joint records) + end-to-end `save_fits` raise
      and resolution on a forced divergence. Suite 1154 default + 180
      slow, 0 failed.
- [x] **B10.3 Optimizer seed producer** — done 2026-08-14 (review
      finding 2; the remedy B10.2's error message points at).
      `fit_wrapper` gains `seed: int | None = None`; `_method_kws`
      forwards it to the **`fit_alg_1` stage only** (the two-stage
      contract designates stage 2 as deterministic refinement;
      `fit_alg_2` stays free-form, so a stochastic second stage stays
      unseeded and is surfaced by the collision rules);
      `build_fit_settings` records
      `seed` only when supplied, and `optimizer_settings_from_provenance`
      already keys it into the hash. Every fit API inherits the kwarg
      through its `**fit_wrapper_kwargs` passthrough — no per-API
      signature changes (SbS included; its `seed_source`/`seed_values`
      choose initial parameter values, a different thing, documented in
      the `fit_wrapper` docstring). Per principles, no capability table:
      a seed on a non-accepting method surfaces as SciPy's own TypeError.
      Tests pin the principles scenario table on real
      differential_evolution fits: seeded re-run → same handle,
      bit-identical values, silent dedup; unseeded vs seeded → two
      slots; leastsq+seed → TypeError, no slot captured; two-stage seed
      reaches only the global stage. Docs: principles seed paragraph and
      the schema plan "Still open" bullet marked landed.
      Addendum 2026-08-21 (reviewer accepted the stage-1 contract and
      withdrew the finding; in tree, rides with the B9 commit): wording
      tightened — "forwarded to the stage-1 optimizer"; stage 2 is
      *designated* deterministic refinement by the two-stage contract,
      not enforced (`fit_alg_2` stays free-form) — and the collapse
      divergence message is seed-aware via `_divergence_remedy`: with a
      recorded seed it points at a stochastic `fit_alg_2` / environment
      difference instead of re-recommending the seed the user already
      supplied.
- [x] **B11. Docs** — done 2026-08-26 (in tree).
      `fit_archive_schema.md` rewritten as the self-contained schema-7
      spec from the landed writer/reader (conventions incl. compression
      and read-only rehydration, top/file/slot/joint layouts, the
      identity-chain table, the collision table with the equivalence
      tolerances, reader mapping incl. selection recovered from
      `input_files`, cheat sheet, deliberately-not-stored list); the
      schema 1–6 version history is compressed to a break note (it
      lives in git history). `fit_archive_schema_plan.md` and
      `joint_fit_result.md` moved to `docs/design/archive/` with the
      archived-doc conventions (blockquote header, `../` links,
      `../../../src` paths); every incoming reference repointed
      (principles, repo_architecture, TODO, PLAN, CHANGELOG, one test
      docstring) and the three `fit_io.py` docstring references now
      point at the living spec's §Joint record group. The plan's last
      "Still open" bullet (`frequency` persisted state) truthed up as
      landed via `encode_model_structure`. `repo_architecture.md`
      updated: provider association (parent-owned for archives),
      persisted `PlotConfig`, the B9 query layer, and the fit_io
      section's schema-7 helper inventory. `llms.txt` gains a
      user-facing query-layer paragraph; `AGENTS.md` / `CLAUDE.md`
      needed no changes. CHANGELOG: four entries — the query layer, the
      optimizer seed, the breaking schema-7 bullet, and the
      save-all/export-latest default change.
      Review addendum (2026-08-27, four findings — all valid, all
      fixed): (1) the claimed clean build was checked with incremental
      `make html` + grep, which missed a docutils error — the clean
      `sphinx -W` build failed on `|x − 1|` in the `save_fits`
      docstring (RST parses it as a substitution reference); wrapped in
      inline code there and in `select_snapshot_slots`, and the verify
      convention is now a **clean** `python -m sphinx -W` build;
      (2) `TODO.md` truthed up — the milestone item's defect list is
      past-tense with step (4) marked done (B12 was then the only open
      step) and the joint-sidecar item is complete; (3) `llms.txt` no
      longer claims `handle=` on *any* accessor — scoped to the eight
      single-fit accessors (`get_joint` / `plot_joint_mcmc` /
      `plot_residuals` do not take it); (4) `llms.txt` dedup wording
      fixed — exact re-runs stay in the session history (and appear in
      `variants()`) and deduplicate at save time, with reference
      resolution picking the latest same-handle run.
- [x] **B11.5. Pre-release accessor renames** — done 2026-08-27 (in tree).
      Naming review before 0.14 ships: `get_fit_results` → `get_parameters`
      (next to `FitResults.get()`, which returns the record, the old name
      read as the record accessor), `get_conf_intervals` →
      `get_confidence_intervals`, `FitResults.label` → `set_label` (the
      query layer's only mutating method). Applied identically to the
      `File.*` wrappers; whole-repo rename incl. notebooks, docs, llms.txt,
      benchmark script; archived design docs untouched (point-in-time).
      Declined: `Project.load_fits` → `load_results` (breaks the
      `save_fits`/`load_fits` pairing, and next to the `project.results`
      property it would read as mutating that property — worse than the
      ambiguity it fixes).
- [x] **B12. Verify** — done 2026-08-27 (in tree). Full suite green (fast
      1210 + slow 181), Ruff, mypy, pyright, clean `sphinx -W` build.
      Whole-repo grep (notebooks, YAML, rst, docs included): retired names
      (`observed_sha256` / `history_key` / `archive_slot_key` /
      `yaml_filename`) survive only in the writer guard test, the
      principles doc's conversion rationale, archived docs, and released
      CHANGELOG history — all intentional. Fixed stale references found:
      `10_model_comparison` notebook claimed an `observed_sha256`
      cross-check (→ `fit_view_sha256`); six src docstrings promised
      schema-2/6 loading behavior that no longer exists (pre-7 archives
      unreadable); `_config_for` + `docs/api/fit_results.rst` still said
      styling / joint records are not persisted "until schema 7"; three
      Unreleased CHANGELOG entries used future-tense "until schema 7" and
      the guarded-name bullet still described the retired `history_key`
      mechanics (trimmed — the schema-7 entry owns identity); TODO item
      on array mutation described slot capture with pre-schema-7
      by-reference semantics; rollout-story schema tags pruned from test
      comments.
      Review addendum (2026-08-27): a second external pass caught seven
      more clusters, all verified and fixed in tree — (1) two `FitResults`
      docstrings claimed loaded archives carry no joint records (the
      reader rehydrates them); (2) `11_save_load_export` still taught
      schema-6 save semantics (latest-per-key snapshot, deferred
      `keep_history`, fingerprint prose, same-selection collision) —
      retaught as `select="all"`, handle dedup, and the collision rule,
      with `examples_upgrade.md`'s latest-per-selection line updated;
      (3) four "chi2_red_raw is always present" claims softened (NaN on
      joint projections, dropped from defaults when every row lacks it);
      (4) `Project.results` docstring called Files plot-config providers;
      (5) save/export default paths were conflated in five spots (save is
      `fit_results/<name>.fit.h5`, export `fit_results/<name>/` —
      `Project.name` docstring, `path_results` migration message,
      repo_architecture, TODO, and the Unreleased "fits never write"
      CHANGELOG entry); (6) PLAN/TODO status lines still said Part B / B12
      open; (7) three rollout-era schema comments and one
      fingerprint-as-identity docstring pruned. Retired names now
      additionally survive in TODO's milestone-history item (deliberate
      past tense).

### Part C — renderer consolidation (queued behind schema 7)

Surviving steps from the previous plan; the config steps (old 1, 2, 3, 7)
dissolved into the milestone per schema plan §Interaction with the current
PLAN.md. All five steps done 2026-08-28 (in tree). The post-schema-7 gap
was smaller than this plan's framing: all live/pre-fit entry points
already rendered through `utils/plot.py`, so the substance was moving
`fitlib`'s three `plt_fit_res_*` bodies (evaluation split out as
`fitlib.eval_model_curves_1d`; rendering became the array-based
`plot_fit_overlay_1d` / `plot_fit_res_2d` / `plot_par_series`), the three
lazy-matplotlib `FitResults` methods (MCMC diagnostics and the two
residual comparisons — `fit_results` keeps slot selection/axes/titles and
passes panel dicts), and `sbs.py`'s backend switch
(`use_headless_backend`). `plt_fit_res_*` removed without shims
(pre-1.0, callers were internal; CHANGELOG carries the breaking note);
tests repointed their monkeypatch targets to the `utils.plot` renderers
and the source-boundary test (`TestRenderingImportBoundary`,
ast-parsed imports, docstring examples exempt) enforces the end state:

- [x] **C1. Move every production rendering primitive into `utils/plot.py`**
      (old step 4): 1D fit panels, 2D data/fit/residual panels (move the
      Matplotlib body of `fitlib.plt_fit_res_2d`), MCMC walker/corner
      diagnostics, side-by-side 1D residuals and 2D heatmaps,
      parameter-evolution plots. Each renderer owns figure creation, layout,
      saving, show/close, and return value; finalize against the explicit
      Figure, never pyplot's implicit current one.
- [x] **C2. Remove Matplotlib ownership from orchestration/fitting modules**
      (old step 5): `fit_results.py` keeps selection/assembly/titles and
      delegates rendering; `fitlib.py` loses its pyplot import (thin adapters
      or coherent removal for `plt_fit_res_*` after a whole-repo grep);
      `fit_io.py` exports PNGs via `utils.plot` directly; `utils/sbs.py`
      drops the worker-side backend switch. End state: no production
      `matplotlib`/`pyplot` import outside `utils/plot.py`, enforced by a
      source-boundary test.
- [x] **C3. Adapt live/pre-fit entry points to shared renderers** (old
      step 6): `describe_model`, `Model.plot_*`, `Component.plot`,
      `define_baseline`, `set_fit_limits`, simulator plots stay live-state
      exceptions but pass plain arrays + `PlotConfig` into the shared
      renderers; renderers never receive `File`/`Model`/lmfit objects.
- [x] **C4. Figure lifecycle and compatibility** (old step 8): preserve
      `show_plot=False` / `save_img=-2` semantics, returned-Figure contracts,
      and decide `fitlib.plt_fit_res_*` compatibility after a repo grep.
- [x] **C5. Tests, docs, close-out** (old steps 9–11, renderer-scoped):
      exercise `Project.results` and `FitResults.load` per plot; update
      `CLAUDE.md` module ownership, `repo_architecture.md`, API docs,
      CHANGELOG; then run the archive-or-changelog close-out question for
      this plan.
- [x] **C.6 Review fix pack** — done 2026-09-02, after an external pass
      found the written C1/C4/C5 contracts unmet despite the boundary
      landing: (1) figure lifecycle made explicit — `_finalize_plot(fig,
      ...)` / `img_save(..., fig=)` operate on the passed Figure, every
      renderer (incl. MCMC/residual, which hand-rolled show/close) routes
      through it; (2) `plot_fit_panel_1d` accepts `PlotConfig` (it was
      silently dropping `dpi_plot` and `z_label` — the most-used figure
      was the one a customized config didn't style), the residual maps
      honor `config.z_colormap_res` instead of hardcoding `RdBu_r`, and
      the `mcp` subcycle diagnostic passes `self.plot_config`; a second
      pocket pass (same day) threaded the rest: `plot_mcmc_diagnostics`
      gains `config=` (walker/corner figures render at `dpi_plot` — the
      hardcoded dpi=75 was arbitrary legacy) and `plot_fit_panel_1d`
      applies `x_lim`/`y_lim`/`x_type`/`y_type` via `_apply_axis_settings`
      like its overlay sibling; a third pass (2026-09-02) closed the
      last pockets — `plot_fit_res_2d` honors `dpi_plot`/`dpi_save` and
      labels its colorbar with `z_label` (the docstring had promised it),
      the residual panels/maps render at `dpi_plot` and apply the
      configured axis direction/scale, and all three fit renderers style
      their fit-limit lines from `refline_color`/`refline_style` (only
      the 2D maps did before). Deliberately NOT config-driven: the fit
      renderers' trace-role styling — observed/init/component/fit colors
      and line shapes are fixed semantic styling so every fit figure
      reads the same way; `PlotConfig.colors`/`linestyles`/`linewidths`
      style generic `plot_1d` traces, not fit roles. Also from that pass:
      overlay validates fit=/init= before creating its figure (a bad
      call used to leak a half-drawn one) and the explicit-figure
      lifecycle got its regression test (decoy-current-figure vs target
      in `TestFinalizePlotExplicitFigure`);
      (3) docs state the true boundary — plain *data* (arrays, DataFrames,
      data dataclasses) + `PlotConfig`, not "plain arrays";
      (4) `_symmetric_range` owns the duplicated diverging-scale math
      (the 1D curve-styling merge was declined: the two renderers'
      styles differ deliberately, a shared helper would just parameterize
      every difference); (5) both preliminary `create_value_1d` calls in
      `describe_model` removed — `Model.plot_1d` re-evaluates internally
      (an earlier rebuttal claiming the Dynamics one was load-bearing was
      wrong — mcp.py:1182 evaluates unconditionally); (6) coverage
      truthed up to C5's claim — output-level `plot_fit_overlay_1d`
      tests, loaded-archive tests for `plot_residuals` /
      `plot_param_evolution` / `plot_joint_mcmc`, and notebook 10's stale
      "index axes" `plot_residuals` prose fixed (providers supply real
      axes).
- [x] **C.7 Completed-fit providers are captured, not live** — done
      2026-09-02 (review round 3). `Project.results` attached live `File`
      objects as axes/full-range providers while `save_fits`/`export_fits`
      already built exclusively from the captured `SavedFile` payloads
      (`Project._captured_files`, registered at each file's first fit) —
      so an in-session completed-fit plot could depend on later live
      mutation while the same fit loaded from an archive could not.
      `Project.results` now attaches each file's captured payload with
      its history slots (mirroring the reader), making the completed-fit
      contract identical before and after serialization; a follow-up
      review pass (same day) then **eliminated** the name-matching
      fallback outright — parent association is the only provider
      mechanism (no in-repo user relied on name matching; an unowned
      slot falls back to index axes) — and made `Project.results` raise
      loudly if any history slot lacks a captured payload (an
      impossible-normal state that previously degraded silently).
      Principle
      0's capture bullet now states the provider rule explicitly;
      `repo_architecture.md`'s two provider passages updated; regression
      test pins that in-place `data_raw` mutation after a fit never
      reaches rendering. `PlotConfig` stays live-resolved (Principle 2);
      `describe`/`describe_model`/`define_baseline`/`set_fit_limits`
      stay live by design. Also fixed here: the plan intro still called
      Part C "queued".

### Non-goals (carried over)

- 2D per-component decomposition (evaluator does not produce it).
- Matplotlib in tests/examples — the ownership rule is for `src/trspecfit/`.
- The systemic array-mutation policy (independent TODO; archive ownership
  rules do not wait for it — principles §Sequencing item 4).
