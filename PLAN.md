# Active Plan

## Fit-archive schema 7 (milestone)

Authoritative design: `docs/design/fit_archive_principles.md` (settled).
Conversion detail: `docs/design/fit_archive_schema_plan.md` — this plan is the
working checklist; it references that document rather than restating it.

Sequencing status (principles §Sequencing): step 1 (identity guards —
`ccb4da0`, `ef0339a`, `d6b2cf9`), step 2 (project-owned `PlotConfig` —
`7befe82`), and step 3 (first-class joint results — `61bcd61`, `67587e3`) are
**done**. What remains is Part A below (review-findings hardening) and Part B
(schema 7 itself). Part C (renderer consolidation) is queued behind the
milestone and orthogonal to it.

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
- [ ] **A4. Load-path provider association** (finding 4 —
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
- [ ] **A5. Truth-up docs and stale identity claims** (finding 6).
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

Follow `docs/design/fit_archive_schema_plan.md` §Execution order; steps 1–2
are done (identity guards, PlotConfig refactor). Checklist mirrors its
numbering — scope details live there, not here:

- [ ] **B3. `PlotConfig` (de)serialization** — canonical JSON helpers,
      round-trip tests first. Entirely new code (no to_dict/from_dict exists).
- [ ] **B4. Hash construction** — the two independent families (identity →
      `optimization_hash` → `handle`; comparability → `fit_view_sha256`),
      unit-tested with no I/O. Tagged encodings only (A2 sets the precedent).
- [ ] **B5. Object model** — `SavedProject`/`SavedFile`/`SavedFitSlot` field
      changes, new `SavedJointFit`, copy-and-freeze at capture.
- [ ] **B6. Writer** — `project/` group, `joint/` sidecar, project-name check,
      same-name/different-content raise (A3 is the schema-6 prototype),
      compression, the four-case collision table.
- [ ] **B7. Reader** — `SUPPORTED_READ_VERSIONS = ("7",)`; delete every
      pre-7 fallback branch (this retires the A5 legacy-reader note).
- [ ] **B8. Capture** — first-slot `SavedFile` capture, per-slot correction
      snapshots, serialization of `JointFitResult` (consumes the step-3
      record; layout follows the record).
- [ ] **B9. `FitResults` query layer** — handles, variant table, diffs,
      `select=`, pruning, regrouped comparability, σ tiers, bundle-level
      joint diffs. (A4 pre-completes the association item.)
- [ ] **B10. Tests** — the full matrix in schema plan §Test coverage.
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
