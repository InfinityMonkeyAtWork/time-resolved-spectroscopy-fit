# Active Plan: export by reference, exact references, unique labels (branch `export-by-handle`)

Target: v0.16.0 (the `sigma-weighting` plan moves to v0.17.0). Started 2026-09-17,
branched from `main` at 9779f83. Replaces the TODO item "Label-based export
directory naming", which is superseded rather than implemented.

Goal in one line: an export writes exactly one fit, addressed the same way as
everything else — by an exact handle or a label that is a project-wide unique id —
so a result can be handed to someone without trspecfit and without a directory-
naming scheme to decode.

## Design (settled 2026-09-17)

**Export.** `Project.export_fit(ref, *, filepath=None, overwrite=False, show_output=1)`
and `File.export_fit(ref, *, ...)`. One fit per call. `filepath` is a *root*
(default `./fit_results/<project_name>/`, next to the archive's default path); the
tree below it is fixed:

| reference resolves to | directory |
|---|---|
| a slot | `<root>/<file_name>/<model_name>/<handle[:8]>/` |
| a joint record, or any of its projections | `<root>/joint/<model_name>/<optimization_hash[:8]>/` with one `<file_name>/` subdirectory per projection |

No `__fit_type` segment: `fit_type` is inside the optimization hash, so the handle
already separates a baseline from a spectrum fit of the same model. No `__000`
file ordinal: the handle carries the file's version stamp. Every directory gets a
`fit_info.csv` (two columns, `field,value`): full `handle` / `optimization_hash`,
`file` (or `files`), `model`, `fit_type`, `label`, `joint_ref`, `timestamp`,
`trspecfit_version`. Slot payload is today's `_export_slot` (params, metrics,
conf_ci, mcmc/, fit_1d/fit_2d CSVs, PNGs). Joint payload: `params.csv`,
`metrics.csv`, `conf_ci.csv`, `correl.csv`, `mcmc/`, then the per-file
subdirectories written by `_export_slot`. Joint records were never exported before.
`overwrite=False`: a non-empty target directory raises `FileExistsError` before any
write; `True` clears it first. `File.export_fit` refuses a reference whose fit does
not belong to the file (a bundle counts if the file is one of its projections).

Deleted: `Project.export_fits`, `select=` / `by=` / filters on export,
`write_csv_export`, `_resolve_export_dirs`, `_slot_dir_name`, the multi-slot
`_precheck_export_collisions`, and the `expand_joint_bundles=False` path of
`_build_saved_project_from_history` (the archive path is its only caller).

**References.** `resolve_fit_reference` accepts exactly three forms: the 8-character
display form of a handle or joint hash, the full 64-character digest (both
case-insensitive), or an exact label. No prefix matching. The wrong-length error
names the three forms. The ambiguity branch stays as a defensive backstop (labels
are unique by construction; an 8-char collision is astronomically unlikely). Rule
change recorded in `fit_archive_principles.md` §"Slot handles" (was: "any prefix,
git-style").

**Labels.** Non-empty, not in `SELECT_RESERVED`, and **unique project-wide across
slots and joint records** — a label is the human-readable twin of the handle, which
`compute_slot_handle` already makes project-unique by hashing the file name in.
Enforced twice:

- in session, `FitResults.set_label` raises `ValueError` naming the current holder
  when another record in the view carries the label (relabelling the same record is
  a no-op);
- at archive append, an attrs-only scan of stored slot and joint metadata (extends
  `_stored_identity_sets`; no slot rehydration) collects taken labels; an incoming
  label held by a *different* handle / hash raises before any mutation; with
  `overwrite=True` the incoming record wins and the stored holder's `label` attr is
  deleted (this is also the only way to move a label to a newer run from a later
  session). A same-handle re-save rewrites the label, as today.

Labels are valid export references. `select=` on `save_fits` keeps `"all"` /
`"latest"` / `"best"` + a reference, unchanged.

**Rejected (recorded in the principles doc, do not re-open here):** dropping export
entirely; label as export-directory suffix; per-file label uniqueness with
filter-scoped resolution; hex-only and filesystem-charset label rules; prefix
resolution with a warning; positional `filepath`.

## Steps

### A — docs first
- [x] A1 `PLAN.md` (this file); `TODO.md`: export-naming item → `[ACTIVE]` entry.
- [x] A2 `docs/design/fit_archive_principles.md`: §Slot handles lookup rule; §Labels
      (uniqueness + rationale, rejected scopes); §Archive vs export rewritten for
      one-fit export (the `select=` defaults table goes); §Pruning and selection
      intro no longer says "at save/export time".

### B — references
- [ ] B1 `resolve_fit_reference` (fit_io.py ~L2011): exact 8/64 hex or exact label;
      helper `_is_reference_form(ref)`; error text names the three forms.
- [ ] B2 Docstrings that say "prefix": fit_results.py (`get`, `_resolve_ref`,
      `set_label`, `diff`, `_resolve_slot`, every `handle=` parameter block),
      trspecfit.py (`save_fits`, `drop_fits`, `_build_saved_project_from_history`),
      `llms.txt` L55-56.
- [ ] B3 Tests (test_fit_history.py `TestResolveFitReference`): 8-char accepted,
      full accepted, case-insensitive, 7- and 9-char rejected with the three-forms
      message, label still exact; drop the prefix tests.

### C — labels
- [ ] C1 `FitResults.set_label`: uniqueness check over `self._slots` + `self._joint`
      (skip the target itself); `set_fit_label` docstring states the contract.
- [ ] C2 fit_io.py: `_stored_labels(archive)` (attrs only) →
      `{label: (kind, identity, file_name)}`; `_precheck_labels(archive, project,
      overwrite)` called from `write_archive` next to `_precheck_bundle_integrity`,
      before any mutation; on `overwrite=True` return the stored holders to strip,
      and delete their `label` attr before the write loop.
- [ ] C3 Tests (test_fit_query.py `TestLabelFlow`, test_fit_history.py): duplicate
      `set_label` raises naming the holder; relabel-same-record ok; slot vs joint
      collision raises; append with a taken label raises; `overwrite=True` moves it
      (reload shows one holder); same-handle relabel re-save still rewrites.

### D — export
- [ ] D1 `Project.export_fit`: resolve → slot or joint; slot path uses the captured
      `SavedFile` provider (`_captured_files[file_name]`) for axes; joint path writes
      the joint payload + per-projection subdirs; `fit_info.csv` in every directory;
      single-directory overwrite pre-check; summary line.
- [ ] D2 `File.export_fit(ref, *, ...)`: ownership guard, delegate.
- [ ] D3 Delete `Project.export_fits`, `write_csv_export`, `_resolve_export_dirs`,
      `_slot_dir_name`, multi-slot `_precheck_export_collisions`; collapse
      `expand_joint_bundles` in `_build_saved_project_from_history`.
- [ ] D4 Tests: test_export_fits_parity.py → two exports of one reference are
      byte-identical trees; test_fit_query.py export tests → layout
      `<file>/<model>/<handle8>`; test_fit_side_effects.py `test_export_fits_writes`
      → `export_fit`; test_project_fit.py `File.export_fit` after a project fit;
      new: joint bundle export layout + `fit_info.csv` fields, projection ref →
      bundle, `overwrite` refusal and clear, `File.export_fit` ownership error,
      label as export reference.

### E — docs, examples, release
- [ ] E1 Notebook 11 §2: export one fit by handle and by label, print the tree;
      §1 sentence "per-file wrappers take just the keyword forms" goes. Notebook 20:
      export cell becomes a loop over each file's handle.
      Run `scripts/normalize_notebooks.py` before staging.
- [ ] E2 `docs/design/repo_architecture.md` (Export bullet + defaults paragraph),
      `docs/design/examples_architecture.md` L66/78/130/136, `docs/api/trspecfit.rst`
      (`Project.export_fit` if the class lists methods explicitly), `llms.txt` L120,
      trspecfit.py `_removed_keys` messages (`export_fits()` → `export_fit()`),
      `drop_fits` docstring, `docs/ai/check-example.md` L225/L495 wording.
- [ ] E3 `CHANGELOG.md` `[0.16.0]`: Removed (`export_fits`, prefix resolution),
      Changed (labels unique, `select=` forms), Added (`export_fit`, joint export,
      `fit_info.csv`). `pyproject.toml` → 0.16.0. `TODO.md` item → done, tag off.

### F — verify
- [ ] F1 `pytest -q`; pre-commit (ruff, mypy --no-incremental, pyright);
      `sphinx -W` docs build.
- [ ] F2 Execute notebooks 10, 11, 20 end to end; `/check-example 11`.
- [ ] F3 Archive: ask whether `PLAN.md` moves to `docs/design/archive/` or the
      changelog suffices; clear `PLAN.md`; update `TODO.md`.
