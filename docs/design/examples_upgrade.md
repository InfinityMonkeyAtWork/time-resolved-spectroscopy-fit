# Examples Upgrade Plan

Design note for a future branch that reorganizes the example notebooks around
the way users actually approach the package. This is deliberately separate from
the fit-results save/load branch: the current branch should finish the archive
feature with minimal examples/docs coverage, then merge. The broader examples
upgrade is a teaching and UX project with enough file movement and narrative
work to deserve its own branch.

## Decisions (locked)

- Track-based navigation replaces the linear "walk forward" path. The
  quickstart still recommends `01_basic_fitting` as the first notebook, but
  does not imply that every user should walk every example in order.
- Top-level directories for this examples-upgrade pass: `fitting_workflows/`
  (existing name kept) and `synthetic_data/` (renamed from
  `data_generation/`). No `data_preparation/` track in this pass.
- Inside `fitting_workflows/`, layout is flat with three numeric blocks:
  **01–04** = fitting skills on a single file; **10–11** = post-fit work
  (comparison, persistence, export); **20+** = multi-file workflows.
  `fitting_workflows/README.md` documents the legend.
- `10_model_comparison` is strictly about model comparison. The
  persistence / inspection / export side (save/load h5, browse loaded
  archives, ship single slots, the two-channels framing) moves to a
  sibling notebook `11_save_load_export`. Each notebook has one job, and
  notebook 11 gets a discoverable filesystem location that can be linked
  from `save_fit` / `export_fit` docstrings, the README, and the CHANGELOG.
- Notebook 11 uses notebook 10's full pipeline as its preamble via the
  IPython `%run` magic (`%run ../10_model_comparison/example.ipynb`),
  preceded by a one-line markdown pointer ("see `10_model_comparison`
  for the fitting/comparison details; this notebook focuses on what to
  do with the results"). Output suppression rides on the existing
  `project.yaml` knobs already used by `10_model_comparison`
  (`show_output: 0`, `auto_export: false`); `%%capture` is the fallback,
  not the default mechanism. Expected preamble runtime ~30–40 s
  (baseline fits <1 s each, SbS ~10 s each, 2D fits a few seconds each)
  — acceptable for a notebook a reader opens deliberately. Single
  source of truth: notebook 10 owns the fit pipeline; notebook 11
  inherits any future updates automatically and ends up with rich state
  (baseline + SbS + 2D slots, σ snapshot, conf_ci) available for the
  persistence demos.
- Casual user's mental model = `File`. `file.save_fit()` saves a snapshot of
  completed fits for this file, keeping the latest slot per model / fit type /
  selection (snapshot semantics inherited from `Project.save_fits`). It is the
  casual user's "save all current fits for this file" API.
- Model comparison (`FitResults.load` + `compare_models`) is its own notebook
  (`10_model_comparison`), **not** part of `01_basic_fitting` — comparison
  requires fitting two models, which doubles the cognitive load of the basic
  notebook.
- Export terminology: **export** = one-way CSV/PNG for humans + tools like
  Origin; **save/load** = HDF5 round-trip via the `FitResults` archive.
- `Project.auto_export` (default `True`, configurable via `project.yaml` or
  post-init mutation) gates the fit-completion CSV/PNG side effects. The basic
  notebook documents the default behavior and shows the opt-out.

## Amendments (2026-05-29)

Decisions revised while stress-testing notebook 01 after the upgrade landed:

- **All persistence moved out of `01_basic_fitting`.** The basic notebook now
  ends at `get_fit_results()` + plots. `save_fit` / `export_fit` /
  `FitResults.load` are gone from 01 (it had drifted into showing `load`,
  which was always meant to live in 11). `11_save_load_export` already covers
  the full save/load/export story comprehensively, so this was a deletion from
  01, not a rebuild of 11. Supersedes the §01 content target and migration
  step 1 below.
- **`01` sets `auto_export: false` in its `project.yaml`** so the introductory
  notebook produces no surprise files on disk. The `auto_export` default/opt-out
  explanation lives in `11_save_load_export`'s tips (which inherits 10's
  `auto_export: false`), not in 01.
- **New `12_uncertainty_mcmc` (block 1x).** MCMC was dead code in 01
  (`MC(use_mc=0)`, never run). It now has its own notebook that `%run`-reuses
  01 (same preamble pattern as 11←10) and samples the posterior for real via
  `lmfit.emcee`. The 1x block legend gains "MCMC uncertainty".
- **Notebooks use `pathlib.Path.cwd()`** instead of `import os` /
  `os.getcwd()` for `Project(path=...)`. Applied to 01 first; the same swap is
  made to the other notebooks as each is revisited.

## Motivation

The fit-results work introduces a clearer split between:

- `File` as the natural surface for fitting and exporting one dataset.
- `Project` as configuration, workspace, multi-file coordination, and archive
  ownership.
- `FitResults` as the inspection/comparison object for completed fits.

The examples should reinforce that split. Users who are thinking "fit file 1,
export the fit, load it into Origin" should not feel that they need to
understand the full `Project` model. Power users should still have a clear path
to multi-file fitting, project-level shared fits, archives, and model
comparison.

## Current State

The examples are currently organized as:

```text
examples/
  data_generation/
    simulator/
    ml_training/
  fitting_workflows/
    01_basic_fitting/
    02_dependent_parameters/
    03_multi_cycle/
    04_par_profiles/
    05_project_level_fitting/
```

This is already close in one respect: simulation and ML training data are
separate from fitting. The weakness is that all fitting notebooks live in one
linear sequence, even though they represent different user mindsets:

- Single-file fitting skills (`01` through `04`).
- Post-fit comparison (currently absent).
- Multi-file / project-level fitting (`05`, with no bridge between the
  single-file basics and full project-level shared fits).

The current quickstart also tells users to start at `01` and work forward. That
is useful for a tutorial path, but less useful once examples cover multiple
tracks.

## User Workflows

### Single-file / Origin-style user

This user has one processed dataset and wants to fit it, export tables/plots,
and keep working in external tools.

Primary API:

```python
file.fit_baseline(...)
file.fit_2d(...)
file.export_fit()                       # CSV + PNG, Origin-friendly
file.save_fit()                         # HDF5 archive snapshot for this file
file.get_fit_results(fit_type="2d")
```

The `Project` should appear as setup/context, not as the main conceptual object.
For this user, `Project` is mostly where config, paths, plotting defaults, and
model files live.

### Multi-file individual fitting user

This user has several files but wants to fit each file separately. They want a
shared loop, consistent settings, per-file exports, and a summary view.

Primary API:

```python
project = trspecfit.Project(...)
files = [...]

for file in files:
    file.fit_baseline(...)
    file.fit_2d(...)

project.export_fits()                              # one coherent tree across files
project.save_fits()                                # one portable HDF5 for the batch
project.results.compare_models(file=files[0], ...)
```

This is the bridge workflow: `Project` is useful as a collection and session
workspace, but each fit remains file-scoped.

### Project-level / shared-fit user

This user intentionally wants shared parameters across multiple files and is
ready for `Project` to be an active fitting object.

Primary API:

```python
project.fit_2d(...)
project.save_fits(...)
project.results.compare_models(...)
```

This workflow is more complex and should come after multi-file individual
fitting, not immediately after single-file basics.

### Synthetic-data / ML user

This user is working with forward simulation, validation, or training data.
They are not preparing experimental data and not primarily fitting an existing
file. The current simulator and ML training notebooks belong together.

## Proposed Directory Layout

```text
examples/
  fitting_workflows/                              # existing name kept
    01_basic_fitting/                             # block 0x: fitting skills (single-file)
    02_dependent_parameters/
    03_multi_cycle_dynamics/                      # renamed from 03_multi_cycle
    04_parameter_profiles/                        # renamed from 04_par_profiles
    10_model_comparison/                          # block 1x: post-fit work
    11_save_load_export/                          # block 1x: post-fit work (NEW)
    20_fit_each_separately/                       # block 2x: multi-file (NEW, bridge)
    21_project_level_shared_fit/                  # was 05_project_level_fitting
  synthetic_data/                                 # renamed from data_generation
    01_simulator/
    02_ml_training_data/
```

Numbering convention documented in `fitting_workflows/README.md`:

- **0x** — fitting skills on a single file.
- **1x** — post-fit work (comparison, persistence, export).
- **2x** — multi-file workflows.

The flat layout with three numeric blocks keeps alphabetical sort intact while
making the category structure visible without an extra directory level.
Numbering restarts at the next block boundary as new notebooks are added
within a category.

## Notebook Content Targets

### `fitting_workflows/01_basic_fitting`

Core "one file" story:

- Load processed `data`, `energy`, and `time`.
- Fit baseline, optional slice-by-slice, and 2D model.
- Show `file.get_fit_results(...)`.
- Show `file.export_fit()` as the Origin-friendly CSV/PNG workflow.
- Show `file.save_fit()` as the archive-snapshot persistence (keeps the latest
  slot per model / fit type / selection for this file).
- **Callout**: `fit_*` methods auto-write CSVs/PNGs to `project.path_results`
  on completion by default. The notebook shows both the default and the
  `project.auto_export = False` opt-out (also settable via `project.yaml`).
- **Out of scope here**: `FitResults.load` and `compare_models` — those move
  to `10_model_comparison` so this notebook stays focused on the casual user's
  single-file path.

### `fitting_workflows/10_model_comparison`

Post-fit comparison story (NEW). Strictly model selection — persistence,
inspection, and export move to the sibling notebook `11_save_load_export`
so each notebook has one job.

- Two models, one file. Compress the fitting cells — readers have seen
  the fit API in 01–04, so this notebook glosses over fitting and
  focuses on comparison.
- Three comparison stories, each isolating one structural choice:
  baseline (line shape), SbS (parsimony), 2D (instrument response).
- In-session comparison via `project.results.compare_models(...)` and
  the sugar delegate `file.compare_models(...)`.
- `compare_models` aggregation modes: default `median`, `sum`, and
  `long` for per-slice rows. Close §6 with a "two practical questions"
  payoff — "which model fits spectrum #4 best?" (long form + slice
  filter) and "which model fits best across the board?" (sum-aggregated,
  sorted by AIC). This motivates `long` form for the batch-of-spectra
  use case (SbS as N independent 1D fits, not necessarily time).
- `plot_residuals` at both ends: 1D obs+fit+residual for baseline,
  shared-scale residual heatmaps for 2D (where the IRF residual band is
  the decisive visual).

Persistence content (`save_fit` / `FitResults.load` / `compare_models`
on loaded archives / slot anatomy / filtered single-slot ship / two
channels / overwrite semantics / σ-snapshot recalibration) does **not**
live here. See `11_save_load_export`.

### `fitting_workflows/11_save_load_export`

Save / load / export story (NEW). The canonical reference for the
`FitResults` archive API, used after a reader has seen fitting and
comparison.

**Preamble pattern** (first two cells):

1. Markdown pointer to `10_model_comparison` ("see that notebook for the
   fits' details; this one focuses on what to do with the results").
2. A single code cell that runs notebook 10's content with suppressed
   output:

   ```python
   %run ../10_model_comparison/example.ipynb
   ```

   IPython `%run` executes the target notebook in the current kernel,
   so all of notebook 10's variables (`file`, `project`, fitted models)
   are in scope below. Output suppression rides on the existing
   `project.yaml` knobs (`show_output: 0`, `auto_export: false`).
   `%%capture` is the fallback if any output leaks past the YAML knobs.
   Runtime ~30–40 s.

After the preamble, the actual content:

- `file.save_fit("comparison.fit.h5")` and `FitResults.load(path)` —
  the canonical round-trip with no live `Project` on the reload side.
- `loaded.compare_models(...)` showing the same comparison API works
  identically against an on-disk archive (sanity check, not the primary
  point).
- Filtered single-slot save (`save_fit(path, model=..., fit_type=...)`)
  and the parallel `export_fit` with the same filters. "Ship the
  winners" as the natural next step once a reader has a verdict.
- The two channels framed by audience: HDF5 (structured, lossless,
  σ-snapshot included, round-trips back into trspecfit — for *future
  you* and other trspecfit users) vs CSV/PNG tree (one-way — for
  Origin, MATLAB, paper plots, non-trspecfit colleagues).
- `FitResults` query API: `files()`, `models()`, `find()`, `get()`,
  and slot anatomy via `dataclasses.fields(slot)` rendering shapes for
  arrays/frames and keys for dicts (every constituent part is
  discoverable without opening the `.h5`).
- Overwrite / slot-collision semantics on `save_fit` (append-by-default,
  `FileExistsError` on slot collision unless `overwrite=True`).
- σ-snapshot semantics — calibrated columns survive load without
  re-`set_sigma()`; what-if recalibration via `chi2_red_raw`.

The preamble pattern is preferred over an inline stripped-down setup
because (a) single source of truth — notebook 10 owns the fit pipeline,
notebook 11 inherits future updates automatically — and (b) rich state:
all slot types (baseline + SbS + 2D, with conf_ci on baseline) are
available for the persistence demos, not just a minimum quorum. The
~30–40 s runtime cost is acceptable for a notebook a reader opens
deliberately.

### `fitting_workflows/20_fit_each_separately`

Bridge story (NEW):

- One model definition and one set of fit limits applied across N files
  (avoids the duplicated setup code a bare `for file in files: ...` loop
  would require without `Project`).
- `project.export_fits()` produces a single coherent directory tree
  (`<root>/<file_name>/<model>__<fit_type>/...`) — easier to diff or zip than
  N separate per-file dumps.
- `project.results.compare_models(file=...)` works across the full batch,
  including replicates of the same physical sample.
- `project.save_fits(path)` packages the whole batch into one portable HDF5.
- Concrete contrast: mention what is lost when running the loop without
  `Project` (the four points above) — makes the value prop explicit rather
  than implicit.

This notebook makes the distinction clear: multi-file workspace does not
necessarily mean shared/project-level fitting.

### `fitting_workflows/21_project_level_shared_fit`

Power-user story:

- Load multiple related datasets.
- Define shared and per-file parameters.
- Run `project.fit_2d(...)`.
- Save/archive results with `project.save_fits(...)`.
- Compare/inspect via `project.results` or loaded `FitResults`.

This is where `Project` becomes the main object. The notebook should note that
the joint multi-file residual is currently in MVP state: it is not yet lowered
to GIR (see `TODO.md`), which is the source of the slowness — not a permanent
characterization.

### `synthetic_data`

Forward-model story:

- `01_simulator`: generate known-truth spectra for validation and demos.
- `02_ml_training_data`: sweep parameter space and save training datasets.

These examples can keep using `Project`/`File` internally because the simulator
needs a model, but the section is described as synthetic data generation, not
as a fitting tutorial.

## Docs Navigation

The examples documentation moves from a single linear path to a "choose your
track" entry point:

- New user with one processed file: start at
  `fitting_workflows/01_basic_fitting`.
- Comparing two fits on one file: start at
  `fitting_workflows/10_model_comparison`.
- Saving, loading, or exporting fit results (HDF5 archive or CSV/PNG
  tree): start at `fitting_workflows/11_save_load_export`.
- Many files, separate fits: start at
  `fitting_workflows/20_fit_each_separately`.
- Shared/global fit: start at
  `fitting_workflows/21_project_level_shared_fit`.
- Simulation or ML training data: start at `synthetic_data`.

The quickstart can still recommend the basic fitting notebook as the first
notebook, but it should not imply that every user should walk every example in
numerical order.

## Save/Export/Load Presentation

The examples should be careful about language:

- Use **export** for one-way CSV/PNG output intended for humans and tools like
  Origin: `file.export_fit()` / `project.export_fits()`.
- Use **save/load** for round-trippable HDF5 fit-result archives:
  `file.save_fit()`, `project.save_fits()`, `FitResults.load(...)`,
  `project.load_fits(...)`.
- Keep individual export visibly supported. The deprecated methods are the old
  method names and legacy implementations, not the single-file export workflow.
- Present `FitResults` as the result browser/comparison object, not as
  something casual single-file users must understand before exporting.

**Auto-export side effect.** `fit_*` methods write CSVs/PNGs to
`project.path_results` automatically on completion by default. Explicit
`file.export_fit()` / `project.export_fits()` calls are the re-runnable,
slot-filtered version of that same content. `project.auto_export = False`
(also settable via `auto_export: false` in `project.yaml`) makes the explicit
path the only one that writes — useful for parameter sweeps, ML training-data
generation, and the long-term real-time fitting goal. Notebooks should
describe both the default and the opt-out, so users are not surprised by
files appearing on disk before they "exported."

## Migration Plan

1. Finish the current fit-results save/load branch with minimal notebook/docs
   coverage:
   - Extend `01_basic_fitting/example.ipynb` with a final section demonstrating
     `file.save_fit()` + `file.export_fit()` (no `compare_models` / `load` —
     those belong in `10_model_comparison`, written in the follow-up branch).
   - Make sure `file.export_fit()` is presented as the Origin-style path.
   - Run tests and merge.

2. Start a new branch for the examples upgrade.

3. Move current notebooks into the new structure:
   - `fitting_workflows/01_basic_fitting` → unchanged
   - `fitting_workflows/02_dependent_parameters` → unchanged
   - `fitting_workflows/03_multi_cycle` →
     `fitting_workflows/03_multi_cycle_dynamics`
   - `fitting_workflows/04_par_profiles` →
     `fitting_workflows/04_parameter_profiles`
   - `fitting_workflows/05_project_level_fitting` →
     `fitting_workflows/21_project_level_shared_fit`
   - `data_generation/simulator` → `synthetic_data/01_simulator`
   - `data_generation/ml_training` → `synthetic_data/02_ml_training_data`

4. Split and add notebooks:
   - `fitting_workflows/10_model_comparison/` — already exists post
     fit-saving merge. Trim to comparison-only: lift §8 (Save → Load →
     Compare Across Sessions), §9 (Browse the Loaded Archive), the
     "Ship just the winning fits" subsection, and the persistence
     bullets/tips into `11_save_load_export`. Update its intro
     table-of-contents (drop bullets about save/load/export) and the
     Tips block accordingly.
   - `fitting_workflows/11_save_load_export/` — NEW. Two-cell preamble
     (markdown pointer + `%run ../10_model_comparison/example.ipynb`),
     then the content lifted from the pre-split notebook 10. See the
     content target above for the full scope.
   - `fitting_workflows/20_fit_each_separately/` — NEW.

5. Add `fitting_workflows/README.md` documenting the 0x / 1x / 2x numeric-block
   legend.

6. Update `examples/README.md`, `docs/examples/index.rst`, and
   `docs/quickstart.md` to use the track-based navigation. Grep for hardcoded
   old paths first.

7. Run notebook smoke checks or at least path/import checks after the moves.

## Non-goals For The Save/Load Branch

- Do not reorganize the full `examples/` tree in the save/load branch.
- Do not rewrite every existing notebook to the new teaching architecture
  before merging the archive work.

The save/load branch should only make the new feature discoverable enough that
users are not stranded. The full teaching architecture belongs in the follow-up
examples branch.

**Data-preparation workflows.** Dark subtraction, detector calibration, and
pixel-to-energy mapping are upstream preprocessing — out of scope for this
examples-upgrade pass. Dark subtraction and detector calibration may be too
instrument-specific for this repo's core example tree. Energy-axis calibration
by fitting reference spectra is closer to `trspecfit`'s value proposition, so
it can be revisited later if we have a compact, shareable Au 4f / valence-band
style dataset and a workflow that teaches calibration without turning into an
instrument-control tutorial.

## Open Questions

- Should moved notebooks preserve old numeric prefixes exactly, or use the new
  block scheme? **Resolved**: use the new 0x / 1x / 2x block scheme; document
  the legend in `fitting_workflows/README.md`.
- Should the examples upgrade include a compatibility note for old paths, or
  is this acceptable as a clean pre-1.0 examples reorganization? **Resolve via
  grep** of `docs/`, `README.md`, `examples/README.md`, and any reference in
  the docstrings before the rename branch starts — the answer follows the
  number of hits.
