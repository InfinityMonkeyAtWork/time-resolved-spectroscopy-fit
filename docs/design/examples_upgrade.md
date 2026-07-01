# Examples Architecture

How the example tree under `examples/` is organized and the decisions behind
it. Examples are grouped by how users actually approach the package — by task
and mindset — rather than as one linear tutorial.

This doc is about *structure*. For the quality bar every notebook must meet
(runs clean, truth-anchored, one clear message, …) see
[`../ai/check-example.md`](../ai/check-example.md), the source of truth for the
`/check-example` skill.

## Conceptual split

The example set reinforces a three-object split:

- **`File`** — the natural surface for fitting and exporting one dataset.
- **`Project`** — configuration, workspace, multi-file coordination, archive
  ownership.
- **`FitResults`** — the inspection/comparison object for completed fits.

A user thinking "fit file 1, export the fit, load it into Origin" should not
need the full `Project` model; a power user still has a clear path to
multi-file and shared-parameter fitting, archives, and comparison.

## Directory layout

```text
examples/
  fitting_workflows/
    01_basic_fitting/               # 0x: single-file fitting skills
    02_dependent_parameters/
    03_multi_cycle_dynamics/
    04_parameter_profiles/
    10_model_comparison/            # 1x: post-fit work
    11_save_load_export/
    12_uncertainty_mcmc/
    20_multi_file_independent_fit/  # 2x: multi-file workflows
    21_multi_file_shared_fit/
  synthetic_data/
    01_simulator/
    02_ml_training_data/
```

Numeric-block legend (documented in `fitting_workflows/README.md`):

- **0x** — fitting skills on a single file.
- **1x** — post-fit work (comparison, persistence, export, uncertainty).
- **2x** — multi-file workflows.

Flat layout with three blocks keeps alphabetical sort intact while making the
category structure visible without an extra directory level. Numbering restarts
at each block boundary as notebooks are added.

## User workflows → entry points

Track-based navigation replaces a linear "walk every notebook in order" path.
The quickstart still recommends `01_basic_fitting` first, but does not imply
every user walks the whole sequence.

**Single-file / Origin-style user** — one dataset; fit, export, keep working in
external tools. Entry: `01_basic_fitting`.

```python
file.fit_baseline(...); file.fit_2d(...)
file.get_fit_results(fit_type="2d")
file.export_fit()     # one-way CSV + PNG, Origin-friendly
file.save_fit()       # HDF5 archive snapshot for this file
```

`Project` is setup/context here (paths, plotting defaults, model files), not the
main object.

**Multi-file independent user** — several files, each fit separately, with
shared setup and a summary view. Entry: `20_multi_file_independent_fit`.

```python
for f in files: f.fit_baseline(...); f.fit_2d(...)
project.export_fits()                 # one coherent tree across files
project.save_fits()                   # one portable HDF5 for the batch
project.results.compare_models(...)   # cross-file survey (no file= filter)
```

The bridge workflow: `Project` as a collection/session workspace, each fit still
file-scoped.

**Shared-fit / power user** — shared parameters across files; `Project` is an
active fitting object. Entry: `21_multi_file_shared_fit`.

```python
project.fit_2d(...)                   # true joint fit
project.save_fits(...)
project.results.compare_models(...)
```

More complex; comes after multi-file independent, not straight after the basics.
The joint multi-file residual is currently MVP — not yet lowered to GIR (see
`TODO.md`), which is the present source of its slowness, not a permanent trait.

**Synthetic-data / ML user** — forward simulation, validation, training data.
Entry: `synthetic_data/` (`01_simulator`, `02_ml_training_data`). These keep
using `Project`/`File` internally because the simulator needs a model, but the
section is framed as data generation, not a fitting tutorial.

## Structural decisions

- **One job per notebook.** `10_model_comparison` is strictly model selection;
  persistence / inspection / export live in the sibling `11_save_load_export`.
  Comparison is its own notebook (not folded into `01`) — comparing two models
  would double the cognitive load of the basic notebook.
- **Documented cross-notebook handoffs.** A notebook may reuse a sibling's
  setup or data when it signposts the reuse up front:
  - `11_save_load_export` re-runs `10_model_comparison` in-kernel via a
    `%%capture` + `%cd -q ../10_model_comparison` + `%run example.ipynb` +
    `%cd -q -` preamble, so all of 10's fitted state (`file`, `project`,
    baseline/SbS/2D slots, σ snapshot, conf_ci) is in scope. A short heartbeat
    cell then prints file/model/slot counts. Notebook 10 keeps `show_output: 1`
    so it stays interactive when opened on its own; `%%capture` suppresses that
    output during the `%run`.
  - `12_uncertainty_mcmc` reuses `01_basic_fitting` the same way, then samples
    the posterior for real via `lmfit.emcee`.
  - `20_multi_file_independent_fit` loads `21_multi_file_shared_fit`'s six-file
    dataset by relative path (copying only its small YAMLs) rather than
    duplicating the CSVs.
- **Casual user's mental model is `File`.** `file.save_fit()` snapshots this
  file's completed fits (latest slot per model / fit type / selection).
- **`auto_export` opt-out.** `fit_*` methods auto-write CSV/PNG on completion by
  default; example `project.yaml` files set `auto_export: False` (with a
  comment) so notebooks leave no surprise files. The default + opt-out is taught
  where export is the topic.
- **`pathlib.Path.cwd()`** for `Project(path=...)`, not `import os`.

## Save / export / load language

- **export** = one-way CSV/PNG for humans and tools like Origin
  (`file.export_fit()`, `project.export_fits()`).
- **save/load** = round-trippable HDF5 archive (`file.save_fit()`,
  `project.save_fits()`, `FitResults.load(...)`, `project.load_fits(...)`).
- `FitResults` is the result browser/comparison object, not something a casual
  single-file user must understand before exporting.

## Benchmark harness

The GIR benchmark (`docs/ai/benchmark.md`) discovers example folders by `NN_`
prefix. Batch mode (`--example 0`) iterates the single-file fitting examples
(01–04); the 1x notebooks and `21_multi_file_shared_fit` sit outside the harness.
