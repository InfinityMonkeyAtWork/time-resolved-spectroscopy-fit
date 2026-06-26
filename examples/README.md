# Examples

Hands-on walkthroughs for using `trspecfit`. The notebooks are organized
by **what you're trying to do**, not in a linear "walk forward" order — pick
the track that matches your task.

## Tracks

### [Fitting Workflows](fitting_workflows/)

Loading data, building models, fitting, comparing, and exporting results.
See [`fitting_workflows/README.md`](fitting_workflows/README.md) for the full
notebook list and the 0x / 1x / 2x numeric-block legend.

| Notebook | What you learn |
|----------|----------------|
| [01_basic_fitting](fitting_workflows/01_basic_fitting/)             | Load data, define a model, fit baseline + 2D, visualize results. |
| [02_dependent_parameters](fitting_workflows/02_dependent_parameters/) | Link parameters with expressions and physical constraints. |
| [03_multi_cycle_dynamics](fitting_workflows/03_multi_cycle_dynamics/) | Multi-cycle dynamics with subcycles and frequency. |
| [04_parameter_profiles](fitting_workflows/04_parameter_profiles/)   | Depth-dependent parameters via profile functions (with optional time-dependence). |
| [10_model_comparison](fitting_workflows/10_model_comparison/)       | Compare two models on the same file (baseline / SbS / 2D). |
| [11_save_load_export](fitting_workflows/11_save_load_export/)       | `FitResults` HDF5 round-trip, CSV/PNG export, "ship just the winners". |
| [12_uncertainty_mcmc](fitting_workflows/12_uncertainty_mcmc/)       | Three tiers of parameter uncertainty — `stderr`, profiled CIs, and MCMC — checked against truth. |
| [20_multi_file_independent_fit](fitting_workflows/20_multi_file_independent_fit/) | Multi-file workspace, per-file independent fits (bridge to shared-parameter fitting). |
| [21_project_level_shared_fit](fitting_workflows/21_project_level_shared_fit/) | Multi-file workspace, shared-parameter fits across files. |

### [Synthetic Data](synthetic_data/)

Forward simulation and ML training data. These are not fitting tutorials —
they cover how to generate datasets from a known model.

| Notebook | What you learn |
|----------|----------------|
| [01_simulator](synthetic_data/01_simulator/)            | Generate synthetic noisy data from a known ground truth. |
| [02_ml_training_data](synthetic_data/02_ml_training_data/) | Sweep parameter space for ML training datasets. |

## Choose your starting point

- **One processed file, want to fit it:** [`fitting_workflows/01_basic_fitting`](fitting_workflows/01_basic_fitting/).
- **Compare two candidate models on one file:** [`fitting_workflows/10_model_comparison`](fitting_workflows/10_model_comparison/).
- **Save, load, or export fit results:** [`fitting_workflows/11_save_load_export`](fitting_workflows/11_save_load_export/).
- **Estimate uncertainties with MCMC:** [`fitting_workflows/12_uncertainty_mcmc`](fitting_workflows/12_uncertainty_mcmc/).
- **Many files, fit each independently:** [`fitting_workflows/20_multi_file_independent_fit`](fitting_workflows/20_multi_file_independent_fit/).
- **Many files, shared-parameter fit:** [`fitting_workflows/21_project_level_shared_fit`](fitting_workflows/21_project_level_shared_fit/).
- **Generate synthetic / ML training data:** [`synthetic_data/`](synthetic_data/).

## Running examples

Most examples are self-contained with:

- `example.ipynb` — Jupyter notebook with step-by-step walkthrough.
- `data/` — input data files (where applicable; some notebooks generate data inline).
- `models_energy.yaml` / `models_time.yaml` / `models_profile.yaml` — model definitions.
- `project.yaml` — project configuration.

The two post-fit notebooks are the exception: **`11_save_load_export`** and
**`12_uncertainty_mcmc`** carry no model files of their own. Their first cell
`%run`s an upstream notebook (11 → 10, 12 → 01) to bring a finished fit into
scope, then shows what to do with it — open them like any other notebook (the
preamble handles the dependency), but expect that first cell to run for the
better part of a minute.

Install the notebook dependencies first:

```bash
python -m pip install "trspecfit[lab]"
```

Open a notebook:

```bash
cd examples/fitting_workflows/01_basic_fitting
jupyter lab example.ipynb
```
