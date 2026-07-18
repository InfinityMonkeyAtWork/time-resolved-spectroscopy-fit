# Fitting Workflows

Track-based, not linear. Pick the notebook that matches what you're trying to
do; each one is self-contained.

## Numeric-block legend

- **0x — single-file fitting skills.** One file at a time. Start here if you
  have one processed dataset.
- **1x — post-fit work.** Model comparison, archive save/load, CSV/PNG export,
  MCMC uncertainty. Use after you have a fit to compare, ship, or characterize.
- **2x — multi-file workflows.** Several files in one project, with or without
  shared parameters.

The numbering restarts at the next block boundary as new notebooks are added.

## Notebooks

| Notebook | What you learn |
|----------|----------------|
| [`01_basic_fitting`](01_basic_fitting/)                       | Load data, define a model, fit baseline + 2D, visualize results. |
| [`02_dependent_parameters`](02_dependent_parameters/)         | Link parameters with expressions (e.g. spin-orbit doublets). |
| [`03_multi_cycle_dynamics`](03_multi_cycle_dynamics/)         | Multi-cycle dynamics with subcycles and frequency. |
| [`04_parameter_profiles`](04_parameter_profiles/)             | Depth-dependent parameters with profile functions (single or combined with time-dependence). |
| [`10_model_comparison`](10_model_comparison/)                 | Compare two models on the same file (baseline, slice-by-slice, 2D). |
| [`11_save_load_export`](11_save_load_export/)                 | `FitResults` HDF5 round-trip, CSV/PNG export, "ship just the winners". |
| [`12_uncertainty_mcmc`](12_uncertainty_mcmc/)                 | Three tiers of parameter uncertainty — `stderr`, profiled CIs, and MCMC — checked against truth. |
| [`20_multi_file_independent_fit`](20_multi_file_independent_fit/)           | Multi-file workspace, **per-file independent** fits (bridge to shared-parameter fitting). |
| [`21_multi_file_shared_fit`](21_multi_file_shared_fit/) | Multi-file workspace, **shared-parameter** fits across files. |

## Choose your track

- **Single processed file → fit:** `01_basic_fitting`.
- **Compare two models on one file:** `10_model_comparison`.
- **Save / load / export fit results (HDF5 or CSV/PNG):** `11_save_load_export`.
- **Estimate uncertainties with MCMC:** `12_uncertainty_mcmc`.
- **Many files, fit each independently:** `20_multi_file_independent_fit`.
- **Many files, shared parameters across them:** `21_multi_file_shared_fit`.
- **Synthetic / ML training data:** see [`../synthetic_data/`](../synthetic_data/).

## Notebook layout

Each notebook directory contains:

- `example.ipynb` — the walkthrough.
- `data/` — input CSVs (where applicable; some notebooks generate data inline).
- `models_energy.yaml` / `models_time.yaml` / `models_profile.yaml` — model
  definitions.
- `project.yaml` — project-level configuration (display, axis labels,
  plotting, etc.).

**Exception — post-fit notebooks.** `11_save_load_export` and
`12_uncertainty_mcmc` have no model/`project.yaml` of their own; their preamble
`%run`s an upstream notebook (11 → 10, 12 → 01) to reuse its fits.
