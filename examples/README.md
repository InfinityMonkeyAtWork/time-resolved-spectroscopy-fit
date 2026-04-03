# Examples

This directory contains examples demonstrating how to use `trspecfit` for time-resolved spectroscopy analysis.

## Fitting Workflows

Tutorials showing typical analysis workflows:

### [01_basic_fitting](fitting_workflows/01_basic_fitting/)
Basic workflow:
- Load spectroscopy data
- Define a simple energy-resolved model
- Add time-dependence to a fit parameter
- Fit and visualize results

### [02_dependent_parameters](fitting_workflows/02_dependent_parameters/)
Parameters that depend on each other:
- Link parameters across components
- Use expressions to define parameter relationships
- Constrain fits using physical relationships

### [03_multi_cycle](fitting_workflows/03_multi_cycle/)
Multi-cycle time-dependent models:
- Model dynamics with multiple subcycles
- Parameters can have separate dynamics in different subcycles
- Models allow for global time-dependent behavior (e.g. IRF) across all subcycles

### [04_par_profiles](fitting_workflows/04_par_profiles/)
Profile-aware fitting:
- Attach profile functions to model parameters along an auxiliary axis
- Combine profiles with time-dependence for full 2D models

### [05_project_level_fitting](fitting_workflows/05_project_level_fitting/)
Project-level fitting across multiple files:
- Load shared models across files in a single project
- Fit with shared parameters and per-file results

## Data Generation

Tools for generating synthetic data and machine learning training sets. These examples are primarily for:
- Testing and validation
- Developing ML models
- Understanding the forward model

### [simulator](data_generation/simulator/)
Generate synthetic, noisy spectroscopy data with known ground truth for testing and validation.

### [ml_training](data_generation/ml_training/)
Generate large training datasets for machine learning applications (parameter space exploration).

## Running Examples

Each example is self-contained with:
- `example.ipynb` - Jupyter notebook with step-by-step workflow
- `data/` - Example data files (if applicable)
- `models_energy.yaml` - Energy-resolved model definitions
- `models_time.yaml` - Time-resolved model definitions
- `project.yaml` - Project configuration

Install the notebook dependencies first:

```bash
python -m pip install "trspecfit[lab]"
```

To run an example:

```bash
cd examples/fitting_workflows/01_basic_fitting
jupyter lab example.ipynb
```
