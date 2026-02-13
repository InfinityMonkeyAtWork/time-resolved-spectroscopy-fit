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

To run an example:

```bash
cd examples/fitting_workflows/01_basic_fitting
jupyter notebook example.ipynb
```
