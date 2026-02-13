# Examples

This directory contains examples demonstrating how to use `trspecfit` for time-resolved spectroscopy analysis.

## Quick Start

**New users should start here:** [01_basic_fitting](fitting_workflows/01_basic_fitting/)

Examples are organized into two main categories:

## 📊 Fitting Workflows

Progressive tutorials showing typical analysis workflows. Work through these in order:

### [01_basic_fitting](fitting_workflows/01_basic_fitting/)
**Start here!** Learn the basic workflow:
- Load spectroscopy data
- Define a simple energy-resolved model
- Set fitting limits
- Fit and visualize results

### [02_dependent_parameters](fitting_workflows/02_dependent_parameters/)
**Advanced:** Parameters that depend on each other
- Link parameters across components
- Use expressions to define parameter relationships
- Constrain fits using physical relationships

### [03_multi_cycle](fitting_workflows/03_multi_cycle/)
**Advanced:** Multi-cycle time-dependent models
- Add time-dependence to parameters
- Model dynamics with multiple subcycles
- Global 2D fitting (time + energy)

## 🔧 Data Generation

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

## Prerequisites

Make sure you have `trspecfit` installed:

```bash
pip install trspecfit
# or for development:
pip install -e .
```

## Need Help?

- **Documentation:** https://trspecfit.readthedocs.io
- **Issues:** https://github.com/username/trspecfit/issues
- **Questions:** Check the docs or open a discussion

## Contributing Examples

Have a useful workflow to share? We welcome contributions! Please ensure your example:
- Is self-contained (includes data, models, configs)
- Has clear markdown cells explaining each step
- Follows the existing naming conventions
- Includes a brief description at the top

See [CONTRIBUTING.md](../CONTRIBUTING.md) for details.
