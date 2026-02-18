# trspecfit - 2D Time- and Energy-resolved Spectroscopy Fitting

[![Documentation Status](https://readthedocs.org/projects/time-resolved-spectroscopy-fit/badge/?version=latest)](https://time-resolved-spectroscopy-fit.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://img.shields.io/pypi/v/trspecfit.svg)](https://pypi.org/project/trspecfit/)

`trspecfit` is a Python package for modeling and fitting 1D energy-resolved and 2D time-and-energy-resolved spectroscopy data. It extends lmfit with composable spectral components, parameter-level time dynamics, convolution kernels, and simulation tools so you can build, fit, and validate physically meaningful models in one workflow.

## Capabilities

- Modular components (Gaussian, Voigt/GLP/GLS, Doniach-Sunjic, backgrounds, kernels)
- 1D and 2D model construction with time-dependent parameters
- Global fitting via `lmfit`, including CI and optional MCMC (`lmfit.emcee`)
- Synthetic data generation (single spectra, 2D datasets, noisy realizations)
- Parameter-sweep simulation for validation and ML training data generation

## Documentation

Full docs are hosted on Read the Docs:
- Docs home: https://time-resolved-spectroscopy-fit.readthedocs.io/en/latest/
- Installation: https://time-resolved-spectroscopy-fit.readthedocs.io/en/latest/installation.html
- Quick start: https://time-resolved-spectroscopy-fit.readthedocs.io/en/latest/quickstart.html
- Examples: https://time-resolved-spectroscopy-fit.readthedocs.io/en/latest/examples/index.html
- API reference: https://time-resolved-spectroscopy-fit.readthedocs.io/en/latest/api/index.html

For consistent, central plot behavior, set plotting defaults at `Project` creation (typically via `project.yaml`), and see PlotConfig details and override patterns here: https://time-resolved-spectroscopy-fit.readthedocs.io/en/latest/api/plot_config.html

## Installation

Install from PyPI:

```bash
pip install trspecfit
```

Install from GitHub:

```bash
pip install git+https://github.com/InfinityMonkeyAtWork/time-resolved-spectroscopy-fit.git
```

## Quick Usage

```python
from trspecfit import Project, File

project = Project(path='examples/simulator', name='local-test')
file = File(parent_project=project, path='simulated_dataset')

file.load_model('models_energy.yaml', ['ModelName'])
file.describe_model()

file.add_time_dependence(
    model_yaml='models_time.yaml',
    model_info=['TimeModelName'],
    par_name='EnergyModelComponent_NN_par',
)

file.model_active.create_value2D()
value_2d = file.model_active.value2D
```

For full workflows, see the docs examples page and the notebooks in `examples/`.

## Development

```bash
# Create env (same on all platforms)
python -m venv .venv

# Activate virtual environment
# Linux / macOS
source .venv/bin/activate
# OR Windows PowerShell
.\.venv\Scripts\Activate

# Install and setup (same on all platforms)
pip install -U pip
pip install -e ".[dev]"
python -m pre_commit install --install-hooks
python -m pre_commit run --all-files
```

## Repository Layout

- `src/trspecfit/` - package source
- `docs/` - Sphinx docs source
- `examples/` - notebooks and YAML models
- `tests/` - pytest test suite

## Copyright Notice

time-resolved spectroscopy fit (trspecfit) Copyright (c) 2025, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative
works, and perform publicly and display publicly, and to permit others to do so.
