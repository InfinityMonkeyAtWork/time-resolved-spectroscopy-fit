# trspecfit - 2D Time- and Energy-resolved Spectroscopy Fitting

[![Documentation Status](https://readthedocs.org/projects/time-resolved-spectroscopy-fit/badge/?version=latest)](https://time-resolved-spectroscopy-fit.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://img.shields.io/pypi/v/trspecfit.svg)](https://pypi.org/project/trspecfit/)

`trspecfit` is a Python package for modeling and fitting 1D energy-resolved and 2D time-and-energy-resolved spectroscopy data. It extends lmfit with composable spectral components, parameter-level time dynamics, convolution kernels, and simulation tools so you can build, fit, and validate physically meaningful models in one workflow.

## Capabilities

- Modular components (Gaussian, Voigt/GLP/GLS, Doniach-Sunjic, backgrounds, kernels)
- 1D and 2D model construction with time-dependent parameters
- Auxiliary-axis parameter profiles via `add_par_profile(...)`
- Global fitting via `lmfit`, including CI and optional MCMC (`lmfit.emcee`)
- Synthetic data generation (single spectra, 2D datasets, noisy realizations)
- Parameter-sweep simulation for validation and ML training data generation
- Centralized plot configuration via `PlotConfig` with per-plot overrides

## Documentation

Full docs are hosted on Read the Docs:
- Docs home: https://time-resolved-spectroscopy-fit.readthedocs.io/en/latest/
- Installation: https://time-resolved-spectroscopy-fit.readthedocs.io/en/latest/installation.html
- Quick start: https://time-resolved-spectroscopy-fit.readthedocs.io/en/latest/quickstart.html
- Examples: https://time-resolved-spectroscopy-fit.readthedocs.io/en/latest/examples/index.html
- API reference: https://time-resolved-spectroscopy-fit.readthedocs.io/en/latest/api/index.html

## Support and Community

- Questions and usage help: GitHub Discussions (Q&A category)
- Ideas and brainstorming: GitHub Discussions (Ideas category)
- Reproducible bugs and trackable feature requests: GitHub Issues
- Maintainer response target: within 7 days

Links:
- Discussions: https://github.com/InfinityMonkeyAtWork/time-resolved-spectroscopy-fit/discussions
- Issues: https://github.com/InfinityMonkeyAtWork/time-resolved-spectroscopy-fit/issues

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

project = Project(path='my_project', name='my_experiment')
file = File(parent_project=project, path='my_dataset',
            data=..., energy=..., time=...)

# Load model from YAML, set limits, fit
file.load_model('models_energy.yaml', 'my_base_model')
file.set_fit_limits(energy_limits=[...], time_limits=[...])
file.fit_baseline('my_base_model')

file.load_model('models_energy.yaml', 'my_2d_model')
file.add_time_dependence('my_2d_model', 'my_par', 'models_time.yaml', 'my_dynamics')
file.fit_2d('my_2d_model')

# Inspect results
df = file.get_fit_results(fit_type='2d')
```

For global fits, dynamics, profiles, and advanced workflows see the
[Quick Start](https://time-resolved-spectroscopy-fit.readthedocs.io/en/latest/quickstart.html)
and [Examples](https://time-resolved-spectroscopy-fit.readthedocs.io/en/latest/examples/index.html).

## Development

```bash
# Create and activate virtual environment
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
# OR Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate

# [same commands for all platforms from here on]
# Install and setup
pip install -U pip
pip install -e ".[dev]"
python -m pre_commit install --install-hooks

# Commit changes
pytest
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
