# trspecfit — 2D Time- and Energy-resolved Spectroscopy Fitting

**A Python library for fitting multi-component spectral models to time-resolved spectroscopy data.**

trspecfit is a Python library to define and fit multi-component spectral models for 1D energy-resolved and 2D time-and-energy-resolved spectroscopy data. The package extends lmfit-based workflows to support component-based models, time-dynamics, convolution kernels and helpers for generating simulated data.

## Key capabilities
- Define modular spectral components (Gaussian, Voigt/GLP/GLS, Doniach–Sunjic,
	backgrounds, convolution kernels).
- Build 1D (energy) and 2D (time × energy) spectra from components and
	attach time-dynamics to individual model parameters.
- Fit models with lmfit and wrappers for confidence intervals and optional
	MCMC sampling (lmfit.emcee).

## Repository layout
- `src/trspecfit/` — library source code (core engine, functions, and utils)
- `examples/` — Jupyter notebooks and YAML model files that demonstrate usage
- `tests/` — pytest unit tests for core functionality
- `pyproject.toml` — project metadata and dependency declaration

## Installation

### Install from GitHub
```bash
pip install git+https://github.com/InfinityMonkeyAtWork/time-resolved-spectroscopy-fit.git
```

### Install for development
For local development, clone the repository and install in editable mode:
<br>
Windows PowerShell example:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate
pip install -U pip
pip install -e .
```

macOS / Linux example:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Example Jupyter notebooks and YAML files

Open the notebooks in `examples/` for runnable examples:
- `examples/simulator/example.ipynb` — simulate noisy datasets based on your model input
- `examples/dependent_parameters/example.ipynb` — parameters of one peak depend on another peak
- `examples/subcycles/example.ipynb` — multiple subcycles inside one pump-probe cycle

## Fitting

Use `trspecfit.fitlib.residual_fun` together with `fitlib.fit_wrapper` to run
fits using lmfit. The wrapper supports sequential optimization, `lmfit.conf_interval`
and optional MCMC sampling with `lmfit.emcee` for robust uncertainty estimates.

## Testing

Run the unit tests with pytest (install pytest if needed):

```powershell
pip install pytest
pytest -q
```

# Copyright Notice

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