# Quick Start

## Get running

These steps assume you have `trspecfit[lab]` installed — see [Installation](installation.md) if not.

If you followed the step-by-step install guide, you already have the repo (and the [example notebooks](examples/index.rst)) on disk. If you installed from PyPI instead, clone the repo to get the examples:

```bash
git clone https://github.com/InfinityMonkeyAtWork/time-resolved-spectroscopy-fit.git
```

**Open the first example.** New users should start with `01_basic_fitting`. Past that, examples are organized by **what you're trying to do**, not in a linear order — pick the track that matches your task (see [Examples](examples/index.rst) for the full track list and notebook legend).

```bash
cd time-resolved-spectroscopy-fit/examples/fitting_workflows/01_basic_fitting
jupyter lab example.ipynb
```

Run all cells. The notebook will load data, build a model, fit it, and plot the results.

## Examples overview

Fitting workflow examples — **0x = single-file skills, 1x = post-fit work, 2x = multi-file**:

| Example | What you learn |
|---------|---------------|
| 01_basic_fitting | Load data, define a model, fit baseline + 2D, visualize results |
| 02_dependent_parameters | Link parameters with expressions (e.g. spin-orbit doublets) |
| 03_multi_cycle_dynamics | Multi-cycle dynamics with subcycles and frequency |
| 04_parameter_profiles | Depth-dependent parameters with profile functions |
| 10_model_comparison | Compare two models on the same file (baseline / SbS / 2D) |
| 11_save_load_export | `FitResults` HDF5 round-trip, CSV/PNG export, ship just the winners |
| 12_uncertainty_mcmc | MCMC-based uncertainty estimation on a fitted 2D model |
| 20_fit_each_separately | Multi-file workspace, per-file independent fits |
| 21_project_level_shared_fit | Multi-file workspace, shared-parameter fits across files |

Synthetic-data examples (for testing/ML):

| Example | What you learn |
|---------|---------------|
| synthetic_data/01_simulator | Generate synthetic noisy data from a known model |
| synthetic_data/02_ml_training_data | Sweep parameter space for ML training datasets |

## Typical workflow

Every fitting workflow follows the same pattern:

```
1. Project(path=...)                    # set up workspace
2. File(data=..., energy=..., time=...) # load your data
3. file.define_baseline(...)            # select baseline time region
4. file.load_model('models.yaml', ...)  # load baseline model from YAML
5. file.fit_baseline(...)               # fit the baseline spectrum
6. file.load_model('models.yaml', ...)  # load 2D energy model from YAML
7. file.add_time_dependence(...)        # make a parameter evolve in time
8. file.fit_2d(...)                     # global 2D fit
9. file.get_fit_results(fit_type='2d')  # extract results as DataFrame
```

Optional extension after step 6:
`file.add_par_profile(...)` — make a parameter vary over an auxiliary axis

## YAML model format

Models are defined in YAML files. Each file can contain multiple named models.
Each model lists components (functions) and their parameters:

```yaml
model_name:
    ComponentFunction:
      param: [initial_value, vary?, min, max]  # full form
      param: [initial_value, False]             # fixed parameter (short form)
      param: ["expression"]                     # expression dependency
```

**Available component types:**

- **Peaks**: `Gauss`, `GaussAsym`, `Lorentz`, `Voigt`, `GLP`, `GLS`, `DS`
- **Backgrounds**: `LinBack`, `Shirley`, `Offset`
- **Time dynamics**: `linFun`, `expFun`, `sinFun`, `sinDivX`, `erfFun`, `sqrtFun`
- **Convolution kernels**: `gaussCONV`, `lorentzCONV`, `expDecayCONV`, `expRiseCONV`
- **Profile functions**: `pExpDecay`, `pLinear`, `pGauss`

See the [API Reference](api/index.rst) for all function signatures and
parameter descriptions.

## Model composition rules

- **Supported:** base parameter -> profile, then profile parameter -> dynamics
- **Disallowed:** profile + dynamics on the same base parameter
  (disabled to avoid strongly correlated fits)

See [Supported Models](design/supported_models.md) for the full reference on
supported model combinations, expression semantics, and edge cases.
