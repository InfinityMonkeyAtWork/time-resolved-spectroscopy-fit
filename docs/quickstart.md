# Quick Start

## Get running in 3 steps

**1. Install with Jupyter notebook support**

```bash
python -m pip install "trspecfit[lab]"
```

**2. Download the examples**

The [example notebooks](examples/index.rst) and data are not included in the pip package.
Download them from GitHub:

```bash
curl -L https://github.com/InfinityMonkeyAtWork/time-resolved-spectroscopy-fit/archive/refs/heads/main.zip -o trspecfit-examples.zip
unzip trspecfit-examples.zip "time-resolved-spectroscopy-fit-main/examples/*" -d .
mv time-resolved-spectroscopy-fit-main/examples ./trspecfit-examples
rm -rf time-resolved-spectroscopy-fit-main trspecfit-examples.zip
```

Or clone the full repo:

```bash
git clone https://github.com/InfinityMonkeyAtWork/time-resolved-spectroscopy-fit.git
mv time-resolved-spectroscopy-fit/examples ./trspecfit-examples
```

**3. Open the first example**

Start with **01** and work forward — each builds on the previous one.

```bash
cd trspecfit-examples/fitting_workflows/01_basic_fitting
jupyter lab example.ipynb
```

Run all cells. The notebook will load data, build a model, fit it, and plot the results.

## Examples overview

Fitting workflow examples:

| Example | What you learn |
|---------|---------------|
| 01_basic_fitting | Load data, define a model, fit baseline + 2D, visualize results |
| 02_dependent_parameters | Link parameters with expressions (e.g. spin-orbit doublets) |
| 03_multi_cycle | Multi-cycle dynamics with subcycles and frequency |
| 04_par_profiles | Depth-dependent parameters with profile functions |

Data generation examples (for testing/ML):

| Example | What you learn |
|---------|---------------|
| simulator | Generate synthetic noisy data from a known model |
| ml_training | Sweep parameter space for ML training datasets |

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

- **Peaks**: `GLP`, `Gauss`, `Lorentz`, `GLS`, `DS`
- **Backgrounds**: `LinBack`, `Shirley`, `Offset`
- **Time dynamics**: `expFun`, `expRiseFun`, `SinFun`, `SinDivX`
- **Convolution kernels**: `gaussCONV`, `lorentzCONV`, `expDecayCONV`, `expRiseCONV`
- **Profile functions**: `pExpDecay`, `pLinear`, `pGauss`

See the [API Reference](api/index.rst) for all function signatures and
parameter descriptions.

## Model composition rules

- **Supported:** base parameter -> profile, then profile parameter -> dynamics
- **Disallowed:** profile + dynamics on the same base parameter
  (disabled to avoid strongly correlated fits)

## Next steps

- [Examples](examples/index.rst) — full notebooks with real data
- [API Reference](api/index.rst) — complete method and function documentation
