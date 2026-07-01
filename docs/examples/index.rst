Examples
========

The examples are organized by **what you're trying to do**, not in a linear
"walk forward" order. Pick the track that matches your task. Each notebook is
self-contained.

Fitting Workflows
-----------------

Single-file fitting skills (block 0x):

- ``examples/fitting_workflows/01_basic_fitting/``
  Load data, define a model, fit baseline + 2D, visualize results.
- ``examples/fitting_workflows/02_dependent_parameters/``
  Link parameters with expressions and physical constraints.
- ``examples/fitting_workflows/03_multi_cycle_dynamics/``
  Multi-cycle dynamics with subcycles and frequency.
- ``examples/fitting_workflows/04_parameter_profiles/``
  Profile-aware fitting with auxiliary-axis parameter profiles and profile-parameter dynamics.

Post-fit work (block 1x):

- ``examples/fitting_workflows/10_model_comparison/``
  Compare two models on the same file (baseline / SbS / 2D).
- ``examples/fitting_workflows/11_save_load_export/``
  ``FitResults`` HDF5 round-trip, CSV/PNG export, "ship just the winners".
- ``examples/fitting_workflows/12_uncertainty_mcmc/``
  Three tiers of parameter uncertainty (``stderr``, profiled CIs, MCMC),
  checked against truth.

Multi-file workflows (block 2x):

- ``examples/fitting_workflows/20_multi_file_independent_fit/``
  Multi-file workspace, per-file independent fits (bridge to shared-parameter fitting).
- ``examples/fitting_workflows/21_multi_file_shared_fit/``
  Multi-file workspace with shared parameters across files.

Synthetic Data
--------------

- ``examples/synthetic_data/01_simulator/``
  Synthetic noisy data generation from a known ground truth.
- ``examples/synthetic_data/02_ml_training_data/``
  Training dataset generation via parameter-space exploration.

Choose Your Track
-----------------

- New user, one processed file: start at
  `01 basic fitting notebook <../../examples/fitting_workflows/01_basic_fitting/example.ipynb>`_.
- Comparing two models on one file:
  `10 model comparison notebook <../../examples/fitting_workflows/10_model_comparison/example.ipynb>`_.
- Saving, loading, or exporting fit results:
  `11 save / load / export notebook <../../examples/fitting_workflows/11_save_load_export/example.ipynb>`_.
- Estimating uncertainties with MCMC:
  `12 uncertainty (MCMC) notebook <../../examples/fitting_workflows/12_uncertainty_mcmc/example.ipynb>`_.
- Many files, fit each independently:
  `20 multi-file independent fit notebook <../../examples/fitting_workflows/20_multi_file_independent_fit/example.ipynb>`_.
- Many files, shared-parameter fit:
  `21 multi-file shared fit notebook <../../examples/fitting_workflows/21_multi_file_shared_fit/example.ipynb>`_.
- Simulation or ML training data:
  `Simulator data generation notebook <../../examples/synthetic_data/01_simulator/example.ipynb>`_
  or `ML training data generation notebook <../../examples/synthetic_data/02_ml_training_data/example.ipynb>`_.

All Notebooks
-------------

- `01 basic fitting <../../examples/fitting_workflows/01_basic_fitting/example.ipynb>`_
- `02 dependent parameters <../../examples/fitting_workflows/02_dependent_parameters/example.ipynb>`_
- `03 multi-cycle dynamics <../../examples/fitting_workflows/03_multi_cycle_dynamics/example.ipynb>`_
- `04 parameter profiles <../../examples/fitting_workflows/04_parameter_profiles/example.ipynb>`_
- `10 model comparison <../../examples/fitting_workflows/10_model_comparison/example.ipynb>`_
- `11 save / load / export <../../examples/fitting_workflows/11_save_load_export/example.ipynb>`_
- `12 uncertainty (MCMC) <../../examples/fitting_workflows/12_uncertainty_mcmc/example.ipynb>`_
- `20 multi-file independent fit <../../examples/fitting_workflows/20_multi_file_independent_fit/example.ipynb>`_
- `21 multi-file shared fit <../../examples/fitting_workflows/21_multi_file_shared_fit/example.ipynb>`_
- `Simulator data generation <../../examples/synthetic_data/01_simulator/example.ipynb>`_
- `ML training data generation <../../examples/synthetic_data/02_ml_training_data/example.ipynb>`_
