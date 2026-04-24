trspecfit — 2D Time- and Energy-resolved Spectroscopy Fitting
==============================================================

A Python library for fitting multi-component spectral models to time-resolved spectroscopy data.

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   installation
   quickstart
   examples/index
   design/supported_models

.. toctree::
   :maxdepth: 2
   :caption: Developer Info:

   design/repo_architecture
   design/lowered_evaluator
   ai/index

.. toctree::
   :maxdepth: 2
   :caption: Reference:

   api/index

Overview
--------

**trspecfit** is a Python library to define and fit multi-component spectral models for 1D 
energy-resolved and 2D time-and-energy-resolved spectroscopy data. The package extends 
lmfit-based workflows to support component-based models, time-dynamics, convolution kernels 
and helpers for generating simulated data.

Key Capabilities
----------------

* Define modular spectral components (Gaussian, Voigt/GLP/GLS, Doniach–Sunjic, backgrounds, convolution kernels)
* Build 1D (energy) and 2D (time × energy) spectra from components with time-dynamics
* Attach auxiliary-axis parameter profiles (e.g. depth/position) and compose with dynamics
* Fit models with lmfit and wrappers for confidence intervals and optional MCMC sampling
* Simulate individual 1D/2D spectra to validate models
* Generate ML training data via parameter space exploration

Design Guidance
---------------

For the current supported model-composition rules and expression semantics,
see :doc:`design/supported_models`. This is the reference guide for which
model combinations are intended to work today, which are explicitly excluded,
and which edge cases are tolerated for now but may change in the future.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
