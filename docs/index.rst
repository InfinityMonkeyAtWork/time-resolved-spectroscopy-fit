trspecfit — 2D Time- and Energy-resolved Spectroscopy Fitting
==============================================================

A Python library for fitting multi-component spectral models to time-resolved spectroscopy data.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   examples/index
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
* Fit models with lmfit and wrappers for confidence intervals and optional MCMC sampling
* Simulate individual 1D/2D spectra to validate models
* Generate ML training data via parameter space exploration

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`