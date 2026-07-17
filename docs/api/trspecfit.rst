Main Module
===========

Project Class
-------------

.. autoclass:: trspecfit.trspecfit.Project
   :members:
   :undoc-members:
   :show-inheritance:

File Class
----------

The File class provides a complete workflow for spectroscopy data analysis.
Methods are organized below in the typical order of use:

Data Inspection
~~~~~~~~~~~~~~~

.. automethod:: trspecfit.trspecfit.File.__init__
.. automethod:: trspecfit.trspecfit.File.describe

Model Management
~~~~~~~~~~~~~~~~

.. automethod:: trspecfit.trspecfit.File.load_model
.. automethod:: trspecfit.trspecfit.File.set_active_model
.. automethod:: trspecfit.trspecfit.File.describe_model
.. automethod:: trspecfit.trspecfit.File.select_model
.. automethod:: trspecfit.trspecfit.File.delete_model
.. automethod:: trspecfit.trspecfit.File.reset_models

Data Corrections
~~~~~~~~~~~~~~~~

.. automethod:: trspecfit.trspecfit.File.subtract_dark
.. automethod:: trspecfit.trspecfit.File.calibrate_data
.. automethod:: trspecfit.trspecfit.File.reset_dark
.. automethod:: trspecfit.trspecfit.File.reset_calibration

Fitting Workflow
~~~~~~~~~~~~~~~~

.. note::
   Recommended composition for profile-aware 2D models is serial:
   attach a profile to a base parameter with ``add_par_profile()``, then
   attach dynamics to a profile parameter with ``add_time_dependence()``.
   Adding profile and dynamics directly to the same base parameter is
   currently disabled to avoid strongly correlated fits.

.. automethod:: trspecfit.trspecfit.File.define_baseline
.. automethod:: trspecfit.trspecfit.File.set_fit_limits
.. automethod:: trspecfit.trspecfit.File.fit_baseline
.. automethod:: trspecfit.trspecfit.File.fit_slice_by_slice
.. automethod:: trspecfit.trspecfit.File.add_par_profile
.. automethod:: trspecfit.trspecfit.File.add_time_dependence
.. automethod:: trspecfit.trspecfit.File.fit_2d
.. automethod:: trspecfit.trspecfit.File.fit_spectrum

Results, Plotting, and Persistence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These read the persisted fit record (latest matching fit), so they work
identically on a live session and on loaded archives — see
:class:`trspecfit.fit_results.FitResults` for the underlying API.

.. automethod:: trspecfit.trspecfit.File.get_fit_results
.. automethod:: trspecfit.trspecfit.File.get_correlations
.. automethod:: trspecfit.trspecfit.File.get_conf_intervals
.. automethod:: trspecfit.trspecfit.File.get_mcmc
.. automethod:: trspecfit.trspecfit.File.plot_fit
.. automethod:: trspecfit.trspecfit.File.plot_param_evolution
.. automethod:: trspecfit.trspecfit.File.compare_models
.. automethod:: trspecfit.trspecfit.File.save_fit
.. automethod:: trspecfit.trspecfit.File.export_fit

Utility Methods
~~~~~~~~~~~~~~~

.. note::
   These methods are typically called automatically by the fitting workflow.
   Most users won't need to call these directly.

.. automethod:: trspecfit.trspecfit.File.model_list_to_name
.. automethod:: trspecfit.trspecfit.File.model_path
