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

Fitting Workflow
~~~~~~~~~~~~~~~~

.. automethod:: trspecfit.trspecfit.File.define_baseline
.. automethod:: trspecfit.trspecfit.File.set_fit_limits
.. automethod:: trspecfit.trspecfit.File.fit_baseline
.. automethod:: trspecfit.trspecfit.File.fit_SliceBySlice
.. automethod:: trspecfit.trspecfit.File.add_time_dependence
.. automethod:: trspecfit.trspecfit.File.fit_2Dmodel

Utility Methods
~~~~~~~~~~~~~~~

.. note::
   These methods are typically called automatically by the fitting workflow.
   Most users won't need to call these directly.

.. automethod:: trspecfit.trspecfit.File.model_list_to_name
.. automethod:: trspecfit.trspecfit.File.create_model_path
.. automethod:: trspecfit.trspecfit.File.save_SliceBySlice_fit
.. automethod:: trspecfit.trspecfit.File.save_2Dmodel_fit