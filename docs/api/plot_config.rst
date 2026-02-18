Plot Configuration
==================

The :class:`trspecfit.config.plot.PlotConfig` dataclass controls plotting
appearance and behavior across trspecfit.

Use PlotConfig to keep plotting settings consistent across a project while
still allowing local overrides for specific plots.

Usage Patterns
--------------

Basic usage
^^^^^^^^^^^

Create a standalone PlotConfig and pass it to plotting helpers.

.. code-block:: python

   from trspecfit import PlotConfig
   from trspecfit.utils import plot as uplt

   # Create config with custom settings
   config = PlotConfig(x_label='Energy (eV)', x_dir='rev', dpi_plot=150)
   uplt.plot_1D(data, x, config=config)

Using project settings
^^^^^^^^^^^^^^^^^^^^^^

Load defaults from a ``Project`` (including values from ``project.yaml``), then
reuse the resulting config across plots.

.. code-block:: python

   from trspecfit import Project, PlotConfig
   from trspecfit.utils import plot as uplt

   project = Project(path='my_project', config_file='project.yaml')
   config = PlotConfig.from_project(project)

   # All plots use project settings
   uplt.plot_1D(data, x, config=config)
   uplt.plot_2D(data, x, y, config=config)

Per-plot overrides
^^^^^^^^^^^^^^^^^^

Override selected settings in an individual plot call without changing the
underlying config object.

.. code-block:: python

   uplt.plot_1D(data, x, config=config, x_dir='rev', colors=['red', 'blue'])

Create alternate configs
^^^^^^^^^^^^^^^^^^^^^^^^

Create lightweight variations from a base config for publication, talks, or
interactive analysis.

.. code-block:: python

   default_config = PlotConfig.from_project(project)
   pub_config = default_config.copy(dpi_save=600, dpi_plot=150)
   talk_config = default_config.copy(ticksize=14)

API Reference
-------------

.. automodule:: trspecfit.config.plot
   :members:
   :show-inheritance:
