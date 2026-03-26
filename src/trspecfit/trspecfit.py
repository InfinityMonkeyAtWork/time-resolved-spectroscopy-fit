"""
Project and file management for spectroscopy data analysis.

This module provides high-level classes for organizing and managing spectroscopy
fitting workflows. It handles project configuration, data loading, model management,
and orchestrates the complete fitting pipeline from data import to result export.

Core Classes
------------
Project : Project configuration and directory management
    Manages project-wide settings including plot configuration, file I/O formats,
    and fitting parameters. Supports YAML-based configuration for reproducible
    workflows across different users and spectroscopy types.

File : Data container with model fitting capabilities
    Represents a single spectroscopy dataset (1D or 2D) with its associated
    energy/time/auxiliary axes. Manages multiple models for comparison, handles
    baseline extraction, and provides methods for baseline fitting, Slice-by-Slice
    fitting, and full 2D time- and energy-resolved fitting. Supports parameter
    profiles over an auxiliary axis via Profile models.

Workflow
--------
1. **Setup**: Create Project with configuration settings
2. **Load Data**: Create File object with data, energy, time, and optionally aux_axis
3. **Define Models**: Load models from YAML files using File.load_model()
4. **Attach Profiles** (optional): Add parameter profiles with File.add_par_profile()
5. **Add Dynamics** (optional): Add time-dependence with File.add_time_dependence()
6. **Set Limits**: Define fitting regions with File.set_fit_limits()
7. **Fit**: Execute appropriate fitting method:
   - File.fit_baseline() for ground state spectrum
   - File.fit_slice_by_slice() for time-independent analysis
   - File.fit_2d() for global time- and energy-resolved fitting
8. **Export**: Results automatically saved to project directory structure

Features
--------
- Hierarchical project/file organization with automatic directory creation
- Multiple model management and comparison within single File
- YAML-based project and model configuration
- Flexible plot customization via PlotConfig
- Support for 1D (energy-only) and 2D (time + energy) datasets
- Parameter profiles over auxiliary axis (e.g. depth) via Profile models
- Baseline/pre-trigger spectrum extraction and fitting
- Slice-by-Slice time-series fitting with parameter evolution tracking
- Global 2D fitting with time-dependent parameters
- Automatic result export (parameters, plots, confidence intervals)

Examples
--------
See examples/ directory for complete workflows.
"""

import pathlib
import time
import warnings
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal, cast

# TYPE_CHECKING is False at runtime so this import is skipped during execution.
# Exists only for type checkers (mypy/pyright) to resolve types.ModuleType annotations.
if TYPE_CHECKING:
    import types

import numpy as np
import pandas as pd
from IPython.display import display
from ruamel.yaml import YAML

from trspecfit import fitlib, mcp, spectra

# standardized plotting configuration
from trspecfit.config.plot import PlotConfig

# function library for energy, time, and profile components
from trspecfit.functions import energy as fcts_energy
from trspecfit.functions import profile as fcts_profile
from trspecfit.functions import time as fcts_time
from trspecfit.utils import arrays as uarr
from trspecfit.utils import lmfit as ulmfit
from trspecfit.utils import parsing as uparsing
from trspecfit.utils import plot as uplt

PathLike = str | pathlib.Path
ModelRef = str | int | list[str]


# multi-subcycle models allow for convolution only in the "0th subcycle"
# i.e. first model_info element which affects all times t.
# "conv" functions in individual subcycles are currently ignored


#
#
class Project:
    """
    Project-wide configuration and directory structure management.

    Project provides centralized configuration for spectroscopy analysis workflows.
    It manages project directories, plot settings, file I/O formats, and analysis
    parameters. Configuration can be customized via YAML files to support different
    users, instruments, or spectroscopy techniques.

    Parameters
    ----------
    path : str or Path
        Base directory for project data files and YAML configuration.
        If None, defaults to 'test' directory.
    name : str, default='test'
        Name for this analysis run. Creates subdirectory in results folder.
    config_file : str or Path, optional
        YAML configuration file name (located in path directory).
        If None, uses default settings only.

    Attributes
    ----------
    path : Path
        Base project directory containing data and configuration
    path_results : Path
        Results directory (path + '_fits' suffix)
    path_run : Path
        Directory for this specific run (path_results / name)
    name : str
        Name for this analysis run
    files : list of File
        All File instances registered with this Project
    show_output : int
        Verbosity level:

        - 0: Silent / programmatic / API mode -- no prints, no plots
          displayed or saved
        - 1: Interactive / notebook / UI mode -- show timing, fit results,
          save plots and data
    spec_lib : module
        Module containing spectrum fitting functions (default: spectra)
    spec_fun_str : str
        Name of fitting function in spec_lib
    skip_first_n_spec, first_n_spec_only : int
        Slice selection for partial fitting (-1 = all slices)

    Notes
    -----
    **Plot Configuration:**
    Plot-related attributes (axis labels, directions, colormaps, DPI, etc.)
    are used to construct PlotConfig objects. See trspecfit.config.plot.PlotConfig
    for full documentation of available plot settings.

    **YAML Configuration:**
    Create a YAML file in the project directory to override defaults.
    Only specified settings need to be included; others use defaults.

    **File I/O Settings:**
    Attributes ext, fmt, delim, da_fmt, and da_slices_fmt control file
    export formats and can be customized per project via YAML or direct
    attribute assignment.
    """

    #
    def __init__(
        self,
        path: PathLike | None,
        name: str = "test",
        config_file: PathLike | None = "project.yaml",
    ) -> None:
        self.path = pathlib.Path(path) if path is not None else pathlib.Path("test")
        self.path_results = pathlib.Path(f"{path}_fits")
        self.name = name
        self.path_run = self.path_results / name

        self._config_file: PathLike | None = None
        self.files: list[File] = []

        # Set defaults first
        self._set_defaults()

        # Override with YAML config if provided
        if config_file is not None:
            self._load_config(config_file)

    #
    def _set_defaults(self) -> None:
        """Set default project configuration."""

        self.show_output = 1
        # Plot settings
        self.e_label = "Energy"
        self.t_label = "Time"
        self.z_label = "Intensity"
        self.x_dir = "def"
        self.x_type = "lin"
        self.y_dir = "def"
        self.y_type = "lin"
        self.z_colormap = "viridis"
        self.z_colorbar = "ver"
        self.z_type = "lin"
        self.dpi_plt = 100
        self.dpi_save = 300
        self.res_mult = 5
        # File I/O settings
        self.ext = ".dat"
        self.fmt = "%.6e"
        self.delim = ","
        self.da_fmt = "%04d"
        self.da_slices_fmt = "%06d"
        # Advanced settings
        self.spec_lib = spectra
        self.spec_fun_str = "fit_model_mcp"
        self.skip_first_n_spec = -1
        self.first_n_spec_only = -1

    @property
    def spec_fun(self) -> Callable:
        """
        Dynamically get the spectrum fitting function.
        """

        return cast("Callable", getattr(self.spec_lib, self.spec_fun_str))

    #
    def __repr__(self) -> str:
        return f"Project(path='{self.path}', name='{self.name}')"

    #
    def describe(self, detail: int = 0) -> None:
        """
        Display project configuration summary.

        Parameters
        ----------
        detail : int, default=0
            Verbosity level.
            0: project paths and config source.
            1: also list attached Files (path, dim, shape, models).
            2: also show plot and file I/O settings.
        """

        print("Project")
        print(f"  path:         {self.path}")
        print(f"  results:      {self.path_results}")
        print(f"  name:         {self.name}")
        if self._config_file is not None:
            print(f"  config:       {self._config_file}")
        else:
            print("  config:       defaults (no YAML loaded)")

        if detail >= 1:
            print(f"\n  Files ({len(self.files)}):")
            if not self.files:
                print("    (none)")
            for f in self.files:
                shape = f"shape {f.data.shape}" if f.data is not None else "no data"
                n_models = len(f.models)
                model_names = (
                    ", ".join(m.name for m in f.models) if n_models else "none"
                )
                active = (
                    f" [active: {f.model_active.name}]"
                    if f.model_active is not None
                    else ""
                )
                aux = (
                    f", aux_axis len {len(f.aux_axis)}"
                    if f.aux_axis is not None
                    else ""
                )
                print(
                    f"    {f.path}: {f.dim}D {shape}{aux}, "
                    f"models ({n_models}): {model_names}{active}"
                )

        if detail >= 2:
            print("\n  Plot settings:")
            print(f"    e_label:    {self.e_label}")
            print(f"    t_label:    {self.t_label}")
            print(f"    z_label:    {self.z_label}")
            print(f"    x_dir:      {self.x_dir}")
            print(f"    x_type:     {self.x_type}")
            print(f"    y_dir:      {self.y_dir}")
            print(f"    y_type:     {self.y_type}")
            print(f"    z_colormap: {self.z_colormap}")
            print(f"    z_colorbar: {self.z_colorbar}")
            print(f"    z_type:     {self.z_type}")
            print(f"    dpi_plt:    {self.dpi_plt}")
            print(f"    dpi_save:   {self.dpi_save}")
            print(f"    res_mult:   {self.res_mult}")
            print("\n  File I/O settings:")
            print(f"    ext:        {self.ext}")
            print(f"    fmt:        {self.fmt}")
            print(f"    delim:      {repr(self.delim)}")
            print(f"    da_fmt:     {self.da_fmt}")
            print(f"    DA_slices:  {self.da_slices_fmt}")

    #
    def _load_config(self, config_file: PathLike) -> None:
        """
        Load project configuration from YAML file.
        Allow different users or types of spectroscopy to overwrite project attributes

        Parameters
        ----------
        config_file : str or Path
            Name or path of config file (looks in self.path)
        """

        yaml = YAML(typ="safe")
        config_path = self.path / config_file

        try:
            with config_path.open() as f:
                config = yaml.load(f)

            if config is None:
                if self.show_output >= 1:
                    print(f"Warning: {config_file} is empty, using defaults")
                return

            # Update attributes from config
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    if self.show_output >= 1:
                        print(f"Warning: Unknown config key '{key}' ignored")

            self._config_file = config_path

        except FileNotFoundError:
            if self.show_output >= 1:
                print(f"Config file {config_path} not found, using defaults")
        except Exception as e:  # noqa: BLE001
            if self.show_output >= 1:
                print(f"Error loading config: {e}")
                print("Using default settings")


#
#
class File:
    """
    Data container with model management and fitting capabilities.

    File represents a single spectroscopy dataset (1D or 2D) with associated
    energy and time axes. It manages multiple models for comparison, handles
    baseline extraction, set fitting limits, and orchestrates baseline fitting,
    Slice-by-Slice fitting, and full 2D fitting workflows.

    Parameters
    ----------
    parent_project : Project, optional
        Parent Project instance for configuration. If None, creates default
        test project internally.
    path : str or Path, default='test'
        Identifier for this file (used in result directory structure and plots)
    data : ndarray, optional
        Spectroscopy data to fit:

        - 1D: shape [n_energy] for energy-resolved spectra
        - 2D: shape [n_time, n_energy] for time- and energy-resolved data

    energy : array-like, optional
        Energy axis values. If None and data provided, uses indices.
    time : array-like, optional
        Time axis values (for 2D data only). If None and 2D data provided,
        uses indices.
    aux_axis : array-like, optional
        Auxiliary physical axis (e.g. depth, position) for Profile models.
        Propagated to models loaded via ``load_model()``.

    Attributes
    ----------
    p : Project
        Parent project providing configuration
    path : str or Path
        File identifier
    path_da : Path
        Directory path for saving this file's fit results
    data : ndarray
        Spectroscopy data (1D or 2D)
    dim : int
        Data dimensionality (1 or 2)
    energy : ndarray
        Energy axis
    time : ndarray or None
        Time axis (None for 1D data)
    aux_axis : ndarray or None
        Auxiliary physical axis for Profile models (None if unused)
    models : list of Model
        All models loaded for this file
    model_active : Model or None
        Currently active model (default for operations)
    e_lim_abs, e_lim : list
        Energy fitting limits (absolute values and indices)
    t_lim_abs, t_lim : list
        Time fitting limits (absolute values and indices)
    data_base : ndarray or None
        Baseline spectrum (averaged from specified time range)
    base_t_abs, base_t_ind : list
        Time range for baseline extraction (absolute and indices)
    model_base : Model or None
        Model used for baseline fitting
    model_sbs : Model or None
        Model used for Slice-by-Slice fitting
    results_sbs : list
        Slice-by-Slice fit results for all time slices
    plot_config : PlotConfig
        Plot configuration (created from parent Project on first access)

    Notes
    -----
    **Typical Workflow:**
    1. Create File with data, energy, time (and optionally aux_axis)
    2. Load model(s) with load_model()
    3. Set active model with set_active_model()
    4. Optionally attach Profile models with add_par_profile()
    5. Optionally add time-dependence with add_time_dependence()
    6. Define baseline with define_baseline() (for 2D data)
    7. Set fit limits with set_fit_limits()
    8. Fit baseline with fit_baseline()
    9. Perform Slice-by-Slice or 2D fit with fit_slice_by_slice() or fit_2d()

    **Model Management:**
    File can manage multiple models simultaneously for comparison. Use
    select_model() to retrieve specific models by name or index, and
    set_active_model() to change which model is used by default.

    **Baseline Spectrum:**
    For 2D data, extract a baseline/pre-trigger/ground-state reference
    spectrum using define_baseline(). This is required before fitting
    time-dependent data.
    """

    #
    def __init__(
        self,
        parent_project: Project | None = None,
        path: PathLike = "test",
        data: np.ndarray | None = None,
        energy: np.ndarray | None = None,
        time: np.ndarray | None = None,
        aux_axis: np.ndarray | None = None,
    ) -> None:
        # pass parent project or (default) create a functioning test project environment
        self.p = parent_project if parent_project is not None else Project(path=None)
        self.p.files.append(self)  # register with parent project
        self.path = path  # path to load/save [?] data from
        self.path_da = self.p.path_run / path  # path to save fit results to
        self._plot_config: PlotConfig | None = None  # create plot config from project
        self.data = data  # (time-[optional] and) energy-dependent data to fit
        self.dim = 0 if data is None else data.ndim  # 1/2 D for energy/+time
        # take energy and time input or create a generic axis if None is passed
        if energy is not None or data is None:
            self.energy = energy
        elif self.dim == 1:
            self.energy = np.arange(0, data.shape[0])
        else:
            self.energy = np.arange(0, data.shape[1])
        if time is not None or self.dim <= 1 or data is None:
            self.time = time
        else:
            self.time = np.arange(0, data.shape[0])
        self.aux_axis: np.ndarray | None = (
            aux_axis  # auxiliary physical axis (e.g. depth)
        )
        # keep track of models that are used to fit this file/data
        self.models: list[mcp.Model] = []
        self.model_active: mcp.Model | None = None  # default model to work with
        # Energy and time limits for fitting methods
        self.e_lim_abs: list[float] = []  # energy limits (low, high) user-defined
        self.e_lim: list[int] = []  # index [start, stop) for energy[start:stop]
        self.t_lim_abs: list[float] = []  # time limits (low, high) user-defined
        self.t_lim: list[int] = []  # index [start, stop) for time[start:stop]
        #
        self.base_t_abs: list[
            float
        ] = []  # start and stop time of the baseline spectrum
        self.base_t_ind: list[int] = []  # index of the above start and stop time
        self.data_base = None  # average spectrum between above indices
        self.model_base: mcp.Model | None = None
        #
        self.model_sbs: mcp.Model | None = None
        self.model_2d: mcp.Model | None = None
        # all Slice-by-Slice fit results (different from model_sbs.result)
        self.results_sbs: list = []
        # default fit limits to entire dataset (energy is None only for bare File())
        if self.energy is not None:
            self.set_fit_limits(energy_limits=None, show_plot=False)

    @property
    def plot_config(self) -> PlotConfig:
        """
        Get plot config for this File.

        Created from parent Project on first access. File can then customize
        persistently (e.g., for different time axes across files).
        """

        if self._plot_config is None:
            self._plot_config = PlotConfig.from_project(self.p)
        return self._plot_config

    @plot_config.setter
    def plot_config(self, config: PlotConfig) -> None:
        """Allow setting a custom config for this File"""

        self._plot_config = config

    #
    def __repr__(self) -> str:
        shape = self.data.shape if self.data is not None else None
        n_models = len(self.models)
        return (
            f"File(path='{self.path}', data={shape}, {n_models} models, dim={self.dim})"
        )

    #
    def describe(self) -> None:
        """
        Display information about this file's data.

        Plots the data with current fit limits indicated by vertical and
        horizontal lines. For 1D data, shows energy spectrum. For 2D data,
        shows time- and energy-resolved map.
        """

        print(f"File # x [path: {self.path}]")
        if self.data is None:
            warnings.warn("No data loaded; nothing to describe.", stacklevel=2)
            return
        if self.energy is None:
            self.energy = np.arange(self.data.shape[-1])
            warnings.warn("Energy axis missing; using index axis.", stacklevel=2)
        if self.dim == 2 and self.time is None:
            self.time = np.arange(self.data.shape[0])
            warnings.warn("Time axis missing; using index axis.", stacklevel=2)

        config = self.plot_config

        if self.dim == 1:
            uplt.plot_1d(
                data=[
                    self.data,
                ],
                x=self.energy,
                config=config,
                vlines=self.e_lim_abs,
            )

        elif self.dim == 2:
            uplt.plot_2d(
                data=self.data,
                x=self.energy,
                y=self.time,
                config=config,
                vlines=self.e_lim_abs,
                hlines=self.t_lim_abs,
            )

    #
    def model_list_to_name(self, model_list: Sequence[str]) -> str:
        """
        Create composite model name from list of submodel names.

        Joins individual model names with underscores. Used primarily for
        mcp.Dynamics models with multiple subcycles. For single-element lists,
        returns that element unchanged.

        Parameters
        ----------
        model_list : list of str
            List of model names to combine

        Returns
        -------
        str
            Combined model name (e.g., ['model1', 'model2'] -> 'model1_model2')
        """

        return "_".join(model_list)  # see str.join()

    #
    def select_model(self, model_info: ModelRef) -> mcp.Model | None:
        """
        Select model by name, index, or multi-cycle name list.

        Parameters
        ----------
        model_info : str, int, or list of str
            Model identifier: name, position in ``self.models``, or list of
            submodel names (joined with ``_`` to match composite name).

        Returns
        -------
        Model or None
            The matched model, or None if not found.
        """

        if isinstance(model_info, str):
            for m in self.models:
                if m.name == model_info:
                    return m
            return None

        if isinstance(model_info, int):
            if 0 <= model_info < len(self.models):
                return self.models[model_info]
            return None

        if isinstance(model_info, list):
            m_name = self.model_list_to_name(model_info)
            for m in self.models:
                if m.name == m_name:
                    return m

        return None

    #
    def set_active_model(self, model_info: ModelRef) -> None:
        """
        Set model to be used as active model.

        All functions requiring a model input will default to the currently active model
        unless a model is specified as input to the respective function
        (via ``model_info``).

        Parameters
        ----------
        model_info : str or int
            Model identifier (name or index)
        """

        self.model_active = self.select_model(model_info)

    #
    def load_model(
        self,
        model_yaml: PathLike,
        model_info: str | list[str],
        par_name: str = "",
        model_type: Literal["energy", "dynamics", "profile"] = "energy",
    ) -> mcp.Model:
        """
        Load a model from YAML file.

        Loads a model defined in ``model_yaml`` file located in Parent.path.

        Parameters
        ----------
        model_yaml : str or Path
            YAML file name (located in project path) defining the model
        model_info : str or list of str
            Model name(s) to load. A bare string is treated as a single-element
            list.

            - For energy-dependent models (1D or 2D): ``'model_name'`` or
              ``['model_name']`` (single element). Model will be set as active
              model.
            - For standard time-dependent models: ``'model_name'`` or
              ``['model_name']``. Single element applies to entire time axis.
            - For multi-cycle time-dependent models: ``['model1', 'model2', ...]``.
              Element 0 applies to entire time axis, elements 1+ apply to
              respective subcycles only.
            - For profile models: ``'model_name'`` or ``['model_name']``
              (single element).

        par_name : str, default=''
            Parameter name for dynamics and profile models. Empty string (default)
            indicates an energy-dependent model. For ``"dynamics"`` and ``"profile"``
            model types, must match the name of the energy model parameter that the
            loaded model will be attached to.
        model_type : {'energy', 'dynamics', 'profile'}, default='energy'
            Type of model to load:

            - ``'energy'``: energy- (and time-)dependent spectral model
            - ``'dynamics'``: time-dependence of a single model parameter
            - ``'profile'``: spatial profile of a single model parameter over aux_axis

        Returns
        -------
        Model
            The loaded model. For ``'energy'`` models the model is also
            registered in ``self.models`` and set as the active model.
        """

        # normalize bare string to single-element list
        if isinstance(model_info, str):
            model_info = [model_info]
        if model_type == "energy" and len(model_info) != 1:
            raise ValueError(
                'Energy-resolved data (model_type="energy") require a single model'
                " name in model_info.\nPass model name as the only element in the"
                " model_info list."
            )
        if model_type == "profile" and len(model_info) != 1:
            raise ValueError(
                'Profile models (model_type="profile") require a single model name'
                " in model_info."
            )
        if model_type == "energy":
            if self.select_model(model_info) is not None:
                raise ValueError(
                    f'Model named "{self.model_list_to_name(model_info)}"'
                    " already exists. Delete the existing model"
                    " or change the name of the new model."
                )

        # Load and process YAML file with appropriate numbering strategy
        model_yaml_path = self.p.path / pathlib.Path(model_yaml)
        model_info_dict = uparsing.load_and_number_yaml_components(
            model_yaml_path=model_yaml_path,
            model_info=model_info,
            is_dynamics=model_type == "dynamics",
        )

        # Initialize model
        fcts_package: types.ModuleType
        if model_type == "energy":
            if self.p.show_output >= 1:
                print(
                    f"Loading model to describe energy- (and time-)dependent data: "
                    f"{self.model_list_to_name(model_info)}"
                )
            fcts_package = fcts_energy
            loaded_model = mcp.Model(self.model_list_to_name(model_info))
        elif model_type == "dynamics":
            if self.p.show_output >= 1:
                print(
                    f"Loading model to describe time-dependence of a model parameter: "
                    f"{par_name} of {self.model_list_to_name(model_info)} model"
                )
            fcts_package = fcts_time
            loaded_model = mcp.Dynamics(par_name)
        elif model_type == "profile":
            if self.p.show_output >= 1:
                print(
                    f"Loading profile model for parameter: "
                    f"{par_name} of {self.model_list_to_name(model_info)} model"
                )
            fcts_package = fcts_profile
            loaded_model = mcp.Profile(par_name)
        else:
            raise ValueError(
                f'Model type "{model_type}" not recognized. Must be one of:'
                ' "energy", "dynamics", or "profile".'
            )

        # Inherit necessary model attributes from function input, file, and project
        loaded_model.yaml_f_name = pathlib.Path(model_yaml).stem  # yaml file name
        loaded_model.dim = 1  # start with 1, +1 when adding dynamics
        if isinstance(loaded_model, mcp.Dynamics):
            loaded_model.subcycles = len(model_info) - 1
        loaded_model.parent_file = self
        loaded_model.energy = self.energy
        loaded_model.time = self.time
        loaded_model.aux_axis = self.aux_axis

        all_comps = []  # initialize component list

        # Go through (sub)model(s)
        # (for mcp.Dynamics model instances length model_info could be larger than 1)
        for subcycle, submodel in enumerate(model_info):
            # Get the section defined by model_info
            try:
                submodel_info = model_info_dict[submodel]
            except KeyError as err:
                available_models = list(model_info_dict.keys())
                raise ValueError(
                    f'Model "{submodel}" not found in {model_yaml}\n'
                    f"Available models in this file: {available_models}\n"
                    f"Check for typos in model name."
                ) from err

            # Create components for this submodel using existing mcp.Component logic
            for c_name, c_info in submodel_info.items():
                c_temp = mcp.Component(c_name, fcts_package, subcycle)
                c_temp.add_pars(c_info)
                all_comps.append(c_temp)

        # Add all components (and their parameters) to model
        loaded_model.add_components(all_comps)

        # Add model to file
        if model_type == "energy":
            self.models.append(loaded_model)
            self.set_active_model(model_info)  # set as current active model
        return loaded_model

    #
    def describe_model(
        self, model_info: ModelRef | None = None, detail: int = 0
    ) -> None:
        """
        Display information about a specific model.

        Shows model parameters and optionally plots data with initial guess
        and residual. Useful for inspecting models before fitting.

        Parameters
        ----------
        model_info : str or int, optional
            Model identifier (name or index). If None, uses currently active model.
        detail : {0, 1}, default=0
            Level of detail:

            - 0: Show parameter table only
            - 1: Show parameters and plot data/initial guess/residual
        """

        mod = self.model_active if model_info is None else self.select_model(model_info)
        if mod is None:
            warnings.warn("Model not found; nothing to describe.", stacklevel=2)
            return

        # parameter list
        mod.describe(detail=0)

        if detail == 1 and isinstance(mod, mcp.Dynamics):
            mod.create_value_1d(store_1d=1)  # update individual component spectra
            mod.plot_1d(plot_sum=False)  # plot guess only (individual components)

        if detail == 1 and mod.dim == 1:
            if self.energy is None or self.data_base is None:
                warnings.warn(
                    "Energy axis or baseline data missing;"
                    " cannot plot 1D model summary.",
                    stacklevel=2,
                )
                return
            mod.create_value_1d(store_1d=1)  # update individual component spectra
            # plot initial guess (individual components), data, and residual
            title_mod = (
                f"File: {self.path}, "
                f'Model: "{model_info}" (from "{mod.yaml_f_name}.yaml")'
                ": initial guess"
            )
            fitlib.plt_fit_res_1d(
                x=self.energy,
                y=self.data_base,
                fit_fun_str=self.p.spec_fun_str,
                package=self.p.spec_lib,
                par_init=[],
                par_fin=mod.lmfit_pars,
                args=(mod, 1),
                plot_sum=False,
                show_init=False,
                title=title_mod,
                fit_lim=self.e_lim,
                config=self.plot_config,
                legend=[comp.name for comp in mod.components],
            )

        if detail == 1 and mod.dim == 2:
            mod.create_value_2d()  # update spectrum
            if self.data is None or mod.value_2d is None:
                warnings.warn(
                    "2D data/model values missing; cannot plot 2D model summary.",
                    stacklevel=2,
                )
                return
            # plot data, fit, and residual 2D maps
            fitlib.plt_fit_res_2d(
                data=self.data,
                fit=mod.value_2d,
                x=self.energy,
                y=self.time,
                config=self.plot_config,
                x_lim=self.e_lim,
                y_lim=self.t_lim,
            )

    #
    def delete_model(self, model_to_delete: ModelRef | None = None) -> None:
        """
        Remove a model from this file's model list.

        Parameters
        ----------
        model_to_delete : str, int, list of str, or None, optional
            Model to delete (name, index, multi-cycle name list, or None
            for the currently active model).

        Notes
        -----
        After deletion, model_active may be invalid. Set a new active model
        if needed using set_active_model().
        """

        if model_to_delete is None:
            if self.model_active is None:
                warnings.warn("No active model to delete.", stacklevel=2)
                return
            self.models.remove(self.model_active)
            return

        mod = self.select_model(model_to_delete)
        if mod is None:
            warnings.warn(
                f"delete_model: Model {model_to_delete!r} not found.",
                stacklevel=2,
            )
            return
        self.models.remove(mod)

    #
    def reset_models(self) -> None:
        """
        Remove all models from this file.

        Clears the models list and resets model_active to None. Use this
        to start fresh with model loading without creating a new File object.
        """

        self.models = []

    #
    def create_model_path(
        self, model_name: str, subfolders: list[str] | None = None
    ) -> pathlib.Path:
        """
        Create directory structure for saving model fit results.

        Constructs path based on file path, YAML file name, and model name.
        Creates directories if they don't exist.

        Parameters
        ----------
        model_name : str
            Name of model (must exist in self.models)
        subfolders : list of str, default=[]
            Additional subdirs to create (e.g., ['slices'] for Slice-by-Slice fits)

        Returns
        -------
        Path
            Path to model results directory
        """

        mod = self.select_model(model_name)  # get model
        if mod is None:
            warnings.warn(
                f"Model '{model_name}' not found; using fallback output path.",
                stacklevel=2,
            )
            path_model = self.path_da / "model_unknown" / model_name
        else:
            yaml_name = (
                mod.yaml_f_name if mod.yaml_f_name is not None else "model_unknown"
            )
            path_model = self.path_da / yaml_name / model_name
        # path_model = self.path_da / self.model_base.yaml_f_name / model_name
        path_model.mkdir(parents=True, exist_ok=True)
        if subfolders is None:
            subfolders = []
        for subfolder in subfolders:
            (path_model / subfolder).mkdir(parents=True, exist_ok=True)

        return path_model

    #
    def define_baseline(
        self,
        time_start: float,
        time_stop: float,
        *,
        time_type: str = "abs",
        show_plot: bool = True,
    ) -> None:
        """
        Define ground state/pre-trigger/baseline reference spectrum.

        2D data will be cut and averaged between the specified time points
        to produce the baseline spectrum.

        Parameters
        ----------
        time_start : float or int
            Start point in time, inclusive (absolute value or index, see time_type)
        time_stop : float or int
            Stop point in time, inclusive (absolute value or index, see time_type)
        time_type : {'abs', 'ind'}, default='abs'
            Type of time specification:

            - 'abs': Absolute time stamps
            - 'ind': Time array indices

        show_plot : bool, default=True
            If True, plot the resulting baseline spectrum
        """

        if self.dim == 1:
            raise ValueError("Cannot define baseline for 1D data.")
        if self.data is None:
            raise ValueError("No data loaded; cannot define baseline.")
        if self.time is None:
            self.time = np.arange(self.data.shape[0])
            warnings.warn(
                "Time axis missing; using index axis for baseline definition.",
                stacklevel=2,
            )
        if self.energy is None:
            self.energy = np.arange(self.data.shape[1])
            warnings.warn("Energy axis missing; using index axis.", stacklevel=2)
        if time_type not in ("abs", "ind"):
            raise ValueError(
                f"Unknown time_type '{time_type}'. Expected 'abs' or 'ind'."
            )

        if time_type == "abs":
            t_ind_start = int(np.searchsorted(self.time, time_start, side="left"))
            t_ind_stop = int(np.searchsorted(self.time, time_stop, side="right"))
        elif time_type == "ind":
            t_ind_start = int(time_start)
            t_ind_stop = int(time_stop + 1)
        self.base_t_ind = [t_ind_start, t_ind_stop]
        self.base_t_abs = [self.time[t_ind_start], self.time[t_ind_stop - 1]]

        # cut and average
        self.data_base = np.mean(
            self.data[self.base_t_ind[0] : self.base_t_ind[1], :], axis=0
        )

        # plot
        if show_plot:
            if self.data_base is None:
                warnings.warn(
                    "Baseline data is unavailable; skipping baseline plot.",
                    stacklevel=2,
                )
                return
            uplt.plot_1d(
                data=[
                    self.data_base,
                ],
                x=self.energy,
                config=self.plot_config,
                title=f"Baseline data: t in {self.base_t_abs} (idx: {self.base_t_ind})",
            )

    #
    def set_fit_limits(
        self,
        energy_limits: Sequence[float] | None,
        *,
        time_limits: Sequence[float] | None = None,
        show_plot: bool = True,
    ) -> None:
        """
        Set energy (and time) limits for fits.

        Pass absolute values (NOT indices).

        Parameters
        ----------
        energy_limits : list of float or None
            Energy range for fitting ``[min, max]`` in absolute values.
            If None, uses full energy range ``[np.min(energy), np.max(energy)]``.
        time_limits : list of float, optional
            Time range for fitting ``[min, max]`` in absolute values.
            If None, no time limits are applied.
        show_plot : bool, default=True
            If True, plot data with fit limits indicated
        """

        if self.data is None and self.energy is None:
            raise ValueError("No data/energy axis loaded; cannot set fit limits.")
        if self.energy is None and self.data is not None:
            self.energy = np.arange(self.data.shape[-1])
            warnings.warn("Energy axis missing; using index axis.", stacklevel=2)
        if self.energy is None:
            raise ValueError("Energy axis unavailable; cannot set fit limits.")
        energy = self.energy
        if energy_limits is None:
            energy_limits = [float(np.min(energy)), float(np.max(energy))]
        self.e_lim_abs = [np.min(energy_limits), np.max(energy_limits)]

        # convert energy limits to [start, stop) slice indices
        # searchsorted with side='left' for lower bound, side='right' for upper bound
        if energy[0] > energy[-1]:  # descending energy
            rev = energy[::-1]
            start = len(energy) - int(
                np.searchsorted(rev, np.max(energy_limits), side="right")
            )
            stop = len(energy) - int(
                np.searchsorted(rev, np.min(energy_limits), side="left")
            )
            self.e_lim = [start, stop]
        else:  # ascending energy
            start = int(np.searchsorted(energy, np.min(energy_limits), side="left"))
            stop = int(np.searchsorted(energy, np.max(energy_limits), side="right"))
            self.e_lim = [start, stop]

        if time_limits is None and self.time is not None:
            time_limits = [float(np.min(self.time)), float(np.max(self.time))]
        if time_limits is not None:
            if self.time is None:
                if self.data is None or self.dim != 2:
                    raise ValueError("Time axis missing; cannot apply time limits.")
                self.time = np.arange(self.data.shape[0])
                warnings.warn(
                    "Time axis missing; using index axis for time limits.",
                    stacklevel=2,
                )
            self.t_lim_abs = list(time_limits)
            t_start = int(np.searchsorted(self.time, np.min(time_limits), side="left"))
            t_stop = int(np.searchsorted(self.time, np.max(time_limits), side="right"))
            self.t_lim = [t_start, t_stop]

        if show_plot:  # show data with limits
            if self.dim == 1:
                if self.data is None:
                    warnings.warn("Data missing; cannot plot fit limits.", stacklevel=2)
                    return
                x_cut = energy[self.e_lim[0] : self.e_lim[1]]
                y_cut = self.data[self.e_lim[0] : self.e_lim[1]]
                uplt.plot_1d(
                    data=[self.data, y_cut],
                    x=[energy, x_cut],
                    config=self.plot_config,
                    waterfall=(np.max(np.abs(y_cut)) - np.min(np.abs(y_cut))) / 8,
                    legend=["all", "cut"],
                    vlines=self.e_lim_abs,
                )
            elif self.dim == 2:
                if self.data is None:
                    warnings.warn(
                        "Data missing; cannot plot 2D fit limits.",
                        stacklevel=2,
                    )
                    return
                uplt.plot_2d(
                    data=self.data,
                    x=energy,
                    y=self.time,
                    config=self.plot_config,
                    vlines=self.e_lim_abs,
                    hlines=self.t_lim_abs,
                )

    #
    def fit_baseline(
        self, model_name: str, stages: int = 1, **lmfit_wrapper_kwargs
    ) -> None:
        """
        Fit the baseline/ground state/pre-trigger reference spectrum.

        Parameters
        ----------
        model_name : str
            Name of a previously loaded model (use File.load_model first)
        stages : {1, 2}, default=1
            Number of optimization stages:

            - 1: Single optimization with ``fit_alg_1``
            - 2: Two-stage fit (``fit_alg_1`` then ``fit_alg_2``)

        **lmfit_wrapper_kwargs
            Additional keyword arguments passed to fitlib.fit_wrapper
        """

        t_base = time.time()  # start timing for baseline fit

        self.model_base = self._resolve_model(model_name)
        if self.energy is None or self.data_base is None:
            raise ValueError(
                "Baseline data/energy axis missing; cannot fit baseline.\n"
                "Run define_baseline() first to extract the baseline region."
            )

        # get initial guess
        initial_guess = ulmfit.par_extract(
            self.model_base.lmfit_pars, return_type="list"
        )
        # define (and create) path where basline fit results will be saved to
        path_base_results = self.create_model_path(model_name)

        # const = (x, data, package, fnctn string, unpack, energy limits, time limits)
        self.model_base.const = (
            self.energy,
            self.data_base,
            self.p.spec_lib,
            self.p.spec_fun_str,
            0,
            self.e_lim,
            [],
        )
        # args [for fit function called in residual function]
        # model, dimension (dim =1 for baseline and SbS, =2 for 2D (global) fit)
        self.model_base.args = (self.model_base, 1)
        # fit (optionally) with confidence intervals
        self.model_base.result = fitlib.fit_wrapper(
            const=self.model_base.const,
            args=self.model_base.args,
            par_names=self.model_base.parameter_names,
            par=self.model_base.lmfit_pars,
            stages=stages,
            show_output=1 if self.p.show_output >= 1 else 0,
            save_output=1,
            save_path=path_base_results / model_name,
            **lmfit_wrapper_kwargs,
        )

        # update individual component spectra
        # self.model_base.create_value_1d(store_1d=1)

        # display/plot and save baseline fit summary
        title_base = (
            f"File: {self.path}, "
            f'Model: "{model_name}" (from "{self.model_base.yaml_f_name}.yaml")'
        )

        fitlib.plt_fit_res_1d(
            x=self.energy,
            y=self.data_base,
            fit_fun_str=self.p.spec_fun_str,
            package=self.p.spec_lib,
            par_init=initial_guess,
            par_fin=self.model_base.result[1],
            args=self.model_base.args,
            plot_sum=False,
            show_init=True,
            title=title_base,
            fit_lim=self.e_lim,
            config=self.plot_config,
            legend=[comp.name for comp in self.model_base.components],
            save_img=-1 if self.p.show_output < 1 else 1,
            save_path=path_base_results / "base_fit.png",
        )

        if stages >= 1:
            fitlib.time_display(
                t_start=t_base, print_str="Time elapsed for baseline fit: "
            )
            display(self.model_base.result[1].params)  # display final pars below figure

    #
    def load_fit(self) -> None:
        """
        TODO: Do this instead of refitting to try out different models?
        Probably needed to compare fits anyway!
        """

    #
    def fit_slice_by_slice(
        self, model_name: str, stages: int = 1, **fit_wrapper_kwargs
    ) -> None:
        """
        Fit time- and energy-resolved spectrum Slice-by-Slice (SbS).

        Treats every time step as independent from other times. Requires fitting
        the baseline first using fit_baseline().

        Parameters
        ----------
        model_name : str
            Name of a previously loaded model (use File.load_model first)
        stages : {1, 2}, default=1
            Number of optimization stages:

            - 1: Single optimization with ``fit_alg_1``
            - 2: Two-stage fit (``fit_alg_1`` then ``fit_alg_2``)

        **fit_wrapper_kwargs
            Additional keyword arguments passed to fitlib.fit_wrapper

        Notes
        -----
        Note:
        Currently the energy position guesses (x0) are shifted on a per slice basis
        according to the position in energy (x) of the maximum value of the spectrum
        (NOT always a good idea!)
        """

        t_sbs = time.time()  # start timing for SbS fit

        self.model_sbs = self._resolve_model(model_name)
        if self.model_base is None:
            raise ValueError(
                "Baseline model is not fitted yet; run fit_baseline() first."
            )
        if (
            self.data is None
            or self.time is None
            or self.energy is None
            or self.data_base is None
        ):
            raise ValueError(
                "Data/axes/baseline missing; cannot run Slice-by-Slice fit."
            )

        # define (and create) path where SbS fit results will be saved to
        path_sbs_results = self.create_model_path(
            model_name,
            subfolders=[
                "slices",
            ],
        )

        # set all fixed SbS fit parameters equal to baseline model results
        base_df = ulmfit.par_to_df(self.model_base.lmfit_pars, col_type="min")
        self.model_sbs.update_value(
            new_par_values=list(base_df["value"]), par_select="all"
        )

        # find all parameters with names ending in "x0"
        # so they can be updated for every slice
        e_pos_pars = [
            name for name in self.model_sbs.parameter_names if name.endswith("_x0")
        ]
        # find their corresponding values
        e_pos_vals = uarr.get_item(
            base_df, row=["name", e_pos_pars], col="value", astype="series"
        )

        # cycle through all spectra and fit them
        self.results_sbs = []  # (re-)initialize placeholder for results
        for s_i, s in enumerate(self.data):
            print(f"Analyzing slice number {s_i + 1}/{len(self.time)}", end="\r")
            if s_i < self.p.skip_first_n_spec:
                continue  # skip past baseline spectra for debugging
            # define path for files saved for this slice
            path_slice = path_sbs_results / "slices" / str(self.p.da_slices_fmt % s_i)

            # update the "x0" peak energy guess(es) using
            # "max(baseline) -(max current slice)" [ in eV]
            delta_max = (
                self.energy[np.argmax(s)] - self.energy[np.argmax(self.data_base)]
            )
            # update all guesses for parameters with names ending in "x0"
            new_e_vals = list(e_pos_vals.add(delta_max))
            self.model_sbs.update_value(
                new_par_values=new_e_vals, par_select=e_pos_pars
            )
            # get initial guess
            initial_guess = ulmfit.par_extract(
                self.model_sbs.lmfit_pars, return_type="list"
            )

            # const = (x, data, package, fnctn str, unpack, energy limits, time limits)
            self.model_sbs.const = (
                self.energy,
                s,
                self.p.spec_lib,
                self.p.spec_fun_str,
                0,
                self.e_lim,
                [],
            )
            # args [for fit function called in residual function]
            self.model_sbs.args = (self.model_sbs, 1)

            # fit with confidence intervals
            result_sbs = fitlib.fit_wrapper(
                const=self.model_sbs.const,
                args=self.model_sbs.args,
                par_names=self.model_sbs.parameter_names,
                par=self.model_sbs.lmfit_pars,
                stages=stages,
                show_output=0,
                save_output=1,
                save_path=path_slice,
                **fit_wrapper_kwargs,
            )

            # add final fit parameters to list of fit parameters of all spectra
            self.results_sbs.append(result_sbs)

            # (optionally) plot and (always) save fit summary for this slice
            fitlib.plt_fit_res_1d(
                x=self.model_sbs.const[0],
                y=self.model_sbs.const[1],
                fit_fun_str=self.p.spec_fun_str,
                package=self.p.spec_lib,
                par_init=initial_guess,
                par_fin=result_sbs[1],
                args=self.model_sbs.args,
                plot_sum=False,
                show_init=True,
                fit_lim=self.e_lim,
                config=self.plot_config,
                save_img=-1,
                save_path=path_slice.with_suffix(".png"),
            )
            #
            if s_i == self.p.first_n_spec_only:
                break  # for debugging: only fit first N spectra

        if stages >= 1:
            self.save_sbs_fit(save_path=path_sbs_results)
            fitlib.time_display(
                t_start=t_sbs, print_str="Time elapsed for Slice-by-Slice fit: "
            )

    #
    def save_sbs_fit(self, save_path: PathLike) -> None:
        """
        Export Slice-by-Slice fit results.

        Saves parameter evolution as CSV, plots individual parameters vs. time,
        reconstructs 2D fit map, and creates data/fit/residual comparison plots.

        Parameters
        ----------
        save_path : str or Path
            Base directory for saving results
        """

        if self.model_sbs is None or self.time is None:
            raise ValueError(
                "Slice-by-Slice model/results are incomplete; nothing to save."
            )
        if self.data is None:
            raise ValueError("Data missing; cannot save Slice-by-Slice fit.")
        if self.model_sbs.const is None or self.model_sbs.args is None:
            raise ValueError(
                "Slice-by-Slice model const/args missing; cannot reconstruct 2D fit."
            )
        # convert results, specifically par_fin to dataframe and save
        # this also plots all parameters as a function of time
        df_sbs = fitlib.results_to_df(
            results=self.results_sbs,
            x=self.time,
            index=np.arange(0, len(self.time)),
            config=self.plot_config,
            skip_first_n_spec=self.p.skip_first_n_spec,
            first_n_spec_only=self.p.first_n_spec_only,
            save_df=-1 if self.p.show_output == 0 else 1,
            save_path=save_path,
        )

        # get slice-by-slice fit spectra as a 2D map
        df_sbs_pars = df_sbs.loc[:, self.model_sbs.parameter_names]
        fit_2d_sbs = fitlib.results_to_fit_2d(
            results=df_sbs_pars,
            const=self.model_sbs.const,
            args=self.model_sbs.args,
            save_2d=-1 if self.p.show_output == 0 else 1,
            save_path=save_path,
        )

        # plot data, fit, and residual 2D maps
        # (works if full 2D map is fitted/ no slices skipped)
        if self.p.first_n_spec_only == -1 and self.p.skip_first_n_spec == -1:
            fitlib.plt_fit_res_2d(
                data=self.data,
                fit=fit_2d_sbs,
                x=self.energy,
                y=self.time,
                config=self.plot_config,
                x_lim=self.e_lim,
                y_lim=self.t_lim,
                save_img=-1 if self.p.show_output == 0 else 1,
                save_path=save_path,
            )

    #
    def _resolve_model(self, model_name: str | None) -> mcp.Model:
        """
        Resolve a model by name, falling back to model_active.

        Raises ValueError if the model cannot be found.
        """

        if model_name is None:
            if self.model_active is None:
                available = [m.name for m in self.models]
                raise ValueError(
                    "No active model set. Call set_active_model() first "
                    "or pass target_model explicitly.\n"
                    f"Available models: {available or 'none loaded'}"
                )
            return self.model_active
        mod = self.select_model(model_name)
        if mod is None:
            available = [m.name for m in self.models]
            raise ValueError(
                f"Model '{model_name}' not found.\n"
                f"Available models: {available or 'none loaded'}"
            )
        return mod

    #
    def add_time_dependence(
        self,
        target_model: str,
        target_parameter: str,
        dynamics_yaml: PathLike,
        dynamics_model: str | list[str],
        *,
        frequency: float = -1,
    ) -> None:
        """
        Add time dependence for one parameter of a model.

        Loads a "Dynamics"-type model to describe time-dependent behavior.
        The parameter can live either directly in the energy model or inside
        a Profile model attached to an energy model parameter.

        To avoid strongly correlated fits, adding dynamics directly to an
        energy-model parameter that already has a profile (``p_vary=True``)
        is currently disallowed. In that case, add dynamics to a profile
        parameter instead.

        Parameters
        ----------
        target_model : str
            Name of the energy model to add dynamics to.
        target_parameter : str
            Name of parameter to make time-dependent. Can be an energy model
            parameter or a profile model parameter.
        dynamics_yaml : str or Path
            YAML file name defining the Dynamics model.
        dynamics_model : str or list of str
            Model name(s) for time-dependent behavior.
        frequency : float, default=-1
            Repetition frequency for time-dependent behavior.
            -1 (default) means no repetition (single cycle).
        """

        model = self._resolve_model(target_model)
        t_mod = self.load_model(
            dynamics_yaml,
            dynamics_model,
            target_parameter,
            model_type="dynamics",
        )

        # Try energy model first
        ci, pi = model.find_par_by_name(target_parameter)
        if ci is not None and pi is not None:
            target_par = model.components[ci].pars[pi]
            if target_par.p_vary:
                raise ValueError(
                    f"Cannot add time dependence to parameter "
                    f"'{target_parameter}' because it already has a profile "
                    "(p_vary=True). This is currently disabled to avoid strongly "
                    "correlated fits. Add dynamics to a profile parameter "
                    "instead, or remove/fix the profile first."
                )
            model.add_dynamics(cast("mcp.Dynamics", t_mod), frequency)
            model.dim = 2
            return

        # Search profile models attached to energy model parameters
        for comp in model.components:
            for par in comp.pars:
                if par.p_vary and par.p_model is not None:
                    pci, _ppi = par.p_model.find_par_by_name(target_parameter)
                    if pci is not None:
                        # Add dynamics to the profile model
                        par.p_model.add_dynamics(cast("mcp.Dynamics", t_mod), frequency)
                        # Sync dynamics params into the energy model's parameter list
                        par.lmfit_par_list.extend(t_mod.lmfit_par_list)
                        model.update()
                        model.dim = 2
                        return

        # Not found in energy model or any attached profile
        available = model.parameter_names
        profile_pars: list[str] = []
        for comp in model.components:
            for par in comp.pars:
                if par.p_vary and par.p_model is not None:
                    profile_pars.extend(par.p_model.parameter_names)
        if profile_pars:
            available = available + profile_pars
        raise ValueError(
            f"Parameter '{target_parameter}' not found in model "
            f"'{model.name}' or any attached profile models.\n"
            f"Available parameters: {available}"
        )

    #
    def add_par_profile(
        self,
        target_model: str,
        target_parameter: str,
        profile_yaml: PathLike,
        profile_model: str | list[str],
    ) -> None:
        """
        Add a parameter profile over the auxiliary axis to a model.

        Loads a ``"profile"``-type model from a YAML file and attaches it to the
        named parameter, so that the parameter varies over ``aux_axis``
        (uniform averaging over the auxiliary dimension).

        If any parameters inside the profile model are time-dependent
        (``t_vary=True``), the model's dim is automatically set to 2.

        To avoid strongly correlated fits, adding a profile to an
        energy-model parameter that already has time dependence
        (``t_vary=True``) is currently disallowed.

        Parameters
        ----------
        target_model : str
            Name of the energy model to add the profile to.
        target_parameter : str
            Name of the parameter to attach the profile to
            (e.g. ``'GLP_01_A'``).
        profile_yaml : str or Path
            YAML file name defining the Profile model.
        profile_model : str or list of str
            Model name(s) for the profile (single element).
        """

        model = self._resolve_model(target_model)
        p_mod = self.load_model(
            profile_yaml,
            profile_model,
            target_parameter,
            model_type="profile",
        )
        ci, pi = model.find_par_by_name(target_parameter)
        if ci is None or pi is None:
            raise ValueError(
                f"Parameter '{target_parameter}' not found in model '{model.name}'.\n"
                f"Available parameters: {model.parameter_names}"
            )
        target_par = model.components[ci].pars[pi]
        if target_par.t_vary:
            raise ValueError(
                f"Cannot add profile to parameter '{target_parameter}' because "
                "it already has time dependence (t_vary=True). This is currently "
                "disabled to avoid strongly correlated fits. Add profile first "
                "and dynamics to a profile parameter instead, or remove/fix "
                "time dependence first."
            )
        model.add_profile(cast("mcp.Profile", p_mod))
        # auto-promote to 2D if any parameter inside the profile is time-dependent
        if any(p.t_vary for comp in p_mod.components for p in comp.pars):
            model.dim = 2

    #
    def fit_2d(self, model_name: str, stages: int = 1, **fit_wrapper_kwargs) -> None:
        """
        Perform energy- and time-dependent 2D model fit.

        Parameters
        ----------
        model_name : str
            Name of the model to fit (loaded via File.load_model)
        stages : {1, 2}, default=1
            Number of optimization stages:

            - 1: Single optimization with ``fit_alg_1``
            - 2: Two-stage fit (``fit_alg_1`` then ``fit_alg_2``)

        **fit_wrapper_kwargs
            Additional keyword arguments passed to fitlib.fit_wrapper
            (see fitlib.fit_wrapper for details)
        """

        t_2d = time.time()  # start timing for 2D fit

        self.model_2d = self._resolve_model(model_name)
        if self.model_base is None:
            raise ValueError(
                "Baseline model is not fitted yet; run fit_baseline() first."
            )
        if self.energy is None or self.time is None or self.data is None:
            raise ValueError("Data/axes missing; cannot run 2D fit.")

        # define (and create) path where 2D fit results will be saved to
        path_2d_results = self.create_model_path(model_name)

        # set all fixed 2D fit parameters equal to baseline model results
        base_df = ulmfit.par_to_df(self.model_base.lmfit_pars, col_type="min")
        self.model_2d.update_value(
            new_par_values=list(base_df["value"]), par_select=list(base_df["name"])
        )
        # const [x, data, package, function string, unpack, energy limits, time limits]
        self.model_2d.const = (
            self.energy,
            self.data,
            self.p.spec_lib,
            self.p.spec_fun_str,
            0,
            self.e_lim,
            self.t_lim,
        )
        # args [for fit function called in residual function]
        self.model_2d.args = (self.model_2d, 2)  # model, dimension

        # fit (with confidence intervals)
        self.model_2d.result = fitlib.fit_wrapper(
            const=self.model_2d.const,
            args=self.model_2d.args,
            par_names=self.model_2d.parameter_names,
            par=self.model_2d.lmfit_pars,
            stages=stages,
            show_output=1 if self.p.show_output >= 1 else 0,
            save_output=1,
            save_path=path_2d_results / model_name,
            **fit_wrapper_kwargs,
        )
        if stages >= 1:
            self.save_2d_fit(save_path=path_2d_results)
            fitlib.time_display(
                t_start=t_2d, print_str="Time elapsed for 2D model fit: "
            )
            display(self.model_2d.result[1].params)  # display final pars below figure

    #
    def save_2d_fit(self, save_path: PathLike) -> None:
        """
        Export 2D model fit results.

        Evaluates model at final parameters, creates 2D data/fit/residual
        comparison plots, and saves to specified directory.

        Parameters
        ----------
        save_path : str or Path
            Base directory for saving results
        """

        if (
            self.model_2d is None
            or self.energy is None
            or self.time is None
            or self.data is None
        ):
            raise ValueError("2D model/data/axes missing; nothing to save.")
        self.model_2d.create_value_2d()  # update 2D spectrum to final fit result
        if self.model_2d.value_2d is None:
            raise ValueError(
                "2D model evaluation did not produce value_2d; nothing to save."
            )
        # plot data, fit, and residual 2D maps
        fitlib.plt_fit_res_2d(
            data=self.data,
            fit=self.model_2d.value_2d,
            x=self.energy,
            y=self.time,
            config=self.plot_config,
            x_lim=self.e_lim,
            y_lim=self.t_lim,
            save_img=-1 if self.p.show_output == 0 else 1,
            save_path=save_path,
        )
        # dpi_plot = round(1.5 *self.p.dpi_plt), NOT AVAILABLE YET (fig_size)

    #
    def get_fit_results(
        self,
        *,
        fit_type: Literal["baseline", "sbs", "2d"] = "baseline",
    ) -> pd.DataFrame:
        """
        Return fit results as a DataFrame for programmatic access.

        Parameters
        ----------
        fit_type : {'baseline', 'sbs', '2d'}, default='baseline'
            Which fit results to return:

            - 'baseline': Baseline/ground-state fit (from ``fit_baseline``)
            - 'sbs': Slice-by-Slice fit (from ``fit_slice_by_slice``)
            - '2d': 2D global fit (from ``fit_2d``)

        Returns
        -------
        pd.DataFrame
            For 'baseline' and '2d': one row per parameter with columns
            ``['name', 'value', 'stderr', 'init_value', 'min', 'max',
            'vary', 'expr']``.
            For 'sbs': one row per time slice with columns = parameter names.

        Raises
        ------
        ValueError
            If the requested fit has not been performed yet.
        """

        if fit_type == "baseline":
            if self.model_base is None or not self.model_base.result:
                raise ValueError("No baseline fit results. Run fit_baseline() first.")
            return ulmfit.par_to_df(
                self.model_base.result[1].params,
                col_type="min",
                par_names=self.model_base.parameter_names,
            )
        if fit_type == "sbs":
            if not self.results_sbs:
                raise ValueError(
                    "No Slice-by-Slice fit results. Run fit_slice_by_slice() first."
                )
            return ulmfit.list_of_par_to_df(self.results_sbs)
        if fit_type == "2d":
            if self.model_2d is None or not self.model_2d.result:
                raise ValueError("No 2D fit results. Run fit_2d() first.")
            return ulmfit.par_to_df(
                self.model_2d.result[1].params,
                col_type="min",
                par_names=self.model_2d.parameter_names,
            )
        raise ValueError(
            f"Unknown fit_type={fit_type!r}; use 'baseline', 'sbs', or '2d'."
        )

    #
    def compare_models(self) -> None:
        """
        TODO: Compare fit quality across multiple models (not yet implemented).

        Future implementation will compare:
        - Residual maps and statistics (min/max/std)
        - Reduced chi-squared values
        - Model complexity vs. fit quality metrics

        Notes
        -----
        This method is a placeholder for future development.
        """
