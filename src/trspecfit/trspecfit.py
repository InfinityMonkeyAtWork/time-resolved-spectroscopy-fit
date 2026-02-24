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
    energy/time axes. Manages multiple models for comparison, handles baseline
    extraction, and provides methods for baseline fitting, Slice-by-Slice
    fitting, and full 2D time- and energy-resolved fitting.

Workflow
--------
1. **Setup**: Create Project with configuration settings
2. **Load Data**: Create File object with data, energy, and time axes
3. **Define Models**: Load models from YAML files using File.load_model()
4. **Set Limits**: Define fitting regions with File.set_fit_limits()
5. **Fit**: Execute appropriate fitting method:
   - File.fit_baseline() for ground state spectrum
   - File.fit_SliceBySlice() for time-independent analysis
   - File.fit_2Dmodel() for global time- and energy-resolved fitting
6. **Export**: Results automatically saved to project directory structure

Features
--------
- Hierarchical project/file organization with automatic directory creation
- Multiple model management and comparison within single File
- YAML-based project and model configuration
- Flexible plot customization via PlotConfig
- Support for 1D (energy-only) and 2D (time + energy) datasets
- Baseline/pre-trigger spectrum extraction and fitting
- Slice-by-Slice time-series fitting with parameter evolution tracking
- Global 2D fitting with time-dependent parameters
- Automatic result export (parameters, plots, confidence intervals)

Examples
--------
See examples/ directory for complete workflows.
"""

import copy
import os  # replace os.join with "pathlib path /"subfolder" /"file name"
import pathlib
import time
import types
import warnings
from collections.abc import Callable, Sequence
from typing import Literal, cast, overload

import numpy as np
from IPython.display import display
from ruamel.yaml import YAML

from trspecfit import fitlib, mcp, spectra

# from trspecfit.functions import distribution as fcts_dist
# standardized plotting configuration
from trspecfit.config.plot import PlotConfig

# function library for energy, time, and distribution components
from trspecfit.functions import energy as fcts_energy
from trspecfit.functions import time as fcts_time
from trspecfit.utils import arrays as uarr
from trspecfit.utils import lmfit as ulmfit
from trspecfit.utils import parsing as uparsing
from trspecfit.utils import plot as uplt

PathLike = str | pathlib.Path
ModelRef = str | int | list[str]

# what does show_info mean? convert to binary debug by True if show_info >=3 else False

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
    run : str
        Current run name
    show_info : int
        Verbosity level (0=silent, 1=basic, 2=detailed, 3=debug)
    spec_lib : module
        Module containing spectrum fitting functions (default: spectra)
    spec_fun_str : str
        Name of fitting function in spec_lib
    skip_first_N_spec, first_N_spec_only : int
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
    Attributes ext, fmt, delim, DA_fmt, and DA_slices_fmt control file
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
        self.run = name
        self.path_run = self.path_results / name

        # Set defaults first
        self._set_defaults()

        # Override with YAML config if provided
        if config_file is not None:
            self._load_config(config_file)

    #
    def _set_defaults(self) -> None:
        """Set default project configuration."""

        self.show_info = 1
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
        self.DA_fmt = "%04d"
        self.DA_slices_fmt = "%06d"
        # Advanced settings
        self.spec_lib = spectra
        self.spec_fun_str = "fit_model_mcp"
        self.skip_first_N_spec = -1
        self.first_N_spec_only = -1

    @property
    def spec_fun(self) -> Callable:
        """
        Dynamically get the spectrum fitting function.
        """

        return cast("Callable", getattr(self.spec_lib, self.spec_fun_str))

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

        yaml = YAML()  # Standard YAML loading
        config_path = self.path / config_file

        try:
            with open(config_path) as f:
                config = yaml.load(f)

            if config is None:
                if self.show_info >= 1:
                    print(f"Warning: {config_file} is empty, using defaults")
                return

            # Update attributes from config
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    if self.show_info >= 2:
                        print(f"Set {key} = {value}")
                else:
                    if self.show_info >= 1:
                        print(f"Warning: Unknown config key '{key}' ignored")

        except FileNotFoundError:
            if self.show_info >= 1:
                print(f"Config file {config_path} not found, using defaults")
        except Exception as e:
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

    Attributes
    ----------
    p : Project
        Parent project providing configuration
    path : str or Path
        File identifier
    path_DA : Path
        Directory path for saving this file's fit results
    data : ndarray
        Spectroscopy data (1D or 2D)
    dim : int
        Data dimensionality (1 or 2)
    energy : ndarray
        Energy axis
    time : ndarray or None
        Time axis (None for 1D data)
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
    model_SbS : Model or None
        Model used for Slice-by-Slice fitting
    results_SbS : list
        Slice-by-Slice fit results for all time slices
    plot_config : PlotConfig
        Plot configuration (created from parent Project on first access)

    Notes
    -----
    **Typical Workflow:**
    1. Create File with data, energy, time
    2. Load model(s) with load_model()
    3. Set active model with set_active_model()
    4. Define baseline with define_baseline() (for 2D data)
    5. Set fit limits with set_fit_limits()
    6. Fit baseline with fit_baseline()
    7. Perform Slice-by-Slice or 2D fit with fit_SliceBySlice() or fit_2Dmodel()

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
    ) -> None:
        # pass parent project or (default) create a functioning test project environment
        self.p = parent_project if parent_project is not None else Project(path=None)
        self.path = path  # path to load/save [?] data from
        self.path_DA = self.p.path_run / path  # path to save fit results to
        self._plot_config: PlotConfig | None = None  # create plot config from project
        self.data = data  # (time-[optional] and) energy-dependent data to fit
        self.dim = 0 if data is None else len(np.shape(data))  # 1/2 D for energy/+time
        # take energy and time input or create a generic axis if None is passed
        if energy is not None or data is None:
            self.energy = energy
        elif self.dim == 1:
            self.energy = np.arange(0, np.shape(data)[0])
        else:
            self.energy = np.arange(0, np.shape(data)[1])
        if time is not None or self.dim <= 1 or data is None:
            self.time = time
        else:
            self.time = np.arange(0, np.shape(data)[0])
        # keep track of models that are used to fit this file/data
        self.models: list[mcp.Model] = []  # $% could do @property, setter, getter here?
        self.model_active: mcp.Model | None = None  # default model to work with
        # Energy and time limits for fitting methods
        self.e_lim_abs: list[float] = []  # energy limits (low, high) user-defined
        self.e_lim: list[int] = []  # index (from left, from right: energy[left:-right])
        self.t_lim_abs: list[float] = []  # time limits (low, high) user-defined
        self.t_lim: list[int] = []  # index (left to right: time[left:right])
        #
        self.base_t_abs: list[
            float
        ] = []  # start and stop time of the baseline spectrum
        self.base_t_ind: list[int] = []  # index of the above start and stop time
        self.data_base = None  # average spectrum between above indices
        self.model_base: mcp.Model | None = None
        #
        self.model_SbS: mcp.Model | None = None
        self.model_2D: mcp.Model | None = None
        # all Slice-by-Slice fit results (different from model_SbS.result)
        self.results_SbS: list = []

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
            self.energy = np.arange(np.shape(self.data)[-1])
            warnings.warn("Energy axis missing; using index axis.", stacklevel=2)
        if self.dim == 2 and self.time is None:
            self.time = np.arange(np.shape(self.data)[0])
            warnings.warn("Time axis missing; using index axis.", stacklevel=2)

        config = self.plot_config

        if self.dim == 1:
            uplt.plot_1D(
                data=[
                    self.data,
                ],
                x=self.energy,
                config=config,
                vlines=self.e_lim_abs,
            )

        elif self.dim == 2:
            uplt.plot_2D(
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
    @overload
    def select_model(
        self, model_info: ModelRef, return_type: Literal["model"] = "model"
    ) -> mcp.Model | None: ...

    @overload
    def select_model(
        self, model_info: ModelRef, return_type: Literal["index"] = "index"
    ) -> int | None: ...

    def select_model(
        self, model_info: ModelRef, return_type: Literal["model", "index"] = "model"
    ) -> mcp.Model | int | None:
        """
        Select model by name [type(model_info)=str] or position [type(model_info)=int].

        Returns model (``return_type='model'``, default) or
        index of model in File.models (``return_type='index'``).
        Returns None if model name not found or index out of range.

        For time-dependence/ dynamics models with more than one model i.e. submodels:
        pass the list containing all model names (same input as in "load_model")
        """

        if isinstance(model_info, str):
            for m_i, m in enumerate(self.models):
                if m.name == model_info:
                    if return_type == "model":
                        return self.models[m_i]
                    if return_type == "index":
                        return m_i
            return None  # no match found

        if isinstance(model_info, int):
            if model_info not in range(len(self.models)):
                return None  # no match found
            if return_type == "model":
                return self.models[model_info]
            if return_type == "index":
                return model_info

        elif isinstance(model_info, list):
            m_name = self.model_list_to_name(model_info)
            for m_i, m in enumerate(self.models):
                if m.name == m_name:
                    if return_type == "model":
                        return self.models[m_i]
                    if return_type == "index":
                        return m_i
            return None  # no match found

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
        model_info: list[str],
        par_name: str = "",
        debug: bool = False,
    ) -> mcp.Model | None:
        """
        Load a model from YAML file.

        Loads a model defined in ``model_yaml`` file located in Parent.path.

        Parameters
        ----------
        model_yaml : str or Path
            YAML file name (located in project path) defining the model
        model_info : list of str
            Model name(s) to load:

            - For energy-dependent models (1D or 2D): ``['model_name']``
              (single element). Model will be set as active model.
            - For standard time-dependent models: ``['model_name']``.
              Single element applies to entire time axis.
            - For multi-cycle time-dependent models: ``['model1', 'model2', ...]``.
              Element 0 applies to entire time axis, elements 1+ apply to
              respective subcycles only.

        par_name : str, default=''
            Parameter name for time-dependent models. Empty string (default)
            indicates energy-dependent model. For Dynamics models, must match
            the name of the 2D model parameter whose time-dependence is
            described by the model being loaded.
        debug : bool, default=False
            If True, print detailed parameter information during loading

        Returns
        -------
        Model or None
            Returns the loaded model for time-dependent models (when par_name is set),
            None for energy-dependent models (which are set as active model).
        """

        # sanity checks
        if not isinstance(model_info, list):
            raise TypeError(
                "model_info must be a list.\n"
                "Usage:\n"
                "  [name_model1,] for energy-dependent models\n"
                "  [name_model1, name_model2 (optional), ...] for time-dependent models"
            )
        if par_name == "" and len(model_info) != 1:
            raise ValueError(
                'Energy-resolved data (par_name="") require a single model name in'
                " model_info.\nPass model name as the only element in the model_info"
                " list.\nOR pass a non-empty par_name to define mcp.Dynamics model"
                " with one or more model names."
            )
        if self.select_model(model_info) is not None:
            raise ValueError(
                f'Model named "{self.model_list_to_name(model_info)}" already exists.'
                " Delete the existing model or change the name of the new model."
            )

        # Load and process YAML file with appropriate numbering strategy
        model_yaml_path = self.p.path / pathlib.Path(model_yaml)
        model_info_dict = uparsing.load_and_number_yaml_components(
            model_yaml_path=model_yaml_path,
            model_info=model_info,
            is_dynamics=par_name != "",
            debug=debug,
        )

        # Initialize model
        fcts_package: types.ModuleType
        if par_name == "":
            if self.p.show_info >= 1:
                print(
                    f"Loading model to describe energy- (and time-)dependent data: "
                    f"{self.model_list_to_name(model_info)}"
                )
            fcts_package = fcts_energy
            loaded_model = mcp.Model(self.model_list_to_name(model_info))
        else:
            if self.p.show_info >= 1:
                print(
                    f"Loading model to describe time-dependence of a model parameter: "
                    f"{par_name} of {self.model_list_to_name(model_info)} model"
                )
            fcts_package = fcts_time
            loaded_model = mcp.Dynamics(par_name)

        # Inherit necessary model attributes from function input, file, and project
        loaded_model.yaml_f_name = pathlib.Path(model_yaml).stem  # yaml file name
        loaded_model.dim = 1  # start with 1, +1 when adding dynamics
        if isinstance(loaded_model, mcp.Dynamics):
            loaded_model.subcycles = len(model_info) - 1
        loaded_model.energy = self.energy  # $% remove redundancy?
        loaded_model.time = self.time  # $% remove redundancy?

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
        if not isinstance(loaded_model, mcp.Dynamics):
            self.models.append(loaded_model)
            self.set_active_model(model_info)  # set as current active model
            return None
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

        if model_info is None:
            mod = self.model_active
            # model_info = FIND NAME
        else:
            mod = self.select_model(model_info)
        if mod is None:
            warnings.warn("Model not found; nothing to describe.", stacklevel=2)
            return

        # parameter list
        mod.describe(detail=0)

        if detail == 1 and isinstance(mod, mcp.Dynamics):
            mod.create_value1D(store1D=1)  # update individual component spectra
            mod.plot_1D(plot_ind=True)  # plot guess only (individual components)

        if detail == 1 and mod.dim == 1:
            if self.energy is None or self.data_base is None:
                warnings.warn(
                    "Energy axis or baseline data missing;"
                    " cannot plot 1D model summary.",
                    stacklevel=2,
                )
                return
            mod.create_value1D(store1D=1)  # update individual component spectra
            # plot initial guess (individual components), data, and residual
            title_mod = (
                f"File: {self.path}, "
                 f'Model: "{model_info}" (from "{mod.yaml_f_name}.yaml")'
                 ": initial guess"
            )
            fitlib.plt_fit_res_1D(
                x=self.energy,
                y=self.data_base,
                fit_fun_str=self.p.spec_fun_str,
                package=self.p.spec_lib,
                par_init=[],
                par_fin=mod.lmfit_pars,
                args=(mod, 1),
                plot_ind=True,
                show_init=False,
                title=title_mod,
                fit_lim=self.e_lim,
                config=self.plot_config,
                legend=[comp.name for comp in mod.components],
            )

        if detail == 1 and mod.dim == 2:
            mod.create_value2D()  # update spectrum
            if self.data is None or mod.value2D is None:
                warnings.warn(
                    "2D data/model values missing; cannot plot 2D model summary.",
                    stacklevel=2,
                )
                return
            # plot data, fit, and residual 2D maps
            fitlib.plt_fit_res_2D(
                data=self.data,
                fit=mod.value2D,
                x=self.energy,
                y=self.time,
                config=self.plot_config,
                x_lim=self.e_lim,
                y_lim=self.t_lim,
            )

    #
    def delete_model(self, model_to_delete: str | int | None = None) -> None:
        """
        Remove a model from this file's model list.

        Parameters
        ----------
        model_to_delete : str or int, optional
            Model to delete:

            - str: Model name
            - int: Model index in self.models
            - None: Delete currently active model

        Notes
        -----
        After deletion, model_active may be invalid. Set a new active model
        if needed using set_active_model().
        """

        mod_index_del: int | None = None
        if model_to_delete is None:
            if self.model_active is None:
                warnings.warn("No active model to delete.", stacklevel=2)
                return
            mod_index_del = self.models.index(self.model_active)  # list.index(value)

        elif isinstance(model_to_delete, str):
            mod_index_del = self.select_model(model_to_delete, return_type="index")
            if mod_index_del is None:
                print(f"delete_model: Model with name {model_to_delete} not found")
                return

        elif isinstance(model_to_delete, int):
            mod_index_del = copy.deepcopy(model_to_delete)
            if mod_index_del not in range(len(self.models)):
                print("delete_model: Model index out of range")
                return

        else:
            print(f"delete_model: input type {type(model_to_delete)} not supported")
            return

        # delete model from list using index: File.models[index]
        if mod_index_del is None:
            return
        self.models.pop(mod_index_del)

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
            path_model = self.path_DA / "model_unknown" / model_name
        else:
            yaml_name = (
                mod.yaml_f_name if mod.yaml_f_name is not None else "model_unknown"
            )
            path_model = self.path_DA / yaml_name / model_name
        # path_model = self.path_DA / self.model_base.yaml_f_name / model_name
        if self.p.show_info >= 3:
            print(path_model)
        if not os.path.exists(path_model):
            os.makedirs(path_model)
        if subfolders is None:
            subfolders = []
        if len(subfolders) != 0:
            for subfolder in subfolders:
                if not os.path.exists(path_model / subfolder):
                    os.makedirs(path_model / subfolder)
                    if self.p.show_info >= 3:
                        print(path_model / subfolder)

        return path_model

    #
    def define_baseline(
        self,
        time_start: float,
        time_stop: float,
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
            Start point in time (absolute value or index depending on time_type)
        time_stop : float or int
            Stop point in time (absolute value or index depending on time_type)
        time_type : {'abs', 'ind'}, default='abs'
            Type of time specification:

            - 'abs': Absolute time stamps
            - 'ind': Time array indices

        show_plot : bool, default=True
            If True, plot the resulting baseline spectrum
        """

        if self.dim == 1:
            warnings.warn("Cannot define baseline for 1D data.", stacklevel=2)
            return
        if self.data is None:
            warnings.warn("No data loaded; cannot define baseline.", stacklevel=2)
            return
        if self.time is None:
            self.time = np.arange(np.shape(self.data)[0])
            warnings.warn(
                "Time axis missing; using index axis for baseline definition.",
                stacklevel=2,
            )
        if self.energy is None:
            self.energy = np.arange(np.shape(self.data)[1])
            warnings.warn("Energy axis missing; using index axis.", stacklevel=2)
        if time_type not in ("abs", "ind"):
            warnings.warn(
                f"Unknown time_type '{time_type}'. Expected 'abs' or 'ind'.",
                stacklevel=2,
            )
            return

        if time_type == "abs":
            t_ind_start = int(np.searchsorted(self.time, time_start))
            t_ind_stop = int(np.searchsorted(self.time, time_stop))
        elif time_type == "ind":
            t_ind_start = int(time_start)
            t_ind_stop = int(time_stop)
        self.base_t_ind = [t_ind_start, t_ind_stop]
        self.base_t_abs = [self.time[t_ind_start], self.time[t_ind_stop]]

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
            uplt.plot_1D(
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
            warnings.warn(
                "No data/energy axis loaded; cannot set fit limits.",
                stacklevel=2,
            )
            return
        if self.energy is None and self.data is not None:
            self.energy = np.arange(np.shape(self.data)[-1])
            warnings.warn("Energy axis missing; using index axis.", stacklevel=2)
        if self.energy is None:
            warnings.warn(
                "Energy axis unavailable; cannot set fit limits.",
                stacklevel=2,
            )
            return
        energy = self.energy
        if energy_limits is None:
            energy_limits = [float(np.min(energy)), float(np.max(energy))]
        self.e_lim_abs = [np.min(energy_limits), np.max(energy_limits)]

        # convert energy and time limits to index values
        # use data ordering (not plot direction) to pass ascending input to searchsorted
        n_e = np.shape(energy)[0]
        if energy[0] > energy[-1]:  # descending energy
            E_ind_min = int(np.searchsorted(energy[::-1], np.min(energy_limits)))
            E_ind_max = int(np.searchsorted(energy[::-1], np.max(energy_limits)))
            self.e_lim = [n_e - E_ind_max, E_ind_min]  # skip high-E start, low-E end
        else:  # ascending energy
            E_ind_min = int(np.searchsorted(energy, np.min(energy_limits)))
            E_ind_max = int(np.searchsorted(energy, np.max(energy_limits)))
            self.e_lim = [E_ind_min, n_e - E_ind_max]  # skip low-E start, high-E end

        if time_limits is not None:
            if self.time is None:
                if self.data is None or self.dim != 2:
                    warnings.warn(
                        "Time axis missing; cannot apply time limits.",
                        stacklevel=2,
                    )
                    return
                self.time = np.arange(np.shape(self.data)[0])
                warnings.warn(
                    "Time axis missing; using index axis for time limits.",
                    stacklevel=2,
                )
            self.t_lim_abs = list(time_limits)
            t_ind_min = int(np.searchsorted(self.time, np.min(time_limits)))
            t_ind_max = int(np.searchsorted(self.time, np.max(time_limits)))
            self.t_lim = [t_ind_min, t_ind_max]  # "min:max"

        if show_plot:  # show data with limits
            if self.dim == 1:
                if self.data is None:
                    warnings.warn("Data missing; cannot plot fit limits.", stacklevel=2)
                    return
                x_cut = energy[self.e_lim[0] : -self.e_lim[1]]
                y_cut = self.data[self.e_lim[0] : -self.e_lim[1]]
                uplt.plot_1D(
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
                uplt.plot_2D(
                    data=self.data,
                    x=energy,
                    y=self.time,
                    config=self.plot_config,
                    vlines=self.e_lim_abs,
                    hlines=self.t_lim_abs,
                )

    #
    def fit_baseline(self, model_name: str, fit: int, **lmfit_wrapper_kwargs) -> None:
        """
        Fit the baseline/ground state/pre-trigger reference spectrum.

        Parameters
        ----------
        model_name : str
            Name of a previously loaded model (use File.load_model first)
        fit : {0, 1, 2}
            Fitting mode:

            - 0: Show initial guess only (no fit)
            - 1: Perform fit with single method
            - 2: Two-stage fit (global then local optimization)

        **lmfit_wrapper_kwargs
            Additional keyword arguments passed to fitlib.fit_wrapper
        """

        t_base = time.time()  # start timing for baseline fit

        # find model with matching name from list
        self.model_base = self.select_model(model_info=model_name)
        if self.model_base is None:
            warnings.warn(
                f"Model '{model_name}' not found; baseline fit skipped.",
                stacklevel=2,
            )
            return
        if self.energy is None or self.data_base is None:
            warnings.warn(
                "Baseline data/energy axis missing; baseline fit skipped.",
                stacklevel=2,
            )
            return

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
        # model, dimension (dim =1 for baseline and SbS, =2 for 2D (global) fit), debug
        self.model_base.args = (self.model_base, 1, False)
        # fit (optionally) with confidence intervals
        self.model_base.result = fitlib.fit_wrapper(
            const=self.model_base.const,
            args=self.model_base.args,
            par_names=self.model_base.par_names,
            par=self.model_base.lmfit_pars,
            fit_type=fit,
            show_info=1 if self.p.show_info >= 2 else 0,
            save_output=1,
            save_path=path_base_results / model_name,
            **lmfit_wrapper_kwargs,
        )

        # update individual component spectra
        # self.model_base.create_value1D(store1D=1)

        # display/plot and save baseline fit summary
        title_base = (
            f"File: {self.path}, "
            f'Model: "{model_name}" (from "{self.model_base.yaml_f_name}.yaml")'
        )

        fitlib.plt_fit_res_1D(
            x=self.energy,
            y=self.data_base,
            fit_fun_str=self.p.spec_fun_str,
            package=self.p.spec_lib,
            par_init=initial_guess,
            par_fin=self.model_base.result[1],
            args=self.model_base.args,
            plot_ind=True,
            show_init=True,
            title=title_base,
            fit_lim=self.e_lim,
            config=self.plot_config,
            legend=[comp.name for comp in self.model_base.components],
            save_img=-1 if self.p.show_info < 1 else 1,
            save_path=path_base_results / "base_fit.png",
        )

        if fit >= 1:
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

        #$% implement loadging of fit results after refactoring saving to hdf5 output

    #
    def fit_SliceBySlice(self, model_name: str, fit: int, **fit_wrapper_kwargs) -> None:
        """
        Fit time- and energy-resolved spectrum Slice-by-Slice (SbS).

        Treats every time step as independent from other times. Requires fitting
        the baseline first using fit_baseline().

        Parameters
        ----------
        model_name : str
            Name of a previously loaded model (use File.load_model first)
        fit : {0, 1, 2}
            Fitting mode:

            - 0: Show initial guess only (no fit)
            - 1: Perform fit with single method
            - 2: Two-stage fit (global then local optimization)

        **fit_wrapper_kwargs
            Additional keyword arguments passed to fitlib.fit_wrapper

        Notes
        -----
        Note:
        Currently the energy position guesses (x0) are shifted on a per slice basis
        according to the position in energy (x) of the maximum value of the spectrum
        (NOT always a good idea!)
        """

        t_SbS = time.time()  # start timing for SbS fit

        # find model with matching name from list
        self.model_SbS = self.select_model(model_info=model_name)
        if self.model_SbS is None:
            warnings.warn(
                f"Model '{model_name}' not found; Slice-by-Slice fit skipped.",
                stacklevel=2,
            )
            return
        if self.model_base is None:
            warnings.warn(
                "Baseline model is not fitted yet; run fit_baseline() first.",
                stacklevel=2,
            )
            return
        if (
            self.data is None
            or self.time is None
            or self.energy is None
            or self.data_base is None
        ):
            warnings.warn(
                "Data/axes/baseline missing; Slice-by-Slice fit skipped.",
                stacklevel=2,
            )
            return

        # define (and create) path where SbS fit results will be saved to
        path_SbS_results = self.create_model_path(
            model_name,
            subfolders=[
                "slices",
            ],
        )

        # set all fixed SbS fit parameters equal to baseline model results
        base_df = ulmfit.par2df(self.model_base.lmfit_pars, col_type="min")
        self.model_SbS.update_value(
            new_par_values=list(base_df["value"]), par_select="all"
        )

        # find all parameters with names ending in "x0"
        # so they can be updated for every slice
        e_pos_pars = [name for name in self.model_SbS.par_names if name.endswith("_x0")]
        # find their corresponding values
        e_pos_vals = uarr.get_item(
            base_df, row=["name", e_pos_pars], col="value", astype="series"
        )

        # cycle through all spectra and fit them
        self.results_SbS = []  # (re-)initialize placeholder for results
        for s_i, s in enumerate(self.data):
            if self.p.show_info < 3:
                print(f"Analyzing slice number {s_i + 1}/{len(self.time)}", end="\r")
            if s_i < self.p.skip_first_N_spec:
                continue  # skip past baseline spectra for debugging
            if self.p.show_info >= 3:
                print()
                print("Spectrum #" + str(s_i))  # print iteration info
            # define path for files saved for this slice
            path_slice = os.path.join(
                path_SbS_results, "slices", str(self.p.DA_slices_fmt % s_i)
            )

            # update the "x0" peak energy guess(es) using
            # "max(baseline) -(max current slice)" [ in eV]
            deltaMAX = (
                self.energy[np.argmax(s)] - self.energy[np.argmax(self.data_base)]
            )
            if self.p.show_info >= 3:
                print(f"deltaMAX (spectrum with respect to baseline: {deltaMAX}")
            # update all guesses for parameters with names ending in "x0"
            new_e_vals = list(e_pos_vals.add(deltaMAX))
            self.model_SbS.update_value(
                new_par_values=new_e_vals, par_select=e_pos_pars
            )
            # get initial guess
            initial_guess = ulmfit.par_extract(
                self.model_SbS.lmfit_pars, return_type="list"
            )

            # const = (x, data, package, fnctn str, unpack, energy limits, time limits)
            self.model_SbS.const = (
                self.energy,
                s,
                self.p.spec_lib,
                self.p.spec_fun_str,
                0,
                self.e_lim,
                [],
            )
            # args (lmfit2D.Model, dim, debug) [for fit fnctn called in residual fnctn]
            self.model_SbS.args = (self.model_SbS, 1, False)

            # fit with confidence intervals
            result_SbS = fitlib.fit_wrapper(
                const=self.model_SbS.const,
                args=self.model_SbS.args,
                par_names=self.model_SbS.par_names,
                par=self.model_SbS.lmfit_pars,
                fit_type=fit,
                show_info=1 if self.p.show_info >= 3 else 0,
                save_output=1,
                save_path=path_slice,
                **fit_wrapper_kwargs,
            )

            # add final fit parameters to list of fit parameters of all spectra
            self.results_SbS.append(result_SbS)

            # (optionally) plot and (always) save fit summary for this slice
            fitlib.plt_fit_res_1D(
                x=self.model_SbS.const[0],
                y=self.model_SbS.const[1],
                fit_fun_str=self.p.spec_fun_str,
                package=self.p.spec_lib,
                par_init=initial_guess,
                par_fin=result_SbS[1],
                args=self.model_SbS.args,
                plot_ind=True,
                show_init=True,
                fit_lim=self.e_lim,
                config=self.plot_config,
                save_img=-1 if self.p.show_info < 3 else 1,
                save_path=path_slice + ".png",
            )
            #
            if s_i == self.p.first_N_spec_only:
                break  # for debugging: only fit first N spectra

        if fit >= 1:
            self.save_SliceBySlice_fit(save_path=path_SbS_results)
            fitlib.time_display(
                t_start=t_SbS, print_str="Time elapsed for Slice-by-Slice fit: "
            )

    #
    def save_SliceBySlice_fit(self, save_path: PathLike) -> None:
        """
        Export Slice-by-Slice fit results.

        Saves parameter evolution as CSV, plots individual parameters vs. time,
        reconstructs 2D fit map, and creates data/fit/residual comparison plots.

        Parameters
        ----------
        save_path : str or Path
            Base directory for saving results
        """

        if self.model_SbS is None or self.time is None:
            warnings.warn(
                "Slice-by-Slice model/results are incomplete; nothing to save.",
                stacklevel=2,
            )
            return
        if self.data is None:
            warnings.warn("Data missing; cannot save Slice-by-Slice fit.", stacklevel=2)
            return
        if self.model_SbS.const is None or self.model_SbS.args is None:
            warnings.warn(
                "Slice-by-Slice model const/args missing; cannot reconstruct 2D fit.",
                stacklevel=2,
            )
            return
        # convert results, specifically par_fin to dataframe and save
        # this also plots all parameters as a function of time
        df_SbS = fitlib.results2df(
            results=self.results_SbS,
            x=self.time,
            index=np.arange(0, len(self.time)),
            config=self.plot_config,
            skip_first_N_spec=self.p.skip_first_N_spec,
            first_N_spec_only=self.p.first_N_spec_only,
            save_df=-1 if self.p.show_info == 0 else 1,
            save_path=save_path,
        )

        if self.p.show_info >= 3:
            display(df_SbS)

        # get slice-by-slice fit spectra as a 2D map
        df_SbS_pars = df_SbS.loc[:, self.model_SbS.par_names]
        fit2D_SbS = fitlib.results2fit2D(
            results=df_SbS_pars,
            const=self.model_SbS.const,
            args=self.model_SbS.args,
            save_2D=-1 if self.p.show_info == 0 else 1,
            save_path=save_path,
        )

        if self.p.show_info >= 3:
            print(f"size SbS 2D map: {np.shape(fit2D_SbS)}")

        # plot data, fit, and residual 2D maps
        # (works if full 2D map is fitted/ no slices skipped)
        if self.p.first_N_spec_only == -1 and self.p.skip_first_N_spec == -1:
            fitlib.plt_fit_res_2D(
                data=self.data,
                fit=fit2D_SbS,
                x=self.energy,
                y=self.time,
                config=self.plot_config,
                x_lim=self.e_lim,
                y_lim=self.t_lim,
                save_img=-1 if self.p.show_info == 0 else 1,
                save_path=save_path,
            )

    #
    def add_time_dependence(
        self,
        model_yaml: PathLike,
        model_info: list[str],
        par_name: str,
        frequency: float = -1,
    ) -> None:
        """
        Add time dependence for one parameter of currently active model.

        Loads a "Dynamics"-type model to describe time-dependent behavior.

        Parameters
        ----------
        model_yaml : str or Path
            YAML file name defining the Dynamics model
        model_info : list of str
            Model name(s) for time-dependent behavior
        par_name : str
            Name of parameter in active model to make time-dependent
        frequency : float, default=-1
            Repetition frequency for time-dependent behavior.
            -1 (default) means no repetition (single cycle).
        """

        t_mod = self.load_model(
            model_yaml,
            model_info,
            par_name,
            debug=not self.p.show_info < 2,
        )  # load
        if t_mod is None:
            warnings.warn(
                "Dynamics model could not be loaded; time dependence not added.",
                stacklevel=2,
            )
            return
        if self.model_active is None:
            warnings.warn(
                "No active model available; time dependence not added.",
                stacklevel=2,
            )
            return
        self.model_active.add_dynamics(cast("mcp.Dynamics", t_mod), frequency)  # add
        self.model_active.dim = 2  # increase dimension of model to 2

    #
    def fit_2Dmodel(self, model_name: str, fit: int, **fit_wrapper_kwargs) -> None:
        """
        Perform energy- and time-dependent 2D model fit.

        Parameters
        ----------
        model_name : str
            Name of the model to fit (loaded via File.load_model)
        fit : {0, 1, 2}
            Fitting mode:

            - 0: Show initial guess only (no fit)
            - 1: Perform fit with single method
            - 2: Two-stage fit - global minimum search (fit_alg_1)
              followed by local optimization (fit_alg_2)

        **fit_wrapper_kwargs
            Additional keyword arguments passed to fitlib.fit_wrapper
            (see fitlib.fit_wrapper for details)
        """

        t_2D = time.time()  # start timing for 2D fit

        # find model with matching name from list
        self.model_2D = self.select_model(model_info=model_name)
        if self.model_2D is None:
            warnings.warn(
                f"Model '{model_name}' not found; 2D fit skipped.",
                stacklevel=2,
            )
            return
        if self.model_base is None:
            warnings.warn(
                "Baseline model is not fitted yet; run fit_baseline() first.",
                stacklevel=2,
            )
            return
        if self.energy is None or self.time is None or self.data is None:
            warnings.warn("Data/axes missing; 2D fit skipped.", stacklevel=2)
            return

        # define (and create) path where 2D fit results will be saved to
        path_2D_results = self.create_model_path(model_name)

        # set all fixed 2D fit parameters equal to baseline model results
        base_df = ulmfit.par2df(self.model_base.lmfit_pars, col_type="min")
        self.model_2D.update_value(
            new_par_values=list(base_df["value"]), par_select=list(base_df["name"])
        )
        # const [x, data, package, function string, unpack, energy limits, time limits]
        self.model_2D.const = (
            self.energy,
            self.data,
            self.p.spec_lib,
            self.p.spec_fun_str,
            0,
            self.e_lim,
            self.t_lim,
        )
        # args [for fit function called in residual function]
        self.model_2D.args = (self.model_2D, 2, False)  # model, dimension, debug

        # fit (with confidence intervals)
        self.model_2D.result = fitlib.fit_wrapper(
            const=self.model_2D.const,
            args=self.model_2D.args,
            par_names=self.model_2D.par_names,
            par=self.model_2D.lmfit_pars,
            fit_type=fit,
            show_info=1 if self.p.show_info >= 2 else 0,
            save_output=1,
            save_path=path_2D_results / model_name,
            **fit_wrapper_kwargs,
        )
        if fit >= 1:
            self.save_2Dmodel_fit(save_path=path_2D_results)
            fitlib.time_display(
                t_start=t_2D, print_str="Time elapsed for 2D model fit: "
            )
            display(self.model_2D.result[1].params)  # display final pars below figure

    #
    def save_2Dmodel_fit(self, save_path: PathLike) -> None:
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
            self.model_2D is None
            or self.energy is None
            or self.time is None
            or self.data is None
        ):
            warnings.warn(
                "2D model/data/axes missing; nothing to save.",
                stacklevel=2,
            )
            return
        self.model_2D.create_value2D()  # update 2D spectrum to final fit result
        if self.model_2D.value2D is None:
            warnings.warn(
                "2D model evaluation did not produce value2D; nothing to save.",
                stacklevel=2,
            )
            return
        # plot data, fit, and residual 2D maps
        fitlib.plt_fit_res_2D(
            data=self.data,
            fit=self.model_2D.value2D,
            x=self.energy,
            y=self.time,
            config=self.plot_config,
            x_lim=self.e_lim,
            y_lim=self.t_lim,
            save_img=-1 if self.p.show_info == 0 else 1,
            save_path=save_path,
        )
        # dpi_plot = round(1.5 *self.p.dpi_plt), NOT AVAILABLE YET (fig_size)

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

        #$% re-use (at least parts of) the loading fit results functionality
