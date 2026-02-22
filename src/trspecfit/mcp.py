"""
Model/Component/Parameter (MCP) system for spectroscopy fitting.

This module implements the hierarchical model construction system that is the
heart of trspecfit. It provides classes for building complex spectral models
from reusable components with flexible parameter management.

Core Classes
------------
Model : Container for spectral components and parameters
    Represents a complete 1D or 2D spectral model built from components.
    Handles parameter management, model evaluation, and plotting.

Component : Individual spectral or temporal function
    Wraps functions from trspecfit.functions (energy/time) into model-ready
    objects with parameter management and axis handling.

Par : Parameter with optional time-dependence
    Extends lmfit.Parameter to support time-varying parameters through
    the Dynamics system, and handles parameter expressions/constraints.

Dynamics : Model subclass for time-dependent behavior
    Special Model type that describes how parameters evolve over time,
    with support for multi-cycle dynamics and convolution kernels.

Architecture
------------
The MCP system uses composition to build models from bottom-up::

    Par (values + time-dependence)
      ↓
    Component (function + parameters)
      ↓
    Model (components + combination rules)
      ↓
    Spectrum (1D or 2D evaluated model)

Key Features
------------
- Hierarchical model construction from reusable components
- Automatic parameter naming and numbering for multi-component models
- Time-dependent parameters via Dynamics models
- Expression-based parameter constraints and relationships
- Support for convolution with instrumental response functions
- Multi-cycle dynamics with subcycle support
- Automatic component combination (addition, convolution, backgrounds)
"""

import lmfit
from trspecfit.utils import lmfit as ulmfit
from trspecfit.utils import arrays as uarr
from trspecfit.utils import plot as uplt
from trspecfit.utils import parsing as uparsing
import math
import numpy as np
import re
import inspect
import copy
from IPython.display import display
#import concurrent.futures
from typing import List, Optional, Tuple, Union, Dict, Any, Callable, cast
import types
# asteval is used for expressions referencing time-dependent parameters
from asteval import Interpreter
# function library for energy, time, and distribution components
from trspecfit.functions import energy as fcts_energy
from trspecfit.functions import time as fcts_time
#from trspecfit.functions import distribution as fcts_dist
# function configurations
from trspecfit.config.functions import (
    background_functions,
    energy_functions,
    time_functions,
    convolution_functions,
)
# plot configuration
from trspecfit.config.plot import PlotConfig

#
#
class Model:
    """
    Define a 2D time- and energy-resolved fit model using lmfit.
    
    Model is the top-level container for spectral fitting. It manages a
    collection of Component objects and their parameters, handles model
    evaluation, and provides methods for fitting and visualization.
    
    Parameters
    ----------
    model_name : str, default='test'
        Name identifier for this model (used in file I/O and plotting)
    
    Attributes
    ----------
    name : str
        Model identifier
    yaml_f_name : str or None
        Name of YAML file this model was loaded from (without extension)
    peak_fcts : list
        Function objects for all components (extracted from Component.fct)
    components : list of Component
        Component objects that define this model's behavior
    lmfit_par_list : list of lmfit.Parameter
        Flattened list of all individual parameters (spectral + temporal)
    lmfit_pars : lmfit.Parameters
        Complete parameter object for fitting (from lmfit_par_list)
    par_names : list of str
        Names of all parameters in the model
    component_spectra : list of ndarray
        Individual component spectra from last evaluation (when store1D=1)
    value1D : ndarray or None
        1D spectrum (sum of all components) from last evaluation
    value2D : ndarray or None
        2D spectrum (time × energy) from last evaluation
    const : tuple or None
        Constants for residual function (x, data, package, function_str, ...)
    args : tuple or None
        Arguments for fit function (model, dim, debug)
    result : list
        Fit results from fit_wrapper [par_ini, par_fin, conf_CIs, emcee_fin, emcee_CIs]
    parent_file : File or None
        Parent File object (set when model is loaded)
    dim : int or None
        Dimensionality (1 for energy/time only, 2 for time+energy)
    energy : ndarray or None
        Energy axis for spectral components
    time : ndarray or None
        Time axis for temporal dynamics
    
    Notes
    -----
    **Component Combination:**
    Components are combined in reverse order (last to first) via the
    Model.combine() static method, which handles:
    - Addition (regular peaks)
    - Convolution (with kernels)
    - Background addition (requires existing spectrum)
    
    **Parameter Management:**
    All parameters are stored in a flat lmfit.Parameters object for fitting.
    Time-dependent parameters add additional parameters from their Dynamics
    models to this flat structure.
    
    **Inheritance:**
    Model attributes (energy, time, parent_file) are inherited by:
    - Components (need axes for evaluation)
    - Parameters (need time axis for dynamics)
    - Dynamics models (need time axis and parent reference)
    
    See Also
    --------
    Component : Individual spectral/temporal components
    Par : Parameters with optional time-dependence
    Dynamics : Model subclass for time-dependent parameters
    File.load_model : Load models from YAML definitions
    """
    #
    def __init__(self, model_name: str = 'test') -> None:
        self.name: str = model_name
        # file name of yaml file containing model details
        self.yaml_f_name: Optional[str] = None
        # functions of spectral components of fit
        self.peak_fcts: List[Callable] = []
        # list of objects of type defined in Component class
        self.components: List['Component'] = []
        # flattened lmfit parameters list (1D with time- and energy-components)
        self.lmfit_par_list: List[lmfit.Parameter] = [] # (individual lmfit.Parameter objects)
        # lmfit.Parameters object corresponding to lmfit_par_list attribute
        self.lmfit_pars: lmfit.Parameters = lmfit.Parameters()
        # list of all parameter names
        self.par_names: List[str] = []
        # list of component spectra (from last evaluation/ current parameters)
        self.component_spectra: List[np.ndarray] = []
        # 1D spectrum (i.e. sum/ combination of all components)
        self.value1D: Optional[np.ndarray] = None
        # 2D spectrum (i.e. 1D spectra one per time step)
        self.value2D: Optional[np.ndarray] = None # self.value2D = np.empty((len(self.time), len(self.energy)))
        # fit parameters and results
        self.const: Optional[Tuple] = None
        self.args: Optional[Tuple] = None
        self.result: List = []
        # ATTRIBUTES THAT SHOULD BE INHERITED FROM A PARENT ENTITY WHEN LOADING MODEL
        self.parent_file: Optional[Any] = None  # parent reference (set by File when loading model)
        #self.data = None # (currently) not necessary
        self.dim: Optional[int] = None
        self.energy: Optional[np.ndarray] = None # necessarry or should just point to file?
        self.time: Optional[np.ndarray] = None # necessarry or should just point to file?
        #
        return None
    
    @property
    def plot_config(self) -> PlotConfig:
        """
        Get plot configuration from parent File.
        
        Models inherit plot settings from their parent File, ensuring
        consistent plotting across all models for the same dataset.
        
        Returns
        -------
        PlotConfig
            Configuration object with plot settings (axes, colors, DPI, etc.)
        """
        if hasattr(self, 'parent_file') and self.parent_file is not None:
            return cast(PlotConfig, self.parent_file.plot_config)
        
        # Fallback to defaults if no parent
        return PlotConfig()

    #  
    def describe(self, detail: int = 0) -> None:
        """
        Display information about model structure and parameters.
        
        Parameters
        ----------
        detail : int, default=0
            Level of detail to display:
            - 0: Component list and parameters only
            - 1+: Also plot initial guess and (for 2D) data comparison
        """
        # minimum model description
        print('model name: ' +self.name)
        try:
            for comp in self.components:
                comp.describe(detail=detail-1)
        except: print('no elements in this model')
        print('all (1D flattened if applicable) lmfit.Parameters(): [sorted alphabetically]')
        try: self.lmfit_pars.pretty_print()
        except: print('lmfit.Parameters() object is empty')
        print()
        # plot initial guess of model
        if detail >= 1:
            if isinstance(self, Dynamics):
                self.create_value1D(store1D=1)
                self.plot_1D()
            else: # energy-resolved model
                if self.dim == 1:
                    self.create_value1D(store1D=1)
                    self.plot_1D()
                elif self.dim == 2:
                    self.create_value2D()
                    self.plot_2D()            
        #
        return None
    
    #
    def add_components(self, comps_list: List['Component'], debug: bool = False) -> None:
        """
        Add components to model and initialize their parameters.
        
        This is the primary method for building up a model. It takes a list
        of Component objects, assigns them appropriate names/prefixes, creates
        their parameters, and updates the model's parameter structure.
        
        Parameters
        ----------
        comps_list : list of Component
            Components to add to this model. Components should already have
            their parameter dictionaries populated via Component.add_pars().
        debug : bool, default=False
            If True, print detailed parameter information during creation
        
        Notes
        -----
        **Component Naming:**
        Components are expected to have their names already set from YAML
        parsing (e.g., GLP_01, GLP_02, Offset, Shirley). The numbering is
        handled during YAML parsing, not here.
        
        **Parameter Naming:**

        Parameter names are constructed as: prefix + component_name + '_' + param_name

        - For Dynamics models: prefix = model.name (e.g., ``'GLP_01_x0_'``)
        - For regular models: prefix = ``''`` (e.g., ``'GLP_01_A'``)
        
        **Component Preparation:**
        Each component receives:
        - energy/time axes from model
        - parent_model reference (for finding other parameters)
        - subcycle time axis (for multi-cycle Dynamics)
        - kernel time axis (for convolution components)
        
        **Model Updates:**
        After adding components, the model's lmfit_pars and par_names
        are automatically updated via self.update().
        """
        # add list to components attribute 
        self.components = comps_list
        
        # the model calling this method is describing temporal dynamics of a par
        if isinstance(self, Dynamics): prefix = self.name
        # the model calling this method is a general model
        else: prefix = ''
        
        # assemble parameter list for all components
        # [components can be energy or time functions]
        for comp in self.components:
            # add current component function to list of all functions
            self.peak_fcts.append(comp.fct)           
            # pass time and energy axis to the individual components
            comp.energy = self.energy
            comp.time = self.time
            # set parent model reference
            comp.parent_model = self
            # subcycle time axis [updated when using Dynamics.set_frequency] 
            if isinstance(self, Dynamics):
                if self.time is None:
                    raise ValueError("Model time axis must be defined for Dynamics components")
                comp.time_N_sub = np.ones(len(self.time)) # initialize all active
            # if comp should be convoluted it will be defined on a t_kernel axis 
            if comp.comp_type == 'conv':
                comp.time = Component.create_t_kernel(comp)
            # populate pars attribute in the component
            comp.create_pars(prefix=prefix, debug=debug)
        
        # update model lmfit_par_list (+par_names) and components
        self.update(debug=debug)
        if debug: self.lmfit_pars.pretty_print()          
        #
        return None
    
    #
    def find_par_by_name(self, par_name: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Find the component and parameter indices for a given parameter name.
        
        Searches through all components and their parameters to locate the
        indices needed to access a specific parameter by name (exact match).
        
        Parameters
        ----------
        par_name : str
            Full parameter name to search for (e.g., 'GLP_01_x0', 'Offset_y0')
        
        Returns
        -------
        ci : int or None
            Component index in self.components, or None if not found
        pi : int or None
            Parameter index in component.pars, or None if not found
        """
        done = False
        ci: Optional[int] = None
        pi: Optional[int] = None
        for ci, comp in enumerate(self.components):
            for pi, par in enumerate(comp.pars):
                if par.name == par_name:
                    done = True; break
            if done == True: break
        if done == False: #$% should this Raise?
            print(f'parameter "{par_name}" not found in model {self.name}')
            ci = None; pi = None
        #
        return ci, pi
    
    #
    def print_all_pars(self, detail: int = 0) -> None:
        """
        Print information on all parameters individually.
        Debugging utility to inspect parameter structure and values.
        For routine parameter inspection, use model.describe() or
        model.lmfit_pars.pretty_print().
        """
        for c in self.components:
            for p in c.pars:
                p.describe(detail)
        #
        return None
    
    #
    def update(self, debug: bool = False) -> None:
        """
        Update model from bottom up: parameters → components → model.
        
        Recompiles all parameters from all components and recreates the
        flattened lmfit parameter structures. Call this after modifying
        parameter structure.
        (automatically called after add_components or add_dynamics)
        
        Parameters
        ----------
        debug : bool, default=False
            If True, print parameter list after update
        """
        # re-initialize
        self.lmfit_par_list = []
        self.lmfit_pars = lmfit.Parameters()
        self.par_names = []
        
        for ci, comp in enumerate(self.components):
            # create a flattened lmfit.Parameter list for the component
            comp.update_lmfit_par_list()
            # add lmfit.Parameter list of this component to corresponding model list
            self.lmfit_par_list.extend(comp.lmfit_par_list)
            
        # create lmfit.Parameters object from the lmfit_par_list
        self.lmfit_pars.add_many(*self.lmfit_par_list)
        if debug: self.lmfit_pars.pretty_print()
        
        # update list of all parameter names
        self.par_names = [par.name for par in self.lmfit_par_list]
        #
        return None
    
    #
    def update_value(self, new_par_values: Union[List[float], np.ndarray], 
                     par_select: Union[str, List[str]] = 'all') -> None:
        """
        Update model from top down: model → components → parameters.
        
        Updates parameter values in the model's lmfit_pars based on new
        values (e.g., from optimizer). Used during fitting to apply
        proposed parameter values before model evaluation.
        
        Parameters
        ----------
        new_par_values : array-like
            New parameter values to apply. Length must match number of
            parameters being updated.
        par_select : str or list of str, default='all'
            Which parameters to update:

            - 'all': Update all parameters in order
            - list: Update only parameters with names in this list

        Notes
        -----
        Called by spectra.fit_model_mcp() on every iteration during fitting
        to update model parameters before evaluation.
        Does not trigger model re-evaluation; call create_value1D() or
        create_value2D() after updating values.
        """
        p_count = 0 # initialize counter for parameters in par_select
        for i, p in enumerate(self.lmfit_pars):
            # update ALL values in model.lmfit_pars attribute
            if par_select == 'all':
                self.lmfit_pars[p].value = new_par_values[i]
            # update only selected values in model.lmfit_pars
            else:
                if self.lmfit_pars[p].name in par_select:
                    self.lmfit_pars[p].value = new_par_values[p_count]
                    p_count += 1
        #
        return None
    
    #
    def add_dynamics(self, dynamics_model: 'Dynamics', frequency: float = -1, 
                     debug: bool = False) -> None:
        """
        Add temporal dynamics model to a parameter.
        
        Makes a parameter time-dependent by attaching a Dynamics model that
        describes how the parameter evolves over time. The Dynamics model name
        must match the parameter name exactly.
        
        Parameters
        ----------
        dynamics_model : Dynamics
            Dynamics instance describing time evolution. The model name
            must match a parameter in this model (e.g., 'GLP_01_x0').
        frequency : float, default=-1
            Repetition frequency for cyclic dynamics (Hz):
            - -1: Single cycle over entire time axis
            - >0: Dynamics repeat at this frequency
        debug : bool, default=False
            If True, print parameter structure after adding dynamics
        """
        # set the model instance calling this method as parent model for Dynamics 
        dynamics_model.parent_model = self
        
        if frequency != -1: # set a repetition frequency
            dynamics_model.set_frequency(frequency)
        
        # find component and parameter index from Dynamics model name
        ci, pi = self.find_par_by_name(dynamics_model.name)
        if ci is None or pi is None:
            raise ValueError(f'Parameter "{dynamics_model.name}" not found in model {self.name}')
        target_par = self.components[ci].pars[pi]

        # Disallow adding dynamics to expression-linked parameters.
        # Their value is already constrained by the expression.
        if len(target_par.info) == 1 and isinstance(target_par.info[0], str):
            raise ValueError(
                f'Cannot add time dependence to expression parameter "{dynamics_model.name}" '
                f'(expression: {target_par.info[0]}). '
                'Add dynamics to the referenced base parameter instead.'
            )
        
        # add Dynamics model and update corresponding parameter
        target_par.update(dynamics_model)
        # update model lmfit_par_list, par_names and components
        self.update(debug=debug)
        
        # Re-analyze all expressions since time-dependence status may have changed
        self._analyze_expression_dependencies()
        #
        return None
    
    #
    def _analyze_expression_dependencies(self) -> None:
        """
        Analyze all parameter expressions for time-dependent references.
        
        Checks if any parameter expressions reference time-dependent parameters.
        This is needed to handle cases like:
        - Par1 = 10 (time-dependent via Dynamics)
        - Par2 = Par1 + 5 (expression, should track Par1's time-dependence)
        
        Called automatically after add_dynamics() to update expression
        dependency tracking.
        """
        # Get all parameters from all components
        all_parameters = self._get_all_parameters()
        # Analyze all expressions for time-dependent references
        for par in all_parameters:
            par.analyze_expression_dependencies(all_parameters)
        #
        return None
    
    #
    def _get_all_parameters(self) -> List['Par']:
        """
        Get all parameters from all components in this model.
        (Used internally for expression analysis and parameter searches)
        
        Returns
        -------
        list of Par
            All Par objects from all components
        """
        all_parameters = []
        for comp in self.components:
            all_parameters.extend(comp.pars)
        #
        return all_parameters
    
    #
    @staticmethod
    def combine(value: np.ndarray, comp: 'Component', t_ind: int = 0) -> np.ndarray:
        """
        Combine component value with input spectrum via addition or convolution.
        
        This is the core method that defines how components are combined to
        build up a complete spectrum. Components are processed in reverse order
        (last to first) during model evaluation.
        
        Parameters
        ----------
        value : ndarray
            Current spectrum being built up
        comp : Component
            Component to add/convolve with current spectrum
        t_ind : int, default=0
            Time index for evaluation (for time-dependent parameters)
        
        Returns
        -------
        ndarray
            Updated spectrum after combining with component
        """
        # skip 'none' type components entirely (no-op)
        if comp.comp_type == 'none':
            return value
        # add a component to existing spectrum
        if comp.comp_type == 'add':
            return cast(np.ndarray, value + np.asarray(comp.value(t_ind)))
        # add a background component to exisiting spectrum
        elif comp.comp_type == 'back':
            return cast(np.ndarray, value + np.asarray(comp.value(t_ind, spectrum=value)))
        # convolute component with existing spectrum 
        elif comp.comp_type == 'conv':
            if comp.package == fcts_energy:
                x_axis = comp.energy
                print('convolution of spectral components not defined')
            elif comp.package == fcts_time:
                x_axis = comp.time
            else:
                x_axis = None
            #
            if x_axis is None:
                raise ValueError(f"Convolution axis not defined for component '{comp.name}'")
            return uarr.my_conv(x=x_axis, y=value, kernel=np.asarray(comp.value(t_ind)))
        else:
            raise ValueError(f"Unknown component type: {comp.comp_type}")
   
    #
    def create_value1D(self, t_ind: int = 0, store1D: int = 0, 
                       return1D: int = 0, debug: bool = False) -> Optional[np.ndarray]:
        """
        Evaluate model to create 1D spectrum (energy or time).
        
        Combines all components according to their types (addition, convolution,
        background) to generate the complete model spectrum at a specific time
        point (or for a Dynamics model, the complete time evolution).
        
        Parameters
        ----------
        t_ind : int, default=0
            Time index for evaluation. For energy-resolved models, this selects
            which time point to evaluate. For Dynamics models, affects which
            parameters have time-dependence applied.
        store1D : int, default=0
            If 1, store individual component spectra in self.component_spectra
            for later plotting or analysis.
        return1D : int, default=0
            If 1, return the computed spectrum. Otherwise return None and store
            in self.value1D only.
        debug : bool, default=False
            If True, print component numbers and plot each component as it's added
        
        Returns
        -------
        ndarray or None
            If return1D=1, returns the 1D spectrum. Otherwise returns None.
            Spectrum is always stored in self.value1D regardless of return setting.
        
        Notes
        -----
        **Component Combination Order:**
        Components are combined in REVERSE order (last to first):
        1. Initialize with last component
        2. Combine with second-to-last, third-to-last, etc.
        3. This allows backgrounds to access the sum of all peaks
        
        **Stored Components:**
        When store1D=1, self.component_spectra contains individual contributions
        in the ORIGINAL order (not reversed). This matches the order components
        were defined and makes plotting intuitive.
        
        **Performance:**
        For 2D models, this function is called repeatedly (once per time point).
        If components have no time-dependence, their evaluation could be cached,
        but current implementation re-evaluates for simplicity and correctness
        with convolution/background interactions.
        """
        # re-initialize list containing individual component spectra
        if store1D == 1: self.component_spectra = []
        # initialize value1D by evaluating last component
        self.value1D = self.components[-1].value(t_ind)
        if store1D == 1: self.component_spectra.append(self.value1D)
        
        # combine the components into a spectrum/ time dynamics curve
        for N in range(len(self.components) -1):
            if debug: print(N+2); print(self.components[-(N+2)].fct_str)
            if store1D == 1: current_spec = copy.deepcopy(self.value1D)
            #
            self.value1D = Model.combine(self.value1D, 
                                         self.components[-(N+2)],
                                         t_ind)
            # check on last component value added to model
            if store1D == 1: self.component_spectra.append(self.value1D -current_spec)
            if debug: uplt.plot_1D([self.component_spectra[-1],])
            
        # flip component spectra list as components are combined LIFO in this function
        if store1D == 1: self.component_spectra = self.component_spectra[::-1]
        #
        if return1D == 1: return self.value1D
        else: return None

    #
    def create_value2D(self, t_ind: List[int] = [], debug: bool = False) -> None:
        """
        Evaluate model to create 2D spectrum (time × energy).
        
        Generates the complete time- and energy-resolved spectrum by calling
        create_value1D() for each time point. This is where time-dependent
        parameters dynamically modify the model at each time step.
        
        Parameters
        ----------
        t_ind : list, optional
            Time index range to process:
            - [] (empty): Process entire time axis
            - [start, stop]: Process self.time[start:stop] only
        debug : bool, default=False
            If True, print debug information during evaluation
        
        Notes
        -----
        **Performance:**
        Evaluation time scales linearly with:
        - Number of time points (len(self.time))
        - Number of energy points (len(self.energy))
        - Model complexity (number of components, time-dependent parameters)
        
        **Memory:**
        Result stored in self.value2D has shape (n_time, n_energy).
        For 1000 time points × 500 energy points × 8 bytes/float:
        ~4 MB per model evaluation.
        
        **Time-Dependence:**
        For each time point t_i:
        1. Time-dependent parameters evaluate their Dynamics at t_i
        2. Model components use these parameter values
        3. 1D spectrum computed and stored in value2D[t_i, :]
        """
        if self.time is None or self.energy is None:
            raise ValueError("Model time and energy axes must be defined for 2D evaluation")

        if len(t_ind) == 0: # process entire time axis
            self.value2D = np.empty((len(self.time), len(self.energy)))
            for ti, t in enumerate(self.time):
                val = self.create_value1D(t_ind=ti, return1D=1)
                if val is None:
                    raise RuntimeError("create_value1D returned None during 2D evaluation")
                self.value2D[ti,:] = val
        else: # process selection according to t_ind
            self.value2D = np.empty((len(self.time[t_ind[0]:t_ind[1]]), len(self.energy)))
            for ti, t in enumerate(self.time[t_ind[0]:t_ind[1]]):
                val = self.create_value1D(t_ind=ti, return1D=1)
                if val is None:
                    raise RuntimeError("create_value1D returned None during 2D evaluation")
                self.value2D[ti,:] = val
        #
        return None
    
    #
    def plot_1D(self, t_ind: int = 0, plot_ind: bool = False, 
                x_lim: Optional[Tuple[float, float]] = None, 
                y_lim: Optional[Tuple[float, float]] = None, 
                save_img: int = 0, save_path: str = '') -> None:
        """
        Plot 1D model spectrum (energy or time).
        
        Visualizes model evaluation results, either as sum of all components
        or as individual component contributions. Useful for checking initial
        guesses and understanding fit results.
        
        Parameters
        ----------
        t_ind : int, default=0
            Time index for energy-resolved models. Ignored for Dynamics models
            which show time evolution.
        plot_ind : bool, default=False
            Component display mode:
            - False: Plot only sum of all components
            - True: Plot each component separately
        x_lim : tuple of float, optional
            X-axis display range (min, max) in axis coordinates
        y_lim : tuple of float, optional
            Y-axis display range (min, max)
        save_img : int, default=0
            Save/display control:
            - 0: Display only
            - 1: Display and save
            - -1: Save only (no display)
        save_path : str, default=''
            Path for saving figure (if save_img != 0)
        """
        # Get model config for plotting
        config = self.plot_config
        
        # Model calling this method is a ...
        # ... time-resolved model (mcp.Dynamics)
        if isinstance(self, Dynamics):
            x_dir = 'def'  # Always default for Dynamics
            if self.time is None:
                raise ValueError("Dynamics model time axis is not defined")
            x = self.time
            x_label = config.y_label  # t_label from project
            info = ''
        # ... energy-resolved model
        else:
            x_dir = config.x_dir
            if self.energy is None or self.time is None:
                raise ValueError("Model energy/time axes are not defined")
            x = self.energy
            x_label = config.x_label  # e_label from project
            info = f'[{config.y_label}={round(self.time[t_ind],3)} (index={t_ind})]'
            
        # Populate component_spectra argument of the model
        self.create_value1D(t_ind, store1D=1)
        if not plot_ind and self.value1D is None:
            raise RuntimeError("Model evaluation did not produce value1D")
        
        # Plot
        plot_data = self.component_spectra if plot_ind else [cast(np.ndarray, self.value1D)]
        uplt.plot_1D(
            data=plot_data,
            x=x,
            config=config,
            title=f'Model "{self.name}" {info}',
            x_label=x_label,
            y_label=config.z_label,
            x_dir=x_dir,
            x_lim=x_lim, y_lim=y_lim,
            legend=[c.name for c in self.components] if plot_ind else ['sum',],
            save_img=save_img,
            save_path=save_path
        )
        #
        return None

    #
    def plot_2D(self, save_img: int = 0, save_path: str = '', 
                x_lim: Optional[Tuple[float, float]] = None, 
                y_lim: Optional[Tuple[float, float]] = None, 
                z_lim: Optional[Tuple[float, float]] = None) -> None:
        """
        Plot 2D time-and-energy spectrum as heatmap.
        
        Visualizes the complete 2D model evaluation showing how the spectrum
        evolves over time. Essential for understanding time-dependent models
        before and after fitting.
        
        Parameters
        ----------
        save_img : int, default=0
            Save/display control:
            - 0: Display only
            - 1: Display and save
            - -1: Save only (no display)
        save_path : str, default=''
            Path for saving figure (if save_img != 0)
        x_lim : tuple of float, optional
            Energy axis display range (min, max)
        y_lim : tuple of float, optional
            Time axis display range (min, max)
        z_lim : tuple of float, optional
            Color scale limits (min, max)
        """
        if self.value2D is None:
            self.create_value2D()
        if self.value2D is None or self.energy is None or self.time is None:
            raise ValueError("Model value2D, energy, and time must be defined for plot_2D")
        # Plot using the utility plot_2D
        uplt.plot_2D(
            data=self.value2D,
            x=self.energy,
            y=self.time,
            config=self.plot_config,
            title=f'model "{self.name}"',
            x_lim=x_lim, y_lim=y_lim, z_lim=z_lim,
            save_img=save_img,
            save_path=save_path
        )
        #
        return None
    
#    
#
class Component:
    """
    Individual spectral or temporal component with parameter management.
    
    Component wraps a function from trspecfit.functions (energy or time) into
    a model-ready object with parameter management, axis handling, and
    integration within the Model/Component/Parameter hierarchy.
    
    Parameters
    ----------
    comp_name : str
        Component name, possibly with numbering (e.g., 'GLP_01', 'expFun_02').
        Numbering is typically assigned during YAML parsing.
    package : module, optional
        Python module containing the component function (fcts_energy or fcts_time).
        Defaults to fcts_energy (energy-resolved functions)
    comp_subcycle : int, default=0
        Subcycle number for multi-cycle Dynamics models:
        - 0: Active for entire time axis (default)
        - 1, 2, ...: Active only during specific subcycle
    
    Attributes
    ----------
    package : module
        Module containing the function (fcts_energy or fcts_time)
    comp_name : str
        Full component name including any numbering
    fct_str : str
        Base function name without numbering (e.g., 'GLP', 'expFun')
    N : int
        Component number (e.g., 1, 2) or -1 if unnumbered
    comp_type : str
        Component type determining combination method:
        - 'add': Regular addition (peaks, lineshapes)
        - 'back': Background (requires existing spectrum)
        - 'conv': Convolution kernel
        - 'none': Placeholder (no operation)
    par_dict : dict
        Parameter specifications from YAML: {name: [value, vary, min, max]}
    subcycle : int
        Subcycle number for multi-cycle dynamics
    time_N_sub : ndarray or None
        Binary mask (1=active, 0=inactive) for subcycle timing
    time_norm : ndarray or None
        Normalized time axis (resets to 0 at each subcycle start)
    pars : list of Par
        Parameter objects for this component
    lmfit_par_list : list of lmfit.Parameter
        Flattened list of lmfit parameters
    time : ndarray or None
        Time axis (inherited from model, or kernel axis for convolutions)
    energy : ndarray or None
        Energy axis (inherited from model)
    parent_model : Model or None
        Parent model reference (for parameter lookups)

    Notes
    -----
    **Component Properties:**

    The Component class provides several computed properties for accessing
    function information:

    - ``fct`` : callable - Function object (auto-updates if package or fct_str changes)
    - ``fct_specs`` : inspect.FullArgSpec - Function signature information
    - ``fct_args`` : list of str - Function argument names
    - ``prefix`` : str - Prefix for parameter names ('' for exceptions, comp_name+'_' otherwise)
    - ``name`` : str - Component display name
    """
    #
    def __init__(self, comp_name: str, package: Optional[types.ModuleType] = None,
                 comp_subcycle: int = 0) -> None:
        # package containing component (either fcts_energy or fcts_time)
        if package is None:
            package = fcts_energy
        self.package: types.ModuleType = package
        # name of the component (str)
        self.comp_name: str = comp_name
        # parse the component name into function string and component number
        self.fct_str: str
        self.N: Optional[int]
        self.fct_str, self.N = uparsing.parse_component_name(comp_name)       
        # determine component type: 'add', 'conv', 'back', or 'none'
        if self.fct_str in background_functions():
            self.comp_type: str = 'back'
        elif self.fct_str == 'none': # placeholder function (see src/functions/time.py)
            self.comp_type = 'none'
        elif 'CONV' in self.fct_str or self.fct_str.endswith('CONV'):
            self.comp_type = 'conv'
        else:
            self.comp_type = 'add'
        # dict of par_name: par_info from yaml file passed by user
        self.par_dict: Dict[str, List] = {}
        # (for self.package=fcts_time) which subcycle is this component part of 
        self.subcycle: int = comp_subcycle # see "t-dynamics.normalize_time" for more details
        self.time_N_sub: Optional[np.ndarray] = None # 1 where model subcycle equals component subcycle
        self.time_norm: Optional[np.ndarray] = None # restarts at zero for every subcycle
        # list of Par objects needed to construct component
        self.pars: List['Par'] = [] # used to create component value during fit
        # flattened list of all lmfit parameters defining this component
        self.lmfit_par_list: List[lmfit.Parameter] = []
        # time and energy axis of component are inherited from model
        self.time: Optional[np.ndarray] = None
        self.energy: Optional[np.ndarray] = None
        # parent model reference
        self.parent_model: Optional[Model] = None
        #
        return None
    
    # [automatic] create self.fct attribute that will update if either
    # self.package or self.fct_str changes [attribute is read only]
    @property 
    def fct(self) -> Callable:
        """
        Function object for this component.
        
        Automatically retrieves function from package using fct_str.
        Updates dynamically if package or fct_str changes.
        
        Returns
        -------
        callable
            Function object (e.g., fcts_energy.GLP, fcts_time.expFun)
        """
        return cast(Callable, getattr(self.package, self.fct_str))
    
    # [automatic] do the same for function argument specs
    @property 
    def fct_specs(self) -> inspect.FullArgSpec:
        """
        Function signature specifications.
        
        Returns
        -------
        inspect.FullArgSpec
            Complete function signature information
        """
        return inspect.getfullargspec(self.fct)
    
    # [automatic] and function arguments specifically
    @property 
    def fct_args(self) -> List[str]:
        """
        Function argument names.
        
        Returns
        -------
        list of str
            Argument names from function signature
        """
        return self.fct_specs.args
    
    # [automatic] create a prefix for parameters of this component
    @property 
    def prefix(self) -> str:
        """
        Prefix for parameter names.
        
        Returns
        -------
        str
            Parameter prefix:
            comp_name + '_': For regular components
        """
        # component number handled by self.N
        return self.comp_name +'_'
        
    # [automatic] create a name for this component
    @property 
    def name(self) -> str:
        """
        Component display name.
        
        Returns
        -------
        str
            Component name (same as comp_name)
        """
        # use the stored component name (which includes numbering if applicable)
        return self.comp_name
    
    #
    def _add_prefix_to_expression(self, expr: str, prefix: str) -> str:
        """
        Add prefix to time function parameter references in Dynamics models.
        
        For Dynamics models, parameters of time functions need the Dynamics model
        name as a prefix, while energy function parameters reference the parent
        energy model directly without prefix.
        
        Parameters
        ----------
        expr : str
            Expression string from YAML (e.g., "expFun_01_tau + GLP_01_A")
        prefix : str
            Dynamics model name to prepend (e.g., "GLP_01_x0")
        
        Returns
        -------
        str
            Expression with time function parameters prefixed
            (e.g., "GLP_01_x0_expFun_01_tau + GLP_01_A")
        
        Examples
        --------
        For Dynamics model "GLP_01_x0":
        - "expFun_01_tau" -> "GLP_01_x0_expFun_01_tau" (time function)
        - "GLP_01_A" -> "GLP_01_A" (energy function, unchanged)
        - "expFun_01_tau * 0.5 + GLP_01_A" -> "GLP_01_x0_expFun_01_tau * 0.5 + GLP_01_A"
        """
        # Get function names from both libraries
        time_funcs = time_functions()
        energy_funcs = energy_functions()
        conv_funcs = convolution_functions()
        
        # Pattern to match parameter references: function_name_NN_param_name
        # Captures: (function_name)(_NN_param_name)
        pattern = r'\b([A-Za-z_][A-Za-z0-9_]*?)(_\d{2}_[A-Za-z_][A-Za-z0-9_]*)\b'
        
        def replace_with_prefix(match: re.Match[str]) -> str:
            func_name = match.group(1)  # e.g., "expFun" or "GLP"
            rest = match.group(2)        # e.g., "_01_tau"
            full_match = match.group(0)  # e.g., "expFun_01_tau"
            
            # Defensive: Check if already prefixed
            if full_match.startswith(prefix + "_"):
                return full_match
            
            # Time functions need prefix (they're in this Dynamics model)
            if func_name in time_funcs or func_name in conv_funcs:
                return f"{prefix}{func_name}{rest}"
            
            # Energy functions don't need prefix (they reference parent energy model)
            elif func_name in energy_funcs:
                return full_match
            
            # Unknown function - leave unchanged and let lmfit error naturally
            else:
                return full_match
        
        return re.sub(pattern, replace_with_prefix, expr)

    #
    def add_pars(self, par_info_dict: Dict[str, List]) -> None:
        """
        Add parameter specifications to component.
        
        Stores parameter information that will be used to create Par objects
        when create_pars() is called. Typically populated from YAML model
        definitions.
        
        Parameters
        ----------
        par_info_dict : dict
            Parameter specifications: {name: [value, vary, min, max]} for
            constrained or {name: [value], vary} for unconstrained parameters,
            or {name: ['expression']} for dependent parameters
        
        Notes
        -----
        Does not create the actual Par objects yet - that happens in
        create_pars(). This separation allows parameters to be defined
        before axes are known.
        """
        self.par_dict = par_info_dict
        #
        return None
    
    #
    def create_pars(self, prefix: str = '', debug: bool = False) -> None:
        """
        Create Par objects from parameter dictionary.
        
        Populates self.pars with Par objects for each entry in self.par_dict.
        Uses two-pass approach to handle expression parameters that may
        reference parameters defined later in the component.
        
        Parameters
        ----------
        prefix : str, default=''
            Prefix to prepend to parameter names (e.g., for Dynamics models)
        debug : bool, default=False
            If True, print parameter details during creation
        
        Notes
        -----
        **Two-Pass Creation:**
        1. First pass: Create all Par objects with values/bounds
        2. Second pass: Set expressions (so forward references work)
        
        This ensures expressions like 'GLP_02_A' work even when GLP_02
        is defined after GLP_01 in the component list.
        """
        lst = [] # initialize pars list
        if len(prefix) > 0:
            prefix += '_'

        # First pass: create all Par objects, but do not set expressions
        expr_params: List[Tuple['Par', str]] = []
        for p_name, p_info in self.par_dict.items():
            temp = Par(name=prefix + self.prefix + p_name)
            temp.info = p_info  # see Par class for details
            # Set parent model reference
            temp.parent_model = self.parent_model
            # If this is an expression, skip setting it for now
            if len(p_info) == 1 and isinstance(p_info[0], str):
                expr = p_info[0]
                # If this component is part of a Dynamics model, 
                # auto-prefix time function parameter references
                if isinstance(self.parent_model, Dynamics) and prefix:
                    expr = self._add_prefix_to_expression(expr, prefix)
                    if debug:
                        print(f"Expression transform: '{p_info[0]}' -> '{expr}'")
                expr_params.append((temp, expr))
                # Temporarily set a dummy value (needed for lmfit creation)      
                temp.create(expr_skip=True)
            else:
                temp.create()
            if debug:
                temp.describe()
            lst.append(temp)
        self.pars = lst
        
        # Second pass: set expressions now that all parameters exist
        for temp, expr in expr_params:
            # Set the expression on the lmfit parameter
            par_name = temp.name
            try:
                temp.lmfit_par[par_name].set(expr=expr)
            except Exception as e:
                print(f"Failed to set expr '{expr}' for parameter '{par_name}': {e}")
        #
        return None
    
    #
    def update_lmfit_par_list(self) -> None:
        """
        Update flattened list of lmfit parameters.
        
        Collects all lmfit.Parameter objects from all Par objects in this
        component and stores them in a flat list. This includes both spectral
        and temporal parameters (if any Par has time-dependence).
        
        Notes
        -----
        Called automatically by Model.update(). Users typically don't need
        to call this directly.
        
        The flattened list is used by Model to construct the complete
        lmfit.Parameters object for fitting.
        """
        # re-initialize the list
        self.lmfit_par_list = []
        # go through all pars of this component ...
        for pi, p in enumerate(self.pars):
            # ... and add their list of all lmfit.Parameter objects 
            self.lmfit_par_list.extend(p.lmfit_par_list)
        #
        return None
    
    #
    def describe(self, detail: int = 1) -> None:
        """
        Print component information.
        
        Parameters
        ----------
        detail : int, default=1
            Detail level:
            - 0: Function name only
            - 1+: Function name, type, subcycle, and parameters
        """
        # print info on function
        print(f'function: {self.fct_str} from {self.package}')
        # detailed description
        if detail >= 1:
            # addition or convolution?
            if self.comp_type == 'add':
                comp_type_str = 'added to other components'
            elif self.comp_type == 'conv':
                comp_type_str = 'convoluted with other components'
            elif self.comp_type == 'back':
                comp_type_str = 'added as background to other components'
            elif self.comp_type == 'none':
                comp_type_str = 'skipped (no operation)'
            # subcycle info
            if self.subcycle == 0:
                subcycle_str = 'for all times t'
            else:
                subcycle_str = f'within subcycle {self.subcycle}'
            # print info
            print(f'function will be {comp_type_str} [{subcycle_str}]')

            # print info on passed parameters
            if hasattr(self, 'par_dict') and self.par_dict:
                for param_name, param_value in self.par_dict.items():
                    display(f'{self.prefix}{param_name}', param_value)
            else:
                print('No parameter info passed!')
            print()
        #
        return None
    
    #
    def create_t_kernel(self, debug: bool = False) -> np.ndarray:
        """
        Create time axis for convolution kernel.
        
        Convolution kernels need a time axis that extends beyond the data
        time axis to properly handle edge effects. This method creates an
        appropriately sized kernel axis based on the kernel width.
        
        Parameters
        ----------
        debug : bool, default=False
            If True, print kernel axis details
        
        Returns
        -------
        ndarray
            Kernel time axis, centered at 0 and extending ±(width * kernel_parameter)
        """
        # get kernel parameters i.e. component parameters
        parK = cast(list[Any], ulmfit.par_extract(self.par_dict, return_type='list'))
        if debug: print(f'component/kernel parameters as list: {parK}')
        # define kernel time axis
        kernel_width = getattr(fcts_time, self.fct_str +'_kernel_width')()
        if debug: print(f'kernel width loaded from fcts_time: {kernel_width}')
        t_range = parK[0] *kernel_width
        if self.time is None or len(self.time) < 2:
            raise ValueError(f'time axis of component {self.fct_str} not defined')
        t_step = self.time[1] -self.time[0]
        if debug: print(f'delta time (from self.time): {t_step}')
        t_kernel = np.arange(-t_range, t_range+t_step, t_step)
        #
        return t_kernel

    #
    def value(self, t_ind: int = 0, **kwargs) -> np.ndarray:
        """
        Evaluate component at specific time point.
        
        Computes the component's contribution to the spectrum using current
        parameter values. Handles time-dependent parameters, subcycle masking,
        and passes appropriate axes.
        
        Parameters
        ----------
        t_ind : int, default=0
            Time index for evaluation (affects time-dependent parameters)
        **kwargs : dict
            Additional arguments passed to function (e.g., spectrum for backgrounds)
        
        Returns
        -------
        ndarray
            Component value as function of energy or time
        
        Notes
        -----
        **Parameter Evaluation:**
        Each Par object is evaluated at t_ind, which:
        - Returns current value for time-independent parameters
        - Returns value(t_ind) for time-dependent parameters (via Dynamics)
        
        **Subcycle Handling:**
        For multi-cycle Dynamics models (subcycle != 0):
        - Uses time_norm instead of time (resets to 0 each subcycle)
        - Multiplies result by time_N_sub mask (1=active, 0=inactive)
        
        **Background Functions:**
        Background functions receive the 'spectrum' kwarg containing the
        current peak sum, which they use to compute backgrounds like Shirley.
        """
        # get component parameters as list
        pars = []
        # HOTFIX NOTE:
        # Par.value() currently returns a one-element list for compatibility.
        # Future cleanup: make Par.value() return scalar float and switch
        # this call from pars.extend(...) to pars.append(...).
        for p in self.pars:
            #$% slightly hacky to only update Par.t_model for t_ind=0, change?
            pars.extend(p.value(t_ind,
                                update_t_model = True if t_ind==0 else False))

        # get x axis and create component function evaluation
        if self.package == fcts_energy:
            if self.energy is None:
                raise ValueError(f"Energy axis not defined for component '{self.comp_name}'")
            return np.asarray(self.fct(self.energy, *pars, **kwargs))
        elif self.package == fcts_time:
            if self.time is None:
                raise ValueError(f"Time axis not defined for component '{self.comp_name}'")
            if self.subcycle == 0: # single cycle
                return np.asarray(self.fct(self.time, *pars, **kwargs))
            else: # multi-cycle
                # multpliy value with 1 where subcycle applies, 0 otherwise
                # use normalized time instead of standard time for sub!=0]
                if self.time_norm is None or self.time_N_sub is None:
                    raise ValueError(f"Subcycle axes not defined for component '{self.comp_name}'")
                return np.asarray(self.fct(self.time_norm, *pars, **kwargs) *self.time_N_sub)
        else:
            raise ValueError(f"Unsupported function package for component '{self.comp_name}'")
    
    #
    def plot(self, t_ind: int = 0, **kwargs) -> None:
        """
        Plot component as standalone spectrum/dynamics.
        
        Quick visualization of individual component behavior, useful for
        debugging component definitions and understanding parameter effects.
        
        Parameters
        ----------
        t_ind : int, default=0
            Time index for evaluation
        **kwargs : dict
            Additional arguments passed to component function
        """
        # get x axis and its label
        if self.package == fcts_energy: 
            x_axis = self.energy
            x_name = 'Energy'
        elif self.package == fcts_time:
            x_axis = self.time
            x_name = 'Time'
        #
        uplt.plot_1D(data = [self.value(t_ind, **kwargs), ],
                     title = f'function: {self.fct_str} from {self.package}]',
                     x = x_axis, x_label = x_name, y_label = 'Amplitude')
        #
        return None

#
# 
class Par:
    """
    Parameter with optional time-dependence and expression support.
    
    Par extends lmfit.Parameter to support:
    - Time-varying parameters via Dynamics models
    - Expression-based constraints referencing other parameters
    - Tracking of time-dependent expression references
    
    Parameters
    ----------
    name : str
        Parameter name (typically prefixed with component name)
    info : list, default=[]
        Parameter specification:
        - [value, vary, min, max]: Standard parameter
        - ['expression']: Expression-based parameter
    
    Attributes
    ----------
    name : str
        Parameter name
    info : list
        Parameter specification from initialization
    lmfit_par : lmfit.Parameters
        lmfit Parameters object (contains 1+ parameters)
    t_vary : bool
        Whether parameter has time-dependence (via Dynamics)
    t_model : Model
        Dynamics model describing time evolution (if t_vary=True)
    lmfit_par_list : list
        Flattened list of all lmfit parameters (spectral + temporal)
    expr_refs_time_dep : bool
        Whether expression references time-dependent parameters
    expr_string : str or None
        Original expression string (for expression parameters)
    expr_refs : list of str
        Parameter names referenced in expression
    parent_model : Model or None
        Parent model reference (for finding other parameters)
    
    Notes
    -----
    **Time-Dependence:**

    When t_vary=True, the parameter value at time t is::

        value(t) = base_value + dynamics_model.value1D[t]

    **Expression Handling:**

    Expressions are evaluated using asteval (same as lmfit) for safety.
    Can reference other parameters, including time-dependent ones.

    **Expression + Dynamics:**

    A parameter can have an expression that references a time-dependent
    parameter. In this case, expr_refs_time_dep=True and the expression
    is re-evaluated at each time point during model evaluation.

    **Parameter Flattening:**

    lmfit_par_list contains all parameters defining this Par:

    - Without time-dependence: 1 parameter (the spectral one)
    - With time-dependence: N parameters (spectral + all from Dynamics model)
    """
    #
    def __init__(self, name: str, info: Optional[list[Any]] = None) -> None:
        self.name = name
        self.info: list[Any] = [] if info is None else list(info)
        self.lmfit_par: lmfit.Parameters = lmfit.Parameters()
        self.t_vary: bool = False
        self.t_model: Model = Model(f'{name}_tModel')
        self.lmfit_par_list: List[lmfit.Parameter] = []
        # Expression analysis attributes
        self.expr_refs_time_dep: bool = False  # flag for time-dependent references
        self.expr_string: Optional[str] = None  # store original expression
        self.expr_refs: List[str] = []          # list of referenced parameter names
        self.parent_model: Optional[Model] = None  # reference to parent model
        #
        return None
    
    #
    def describe(self, detail: int = 0) -> None:
        """
        Print parameter information.
        
        Parameters
        ----------
        detail : int, default=0
            Detail level for display (passed to t_model.describe if applicable)
        """
        print(f'par name: {self.name} [value: {self.value()}] and its lmfit_par attribute:')
        try: 
            self.lmfit_par.pretty_print()
        except: 
            print('[this is not an lmfit.Parameter instance]')
            display(self.lmfit_par)
        #
        if self.t_vary == False:
            print('parameter has no time dependence')
        elif self.t_vary == True:
            print(f'parameter has time-dependence described by model {self.t_model.name}')
            if detail == 1:
                self.t_model.describe()
        #
        return None
        
    #
    def create(self, prefix: str = '', suffix: str = '', expr_skip: bool = False, debug: bool = False) -> None:
        """
        Create lmfit parameter from info specification.
        
        Initializes the lmfit.Parameters object with the parameter defined
        by self.info. Handles both standard parameters and expression-based
        parameters.
        
        Parameters
        ----------
        prefix : str, default=''
            Prefix to prepend to parameter name
        suffix : str, default=''
            Suffix to append to parameter name
        expr_skip : bool, default=False
            If True and info is expression, create with dummy value first
            (expression set later in two-pass creation)
        debug : bool, default=False
            If True, print creation details
        
        Notes
        -----
        For expression parameters, expr_skip=True creates a temporary
        parameter with dummy value. The actual expression is set in a
        second pass (see Component.create_pars) to handle forward references.
        """
        # create standard lmfit parameter (spectral component)
        if expr_skip and len(self.info) == 1 and isinstance(self.info[0], str):
            # if skipping expression, use a dummy value for now
            lmfit_par = ulmfit.par_create(self.name, [1, True, -np.inf, np.inf], prefix, suffix, debug)
        else:
            lmfit_par = ulmfit.par_create(self.name, self.info, prefix, suffix, debug)
        # add to lmfit_par attribute
        self.lmfit_par.add_many(lmfit_par)
        # and list of individual lmfit paramters
        self.lmfit_par_list.extend([lmfit_par])    
        #
        return None
    
    #
    def update(self, t_model: 'Dynamics') -> None:
        """
        Add time-dependence to parameter via Dynamics model.
        
        Converts a static parameter into a time-dependent one by attaching
        a Dynamics model that describes temporal evolution.
        
        Parameters
        ----------
        t_model : Dynamics
            Dynamics model describing time evolution
        """
        # update t_vary (default = False)
        self.t_vary = True
        # update t_model attribute
        self.t_model = t_model
        # evaluate t_model to update/create model.value1D
        self.t_model.create_value1D()
        # add t_model pars to list of individual lmfit parameters
        self.lmfit_par_list.extend(self.t_model.lmfit_par_list)
        #
        return None
    
    #
    def value(self, t_ind: int = 0, update_t_model: bool = True) -> list[Any]:
        """
        Get parameter value at specific time point.
        
        Returns the parameter value, accounting for time-dependence and
        expressions that may reference time-dependent parameters.
        
        Parameters
        ----------
        t_ind : int, default=0
            Time index for evaluation
        update_t_model : bool, default=True
            If True, recompute Dynamics model before evaluation.
            Set False when calling repeatedly during 2D model evaluation.
        
        Returns
        -------
        float or list
            Parameter value(s) at time point t_ind
        
        Notes
        -----
        **Return Type:**
        Returns list to maintain compatibility with lmfit parameter extraction.
        For single parameters, this is a one-element list.
        
        **Update Control:**
        The update_t_model flag optimizes 2D model evaluation:
        - True (default): Recalculate dynamics (safe, slightly slower)
        - False: Use cached dynamics (faster, assumes unchanged)
        
        During 2D fitting:
        - First time point (t_ind=0): update_t_model=True
        - Subsequent points: update_t_model=False
        
        **Expression Evaluation:**
        For expressions referencing time-dependent parameters, the expression
        is evaluated with current values of all referenced parameters at t_ind.
        """
        if self.t_vary == False:
            if self.expr_refs_time_dep:
                # Custom evaluation for time-dependent expressions
                all_parameters = self._get_all_parameters()
                return self._evaluate_time_dependent_expression(t_ind, all_parameters, update_t_model)
            else:
                # Standard lmfit evaluation
                value = cast(list[Any], ulmfit.par_extract(self.lmfit_par))

        elif self.t_vary == True:
            if update_t_model == True:
                self.t_model.create_value1D() # update t_model, specifically self.t_model.value1D
            base = cast(list[Any], ulmfit.par_extract(self.lmfit_par))
            if self.t_model.value1D is None:
                raise RuntimeError(f'Dynamics model "{self.t_model.name}" has no value1D')
            value = [base[0] + self.t_model.value1D[t_ind]]
            
        else:
            value = [-1]
            print(f't_vary attribute of Par "{self.name}" is not valid')
        #
        return value
    
    #
    def analyze_expression_dependencies(self, all_parameters: list['Par']) -> None:
        """
        Analyze expression for time-dependent parameter references.
        
        Checks if this parameter's expression references any time-dependent
        parameters. If so, sets expr_refs_time_dep flag so value() can
        handle dynamic expression evaluation.
        Called automatically after adding time-dependence to any parameter.

        Parameters
        ----------
        all_parameters : list of Par
            All parameters in parent model (to check time-dependence)
        """
        if len(self.info) == 1 and isinstance(self.info[0], str):
            self.expr_string = self.info[0]
            # Parse expression to find referenced parameters
            self.expr_refs = uparsing.extract_expression_parameters(self.expr_string)
            
            # Check if any referenced parameters are time-dependent
            for ref_name in self.expr_refs:
                ref_par = self._find_parameter_by_name(ref_name, all_parameters)
                if ref_par and ref_par.t_vary:
                    self.expr_refs_time_dep = True
                    break
        #
        return None
    
    #
    def _find_parameter_by_name(self, par_name: str, all_parameters: list['Par']) -> Optional['Par']:
        """
        Find parameter by name in list.
        
        Parameters
        ----------
        par_name : str
            Parameter name to find
        all_parameters : list of Par
            List to search
        
        Returns
        -------
        Par or None
            Found parameter or None if not found
        """
        # Search through all parameters
        for par in all_parameters:
            if par.name == par_name:
                return par
        return None
    
    #
    def _evaluate_time_dependent_expression(self, t_ind: int, all_parameters: list['Par'], update_t_model: bool = True) -> list[Any]:
        """
        Evaluate expression with time-dependent parameter values.
        
        For expressions that reference time-dependent parameters, this
        evaluates the expression using the current values of all referenced
        parameters at the specified time point.
        
        Parameters
        ----------
        t_ind : int
            Time index for evaluation
        all_parameters : list of Par
            All parameters in model (to get current values)
        update_t_model : bool, default=True
            Whether to update Dynamics models before evaluation
        
        Returns
        -------
        list
            Evaluated expression value (as list for lmfit compatibility)
        
        Notes
        -----
        Uses asteval for safe expression evaluation (same as lmfit).
        All referenced parameters are evaluated at t_ind and passed to
        the expression evaluator.
        
        See Also
        --------
        value : Main evaluation method that calls this
        analyze_expression_dependencies : Sets up for this evaluation
        """
        # HOTFIX: use scalar values in asteval namespace.
        # Passing one-element lists (from Par.value) can make expressions like
        # "GLP_01_x0 + 3.6" evaluate to None inside asteval.
        namespace = {}
        for ref_name in self.expr_refs:
            ref_par = self._find_parameter_by_name(ref_name, all_parameters)
            if ref_par:
                ref_val = ref_par.value(t_ind, update_t_model=update_t_model)
                if isinstance(ref_val, list):
                    if len(ref_val) == 0:
                        raise ValueError(f"Referenced parameter '{ref_name}' returned empty value list")
                    namespace[ref_name] = ref_val[0]
                else:
                    namespace[ref_name] = ref_val
        
        # Evaluate expression using asteval (safe, same as lmfit uses)
        try:
            aeval = Interpreter()
            # populate the interpreter symbol table with current parameter values
            # (Interpreter.__call__ doesn't accept a namespace argument)
            for k, v in namespace.items():
                aeval.symtable[k] = v
            expr = cast(str, self.expr_string)
            result = aeval(expr)
            # HOTFIX: asteval may record errors and return None without raising.
            if aeval.error:
                msg = "; ".join(err.get_error()[1] for err in aeval.error)
                raise ValueError(
                    f"Asteval error while evaluating expression '{self.expr_string}': {msg}"
                )
            return result if isinstance(result, list) else [result]
        except Exception as e:
            raise ValueError(f"Error evaluating expression '{self.expr_string}': {e}")
    
    #
    def _get_all_parameters(self) -> list['Par']:
        """
        Get all parameters from parent model.
        
        Returns
        -------
        list of Par
            All parameters in parent model
        """
        if hasattr(self, 'parent_model') and self.parent_model:
            return self.parent_model._get_all_parameters()
        return []

#
#
class Dynamics(Model):
    """
    Time-dependence model for parameters with multi-cycle support.
    
    Dynamics is a specialized Model that describes how a parameter evolves
    over time. It uses temporal functions (from trspecfit.functions.time) to
    model dynamics like exponential decays, rises, oscillations, and can
    include convolution with instrumental response functions.
    
    The name of a Dynamics model must match the parameter it describes
    (e.g., 'GLP_01_x0' for the x0 parameter of the GLP_01 component).
    
    Parameters
    ----------
    model_name : str
        Name matching the parameter to control (e.g., 'GLP_01_x0')
    
    Attributes
    ----------
    All Model attributes, plus:
    
    frequency : float
        Repetition frequency for cyclic dynamics (Hz):
        - -1: Single cycle over entire time axis (default)
        - >0: Dynamics repeat at this frequency
    subcycles : int
        Number of subcycles within each main cycle:
        - 0: No subcycles (single dynamics for entire cycle)
        - N>0: N different dynamics that activate sequentially
    time_norm : ndarray or None
        Normalized time that resets to 0 at start of each subcycle
    N_sub : ndarray or None
        Subcycle number active at each time point (1, 2, ..., subcycles)
    N_counter : ndarray or None
        Cumulative subcycle counter (increments each subcycle)
    
    Notes
    -----
    **Single Cycle (frequency=-1):**
    Dynamics apply once over the entire time axis. Appropriate for:
    - Single pump-probe experiments
    - Irreversible reactions
    - One-time perturbations
    
    **Multi-Cycle (frequency>0):**
    Dynamics repeat periodically. Appropriate for:
    - Lock-in detection experiments
    - Periodic excitation (lasers, electrical pulses)
    - Steady-state oscillations
    
    **Subcycles:**
    Within each main cycle, different dynamics can activate sequentially.
    Example: pump-probe-pump experiments where:
    - Subcycle 1: First pump response
    - Subcycle 2: Second pump response
    
    Components are assigned to subcycles via comp_subcycle parameter:
    - subcycle=0: Active for all times (e.g., IRF convolution)
    - subcycle=1,2,...: Active only during that subcycle
    
    **Time Normalization:**
    For multi-cycle dynamics:
    - time_norm resets to 0 at each subcycle start
    - N_sub tracks which subcycle is active (1, 2, 3, ...)
    - N_counter cumulative count of subcycles
    
    **Evaluation:**
    The dynamics model evaluates to value1D, which is added to the base
    parameter value: param_total(t) = param_base + dynamics.value1D[t]
    """
    #
    def __init__(self, model_name: str) -> None:
        super().__init__(model_name)     
        # repetition frequency of time-dependent model behaviour
        self.frequency: float = -1
        # number of subcycles (within time = 1/frequency)
        self.subcycles: int = 0 # defined via "number of models -1" [model_info in file.load_model()]
        # "normalized time" attributes (all have same length as time axis)
        self.time_norm: Optional[np.ndarray] = None # restart time (at 0) every 1/(freqency *number of subcycles)
        self.N_sub: Optional[np.ndarray] = None # which subcycle is active at time step (t_i)
        self.N_counter: Optional[np.ndarray] = None # cummulative counter of subcycles (at t_i)
        self.parent_model: Optional[Model] = None
        #
        return None
    
    #
    def set_frequency(self, frequency: float, time_unit: int = 0) -> None:
        """
        Set repetition frequency and update time normalization.
        
        Configures the Dynamics for cyclic behavior and updates all components
        with proper subcycle timing information.
        
        Parameters
        ----------
        frequency : float
            Repetition frequency in Hz:
            - -1: No repetition (single cycle)
            - >0: Repeat at this frequency
        time_unit : int, default=0
            Power of 10 for time axis units (e.g., 0 for seconds, -3 for ms).
            Currently disabled/unused.
        
        Notes
        -----
        After setting frequency:
        - time_norm, N_sub, N_counter are computed via normalize_time()
        - Each component receives time_N_sub mask (1=active, 0=inactive)
        - Components with subcycle>0 use time_norm instead of time
        
        **Component Updates:**
        For each component:
        - time_N_sub mask applied (zeros where subcycle doesn't match)
        - Normalized time axis inherited (if subcycle != 0)
        """
        self.frequency = frequency # set the frequency attribute itslef
        self.normalize_time() # update the normalization of the time axis
        if self.N_sub is None:
            raise RuntimeError("N_sub not initialized; call normalize_time() first")
        # update components accordingly
        for comp in self.components:
            # <time_N_sub> is 0/1 where subcomponent is in-/active
            if comp.time_N_sub is None:
                comp.time_N_sub = np.ones(len(self.N_sub))
            comp.time_N_sub[self.N_sub != comp.subcycle] = 0
            # inherit normalized time from Dynamics model
            if comp.subcycle != 0: comp.time_norm = self.time_norm
            # no updating needed on the parameter level as time irrelevant
            # parameter level uses index to refer to par.t_model=Dynamics
        # 
        return None

    #
    def normalize_time(self, time_unit: int = 0, debug: bool = False) -> None:
        """
        Normalize time axis for multi-cycle dynamics with subcycles.
        
        Creates normalized time arrays that reset periodically based on
        frequency and subcycle count. This enables complex multi-cycle
        dynamics like pump-probe-pump experiments.
        
        Parameters
        ----------
        time_unit : int, default=0
            Power of 10 for time units (currently unused)
        debug : bool, default=False
            If True, plot normalized time arrays
        
        Examples
        --------
        >>> t_model = Dynamics('param')
        >>> t_model.time = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
        >>> t_model.frequency = 10  # 10 Hz = 0.1 s period
        >>> t_model.subcycles = 2   # Two subcycles per period
        >>> t_model.normalize_time()
        >>> 
        >>> print(t_model.time_norm)  # Resets every 0.05 s (half period)
        [0, 0.05, 0, 0.05, 0, 0.05, 0]
        >>> print(t_model.N_sub)  # Which subcycle (1 or 2)
        [1, 1, 2, 2, 1, 1, 2]
        >>> print(t_model.N_counter)  # Cumulative count
        [1, 1, 2, 2, 3, 3, 4]
        
        Notes
        -----
        **Normalization Logic:**
        - Subcycle duration = 1 / (frequency * subcycles)
        - time_norm resets to 0 at start of each subcycle
        - N_sub cycles through 1, 2, ..., subcycles
        - N_counter increments by 1 each subcycle
        
        **Negative Times:**
        Times t < 0 are assigned:
        - time_norm = 0
        - N_sub = 0 (baseline/pre-trigger)
        - N_counter = 0
        
        **No Repetition (frequency=-1):**
        - time_norm = time (unchanged)
        - N_sub = 0 (all zeros)
        - N_counter = 0 (all zeros)
        
        **Validation:**
        Raises ValueError for:
        - subcycles = 1 (use subcycles=0 for no subdivision)
        - frequency < 0 and != -1
        - subcycles > 1 with frequency = -1 (inconsistent)
        """
        if self.time is None:
            raise ValueError("Dynamics.time axis must be defined")
        # Sanity checks
        if self.subcycles == 1 or not isinstance(self.subcycles, int):
            raise ValueError(
                'Subcycle (N) must either be zero or a >= 2 integer. '
                f'Got: {self.subcycles} (type: {type(self.subcycles).__name__})'
            )
        
        if self.frequency < 0 and self.frequency != -1:
            raise ValueError(
                f'Frequency (f) must be >0 (or "-1" for no repetition). '
                f'Got: {self.frequency}'
            )

        # No repetition within data/time window
        if self.frequency == -1:
            if self.subcycles > 1:
                raise ValueError(
                    'Cannot use subcycles (N > 1) without a positive frequency. '
                    f'Got subcycles={self.subcycles} with frequency=-1'
                )
            self.time_norm = np.asarray(self.time)
            self.N_sub = np.zeros(len(self.time))
            self.N_counter = np.zeros(len(self.time))

        # Frequency >0 is passed
        else:
            # Compute repetition/normalization number
            norm = 10**(-time_unit) / self.frequency / self.subcycles 
            t_norm = []
            N_sub = []
            N_counter = []
            
            # Go through time axis and perform normalization
            for i, t_i in enumerate(self.time):
                N_temp = math.floor(t_i / norm)
                if t_i >= 0:  # Subcycles start at t=0
                    N_sub.append(math.floor(N_temp % self.subcycles) + 1)  # Which subcycle is active
                    N_counter.append(N_temp + 1)  # Increments by 1 each subcycle
                    t_norm.append(t_i - N_temp * norm)  # Each subcycle starts with t=0
                else:  # Times t<0 are baseline/pre-trigger/ground state spectra
                    N_sub.append(0)
                    N_counter.append(0)
                    t_norm.append(0)
            
            self.time_norm = np.asarray(t_norm)
            self.N_sub = np.asarray(N_sub)
            self.N_counter = np.asarray(N_counter)
            
        if debug:
            legends = ['normalized time', 'subcycle counter', 'cummulative counter']
            uplt.plot_1D(
                data=[self.time_norm, self.N_sub, self.N_counter], 
                x=self.time,
                x_label=f'Time (1E{time_unit}s)',
                y_type='log',
                legend=legends
            )
        #
        return None
