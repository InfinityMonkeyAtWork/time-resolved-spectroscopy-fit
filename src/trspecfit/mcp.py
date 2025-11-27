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
The MCP system uses composition to build models from bottom-up:

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
import math
import numpy as np
import re
import inspect
import copy
from IPython.display import display
import concurrent.futures
# asteval is used for expressions referencing time-dependent parameters
from asteval import Interpreter
# function library for energy, time, and distribution components
from trspecfit.functions import energy as fcts_energy
from trspecfit.functions import time as fcts_time
from trspecfit.functions import distribution as fcts_dist
# function configurations
from trspecfit.config.functions import prefix_exceptions, background_functions, energy_functions
# plot configuration
from trspecfit.config.plot import PlotConfig


#
def parse_component_name(comp_name):
    """
    Parse a component name into base function name and number.
    
    Component names follow the pattern: function_name or function_name_NN
    where NN is a two-digit number. This function extracts both parts.
    
    Parameters
    ----------
    comp_name : str
        Component name to parse (e.g., 'GLP_01', 'expFun_02', 'Offset')
    
    Returns
    -------
    base_name : str
        Function name without number (e.g., 'GLP', 'expFun', 'Offset')
    number : int
        Component number (e.g., 1, 2) or -1 if unnumbered
    
    Examples
    --------
    >>> parse_component_name('expFun_01')
    ('expFun', 1)
    >>> parse_component_name('GLP_02')
    ('GLP', 2)
    >>> parse_component_name('Offset')
    ('Offset', -1)
    >>> parse_component_name('GaussAsym_15')
    ('GaussAsym', 15)
    
    Notes
    -----
    Numbered components allow multiple instances of the same function type
    in a model (e.g., multiple peaks with different parameters).
    
    Background functions and convolution kernels are typically unnumbered
    as only one instance per model makes sense.
    
    The numbering convention (_01, _02, etc.) is enforced during YAML parsing
    to ensure consistent parameter naming across the model.
    
    See Also
    --------
    get_component_parameters : Get expected parameters for a component type
    Component : Component class that uses this parsing
    """
    if '_' in comp_name and comp_name.split('_')[-1].isdigit():
        parts = comp_name.split('_')
        if len(parts) >= 2 and parts[-1].isdigit():
            base_name = '_'.join(parts[:-1])
            number = int(parts[-1])
        else:
            base_name = comp_name
            number = -1
    else:
        base_name = comp_name
        number = -1
    #
    return base_name, number

#
def get_component_parameters(comp_name):
    """
    Get the expected parameter names for a component by inspecting its function.
    
    Extracts parameter names from the function signature, excluding the
    independent variable (first parameter) and special parameters like
    'spectrum' (for background functions).
    
    Parameters
    ----------
    comp_name : str
        Base name of the component function (e.g., 'GLP', 'expFun', 'Shirley')
    
    Returns
    -------
    list of str
        Parameter names expected by this component, excluding:
        - First parameter (x or t axis)
        - 'spectrum' parameter (for background functions)
    
    Examples
    --------
    >>> get_component_parameters('GLP')
    ['A', 'x0', 'F', 'm']
    >>> get_component_parameters('expFun')
    ['A', 'tau', 't0', 'y0']
    >>> get_component_parameters('Shirley')
    ['pShirley']
    
    Notes
    -----
    This function searches for the component in the function libraries:
    - fcts_energy (spectral components and backgrounds)
    - fcts_time (temporal dynamics and convolutions)
    - fcts_dist (parameter distributions)
    
    Used for:
    - YAML validation: Checking if user-provided parameters match expectations
    - Error messages: Providing helpful feedback when parameters are wrong
    - Auto-completion: Suggesting parameter names in interactive use
    
    Returns empty list if function not found in any module.
    
    See Also
    --------
    parse_component_name : Parse component names into base name and number
    Component.create_pars : Create parameters for a component
    """
    # Find which module contains this function
    func = None
    for module in [fcts_energy, fcts_time, fcts_dist]:
        if hasattr(module, comp_name):
            func = getattr(module, comp_name)
            break
    
    if func is None:
        return []
    
    # Get function signature
    sig = inspect.signature(func)
    param_names = list(sig.parameters.keys())
    
    # Remove first parameter (x or t)
    if len(param_names) > 0:
        param_names = param_names[1:]
    
    # Remove last parameter if it's 'spectrum' (background functions)
    if len(param_names) > 0 and param_names[-1] == 'spectrum':
        param_names = param_names[:-1]
    #
    return param_names

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
    
    Examples
    --------
    Create a simple 1D energy-resolved model:
    
    >>> model = Model('Au4f_doublet')
    >>> model.energy = np.linspace(80, 95, 150)
    >>> model.time = np.array([0])  # Single time point
    >>> 
    >>> # Add components (typically done via File.load_model from YAML)
    >>> c1 = Component('GLP')
    >>> c1.add_pars({'A': [20, True, 5, 30], ...})
    >>> model.add_components([c1, ...])
    >>> 
    >>> # Evaluate model
    >>> model.create_value1D(store1D=True)
    >>> model.plot_1D(plot_ind=True)  # Plot individual components
    
    Create a 2D time-and-energy-resolved model:
    
    >>> model = Model('Au4f_2D')
    >>> model.energy = np.linspace(80, 95, 150)
    >>> model.time = np.linspace(-100, 1000, 200)
    >>> 
    >>> # Add components and time-dependence
    >>> model.add_components([...])
    >>> t_model = Dynamics('GLP_01_x0')  # Time-dependence for peak position
    >>> model.add_dynamics(t_model)
    >>> 
    >>> # Evaluate 2D spectrum
    >>> model.create_value2D()
    >>> model.plot_2D()
    
    Notes
    -----
    **Model Construction:**
    Models are typically constructed via File.load_model() which reads
    component definitions from YAML files. Direct construction (as shown
    in examples) is useful for programmatic model building.
    
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
    def __init__(self, model_name='test'):
        self.name = model_name
        # file name of yaml file containing model details
        self.yaml_f_name = None
        # functions of spectral components of fit
        self.peak_fcts = []
        # list of objects of type defined in Component class
        self.components = []
        # flattened lmfit parameters list (1D with time- and energy-components)
        self.lmfit_par_list = [] # (individual lmfit.Parameter objects)
        # lmfit.Parameters object corresponding to lmfit_par_list attribute
        self.lmfit_pars = lmfit.Parameters()
        # list of all parameter names
        self.par_names = None
        # list of component spectra (from last evaluation/ current parameters)
        self.component_spectra = []
        # 1D spectrum (i.e. sum/ combination of all components)
        self.value1D = None
        # 2D spectrum (i.e. 1D spectra one per time step)
        self.value2D = None # self.value2D = np.empty((len(self.time), len(self.energy)))
        # fit parameters and results
        self.const = None
        self.args = None
        self.result = []
        # ATTRIBUTES THAT SHOULD BE INHERITED FROM A PARENT ENTITY WHEN LOADING MODEL
        self.parent_file = None  # parent reference (set by File when loading model)
        #self.data = None # (currently) not necessary
        self.dim = None
        self.energy = None # necessarry or should just point to file?
        self.time = None # necessarry or should just point to file?
        #
        return None
    
    @property
    def plot_config(self):
        """
        Get plot configuration from parent File.
        
        Models inherit plot settings from their parent File, ensuring
        consistent plotting across all models for the same dataset.
        
        Returns
        -------
        PlotConfig
            Configuration object with plot settings (axes, colors, DPI, etc.)
        
        Examples
        --------
        >>> model.plot_1D()  # Uses parent File's plot config
        >>> config = model.plot_config
        >>> config.x_dir = 'rev'  # Customize for this plot
        
        Notes
        -----
        If no parent File exists (standalone model), returns default PlotConfig.
        
        See Also
        --------
        PlotConfig : Plot configuration class
        File.plot_config : File-level configuration
        """
        if hasattr(self, 'parent_file') and self.parent_file is not None:
            return self.parent_file.plot_config
        
        # Fallback to defaults if no parent
        return PlotConfig()

    #  
    def describe(self, detail=0):
        """
        Display information about model structure and parameters.
        
        Parameters
        ----------
        detail : int, default=0
            Level of detail to display:
            - 0: Component list and parameters only
            - 1+: Also plot initial guess and (for 2D) data comparison
        
        Examples
        --------
        >>> model.describe()  # Basic info
        >>> model.describe(detail=1)  # With plots
        
        Notes
        -----
        Useful for:
        - Verifying model loaded correctly from YAML
        - Checking parameter bounds and initial values
        - Visualizing initial guess before fitting
        
        See Also
        --------
        Component.describe : Component-level information
        """
        # minimum model description
        print('model name: ' +self.name)
        try: [comp.describe(detail=detail-1) for comp in self.components]
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
    def add_components(self, comps_list, debug=False):
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
        
        Examples
        --------
        >>> c_offset = Component('Offset')
        >>> c_offset.add_pars({'y0': [2, True, 0, 5]})
        >>> c_peak = Component('GLP')
        >>> c_peak.add_pars({'A': [20, True, 5, 30], ...})
        >>> model.add_components([c_offset, c_peak])
        
        Notes
        -----
        **Component Naming:**
        Components are expected to have their names already set from YAML
        parsing (e.g., GLP_01, GLP_02, Offset, Shirley). The numbering is
        handled during YAML parsing, not here.
        
        **Parameter Naming:**
        Parameter names are constructed as: prefix + component_name + '_' + param_name
        - For Dynamics models: prefix = model.name (e.g., 'GLP_01_x0_')
        - For regular models: prefix = '' (e.g., 'GLP_01_A')
        
        **Component Preparation:**
        Each component receives:
        - energy/time axes from model
        - parent_model reference (for finding other parameters)
        - subcycle time axis (for multi-cycle Dynamics)
        - kernel time axis (for convolution components)
        
        **Model Updates:**
        After adding components, the model's lmfit_pars and par_names
        are automatically updated via self.update().
        
        See Also
        --------
        Component.create_pars : Create parameters for a component
        update : Update model parameter structures
        File.load_model : Typical way to populate components from YAML
        """
        # add list to <components> attribute 
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
    def find_par_by_name(self, par_name):
        """
        Find the component and parameter indices for a given parameter name.
        
        Searches through all components and their parameters to locate the
        indices needed to access a specific parameter by name.
        
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
        
        Examples
        --------
        >>> ci, pi = model.find_par_by_name('GLP_01_x0')
        >>> if ci is not None:
        ...     param = model.components[ci].pars[pi]
        ...     print(f"Found: {param.name} = {param.value()}")
        
        Notes
        -----
        Used internally when adding time-dependence to parameters via
        Model.add_dynamics(). The parameter name must match exactly,
        including component numbering and prefix.
        
        Returns (None, None) if parameter not found. A message is printed
        to console indicating the parameter wasn't found.
        
        See Also
        --------
        add_dynamics : Add time-dependence to a parameter
        Par.name : Parameter naming convention
        """
        done = False
        for ci, comp in enumerate(self.components):
            for pi, par in enumerate(comp.pars):
                if par.name == par_name:
                    done = True; break
            if done == True: break
        if done == False:
            print(f'parameter "{par_name}" not found in model {self.name}')
            ci = None; pi = None
        #
        return ci, pi
    
    #
    def print_all_pars(self, detail=0):
        """
        Print information on all parameters individually.
        
        Parameters
        ----------
        detail : int, default=0
            Detail level passed to Par.describe()
        
        Notes
        -----
        Debugging utility to inspect parameter structure and values.
        For routine parameter inspection, use model.describe() or
        model.lmfit_pars.pretty_print().
        
        See Also
        --------
        describe : More user-friendly model summary
        Par.describe : Parameter-level information
        """
        for c in self.components:
            for p in c.pars:
                p.describe(detail)
        #
        return None
    
    #
    def update(self, debug=False):
        """
        Update model from bottom up: parameters → components → model.
        
        Recompiles all parameters from all components and recreates the
        flattened lmfit parameter structures. Call this after modifying
        parameter structure (e.g., after add_components or add_dynamics).
        
        Parameters
        ----------
        debug : bool, default=False
            If True, print parameter list after update
        
        Notes
        -----
        Updates performed:
        - Recompiles lmfit_par_list from all components
        - Recreates lmfit_pars object
        - Updates par_names list
        
        This is automatically called by add_components() and add_dynamics(),
        so manual calling is rarely needed.
        
        See Also
        --------
        add_components : Adds components and calls update
        add_dynamics : Adds time-dependence and calls update
        Component.update_lmfit_par_list : Component-level update
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
    def update_value(self, new_par_values, par_select='all'):
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
        
        Examples
        --------
        >>> # Update all parameters (during fitting)
        >>> model.update_value([1.5, 84.2, 1.8, 0.3, ...])
        
        >>> # Update specific parameters only
        >>> model.update_value([84.2, 88.0], par_select=['GLP_01_x0', 'GLP_02_x0'])
        
        Notes
        -----
        Called by spectra.fit_model_mcp() on every iteration during fitting
        to update model parameters before evaluation.
        
        Does not trigger model re-evaluation; call create_value1D() or
        create_value2D() after updating values.
        
        See Also
        --------
        spectra.fit_model_mcp : Calls this during fitting
        create_value1D : Evaluate 1D model after update
        create_value2D : Evaluate 2D model after update
        """
        p_count = 0 # initialize counter for parameters in <par_select>
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
    def add_dynamics(self, dynamics_model, frequency=-1, debug=False):
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
        
        Examples
        --------
        >>> # Create 2D model
        >>> model = Model('Au4f_2D')
        >>> model.energy = np.linspace(80, 95, 150)
        >>> model.time = np.linspace(-100, 1000, 200)
        >>> # ... add components ...
        
        >>> # Create time-dependence for peak position
        >>> t_model = Dynamics('GLP_01_x0')
        >>> t_model.time = model.time
        >>> # ... add temporal components to t_model ...
        
        >>> # Attach to parameter
        >>> model.add_dynamics(t_model, frequency=10)  # 10 Hz repetition
        
        Notes
        -----
        **Parameter Naming:**
        The dynamics_model.name must exactly match a parameter name in the
        model. For example, if you have a GLP component numbered as GLP_01,
        and you want to make its x0 parameter time-dependent, the Dynamics
        model must be named 'GLP_01_x0'.
        
        **Dimensionality:**
        Adding dynamics does not change model.dim from 1 to 2. The parent
        must set model.dim = 2 explicitly (typically done by File class).
        
        **Expression Parameters:**
        If a parameter has an expression (depends on another parameter),
        and the referenced parameter becomes time-dependent, the expression
        will automatically track time-dependent values. However, adding
        dynamics directly to an expression parameter is not recommended.
        
        **Multi-Cycle Dynamics:**
        When frequency > 0, the dynamics repeat periodically. Subcycles
        within each period can be defined in the Dynamics model for complex
        multi-step processes (e.g., pump-probe-pump experiments).
        
        See Also
        --------
        Dynamics : Time-dependence model class
        Dynamics.set_frequency : Configure repetition frequency
        Par.update : Parameter update with Dynamics
        """
        # set the model instance calling this method as parent model for Dynamics 
        dynamics_model.parent_model = self
        
        if frequency != -1: # set a repetition frequency
            dynamics_model.set_frequency(frequency)
        
        # find component and parameter index from Dynamics model name
        ci, pi = self.find_par_by_name(dynamics_model.name)
        
        # add Dynamics model and update corresponding parameter
        self.components[ci].pars[pi].update(dynamics_model)
        # update model lmfit_par_list, par_names and components
        self.update(debug=debug)
        
        # Re-analyze all expressions since time-dependence status may have changed
        self._analyze_expression_dependencies()
        #
        return None
    
    #
    def _analyze_expression_dependencies(self):
        """
        Analyze all parameter expressions for time-dependent references.
        
        Checks if any parameter expressions reference time-dependent parameters.
        This is needed to handle cases like:
        - Par1 = 10 (time-dependent via Dynamics)
        - Par2 = Par1 + 5 (expression, should track Par1's time-dependence)
        
        Called automatically after add_dynamics() to update expression
        dependency tracking.
        
        Notes
        -----
        Internal method. Users typically don't need to call this directly.
        
        See Also
        --------
        Par.analyze_expression_dependencies : Parameter-level analysis
        add_dynamics : Triggers this analysis
        """
        # Get all parameters from all components
        all_parameters = self._get_all_parameters()
        # Analyze all expressions for time-dependent references
        for par in all_parameters:
            par.analyze_expression_dependencies(all_parameters)
        #
        return None
    
    #
    def _get_all_parameters(self):
        """
        Get all parameters from all components in this model.
        
        Returns
        -------
        list of Par
            All Par objects from all components
        
        Notes
        -----
        Used internally for expression analysis and parameter searches.
        
        See Also
        --------
        _analyze_expression_dependencies : Uses this to get parameter list
        """
        all_parameters = []
        for comp in self.components:
            all_parameters.extend(comp.pars)
        #
        return all_parameters
    
    #
    def combine(value, comp, t_ind=0):
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
        
        Notes
        -----
        **Component Types:**
        - 'add': Regular addition (peaks, lineshapes)
        - 'back': Background addition (requires existing spectrum)
        - 'conv': Convolution with kernel (IRF, broadening)
        - 'none': Placeholder, returns value unchanged
        
        **Order Matters:**
        Components are combined in reverse order (last to first) to handle
        backgrounds properly. Backgrounds need the sum of all peaks evaluated
        before them.
        
        **Subcycles:**
        Component subcycle handling (for multi-cycle dynamics) is performed
        in Component.value(), not here. This method just handles the
        combination logic.
        
        Examples
        --------
        Typical component order in YAML:
```yaml
        model:
          Offset:      # Applied last (added to everything)
          Shirley:     # Applied second-to-last
          GLP:         # Peak 1
          GLP:         # Peak 2
```
        
        Evaluation order (reverse):
        1. GLP_02 -> value
        2. GLP_01 -> value + GLP_02
        3. Shirley -> value + peaks + Shirley(peaks)
        4. Offset -> value + peaks + Shirley + Offset
        
        See Also
        --------
        create_value1D : Uses this to build 1D spectra
        Component.value : Component evaluation at time point
        uarr.my_conv : Convolution implementation
        """
        # skip 'none' type components entirely (no-op)
        if comp.comp_type == 'none':
            return value
        # add a component to existing spectrum
        if comp.comp_type == 'add':
            return value +comp.value(t_ind)
        # add a background component to exisiting spectrum
        elif comp.comp_type == 'back':
            return value +comp.value(t_ind, spectrum=value)
        # convolute component with existing spectrum 
        elif comp.comp_type == 'conv':
            if comp.package == fcts_energy:
                x_axis = comp.energy
                print('convolution of spectral components not defined')
            elif comp.package == fcts_time:
                x_axis = comp.time
            #
            return uarr.my_conv(x=x_axis, y=value, kernel=comp.value(t_ind))
   
    #
    def create_valueTEST(self, t_ind=0, store1D=0, return1D=0, debug=False):
        """
        Experimental version of create_value1D with efficiency improvements.
        
        **Note: This is a test/experimental method. Use create_value1D() for
        production code.**
        
        Attempts to optimize model evaluation by avoiding redundant component
        evaluations for time-independent components. However, the interaction
        with convolutions and backgrounds makes this optimization non-trivial.
        
        Parameters
        ----------
        t_ind : int, default=0
            Time index for evaluation
        store1D : int, default=0
            If 1, store individual component spectra
        return1D : int, default=0
            If 1, return the 1D spectrum
        debug : bool, default=False
            If True, print debug information
        
        Notes
        -----
        Current implementation is incomplete and may not correctly handle:
        - Time-independent components with convolution
        - Background functions with time-dependent spectrum
        - Subcycle interactions
        
        For reliable results, use create_value1D() instead.
        
        See Also
        --------
        create_value1D : Stable production version
        """
        # re-initialize list containing individual component spectra
        if store1D == 1: self.component_spectra = []
        # re-initialize value1D
        if isinstance(self, Dynamics):
            self.value1D = np.zeros(len(self.time))
        else:
            self.value1D = np.zeros(len(self.energy))
        
        # combine the components into a spectrum/ time dynamics curve
        for N in range(len(self.components)):
            if debug: print(N+1); print(self.components[-(N+1)].fct_str)
            if store1D == 1: current_spec = copy.deepcopy(self.value1D)
            #
            self.value1D = Model.combine(self.value1D, 
                                         self.components[-(N+1)],
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
    def create_value1D(self, t_ind=0, store1D=0, return1D=0, debug=False):
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
        
        Examples
        --------
        >>> # Evaluate model at time index 0
        >>> model.create_value1D(t_ind=0)
        >>> plt.plot(model.energy, model.value1D)
        
        >>> # Store individual components for plotting
        >>> model.create_value1D(t_ind=0, store1D=1)
        >>> for i, comp_spec in enumerate(model.component_spectra):
        ...     plt.plot(model.energy, comp_spec, label=f'Component {i}')
        
        >>> # Use return value directly
        >>> spectrum = model.create_value1D(t_ind=0, return1D=1)
        >>> residual = data - spectrum
        
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
        
        See Also
        --------
        create_value2D : Evaluate 2D model (calls this for each time point)
        Model.combine : Static method that combines components
        Component.value : Evaluate individual component
        plot_1D : Visualize 1D results
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
    def create_value2D(self, t_ind=[], debug=False):
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
        
        Examples
        --------
        >>> # Evaluate full 2D spectrum
        >>> model.create_value2D()
        >>> plt.pcolormesh(model.energy, model.time, model.value2D)
        
        >>> # Evaluate partial time range (e.g., for testing)
        >>> model.create_value2D(t_ind=[0, 50])  # First 50 time points
        
        >>> # After fitting, visualize result
        >>> model.create_value2D()
        >>> model.plot_2D()
        
        Notes
        -----
        **Performance:**
        Evaluation time scales linearly with:
        - Number of time points (len(self.time))
        - Number of energy points (len(self.energy))
        - Model complexity (number of components, time-dependent parameters)
        
        For large models, expect:
        - ~0.01-0.1 seconds per time point for typical models
        - Minutes for high-resolution 2D scans
        - Consider create_value2D_parallel() for multi-core speedup
        
        **Memory:**
        Result stored in self.value2D has shape (n_time, n_energy).
        For 1000 time points × 500 energy points × 8 bytes/float:
        ~4 MB per model evaluation.
        
        **Time-Dependence:**
        For each time point t_i:
        1. Time-dependent parameters evaluate their Dynamics at t_i
        2. Model components use these parameter values
        3. 1D spectrum computed and stored in value2D[t_i, :]
        
        This allows complex dynamics like:
        - Peak positions shifting over time
        - Amplitudes decaying exponentially
        - Widths evolving with multi-exponential kinetics
        
        See Also
        --------
        create_value1D : Evaluates spectrum at single time point
        create_value2D_parallel : Multi-threaded version for speedup
        plot_2D : Visualize 2D results
        """
        if len(t_ind) == 0: # process entire time axis
            self.value2D = np.empty((len(self.time), len(self.energy)))
            for ti, t in enumerate(self.time):
                self.value2D[ti,:] = self.create_value1D(t_ind=ti, return1D=1)
        else: # process selection according to t_ind
            self.value2D = np.empty((len(self.time[t_ind[0]:t_ind[1]]), len(self.energy)))
            for ti, t in enumerate(self.time[t_ind[0]:t_ind[1]]):
                self.value2D[ti,:] = self.create_value1D(t_ind=ti, return1D=1)
        #
        return None
    
    #
    def create_value2D_parallel(self, t_ind=[], debug=False):
        """
        Evaluate 2D model using parallel computation.
        
        **Experimental:** Multi-threaded version of create_value2D() that
        evaluates time points in parallel. May provide speedup for complex
        models on multi-core systems.
        
        Parameters
        ----------
        t_ind : list, optional
            Time index range to process (see create_value2D)
        debug : bool, default=False
            Debug output flag
        
        Notes
        -----
        **Status:** Experimental, needs testing and optimization.
        
        **Potential Issues:**
        - Thread safety with model parameter updates
        - GIL limitations for pure Python code
        - Memory overhead from multiple threads
        
        **When to Use:**
        Consider this if:
        - Model has many time-independent operations
        - System has multiple CPU cores available
        - Standard create_value2D() is bottleneck
        
        **When NOT to Use:**
        - Small models (overhead > benefit)
        - Memory-constrained systems
        - Untested production code
        
        For most cases, standard create_value2D() is recommended until
        this method is fully validated.
        
        See Also
        --------
        create_value2D : Standard serial version (recommended)
        """
        if len(t_ind) == 0: # process entire time axis
            n_times = len(self.time)
            self.value2D = np.empty((n_times, len(self.energy)))
            # Parallelize the 1D computations
            def compute_1d(ti):
                return self.create_value1D(t_ind=ti, return1D=1)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(executor.map(compute_1d, range(n_times)))
            for ti, val in enumerate(results):
                self.value2D[ti, :] = val
        else: # process selection according to t_ind
            t_start, t_stop = t_ind[0], t_ind[1]
            n_times = len(self.time[t_start:t_stop])
            self.value2D = np.empty((n_times, len(self.energy)))
            def compute_1d(ti):
                return self.create_value1D(t_ind=ti, return1D=1)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(executor.map(compute_1d, range(t_start, t_stop)))
            for ti, val in enumerate(results):
                self.value2D[ti, :] = val
        #
        return None

    #
    def plot_1D(self, t_ind=0, plot_ind=False, x_lim=None, y_lim=None, save_img=0, save_path=''):
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
        
        Examples
        --------
        >>> # Plot total spectrum
        >>> model.plot_1D(t_ind=0)
        
        >>> # Plot individual components with legend
        >>> model.plot_1D(t_ind=0, plot_ind=True)
        
        >>> # Zoom in on region of interest
        >>> model.plot_1D(t_ind=0, plot_ind=True, 
        ...               x_lim=(83, 89), y_lim=(0, 25))
        
        >>> # Save without displaying
        >>> model.plot_1D(t_ind=0, plot_ind=True,
        ...               save_img=-1, save_path='model_components.png')
        
        Notes
        -----
        **Automatic Configuration:**
        Plot settings are inherited from parent File via plot_config property.
        This ensures consistency with data plots and other model plots.
        
        **Component Colors:**
        When plot_ind=True, components are automatically assigned different
        colors from the default matplotlib color cycle.
        
        **Dynamics Models:**
        For Dynamics instances, x-axis is always time (regardless of x_dir
        setting) and shows temporal evolution of the parameter.
        
        See Also
        --------
        plot_2D : Plot 2D time-and-energy spectrum
        create_value1D : Must be called first with store1D=1 for plot_ind=True
        PlotConfig : Plot configuration system
        """
        # Get model config for plotting
        config = self.plot_config
        
        # Model calling this method is a ...
        # ... time-resolved model (mcp.Dynamics)
        if isinstance(self, Dynamics):
            x_dir = 'def'  # Always default for Dynamics
            x = self.time
            x_label = config.y_label  # t_label from project
            info = ''
        # ... energy-resolved model
        else:
            x_dir = config.x_dir
            x = self.energy
            x_label = config.x_label  # e_label from project
            info = f'[{config.y_label}={round(self.time[t_ind],3)} (index={t_ind})]'
            
        # Populate <component_spectra> argument of the model
        self.create_value1D(t_ind, store1D=1)
        
        # Plot
        uplt.plot_1D(
            data=self.component_spectra if plot_ind else [self.value1D,],
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
    def plot_2D(self, save_img=0, save_path='', x_lim=None, y_lim=None, z_lim=None):
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
        
        Examples
        --------
        >>> # Plot full 2D spectrum
        >>> model.create_value2D()
        >>> model.plot_2D()
        
        >>> # Focus on specific time/energy region
        >>> model.plot_2D(x_lim=(83, 89), y_lim=(0, 500))
        
        >>> # Set color scale limits
        >>> model.plot_2D(z_lim=(0, 20))
        
        >>> # Save high-res figure for publication
        >>> model.plot_2D(save_img=-1, save_path='model_2D_final.png')
        
        Notes
        -----
        **Model Evaluation:**
        You must call model.create_value2D() before plotting. The plot shows
        the content of model.value2D, which is only populated by create_value2D().
        
        **Configuration:**
        Plot appearance (colormap, axis direction, labels) inherited from
        parent File via plot_config property.
        
        **Performance:**
        For large 2D arrays (1000+ time points), rendering may take a few
        seconds. Consider downsampling for quick visualization during
        model development.
        
        See Also
        --------
        plot_1D : Plot 1D spectrum or time slice
        create_value2D : Must be called first to populate value2D
        PlotConfig : Plot configuration system
        """
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
    def par_distribution_brood_force_test(self, comp_ind, par_ns, A_dist, x_dist):
        """
        Test parameter distribution effects via brute-force sampling.
        
        **Experimental:** Explores how parameter distributions (e.g., from
        ensemble averaging or inhomogeneous samples) affect the resulting
        spectrum by evaluating multiple parameter combinations.
        
        Parameters
        ----------
        comp_ind : int
            Component index to test
        par_ns : list of str
            Parameter names to distribute (e.g., ['A', 'x0'])
        A_dist : list of float
            Distribution of amplitude values to test
        x_dist : list of float
            Distribution of position values to test
        
        Returns
        -------
        out_lst : list of ndarray
            Individual spectra for each parameter combination
        out_sum : ndarray
            Sum of all spectra (ensemble average)
        
        Examples
        --------
        >>> # Test effect of position distribution
        >>> A_vals = [10, 9, 8, 7, 6]  # Amplitude distribution
        >>> x0_vals = [85.5, 85.3, 85.1, 84.9, 84.7]  # Position distribution
        >>> spectra, avg = model.par_distribution_brood_force_test(
        ...     comp_ind=0, par_ns=['A', 'x0'], A_dist=A_vals, x_dist=x0_vals)
        >>> 
        >>> # Compare individual and average
        >>> for spec in spectra:
        ...     plt.plot(model.energy, spec, alpha=0.3)
        >>> plt.plot(model.energy, avg, 'k-', linewidth=2, label='Average')
        
        Notes
        -----
        **Use Cases:**
        - Modeling inhomogeneous broadening
        - Ensemble averaging effects
        - Parameter uncertainty impact
        - Sample heterogeneity effects
        
        **Limitations:**
        - Only handles two parameters currently
        - No proper distribution weighting
        - Computationally expensive for many samples
        
        **Future Work:**
        This could be expanded into a proper distribution system using
        trspecfit.functions.distribution module for realistic distributions
        (Gaussian, Lorentzian, etc.) with proper weighting.
        
        See Also
        --------
        trspecfit.functions.distribution : Distribution functions (underdeveloped)
        """
        # initialize
        out = np.zeros(len(self.energy))
        out_lst = []
        out_sum = np.zeros(len(self.energy))
        
        # select component from model as defined by user (index) input
        #c = self.components[comp_ind]
        
        # create sum over all components with A and x0 values passed by user
        for Atemp, xtemp in zip(A_dist, x_dist):
            # update parameters in model
            self.update_value(new_par_values = [Atemp, xtemp],
                              par_select = par_ns)
            display(self.lmfit_pars) # debug
            # add to list of all component versions
            out_lst.append(Model.combine(out,
                                         self.components[comp_ind]))
            # add to summation spectrum
            out_sum += out_lst[-1]
        #
        return out_lst, out_sum
    
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
    package : module, default=fcts_energy
        Python module containing the component function (fcts_energy or fcts_time)
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
    
    Properties
    ----------
    fct : callable
        Function object (auto-updates if package or fct_str changes)
    fct_specs : inspect.FullArgSpec
        Function signature information
    fct_args : list of str
        Function argument names
    prefix : str
        Prefix for parameter names ('' for exceptions, comp_name+'_' otherwise)
    name : str
        Component display name
    
    Examples
    --------
    Create a peak component manually:
    
    >>> comp = Component('GLP_01', package=fcts_energy)
    >>> comp.add_pars({
    ...     'A': [20, True, 5, 30],
    ...     'x0': [84.5, True, 82, 88],
    ...     'F': [1.6, True, 1, 2.5],
    ...     'm': [0.3, False, 0, 1]
    ... })
    >>> comp.energy = np.linspace(80, 95, 150)
    >>> comp.create_pars()
    
    Create a time dynamics component:
    
    >>> comp = Component('expFun_01', package=fcts_time)
    >>> comp.add_pars({
    ...     'A': [2, True, 0, 5],
    ...     'tau': [1000, True, 100, 5000],
    ...     't0': [0, False, 0, 1],
    ...     'y0': [0, False, 0, 1]
    ... })
    >>> comp.time = np.linspace(-100, 1000, 200)
    >>> comp.create_pars()
    
    Notes
    -----
    **Component Types:**
    Type is determined automatically from function name:
    - Background functions (Offset, Shirley, etc.) -> 'back'
    - Functions ending in 'CONV' -> 'conv'
    - 'none' function -> 'none'
    - All others -> 'add'
    
    **Parameter Naming:**
    Parameters are named with prefix to avoid conflicts in multi-component models:
    - Exception functions (backgrounds, convolutions): no prefix
    - Regular functions: comp_name + '_' + param_name
    
    **Subcycles:**
    For multi-cycle Dynamics models (e.g., pump-probe-pump), components can
    be assigned to specific subcycles. The time_N_sub mask controls when the
    component is active.
    
    **Convolution:**
    Convolution components receive a special kernel time axis (via
    create_t_kernel) that is wider than the main time axis to properly
    handle edge effects.
    
    See Also
    --------
    Model : Container for components
    Par : Parameter management
    create_t_kernel : Kernel axis construction
    trspecfit.functions.energy : Available energy/spectral functions
    trspecfit.functions.time : Available time/dynamics functions
    """
    #
    def __init__(self, comp_name, package=fcts_energy, comp_subcycle=0):
        # package containing component (either fcts_energy or fcts_time)
        self.package = package
        # name of the component (str)
        self.comp_name = comp_name
        # parse the component name into function string and component number
        self.fct_str, self.N = parse_component_name(comp_name)       
        # determine component type: 'add', 'conv', 'back', or 'none'
        if self.fct_str in background_functions():
            self.comp_type = 'back'
        elif self.fct_str == 'none': # placeholder function (see src/functions/time.py)
            self.comp_type = 'none'
        elif 'CONV' in self.fct_str or self.fct_str.endswith('CONV'):
            self.comp_type = 'conv'
        else:
            self.comp_type = 'add'
        # dict of par_name: par_info from yaml file passed by user
        self.par_dict = {}
        # (for self.package=fcts_time) which subcycle is this component part of 
        self.subcycle = comp_subcycle # see "t-dynamics.normalize_time" for more details
        self.time_N_sub = None # 1 where model subcycle equals component subcycle
        self.time_norm = None # restarts at zero for every subcycle
        # list of Par objects needed to construct component
        self.pars = [] # used to create component value during fit
        # flattened list of all lmfit parameters defining this component
        self.lmfit_par_list = []
        # time and energy axis of component are inherited from model
        self.time = None
        self.energy = None
        # parent model reference
        self.parent_model = None
        #
        return None
    
    # [automatic] create self.fct attribute that will update if either
    # self.package or self.fct_str changes [attribute is read only]
    @property 
    def fct(self):
        """
        Function object for this component.
        
        Automatically retrieves function from package using fct_str.
        Updates dynamically if package or fct_str changes.
        
        Returns
        -------
        callable
            Function object (e.g., fcts_energy.GLP, fcts_time.expFun)
        """
        return getattr(self.package, self.fct_str)
    
    # [automatic] do the same for function argument specs
    @property 
    def fct_specs(self):
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
    def fct_args(self):
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
    def prefix(self):
        """
        Prefix for parameter names.
        
        Returns
        -------
        str
            Parameter prefix:
            - '': For exception functions (backgrounds, convolutions, none)
            - comp_name + '_': For regular components
        
        Examples
        --------
        >>> comp = Component('GLP_01')
        >>> comp.prefix
        'GLP_01_'
        >>> comp = Component('Offset')
        >>> comp.prefix
        ''
        """
        # define prefix for parameter names
        if self.fct_str in prefix_exceptions():
            return ''
        else: # number the components starting from N=1
            return self.comp_name +'_'
        
    # [automatic] create a name for this component
    @property 
    def name(self):
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
    def add_pars(self, par_info_dict):
        """
        Add parameter specifications to component.
        
        Stores parameter information that will be used to create Par objects
        when create_pars() is called. Typically populated from YAML model
        definitions.
        
        Parameters
        ----------
        par_info_dict : dict
            Parameter specifications: {name: [value, vary, min, max]} or
            {name: ['expression']} for constrained parameters
        
        Examples
        --------
        >>> comp = Component('GLP_01')
        >>> comp.add_pars({
        ...     'A': [20, True, 5, 30],
        ...     'x0': [84.5, True, 82, 88],
        ...     'F': [1.6, True, 1, 2.5],
        ...     'm': [0.3, False, 0, 1]
        ... })
        
        >>> # Parameter with expression
        >>> comp.add_pars({
        ...     'x0': ['GLP_01_x0 + 3.6']  # Depends on another parameter
        ... })
        
        Notes
        -----
        Does not create the actual Par objects yet - that happens in
        create_pars(). This separation allows parameters to be defined
        before axes are known.
        
        See Also
        --------
        create_pars : Actually create Par objects from this dictionary
        """
        self.par_dict = par_info_dict
        #
        return None
    
    #
    def create_pars(self, prefix='', debug=False):
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
        
        Examples
        --------
        >>> comp = Component('GLP_01')
        >>> comp.add_pars({'A': [20, True, 5, 30], ...})
        >>> comp.energy = np.linspace(80, 95, 150)
        >>> comp.create_pars()
        >>> print([p.name for p in comp.pars])
        ['GLP_01_A', 'GLP_01_x0', 'GLP_01_F', 'GLP_01_m']
        
        Notes
        -----
        **Two-Pass Creation:**
        1. First pass: Create all Par objects with values/bounds
        2. Second pass: Set expressions (so forward references work)
        
        This ensures expressions like 'GLP_02_A' work even when GLP_02
        is defined after GLP_01 in the component list.
        
        **Parent References:**
        Each Par receives a parent_model reference for expression evaluation
        and time-dependence tracking.
        
        **Axes:**
        Each Par inherits the time axis from the component for use in
        Dynamics models.
        
        See Also
        --------
        add_pars : Define parameters before calling this
        Par.create : Individual parameter creation
        """
        lst = [] # initialize pars list
        if len(prefix) > 0:
            prefix += '_'
        # First pass: create all Par objects, but do not set expressions
        expr_params = []
        for p_name, p_info in self.par_dict.items():
            temp = Par(name=prefix + self.prefix + p_name)
            temp.info = p_info  # see Par class for details
            # Set parent model reference
            temp.parent_model = self.parent_model
            # If this is an expression, skip setting it for now
            if len(p_info) == 1 and isinstance(p_info[0], str):
                expr_params.append((temp, p_info[0]))
                # Temporarily set a dummy value (needed for lmfit creation)      
                temp.create(expr_skip=True)
            else:
                temp.create()
            temp.time = self.time  # inherit time axis from component
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
    def update_lmfit_par_list(self):
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
        
        See Also
        --------
        Model.update : Calls this for all components
        Par.lmfit_par_list : Individual parameter lists
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
    def describe(self, detail=1):
        """
        Print component information.
        
        Parameters
        ----------
        detail : int, default=1
            Detail level:
            - 0: Function name only
            - 1+: Function name, type, subcycle, and parameters
        
        Examples
        --------
        >>> comp.describe(detail=0)
        function: GLP from <module 'trspecfit.functions.energy'>
        
        >>> comp.describe(detail=1)
        function: GLP from <module 'trspecfit.functions.energy'>
        function will be added to other components [for all times t]
        GLP_01_A: [20, True, 5, 30]
        GLP_01_x0: [84.5, True, 82, 88]
        ...
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
    def create_t_kernel(self, debug=False):
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
        
        Examples
        --------
        >>> comp = Component('gaussCONV', package=fcts_time)
        >>> comp.add_pars({'SD': [80, True, 0, 1000]})
        >>> comp.time = np.linspace(-100, 1000, 200)
        >>> t_kernel = comp.create_t_kernel()
        >>> print(f"Kernel spans {t_kernel[0]:.1f} to {t_kernel[-1]:.1f}")
        Kernel spans -320.0 to 320.0
        
        Notes
        -----
        **Kernel Width:**
        Each convolution function has an associated width function (e.g.,
        gaussCONV_kernel_width()) that returns a multiplier. The kernel axis
        extends to ±(first_parameter × multiplier).
        
        **Time Step:**
        Uses the same time step as the main time axis for consistent
        convolution behavior.
        
        **Function Names:**
        Only works with functions ending in 'CONV' that have corresponding
        kernel_width functions in fcts_time module.
        
        See Also
        --------
        trspecfit.functions.time : Convolution functions and kernel widths
        Model.combine : Uses kernel for convolution operation
        """
        # get kernel parameters i.e. component parameters
        parK = ulmfit.par_extract(self.par_dict, return_type='list')
        if debug: print(f'component/kernel parameters as list: {parK}')
        # define kernel time axis
        kernel_width = getattr(fcts_time, self.fct_str +'_kernel_width')()
        if debug: print(f'kernel width loaded from fcts_time: {kernel_width}')
        t_range = parK[0] *kernel_width
        try: t_step = self.time[1] -self.time[0]
        except: print(f'time axis of component {self.fct_str} not defined')
        if debug: print(f'delta time (from self.time): {t_step}')
        t_kernel = np.arange(-t_range, t_range+t_step, t_step)
        #
        return t_kernel

    #
    def value(self, t_ind=0, **kwargs):
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
        
        Examples
        --------
        >>> # Evaluate energy-resolved component
        >>> comp.energy = np.linspace(80, 95, 150)
        >>> spec = comp.value(t_ind=0)
        >>> plt.plot(comp.energy, spec)
        
        >>> # Evaluate time-resolved component
        >>> comp.time = np.linspace(-100, 1000, 200)
        >>> dynamics = comp.value(t_ind=0)
        >>> plt.plot(comp.time, dynamics)
        
        >>> # Background component (needs existing spectrum)
        >>> shirley = back_comp.value(t_ind=0, spectrum=peak_sum)
        
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
        
        See Also
        --------
        Par.value : Parameter evaluation at time point
        Model.combine : Combines component values into spectrum
        """
        # get component parameters as list
        pars = []
        for p in self.pars:
            #$% slightly hacky to only update Par.t_model for t_ind=0, change?
            pars.extend(p.value(t_ind, 
                                update_t_model = True if t_ind==0 else False))
                  
        # get x axis and create component function evaluation
        if self.package == fcts_energy:
            return self.fct(self.energy, *pars, **kwargs)
        elif self.package == fcts_time:
            if self.subcycle == 0: # single cycle
                return self.fct(self.time, *pars, **kwargs)
            else: # multi-cycle
                # multpliy value with 1 where subcycle applies, 0 otherwise
                # use normalized time instead of standard time for sub!=0]
                return self.fct(self.time_norm, *pars, **kwargs) *self.time_N_sub
    
    #
    def plot(self, t_ind=0, **kwargs):
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
        
        Examples
        --------
        >>> comp.energy = np.linspace(80, 95, 150)
        >>> comp.plot()  # Shows component shape
        
        >>> # Check time-dynamics
        >>> t_comp.time = np.linspace(-100, 1000, 200)
        >>> t_comp.plot()  # Shows temporal evolution
        
        Notes
        -----
        Plots use default styling and don't inherit File plot configuration.
        For publication-quality plots, evaluate component and use uplt.plot_1D
        with proper configuration.
        
        See Also
        --------
        value : Underlying evaluation method
        uplt.plot_1D : More configurable plotting
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
    
    Examples
    --------
    Create a standard parameter:
    
    >>> par = Par('GLP_01_A', [20, True, 5, 30])
    >>> par.create()
    >>> print(par.value())
    20
    
    Create an expression parameter:
    
    >>> par = Par('GLP_02_A', ['3/4 * GLP_01_A'])
    >>> par.create()
    >>> # Value will be computed from GLP_01_A when evaluated
    
    Add time-dependence:
    
    >>> par = Par('GLP_01_x0', [84.5, True, 82, 88])
    >>> par.create()
    >>> t_model = Dynamics('GLP_01_x0')
    >>> # ... configure t_model ...
    >>> par.update(t_model)
    >>> par.t_vary
    True
    
    Notes
    -----
    **Time-Dependence:**
    When t_vary=True, the parameter value at time t is:
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
    
    See Also
    --------
    Component : Contains Par objects
    Dynamics : Time-dependence model
    Model.add_dynamics : Attach Dynamics to parameter
    """
    #
    def __init__(self, name, info=[]):
        self.name = name
        self.info = info
        self.lmfit_par = lmfit.Parameters()
        self.t_vary = False
        self.t_model = Model(f'{name}_tModel')
        self.lmfit_par_list = []
        # Expression analysis attributes
        self.expr_refs_time_dep = False  # flag for time-dependent references
        self.expr_string = None          # store original expression
        self.expr_refs = []              # list of referenced parameter names
        self.parent_model = None         # reference to parent model
        #
        return None
    
    #
    def describe(self, detail=0):
        """
        Print parameter information.
        
        Parameters
        ----------
        detail : int, default=0
            Detail level for display (passed to t_model.describe if applicable)
        
        Examples
        --------
        >>> par.describe()
        par name: GLP_01_A [value: 20] and its <lmfit_par> attribute:
        Name      Value      Min      Max   Stderr     Vary     Expr Brute_Step
        GLP_01_A      20        5       30     None     True     None      None
        parameter has no time dependence
        """
        print(f'par name: {self.name} [value: {self.value()}] and its <lmfit_par> attribute:')
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
                display(self.t_model.describe())
        #
        return None
        
    #
    def create(self, prefix='', suffix='', expr_skip=False, debug=False):
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
        
        See Also
        --------
        Component.create_pars : Two-pass creation for expressions
        ulmfit.par_create : Underlying parameter creation
        """
        # create standard lmfit parameter (spectral component)
        if expr_skip and len(self.info) == 1 and isinstance(self.info[0], str):
            # if skipping expression, use a dummy value for now
            lmfit_par = ulmfit.par_create(self.name, [0, True, -np.inf, np.inf], prefix, suffix, debug)
        else:
            lmfit_par = ulmfit.par_create(self.name, self.info, prefix, suffix, debug)
        # add to lmfit_par attribute
        self.lmfit_par.add_many(lmfit_par)
        # and list of individual lmfit paramters
        self.lmfit_par_list.extend([lmfit_par])    
        #
        return None
    
    #
    def update(self, t_model):
        """
        Add time-dependence to parameter via Dynamics model.
        
        Converts a static parameter into a time-dependent one by attaching
        a Dynamics model that describes temporal evolution.
        
        Parameters
        ----------
        t_model : Dynamics
            Dynamics model describing time evolution
        
        Examples
        --------
        >>> par = Par('GLP_01_x0', [84.5, True, 82, 88])
        >>> par.create()
        >>> 
        >>> t_model = Dynamics('GLP_01_x0')
        >>> t_model.time = np.linspace(-100, 1000, 200)
        >>> # ... add components to t_model ...
        >>> 
        >>> par.update(t_model)
        >>> print(par.t_vary)
        True
        >>> print(len(par.lmfit_par_list))  # Now includes temporal parameters
        5  # (1 base + 4 from dynamics components, for example)
        
        Notes
        -----
        After calling update:
        - t_vary flag set to True
        - t_model attribute stores the Dynamics model
        - lmfit_par_list extended with temporal parameters
        - value() method will return time-dependent values
        
        See Also
        --------
        Model.add_dynamics : High-level method that calls this
        Dynamics : Time-dependence model class
        value : Evaluation with time-dependence
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
    def value(self, t_ind=0, update_t_model=True):
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
        
        Examples
        --------
        >>> # Time-independent parameter
        >>> par = Par('GLP_01_A', [20, True, 5, 30])
        >>> par.create()
        >>> par.value()
        20
        
        >>> # Time-dependent parameter
        >>> par.t_vary = True
        >>> par.t_model.value1D = np.array([0, 1, 2, 3, ...])  # Dynamics result
        >>> par.value(t_ind=10)
        20 + par.t_model.value1D[10]  # Base + dynamics
        
        >>> # Expression parameter referencing time-dependent parameter
        >>> # Automatically tracks referenced parameter's time-dependence
        
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
        
        See Also
        --------
        update : Add time-dependence to parameter
        _evaluate_time_dependent_expression : Expression evaluation
        """
        if self.t_vary == False:
            if self.expr_refs_time_dep:
                # Custom evaluation for time-dependent expressions
                all_parameters = self._get_all_parameters()
                return self._evaluate_time_dependent_expression(t_ind, all_parameters, update_t_model)
            else:
                # Standard lmfit evaluation
                value = ulmfit.par_extract(self.lmfit_par)

        elif self.t_vary == True:
            if update_t_model == True:
                self.t_model.create_value1D() # update t_model, specifically self.t_model.value1D
            value = ulmfit.par_extract(self.lmfit_par) + self.t_model.value1D[t_ind]
            
        else:
            value = -1
            print(f't_vary attribute of Par "{self.name}" is not valid')
        #
        return value
    
    #
    def analyze_expression_dependencies(self, all_parameters):
        """
        Analyze expression for time-dependent parameter references.
        
        Checks if this parameter's expression references any time-dependent
        parameters. If so, sets expr_refs_time_dep flag so value() can
        handle dynamic expression evaluation.
        
        Parameters
        ----------
        all_parameters : list of Par
            All parameters in parent model (to check time-dependence)
        
        Notes
        -----
        Called automatically by Model._analyze_expression_dependencies after
        adding time-dependence to any parameter.
        
        Users don't typically call this directly.
        
        See Also
        --------
        Model._analyze_expression_dependencies : Triggers this analysis
        _evaluate_time_dependent_expression : Uses the analysis results
        """
        if len(self.info) == 1 and isinstance(self.info[0], str):
            self.expr_string = self.info[0]
            # Parse expression to find referenced parameters
            self.expr_refs = self._extract_parameter_references(self.expr_string)
            
            # Check if any referenced parameters are time-dependent
            for ref_name in self.expr_refs:
                ref_par = self._find_parameter_by_name(ref_name, all_parameters)
                if ref_par and ref_par.t_vary:
                    self.expr_refs_time_dep = True
                    break
        #
        return None
    
    #
    def _extract_parameter_references(self, expr_string):
        """
        Extract parameter names referenced in expression.
        
        Parses expression string to find parameter names, which are identified
        as strings starting with known function names (since parameter naming
        follows function_name_NN_paramname pattern).
        
        Parameters
        ----------
        expr_string : str
            Expression to parse (e.g., "GLP_01_A * 0.75 + GLP_02_x0")
        
        Returns
        -------
        list of str
            Parameter names found in expression
        
        Notes
        -----
        Uses pattern matching to find valid parameter names. A valid reference
        must start with a known function name from the energy functions list.
        
        See Also
        --------
        analyze_expression_dependencies : Uses this to find references
        """        
        # Pattern to match parameter names (letters, numbers, underscores, but not starting with number)
        pattern = r'\b[A-Za-z_][A-Za-z0-9_]*\b'
        matches = re.findall(pattern, expr_string)
        
        # Filter to keep only strings that start with known function names
        # This catches parameter names like GLP_01_A, GLP_02_x0, etc.
        parameter_refs = []
        for match in matches:
            #$% if a mcp.Dynamics Par references another mcp.Dynamics Par this doesn't work yet!
            for func_name in set(energy_functions()):
                if match.startswith(func_name + '_'):
                    parameter_refs.append(match)
                    break  # Found a match, no need to check other function names
        
        return parameter_refs
    
    #
    def _find_parameter_by_name(self, par_name, all_parameters):
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
    def _evaluate_time_dependent_expression(self, t_ind, all_parameters, update_t_model=True):
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
        # Create namespace with current values of referenced parameters
        namespace = {}
        for ref_name in self.expr_refs:
            ref_par = self._find_parameter_by_name(ref_name, all_parameters)
            if ref_par:
                namespace[ref_name] = ref_par.value(t_ind, update_t_model=update_t_model)
        
        # Evaluate expression using asteval (safe, same as lmfit uses)
        try:
            aeval = Interpreter()
            # populate the interpreter symbol table with current parameter values
            # (Interpreter.__call__ doesn't accept a namespace argument)
            for k, v in namespace.items():
                aeval.symtable[k] = v
            return aeval(self.expr_string)
        except Exception as e:
            raise ValueError(f"Error evaluating expression '{self.expr_string}': {e}")
    
    #
    def _get_all_parameters(self):
        """
        Get all parameters from parent model.
        
        Returns
        -------
        list of Par
            All parameters in parent model
        
        Notes
        -----
        Used for expression evaluation to find referenced parameters.
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
    
    Examples
    --------
    Simple exponential decay:
    
    >>> t_model = Dynamics('GLP_01_x0')
    >>> t_model.time = np.linspace(-100, 1000, 200)
    >>> 
    >>> c_exp = Component('expFun', fcts_time)
    >>> c_exp.add_pars({
    ...     'A': [2, True, 0, 5],
    ...     'tau': [500, True, 100, 2000],
    ...     't0': [0, False, 0, 1],
    ...     'y0': [0, False, 0, 1]
    ... })
    >>> t_model.add_components([c_exp])
    
    With instrumental response function:
    
    >>> t_model = Dynamics('GLP_01_x0')
    >>> t_model.time = np.linspace(-100, 1000, 200)
    >>> 
    >>> c_irf = Component('gaussCONV', fcts_time)
    >>> c_irf.add_pars({'SD': [50, True, 10, 200]})
    >>> 
    >>> c_exp = Component('expFun', fcts_time)
    >>> c_exp.add_pars({...})
    >>> 
    >>> t_model.add_components([c_irf, c_exp])
    
    Multi-cycle with subcycles (pump-probe-pump):
    
    >>> t_model = Dynamics('GLP_01_x0')
    >>> t_model.time = np.linspace(0, 1000, 1000)  # Multiple cycles
    >>> t_model.subcycles = 2  # Two subcycles per main cycle
    >>> 
    >>> # Load from YAML with two submodels
    >>> file.load_model('dynamics.yaml', ['model_sub1', 'model_sub2'], 
    ...                 par_name='GLP_01_x0')
    >>> file.add_time_dependence('dynamics.yaml', ['model_sub1', 'model_sub2'],
    ...                          par_name='GLP_01_x0', frequency=10)
    
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
    
    See Also
    --------
    Model.add_dynamics : Attach Dynamics to parameter
    Par.update : Parameter-level attachment
    trspecfit.functions.time : Available dynamics functions
    normalize_time : Time axis normalization for multi-cycle
    """
    #
    def __init__(self, model_name):
        super().__init__(model_name)     
        # repetition frequency of time-dependent model behaviour
        self.frequency = -1
        # number of subcycles (within time = 1/frequency)
        self.subcycles = 0 # defined via "number of models -1" [model_info in file.load_model()]
        # "normalized time" attributes (all have same length as time axis)
        self.time_norm = None # restart time (at 0) every 1/(freqency *number of subcycles)
        self.N_sub = None # which subcycle is active at time step (t_i)
        self.N_counter = None # cummulative counter of subcycles (at t_i)
        #
        return None
    
    #
    def set_frequency(self, frequency, time_unit=0):
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
        
        Examples
        --------
        >>> t_model = Dynamics('GLP_01_x0')
        >>> t_model.time = np.linspace(0, 1, 1000)  # 0-1 second
        >>> t_model.subcycles = 2
        >>> t_model.set_frequency(10)  # 10 Hz = 0.1 s period
        >>> # Now time_norm resets every 0.05 s (half period for 2 subcycles)
        
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
        
        See Also
        --------
        normalize_time : Computes normalized time arrays
        Component.value : Uses time_N_sub and time_norm during evaluation
        """
        self.frequency = frequency # set the frequency attribute itslef
        self.normalize_time() # update the normalization of the time axis
        # update components accordingly
        for comp in self.components:
            # <time_N_sub> is 0/1 where subcomponent is in-/active
            comp.time_N_sub[self.N_sub != comp.subcycle] = 0
            # inherit normalized time from Dynamics model
            if comp.subcycle != 0: comp.time_norm = self.time_norm
            # no updating needed on the parameter level as time irrelevant
            # parameter level uses index to refer to par.t_model=Dynamics
        # 
        return None

    #
    def normalize_time(self, time_unit=0, debug=False):
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
        
        See Also
        --------
        set_frequency : High-level method that calls this
        Component.value : Uses normalized time for subcycle components
        """
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