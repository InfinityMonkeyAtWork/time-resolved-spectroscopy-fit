"""
1D/2D Spectroscopy Fitting Module
"""

from trspecfit import mcp
from trspecfit import fitlib
from trspecfit import spectra
from trspecfit.utils import lmfit as ulmfit
from trspecfit.utils import arrays as uarr
from trspecfit.utils import plot as uplt
import numpy as np
import os # replace os.join with "pathlib path /"subfolder" /"file name"
import pathlib
import copy
import inspect
import time
from IPython.display import display
# function library for energy, time, and distribution components
from trspecfit.functions import energy as fcts_energy
from trspecfit.functions import time as fcts_time
from trspecfit.functions import distribution as fcts_dist
#ruaml.yaml mod (instead of "import yaml"):
from ruamel.yaml import YAML
from ruamel.yaml.constructor import SafeConstructor
from ruamel.yaml.error import YAMLError
# yaml parser needs to know which components to number
from trspecfit.config.functions import prefix_exceptions
# standardized plotting configuration
from trspecfit.config.plot import PlotConfig

# what does show_info mean? convert to binary debug by True if show_info >=3 else False

# multi-subcycle models allow for convolution only in the "0th subcycle" i.e. first model_info element
# which affects all times t. "conv" functions in individual subcycles are currently ignored

#
def get_available_function_names():
    """
    Dynamically discover all available function names from the functions modules.
    Returns a set of all function names that can be used as components.
    """
    function_names = set()
    
    # Get all function names from each module
    for module in [fcts_energy, fcts_time, fcts_dist]:
        for name in dir(module):
            # Only include callable functions (not constants or classes)
            if callable(getattr(module, name)) and not name.startswith('_'):
                function_names.add(name)
    
    return function_names

# yaml is by default read as dictionary which creates problems with non-unique keys
# this function enables multiple components of the same type with automatic numbering
# by reading the yaml as a list of tuples, numbering, then converting to a dictionary
def construct_yaml_map(self, node):
    """
    Enable multiple components of the same type with automatic numbering.
    All components get numbered starting from _01: GLP -> GLP_01, GLP_02, etc.
    Exceptions(functions that don't get numbered): background, convolutions, offset.
    """
    data = []
    yield data

    # Get all available function names
    available_functions = get_available_function_names()
    # Get exceptions (functions that don't get numbered)
    exceptions = prefix_exceptions()
    
    # Track component names to handle duplicates
    component_counts = {}
    
    for key_node, value_node in node.value:
        key = self.construct_object(key_node, deep=True)
        val = self.construct_object(value_node, deep=True)
        
        # Check if this key is a component name (function name)
        if isinstance(key, str) and key in available_functions:
            # Check if this is an exception (background/offset function)
            if key in exceptions:
                # Don't number exceptions, just use the original name
                data.append((key, val))
            else:
                # This is a regular component, always number it
                if key in component_counts:
                    component_counts[key] += 1
                else:
                    component_counts[key] = 1
                
                numbered_key = f"{key}_{component_counts[key]:02d}"
                data.append((numbered_key, val))
        else:
            # This is a model name or other key, don't number it
            data.append((key, val))

SafeConstructor.add_constructor(u'tag:yaml.org,2002:map', construct_yaml_map)
yaml = YAML(typ='safe')

#
#
class ModelValidationError(ValueError):
    """Exception raised for errors in model YAML validation."""
    # see _validate_model_components()
    pass

#
#
class Project:
    """
    Project configuration and management.
    """
    def __init__(self, path, name='test', config_file='project.yaml'):
        self.path = pathlib.Path(path) if path is not None else pathlib.Path('test')
        self.path_results = pathlib.Path(f'{path}_fits')
        self.run = name
        self.path_run = self.path_results / name
        
        # Set defaults first
        self._set_defaults()
        
        # Override with YAML config if provided
        if config_file is not None:
            self._load_config(config_file)
    
    #
    def _set_defaults(self):
        """Set default project configuration."""
        self.show_info = 1
        # Plot settings
        self.e_label = 'Energy'
        self.t_label = 'Time'
        self.z_label = 'Intensity'
        self.x_dir = 'def'
        self.x_type = 'lin'
        self.y_dir = 'def'
        self.y_type = 'lin'
        self.z_colormap = 'viridis'
        self.z_colorbar = 'ver'
        self.z_type = 'lin'
        self.dpi_plt = 100
        self.dpi_save = 300
        self.res_mult = 5
        # File I/O settings
        self.ext = '.dat'
        self.fmt = '%.6e'
        self.delim = ','
        self.DA_fmt = '%04d'
        self.DA_slices_fmt = "%06d"
        # Advanced settings
        self.spec_lib = spectra
        self.spec_fun_str = 'fit_model_mcp'
        self.skip_first_N_spec = -1
        self.first_N_spec_only = -1
    
    @property
    def spec_fun(self):
        """
        Dynamically get the spectrum fitting function.
        """
        return getattr(self.spec_lib, self.spec_fun_str)
    
    #
    def _load_config(self, config_file):
        """
        Load project configuration from YAML file.
        Allow different users or types of spectroscopy to overwrite (a subset of) project attributes
        
        Parameters
        ----------
        config_file : str or Path
            Name or path of config file (looks in self.path)
        """
        from ruamel.yaml import YAML
        
        yaml = YAML()  # Standard YAML loading
        config_path = self.path / config_file
        
        try:
            with open(config_path, 'r') as f:
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
    Load data, energy, and time axis into a file object to easily fit different models
    compare the fit results, etc.
    
    # str(self.p.DA_fmt%self.nr)
    """
    def __init__(self, parent_project=None, path='test', data=None, energy=None, time=None):
        # pass parent project or (default) create a functioning test project environment 
        self.p = parent_project if parent_project is not None else Project(path=None)
        self.path = path # path to load/save [?] data from
        self.path_DA = self.p.path_run / path # path to save fit results to
        self._plot_config = None # create plot config from project (but File can customize it)
        #
        self.data = data # (time-[optional] and) energy-dependent data to fit
        self.dim = len(np.shape(data)) # 1 for energy (1D) # 2 for energy+time (2D)
        # take energy and time input or create a generic axis if None is passed
        self.energy = energy if (energy is not None or data is None) else np.arange(0,(np.shape(data)[1]))
        self.time = time if (time is not None or self.dim<=1) else np.arange(0,(np.shape(data)[0]))
        # keep track of models that are used to fit this file/data
        self.models = [] # list for now, could do @property, setter, getter
        self.model_active = None # default model to work with [no need to pass same name again and again]
        # [i.e. let user define models via file class attributes]
        self.e_lim_abs = [] # fitting: energy limits (low, high) user-defined
        self.e_lim = [] # index of energy limits (from left, from right: energy[left:-right])
        self.t_lim_abs = [] # fitting: time limits (low, high) user-defined
        self.t_lim = [] # index of time limits (left to right: time[left:right])
        #
        self.base_t_abs = [] # start and stop time of the baseline spectrum
        self.base_t_ind = [] # index of the above start and stop time
        self.data_base = None # average spectrum between above indices 
        self.model_base = None
        #
        self.model_SbS = None
        self.results_SbS = [] # all Slice-by-Slice fit results (different from model_SbS.result)
        #
        return None
    
    @property
    def plot_config(self):
        """
        Get plot config for this File.
        
        Created from parent Project on first access. File can then customize
        persistently (e.g., for different time axes across files).
        """
        if self._plot_config is None:
            self._plot_config = PlotConfig.from_project(self.p)
        #
        return self._plot_config
    
    @plot_config.setter
    def plot_config(self, config):
        """Allow setting a custom config for this File"""
        self._plot_config = config

    #
    def describe(self):
        """
        Display info about file
        """
        print(f"File # x [path: {self.path}]")
        
        config = self.plot_config
        
        if self.dim == 1:
            uplt.plot_1D(
                data=[self.data,],
                x=self.energy,
                config=config,
                vlines=self.e_lim_abs
            )
            
        elif self.dim == 2:
            uplt.plot_2D(
                data=self.data,
                x=self.energy,
                y=self.time,
                config=config,
                vlines=self.e_lim_abs,
                hlines=self.t_lim_abs
            )
        #
        return None
   
    #
    def model_list_to_name(self, model_list):
        """
        Create model name for mcp.Dynamics models with more than one submodel
        Join individual model names with underscores between them
        For lists with one element this function returns the name of the element
        """
        #
        return '_'.join(model_list) # see str.join()
    
    #
    def select_model(self, model_info, return_type='model'):
        """
        Select model by name [type(model_info)=str] or position [type(model_info)=int]
        Returns model (<return_type>='model', default) or 
                index of model in File.models (<return_type>='index')
        Returns None if model name not found or index out of range
        
        For time-dependence/ dynamics models with more than one model i.e. submodels:
        pass the list containing all model names (same input as in "load_model")
        """
        #
        if isinstance(model_info, str):
            for m_i, m in enumerate(self.models):
                if m.name == model_info:
                    if return_type == 'model':
                        return self.models[m_i] 
                    elif return_type == 'index':
                        return m_i
            return None # no match found
        #
        elif isinstance(model_info, int):
            if model_info not in range(len(self.models)):
                return None # no match found
            else:
                if return_type == 'model':
                    return self.models[model_info] 
                elif return_type == 'index':
                    return model_info
        #
        elif isinstance(model_info, list):
            m_name = self.model_list_to_name(model_info)
            for m_i, m in enumerate(self.models):
                if m.name == m_name:
                    if return_type == 'model':
                        return self.models[m_i]
                    elif return_type == 'index':
                        return m_i
            return None # no match found
    
    #
    def set_active_model(self, model_info):
        """
        All functions requiring a model input will default to the currently active model unless
        a model is specified as input to the respective function (via <model_info>)
        """
        self.model_active = self.select_model(model_info)

    #
    def _validate_model_components(self, model_info_dict, model_info, model_yaml_path):
        """
        Validate model components and parameters for common errors.
        
        Checks for:
        - Invalid component names (not in available functions)
        - Invalid parameter attributes (not value/vary/min/max or expr)
        - Invalid parameter structure
        - Missing required parameter attributes
        - Invalid bounds (min > max, value outside bounds)
        - Wrong parameter names for component type
        """
        available_functions = get_available_function_names()
        #valid_param_keys = {'value', 'vary', 'min', 'max', 'expr'}
        example_dir = pathlib.Path(__file__).parent.parent / "examples"
        
        # Only validate models that are being loaded
        for model_name in model_info:
            if model_name not in model_info_dict:
                continue
                
            components = model_info_dict[model_name]
            
            for comp_name, params in components.items():
                # Extract base component name (remove _01, _02 suffixes)
                base_comp_name, _ = mcp.parse_component_name(comp_name)
                
                # Check 1: Component type exists
                if base_comp_name not in available_functions:
                    raise ModelValidationError(
                        f"Unknown component type '{base_comp_name}' in model '{model_name}' in {model_yaml_path}\n"
                        f"Available components: {sorted(available_functions)}\n"
                        f"Check for typos in component name."
                    )
                
                # Check 2: Parameters should be a dictionary
                if not isinstance(params, dict):
                    raise ModelValidationError(
                        f"Parameters for '{comp_name}' in model '{model_name}' must be a dictionary.\n"
                        f"Found: {type(params).__name__}\n"
                        f"See 'models_energy.yaml' in example directory: {example_dir}"
                    )
                
                # Get expected parameters for this component type
                expected_params = mcp.get_component_parameters(base_comp_name)

                # Check parameter count matches
                if len(params) != len(expected_params):
                    raise ModelValidationError(
                        f"Component '{comp_name}' (type: {base_comp_name}) in model '{model_name}' has wrong number of parameters.\n"
                        f"Expected {len(expected_params)} parameters: {expected_params}\n"
                        f"Got {len(params)} parameters: {list(params.keys())}\n"
                        f"Check {model_yaml_path}"
                    )
                
                # Check 3: Validate each parameter
                for param_name, param_value in params.items():
                    
                    # Check if parameter name is valid for this component
                    if param_name not in expected_params:
                        raise ModelValidationError(
                            f"Invalid parameter '{param_name}' for component '{comp_name}' (type: {base_comp_name}) in model '{model_name}'.\n"
                            f"Expected parameters: {expected_params}\n"
                            f"Check for typos or wrong component type."
                        )
                    
                    # Parameter value can be a list [value, vary, min, max] or [value, vary]
                    # or a single expression string
                    if isinstance(param_value, list):
                        
                        if (len(param_value) == 4) or (len(param_value) == 2):
                            if len(param_value) == 4: # Standard format: [value, vary, min, max]
                                value, vary, min_val, max_val = param_value
                            elif len(param_value) == 2: # Unbound format: [value, vary]
                                value, vary = param_value
                                min_val = -np.inf
                                max_val = np.inf

                            # Check that 'vary' is boolean
                            if not isinstance(vary, bool):
                                raise ModelValidationError(
                                    f"Parameter '{param_name}' in '{comp_name}' (model '{model_name}'):\n"
                                    f"'vary' (2nd element) must be True or False.\n"
                                    f"Got: {vary} ({type(vary).__name__})"
                                )
                                                    
                            # Check bounds validity
                            if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                                if min_val > max_val:
                                    raise ModelValidationError(
                                        f"Parameter '{param_name}' in '{comp_name}' (model '{model_name}'):\n"
                                        f"min ({min_val}) is greater than max ({max_val})"
                                    )
                                
                                # Check if value is within bounds
                                if isinstance(value, (int, float)):
                                    if value < min_val or value > max_val:
                                        raise ModelValidationError(
                                            f"Parameter '{param_name}' in '{comp_name}' (model '{model_name}'):\n"
                                            f"value ({value}) is outside bounds [{min_val}, {max_val}]"
                                        )
                        
                        elif len(param_value) == 1:
                            # Expression format: ["expression"]
                            if not isinstance(param_value[0], str):
                                raise ModelValidationError(
                                    f"Parameter '{param_name}' in '{comp_name}' (model '{model_name}'):\n"
                                    f"Single-element list must contain a string expression.\n"
                                    f"Got: {param_value[0]} ({type(param_value[0]).__name__})\n"
                                    f'Example: ["GLP_01_x0 + 3.6"]'
                                )
                        else:
                            raise ModelValidationError(
                                f"Parameter '{param_name}' in '{comp_name}' (model '{model_name}') has invalid format.\n"
                                f'Expected: [value, vary, min, max] or [value, vary] or ["expr"]\n'
                                f"Got: {param_value} ({len(param_value)} elements)\n"
                                f"See 'models_energy.yaml' in example directory: {example_dir}"
                            )
                    
                    else:
                        raise ModelValidationError(
                            f"Parameter '{param_name}' in '{comp_name}' (model '{model_name}') has invalid format.\n"
                            f"Expected either:\n"
                            f"  - [value, vary, min, max] for standard parameters\n"
                            f"  - [value, vary] for unbound parameters\n"
                            f"  - ['expression'] for linked parameters\n"
                            f"Got: {param_value}\n"
                            f"See 'models_energy.yaml' in example directory: {example_dir}"
                        )

    #
    def _load_and_number_yaml_components(self, model_yaml, model_info, par_name='', debug=False):
        """
        Load YAML file and apply appropriate component numbering strategy.
        For energy models: use component numbering from construct_yaml_map
        For dynamics models with subcycles: number components globally 
        """
        if debug:
            print(f'"model.yaml" file: {model_yaml}')
        model_yaml_path = self.p.path / model_yaml
        
        try:
            with open(model_yaml_path) as f_yaml:
                model_info_ALL = yaml.load(f_yaml)
                
                # Convert YAML structure to dictionary format
                if isinstance(model_info_ALL, list):
                    model_info_dict = {}
                    for model_entry in model_info_ALL:
                        if not (isinstance(model_entry, tuple) and len(model_entry) == 2):
                            raise ValueError(f"Malformed model entry: {model_entry}")
                        model_name, components = model_entry
                        if model_name in model_info_dict:
                            raise ValueError(f"Duplicate model name found: '{model_name}'")
                        # Convert components to dict format
                        model_info_dict[model_name] = dict(components) if isinstance(components, list) else components
                        # Convert parameters to dict format
                        for comp_name, params in model_info_dict[model_name].items():
                            if isinstance(params, list):
                                model_info_dict[model_name][comp_name] = dict(params)
                else: # should never happen unless something is wrong with construct_yaml_map
                    raise ValueError(f"Unexpected YAML structure in {model_yaml_path}")
                
                if debug:
                    print('model_info_ALL:')
                    print(model_info_ALL)
                    print('model_info_dict:')
                    print(model_info_dict)
                
                # Apply appropriate numbering strategy
                if par_name != '':
                    # This is a dynamics model - resolve numbering conflicts across subcycles
                    model_info_dict = self._resolve_dynamics_numbering_conflicts(model_info_dict, model_info, debug)

                # Validate the loaded model structure
                self._validate_model_components(model_info_dict, model_info, model_yaml_path)
                
                # For energy models, numbering is already complete from construct_yaml_map
                return model_info_dict
                
        except FileNotFoundError:
            raise FileNotFoundError(
                f"FileNotFound: <model_yaml> file input\n"
                f"File should be located in: {self.p.path}\n"
                f"Check file name for typos: {model_yaml_path}"
            )
        except ModelValidationError:
            # Validator errors are already user-friendly, just pass through
            raise
        except ValueError as e:
            # Structural errors (malformed entries, duplicates)
            if "Malformed model entry" in str(e) or "Duplicate model name" in str(e):
                raise
            # Unexpected YAML parsing error
            raise ValueError(
                f"Unexpected error parsing {model_yaml_path}\n"
                f"Original error: {e}\n\n"
                f"This may be a bug in the YAML parser.\n"
                f"Please report this error at: https://github.com/InfinityMonkeyAtWork/time-resolved-spectroscopy-fit/issues\n"
                f"Include your YAML file and this error message. Thank you!"
            ) from e
        except YAMLError as exc:
            raise RuntimeError(
                f"YAML syntax error in {model_yaml_path}\n"
                f"Please check for:\n"
                f"  - Proper indentation (use spaces, not tabs)\n"
                f"  - Matching brackets and quotes\n"
                f"  - Valid YAML syntax\n"
                f"Original error: {exc}"
            )

    #
    def _resolve_dynamics_numbering_conflicts(self, model_info_dict, model_info, debug=False):
        """
        Resolve numbering conflicts for dynamics models by tracking used numbers globally
        and reassigning conflicting numbers to the next available number.
        
        This preserves the existing YAML numbering where possible and only changes
        numbers when there are conflicts across subcycles.
        """
        if debug:
            print("=== STARTING CONFLICT RESOLUTION ===")
            print(f"model_info: {model_info}")
            print(f"\nmodel_info_dict BEFORE resolution:")
            for submodel, comps in model_info_dict.items():
                if submodel in model_info:
                    print(f"  {submodel}: {list(comps.keys())}")

        # Get all available function names and exceptions
        available_functions = get_available_function_names()
        exceptions = prefix_exceptions()
        
        # Track the next available number for each function type globally
        global_next_available = {}
        # Track all used numbers for each function type
        used_numbers = {}
        

        # First pass: collect all existing numbers and find conflicts
        for submodel in model_info:
            if submodel not in model_info_dict:
                continue
                
            for comp_name, comp_params in model_info_dict[submodel].items():
                base_name, number = mcp.parse_component_name(comp_name)
                
                if base_name in available_functions and base_name not in exceptions:
                    if number == -1:
                        number = 1  # Default numbering
                        
                    # Track used numbers
                    if base_name not in used_numbers:
                        used_numbers[base_name] = set()
                        global_next_available[base_name] = 1
                    
                    used_numbers[base_name].add(number)
                    global_next_available[base_name] = max(global_next_available[base_name], number + 1)
        
        if debug:
            print(f"\nAfter first pass - used_numbers: {used_numbers}")
            print(f"global_next_available: {global_next_available}")

        # Second pass: resolve conflicts by reassigning duplicate numbers
        processed_dict = {}
        assigned_numbers = {}  # Track what we've already assigned in this pass
        
        for submodel in model_info:
            if submodel not in model_info_dict:
                continue
                
            processed_dict[submodel] = {}
            

            for comp_name, comp_params in model_info_dict[submodel].items():
                base_name, current_number = mcp.parse_component_name(comp_name)
                
                if base_name in available_functions and base_name not in exceptions:
                    if current_number == -1:
                        current_number = 1  # Default numbering
                
                    # Initialize tracking for this base name
                    if base_name not in assigned_numbers:
                        assigned_numbers[base_name] = set()
                    
                    # Check if this number is already assigned in this dynamics model
                    if current_number in assigned_numbers[base_name]:
                        # Conflict! Find next available number
                        while global_next_available[base_name] in assigned_numbers[base_name]:
                            global_next_available[base_name] += 1
                        new_number = global_next_available[base_name]
                        global_next_available[base_name] += 1
                        
                        if debug:
                            print(f"Conflict resolved: {comp_name} -> {base_name}_{new_number:02d} in {submodel}")
                    else:
                        # No conflict, use current number
                        new_number = current_number
                    
                    # Mark this number as assigned
                    assigned_numbers[base_name].add(new_number)
                    
                    # Create the final component name
                    final_name = f"{base_name}_{new_number:02d}"
                    processed_dict[submodel][final_name] = comp_params
                    
                else:
                    # Not a component function, keep as-is
                    processed_dict[submodel][comp_name] = comp_params

            if debug:
                print(f"\nProcessed submodel: {submodel}")
                print(f"  {submodel}: {list(processed_dict[submodel].keys())}")
        
        if debug:
            print(f"\nFINAL processed_dict:")
            for submodel in model_info:
                if submodel in processed_dict:
                    print(f"  {submodel}: {list(processed_dict[submodel].keys())}")
        #
        return processed_dict

    #
    def load_model(self, model_yaml, model_info, par_name='', debug=False):
        """
        Loads a model defined in <model_yaml> file located in Parent.path based on <model_info>:
        1) for energy-dependent models (1D or 2D) pass a list with one element, i.e. model name
        (<par_name>: pass an empty string [default]); model will be set as active model
        2) for a time-dependent model pass (that describes time-dependence of a parameter)
        pass a list with one element to load one model passed (e.g. for pump-probe data)
        lists with >1 elements: zero-th element is a model applied to the entire time axis and 
        following elements will be applied to respective subcycle only (see "eChem_example.ipynb")
        (<par_name> has to match the name of the 2D model parameter whose time-dependance is 
        described by the model being loaded; model will be returned
        """
        # sanity checks
        if not isinstance(model_info, list):
            raise TypeError(
                "model_info must be a list.\n"
                "Usage:\n"
                "  [name_model1,] for energy-dependent models\n"
                "  [name_model1, name_model2 (optional), ...] for time-dependent models"
            )
        if par_name == '' and len(model_info) != 1:
            raise ValueError(
                'Energy-resolved data (par_name="") require a single model name in model_info.\n'
                'Pass model name as the only element in the model_info list.\n'
                'OR pass a non-empty par_name to define mcp.Dynamics model with one or more model names.'
            )
        if self.select_model(model_info) is not None:
            raise ValueError(
                f'Model with name "{self.model_list_to_name(model_info)}" already exists. '
                'Delete the existing model or change the name of the new model.'
            )

        # Load and process YAML file with appropriate numbering strategy
        model_info_dict = self._load_and_number_yaml_components(model_yaml, model_info, par_name, debug)

        # initialize model
        if par_name == '':
            if self.p.show_info >= 1:
                print(
                    f'Loading model to describe energy- (and time-)dependent data: '
                    f'{self.model_list_to_name(model_info)}'
                )
            fcts_package = fcts_energy
            loaded_model = mcp.Model(self.model_list_to_name(model_info))
        else:
            if self.p.show_info >= 1:
                print(
                    f'Loading model to describe time-dependence of a model parameter: '
                    f'{par_name} of {self.model_list_to_name(model_info)} model'
                )
            fcts_package = fcts_time
            loaded_model = mcp.Dynamics(par_name)
        
        # inherit necessary model attributes from function input, file, and project
        loaded_model.yaml_f_name = model_yaml.split(".")[0] # yaml file name
        loaded_model.dim = 1 # start with 1, +1 when adding dynamics
        loaded_model.subcycles = len(model_info)-1
        loaded_model.energy = self.energy #$% remove redundancy?
        loaded_model.time = self.time #$% remove redundancy?
        
        all_comps = [] # initialize component list
        
        # go through (sub)model(s)
        # (for mcp.Dynamics model instances length model_info could be larger than 1)
        for subcycle, submodel in enumerate(model_info):
            # get the section defined by model_info
            try:
                submodel_info = model_info_dict[submodel]
            except KeyError:
                available_models = list(model_info_dict.keys())
                raise ValueError(
                    f'Model "{submodel}" not found in {model_yaml}\n'
                    f"Available models in this file: {available_models}\n"
                    f"Check for typos in model name."
                )
                    
            # Create components for this submodel using existing mcp.Component logic
            for c_name, c_info in submodel_info.items():
                c_temp = mcp.Component(c_name, fcts_package, subcycle)
                c_temp.add_pars(c_info)
                all_comps.append(c_temp)
                
        # add all components (and their parameters) to model
        loaded_model.add_components(all_comps)
        
        # add model to file
        if not isinstance(loaded_model, mcp.Dynamics):
            self.models.append(loaded_model)
            self.set_active_model(model_info) # set as current active model
            return None
        else:
            return loaded_model
            
    #
    def describe_model(self, model_info=None, detail=0):
        """
        Describe model selected via <model_info> (str or int or None[currently active model])
        <detail> =0: show parameters and parameter info
                 =1: show par&info and data/initial guess/residual
        """
        if model_info is None:
            mod = self.model_active
            #model_info = FIND NAME
        else:
            mod = self.select_model(model_info)
        
        # parameter list
        mod.describe(detail=0)
            
        if detail == 1 and isinstance(mod, mcp.Dynamics):
            mod.create_value1D(store1D=1) # update individual component spectra
            mod.plot_1D(plot_ind=True) # plot guess only (individual components)
        
        if detail == 1 and mod.dim == 1:
            mod.create_value1D(store1D=1) # update individual component spectra
            # plot initial guess (individual components), data, and residual
            title_mod = f'File: {self.path}, ' +\
                        f'Model: "{model_info}" (from "{mod.yaml_f_name}.yaml")' +\
                        f': initial guess'
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
                legend=[comp.name for comp in mod.components]
                )
            
        if detail == 1 and mod.dim == 2:
            mod.create_value2D() # update spectrum
            # plot data, fit, and residual 2D maps
            fitlib.plt_fit_res_2D(
                data=self.data,
                fit=mod.value2D,
                x=self.energy,
                y=self.time,
                config=self.plot_config,
                x_lim=self.e_lim,
                y_lim=self.t_lim
                )
        #
        return None
    
    #
    def delete_model(self, model_to_delete=None):
        """
        Delete a model from all models list of this file
        <model_to_delete>: model name (str) or
                           index (int) or 
                           None (deleting active model)
        """
        if model_to_delete is None:
            mod_index_del = self.models.index(self.model_active) # list.index(value)
            
        elif isinstance(model_to_delete, str):
            mod_index_del = self.select_model(model_to_delete, return_type='index')
            if mod_index_del is None:
                print(f'<delete_model>: Model with name {model_to_delete} not found')
                return None
            
        elif isinstance(model_to_delete, int):
            mod_index_del = copy.deepcopy(model_to_delete)
            if mod_index_del is None:
                print('<delete_model>: Model index out of range')
                return None
            
        else:
            print(f'<delete_model>: input type {type(model_to_delete)} not supported')
        
        # delete model from list using index: File.models[index]
        self.models.pop(mod_index_del)
        #
        return None
    
    #
    def reset_models(self):
        """
        Delete all models associated with this file
        """
        self.models = []
        #
        return None
    
    #
    def create_model_path(self, model_name, subfolders=[]):
        """
        Define (and create) path where model fit results will be saved to
        """
        mod = self.select_model(model_name) # get model
        path_model = self.path_DA / mod.yaml_f_name / model_name
        #path_model = self.path_DA / self.model_base.yaml_f_name / model_name
        if self.p.show_info >= 3: print(path_model)
        if not os.path.exists(path_model):
            os.makedirs(path_model)
        if len(subfolders) != 0:
            for subfolder in subfolders:
                if not os.path.exists(path_model / subfolder):
                    os.makedirs(path_model / subfolder) 
                    if self.p.show_info >= 3: print(path_model / subfolder)
        #
        return path_model
    
    #
    def define_baseline(self, time_start, time_stop, time_type='abs', show_plot=True):
        """
        Define a ground state/ pre-trigger/ baseline or other reference spectrum
        by passing a start and stop point in time.
        2D data will be cut and averaged to produce the baseline spectrum
        
        <time_type> is either "abs" for absolute time stamps or "ind" for index
        """
        if self.dim == 1:
            print("ERROR. Can not define baseline for 1D data")
            return None
            
        if time_type == 'abs':
            t_ind_start = np.searchsorted(self.time, time_start)
            t_ind_stop = np.searchsorted(self.time, time_stop)
        elif time_type == 'ind':
            t_ind_start = time_start
            t_ind_stop = time_stop
        self.base_t_ind = [t_ind_start, t_ind_stop]
        self.base_t_abs = [self.time[t_ind_start], self.time[t_ind_stop]]
        
        # cut and average
        self.data_base = np.mean(self.data[self.base_t_ind[0] : self.base_t_ind[1], :], axis=0)
        
        # plot
        if show_plot:
            uplt.plot_1D(
                data=[self.data_base,],
                x=self.energy,
                config=self.plot_config,
                title=f"Baseline data: t in {self.base_t_abs} (index: {self.base_t_ind})"
            )
        #
        return None
    
    #
    def set_fit_limits(self, energy_limits, time_limits=None, show_plot=True):
        """
        Set energy (and time) limits for fits (and show data with limits)
        Pass absolute values (NOT index)
        <energy_limits> = None will be converted to [np.min(energy), np.max(energy)]
        """
        if energy_limits is None:
            energy_limits = [np.min(self.energy), np.max(self.energy)]
        self.e_lim_abs = [np.min(energy_limits), np.max(energy_limits)]
        
        # convert energy and time limits to index values
        if self.p.x_dir == 'rev':
            E_ind_min = np.searchsorted(self.energy[::-1], np.min(energy_limits))
            E_ind_max = np.searchsorted(self.energy[::-1], np.max(energy_limits))
        else:
            E_ind_min = np.searchsorted(self.energy, np.min(energy_limits))
            E_ind_max = np.searchsorted(self.energy, np.max(energy_limits))
        self.e_lim = [np.shape(self.energy)[0] -E_ind_max, E_ind_min] # "min:-max"
        
        if time_limits is not None:
            self.t_lim_abs = time_limits
            t_ind_min = np.searchsorted(self.time, np.min(time_limits))
            t_ind_max = np.searchsorted(self.time, np.max(time_limits))
            self.t_lim = [t_ind_min, t_ind_max] # "min:max"
        
        if show_plot: # show data with limits
            if self.dim == 1:
                x_cut = self.energy[self.e_lim[0]:-self.e_lim[1]]              
                y_cut = self.data[self.e_lim[0]:-self.e_lim[1]]
                uplt.plot_1D(
                    data=[self.data, y_cut],
                    x=[self.energy, x_cut],
                    config=self.plot_config,
                    waterfall=(np.max(abs(y_cut))-np.min(abs(y_cut)))/8,
                    legend=['all', 'cut'],
                    vlines=self.e_lim_abs
                )
            elif self.dim == 2:
                uplt.plot_2D(
                    data=self.data,
                    x=self.energy,
                    y=self.time,
                    config=self.plot_config,
                    vlines=self.e_lim_abs,
                    hlines=self.t_lim_abs
                )
        #
        return None

    #
    def fit_baseline(self, model_name, fit, **lmfit_wrapper_kwargs):
        """
        Fit the baseline/ground state/pre-trigger or similar reference spectrum
        <model_name> is the name of a loaded model (use File.load_model)
        """
        t_base = time.time() # start timing for baseline fit
        
        # find model with matching name from list
        self.model_base = self.select_model(model_info=model_name)
        
        # get initial guess
        initial_guess = ulmfit.par_extract(self.model_base.lmfit_pars, return_type='list')
        # define (and create) path where basline fit results will be saved to
        path_base_results = self.create_model_path(model_name)
        
        # const = (x, data, package, function string, unpack, energy limits, time limits)
        self.model_base.const = (self.energy, self.data_base, self.p.spec_lib, self.p.spec_fun_str, 0, self.e_lim, [])
        # args [for fit function called in residual function]
        # model, dimension (dim =1 for baseline and SbS, =2 for 2D (global) fit), debug
        self.model_base.args = (self.model_base, 1, False)
        # fit (optionally) with confidence intervals
        self.model_base.result = fitlib.fit_wrapper(const=self.model_base.const,
                                                    args=self.model_base.args,
                                                    par_names=self.model_base.par_names,
                                                    par=self.model_base.lmfit_pars,
                                                    fit_type=fit,
                                                    show_info=1 if self.p.show_info>=2 else 0,
                                                    save_output=1,
                                                    save_path=path_base_results / model_name,
                                                    **lmfit_wrapper_kwargs)
        
        # display/plot and save baseline fit summary
        #self.model_base.create_value1D(store1D=1) # update individual component spectra
        title_base = f'File: {self.path}, ' +\
                     f'Model: "{model_name}" (from "{self.model_base.yaml_f_name}.yaml")'
        
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
            save_img=-1 if self.p.show_info<1 else 1,
            save_path=path_base_results / 'base_fit.png'
        )

        if fit >= 1:
            fitlib.time_display(t_start=t_base, print_str='Time elapsed for baseline fit: ')
            display(self.model_base.result[1].params) # display the final parameters below figure
        #
        return None
    
    #
    def load_fit(self):
        """
        Do this instead of refitting to try out different models?
        Probably needed to compare fits anyway!
        """
        # 
        return None
    
    #
    def fit_SliceBySlice(self, model_name, fit, **fit_wrapper_kwargs):
        """
        Fit time- and energy-resolved spectrum Slice-by-Slice (SbS) i.e. treat every time step as
        independent from other times [requires fitting the baseline first (fit_baseline)]
        <model_name> refers to the name of a model previously loaded using File.load_model()
        
        Note:
        Currently the energy position guesses (x0) are shifted on a per slice basis according to
        the position in energy (x) of the maximum value of the spectrum (NOT always a good idea!)
        """
        t_SbS = time.time() # start timing for SbS fit
        
        # find model with matching name from list
        self.model_SbS = self.select_model(model_info=model_name)
        
        # define (and create) path where SbS fit results will be saved to
        path_SbS_results = self.create_model_path(model_name, subfolders=['slices',])
                
        # set all fixed SbS fit parameters equal to baseline model results
        base_df = ulmfit.par2df(self.model_base.lmfit_pars, col_type='min')
        self.model_SbS.update_value(new_par_values=list(base_df['value']), par_select='all')
        
        # find all parameters with names ending in "x0" so they can be updated for every slice
        e_pos_pars = [name for name in self.model_SbS.par_names if name.endswith('_x0')]
        # find their corresponding values
        e_pos_vals = uarr.get_item(base_df, row=['name', e_pos_pars], col='value', astype='series')
        
        # cycle through all spectra and fit them
        self.results_SbS = [] # (re-)initialize placeholder for results
        for s_i, s in enumerate(self.data):
            if self.p.show_info < 3: print(f'Analyzing slice number {s_i+1}/{len(self.time)}', end='\r')
            if s_i < self.p.skip_first_N_spec: continue # skip past baseline spectra for debugging
            if self.p.show_info >= 3: print(); print('Spectrum #' +str(s_i)) # print iteration info
            # define path for files saved for this slice 
            path_slice = os.path.join(path_SbS_results, 'slices', str(self.p.DA_slices_fmt%s_i))

            # update the "x0" peak energy guess(es) using "max(baseline) -(max current slice)"
            deltaMAX = self.energy[np.argmax(s)] -self.energy[np.argmax(self.data_base)] # in eV
            if self.p.show_info >= 3: print(f'deltaMAX (spectrum with respect to baseline: {deltaMAX}')
            # update all guesses for parameters with names ending in "x0"
            new_e_vals = list(e_pos_vals.add(deltaMAX))
            self.model_SbS.update_value(new_par_values=new_e_vals, par_select=e_pos_pars)
            # get initial guess
            initial_guess = ulmfit.par_extract(self.model_SbS.lmfit_pars, return_type='list')
        
            # const = (x, data, package, function string, unpack, energy limits, time limits)
            self.model_SbS.const = (self.energy, s, self.p.spec_lib, self.p.spec_fun_str, 0, self.e_lim, [])
            # args (lmfit2D.Model, dimension, debug) [for fit function called in residual function]
            self.model_SbS.args = (self.model_SbS, 1, False)

            # fit with confidence intervals
            result_SbS = fitlib.fit_wrapper(const=self.model_SbS.const,
                                            args=self.model_SbS.args,
                                            par_names=self.model_SbS.par_names,
                                            par=self.model_SbS.lmfit_pars,
                                            fit_type=fit,
                                            show_info=1 if self.p.show_info>=3 else 0,
                                            save_output=1,
                                            save_path=path_slice,
                                            **fit_wrapper_kwargs)

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
                save_img=-1 if self.p.show_info<3 else 1,
                save_path=path_slice+'.png'
            )
            #
            if s_i == self.p.first_N_spec_only: break # for debugging: only fit first N spectra
        
        if fit >= 1:
            self.save_SliceBySlice_fit(save_path=path_SbS_results)
            fitlib.time_display(t_start=t_SbS, print_str='Time elapsed for Slice-by-Slice fit: ')
        #
        return None
    
    #
    def save_SliceBySlice_fit(self, save_path):
        """
        Save (additional) results from Slice-by-Slice model/component/parameter fit 
        """
        # convert results, specifically par_fin to dataframe and save
        # this also plots all parameters as a function of time
        df_SbS = fitlib.results2df(
            results=self.results_SbS, 
            x=self.time,
            index=np.arange(0, len(self.time)),
            config=self.plot_config,
            skip_first_N_spec=self.p.skip_first_N_spec, 
            first_N_spec_only=self.p.first_N_spec_only,
            save_df=-1 if self.p.show_info==0 else 1,
            save_path=save_path
        )
        
        if self.p.show_info >= 3: 
            display(df_SbS)
        
        # get slice-by-slice fit spectra as a 2D map
        fit2D_SbS = fitlib.results2fit2D(
            results=df_SbS[self.model_SbS.par_names],
            const=self.model_SbS.const, 
            args=self.model_SbS.args,
            save_2D=-1 if self.p.show_info==0 else 1,
            save_path=save_path
        )
        
        if self.p.show_info >= 3: 
            print(f'size SbS 2D map: {np.shape(fit2D_SbS)}')
        
        # plot data, fit, and residual 2D maps (works if full 2D map is fitted/ no slices skipped)
        if self.p.first_N_spec_only == -1 and self.p.skip_first_N_spec == -1:

            fitlib.plt_fit_res_2D(
                data=self.data,
                fit=fit2D_SbS,
                x=self.energy,
                y=self.time,
                config=self.plot_config,
                x_lim=self.e_lim,
                y_lim=self.t_lim,
                save_img=-1 if self.p.show_info==0 else 1,
                save_path=save_path
            )
        #
        return None
    
    #
    def add_time_dependence(self, model_yaml, model_info, par_name, frequency=-1):
        """
        Add time dependence for one parameter (<par_name>) of currently active model
        Load "Dynamics"-type model defined by <model_info> in <model_yaml> .yaml file
        The time-dependent behaviour repeats either not at all (default, -1) or with <frequency>
        """
        t_mod = self.load_model(model_yaml, model_info, par_name, 
                                debug=False if self.p.show_info<2 else True) # load
        self.model_active.add_dynamics(t_mod, frequency) # add
        self.model_active.dim = 2 # increase dimension of model to 2
        #
        return None
    
    #
    def fit_2Dmodel(self, model_name, fit, **fit_wrapper_kwargs):
        """
        Perform a energy- and time-dependent fit using the model with the name <model_name>
        (pass via <add_model> method)
        
        <fit> =0: show initial guess, =1: perform (one method) fit 
              =2: perform a fit to find global minumum (fitAlg1)
              followed by fit to optimize locally (fitAlg2)
        
        [see fitlib.fit_wrapper for details on keyword arguments]
        """
        t_2D = time.time() # start timing for 2D fit
        
        # find model with matching name from list
        self.model_2D = self.select_model(model_info=model_name)
        
        # define (and create) path where 2D fit results will be saved to
        path_2D_results = self.create_model_path(model_name)
                
        # set all fixed 2D fit parameters equal to baseline model results
        base_df = ulmfit.par2df(self.model_base.lmfit_pars, col_type='min')
        self.model_2D.update_value(new_par_values = list(base_df['value']),
                                   par_select = list(base_df['name']))
        # const [x, data, package, function string, unpack, energy limits, time limits]
        self.model_2D.const = (self.energy, self.data, self.p.spec_lib, self.p.spec_fun_str, 0, \
                               self.e_lim, self.t_lim)
        # args [for fit function called in residual function]
        self.model_2D.args = (self.model_2D, 2, False) # model, dimension, debug
        
        # fit (with confidence intervals)
        self.model_2D.result = fitlib.fit_wrapper(const=self.model_2D.const,
                                                  args=self.model_2D.args,
                                                  par_names=self.model_2D.par_names, 
                                                  par=self.model_2D.lmfit_pars,
                                                  fit_type=fit,
                                                  show_info=1 if self.p.show_info>=2 else 0,
                                                  save_output=1,
                                                  save_path=path_2D_results / model_name,
                                                  **fit_wrapper_kwargs)
        if fit >= 1:
            self.save_2Dmodel_fit(save_path=path_2D_results)
            fitlib.time_display(t_start=t_2D, print_str='Time elapsed for 2D model fit: ')
            display(self.model_2D.result[1].params) # display the final parameters below figure
        #
        return None
    
    #
    def save_2Dmodel_fit(self, save_path):
        """
        Save (additional) results from 2D model/component/parameter fit 
        """
        self.model_2D.create_value2D() # update 2D spectrum to final fit result
        # plot data, fit, and residual 2D maps
        fitlib.plt_fit_res_2D(
            data=self.data,
            fit=self.model_2D.value2D,
            x=self.energy,
            y=self.time,
            config=self.plot_config,
            x_lim=self.e_lim,
            y_lim=self.t_lim,
            save_img=-1 if self.p.show_info==0 else 1,
            save_path=save_path
        )
        # dpi_plot = round(1.5 *self.p.dpi_plt), NOT AVAILABLE YET (fig_size)
        #
        return None
    
    #
    def compare_models(self):
        """
        this could be a good feature to build out
        compare residual maps, max/min and std of 2D residuals
        how to measure overall quality of fit? reduced chi2?
        """
        #
        return None