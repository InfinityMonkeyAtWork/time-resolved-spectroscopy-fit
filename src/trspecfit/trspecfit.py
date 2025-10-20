#
# 1D/2D Spectroscopy Fitting Module
#
from trspecfit import mcp
from trspecfit import fitlib
from trspecfit import spectra
from trspecfit.utils import lmfit as ulmfit
import numpy as np
from trspecfit.utils import arrays as uarr
import os # replace os.join with "pathlib path / "subfolder" / "file name"
import pathlib
#from trspecfit.utils import os as uos
from trspecfit.utils import regex as ure
from trspecfit.utils import plot as uplt
import copy
import time
from IPython.display import display
# function library for energy, time, and depth components
from trspecfit.functions import energy as fcts_energy
from trspecfit.functions import time as fcts_time
from trspecfit.functions import depth as fcts_depth
#import yaml
#ruaml.yaml mod:
import sys
from ruamel.yaml import YAML
from ruamel.yaml.constructor import SafeConstructor
# yaml parser needs to know which components to number
from trspecfit.config import prefix_exceptions

# to do:
# - pull load_model from File to Project to enable global fitting across files
# - dimension: add =3 for slicing larger dimensional data?
#
# what does show_info mean? convert to binary debug by True if show_info >=3 else False
#
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
    for module in [fcts_energy, fcts_time, fcts_depth]:
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
class Project:
    """
    copy the bare minimum from TRAPXPS to here
    
    then make TRAPXPS project and files each a class with the project
    and file from here as a superclass! this way you can add 
    functionality like a user would (e.g. energy calibration, detector
    calibration, other fit methods, etc.)
    
    enable different configurations for different methods/users [how?]
    move the current defaults to a yaml file and load them from there?
    
    general framework for global i.e. project-wide fits:
    define_global_variable() -> set vary=False on the file level and make a project-level
    fit including all global variables (?) and a "for file in files" loop that does the
    file level fits sequentially (there must be a better way to do this, but start with that)
    """
    def __init__(self, path, name='test'):
        self.path = pathlib.Path(path) if path is not None else pathlib.Path('test')
        self.path_results = pathlib.Path(f'{path}_fits')
        self.run = name
        self.path_run = self.path_results / name
        # settings (could be loaded from a .yaml file)
        # change according to your spectroscopy method
        self.show_info = 1 # 0: no output, 1: important, 2: all, 3+: debugging
        self.e_label = 'Binding energy (eV)'
        self.t_label = 'Time (s)'
        self.z_label = 'Intensity (arb. units)'
        self.xdir = 'rev' # default ('def') or reverse ('rev')
        self.dpiplt = 100 # standard plot size for all plots
        self.res_mult = 5 # multiply residuals with this scaling factor
        self.ext = '.dat' # define file extension/ format
        self.fmt = '%.6e' # define format of saved data
        self.delim = ',' # define delimiter between data points
        self.DA_fmt = '%04d' # define format of data_analysis file numbering
        self.DA_slices_fmt = "%06d" # format for time slice numbering (SbS only)
        self.lib_spec = spectra # library used to generate spectra
        self.mcp_fun_str = 'fit_model_mcp'
        self.mcp_fun = getattr(self.lib_spec, self.mcp_fun_str)
        # keep these around? (useful for debugging SbS methods)
        self.skip_first_N_spec = -1
        self.first_N_spec_only = -1
        #
        return None
    
    #
    def load_configuration(self, project_yaml):
        """
        To do: create .yaml files for different users/ spectroscopy templates?
        and load them, basically just re-defining (a subset of) project attributes
        """
        #
        return None
    
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
    
    #
    def describe(self):
        """
        
        """
        print(f"File # x [path: {self.path}]")
        
        if self.dim == 1:
            uplt.plot_1D(data = [self.data,], x = self.energy,
                         xlabel = self.p.e_label, ylabel = self.p.z_label,
                         xdir = self.p.xdir, dpi_plot = self.p.dpiplt,
                         vlines = self.e_lim_abs)
            
        elif self.dim == 2:
            uplt.plot_2D(data = self.data, x = self.energy, y = self.time,
                         xlabel = self.p.e_label, ylabel = self.p.t_label,
                         xdir = self.p.xdir, dpi_plot = self.p.dpiplt,
                         vlines = self.e_lim_abs, hlines = self.t_lim_abs)
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
    def _load_and_number_yaml_components(self, model_yaml, model_info, par_name='', DEBUG=False):
        """
        Load YAML file and apply appropriate component numbering strategy.
        For energy models: use component numbering from construct_yaml_map
        For dynamics models with subcycles: number components globally 
        """
        if DEBUG:
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
                else:
                    if not isinstance(model_info_ALL, dict):
                        raise ValueError("YAML root must be a dictionary or a list of (name, components) tuples.")
                    model_info_dict = model_info_ALL
                    if len(model_info_dict) != len(set(model_info_dict.keys())):
                        raise ValueError("Duplicate model names found in YAML dictionary.")
                
                if DEBUG:
                    print('model_info_ALL:')
                    print(model_info_ALL)
                    print('model_info_dict:')
                    print(model_info_dict)
                
                # Apply appropriate numbering strategy
                if par_name != '':
                    # This is a dynamics model - resolve numbering conflicts across subcycles
                    model_info_dict = self._resolve_dynamics_numbering_conflicts(model_info_dict, model_info, DEBUG)
                # For energy models, numbering is already complete from construct_yaml_map
                return model_info_dict
                
        except FileNotFoundError:
            raise FileNotFoundError(
                f"FileNotFound: <model_yaml> file input\n"
                f"File should be located in: {self.p.path}\n"
                f"Check file name for typos: {model_yaml_path}"
            )
        except yaml.YAMLError as exc:
            raise RuntimeError(f"YAML error while loading {model_yaml}: {exc}")

    #
    def _resolve_dynamics_numbering_conflicts(self, model_info_dict, model_info, DEBUG=False):
        """
        Resolve numbering conflicts for dynamics models by tracking used numbers globally
        and reassigning conflicting numbers to the next available number.
        
        This preserves the existing YAML numbering where possible and only changes
        numbers when there are conflicts across subcycles.
        """
        if DEBUG:
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
        
        if DEBUG:
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
                        
                        if DEBUG:
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

            if DEBUG:
                print(f"\nProcessed submodel: {submodel}")
                print(f"  {submodel}: {list(processed_dict[submodel].keys())}")
        
        if DEBUG:
            print(f"\nFINAL processed_dict:")
            for submodel in model_info:
                if submodel in processed_dict:
                    print(f"  {submodel}: {list(processed_dict[submodel].keys())}")
        #
        return processed_dict

    #
    def load_model(self, model_yaml, model_info, par_name='', DEBUG=False):
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
        model_info_dict = self._load_and_number_yaml_components(model_yaml, model_info, par_name, DEBUG)

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
        loaded_model.energy = self.energy
        loaded_model.time = self.time
        loaded_model.xdir = self.p.xdir # file.p is the parent project
        loaded_model.e_label = self.p.e_label # x axis -> energy
        loaded_model.t_label = self.p.t_label # y axis -> time
        loaded_model.z_label = self.p.z_label # z axis -> intensity
        
        all_comps = [] # initialize component list
        
        # go through (sub)model(s)
        # (for mcp.Dynamics model instances length model_info could be larger than 1)
        for subcycle, submodel in enumerate(model_info):
            # get the section defined by model_info
            try:
                submodel_info = model_info_dict[submodel]
            except KeyError:
                print(f'Model "{submodel}" not found in {model_yaml}')
                return None
            
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
            mod.plot_1D(plt_ind=1) # plot guess only (individual components)
        
        if detail == 1 and mod.dim == 1:
            mod.create_value1D(store1D=1) # update individual component spectra
            # plot initial guess (individual components), data, and residual
            title_mod = f'File: {self.path}, ' +\
                        f'Model: "{model_info}" (from "{mod.yaml_f_name}.yaml")' +\
                        f': initial guess'
            fitlib.plt_fit_res_1D(x = self.energy, y = self.data_base,
                                  fit_fun_str = self.p.mcp_fun_str, package = self.p.lib_spec,
                                  par_init = [], par_fin = mod.lmfit_pars, 
                                  args = (mod, 1), plot_ind = 1, show_init = 0,
                                  xlabel = self.p.e_label, ylabel = self.p.z_label,
                                  xdir = self.p.xdir, title = title_mod, dpi_plt = self.p.dpiplt,
                                  fit_lim = self.e_lim, res_mult = self.p.res_mult,
                                  legend = [comp.name for comp in mod.components])
            
        if detail == 1 and mod.dim == 2:
            mod.create_value2D() # update spectrum
            # plot data, fit, and residual 2D maps
            fitlib.plt_fit_res_2D(data = self.data, x = self.energy, y = self.time,
                                  fit = mod.value2D, xdir = self.p.xdir,
                                  xlabel = self.p.e_label, ylabel = self.p.t_label,
                                  xlim = self.e_lim, ylim = self.t_lim)
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
        if show_plot == True:
            uplt.plot_1D(data = [self.data_base,], x = self.energy, dpi_plot = self.p.dpiplt,
                         xdir = self.p.xdir, xlabel = self.p.e_label, ylabel = self.p.z_label,
                         title = f"Baseline data: t in {self.base_t_abs} (index: {self.base_t_ind})")
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
        if self.p.xdir == 'rev':
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
                uplt.plot_1D(data = [self.data, y_cut], 
                             x = [self.energy, x_cut],
                             waterfall = (np.max(abs(y_cut))-np.min(abs(y_cut)))/8,
                             xlabel = self.p.e_label, ylabel = self.p.z_label,
                             xdir = self.p.xdir, legend = ['all', 'cut'], 
                             vlines = self.e_lim_abs, dpi_plot = self.p.dpiplt)
            elif self.dim == 2:
                uplt.plot_2D(data = self.data, x = self.energy, y = self.time,
                             xlabel = self.p.e_label, ylabel = self.p.t_label,
                             xdir = self.p.xdir, dpi_plot = self.p.dpiplt,
                             vlines = self.e_lim_abs, hlines = self.t_lim_abs)
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
        self.model_base.const = (self.energy, self.data_base, self.p.lib_spec, self.p.mcp_fun_str, 0, self.e_lim, [])
        # args [for fit function called in residual function]
        # model, dimension (dim =1 for baseline and SbS, =2 for 2D (global) fit), debug
        self.model_base.args = (self.model_base, 1, False)
        # fit (optionally) with confidence intervals
        self.model_base.result = fitlib.fit_wrapper(const = self.model_base.const,
                                                    args = self.model_base.args,
                                                    par_names = self.model_base.par_names,
                                                    par = self.model_base.lmfit_pars,
                                                    fit_type = fit,
                                                    show_info = 1 if self.p.show_info>=2 else 0,
                                                    save_output = 1,
                                                    save_path = path_base_results / model_name,
                                                    **lmfit_wrapper_kwargs)
        
        # display/plot and save baseline fit summary
        #self.model_base.create_value1D(store1D=1) # update individual component spectra
        title_base = f'File: {self.path}, ' +\
                     f'Model: "{model_name}" (from "{self.model_base.yaml_f_name}.yaml")'
        fitlib.plt_fit_res_1D(x = self.energy, y = self.data_base,
                              fit_fun_str = self.p.mcp_fun_str, package = self.p.lib_spec,
                              par_init = initial_guess, par_fin = self.model_base.result[1],
                              args = self.model_base.args, plot_ind = 1, show_init = 1,
                              xlabel = self.p.e_label, ylabel = self.p.z_label,
                              xdir = self.p.xdir, title = title_base, dpi_plt = self.p.dpiplt,
                              fit_lim = self.e_lim, res_mult = self.p.res_mult,
                              legend = [comp.name for comp in self.model_base.components],
                              save_fig = -1 if self.p.show_info<1 else 1,
                              save_fig_path = path_base_results / 'base_fit.png')
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
        # in the past the energy limits were shifted as well (commented now) because
        # comparing SbS with 2D method requires same limits! search for [*lim]
        
        t_SbS = time.time() # start timing for SbS fit
        
        # find model with matching name from list
        self.model_SbS = self.select_model(model_info=model_name)
        
        # define (and create) path where SbS fit results will be saved to
        path_SbS_results = self.create_model_path(model_name, subfolders=['slices',])
                
        # set all fixed SbS fit parameters equal to baseline model results
        base_df = ulmfit.par2df(self.model_base.lmfit_pars, col_type='min')
        self.model_SbS.update_value(new_par_values=list(base_df['value']), par_select='all')
        
        # find all parameters with names ending in "x0" so they can be updated for every slice
        e_pos_pars = ure.search_line_by_line(lines = self.model_SbS.par_names,
                                             str_search = '_x0', location = 'end',
                                             include_str_search = True, print_info = 0)
        # find their corresponding values
        e_pos_vals = uarr.get_item(base_df, row=['name', e_pos_pars], col='value', astype='series')
        #[*lim]e_lim_OG = copy.deepcopy(self.e_lim) # make a copy of original energy limits
        
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
            #[*lim] update energy limits and index
            #self.set_fit_limits(energy_limits=[lim +deltaMAX for lim in e_lim_OG], show_plot=False)
            #if self.p.show_info >= 3: print(f'Updated energy limits: {self.e_lim}')
            # get initial guess
            initial_guess = ulmfit.par_extract(self.model_SbS.lmfit_pars, return_type='list')
        
            # const = (x, data, package, function string, unpack, energy limits, time limits)
            self.model_SbS.const = (self.energy, s, self.p.lib_spec, self.p.mcp_fun_str, 0, self.e_lim, [])
            # args (lmfit2D.Model, dimension, debug) [for fit function called in residual function]
            self.model_SbS.args = (self.model_SbS, 1, False)

            # fit with confidence intervals
            result_SbS = fitlib.fit_wrapper(const = self.model_SbS.const,
                                            args = self.model_SbS.args,
                                            par_names = self.model_SbS.par_names,
                                            par = self.model_SbS.lmfit_pars,
                                            fit_type = fit,
                                            show_info = 1 if self.p.show_info>=3 else 0,
                                            save_output = 1,
                                            save_path = path_slice,
                                            **fit_wrapper_kwargs)

            # add final fit parameters to list of fit parameters of all spectra
            self.results_SbS.append(result_SbS)

            # (optionally) plot and (always) save fit summary for this slice
            fitlib.plt_fit_res_1D(x = self.model_SbS.const[0], y = self.model_SbS.const[1],
                                  fit_fun_str = self.p.mcp_fun_str, package = self.p.lib_spec,
                                  par_init = initial_guess, par_fin = result_SbS[1],
                                  args = self.model_SbS.args, plot_ind = 1, show_init = 1,
                                  xlabel = self.p.e_label, xdir = self.p.xdir,
                                  ylabel = self.p.z_label, dpi_plt = self.p.dpiplt,
                                  fit_lim = self.e_lim, res_mult = self.p.res_mult,
                                  save_fig = -1 if self.p.show_info<3 else 1,
                                  save_fig_path = path_slice +'.png')
            #
            if s_i == self.p.first_N_spec_only: break
        
        #[*lim]]self.e_lim = e_lim_OG # set energy limits back to original value
        
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
        df_SbS = fitlib.results2df(results = self.results_SbS, 
                                   x = self.time, 
                                   xlabel = self.p.t_label, 
                                   index = np.arange(0, len(self.time)),
                                   skip_first_N_spec = self.p.skip_first_N_spec, 
                                   first_N_spec_only = self.p.first_N_spec_only,
                                   save_df = -1 if self.p.show_info==0 else 1,
                                   save_path = save_path)
        if self.p.show_info >= 3: display(df_SbS)
        
        # get slice-by-slice fit spectra as a 2D map
        fit2D_SbS = fitlib.results2fit2D(results = df_SbS[self.model_SbS.par_names],
                                         const = self.model_SbS.const, 
                                         args = self.model_SbS.args,
                                         save_2D = -1 if self.p.show_info==0 else 1,
                                         save_path = save_path)
        if self.p.show_info >= 3: print(f'size SbS 2D map: {np.shape(fit2D_SbS)}')
        
        # plot data, fit, and residual 2D maps (works if full 2D map is fitted/ no slices skipped)
        if self.p.first_N_spec_only == -1 and self.p.skip_first_N_spec == -1:
            fitlib.plt_fit_res_2D(data = self.data, x = self.energy, y = self.time,
                                  fit = fit2D_SbS, xdir = self.p.xdir,
                                  xlabel = self.p.e_label, ylabel = self.p.t_label,
                                  xlim = self.e_lim, ylim = self.t_lim,
                                  save_img = -1 if self.p.show_info==0 else 1,
                                  save_path = save_path)
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
                                DEBUG=False if self.p.show_info<2 else True) # load
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
        self.model_2D.const = (self.energy, self.data, self.p.lib_spec, self.p.mcp_fun_str, 0, \
                               self.e_lim, self.t_lim)
        # args [for fit function called in residual function]
        self.model_2D.args = (self.model_2D, 2, False) # model, dimension, debug
        
        # fit (with confidence intervals)
        self.model_2D.result = fitlib.fit_wrapper(const = self.model_2D.const,
                                                  args = self.model_2D.args,
                                                  par_names = self.model_2D.par_names, 
                                                  par = self.model_2D.lmfit_pars,
                                                  fit_type = fit,
                                                  show_info = 1 if self.p.show_info>=2 else 0,
                                                  save_output = 1,
                                                  save_path = path_2D_results / model_name,
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
        fitlib.plt_fit_res_2D(data = self.data, x = self.energy, y = self.time,
                              fit = self.model_2D.value2D, xdir = self.p.xdir,
                              xlabel = self.p.e_label, ylabel = self.p.t_label,
                              xlim = self.e_lim, ylim = self.t_lim,
                              save_img = -1 if self.p.show_info==0 else 1,
                              save_path = save_path)
        # dpi_plot = round(1.5 *self.p.dpiplt), NOT AVAILABLE YET (fig_size)
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