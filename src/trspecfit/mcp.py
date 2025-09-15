#
# package to extend lmfit library to multi-dimensional data
#
import lmfit
from trspecfit.utils import util_lmfit as ulmfit
from trspecfit.utils import util_arrays as uarr
from trspecfit.utils import util_plot as uplt
import math
import numpy as np
import re
import inspect
import copy
from IPython.display import display
from asteval import Interpreter # asteval is used for expressions referencing time-dependent parameters
# function library for energy, time, and depth components
from trspecfit.functions import energy as fcts_energy
from trspecfit.functions import time as fcts_time
#from trspecfit.functions import depth as fcts_depth
# configuration functions
from trspecfit.config import prefix_exceptions, background_functions, energy_functions
import concurrent.futures

# To Do:
# - implement depth dependence (or general implicit variables)
# - modify "create_value1D" to include subcycles! be smart about it
#   (N=0 applies to all of t or E, N>0 applies to the Nth subcycle only
#   use "normalize_time" to get the applicable N (N_sub)
# - delete attributes and methods that you don't actually end up using
# - parallelize 2D map generation or somehow speed it up

#
#
class Model:
    """
    Define a 2D time- and energy-resolved fit model using lmfit
    <package> python package containing the function definitions
    <components> is a list of components which are are functions 
    definitions (str) in the specified <package>.
    <lmfit_pars> is a list of the component parameters containing one
    element per component. Within each element are parameter 
    information (see description of par for a details.
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
        #self.data = None # (currently) not necessary
        self.dim = None
        self.energy = None
        self.time = None
        self.xdir = 'def'
        self.e_label = 'Energy'
        self.t_label = 'Time'
        self.z_label = 'Intensity'
        #
        return None
    
    #  
    def describe(self, detail=0):
        """
        Display info about model according to level of detail (>=0)
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
    def add_components(self, comps_list, DEBUG=False):
        """
        user passes a list of all components that define the model <comps_list>
        see component class for details
        
        this method will populate the following attributes of the model:
        peak_fcts, lmfit_par_list, lmfit_pars, par_names
        
        Components are expected to have their names already set from the YAML parser
        (e.g., GLP_01, GLP_02, Offset, Shirley). The component numbering is handled
        during YAML parsing, so this method just processes the pre-named components.
        
        par names are constructed as follows:
        prefix + component name + '_' + parameter name
        [prefix is equal to model.name for models of class Dynamics and '' for model] 
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
            comp.create_pars(prefix=prefix, DEBUG=DEBUG)
        
        # update model lmfit_par_list (+par_names) and components
        self.update(DEBUG=DEBUG)
        if DEBUG == True: self.lmfit_pars.pretty_print()          
        #
        return None
    
    #
    def find_par_by_name(self, par_name):
        """
        Find index of the component and its parameter (par)
        corresponding to the input parameter name <par_name>
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
        Print info on all parameters of the model individually [debugging]
        """
        for c in self.components:
            for p in c.pars:
                p.describe(detail)
        #
        return None
    
    #
    def update(self, DEBUG=False):
        """
        Update model from the bottom up: parameters -> components -> model.
        recompile all pars for all components and recreate lmfit_par_list
        for all components as well as for the model itself (and model par_names)
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
        if DEBUG == True: self.lmfit_pars.pretty_print()
        
        # update list of all parameter names
        self.par_names = [par.name for par in self.lmfit_par_list]
        #
        return None
    
    #
    def update_value(self, new_par_values, par_select='all'):
        """
        Update model from the top down: model -> components -> parameters
        (based on new model.lmfit_pars input)
        <par_select> is either 'all' or a list of parameter names to be updated
        
        Use during fitting before calling (via fit_wrapper, and residual_fun)
        spectra.fit_model_mcp(x, par, plot_ind, model, dim)
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
    def add_dynamics(self, dynamics_model, frequency=-1, DEBUG=False):
        """
        Add temporal dynamics model (<Dynamics> instance) to a parameter
        The name of the <Dynamics_model> has to match parameter name
        <frequency> is the repetition frequency of the temporal dynamics
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
        self.update(DEBUG=DEBUG)
        
        # Re-analyze all expressions since time-dependence status may have changed
        self._analyze_expression_dependencies()
        #
        return None
    
    #
    def _analyze_expression_dependencies(self):
        """
        Analyze all parameter expressions for time-dependent references
        This should be called whenever time-dependence status of parameters changes
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
        Get all parameters from all components in this model
        """
        all_parameters = []
        for comp in self.components:
            all_parameters.extend(comp.pars)
        #
        return all_parameters
    
    #
    def combine(value, comp, t_ind=0):
        """
        Combine component value with input <value> via addition or convolution
        <value> is typically the current spectrum in the process of model evaluation
        Note: component subcycle (only defined in time) is handled in Component.value()
        """
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
            # generalize to enable convolution in energy? #$%
            return uarr.my_conv(x=x_axis, y=value, kernel=comp.value(t_ind))
   
    #
    def create_valueTEST(self, t_ind=0, store1D=0, return1D=0, DEBUG=False):
        """
        Efficiency
        For components that have no time-dependence in any parameter 
        you should find a way to not call them for every t (?)
        if you would only add components, that would be trivial,
        as there are convolutions and background functions it's not 
        that simple unfortunately
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
            if DEBUG == True: print(N+1); print(self.components[-(N+1)].fct_str)
            if store1D == 1: current_spec = copy.deepcopy(self.value1D)
            #
            self.value1D = Model.combine(self.value1D, 
                                         self.components[-(N+1)],
                                         t_ind)
            # check on last component value added to model
            if store1D == 1: self.component_spectra.append(self.value1D -current_spec)
            if DEBUG == True: uplt.plot_1D([self.component_spectra[-1],])
            
        # flip component spectra list as components are combined LIFO in this function
        if store1D == 1: self.component_spectra = self.component_spectra[::-1]
        #
        if return1D == 1: return self.value1D
        else: return None
    
    #
    def create_value1D(self, t_ind=0, store1D=0, return1D=0, DEBUG=False):
        """
        Return 1D (energy or time) data by evaluating the model
        component numbering: comps = [1, 2, 3, ... , N-2, N-1, N] 
        will combine component N with N-1, the result with N-2, and so forth
        pass "store1D = 1" to save individual component spectra to the model
        """
        # re-initialize list containing individual component spectra
        if store1D == 1: self.component_spectra = []
        # initialize value1D by evaluating last component
        self.value1D = self.components[-1].value(t_ind)
        if store1D == 1: self.component_spectra.append(self.value1D)
        
        # combine the components into a spectrum/ time dynamics curve
        for N in range(len(self.components) -1):
            if DEBUG == True: print(N+2); print(self.components[-(N+2)].fct_str)
            if store1D == 1: current_spec = copy.deepcopy(self.value1D)
            #
            self.value1D = Model.combine(self.value1D, 
                                         self.components[-(N+2)],
                                         t_ind)
            # check on last component value added to model
            if store1D == 1: self.component_spectra.append(self.value1D -current_spec)
            if DEBUG == True: uplt.plot_1D([self.component_spectra[-1], ])
            
        # flip component spectra list as components are combined LIFO in this function
        if store1D == 1: self.component_spectra = self.component_spectra[::-1]
        #
        if return1D == 1: return self.value1D
        else: return None

    #
    def create_value2D(self, t_ind=[], DEBUG=False):
        """
        create time- and energy-dependent spectrum for current parameters
        t_ind=[] -> process entire time axis
        t_ind = [t_start, t_stop] -> process self.time[t_start:t_stop]
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
    def create_value2D_parallel(self, t_ind=[], DEBUG=False):
        """
        create time- and energy-dependent spectrum for current parameters
        t_ind=[] -> process entire time axis
        t_ind = [t_start, t_stop] -> process self.time[t_start:t_stop]
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
    def plot_1D(self, t_ind=0, plt_ind=0, dpi_plt=100,
                xlim=[], xtype='lin', ylim=[], ydir='', ytype='lin',
                save_img=0, save_path=[], dpi_save=300):
        """
        plot model value result as a function of <x_axis> (plt_ind=0 [default], set plt_ind
        to 1 if you want to plot the individual components making up the model)
        x axis (all optional): xlim=[lo, hi], xtype='lin'/'log'
        y axis (all optional): see "x axis" with addition of ydir direction ("rev"/"standard")
        <save_img>= 0(no), 1(yes), <save_path> full image path (including extension)
        """
        # the model calling this method is describing temporal dynamics of a par
        if isinstance(self, Dynamics): xdir = 'default'
        else: xdir = self.xdir
                
        # get x axis and its label (all comps should have the same package)
        if self.components[0].package == fcts_energy:
            x_axis = self.energy
            x_name = self.e_label
            info = f'[{self.t_label}={round(self.time[t_ind],3)} (index={t_ind})]'
        elif self.components[0].package == fcts_time: 
            x_axis = self.time
            x_name = self.t_label
            info = ''
            
        # populate <component_spectra> argument of the model
        self.create_value1D(t_ind, store1D=1)
        
        # plot
        uplt.plot_1D(data = self.component_spectra if plt_ind==1 else [self.value1D,],
                     title = f'model "{self.name}" {info}',
                     x = x_axis,
                     xlabel = x_name, ylabel = self.z_label,
                     xlim = xlim, xdir = xdir, xtype = xtype,
                     ylim = ylim, ydir = ydir, ytype = ytype,
                     legend = [c.name for c in self.components] if plt_ind==1 else ['sum',],
                     save_img = save_img, save_path = save_path, dpi_save = dpi_save,
                     dpi_plot = dpi_plt)
        #
        return None
    
    #
    def plot_2D(self, dpi_plt=100, save_img=0, save_path='', dpi_save=300, 
                zlim=[], xlim=[], xtype='lin', ylim=[], ydir='', ytype='lin'):
        """
        Plot model attribute value2D, i.e. time- and energy-dependent spectrum
        """
        #
        uplt.plot_2D(data = self.value2D, x = self.energy, y = self.time, 
                     title = f'model "{self.name}"', ranges = [xlim, ylim, zlim],
                     xlabel = self.e_label, ylabel = self.t_label,
                     xdir = self.xdir, xtype = xtype, ydir = ydir, ytype = ytype,
                     save_img = save_img, save_path = save_path,
                     dpi_save = dpi_save, dpi_plot = dpi_plt)
        #
        return None
    
    #
    def combine_models(name, model1, model2, DEBUG=False):
        """
        Combine two models [haven't tied this yet!]
        """
        # initialize new model object 
        new_model = Model(name)
        # combine components
        new_model.add_components(model1.components.extend(model2.components),
                                 DEBUG = DEBUG)
        # call function to populate other attributes using components
        # or is that all the functionality this needs? 
        #
        return new_model

    #
    def par_distribution_brood_force_test(self, comp_ind, par_ns, A_dist, x_dist):
        """
        brood for test of what the resulting component spectrum should look like 
        if a parameter (like A or x0) has a distribution
        A_dist is distribution of amplitudes (list)
        x_dist is dirtribution of x0 values (list)
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
    Define a spectral component (peak defined in fcts_energy)
    OR
    a temporal dynamics component (functions in fcts_time) of the fit
    """
    #
    def __init__(self, comp_name, package=fcts_energy, comp_type='add', comp_subcycle=0):
        # package containing component (either fcts_energy or fcts_time)
        self.package = package
        # name of the component (str)
        self.comp_name = comp_name
        # retrieve function_str and component number from component name if necessary
        if '_' in comp_name and comp_name.split('_')[-1].isdigit():
            # this is a numbered component name (e.g., "GLP_01")
            name_parts = comp_name.split('_')
            self.fct_str = name_parts[0]
            self.N = int(name_parts[-1])
        else:
            # This is a base function name (e.g., "Offset", "Shirley")
            self.fct_str = comp_name
            self.N = -1  # Background functions don't get numbered
        
        # add (or convolute) this component to (or with) other components ['add'/ 'conv']
        self.comp_type = 'back' if self.fct_str in background_functions() else comp_type 
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
        return getattr(self.package, self.fct_str)
    
    # [automatic] do the same for function argument specs
    @property 
    def fct_specs(self):
        return inspect.getfullargspec(self.fct)
    
    # [automatic] and function arguments specifically
    @property 
    def fct_args(self):
        return self.fct_specs.args
    
    # [automatic] create a prefix for parameters of this component
    @property 
    def prefix(self):
        # define prefix for parameter names
        if self.fct_str in prefix_exceptions():
            return ''
        else: # number the components starting from N=1
            return self.comp_name +'_'
        
    # [automatic] create a name for this component
    @property 
    def name(self):
        # use the stored component name (which includes numbering if applicable)
        return self.comp_name
    
    #
    def add_pars(self, par_info_dict):
        """
        Pass a dictionary containing parameter names as keys and
        par.info-like items as values (see par class description for details)
        """
        self.par_dict = par_info_dict
        #
        return None
    
    #
    def create_pars(self, prefix='', DEBUG=False):
        """
        Populate self.pars with Par objects using the parameter dictionary
        Two-pass: (1) create all parameters without expressions, (2) set expressions.
        This makes sure that forward references are handled correctly.
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
            if DEBUG:
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
        Get a list of all individual lmfit_par objects that make up the component 
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
        # print info on function
        print(f'function: {self.fct_str} from {self.package}')
        # detailed description
        if detail >= 1:
            # addition or convolution?
            if self.comp_type == 'add':
                addORconv = 'added to other components'
            elif self.comp_type == 'conv':
                addORconv = 'convoluted with other components'
            # subcycle info
            if self.subcycle == 0:
                subcycle_str = 'for all times t'
            else:
                subcycle_str = f'within subcycle {self.subcycle}'
            # print info
            print(f'function will be {addORconv} [{subcycle_str}]')

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
    def create_t_kernel(self, DEBUG=False):
        """
        Create time axis for a kernel component
        i.e. a component that is convoluted with other components within a model
        """
        # get kernel parameters i.e. component parameters
        parK = ulmfit.par_extract(self.par_dict, return_type='list')
        if DEBUG == True: print(f'component/kernel parameters as list: {parK}')
        # define kernel time axis
        kernel_width = getattr(fcts_time, self.fct_str +'_kernel_width')()
        if DEBUG == True: print(f'kernel width loaded from fcts_time: {kernel_width}')
        t_range = parK[0] *kernel_width
        try: t_step = self.time[1] -self.time[0]
        except: print(f'time axis of component {self.fct_str} not defined')
        if DEBUG == True: print(f'delta time (from self.time): {t_step}')
        t_kernel = np.arange(-t_range, t_range+t_step, t_step)
        #
        return t_kernel

    #
    def value(self, t_ind=0, **kwargs):
        """
        Evaluate component function with one specific set of parameters at
        independent variable/ control coordinates [either time or energy defined 
        by the <component.package> which is either fcts_energy or fcts_time]
        pass arguments <args> to component like this: function(x, *pars, args)
        (takes one individual time index for now, vectorize later?)
        returns component value as a function of time or energy (as np.array)
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
        Plot component as a function of <x_axis>
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
                     x = x_axis,
                     xlabel = x_name, ylabel = 'Amplitude')
        #
        return None

#
# 
class Par:
    """
    Extend lmfit.Parameter to include option for time-dependent behaviour
    
    (necessary attributes)
    <name>: (str) name of one fit parameter of one spectral component
    <info>: list containing one list per parameter of 
            EITHER [initial guess, vary/fixed(True/False),
                   minimum boundary, maximum boundary]
            OR     [condition] passed as expr=<condition>
    (optional attributes)
    <t_vary>: (True/False) does the parent parameter (i.e. peak/
              spectral component) have a time-dependent behaviour?
    <t_model>: model describing the kinetics/temporal dynamics of the 
               parent parameter (par)
    (attributes created through methods of this class)
    <lmfit_par>: instance of lmfit.Parameter() type if t_vary=False
                 or lmfit.Parameters() [multiple] if t_vary=True
    <lmfit_par_list>: flattened list of all spectral and temporal 
                      lmfit.Paramter objects in the par object
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
    def create(self, prefix='', suffix='', expr_skip=False, DEBUG=False):
        """
        Create par object from self.name and self.info will populate spectral
        (fcts_energy) attributes only.
        To add time-dependence to an par use Model.add_Dynamics() followed by
        Par.update().
        """
        # create standard lmfit parameter (spectral component)
        if expr_skip and len(self.info) == 1 and isinstance(self.info[0], str):
            # if skipping expression, use a dummy value for now
            lmfit_par = ulmfit.par_create(self.name, [0, True, -np.inf, np.inf], prefix, suffix, DEBUG)
        else:
            lmfit_par = ulmfit.par_create(self.name, self.info, prefix, suffix, DEBUG)
        # add to lmfit_par attribute
        self.lmfit_par.add_many(lmfit_par)
        # and list of individual lmfit paramters
        self.lmfit_par_list.extend([lmfit_par])    
        #
        return None
    
    #
    def update(self, t_model):
        """
        Update par after adding time dependence to it
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
        Return value of the parameter 
                
        (t_ind is the index where time_axis[t_ind] = t, where <t> is
        the point in time for which parameter value shall be returned)
        
        <update_t_model> recomputes parameter time dependence (Dynamics)
        set True while fitting, set False while building model 2Dmap
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
            print(f't_vary attribute of par "{self.name}" is not valid')
        #
        return value
    
    #
    def analyze_expression_dependencies(self, all_parameters):
        """
        Check if this parameter's expression references time-dependent parameters
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
        Extract parameter names referenced in an expression string
        Looks for strings that start with function names from functions.energy
        since parameter names follow the pattern {function_name}_{number}_{parameter_name}
        """        
        # Pattern to match parameter names (letters, numbers, underscores, but not starting with number)
        pattern = r'\b[A-Za-z_][A-Za-z0-9_]*\b'
        matches = re.findall(pattern, expr_string)
        
        # Filter to keep only strings that start with known function names
        # This catches parameter names like GLP_01_A, GLP_02_x0, etc.
        parameter_refs = []
        for match in matches:
            for func_name in set(energy_functions()):
                if match.startswith(func_name + '_'):
                    parameter_refs.append(match)
                    break  # Found a match, no need to check other function names
        
        return parameter_refs
    
    #
    def _find_parameter_by_name(self, par_name, all_parameters):
        """
        Find a parameter by name in the list of all parameters
        """
        # Search through all parameters
        for par in all_parameters:
            if par.name == par_name:
                return par
        return None
    
    #
    def _evaluate_time_dependent_expression(self, t_ind, all_parameters, update_t_model=True):
        """
        Evaluate expression using time-dependent values of referenced parameters
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
            return aeval(self.expr_string, namespace)
        except Exception as e:
            raise ValueError(f"Error evaluating expression '{self.expr_string}': {e}")
    
    #
    def _get_all_parameters(self):
        """
        Get all parameters from the parent model
        """
        if hasattr(self, 'parent_model') and self.parent_model:
            return self.parent_model._get_all_parameters()
        return []

#
#
class Dynamics(Model):
    """
    <Model> (superclass) object used to describe time dependence of a <par> 
    from the parent model <parent_model>
    
    Individual functions defining rise and decay dynamics are <components>
    
    Time axis may be divided into cycles (defined by <frequency>) and 
    subcycles defined by "number of models (loaded into a dynamics instance) -1"
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
        Set the frequency of the temporal dynamics model
        [disabled] unit of time axis is 1E[<time_unit>] seconds
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
    def normalize_time(self, time_unit=0, DEBUG=False):
        """
        'Normalize' a <time> axis (unit: 1E[<time_unit>]s) such that it
        will repeat in T=1/<frequency> intervals where <frequency> is 
        the repetition frequency.
        Every T=1/<frequency> cycle is divided into <N> subcycles.
        (N > 0, where N=1 -> no subdivision)

        Creates normalized time (t_norm), the subcycle number (N_sub) 
        for every time step, and a cummulative subcycle counter 
        (N_counter) for every time step (numpy arrays each).
        """
        # sanity check
        if self.subcycles == 1 or not isinstance(self.subcycles, int): 
            print('Subcycle (N) must either be zero or a >= 2 integer')
            self.time_norm = None; self.N_sub = None; self.N_counter = None
        if self.frequency < 0 and self.frequency != -1:
            print('Frequency (f) must be >0 (or "-1" i.e. no repetition)')
            self.time_norm = None; self.N_sub = None; self.N_counter = None

        # no repetition within data/ time window
        if self.frequency == -1:
            if self.subcycles > 1: print('Define a (>0) frequency to use subcycles (N)')
            self.time_norm = np.asarray(self.time)
            self.N_sub = np.zeros(len(self.time))
            self.N_counter = np.zeros(len(self.time))

        # frequency >0 is passed
        else:
            # compute repetition/normalization number
            norm = 10**(-time_unit) /self.frequency /self.subcycles 
            t_norm = []; N_sub = []; N_counter = [] # initialize
            # go through time axis and perform normalization
            for i, t_i in enumerate(self.time):
                N_temp = math.floor(t_i /norm)
                if t_i >=0: # subcycles start at t=0
                    N_sub.append(math.floor(N_temp%self.subcycles) +1) # which subcycle is active
                    N_counter.append(N_temp +1) # increments by 1 each subcycle
                    t_norm.append(t_i -N_temp*norm) # each subcycle starts with t=0
                else: # times t<0 are baseline/pre-trigger/ground state spectra
                    N_sub.append(0); N_counter.append(0); t_norm.append(0)
            self.time_norm = np.asarray(t_norm)
            self.N_sub = np.asarray(N_sub)
            self.N_counter = np.asarray(N_counter)
            
        if DEBUG == True:
            legends = ['normalized time', 'subcycle counter', 'cummulative counter']
            uplt.plot_1D(data = [self.time_norm, self.N_sub, self.N_counter], 
                         x = self.time, xlabel = f'Time (1E{time_unit}s)',
                         ytype = 'log', legend = legends)
        #
        return None

#
#
#