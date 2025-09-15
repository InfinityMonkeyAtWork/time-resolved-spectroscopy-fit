#
# helper functions for lmfit package and scipy.optimize(/minimize)
#
from fileinput import isfirstline
import lmfit
import numpy as np
import pandas as pd

#
# lmfit add-on functions
#
def par_create(par_name, par_info, prefix='', suffix='', DEBUG=0):
    """
    Create lmfit.Parameter object based on <par_name> and <par_info>
    lmfit.Parameter (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
    par_names: list of names of fit parameters
    par_info: list containing one list per parameter of 
              EITHER [initial guess, vary/fixed(True/False),
                      minimum boundary, maximum boundary]
              OR     [condition] passed as expr=<condition>.
    """
    # assemble parameter name
    par_str = prefix +par_name +suffix
    if DEBUG == 1: print(par_str); print(par_info)
    
    # create lmfit.Paramter object
    lmf_par = lmfit.Parameter(par_str)
    
    # (I) 1st option:
    # add parameter name, value, vary, min, max 
    if len(par_info) == 4:
        lmf_par.set(*par_info)
    # (II) 2nd option:
    # add parameter name and expression (expr=[condition])
    elif len(par_info) == 1:
        if DEBUG == 1: print('expr=' +par_info[0])
        try: lmf_par.set(expr=par_info[0])
        except: print(f'exception while adding expression {par_info[0]} to parameter {par_str}')
    #
    return lmf_par

#
def par_extract(lmfit_pars, return_type='list'):
    """
    input: lmfit.Minimizer() output or intial guess (list of items like
           [initial guess, vary/fixed (bool), min bound, max bound]) or
           initial guess in the form of a lmfit.parameter.Parameters.           
    output: either list (return_type='list') of parameter values or
            scipy.optimize.minimize parameter fit result type output
            where the values are in par.x (return_type='par.x')
    """
    # lmfit parameter case
    if isinstance(lmfit_pars, lmfit.parameter.Parameters):
        # get values as dictionary from lmfit.parameter.Parameters
        pars_dict = lmfit_pars.valuesdict()
        # get values from dictionary
        pars = [v for k, v in pars_dict.items()]
        
    #$% [deprecated!] initial guess list case
    elif isinstance(lmfit_pars, list):
        # check if actually [value, vary/fix, min, max] per parameter
        if isinstance(lmfit_pars[0], list):
            pars = [p[0] for p in lmfit_pars]
        else: # if pars is a list of par values just pass the input 
            pars = lmfit_pars
    
    # initial guess dictionary case
    elif isinstance(lmfit_pars, dict):
        pars = [v[0] for k, v in lmfit_pars.items()]
    
    # array containing par vals
    elif isinstance(lmfit_pars, np.ndarray): 
        pars = lmfit_pars.tolist()
        
    # lmfit fit result case
    else:
        pars = [lmfit_pars.params[p].value for p in lmfit_pars.params]
    
    # return
    if return_type == 'list':
        return pars
    elif return_type == 'par.x':
        pars_scipy = par_dummy()
        pars_scipy.x = pars
        return pars_scipy

#
def par_construct(par_names, par_info):
    """
    Returns an lmfit.Parameters object 
    [instead of individual lmfit.Parameter] 
    (see par_create instead for lmfit.Parameter [singular] object)
    """
    # initialize parameters
    lmf_pars = lmfit.Parameters()
    
    # add parameters one by one
    for pi, p_info in enumerate(par_info):
        if len(p_info) == 4: # value, vary, min, max
            lmf_pars.add(par_names[pi], *p_info)
        elif len(p_info) == 1: # expr=[condition]
            lmf_pars.add(par_names[pi], expr=p_info[0])
    #
    return lmf_pars

#
def conf_interval2df(ci, CI_cols):
    """
    conf_interval returns <ci>, a dictionary with key = "parameter name"
    and values: list of (sigma percentage, corresponding parameter value)
    convert this dict to a df that can be "display"ed and save as a csv
    
    CI_cols are the column headers of the dataframe that is returned.
    """
    conf_CIs_list = [] # list that will be filled from dict
    
    for k, vals in ci.items():

        conf_par_CIs = [] # initialize row for this parameter
        conf_par_CIs.append(k) # parameter name

        for val in vals:
            # val = (sigma percentage cutoff, corresponding par value)
            conf_par_CIs.append(val[1])

        # add this row (i.e. parameter) to the list
        conf_CIs_list.append(conf_par_CIs)

    # convert list to df and return
    return pd.DataFrame(data=conf_CIs_list, columns=CI_cols)

#
def par2df(lmfit_params, col_type, par_names=None):
    """
    par_names=None is better than the mutable par_names=[]
    <lmfit_params> could be a minimizer result or any other 
    object/instance of class "lmfit.parameter.Parameters"
    
    <col_type> defines which set of columns is returned, pass either
    - your own list containing elements (str) that should be returned
    - 'ini' (intial guess) to return
      ['name', 'value', 'vary', 'min', 'max', 'expr']
    - 'min' (minimizer result) to return
      ['name', 'value', 'standard error', 'relative error', 
       'initial value', 'min', 'max', 'vary', 'expr']
    
    <par_names> is a list of parameter names (rows) to be returned
    If empty list is passed (default) then select all parameters
    
    Note:
    relative error (df['value']/df['stderr']*100) not saved explicitly
    """
    # if empty list is passed as <par_names> then select all parameters
    if par_names is None:
    #if not par_names: # par_names=[]
        par_names = list(lmfit_params.keys())
    
    # define columns that will be queried
    if col_type == 'ini':
        cols = ['name', 'value', 'vary', 'min', 'max', 'expr']
    elif col_type == 'min':
        cols = ['name', 'value', 'stderr', 'init_value', 'min', 'max',
                'vary', 'expr']
    else: cols = col_type
        
    # go through all parameter names (rows)
    par_info_list = [] # initialize
    for par_name in par_names:
        # go through all attributes (columns)
        par_info = [] # initialize
        for col in cols:
            par_info.append(getattr(lmfit_params.get(par_name), col))
        # append this parameter to main list
        par_info_list.append(par_info)
    
    # convert list to dataframe and return
    return pd.DataFrame(data=par_info_list, columns=cols)

#
def list_of_par2df(results):
    """
    Extract parameter values from a list of lmfit results
    See XPSfit package, function: lmfit_fit_wrapper for the structure
    of the "results" input
    """
    # get list of parameters from result
    lst = [par_extract(results[i][1].params) for i in range(len(results))]
    
    # get the name of the parameters
    # (from first fit results only [all are identical])
    cols = [k for k, v in results[0][1].params.valuesdict().items()]
    
    return pd.DataFrame(lst, columns=cols)

#
# lmfit.emcee helper class
#
class MC:
    """
    Settings for lmfit.emcee
    """
    def __init__(self, useMC=0, steps=5000, nwalkers=1, burn=0, thin=1,
                 ntemps=1, workers=1, is_weighted=False):
        self.use_emcee = useMC 
        self.steps = steps
        self.nwalkers = nwalkers
        self.burn = burn # default 500
        self.thin = thin # default 20
        self.ntemps = ntemps
        self.workers = workers
        self.is_weighted = is_weighted
        #
        return None

#
# scipy.optimize (minimize)
#
class par_dummy:
    """
    Object that represents the fit result parameter returned from
    scipy.optimize.minimize. Use this to avoid unneccessary case
    handling when only showing the initial guess before actually
    fitting data
    """
    def __init__(self):
        self.final_simplex = None
        self.fun = None
        self.message = None
        self.nfev = None
        self.nit = None
        self.status = None
        self.success: True
        self.x = None
        #
        return None

#
#
#