"""
Helper functions for lmfit parameter handling and result management.

This module provides utilities for:
- Creating and constructing lmfit.Parameter objects
- Extracting parameter values from various lmfit objects
- Converting lmfit results to pandas DataFrames for analysis
- Managing MCMC sampling configuration
- Compatibility with scipy.optimize workflows
"""

import lmfit
import numpy as np
import pandas as pd

#
# lmfit parameter creation and extraction
#

#
def par_create(par_name, par_info, prefix='', suffix='', debug=False):
    """
    Create lmfit.Parameter object with optional name modifiers.
    
    Convenience wrapper for creating lmfit parameters that handles both
    standard parameters (value, vary, min, max) and expression-based
    parameters with automatic name prefix/suffix support.
    
    Parameters
    ----------
    par_name : str
        Base parameter name
    par_info : list
        Parameter specification, either:
        - [value, vary, min, max] for standard parameter
        - [value, vary] for unbound fit parameter
        - [expr_string] for expression-based parameter
    prefix : str, default=''
        String to prepend to parameter name
    suffix : str, default=''
        String to append to parameter name
    debug : bool, default=False
        If True, print parameter name and info during creation
    
    Returns
    -------
    lmfit.Parameter
        Configured parameter object
    """
    # Assemble parameter name
    par_str = prefix + par_name + suffix
    if debug:
        print(par_str)
        print(par_info)
    
    # Create lmfit.Parameter object
    lmf_par = lmfit.Parameter(par_str)
    
    # Standard parameter: [value, vary, min, max]
    if len(par_info) == 4:
        lmf_par.set(*par_info)
    # Unbound fit parameter: [value, vary]
    elif len(par_info) == 2:
        lmf_par.set(par_info[0], par_info[1], -np.inf, np.inf)
    # Expression parameter: [expr_string]
    elif len(par_info) == 1:
        if debug:
            print('expr=' + par_info[0])
        try:
            lmf_par.set(expr=par_info[0])
        except Exception as e:
            print(f'Exception while adding expression {par_info[0]} '
                  f'to parameter {par_str}: {e}')
    #
    return lmf_par

#
def par_extract(lmfit_pars, return_type='list'):
    """
    Extract parameter values from lmfit objects.
    
    Converts various lmfit parameter representations into a simple list
    of values or scipy-compatible format. Handles Parameters objects,
    MinimizerResult objects, lists, dicts, and numpy arrays.
    
    Parameters
    ----------
    lmfit_pars : lmfit.Parameters, lmfit.MinimizerResult, list, dict, or ndarray
        Parameter source to extract from:
        - lmfit.Parameters: Extract current values
        - lmfit.MinimizerResult: Extract optimized values
        - list: Pass through directly (list of values)
        - dict: Extract first element of each value (format: {name: [val, vary, min, max]})
        - ndarray: Convert to list
    return_type : {'list', 'par.x'}, default='list'
        Output format:
        - 'list': Return Python list of values
        - 'par.x': Return par_dummy object with .x attribute (scipy compatible)
    
    Returns
    -------
    list or par_dummy
        Parameter values in requested format
    
    Examples
    --------
    >>> # From lmfit.Parameters
    >>> params = lmfit.Parameters()
    >>> params.add('a', value=1.5)
    >>> params.add('b', value=2.0)
    >>> par_extract(params)
    [1.5, 2.0]
    
    >>> # From fit result
    >>> result = minimize(residual, params, ...)
    >>> par_extract(result)
    [1.523, 1.987]
    
    >>> # From list (passthrough)
    >>> par_extract([1.5, 2.0, 3.0])
    [1.5, 2.0, 3.0]
    
    >>> # From dict (initial guess format)
    >>> par_dict = {'a': [1.5, True, 0, 5], 'b': [2.0, True, 0, 10]}
    >>> par_extract(par_dict)
    [1.5, 2.0]
    
    >>> # scipy-compatible format
    >>> par_obj = par_extract(params, return_type='par.x')
    >>> par_obj.x
    [1.5, 2.0]
    """
    # lmfit.Parameters object
    if isinstance(lmfit_pars, lmfit.parameter.Parameters):
        pars_dict = lmfit_pars.valuesdict()
        pars = [v for k, v in pars_dict.items()]
    
    # List of values (passthrough)
    elif isinstance(lmfit_pars, list):
        pars = lmfit_pars
    
    # Initial guess dictionary: {name: [value, vary, min, max]}
    elif isinstance(lmfit_pars, dict):
        pars = [v[0] for k, v in lmfit_pars.items()]
    
    # Numpy array
    elif isinstance(lmfit_pars, np.ndarray):
        pars = lmfit_pars.tolist()
    
    # lmfit.MinimizerResult object
    else:
        pars = [lmfit_pars.params[p].value for p in lmfit_pars.params]
    
    # Return in requested format
    if return_type == 'list':
        return pars
    elif return_type == 'par.x':
        pars_scipy = par_dummy()
        pars_scipy.x = pars
        return pars_scipy
    
#
def par_construct(par_names, par_info):
    """
    Construct lmfit.Parameters object from lists.
    
    Batch version of par_create that builds a complete Parameters object
    from parallel lists of names and parameter specifications.
    
    Parameters
    ----------
    par_names : list of str
        Parameter names
    par_info : list of list
        Parameter specifications, one per name. Each element is either:
        - [value, vary, min, max] for standard parameter
        - [value, vary] for unbound fit parameter
        - [expr_string] for expression-based parameter
    
    Returns
    -------
    lmfit.Parameters
        Complete Parameters object with all parameters added
    """
    # Initialize Parameters object
    lmf_pars = lmfit.Parameters()
    
    # Add parameters one by one
    for par_name, p_info in zip(par_names, par_info):
        if len(p_info) == 4:  # [value, vary, min, max]
            lmf_pars.add(par_name, *p_info)
        elif len(p_info) == 2: # [value, vary]
            lmf_pars.add(par_name, p_info[0], p_info[1], -np.inf, np.inf)
        elif len(p_info) == 1:  # [expr]
            lmf_pars.add(par_name, expr=p_info[0])
    #
    return lmf_pars

#
# Result conversion to pandas DataFrames
#

#
def conf_interval2df(ci, CI_cols):
    """
    Convert lmfit confidence interval results to pandas DataFrame.
    
    Transforms the nested dictionary structure returned by lmfit.conf_interval
    into a tabular DataFrame format suitable for display and saving.
    Each row contains parameter name followed by values at different sigma levels.
    The confidence interval values represent parameter bounds at each sigma level.
    
    Parameters
    ----------
    ci : dict
        Confidence interval results from lmfit.conf_interval.
        Structure: {param_name: [(sigma, value), ...]}
    CI_cols : list of str
        Column headers for the output DataFrame.
        Typically: ['par[v]/sigma[>]', '-3', '-2', '-1', 'best fit', '+1', '+2', '+3']
    
    Returns
    -------
    pd.DataFrame
        DataFrame with rows=parameters, columns=sigma levels
    
    Examples
    --------
    >>> # After running confidence interval calculation
    >>> ci, trace = lmfit.conf_interval(minimizer, result, sigmas=[1,2,3], trace=True)
    >>> CI_cols = ['parameter', '-3', '-2', '-1', 'best', '+1', '+2', '+3']
    >>> df = conf_interval2df(ci, CI_cols)
    >>> df.to_csv('confidence_intervals.csv', index=False)
    """
    conf_CIs_list = []
    
    for param_name, values in ci.items():
        conf_par_CIs = [param_name]  # Start with parameter name
        
        # Extract parameter values at each sigma level
        # values is list of (sigma_percentage, param_value) tuples
        for val in values:
            conf_par_CIs.append(val[1])  # val[1] is the parameter value
        
        conf_CIs_list.append(conf_par_CIs)
    #
    return pd.DataFrame(data=conf_CIs_list, columns=CI_cols)

#
def par2df(lmfit_params, col_type, par_names=None):
    """
    Convert lmfit.Parameters object to pandas DataFrame.
    
    Extracts parameter information into tabular format for easy display,
    analysis, and saving. Supports different column sets for initial
    guesses vs. fit results.
    
    Parameters
    ----------
    lmfit_params : lmfit.Parameters or lmfit.MinimizerResult
        Parameters object or fit result to convert
    col_type : str or list of str
        Column selection:
        - 'ini': Initial guess columns ['name', 'value', 'vary', 'min', 'max', 'expr']
        - 'min': Fit result columns ['name', 'value', 'stderr', 'init_value', 
          'min', 'max', 'vary', 'expr']
        - list: Custom list of attribute names to extract
    par_names : list of str, optional
        Subset of parameter names to include. If None, includes all parameters.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with rows=parameters, columns=attributes
    
    Examples
    --------
    >>> # Initial guess parameters
    >>> params = lmfit.Parameters()
    >>> params.add('amplitude', value=10, vary=True, min=0, max=100)
    >>> df = par2df(params, col_type='ini')
    >>> df.to_csv('initial_parameters.csv', index=False)
    
    >>> # Fit results
    >>> result = minimize(residual, params, ...)
    >>> df = par2df(result.params, col_type='min')
    >>> print(df[['name', 'value', 'stderr']])
    
    >>> # Custom columns
    >>> df = par2df(params, col_type=['name', 'value', 'vary'])
    
    >>> # Subset of parameters
    >>> df = par2df(params, col_type='ini', par_names=['amplitude', 'center'])
    
    Notes
    -----
    Relative error (value/stderr*100) not included but easily computed from output.
    """
    # Select all parameters if none specified
    if par_names is None:
        par_names = list(lmfit_params.keys())
    
    # Define columns based on type
    if col_type == 'ini':
        cols = ['name', 'value', 'vary', 'min', 'max', 'expr']
    elif col_type == 'min':
        cols = ['name', 'value', 'stderr', 'init_value', 'min', 'max', 'vary', 'expr']
    else:
        cols = col_type
    
    # Extract parameter attributes
    par_info_list = []
    for par_name in par_names:
        par_info = []
        for col in cols:
            par_info.append(getattr(lmfit_params.get(par_name), col))
        par_info_list.append(par_info)
    #
    return pd.DataFrame(data=par_info_list, columns=cols)

#
def list_of_par2df(results):
    """
    Extract parameter values from multiple fit results into DataFrame.
    
    Collects optimized parameter values from a list of lmfit fit results
    (e.g., from slice-by-slice fitting) and organizes them in a DataFrame
    with rows=fits and columns=parameters.
    Assumes all fits have the same parameter names (typical for slice-by-slice).
    Parameter names are extracted from the first result.
    
    Parameters
    ----------
    results : list
        List of fit results from fit_wrapper or similar.
        Each element is expected to be a tuple/list where element [1] contains
        the lmfit.MinimizerResult with a .params attribute.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with rows=individual fits, columns=parameter values
    
    Examples
    --------
    >>> # After slice-by-slice fitting
    >>> results_list = []
    >>> for spectrum in data_2D:
    ...     result = fit_wrapper(spectrum, ...)
    ...     results_list.append(result)
    >>> 
    >>> df = list_of_par2df(results_list)
    >>> df.columns
    Index(['amplitude', 'center', 'width', ...], dtype='object')
    >>> 
    >>> # Plot parameter evolution
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(df['center'])
    >>> plt.xlabel('Slice number')
    >>> plt.ylabel('Peak center')
    """
    # Extract parameter values from each result
    param_values_list = [par_extract(results[i][1].params) for i in range(len(results))]
    
    # Get parameter names from first result (all should be identical)
    param_names = [k for k, v in results[0][1].params.valuesdict().items()]
    #
    return pd.DataFrame(param_values_list, columns=param_names)

#
# Configuration and compatibility classes
#

#
#
class MC:
    """
    Configuration for lmfit.emcee MCMC sampling.
    
    Stores settings for Markov Chain Monte Carlo parameter space exploration
    using lmfit's emcee wrapper. Provides a convenient way to manage and pass
    MCMC configuration settings.
    
    Parameters
    ----------
    useMC : int, default=0
        MCMC usage flag:
        - 0: Don't use MCMC
        - 1: Always use MCMC
        - 2: Use MCMC if conf_interval fails
    steps : int, default=5000
        Number of MCMC steps per walker
    nwalkers : int, default=1
        Number of MCMC walkers (should be >> number of parameters)
    burn : int, default=0
        Number of burn-in steps to discard (default 500 if starting far from optimum)
    thin : int, default=1
        Thinning factor (keep every Nth sample, default 20 for independence)
    ntemps : int, default=1
        Number of temperatures for parallel tempering
    workers : int, default=1
        Number of parallel workers (1 = serial)
    is_weighted : bool, default=False
        Whether to use weighted samples
    
    Attributes
    ----------
    use_emcee : int
        MCMC usage flag
    steps : int
        Number of MCMC steps
    nwalkers : int
        Number of walkers
    burn : int
        Burn-in steps
    thin : int
        Thinning factor
    ntemps : int
        Temperature levels
    workers : int
        Parallel workers
    is_weighted : bool
        Use weighted samples
    
    Examples
    --------
    >>> # Basic MCMC settings
    >>> mc_config = MC(useMC=1, steps=10000, nwalkers=50)
    >>> result = fit_wrapper(..., MCsettings=mc_config)
    
    >>> # Parallel tempering with multiple workers
    >>> mc_config = MC(useMC=1, steps=5000, nwalkers=100, 
    ...                ntemps=10, workers=4)
    
    >>> # Use MCMC as fallback if conf_interval fails
    >>> mc_config = MC(useMC=2, steps=5000, nwalkers=50)
    
    Notes
    -----
    - nwalkers should be at least 2 * n_parameters
    - burn-in needed if starting point far from optimum (set burn=0 if starting from fit)
    - thin > 1 reduces autocorrelation in samples
    - workers > 1 enables parallel sampling (requires multiprocessing support)
    
    See Also
    --------
    lmfit.emcee : lmfit's MCMC wrapper
    emcee : Underlying MCMC library
    """
    #
    def __init__(self, useMC=0, steps=5000, nwalkers=1, burn=0, thin=1,
                 ntemps=1, workers=1, is_weighted=False):
        self.use_emcee = useMC
        self.steps = steps
        self.nwalkers = nwalkers
        self.burn = burn
        self.thin = thin
        self.ntemps = ntemps
        self.workers = workers
        self.is_weighted = is_weighted

#
#
class par_dummy:
    """
    Dummy parameter object for scipy.optimize compatibility.
    
    Mimics the structure of scipy.optimize.minimize result objects to allow
    uniform handling of initial guesses and fit results. Useful for displaying
    initial parameter guesses using the same code that handles fit results.
    
    Attributes
    ----------
    final_simplex : None
        Final simplex (placeholder)
    fun : None
        Objective function value (placeholder)
    message : None
        Optimization message (placeholder)
    nfev : None
        Number of function evaluations (placeholder)
    nit : None
        Number of iterations (placeholder)
    status : None
        Optimization status (placeholder)
    success : bool
        Optimization success flag (always True for dummy)
    x : None or array
        Parameter values (set by par_extract when return_type='par.x')
    
    Examples
    --------
    >>> # Create dummy result for initial guess
    >>> params_init = par_extract(initial_params, return_type='par.x')
    >>> params_init.x
    [10, 5.0, 1.0]
    
    >>> # Can now use same plotting code for initial guess and fit result
    >>> plot_parameters(params_init)  # Initial guess
    >>> result = minimize(...)
    >>> plot_parameters(result)        # Fit result
    """
    #
    def __init__(self):
        self.final_simplex = None
        self.fun = None
        self.message = None
        self.nfev = None
        self.nit = None
        self.status = None
        self.success = True
        self.x = None