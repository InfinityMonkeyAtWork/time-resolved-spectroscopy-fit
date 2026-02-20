"""
1D and 2D peak fitting functions based on lmfit.

This module provides the complete fitting workflow for spectroscopy data:
- Residual calculation for optimization
- Fitting with lmfit (including global + local optimization)
- Confidence interval estimation (lmfit.conf_interval)
- MCMC sampling via lmfit.emcee for uncertainty quantification
- Result visualization and export

The fitting functions here work with the MCP (Model/Component/Parameter) system
from trspecfit.mcp to provide a flexible, component-based fitting framework.

Key Functions
-------------
residual_fun : Compute residual for optimizer
fit_wrapper : Main fitting function with CI and MCMC support
plt_fit_res_1D : Plot 1D fit results with residuals
plt_fit_res_2D : Plot 2D fit results with residual maps
"""

import lmfit
from lmfit.minimizer import MinimizerResult
from trspecfit.utils import lmfit as ulmfit
#import emcee
import corner
import math
import numpy as np
import pandas as pd
import os
import copy
import time
from IPython.display import display, display_pretty
import matplotlib.pyplot as plt
from trspecfit.utils import plot as uplt
from trspecfit.config.plot import PlotConfig
from typing import Any, Optional, Sequence, Union, cast
import pathlib
from numpy.typing import ArrayLike

PathLike = Union[str, pathlib.Path]

def _result_params(result: MinimizerResult) -> lmfit.parameter.Parameters:
    """Return typed lmfit parameters from a MinimizerResult-like object."""
    params = getattr(result, "params", None)
    if not isinstance(params, lmfit.parameter.Parameters):
        raise TypeError(
            f"Expected lmfit.Parameters in result.params, got {type(params).__name__}"
        )
    return params


def _result_errorbars(result: MinimizerResult) -> bool:
    """Safely read MinimizerResult.errorbars with fallback."""
    return bool(getattr(result, "errorbars", False))


# Changes:
# - residual_fun: pass function instead of package + function 
# - rewrite a peak finder using ML
# - move from 0/1 to False/True whereever there are only 2 options

#
def residual_fun(
    par: Any,
    x: ArrayLike,
    data: np.ndarray,
    package: Any,
    fit_fun_str: str,
    unpack: int = 0,
    e_lim: Optional[list[int]] = None,
    t_lim: Optional[list[int]] = None,
    res_type: str = 'lmfit',
    args: Optional[Sequence[Any]] = None
) -> Union[np.ndarray, float]:
    """
    Compute residual (data - fit) for optimization and analysis.
    
    This function is called repeatedly by optimizers to evaluate how well
    current parameters fit the data. It supports both 1D and 2D fitting,
    with options for limiting the fitting region and returning different
    residual representations.
    
    Parameters
    ----------
    par : list or lmfit.Parameters
        Current parameter values. If lmfit.Parameters, values are extracted.
        Order must match the fit function's parameter expectations.
    x : ndarray
        Independent variable axis (energy or time). Passed to fit function
        but may not be used if function has its own axes.
    data : ndarray (1D or 2D)
        Experimental data to fit. Shape determines dimensionality:

        - 1D: [n_energy] for energy-resolved fits
        - 2D: [n_time, n_energy] for time- and energy-resolved fits

    package : module
        Python module containing the fit function (typically trspecfit.spectra)
    fit_fun_str : str
        Name of fit function within package (e.g., 'fit_model_mcp')
    unpack : {0, 1}, default=0
        Parameter passing mode:

        - 0: Pass parameters as list: ``fit_fun(x, par, ...)``
        - 1: Unpack parameters: ``fit_fun(x, *par, ...)``

    e_lim : list of int, default=[]
        Energy axis limits [left, right] for residual calculation.
        Uses slice notation: data[e_lim[0]:-e_lim[1]]
        Empty list uses full energy range.
    t_lim : list of int, default=[]
        Time axis limits [start, stop] for residual calculation.
        Uses slice notation: data[t_lim[0]:t_lim[1]]
        Empty list uses full time range.
    res_type : {'lmfit', 'RSS', 'abs', 'res', 'fit'}, default='lmfit'
        Return type:

        - 'lmfit': Residual array (1D) or flattened residual (2D) for lmfit
        - 'RSS': Residual sum of squares (scalar, for scipy.optimize)
        - 'abs': Sum of absolute residuals (scalar, L1 norm)
        - 'res': Raw residual array (data - fit)
        - 'fit': Return fit itself instead of residual

    args : tuple, default=()
        Additional arguments for fit function, passed via ``*args``
    
    Returns
    -------
    ndarray or float
        Return type depends on res_type:
        - 'lmfit': ndarray (1D for 1D data, flattened for 2D data)
        - 'RSS', 'abs': float (scalar metric)
        - 'res': ndarray (same shape as data)
        - 'fit': ndarray (same shape as data)
    """
    if e_lim is None:
        e_lim = []
    if t_lim is None:
        t_lim = []
    if args is None:
        args = ()

    # define the fit function
    fit_fun = getattr(package, fit_fun_str)
    
    # if the minimizer calling this is from the lmfit package, then
    # extract the value from their lmfit.Parameter() (dictionary)
    # or if list of [value, vary(, min, max)] transition to val list
    par = ulmfit.par_extract(par, return_type='list')
    
    # compute the fit curve [plot_ind has to be hardcoded as False/0 here]
    if unpack == 1:
        fit = fit_fun(x, *par, 0, *args)
    elif unpack == 0:
        fit = fit_fun(x, par, 0, *args)
    else:
        raise ValueError("unpack must be 0 or 1")
    fit_arr = np.asarray(fit)
    data_arr = np.asarray(data)
    
    # select user-defined region to consider for residual computation
    if len(data_arr.shape) == 1: # 1D data
        if len(e_lim) != 0:
            residual = data_arr[e_lim[0]:-e_lim[1]] - fit_arr[e_lim[0]:-e_lim[1]]
        else: # use entire data and fit array to compute RSS 
            residual = data_arr - fit_arr
    elif len(data_arr.shape) == 2: # 2D data
        if (len(e_lim) != 0) and (len(t_lim) == 0):
            residual = data_arr[:, e_lim[0]:-e_lim[1]] - fit_arr[:, e_lim[0]:-e_lim[1]]
        elif (len(e_lim) == 0) and (len(t_lim) != 0):
            residual = data_arr[t_lim[0]:t_lim[1], :] - fit_arr[t_lim[0]:t_lim[1], :]
        elif (len(e_lim) != 0) and (len(t_lim) != 0):
            residual = data_arr[t_lim[0]:t_lim[1], e_lim[0]:-e_lim[1]] \
                       - fit_arr[t_lim[0]:t_lim[1], e_lim[0]:-e_lim[1]]
        # or use entire data and fit array to compute RSS
        else:
            residual = data_arr - fit_arr
    else:
        raise ValueError("data must be 1D or 2D")
    
    # type of residual to return
    if res_type == 'RSS':
        return float(np.sum(residual**2))
    elif res_type == 'abs':
        return float(np.sum(np.abs(residual)))
    elif res_type == 'res':
        return np.asarray(residual)
    elif res_type == 'lmfit':
        if len(data_arr.shape) == 1: # 1D data
            return np.asarray(residual)
        elif len(data_arr.shape) == 2: # 2D data
            return np.asarray(residual).flatten()
    elif res_type == 'fit':
        return fit_arr
    raise ValueError(f"Unknown res_type '{res_type}'")

#
def time_display(t_start: float, print_str: str = '', return_delta_seconds: bool = False) -> Optional[float]:
    """
    Display elapsed time in human-readable format.
    
    Computes time elapsed since t_start and displays it with appropriate
    units (seconds, minutes, hours, or days). Useful for benchmarking
    fitting operations.
    
    Parameters
    ----------
    t_start : float
        Start time from time.time()
    print_str : str, default=''
        Prefix string for display (e.g., 'Fitting completed in: ')
    return_delta_seconds : bool, default=False
        If True, also return elapsed seconds as float

    Returns
    -------
    None or float
        None normally, or elapsed seconds if return_delta_seconds=True
    
    Examples
    --------
    >>> import time
    >>> t0 = time.time()
    >>> # ... do some work ...
    >>> time_display(t0, 'Operation completed in: ')
    Operation completed in: 01:23.456(mm:ss.ms)
    """
    t_stop = time.time()
    seconds = t_stop - t_start
    
    str_format = 'ss.ms'
    minutes, seconds = divmod(seconds, 60)
    delta_format = f'{seconds:06.3f}'
    
    if minutes > 0:
        str_format = 'mm:' + str_format
        hours, minutes = divmod(minutes, 60)
        delta_format = f'{math.floor(minutes):02d}:{delta_format}'
        
        if hours > 0:
            str_format = 'hh:' + str_format
            days, hours = divmod(hours, 24)
            delta_format = f'{math.floor(hours):02d}' + ':' + delta_format
            
            if days > 0:
                str_format = 'ddd:' + str_format
                delta_format = f'{math.floor(days):03d}' + ':' + delta_format
        
    print(print_str + delta_format + f'({str_format})')
    #
    if return_delta_seconds:
        return seconds
    else:
        return None

#
# error estimation helper functions
#
def sigma_dict() -> dict[str, float]:
    """
    Get percentage of distribution within N-sigma intervals.
    
    Returns dictionary mapping sigma values (as strings) to the percentage
    of a normal distribution contained within ±N sigma of the mean.
    
    Returns
    -------
    dict
        Keys: Sigma values as strings ('0.5', '1.0', ..., '5.0')
        Values: Percentage of distribution (float)
    """
    sigma_dict = {'0.5' : 38.2924922548026,
                  '1.0' : 68.2689492137086,
                  '1.5' : 86.6385597462284,
                  '2.0' : 95.4499736103642,
                  '2.5' : 98.7580669348448,
                  '3.0' : 99.7300203936740,
                  '3.5' : 99.9534741841929,
                  '4.0' : 99.9936657516334,
                  '4.5' : 99.9993204653751,
                  '5.0' : 99.9999426696856
                 }
    return sigma_dict

#
def sigma_start_stop_percent(sigma_list: Sequence[float]) -> list[list[float]]:
    """
    Calculate percentile bounds for symmetric confidence intervals.
    
    For each sigma level, computes the lower and upper percentiles that
    define a symmetric interval containing the specified percentage of
    a normal distribution.
    
    Parameters
    ----------
    sigma_list : list of float
        Sigma values (e.g., [1, 2, 3] for 1σ, 2σ, 3σ intervals).
        Values must be in range [0.5, 5.0] in 0.5 increments.
    
    Returns
    -------
    list of [float, float]
        Percentile bounds for each sigma level.
        Each element: [lower_percentile, upper_percentile]
        Returns empty list if any sigma value not supported.
    
    Examples
    --------
    >>> # 1σ, 2σ, 3σ intervals
    >>> bounds = sigma_start_stop_percent([1.0, 2.0, 3.0])
    >>> print(bounds)
    [[15.87, 84.13], [2.28, 97.72], [0.14, 99.86]]
    
    >>> # Use with numpy.percentile on MCMC samples
    >>> import numpy as np
    >>> samples = np.random.normal(0, 1, 10000)
    >>> bounds_1sigma = sigma_start_stop_percent([1.0])[0]
    >>> ci = np.percentile(samples, bounds_1sigma)
    >>> print(f"1σ interval: [{ci[0]:.2f}, {ci[1]:.2f}]")
    1σ interval: [-1.01, 1.02]
    """
    borders_pc: list[list[float]] = []  # low/high borders in percent
    for sigma in sigma_list:
        a2a_total_raw = sigma_dict().get("{:.1f}".format(sigma), "sigma value not supported")
        if isinstance(a2a_total_raw, str):
            print(a2a_total_raw)
            borders_pc = []
        else:
            a2a_total = float(a2a_total_raw)
            A_exclude2A_total = 100 - a2a_total
            borders_pc.append([A_exclude2A_total/2, 100 -A_exclude2A_total/2])
    #
    return borders_pc

#
# wrapper for lmfit fit, confidence interval, and lmfit.emcee functions
#
def fit_wrapper(
    const: tuple[Any, ...],
    args: tuple[Any, ...],
    par_names: list[str],
    par: Any,
    fit_type: int,
    sigmas: Optional[list[float]] = None,
    try_CI: int = 1,
    MCsettings: Optional[ulmfit.MC] = None,
    fit_alg_1: str = 'Nelder',
    fit_alg_2: str = 'leastsq',
    show_info: float = 0,
    save_output: int = 0,
    save_path: PathLike = ''
) -> list[Any]:
    """
    if sigmas is None:
        sigmas = [1, 2, 3]
    if MCsettings is None:
        MCsettings = ulmfit.MC()

    Comprehensive fitting wrapper with optimization, CI, and MCMC.
    
    This is the main fitting function in trspecfit. It handles:
    - Single or two-stage optimization
    - Confidence interval estimation via lmfit.conf_interval
    - MCMC sampling via lmfit.emcee
    - Result visualization and export
    
    Two-stage fitting (fit_type=2) is recommended for robust optimization:
    first finds global minimum with Nelder-Mead, then refines locally with
    leastsq for accurate error bars.
    
    Parameters
    ----------
    const : tuple
        Constants for residual_fun:
        (x, data, package, function_str, unpack, e_lim, t_lim)
    args : tuple
        Arguments for fit function (passed to residual_fun):
        Typically (model, dim, debug) for MCP models
    par_names : list of str
        Parameter names in order (for display and export)
    par : lmfit.Parameters or list
        Initial parameter guess:

        - lmfit.Parameters: Use directly
        - list: Convert to lmfit.Parameters using par_names.
          Each element: ``[value, vary, min, max]`` or ``['expression']``

    fit_type : {0, 1, 2}
        Fitting strategy:

        - 0: No fit (return initial guess only, for debugging)
        - 1: Single fit with fit_alg_1
        - 2: Two-stage fit (global with fit_alg_1, local with fit_alg_2)

    sigmas : list of int or float, default=[1,2,3]
        Confidence levels for CI and MCMC (e.g., [1,2,3] for 1σ, 2σ, 3σ)
    try_CI : {0, 1}, default=1
        Confidence interval estimation:

        - 0: Skip CI calculation
        - 1: Calculate CI if error bars available (result.errorbars=True)

    MCsettings : ulmfit.MC, default=ulmfit.MC()
        MCMC configuration object:

        - use_emcee: 0 (skip), 1 (always), 2 (if CI fails)
        - steps: Number of MCMC steps per walker
        - nwalkers: Number of MCMC walkers
        - burn, thin, ntemps, workers, is_weighted: MCMC parameters

        See ulmfit.MC class for details
    fit_alg_1 : str, default='Nelder'
        First/only optimization method. Common options:

        - 'Nelder': Nelder-Mead (robust, no gradients, slower)
        - 'powell': Powell (robust, no gradients)
        - 'leastsq': Levenberg-Marquardt (fast, needs good initial guess)

        See lmfit documentation for full list
    fit_alg_2 : str, default='leastsq'
        Second optimization method (fit_type=2 only).
        Typically 'leastsq' for accurate local optimization and error bars.
    show_info : {0, 1, 2, 3}, default=0
        Verbosity level:

        - 0: Silent
        - 1: Basic timing and final results
        - 1.5: Also show initial parameters
        - 2: Also show constants and arguments

    save_output : {-1, 0, 1}, default=0
        Save results to files:

        - 0: Don't save
        - 1: Save all results (parameters, CIs, MCMC, plots)
        - -1: Same as 1 (for compatibility)

    save_path : str or Path, default=''
        Base path for saved files (without extension).
        Files saved: _par_ini.csv, _par_fin.txt, _par_fin.csv,
        _conf_CIs.csv, _emcee_fin.txt, _emcee_flatchain.csv,
        _emcee_CIs.csv, _emcee_walker_acceptance_ratio.png,
        _emcee_corner_plot.png
    
    Returns
    -------
    list
        Five-element list containing results:
        [par_ini, par_fin, conf_CIs, emcee_fin, emcee_CIs]
        
        par_ini : lmfit.Parameters
            Initial parameter guess
        par_fin : lmfit.MinimizerResult or []
            Final fit result from lmfit.minimize
            Empty list if fit_type=0
        conf_CIs : pd.DataFrame or pd.DataFrame()
            Confidence intervals from lmfit.conf_interval
            Columns: ['par[v]/sigma[>]', '-3σ', '-2σ', '-1σ', 'best', '+1σ', '+2σ', '+3σ']
            Empty DataFrame if CI not calculated/failed
        emcee_fin : lmfit.MinimizerResult or []
            MCMC result from lmfit.emcee
            Empty list if MCMC not used
        emcee_CIs : pd.DataFrame or pd.DataFrame()
            MCMC confidence intervals from quantiles of flatchain
            Same column structure as conf_CIs
            Empty DataFrame if MCMC not used
    
    Examples
    --------
    >>> # Basic single-stage fit
    >>> const = (energy, spectrum, spectra, 'fit_model_mcp', 0, [], [])
    >>> args = (model, 1, False)
    >>> results = fit_wrapper(
    ...     const=const,
    ...     args=args,
    ...     par_names=model.par_names,
    ...     par=model.lmfit_pars,
    ...     fit_type=1,
    ...     show_info=1
    ... )
    >>> par_ini, par_fin, conf_CIs, emcee_fin, emcee_CIs = results
    
    >>> # Two-stage fit with confidence intervals
    >>> results = fit_wrapper(
    ...     const=const,
    ...     args=args,
    ...     par_names=model.par_names,
    ...     par=model.lmfit_pars,
    ...     fit_type=2,
    ...     try_CI=1,
    ...     sigmas=[1, 2, 3],
    ...     show_info=1,
    ...     save_output=1,
    ...     save_path='fit_results/baseline_fit'
    ... )
    
    >>> # Fit with MCMC for uncertainty quantification
    >>> mc = ulmfit.MC(use_emcee=1, steps=5000, nwalkers=50)
    >>> results = fit_wrapper(
    ...     const=const,
    ...     args=args,
    ...     par_names=model.par_names,
    ...     par=model.lmfit_pars,
    ...     fit_type=2,
    ...     try_CI=1,
    ...     MCsettings=mc,
    ...     show_info=1
    ... )
    
    >>> # Debug mode (no fitting, just show initial guess)
    >>> results = fit_wrapper(
    ...     const=const,
    ...     args=args,
    ...     par_names=model.par_names,
    ...     par=model.lmfit_pars,
    ...     fit_type=0
    ... )
    
    Notes
    -----
    **Two-Stage Fitting:**
    Recommended approach for robust results:
    1. Nelder-Mead finds global minimum (no gradient needed)
    2. Levenberg-Marquardt refines locally (fast, accurate errors)
    
    This combination avoids local minima while providing reliable error estimates.
    
    **Error Estimation Methods:**
    - Confidence intervals (CI): Profile likelihood method, exact but slow
    - MCMC: Samples parameter space, handles correlations, provides full distributions
    - Use MCMC for complex models or when CI fails
    
    **MCMC Diagnostics:**
    When using MCMC, check:
    - Acceptance ratios (saved plot): Should be 0.2-0.5
    - Corner plot (saved): Should show well-defined peaks
    - Chain length: Increase steps if distributions look noisy
    
    **Performance Tips:**
    - Use fit_type=1 for quick fits during model development
    - Use fit_type=2 for final/publication fits
    - MCMC is slow (minutes for complex models) but provides best uncertainties
    
    **File Outputs:**
    When save_output=1:
    - CSV files: Comma-separated, easy to read in Excel/pandas
    - TXT files: Human-readable lmfit.fit_report format
    - PNG files: High-resolution plots for documentation
    """
    if sigmas is None:
        sigmas = [1.0, 2.0, 3.0]
    if MCsettings is None:
        MCsettings = ulmfit.MC()

    # construct the lmfit parameters if necessary
    if isinstance(par, lmfit.parameter.Parameters):
        par_ini = copy.deepcopy(par); prnt_str = 'passed in '
    else:
        par_ini = ulmfit.par_construct(par_names=par_names, par_info=par)
        prnt_str = 'converted to '
    if show_info >= 1.5: 
        print('Parameters '+prnt_str +'lmfit format:')
        display(par_ini)
    # convert par_ini to pandas dataframe and save all lmfit info
    df_par_ini = ulmfit.par2df(par_ini, 'ini', par_names)
    
    if show_info >= 2: # (optionally) show constants and args input
        print(); print('Constants input to residual function:')
        display_pretty(const)
        print('Arguments input to fit function:')
        display_pretty(args)
    if show_info >= 1: t_0 = time.time() # start time
        
    if fit_type == 0: 
        #$% deprecated: use plt_fit_res_1D/2D to show data +initial guess +residual
        # instead use file.model.describe() to see initial guess!
        if show_info >= 1:
            print(); print('Deprecated. Option will be removed.')
            print('Use file.model.describe() to see initial guess.')
            print('Returning initial parameters without fitting.')
        #
        return [par_ini, [], pd.DataFrame(), [], pd.DataFrame()]

    # construct lmfit minimizer
    mini = lmfit.Minimizer(residual_fun,
                           par_ini, 
                           fcn_args = const +('lmfit', ) +(args, )
                          )
    # perform fit(s)
    if show_info >= 1:
        t_ini = time.time()
        print(f'\nTime initialize: {t_ini -t_0} s')
    #
    if fit_type == 1: # one fit only
        par_fin = mini.minimize(method=fit_alg_1)
        par_fin_params = _result_params(par_fin)
        if show_info >= 1:
            print(); print(f'Results fit (method={fit_alg_1}): ') 
            lmfit.report_fit(par_fin_params)
            t_fit = time.time(); print(f'Time fit: {t_fit -t_ini} s')
    #
    if fit_type == 2: # find global minimum + local optimization
        par_fin_GM = mini.minimize(method=fit_alg_1)
        par_fin_GM_params = _result_params(par_fin_GM)
        if show_info >= 1:
            print()
            print(f'Results global minumum fit (method={fit_alg_1}): ')
            lmfit.report_fit(par_fin_GM_params)
            t_fit0 = time.time()
            print(f'Time fit (global minimum): {t_fit0 -t_ini} s')
        #
        par_fin = mini.minimize(method=fit_alg_2, params=par_fin_GM_params)
        par_fin_params = _result_params(par_fin)
        if show_info >= 1:
            print()
            print(f'Results local optimization fit (method={fit_alg_2}): ')
            lmfit.report_fit(par_fin_params)
            t_fit = time.time()
            print(f'Time fit (local optimization): {t_fit -t_fit0} s')
    
    
    # confidence intervals
    
    # define column headers for the confidence interval dataframes
    # (conf_interval and emcee)
    CI_cols = ['par[v]/sigma[>]'] +\
              ["-"+str(sigma) for sigma in sigmas[::-1]] +\
              ['best fit'] +\
              ["+"+str(sigma) for sigma in sigmas]
    
    # conf_interval (https://lmfit.github.io/lmfit-py/confidence.html)
    if try_CI == 1:
        if _result_errorbars(par_fin):
            ci_fin, trace_fin = lmfit.conf_interval(mini,
                                                    par_fin,
                                                    sigmas=sigmas,
                                                    trace=True)
            if show_info >= 1: 
                print(); lmfit.printfuncs.report_ci(ci_fin)
            # convert ci_fin to standard CI dataframe
            conf_CIs = ulmfit.conf_interval2df(ci_fin, CI_cols)
        else: 
            conf_CIs = pd.DataFrame()
            if show_info >= 1:
                print()
                print('No successful error bar determination via conf_interval')
            if MCsettings.use_emcee == 2: 
                # conf_interval didn't work -> use lmfit.emcee()
                MCsettings.use_emcee = 1
    elif try_CI == 0:
        conf_CIs = pd.DataFrame()

    # lmfit.emcee() [not a fit, it is a way to sample the parameter space!]
    if MCsettings.use_emcee == 1:
        t_emcee0 = time.time()
        par_fin_params = _result_params(par_fin)
        # make optional for user to pass value and min/max
        #$% rely on defaults here instead?
        par_fin_params.add('__lnsigma', value=np.log(0.1),
                           min=np.log(0.001), max=np.log(2))
        # always print progress bar
        print()
        print('Progress of lmfit.emcee confidence interval determination')
        print('(based on Markov chain Monte Carlo parameter space sampling):')
        # burn necessary if starting point not close to max(probability distribution)
        # i.e. not close to the optimized parameter set, so burn=0 is ok here!
        emcee_fin = mini.emcee(params=par_fin_params, 
                               steps=MCsettings.steps,
                               nwalkers=MCsettings.nwalkers, 
                               burn=MCsettings.burn,
                               thin=MCsettings.thin, 
                               ntemps=MCsettings.ntemps, 
                               workers=MCsettings.workers, 
                               is_weighted=MCsettings.is_weighted, 
                               progress=True
                               )
        emcee_fin_params = _result_params(emcee_fin)
        emcee_flatchain = cast(pd.DataFrame, getattr(emcee_fin, "flatchain", pd.DataFrame()))
        emcee_var_names = cast(list[str], getattr(emcee_fin, "var_names", []))
        emcee_lnprob = np.asarray(getattr(emcee_fin, "lnprob", np.array([])))
        emcee_chain = np.asarray(getattr(emcee_fin, "chain", np.array([])))
        emcee_acceptance_fraction = np.asarray(
            getattr(emcee_fin, "acceptance_fraction", np.array([]))
        )
        # lmfit.emcee() results
        if show_info >= 1:
            print()
            print('Results lmfit.emcee() confidence interval determination:')
            lmfit.report_fit(emcee_fin_params)
            t_emcee1 = time.time()
            print(f'Time lmfit.emcee: {t_emcee1-t_emcee0} s')
        # acceptence fraction of all walkers (plot)
        fig_emcee_walker, ax = plt.subplots(1, 1, dpi=75)
        plt.plot(emcee_acceptance_fraction, 'o')
        plt.xlabel('Walker number'); plt.ylabel('Acceptance fraction')
        if abs(save_output)==1:
            uplt.img_save(f'{save_path}_emcee_walker_acceptance_ratio.png')
        if show_info >= 1: plt.show()
        else: plt.close(fig_emcee_walker)
        # draw all combinations of the typically ellipsoidal chi plot
        # [<x=par1, y=par2, z=chi2> plot]
        emcee_truths = [emcee_fin_params.valuesdict().get(par_name)
                        for par_name in emcee_var_names]
        fig_emcee_corner = plt.figure(figsize=(10,10))
        emcee_corner = corner.corner(emcee_flatchain,
                                     labels=emcee_var_names,
                                     truths=emcee_truths,
                                     fig=fig_emcee_corner)
        if abs(save_output) == 1:
            uplt.img_save(f'{save_path}_emcee_corner_plot.png')
        if show_info >= 1: plt.show()
        else: plt.close(fig_emcee_corner)
        # find highest probability parameter combination
        highest_prob = np.argmax(emcee_lnprob)
        hp_loc = np.unravel_index(highest_prob, emcee_lnprob.shape)
        mle_soln = emcee_chain[hp_loc]
        # get percentage borders to categorize emcee.flatchain data
        sigma_borders = sigma_start_stop_percent(sigmas)
        # go through all combinations of parameters and sigmas to find
        # lmfit.emcee() confidence intervals
        emcee_CIs_list = [] # initialize results
        for par_name in par_names+['__lnsigma']:
            emcee_par_CIs: list[Any] = [par_name] # initialize results for this parameter
            if par_name in emcee_var_names:
                # get quantiles if fit parameter is variable
                for s, sigma_b in enumerate(sigma_borders):
                    # get cutoff values that meet this sigma threshold (+/-)
                    quantiles = np.percentile(emcee_flatchain[par_name],
                                              sigma_b)
                    # lower threshold; 0 is par_name
                    emcee_par_CIs.insert(1, quantiles[0])
                    # upper threshold
                    emcee_par_CIs.insert(len(emcee_par_CIs), quantiles[1])
            else: # pass a list of "-1" (int) as confidence intervals
                emcee_par_CIs.extend(2*len(sigmas)*[-1,])
            # append this line to list containing all parameters
            emcee_CIs_list.append(emcee_par_CIs)
        # convert confidence interval cutoffs to a dataframe 
        # and add the "best fit result" in the middle
        emcee_CIs = pd.DataFrame(data=emcee_CIs_list) 
        emcee_CIs.insert(loc=len(sigmas)+1, column='bla',
                         value=list(emcee_fin_params.valuesdict().values()))
        emcee_CIs.columns = CI_cols
        if show_info >= 1:
            print(display(emcee_CIs))
    else: # use_emcee equal to 0, or equal to 2 and conf_interval worked
        emcee_fin = None; emcee_CIs = pd.DataFrame()
    
    # optional save (figures are saved above)
    # [if statements check for empty list/dataframe]
    if abs(save_output) == 1:
        # par_ini (pandas DataFrame) as csv file
        df_par_ini.to_csv(str(save_path) +'_par_ini.csv', index=False)
        # par_fin as text dump
        if par_fin:
            with open(str(save_path) +'_par_fin.txt', 'w') as par_fin_file:
                par_fin_file.write(lmfit.fit_report(par_fin))
        # par_fin variables as csv file
        df_par_fin = ulmfit.par2df(_result_params(par_fin), 'min', par_names)
        df_par_fin.to_csv(str(save_path) +'_par_fin.csv', index=False)
        # conf_CIs using pandas as it is a pd.DataFrame
        if not conf_CIs.empty:
            conf_CIs.to_csv(str(save_path) +'_conf_CIs.csv', index=False)
        # emcee_fin (fit_report) as text dump, emcee flatchain as csv
        if emcee_fin is not None:
            with open(str(save_path) +'_emcee_fin.txt', 'w') as emcee_fin_file:
                emcee_fin_file.write(lmfit.fit_report(emcee_fin))
            emcee_flatchain = cast(
                pd.DataFrame, getattr(emcee_fin, "flatchain", pd.DataFrame())
            )
            emcee_flatchain.to_csv(f'{save_path}_emcee_flatchain.csv',
                                   index=False)
        # emcee_CIs using pandas as it is a pd.DataFrame
        if not emcee_CIs.empty:
            emcee_CIs.to_csv(str(save_path) +'_emcee_CIs.csv', index=False)
    #
    return [par_ini, par_fin, conf_CIs, emcee_fin, emcee_CIs]

#
# plotting fit results for Slice-by-Slice methods
#
def results_select(data: Any, skip: int = -1, N: int = -1, dim: int = 1) -> Any:
    """
    Select slice of results array for partial fitting analysis.
    
    Parameters
    ----------
    data : array-like
        Data array to slice
    skip : int, default=-1
        Number of initial elements to skip. -1 means skip none.
    N : int, default=-1
        Number of elements to include. -1 means include all (after skip).
    dim : int, default=1
        Dimensionality (currently only dim=1 supported).
    
    Returns
    -------
    array-like
        Sliced data
    
    Examples
    --------
    >>> # Get all data
    >>> results_select(data)
    
    >>> # Skip first 10 points
    >>> results_select(data, skip=10)
    
    >>> # Get first 50 points only
    >>> results_select(data, N=50)
    
    >>> # Get points 10-60
    >>> results_select(data, skip=10, N=60)
    """
    if dim == 1:
        if N == -1: 
            if skip == -1:
                return data # full data set
            else:
                out = data[skip:]
        else: # "+1" accounts for Python's exclusive upper bound 
            if skip == -1:
                out = data[:N+1]
            else:
                out = data[skip:N+1]
    else:
        raise ValueError(f"Unsupported dim={dim}; only dim=1 is implemented")

    return out

#
def results2df(
    results: list[Any],
    x: Optional[ArrayLike] = None,
    index: Optional[ArrayLike] = None,
    config: Optional[PlotConfig] = None,
    first_N_spec_only: int = -1,
    skip_first_N_spec: int = -1,
    save_df: int = 0,
    save_path: PathLike = ''
) -> pd.DataFrame:
    """
    Convert Slice-by-Slice fit results to DataFrame with parameter plots.
    
    Transforms list of fit results (from slice-by-slice fitting) into
    a pandas DataFrame with time/index as rows and parameters as columns.
    Optionally creates individual plots for each parameter vs. time.
    
    Parameters
    ----------
    results : list
        List of fit results from fit_wrapper, one per time slice.
        Each element: [par_ini, par_fin, conf_CIs, emcee_fin, emcee_CIs]
    x : array-like, optional
        Time axis values. If provided, included as column in DataFrame.
    index : array-like, optional
        Index values (e.g., slice numbers). If provided, included as column.
    config : PlotConfig, optional
        Plot configuration. If None, uses defaults.
    first_N_spec_only : int, default=-1
        If != -1, include only first N results (for partial fitting).
    skip_first_N_spec : int, default=-1
        If != -1, skip first N results (for partial fitting).
    save_df : {-1, 0, 1}, default=0
        Save outputs:
        
        - 0: Don't save
        - 1: Save DataFrame and parameter plots
        - -1: Same as 1
        
        When save_df != 0, plots only varied (not fixed) parameters
    save_path : str or Path, default=''
        Directory path for saving files (not full filename) (created if not exists).
        Files saved: 'fit_pars.csv', '{param_name}.png' for each parameter
    
    Returns
    -------
    pd.DataFrame
        Results with structure:
        - Columns: ['index', config.y_label, param1, param2, ...]
        - Rows: One per fitted slice
        - Values: Optimized parameter values
    """
    # Use default config if none provided
    if config is None:
        config = PlotConfig()
    
    # transform lmfit_wrapper results to dataframe
    df = ulmfit.list_of_par2df(results)
    # get columns names for plot before adding x/index
    cols_plt = df.columns
    
    # select x (time) and index data if passed
    if x is not None:
        x_save = results_select(data=x,
                                skip=skip_first_N_spec,
                                N=first_N_spec_only)
        df.insert(0, config.y_label, x_save) # and insert into dataframe
    if index is not None:
        ind_save = results_select(data=index,
                                  skip=skip_first_N_spec,
                                  N=first_N_spec_only)
        df.insert(0, "index", ind_save) # and insert into dataframe
    
    # get par_fin([1]) of first slice(index=0)
    # (their "vary" attribute is the same for all)
    df_par_fin_slice0 = ulmfit.par2df(lmfit_params=results[0][1].params,
                                        col_type='min')
    save_array = [-1 if vary==False else 1 for vary in df_par_fin_slice0['vary']]
    
    if save_df != 0:
        # save the dataframe (index, x axis, parameter1, parameter2, ...
        df.to_csv(os.path.join(save_path, 'fit_pars.csv'))
        # plot individual parameters as a function of time (s)
        plt_fit_res_pars(df=df.loc[:, list(cols_plt)], x=x_save if x is not None else None,
                         config=config, save_img=save_array, save_path=save_path)
    #
    return df

#
def results2fit2D(
    results: Union[list[Any], pd.DataFrame],
    const: tuple[Any, ...],
    args: tuple[Any, ...],
    num_fmt: str = '%.6e',
    delim: str = ',',
    save_2D: int = 0,
    save_path: PathLike = ''
) -> np.ndarray:
    """
    Reconstruct 2D fit spectrum from Slice-by-Slice fit results.
    
    Takes individual 1D fit results (one per time slice) and stacks them
    into a complete 2D fit array. This allows visualization and comparison
    with the measured 2D data for Slice-by-Slice fitting.
    
    Parameters
    ----------
    results : list or pd.DataFrame
        Fit results, either:

        - list: Output from fit_wrapper for each slice.
          Each element: ``[par_ini, par_fin, conf_CIs, emcee_fin, emcee_CIs]``
        - pd.DataFrame: From results2df() with parameters as columns

    const : tuple
        Constants for residual_fun:
        (x, data, package, function_str, unpack, e_lim, t_lim)
        Used to evaluate fit function at each time point.
    args : tuple
        Arguments for fit function (model, dim, debug).
        Passed to residual_fun for spectrum generation.
    num_fmt : str, default='%.6e'
        Number format for saving (scientific notation with 6 decimals)
    delim : str, default=','
        Delimiter for CSV output
    save_2D : {-1, 0, 1}, default=0
        Save 2D fit to file:

        - 0: Don't save
        - 1: Save to CSV
        - -1: Save to CSV (same as 1)

    save_path : str or Path, default=''
        Directory path for saving. File saved as: save_path/fit2D.csv
        Directory created if doesn't exist.
    
    Returns
    -------
    ndarray
        2D fit array (shape: [n_time, n_energy])
        Each row is the fitted spectrum for one time slice
    """
    x_const, data_const, package_const, fit_fun_const, unpack_const, e_lim_const, t_lim_const = const
    lst = [] # intialize
    for N in range(len(results)):
        # list of lmfit_wrapper fit results
        if isinstance(results, list):
            lst.append(
                residual_fun(
                    results[N][1].params,
                    x_const,
                    np.asarray(data_const),
                    package_const,
                    fit_fun_const,
                    unpack=cast(int, unpack_const),
                    e_lim=cast(list[int], e_lim_const),
                    t_lim=cast(list[int], t_lim_const),
                    res_type='fit',
                    args=args,
                )
            )
        # pandas dataframe containing parameters as columns
        elif isinstance(results, pd.DataFrame):
            lst.append(
                residual_fun(
                    results.iloc[N].values,
                    x_const,
                    np.asarray(data_const),
                    package_const,
                    fit_fun_const,
                    unpack=cast(int, unpack_const),
                    e_lim=cast(list[int], e_lim_const),
                    t_lim=cast(list[int], t_lim_const),
                    res_type='fit',
                    args=args,
                )
            )
    fit2D = np.asarray(lst)
    #
    if abs(save_2D) == 1:
        np.savetxt(os.path.join(save_path, 'fit2D.csv'),
                   fit2D, fmt=num_fmt, delimiter=delim)
    #
    return fit2D

#
# Plot fit results 1D and 2D functions
#
def plt_fit_res_1D(
    x: ArrayLike,
    y: ArrayLike,
    fit_fun_str: str,
    package: Any,
    par_init: Any,
    par_fin: Any,
    args: Optional[tuple[Any, ...]] = None,
    plot_ind: bool = True,
    show_init: bool = True,
    title: str = '',
    fit_lim: Optional[list[int]] = None,
    config: Optional[PlotConfig] = None,
    legend: Optional[list[str]] = None,
    **kwargs: Any
) -> None:
    """
    Plot 1D fit results: data, initial guess, final fit, components, and residual.
    
    Creates a comprehensive visualization showing data, model components,
    total fit, and residual. Essential for evaluating fit quality and
    understanding multi-component models.
    
    Parameters
    ----------
    x : array
        X-axis data (energy or time)
    y : array
        Y-axis data (spectrum to be fitted)
    fit_fun_str : str
        Name of fitting function in package (e.g., 'fit_model_mcp')
    package : module
        Python module containing fit_fun_str (typically trspecfit.spectra)
    par_init : list or lmfit.Parameters
        Initial parameter guess. Can be empty list [] if show_init=False.
    par_fin : lmfit.MinimizerResult or lmfit.Parameters or list
        Final fit parameters:

        - lmfit.MinimizerResult: From fit_wrapper result[1]
        - lmfit.Parameters: Manual parameter object
        - list: Empty list shows initial guess only (no final fit)

    args : tuple, optional
        Additional arguments for fit function (model, dim, debug).
        If None, defaults to empty tuple.
    plot_ind : bool, default=True
        Plot individual components:

        - True: Show each component separately (colored + filled)
        - False: Show only total fit (faster, cleaner for many components)

    show_init : bool, default=True
        Show initial parameter guess:

        - True: Plot initial guess as dotted gold line
        - False: Skip initial guess (cleaner when guess is far off)

    title : str, default=''
        Plot title. Use for file/model identification.
    fit_lim : list of int, optional
        Fit limit indices [left, right] to show as vertical dashed lines.
        Visualizes which data region was used for optimization.
    config : PlotConfig, optional
        Plot configuration object. If None, uses defaults.
    legend : list of str, optional
        Legend labels for components (used only if plot_ind=True).
        If None, auto-generates 'component 0', 'component 1', etc.
    **kwargs : dict
        Override config attributes for this plot:
        x_label, y_label, x_lim, y_lim, x_dir, y_dir, res_mult,
        save_img, save_path, dpi_plot, dpi_save
    
    Notes
    -----
    When saving (save_img=1 or -1), provide full path with extension:
    save_path='results/baseline_fit.png'
    """
    if config is None:
        config = PlotConfig()
    
    if args is None:
        args = ()
    
    # Extract settings from config
    x_label = kwargs.get('x_label', config.x_label)
    y_label = kwargs.get('z_label', config.z_label) # y is Intensity in 1D plot
    x_dir = kwargs.get('x_dir', config.x_dir)
    x_type = kwargs.get('x_type', config.x_type)
    y_type = kwargs.get('y_type', config.y_type)
    x_lim = kwargs.get('x_lim', config.x_lim)
    y_lim = kwargs.get('y_lim', config.y_lim)
    dpi_plot = kwargs.get('dpi_plot', config.dpi_plot)
    dpi_save = kwargs.get('dpi_save', config.dpi_save)
    res_mult = kwargs.get('res_mult', 5)
    save_img = kwargs.get('save_img', 0)
    save_path = kwargs.get('save_path', '')
    
    # Get fit function
    fit_fun = getattr(package, fit_fun_str)

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    
    # Get standard colors
    colors: list[str] = list(plt.rcParams['axes.prop_cycle'].by_key().get('color', ['#1f77b4']))
    
    # Create figure
    fig, ax = plt.subplots(1, 1, dpi=dpi_plot)
    
    # Plot data
    plt.plot(x_arr, y_arr, color=colors[0], linewidth=2, label='data')
    
    # Plot initial guess if requested
    if show_init: 
        par_ini = ulmfit.par_extract(par_init, return_type='list')
        plt.plot(x_arr, fit_fun(x_arr, par_ini, 0, *args), color='#FFD700',
                linestyle=':', linewidth=2, label='initial guess')
    
    # Plot final fit (components and/or sum)
    if isinstance(par_fin, (lmfit.minimizer.MinimizerResult, lmfit.parameter.Parameters)):
        par_fin_vals = ulmfit.par_extract(par_fin, return_type='list')
        
        # Plot individual components if requested
        if plot_ind:
            peaks = fit_fun(x_arr, par_fin_vals, 1, *args)
            for p, peak in enumerate(peaks):
                label = legend[p] if legend and p < len(legend) else f'component {p}'
                color_idx = (p + 1) % len(colors)
                plt.plot(x_arr, peak, color=colors[color_idx], linestyle='-', 
                        linewidth=2, label=label)
                ax.fill_between(x_arr, 0, peak, facecolor=colors[color_idx], alpha=0.5)
        
        # Plot final fit sum
        plt.plot(x_arr, fit_fun(x_arr, par_fin_vals, 0, *args), color='#000000',
                linestyle='-', linewidth=1, label='final fit')
        
        # Calculate residual
        res = y_arr - fit_fun(x_arr, par_fin_vals, 0, *args)
    else:
        # Initial guess only
        par_ini = ulmfit.par_extract(par_init, return_type='list')
        res = y_arr - fit_fun(x_arr, par_ini, 0, *args)
    
    # Plot residual (scaled for visibility)
    plt.plot(x_arr, res * res_mult, color='#808080',
            linestyle='-', linewidth=2, label=f'{res_mult}*residual')
    
    # Set axis labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.title(title, loc='left', fontsize=10)
    
    # Apply axis limits, direction, and scale
    if x_type == 'log':
        ax.set_xscale('log')
    if x_lim is not None:
        ax.set_xlim(x_lim[0], x_lim[1])
    if x_dir == 'rev':
        plt.gca().invert_xaxis()
    if y_type == 'log':
        ax.set_yscale('log')
    if y_lim is not None:
        ax.set_ylim(y_lim[0], y_lim[1])
    
    # Draw zero line
    if x_lim is not None:
        ax.hlines(y=0, xmin=x_lim[0], xmax=x_lim[1],
                 color='#A9A9A9', linestyle=':')
    else:
        ax.hlines(y=0, xmin=np.min(x_arr), xmax=np.max(x_arr),
                 color='#A9A9A9', linestyle=':')
    
    # Draw vertical lines showing fit limits
    if fit_lim is not None and len(fit_lim) == 2:
        ax.vlines(x=[x_arr[fit_lim[0]], x_arr[-fit_lim[1]] if fit_lim[1] > 0 else x_arr[-1]],
                 ymin=np.min(res), ymax=np.max(y_arr),
                 colors='#A9A9A9', linestyle='--')
    
    # Legend
    plt.legend(bbox_to_anchor=(1.35, 1))
    
    # Save with predetermined filename
    if abs(save_img) == 1:
        plt.savefig(save_path, dpi=dpi_save, bbox_inches='tight',
                   pad_inches=0.05, facecolor='white', edgecolor='auto')
    
    # Display or close
    if save_img >= 0:
        plt.show()
    else:
        plt.close()
    #
    return None

#
def plt_fit_res_2D(
    data: np.ndarray,
    fit: np.ndarray,
    x: Optional[ArrayLike] = None,
    y: Optional[ArrayLike] = None,
    config: Optional[PlotConfig] = None,
    **kwargs: Any
) -> None:
    """
    Plot 2D fit results: data, fit, and residual maps.
    
    Creates a three-panel visualization showing measured data, fitted data,
    and residual (data - fit) as 2D color maps. Use to improve fit by switching
    component type, changing number of components, etc.
    
    Parameters
    ----------
    data : 2D array
        Measured data (shape: [n_time, n_energy])
    fit : 2D array
        Fitted data (same shape as data)
    x : array-like, optional
        X-axis (energy) coordinates. If None, uses column indices.
    y : array-like, optional
        Y-axis (time) coordinates. If None, uses row indices.
    config : PlotConfig, optional
        Plot configuration object. If None, uses defaults.
    **kwargs : dict
        Override config attributes for this plot.

        Common options:

        - x_label, y_label : Axis labels (z_label used for colorbar title)
        - x_lim, y_lim : Fit limit indices ``[left, right]`` or ``[start, stop]``.
          Used for both slicing residual and drawing limit lines
        - z_lim_top : Color scale ``[min, max]`` for data and fit panels.
          Synchronized scale enables direct comparison
        - z_lim_res : Color scale ``[min, max]`` for residual panel.
          Independent scale optimizes residual visibility
        - z_colormap : Colormap name (default 'viridis')
        - x_dir, y_dir : 'def' or 'rev' for axis direction
        - x_type, y_type : 'lin' or 'log' for axis scale
        - save_img : 0 (display), 1 (save+display), -1 (save only)
        - save_path : Directory path (file saved as '2D_data_fit_res.png')
    """
    if config is None:
        config = PlotConfig()
    
    # Extract settings from config
    x_label = kwargs.get('x_label', config.x_label)
    y_label = kwargs.get('y_label', config.y_label)
    z_colormap = kwargs.get('z_colormap', config.z_colormap)
    x_dir = kwargs.get('x_dir', config.x_dir)
    x_type = kwargs.get('x_type', config.x_type)
    y_dir = kwargs.get('y_dir', config.y_dir)
    y_type = kwargs.get('y_type', config.y_type)
    save_img = kwargs.get('save_img', 0)
    save_path = kwargs.get('save_path', '')
    
    # Fit limit indices
    x_lim = kwargs.get('x_lim', None)
    y_lim = kwargs.get('y_lim', None)
    
    # Color scale limits
    z_lim_top = kwargs.get('z_lim_top', None)  # Shared for data and fit
    z_lim_res = kwargs.get('z_lim_res', None)  # Independent for residual
    
    # Calculate residual
    res = data - fit
    
    # Cut residual according to x_lim and y_lim for statistics
    if x_lim is not None and y_lim is not None:
        res_cut = res[y_lim[0]:y_lim[1], x_lim[0]:-x_lim[1]]
    elif x_lim is not None:
        res_cut = res[:, x_lim[0]:-x_lim[1]]
    elif y_lim is not None:
        res_cut = res[y_lim[0]:y_lim[1], :]
    else:
        res_cut = res
    
    res_sum = np.sum(np.abs(res_cut))
    res_dim = np.shape(res_cut)
    
    # Create default axes if not provided
    if x is None:
        x_arr = np.arange(0, np.shape(data)[1], 1, dtype=float)
    else:
        x_arr = np.asarray(x, dtype=float)
    if y is None:
        y_arr = np.arange(0, np.shape(data)[0], 1, dtype=float)
    else:
        y_arr = np.asarray(y, dtype=float)
    
    # Determine color scale ranges
    # Data and fit share the same scale for comparison
    if z_lim_top is None:
        range_dat_fit = [min(np.min(data), np.min(fit)), 
                         max(np.max(data), np.max(fit))]
    else:
        range_dat_fit = z_lim_top
    
    # Residual has independent scale
    if z_lim_res is None:
        range_res = [np.min(res_cut), np.max(res_cut)]
    else:
        range_res = z_lim_res
    
    # Create figure layout
    fig, axs = plt.subplot_mosaic([['left', 'right'],
                                   ['bottom', 'bottom'],
                                   ['bottom', 'bottom']],
                                  constrained_layout=True, figsize=(9, 12))
    
    # Data panel (uses shared scale)
    pc_dat = axs['left'].pcolormesh(x_arr, y_arr, data, cmap=z_colormap,
                                    vmin=range_dat_fit[0], vmax=range_dat_fit[1],
                                    shading='nearest')
    axs['left'].set_title('Data [min: ' + str('{0:.3E}'.format(np.min(data))) +
                          ', max: ' + str('{0:.3E}'.format(np.max(data))) + ']')
    
    # Fit panel (uses shared scale)
    pc_fit = axs['right'].pcolormesh(x_arr, y_arr, fit, cmap=z_colormap,
                                     vmin=range_dat_fit[0], vmax=range_dat_fit[1],
                                     shading='nearest')
    axs['right'].set_title('Fit [min: ' + str('{0:.3E}'.format(np.min(fit))) +
                           ', max: ' + str('{0:.3E}'.format(np.max(fit))) + ']')
    
    # Residual panel (independent scale)
    pc_res = axs['bottom'].pcolormesh(x_arr, y_arr, res, cmap=z_colormap,
                                      vmin=range_res[0], vmax=range_res[1],
                                      shading='nearest')
    axs['bottom'].set_title('Residual (Data-Fit) [min: ' +
                            str('{0:.3E}'.format(np.min(res_cut))) +
                            ', max: ' + str('{0:.3E}'.format(np.max(res_cut))) +
                            ']' + '\n' + 'total residual (sum within black dotted lines): ' +
                            str('{0:.3E}'.format(res_sum)) + '\n' + str('per spectrum: ') +
                            str('{0:.3E}'.format(res_sum/res_dim[0])) + str(', per pixel: ') +
                            str('{0:.3E}'.format(res_sum/res_dim[0]/res_dim[1])))
    
    # Colorbar only on residual map
    fig.colorbar(pc_res, orientation='vertical')
    
    # Labels only on residual map
    axs['bottom'].set_ylabel(y_label)
    axs['bottom'].set_xlabel(x_label)
    
    # Draw horizontal and vertical lines showing fit limits
    if y_lim is not None:
        axs['bottom'].axhline(y=float(y_arr[y_lim[0]]), xmin=0, xmax=1,
                             color='#000000', linestyle=':')
        axs['bottom'].axhline(y=float(y_arr[y_lim[1]]), xmin=0, xmax=1,
                             color='#000000', linestyle=':')
    if x_lim is not None:
        axs['bottom'].axvline(x=float(x_arr[x_lim[0]]), ymin=0, ymax=1,
                             color='#000000', linestyle=':')
        axs['bottom'].axvline(x=float(x_arr[np.shape(res)[1]-x_lim[1]]), ymin=0, ymax=1,
                             color='#000000', linestyle=':')
    
    # Apply axis settings to all three plots
    if x_type == 'log':
        axs['left'].set_xscale('log')
        axs['right'].set_xscale('log')
        axs['bottom'].set_xscale('log')
    if x_dir == 'rev':
        axs['left'].invert_xaxis()
        axs['right'].invert_xaxis()
        axs['bottom'].invert_xaxis()
    if y_type == 'log':
        axs['left'].set_yscale('log')
        axs['right'].set_yscale('log')
        axs['bottom'].set_yscale('log')
    if y_dir == 'rev':
        axs['left'].invert_yaxis()
        axs['right'].invert_yaxis()
        axs['bottom'].invert_yaxis()
    
    # Save
    if abs(save_img) == 1:
        uplt.img_save(os.path.join(save_path, '2D_data_fit_res.png'))
    
    # Show or close
    if save_img >= 0:
        plt.show()
    else:
        plt.close()
    #
    return None

#
def plt_fit_res_pars(
    df: pd.DataFrame,
    x: Optional[ArrayLike] = None,
    config: Optional[PlotConfig] = None,
    save_img: Union[int, list[int]] = 0,
    save_path: PathLike = ''
) -> None:
    """
    Plot fit parameters individually as functions of time/index.
    
    Creates separate plots for each parameter column in the DataFrame,
    showing how parameters evolve over time (from Slice-by-Slice fitting).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with parameters as columns. Typically from results2df().
        Each row represents one fitted slice/time point.
    x : array-like, optional
        X-axis (time) values for plotting. If None, uses row indices.
    config : PlotConfig, optional
        Plot configuration object. If None, uses defaults.
    save_img : int or list, default=0
        Save/display control for each plot:

        - int: Apply same setting to all parameters

          - 0: Display only
          - 1: Display and save
          - -1: Save only (no display)

        - list: One element per parameter (per row in df).
          Allows selective saving (e.g., save only varied parameters)
          
    save_path : str or Path, default=''
        Directory path for saving plots.
        Each plot saved as: save_path/{parameter_name}.png
        Directory created if doesn't exist.
    """
    # Use default config if none provided
    if config is None:
        config = PlotConfig()
    
    # if save_img is passed as int, make array of length = number of parameters
    if isinstance(save_img, int):
        save_img_list = len(df.columns) * [save_img]
    else:
        save_img_list = save_img
    
    # plot all parameters as function of time
    for c, col in enumerate(df.columns):
        uplt.plot_1D(
            data=[df[col]],
            x=x,
            config=config,
            title=col,
            x_dir='def',
            x_type=config.y_type,
            x_label=config.y_label,
            y_label=col,
            save_img=save_img_list[c],
            save_path=os.path.join(save_path, col)
        )
    #
    return None
