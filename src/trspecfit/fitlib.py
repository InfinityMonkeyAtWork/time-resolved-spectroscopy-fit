#
# 1D and 2D peak fitting functions
#
import lmfit
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
from matplotlib.cm import get_cmap
from trspecfit.utils import plot as uplt

# Changes:
# - residual_fun: pass function instead of package + function 
# - rewrite a peak finder using ML
# - move from 0/1 to False/True whereever there are only 2 options

#
def residual_fun(par, x, data, package, fit_fun_str, unpack=0, e_lim=[],
                 t_lim=[], res_type='lmfit', args=[]):
    """
    Residual function returns the residual of a fit for all lmfit or
    scipy.optimize.minimize instances 
        
    <par> can be a list of parameter values or a lmfit.Parameters()
    object which will be converted to a list of parameter values. This
    list will be passed to fit function [getattr(package, fit_fun_str)]
    
    <x> (np.array) is the axis
    <data> (np.array [1D or 2D] is the y axis.
    
    <unpack> =0 or =1 defines whether parameters are passed to fit 
    function as <par> or <*par>
    
    Limits define which area of data and fit will be considered for 
    optimization and RSS computation: [read the examples carefully!]
    energy: data = data[left : -right], time: data = data[start : stop]
    <e_lim> are defining the energy axis limits (x axis), 
    i.e. data[e_lim[0]:-e_lim[1]] (cut left and right)
    <t_lim> are defining the time axis limits (y axis),
    i.e. data[t_lim[0]:t_lim[1], :] (start and stop index)
    
    Options for return are selected via <res_type>:
     - ='res': residual, i.e. data -fit
     - ='lmfit': residual (1D data) or flattened residual (2D data)
     - ='RSS': residual sum square (0D)
     - ='abs': sum of the absolute residual
     - ='fit': return the fit itself (np.array)
    
    <args> (tuple) additional arguments for the fit function are to be
    passed via args (defaults to empty tuple) which will be unpacked 
    via *args in the fit function itself
    """
    # define the fit function
    fit_fun = getattr(package, fit_fun_str)
    
    # if the minimizer calling this is from the lmfit package, then
    # extract the value from their lmfit.Parameter() (dictionary)
    # or if list of [value, vary/fix, min, max] transition to val list
    par = ulmfit.par_extract(par, return_type='list')
    
    # compute the fit curve [plt_ind has to be hardcoded as zero here!(?)]
    if unpack == 1: fit = fit_fun(x, *par, 0, *args)
    elif unpack == 0: fit = fit_fun(x, par, 0, *args)
    
    # select user-defined region to consider for residual computation
    if len(data.shape) == 1: # 1D data
        if len(e_lim) != 0:
            residual = data[e_lim[0]:-e_lim[1]] -fit[e_lim[0]:-e_lim[1]]
        else: # use entire data and fit array to compute RSS 
            residual = data -fit
    elif len(data.shape) == 2: # 2D data
        if (len(e_lim) != 0) and (len(t_lim) == 0):
            residual = data[:, e_lim[0]:-e_lim[1]] -fit[:, e_lim[0]:-e_lim[1]]
        elif (len(e_lim) == 0) and (len(t_lim) != 0):
            residual = data[t_lim[0]:t_lim[1], :] -fit[t_lim[0]:t_lim[1], :]
        elif (len(e_lim) != 0) and (len(t_lim) != 0):
            residual = data[t_lim[0]:t_lim[1], e_lim[0]:-e_lim[1]] \
                       -fit[t_lim[0]:t_lim[1], e_lim[0]:-e_lim[1]]
        # or use entire data and fit array to compute RSS
        else: residual = data -fit
    
    # DEBUGGING, un/comment
    #print(f'DEBUGGING: RSS={np.sum(residual**2)}')
    
    # type of residual to return
    if res_type == 'RSS':
        return np.sum(residual**2)
    elif res_type == 'abs':
        return np.sum(np.abs(residual))
    elif res_type == 'res':
        return residual
    elif res_type == 'lmfit':
        if len(data.shape) == 1: # 1D data
            return residual
        elif len(data.shape) == 2: # 2D data
            return residual.flatten()
    elif res_type == 'fit':
        return fit

#
def time_display(t_start, print_str='', return_delta_seconds=False):
    """
    Displays delta between <t_start> and now [t_start = time.time()]
    (works up to days, i.e. ddd:hh:mm:ss [rounded to millisecond])
    """
    t_stop = time.time()
    seconds = t_stop -t_start
    
    str_format = 'ss.ms'
    minutes, seconds = divmod(seconds, 60)
    delta_format = f'{seconds:06.3f}'
    
    if minutes > 0:
        str_format = 'mm:' +str_format
        hours, minutes = divmod(minutes, 60)
        delta_format = f'{math.floor(minutes):02d}:{delta_format}'
        
        if hours > 0:
            str_format = 'hh:' +str_format
            days, hours = divmod(hours, 24)
            delta_format = f'{math.floor(hours):02d}' +':' +delta_format
            
            if days > 0:
                str_format = 'ddd:' +str_format
                # could do weeks, months, years here
                delta_format = f'{math.floor(days):03d}' +':' +delta_format
        
    print(print_str +delta_format +f'({str_format})')
    #
    if return_delta_seconds:
        return seconds
    else:
        return None

#
# error estimation helper functions
#
def sigma_dict():
    """
    Returns dictionary of sigma values (str, "key") and their area
    over total area ratios (float [in percent], "value")
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
def sigma_start_stop_percent(sigma_list):
    """
    <sigma_list> is a list of sigma values between (including) 0.5 and
    5 passed as numbers (0.5 increment)
    Returns a list of [start, stop] values (in percent) for each sigma
    value (in sigma_list) e.g. to select/reject values from a discrete 
    distribution based on whether they lie inside/outside of Nth-sigma
    interval
    """   
    borders_pc = [] # low/high borders in percent
    for sigma in sigma_list:
        A2A_total = sigma_dict().get("{:.1f}".format(sigma),
                                     "sigma value not supported")
        if type(A2A_total) == str:
            print(A2A_total); borders_pc = []
        else:
            A_exclude2A_total = 100 -A2A_total
            borders_pc.append([A_exclude2A_total/2, 100 -A_exclude2A_total/2])
    #
    return borders_pc

#
# wrapper for lmfit fit, confidence interval, and lmfit.emcee functions
#
def fit_wrapper(const, args, par_names, par, fit_type, sigmas=[1,2,3], 
                try_CI=1, MCsettings=ulmfit.MC(),
                fit_alg_1='Nelder', fit_alg_2='leastsq',
                show_info=0, save_output=0, save_path=''):
    """
    wrapper for lmfit fit, confidence interval, lmfit.emcee functions
    <par_names> list of fit parameter names 
    <par> is a list of fit parameters [see lmfit.Parameter.add()] each
    passed as a list itself; pass either "[value=initial guess,
    vary=True/False, min=min.boundary, max=max.boundary]" or a 
    condition (string) only, passed as "[expr=<condition>]"; e.g. to
    make one parameter conditional on another
    (see https://lmfit.github.io/lmfit-py/constraints.html for details)
    
    <const> are the constants passed to fit function through
    <residual_fun> (see details there)
    const = (x, data, package, function str, unpack, x limits, y limits)
    <args> are the arguments that will be passed to fit function itself
    
    <fit_type> =0: no fit (debugging); =1: perform one fit with 
    fit_alg_1 as method; =2: perform a first fit (with
    method=<fit_alg_1>) to find the global minimum. 'Nelder' performs
    most robustly; followed by a second fit (with method=<fit_alg_2> to
    optimize locally around global minimum. 'leastsq' allows confidence 
    interval determination with lmfit.conf_interval() if 
    result.errorbars is True
    Any error estimate will only be performed on the last/final fit
     
    <try_CI>=0: no confidence interval determination; =1: use 
    conf_interval if possible
    <MCsettings.use_emcee>=0: don't use lmfit.emcee(); =1: always use;
    =2: use if conf_interval does not work. see "MCsettings" class
    defined in utils.lmfit.py for more details
    <sigmas> is a list of sigma values (int or float) that defines the
    computed confidence intervals for both the conf_interval() and the
    lmfit.emcee() methods

    returns:
    [par_ini, par_fin, conf_CIs, emcee_fin, emcee_CIs] = results
    """
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
    
    if show_info >= 2: # (optionally) show <constants> and <args> input
        print(); print('Constants input to residual function:')
        display_pretty(const)
        print('Arguments input to fit function:')
        display_pretty(args)
    if show_info >= 1: t_0 = time.time() # start time
        
    if fit_type == 0:
        # use plot_fit_res_1D to show data +initial guess +residual
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
        if show_info >= 1:
            print(); print(f'Results fit (method={fit_alg_1}): ') 
            lmfit.report_fit(par_fin.params)
            t_fit = time.time(); print(f'Time fit: {t_fit -t_ini} s')
    #
    if fit_type == 2: # find global minimum + local optimization
        par_fin_GM = mini.minimize(method=fit_alg_1)
        if show_info >= 1:
            print()
            print(f'Results global minumum fit (method={fit_alg_1}): ')
            lmfit.report_fit(par_fin_GM.params)
            t_fit0 = time.time()
            print(f'Time fit (global minimum): {t_fit0 -t_ini} s')
        #
        par_fin = mini.minimize(method=fit_alg_2, params=par_fin_GM.params)
        if show_info >= 1:
            print()
            print(f'Results local optimization fit (method={fit_alg_2}): ')
            lmfit.report_fit(par_fin.params)
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
        if par_fin.errorbars:
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
        # make optional for user to pass value and min/max
        par_fin.params.add('__lnsigma', value=np.log(0.1),
                           min=np.log(0.001), max=np.log(2))
        if show_info >= 1:
            print()
            print('Progress of lmfit.emcee confidence interval determination')
            print('(based on Markov chain Monte Carlo parameter space sampling):')
        # burn necessary if starting point not close to max(probability distribution)
        # i.e. not close to the optimized parameter set, so burn=0 is ok here!
        emcee_fin = mini.emcee(params = par_fin.params, 
                               steps = MCsettings.steps,
                               nwalkers = MCsettings.nwalkers, 
                               burn = MCsettings.burn,
                               thin = MCsettings.thin, 
                               ntemps = MCsettings.ntemps, 
                               workers = MCsettings.workers, 
                               is_weighted = MCsettings.is_weighted, 
                               progress = bool(show_info))
        # lmfit.emcee() results
        if show_info >= 1:
            print()
            print('Results lmfit.emcee() confidence interval determination:')
            lmfit.report_fit(emcee_fin.params)
            t_emcee1 = time.time()
            print(f'Time lmfit.emcee: {t_emcee1-t_emcee0} s')
        # acceptence fraction of all walkers (plot)
        fig_emcee_walker, ax = plt.subplots(1, 1, dpi=75)
        plt.plot(emcee_fin.acceptance_fraction, 'o')
        plt.xlabel('Walker number'); plt.ylabel('Acceptance fraction')
        if abs(save_output)==1:
            uplt.img_save(f'{save_path}_emcee_walker_acceptance_ratio.png')
        if show_info >= 1: plt.show()
        else: plt.close(fig_emcee_walker)
        # draw all combinations of the typically ellipsoidal chi plot
        # [<x=par1, y=par2, z=chi2> plot]
        emcee_truths = [emcee_fin.params.valuesdict().get(par_name) \
                        for par_name in emcee_fin.var_names]
        fig_emcee_corner = plt.figure(figsize=(10,10))
        emcee_corner = corner.corner(emcee_fin.flatchain, 
                                     labels = emcee_fin.var_names,
                                     truths = emcee_truths,
                                     fig = fig_emcee_corner)
        if abs(save_output) == 1:
            uplt.img_save(f'{save_path}_emcee_corner_plot.png')
        if show_info >= 1: plt.show()
        else: plt.close(fig_emcee_corner)
        # find highest probability parameter combination
        highest_prob = np.argmax(emcee_fin.lnprob)
        hp_loc = np.unravel_index(highest_prob, emcee_fin.lnprob.shape)
        mle_soln = emcee_fin.chain[hp_loc]
        # get percentage borders to categorize emcee.flatchain data
        sigma_borders = sigma_start_stop_percent(sigmas)
        # go through all combinations of parameters and sigmas to find
        # lmfit.emcee() confidence intervals
        emcee_CIs_list = [] # initialize results
        for par_name in par_names+['__lnsigma']:
            emcee_par_CIs = [par_name] # initialize results for this parameter
            if par_name in emcee_fin.var_names:
                # get quantiles if fit parameter is variable
                for s, sigma_b in enumerate(sigma_borders):
                    # get cutoff values that meet this sigma threshold (+/-)
                    quantiles = np.percentile(emcee_fin.flatchain[par_name],
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
                         value=list(emcee_fin.params.valuesdict().values()))
        emcee_CIs.columns = CI_cols
        if show_info >= 1:
            print(display(emcee_CIs))
    else: # use_emcee equal to 0, or equal to 2 and conf_interval worked
        emcee_fin = []; emcee_CIs = pd.DataFrame()
    
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
        df_par_fin = ulmfit.par2df(par_fin.params, 'min', par_names)
        df_par_fin.to_csv(str(save_path) +'_par_fin.csv', index=False)
        # conf_CIs using pandas as it is a pd.DataFrame
        if not conf_CIs.empty:
            conf_CIs.to_csv(str(save_path) +'_conf_CIs.csv', index=False)
        # emcee_fin (fit_report) as text dump, emcee flatchain as csv
        if emcee_fin:
            with open(str(save_path) +'_emcee_fin.txt', 'w') as emcee_fin_file:
                emcee_fin_file.write(lmfit.fit_report(emcee_fin))
            emcee_fin.flatchain.to_csv(f'{save_path}_emcee_flatchain.csv',
                                       index=False)
        # emcee_CIs using pandas as it is a pd.DataFrame
        if not emcee_CIs.empty:
            emcee_CIs.to_csv(str(save_path) +'_emcee_CIs.csv', index=False)
    #
    return [par_ini, par_fin, conf_CIs, emcee_fin, emcee_CIs]

#
# plotting fit results for Slice-by-Slice methods
#
def results_select(data, skip=-1, N=-1, dim=1):
    """
    Apply start (<skip>) and stop (<N>) index to 1D data like so:
    return data[skip:N+1] (if skip and N deviate from default, i.e. -1)
    x[:first_N_spec_only+1] : "+1" as "break" in for loop comes at end
    of iteration
    
    Note: add 2D options
    """
    if dim == 1:
        if N == -1: 
            if skip == -1:
                return data # full data set
            else:
                out = data[skip:]
        else:
            if skip == -1:
                out = data[:N+1]
            else:
                out = data[skip:N+1]
    #
    return out

#
def results2df(results, x=[], xlabel='x', index=[], first_N_spec_only=-1,
               skip_first_N_spec=-1, save_df=0, save_path=''):
    """
    Convert Slice-by-Slice fit <results> list of lmfit_wrapper()
    elements into a pandas dataframe
    
    Save dataframe of results and plots (parameter over time, 1 plot
    each) if save_df != 0
    Displays only the plots of parameters that are varied during fit
    (saves all plots)
    
    If not all slices have been fit, i.e. <skip_first_N_spec> and/or
    <first_N_spec_only> != -1 
    then the partial results will be displayed (if <save_df> >= 0)
    and/or saved to <save_path>
    returns a pandas dataframe of the fit results with <index> and <x>
    as rows and parameters (retrieved from <results>) as columns.
    """
    # transform lmfit_wrapper results to dataframe
    df = ulmfit.list_of_par2df(results)
    # get columns names for plot before adding x/index
    cols_plt = df.columns
    
    # select <x> and <index> data if passed
    if len(x) != 0:
        x_save = results_select(data=x,
                                skip=skip_first_N_spec,
                                N=first_N_spec_only)
        df.insert(0, xlabel, x_save) # and insert into dataframe
    if len(index) != 0:
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
        plt_fit_res_pars(df=df[cols_plt], x=x_save, xlabel=xlabel,
                         save_img=save_array, save_path=save_path)
    #
    return df

#
def results2fit2D(results, const, args, num_fmt='%.6e', delim=',',
                  save_2D=0, save_path=''):
    """
    Create 2D fit data (numpy array) output from slice-by-slice fit
    results.
    
    <results> is a list of individual fit results, each structured as
    defined by "lmfit_fit_wrapper(...)" function in this package
    OR
    a pandas dataframe where columns are fit parameters and rows are
    the index/ x (e.g. from "results2df" function from this package)
    
    <const> = (x, data, package, function (str), unpack, e_limits, t_limits)
    where args = (arguments for fit function called in residual_fun)
    see "residual_fun" (this package) for more details!
    
    <num_fmt> is the number format and <delim> is the delimiter used
    to save data
    <save_2D> =0: don't save, =1 (or -1): save as .csv ('%.6e') 
    location of saved file is <save_path> / 'fit2D.csv'
    """
    lst = [] # intialize
    for N in range(len(results)):
        # list of lmfit_wrapper fit results
        if isinstance(results, list):
            lst.append(residual_fun(results[N][1].params, *const, 'fit', args))
        # pandas dataframe containing parameters as columns
        elif isinstance(results, pd.DataFrame):
            lst.append(residual_fun(results.iloc[N].values, *const, 'fit', args))
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
def plt_fit_res_1D(x, y, fit_fun_str, package, par_init, par_fin, args=[0], 
                   plot_ind=0, show_init=1, title='', fit_lim=[], res_mult=5, 
                   xlim=[], xtype='lin', ytype='lin', xdir='rev', ydir='',
                   xlabel='x_axis', ylabel='y_axis', legend=[],
                   dpi_plt=125, save_fig=0, save_fig_path=''):
    """
    Input data: <x> (np.array) axis, <y> (np.array) y axis,
    <fit_fun_str> (string) is the name of the function contained in the
    module <package> that will be used to generate a spectrum to fit 
    the data (y)
    <args> will be passed in an unpacked form (*args) to the fit
    function <package>.<fit_fun_str>
    <dpi_plt> defines the dpi (dots per inch) for the plot produced by
    this function
    Use <show_init> (either 0 or 1) to show the initial guess for the
    fit in the output plot. If the initial guess is shown, <par_init>
    defines the parameter set with which this initial guess will be
    produced (pass empty list if show_init==0). The final fit is always
    plotted. The individual components of the fit can either be shown
    in the fit or not, <plt_ind>=1 or =0, respectively
    <res_mult> is the multiplier used on the residual between data (y)
    and final fit
    
    The output figure can be saved (<save_fig>=1) by passing either the
    full path, a folder, or an empty string (which will save a png with
    no file name in said folder or the current directory, respectively)
    via the <save_path> input (string)
    <save_fig> =0: plot only, =1: save & plot, =-1: save only
    """
    # function = getattr(package, function name as string)
    fit_fun = getattr(package, fit_fun_str)
    # get standard colors that matplotlib cycles through as a list
    colors = get_cmap('tab10').colors
    # define figure 
    fig, ax = plt.subplots(1, 1, dpi=dpi_plt)
    # plot data input (x and y axis)
    plt.plot(x, y, color=colors[0], linewidth=2, label='data')
    
    # plot fit resulting from the initial guess parameters (gold)
    if show_init == 1: 
        par_ini = ulmfit.par_extract(par_init, return_type='list')
        plt.plot(x, fit_fun(x, par_ini, 0, *args), color='#FFD700',
                            linestyle=':', linewidth=2, label='initial guess')
    
    # plot final fit components if they exist
    # or don't if no fit performed (init only)
    if isinstance(par_fin, (lmfit.minimizer.MinimizerResult,
                            lmfit.parameter.Parameters)):
        par_fin_vals = ulmfit.par_extract(par_fin, return_type='list')
        # plot individual components if option selected
        if plot_ind == 1:
            # get individual components of the fit
            peaks = fit_fun(x, par_fin_vals, 1, *args)            
            # plot individual components of the fit
            for p, peak in enumerate(peaks):
                plt.plot(x, peak, color=colors[p+1], linestyle='-', linewidth=2,
                         label=f'component {p}' if len(legend)==0 else legend[p])
                ax.fill_between(x, 0, peak, facecolor=colors[p+1], alpha=0.5)
        # plot the final fit
        plt.plot(x, fit_fun(x, par_fin_vals, 0, *args), color='#000000',
                 linestyle='-', linewidth=1, label='final fit')
        # plot the scaled (for better visibility) residual, i.e. data - fit
        res = y -fit_fun(x, par_fin_vals, 0, *args)
    else: # initial guess only
        # 'fit result is not of type "lmfit.minimizer.MinimizerResult"'
        res = y -fit_fun(x, par_ini, 0, *args)
    
    # plot residual
    plt.plot(x, res*res_mult, color='#808080',
             linestyle='-', linewidth=2, label=str(res_mult)+'*residual')
    
    # set axis labels
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    # define x and y direction, log/linear, or limits (if passed)
    if xtype == 'log': ax.set_xscale('log')
    if len(xlim)!=0: ax.set_xlim(xlim[0], xlim[1])
    if xdir == 'rev': plt.gca().invert_xaxis()
    if ytype == 'log': ax.set_yscale('log')
    #if len(ylim)!=0: ax.set_ylim(ylim[0], ylim[1])
    if ydir == 'rev': plt.gca().invert_yaxis()
    # horizontal line (zero line) and xlim vertical lines defining fit boundaries
    if xlim: 
        ax.hlines(y=0, xmin=np.min(xlim), xmax=np.max(xlim),
                  color='#A9A9A9', linestyle=':')
    else:
        ax.hlines(y=0, xmin=np.min(x), xmax=np.max(x),
                  color='#A9A9A9', linestyle=':')
    # vertical lines showing fit limits
    if fit_lim: 
        ax.vlines(x=[x[fit_lim[0]], x[-fit_lim[1]]],
                  ymin=np.min(res), ymax=np.max(y),
                  colors='#A9A9A9', linestyle='--')
    # title
    plt.title(title, loc='left', fontsize=10)
    # legend
    plt.legend(bbox_to_anchor = (1.35,1))
    
    # save
    if abs(save_fig) == 1:
        plt.savefig(save_fig_path, dpi=300, bbox_inches = 'tight',
                    pad_inches=0.05, facecolor='white', edgecolor='auto')
    # display
    if save_fig >=0 : plt.show()
    else: plt.close()
    #
    return None

#
def plt_fit_res_2D(data, fit, x=[], y=[], xlabel='x', ylabel='y',
                   xlim=[], ylim=[], xdir='', ydir='',
                   xtype='lin', ytype='lin', colormap='RdBu', 
                   range_dat=[], range_res=[], save_img=0, save_path=''):
    """
    <data> and <fit> 2D numpy arrays
    <x> and <y> are the horizontal and vertical axis (1D numpy array)
    
    total residual will be computed from xlim[0] to -xlim[1], and from
    ylim[0] to ylim[1] (and printed in the title of the residual map)
    
    <colormap> is the "cmap" attribute of matplotlib.pyplot.pcolormesh 
    <range_dat> and <range_res> are the z (color coded) axis ranges
    (i.e. [min, max])
    
    <save_img> =0: don't save, =1 [show fig] (or -1 [do not show fig]): save
    if abs(save_img)==1: figure is saved to <save_path>/'2D_data_fit_res.png'
    """
    # define residual
    res = data -fit
    # cut residual according to xlim and ylim
    if len(xlim)!=0 and len(ylim)!=0:
        res_cut = res[ylim[0]:ylim[1], xlim[0]:-xlim[1]]
    elif len(xlim)!=0:
        res_cut = res[:, xlim[0]:-xlim[1]]
    elif len(ylim)!=0:
        res_cut = res[ylim[0]:ylim[1], :]
    else: res_cut = res
    res_sum = np.sum(np.abs(res_cut)) # sum of abs(residual)
    res_dim = np.shape(res_cut) # dimensions of cut residual
    
    # if no x and/or y axis is passed, create one
    if len(x)==0: x = np.arange(0, np.shape(data)[1], 1)
    if len(y)==0: y = np.arange(0, np.shape(data)[0], 1)
    
    # plot ranges
    if len(range_dat) == 0:
        range_dat = [np.min(data), np.max(data)]
    if len(range_res) == 0:
        range_res = [np.min(res_cut), np.max(res_cut)]
    
    # plot layout
    fig, axs = plt.subplot_mosaic([['left', 'right'],
                                   ['bottom', 'bottom'],
                                   ['bottom', 'bottom']],
                                  constrained_layout=True, figsize=(9,12))
    # data
    pc_dat = axs['left'].pcolormesh(x, y, data, cmap=colormap,
                                    vmin=range_dat[0], vmax=range_dat[1])
    axs['left'].set_title('Data [min: ' +str('{0:.3E}'.format(np.min(data))) +\
                          ', max: ' +str('{0:.3E}'.format(np.max(data))) +']') 
    # fit
    pc_fit = axs['right'].pcolormesh(x, y, fit, cmap=colormap,
                                     vmin=range_dat[0], vmax=range_dat[1])
    axs['right'].set_title('Fit [min: ' +str('{0:.3E}'.format(np.min(fit))) +\
                           ', max: ' +str('{0:.3E}'.format(np.max(fit))) +']') 
    # residual
    pc_res = axs['bottom'].pcolormesh(x, y, res, cmap=colormap,
                                      vmin=range_res[0], vmax=range_res[1])
    # fix #$% how to make nice multi-line strings?
    axs['bottom'].set_title('Residual (Data-Fit) [min: ' +\
                            str('{0:.3E}'.format(np.min(res_cut))) +\
                            ', max: ' +str('{0:.3E}'.format(np.max(res_cut))) +\
                            ']' +'\n' +'total residual (sum within black dotted lines): ' +\
                            str('{0:.3E}'.format(res_sum)) +'\n' +str('per spectrum: ') +\
                            str('{0:.3E}'.format(res_sum/res_dim[0])) +str(', per pixel: ') +\
                            str('{0:.3E}'.format(res_sum/res_dim[0]/res_dim[1])))
    # show colorbar only on the large, residual map
    fig.colorbar(pc_res, orientation = 'vertical')
    # lables also only on res map
    axs['bottom'].set_ylabel(ylabel); axs['bottom'].set_xlabel(xlabel)
    # draw horizontal and vertical lines around x and y limits
    if len(ylim) != 0:
        plt.axhline(y=y[ylim[0]], xmin=0, xmax=1,
                    color='#000000', linestyle=':')
        plt.axhline(y=y[ylim[1]], xmin=0, xmax=1,
                    color='#000000', linestyle=':')
    if len(xlim) != 0:
        plt.axvline(x=x[xlim[0]], ymin=0, ymax=1,
                    color='#000000', linestyle=':')
        plt.axvline(x=x[np.shape(res)[1]-xlim[1]], ymin=0, ymax=1,
                    color='#000000', linestyle=':')
    # x and y direction and type (lin/log) for all three plots
    if xtype == 'log':
        axs['left'].set_xscale('log')
        axs['right'].set_xscale('log')
        axs['bottom'].set_xscale('log')
    if xdir == 'rev':
        axs['left'].invert_xaxis()
        axs['right'].invert_xaxis()
        axs['bottom'].invert_xaxis()
    if ytype == 'log':
        axs['left'].set_yscale('log')
        axs['right'].set_yscale('log')
        axs['bottom'].set_yscale('log')
    if ydir == 'rev':
        axs['left'].invert_yaxis()
        axs['right'].invert_yaxis()
        axs['bottom'].invert_yaxis()
    
    # save
    if abs(save_img) == 1:
        uplt.img_save(os.path.join(save_path, '2D_data_fit_res.png'))
    #
    if save_img >=0 : plt.show()
    else: plt.close()
    #
    return None

#
def plt_fit_res_pars(df, x=[], xlabel='', save_img=0, save_path=''):
    """
    Plot fit result parameters individually and optionally save figures
    <save_img> =1 plot and save, =0 plot only, =-1 save only
    <save_img> type is list: one element per row of <df>
                       int: apply this setting to all parameters
    """
    # if save_img is passed as int, make array of length = rows in df
    if isinstance(save_img, int):
        save_img = np.shape(df)[0]*[save_img]
    # plot all parameters as function of time
    for c, col in enumerate(df.columns):
        uplt.plot_1D([df[col]], x, title=col, xlabel=xlabel, ylabel=col,
                       save_img=save_img[c],
                       save_path=os.path.join(save_path, col))
    #
    return None