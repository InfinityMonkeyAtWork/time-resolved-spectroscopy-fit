#
# generate 1D and 2D (time-resolved) spectra
# ["x, par, plt_ind, args" is the typical fit function structure]
#
from trspecfit.mcp import Model
from scipy.ndimage import gaussian_filter1d
import math
import numpy as np
#import inspect
import matplotlib.pyplot as plt
from IPython.display import display

# INFORMATION
# This package contains any functions that create spectra by
# - the model/component/parameter approach (see mcp.py)
# - shifting (and broadening) a 1D or 2D (using bias_function) spectrum
#
# To Do: 
# - replace bias_function with the more general mcp.t_dynamics!
# - implement different types of <combinations>
#   * implicit variables (like depth):
#   * time-dependent (based on applied bias) linear combination of input spectrum
#   * [inelastic mean free path weighted-] depth-dependent spectral combination
#   * combining the above two i.e. time- and depth-dependent modelling

#
def fit_model_mcp(x, par, plot_ind, model, dim, DEBUG):
    """
    <model> is a mcp.Model()
    <dim> =1: 1D spectra, =2: 2D data
    <plt_ind> =0: return the sum, =1: or individual component spectra
    [meaningless for dim=2 where the entire 2D dataset is returned]
    """
    model.update_value(new_par_values=par) # update lmfit parameters
    
    if DEBUG == True: 
        display(model.lmfit_pars)
        model.print_all_pars(detail=1)
    
    # create energy- (and time-)resolved spectrum/ data
    if dim == 1: # 1D
        if plot_ind == 0:
            model.create_value1D()
            return model.value1D
        elif plot_ind == 1:
            model.create_value1D(store1D=1)
            return model.component_spectra
        
    elif dim == 2: # 2D
        model.create_value2D()
        return model.value2D

#
# Everything below could be replaced or improved to be more general
# THIS IS CODE ASSOCIATED WITH THE LARGER BL931 PROJECT AND WILL
# EITHER BE SEPARATED OR PROPERLY INTEGRATED INTO trspecfit
#

#
def XPS_shift_GaussKernel(x, par, plt_ind, input_spectrum):
    """
    Shift and broadening (via Gaussian kernel) of <input_spectrum>
    
    args[0]: input_spectrum is a numpy array (1D)
    par[0]: shifting via np.roll by a number of steps defined by steps_shift
    par[1]: broadening via 1D Gaussian filter kernel of sigma=sigma_GaussKernel 
    returns a 1D numpy array
    
    [x is not used in function (but is part of standard fit_function structure)]
    """
    # assign parameters (unpack) and get input spectrum
    [steps_shift, sigma_GaussKernel] = par
    
    # shift input spectrum
    spectrum_shifted = np.roll(input_spectrum, int(steps_shift))

    # broaden with Gaussian kernel
    if sigma_GaussKernel != 0:
        # sigma needs a np.abs() so scipy.minimize algos work without bounds
        return gaussian_filter1d(spectrum_shifted, np.abs(sigma_GaussKernel),
                                 axis=-1, order=0, output=None, truncate=4.0)
    elif sigma_GaussKernel == 0: # sigma=0 is a problem in gaussian_filter1d
        return spectrum_shifted
#
def XPS_shift_GaussKernel_parameters():
    """
    Return parameter names for the function XPS_shift_GaussKernel
    """
    return ['shift_steps', 'sigma_GK']

#
# linear combination of shifted ground state to describe avg pumped spectrum
#
def XPS_lin_combo(y, par, plt_ind, f_bias, method, circuit, show_info=0):
    """
    Create a linear combination of the (baseline) input spectrum <y> which
    is defined on a 1meV grid).
    
    <method> is the EC-lab method that controls the functional shape of the
    bias. Options are 'CA' for a square wave bias, 'LASV' for a sine wave bias.
    <f_bias> is the bias frequency [Hz], <par> are the parameters of the
    (1) exponential decay following an a square wave bias change (method='CA')
    [parameters are an amplitude <A> and an exponential decay constant <tau>];
    or (2) a sinusoidal function[with the parameters amplitude <A>, and phase 
    <phi>] (differing from the bias input amplitude and phase).
    
    UNITS: <A>: mV, <tau>: ms, <phi>: "sin(2pi*f + phi)", f_bias: Hz
    
    <plt_ind>: =1 return individual fit components
               =0 return resulting linear combination spectrum
    """
    # get time t and bias function for this f_bias frequency
    bias, t = bias_function([], par, 0, f_bias, method, circuit, 0)
    
    # initialize spectrum and individual components
    spectrum_sum = np.zeros(np.size(y))
    if plt_ind == 1: comps = []
    
    # go through depth axis to sum up individual contributions
    for i, t_i in enumerate(t): # go through f_bias function in dt steps and ...
        if plt_ind == 0: # ... add shifted baseline spectrum (y) output
            spectrum_sum += np.roll(y, round(bias[i]))
        elif plt_ind == 1: # or keep track of individual shifted spectra
            comps.append(np.roll(y, round(bias[i])))
            spectrum_sum += np.array(comps[-1])
            
    # divide by number of overlayed spectra
    spectrum = spectrum_sum/len(t)

    # sanity check
    if show_info >= 2:
        print('size t: ' + str(len(t)))
        for p, para in enumerate(par): print('par[' + str(p) +']= ' + str(para))
        fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        fig.suptitle('simulated measured bias [mV] over time [ms] (left) ' + \
                     'and measured XPS over BE [eV] (right)')
        ax1.set_xlabel('Time (ms)'); ax1.set_ylabel('Simulated, measured bias (mV)')
        ax1.plot(t, bias)
        ax1.hlines(y=0, xmin=np.min(t), xmax=np.max(t), color='#808080', linestyle=':')
        ax2.set_xlabel('Binding energy (NOT eV)'); ax2.set_ylabel('Intensity (arb. u.)')
        ax2.plot(spectrum, label='linear comb.')
        ax2.plot(y/np.max(y)*np.max(spectrum), label='zeroV (scaled)')
        plt.legend(); plt.show()
    
    if plt_ind == 1: return comps
    elif plt_ind == 0: return spectrum

#
def XPS_2Dshift(y, par, plt_ind, f_bias, method, circuit, t, show_info=0):
    """
    Create time- and energy-dependent (binding/kinetic energy) 2D XPS data
    MORE THAN ONE (BIAS) CYCLE in 2D XPS data
    
    Spectra making up <y> (2D numpy array) need to be interpolated on 1 meV scale
    
    add option to just have zeros (or ground state spectrum) at the end
    i.e. for all times t > t_cutoff [start of dead time]
    """
    # get bias function for this f_bias frequency
    bias = bias_function(t, par, 0, f_bias, method, circuit, 1)
    
    # shift ground state spectrum for every time step and append to 2D data map
    data = np.empty((len(t),len(y))) # initialize placeholder
    for i, t_i in enumerate(t): # go through f_bias function in dt steps and ...
        data[i,:] = np.roll(y, round(bias[i])) # ... add shifted baseline spectrum (y) to output
        
    # data_list = [] # initialize spectrum and individual components
    # for i, t_i in enumerate(t):
    #     data_list.append(np.roll(y, round(bias[i])))
    # return np.asarray(data_list)
    
    # sanity check
    if show_info >= 2:
        print('size t: ' +str(len(t)))
        for p, para in enumerate(par): 
            print('par[' +str(p) +']= ' +str(para))
        plt.plot(t, bias); plt.show()
    #
    return data

#
# create the bias response of an equivalent circuit (plus optional time axis)
#
def bias_function(t, par, plt_ind, f_bias, method, circuit, ext_t):
    """
    Create simulated bias (over time) profile for EC-lab <method> 'CA' 
    i.e. a square wave bias, or 'LASV' i.e. a sine wave bias.
    <circuit> is the equivalent circuit of the sample interface. Options
    so far are 'RC-R', xxx, xxx.
    LIST:
    CA + RC-R -> mono-exponential decay + offset (i.e. I_{t=infinity})
    LASV + RC-R -> sinusoidal function + offset (regular offset)
    ... to be expanded
    <f_bias> is the bias frequency [Hz]. Set <plt_ind> to zero (=0).
    <t> is the input time axis (ms) [<ext_t>=1]. If this function 
    should create its own time axis based on <f_bias>, then <ext_t> <=0.
    Specifically: <ext_t>=0 returns bias, time, while <ext_t>=-1 returns
    time t only.
    <par> are the parameters which depend on <method> and <circuit>. For
    circuit=='RC-R' they are defining the bias amplitude (par[0] in mV) 
    and exponential decay tau [for 'CA'] or pase shift phi [for 'LASV']
    (par[1], in ms and ~pi, respectively); plus an offset for both.
    Use this function to fit another bias over time data set by setting
    <ext_t>==1 and supplying an external time axis <t>. This function
    will return the bias function guess (e.g. to a minimizer function).
    Use this function to generate a bias curve guess simply based on an
    input bias frequency <f_bias> and return the corresponding time axis
    as well by setting <ext_t>=0. Returns bias (with an amplitude of 
    par[0]) and optionally time in milliseconds.
    """
    # time axis (generate time axis based on <f_bias> or use input <t>)
    if ext_t <= 0: # generate time axis based on f_bias
        # time t goes from 0 to 1/f_bias
        delta = 1*10**OoM(1/f_bias) # time step is dynamic 
        # such that len(t) is between 1k and 10k 
        t = np.arange(0, 1/f_bias *1E3, delta) # in ms (1E3)
    
    # return time only for ext_t = -1, skip bias computation below
    if ext_t == -1: return t

    # "normalize" time axis into repeating pattern
    t_norm, N_t_norm = f_norm(t/1E3, f_bias, N=2)
    # encode the sign of the applied bias in an array
    even_odd = np.asarray([(-1)**np.abs(N) for N in N_t_norm])
        
    # create bias function
    if (method == 'CA') and (circuit == 'RC-R'): # t_norm [s] and tau [ms] -> "*1E3" 
        bias = par[0]*even_odd *np.exp(-t_norm*1E3/par[1]) +par[2]*even_odd
    elif (method == 'LASV') and (circuit == 'RC-R'): #  t [ms] -> "*1E3"
        bias = par[0]*np.sin(2*np.pi*f_bias*t/1E3 +par[1]) +par[2]
    
    # set bias values to 0 for t<0 (t=0 is "pump-probe overlap")
    bias[t<0] = 0
    #
    if ext_t == 0: return bias, t
    elif ext_t == 1: return bias
#
def bias_function_parameters(method, circuit, ext=0, par=[], print_info=0):
    """
    EC-lab (or Keithley) electrochemistry biasing <method>
    Options so far: CA (square wave), LASV (sine wave)
    print_info = 0: only return the fit parameter names
               = 1: additionally print fit info
               = 2: don't print but return fit info
               = -1: only print, return None
    """
    # harcode lists of the EC-lab method, fit function, and parameters
    if (method == 'CA') and (circuit == 'RC-R'):
        prnt_method = 'square wave (SQW)'
        prnt_fit_fun = 'mono-exponential decay'
        prnt_par_names = ['amp_mV', 'tau_ms', 'amp_inf_mV']
    elif (method == 'LASV') and (circuit == 'RC-R'):
        prnt_method = 'sine wave (SinW)'
        prnt_fit_fun = 'sine wave'
        prnt_par_names = ['amp_mV', 'phi_pi', 'amp_0_mV']
    # excpetion for default trspecfit.File()
    elif (method == 'NA') and (circuit == 'NA'):
        prnt_method = 'NA'; prnt_fit_fun = 'NA'; prnt_par_names = []
    else:
        print('NOT a valid combination of <method> and <circuit> [bias_function_parameters]')
    # generate string containing general info about fit
    prnt_str = 'applied bias (method=' +method +') is a ' + \
                prnt_method +'\nexpected measured response at interface\n' + \
                '(represented by an ' +circuit +' equivalent circuit) is:\n' + \
                prnt_fit_fun +' with parameters...\n'
    # define list of empty parameters (this is helpful if the user 
    # just wants a list of parameter names, i.e. ext=0)
    if ext == 0: par = len(prnt_par_names)*['x']
    # go through all parameters and add them to general info string
    for p, par_name in enumerate(prnt_par_names):
        prnt_str += str(par_name) +' = ' +str(par[p]) +'\n'
    # print general info about linear combination 
    if abs(print_info) == 1: print(prnt_str)
    #
    # how will this look like with Keithley?
    #
    # optionally return string (to be printed outside) and other info
    if print_info == 2:
        return [prnt_par_names, prnt_str]
    elif print_info < 0:
        return None
    else:
        return prnt_par_names

#
# Helper functions
#
def OoM(x):
    """
    copy from infinitymonkey_plot package
    """
    return int(math.floor(math.log10(x)))

#
def f_norm(time, f, N):
    """
    'normalize' a <time> axis (unit: s) to be repeating in T=1/<f> intervals
    where <f> is the repetition frequency. Every T=1/<f> cycle is divided into
    <N> (int) sub-cycles. E.g. N=2 for a square-wave response (multiply every
    other sub-cycle with "-1".
    
    returns normalized time (<t_norm>) and N_counter as numpy array each.
    """
    #
    norm = 1 /f /N
    #
    t_norm = []; N_counter = []
    for i, t_i in enumerate(time):
        #
        N_temp = math.floor(t_i /norm)
        N_counter.append(N_temp)
        t_norm.append(t_i -N_temp*norm)
        
    return np.asarray(t_norm), np.asarray(N_counter)

#
# LEGACY
#

# # actual peak fit functions DEPTH_RESOLVED [from ZnO project]
# def lin_comb_spec(x, z, IMFP, BB, d_SCL, FWHM_IC, basis_fun, delta=1000, amp_ratio=2):
#     """
#     create linear combination made up of basis function
#     to describe the shape of an XPS spectrum based on 
#     the amount of band bending present within the
#     inelastic mean free path (IMFP) of the semiconductor
#     surface region. The band bending is assumed as to
#     have exponential shape. 1/e value is the width of 
#     the space charge layer (d_SCL).
#     x: spectral axis (usually binding/ kinetic energy)
#     z: depth axis (semiconductor surface normal)
#     IMFP: inelastic mean free path (nm)
#     d_SCL: depth of space charge layer (nm) [1/e value]
#     BB: maximal band bending [at surface] (meV)
#     basis_fun: Gaussian (0), Lorentzian (1), Voigt (2), 
#     asymmetrical Gaussian (3), or Zn3d doublet made of 
#     Gauss (11), Lorentz (12), Voigt (13), asymGauss(14).
#     FWHM_IC: full width at half maximum value of the 
#     individual contributions whose linear combination 
#     will make up the output overall spectrum,
#     FWHM_IC[0] is Gaussian, FWHM_IC[1] is Lorentzian;
#     for asymmetrical Gauss, FWHM_IC[1] is the ratio.
#     """
#     # get standard deviation for Gaussian FHWM
#     SD_G = FWHM_IC[0] /(2*np.sqrt(2*np.log(2)))
#     # get FWHM for Lorentzian or ratio for asymGauss
#     FWHM_L = FWHM_IC[1]
#     # initialize spectrum
#     spectrum = np.zeros(np.size(x))
#     # go through depth axis to sum up individual contributions
#     for z_i in z:
#         #gauss_fun_amp1(x, A, x0, SD)
#         # weight * spectrum (Gauss[x0=VB_position(z)])
#         spectrum += gauss_fun_amp1(x, np.exp(-z_i /IMFP), BB* np.exp(-z_i /d_SCL), SD_G)
#     #
#     return spectrum

#
#
#