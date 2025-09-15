import math
import numpy as np

# This package contains any functions that create fits by combining individual spectra
# as defined in "spectra.py" or based on existing data (e.g. based on a pre-trigger
# baseline spectrum from the time-resolved (ambient pressure) XPS data itself
# Different types of <combinations>:
#     - time-dependent (based on applied bias) linear combination of input spectrum 
#     - [inelastic mean free path weighted-] depth-dependent spectral combination
#     - combining the above two i.e. time- and depth-dependent modelling
#     - shifting and/ or broadening the input spectrum

#
def OoM(x):
    """
    copy from infinitymonkey_plot package
    """
    return int(math.floor(math.log10(x)))
#
#
def fNorm(time, f, N):
    """
    'normalize' a <time> axis (unit: s) to be repeating in T=2pi/<f> intervals
    where <f> is the repetition frequency. Every 2*pi*f cycle is divided into
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
