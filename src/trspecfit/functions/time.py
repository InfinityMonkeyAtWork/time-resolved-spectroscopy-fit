#
# temporal dynamics/ kinetics functions
# all functions must be f(t) = 0 (for t<t0)
#
import numpy as np
from scipy.special import erf
from scipy.special import wofz
from scipy.signal import square

#
def none(t):
    """
    Placeholder function
    Use to define an empty subcycle in a multi-cycle mcp.Dynamics instance
    This function should never actually be called:
    It is caught in mcp.Model.combine() and skipped entirely.
    
    Example usage in a yaml file:
    model_sub2:
      none: {}
    """
    return np.zeros_like(t)  # Should never be evaluated

#
def linFun(t, m, t0, y0):
    """
    Define linear function for every value in the 1D array t,
    where m is the slope, and y0 is the y intersection at t0
    """
    #return m*(t-t0) +y0
    return np.concatenate((np.zeros(np.shape(t[t<t0])[0]), \
                           (m*(t-t0) +y0)[t>=t0])) 

#
def expFun(t, A, tau, t0, y0):
    """
    Define exponential decay function f(t) defined as
    f(t) = A* exp(-1/tau *[t-t0]) +y0 (for t>t0)
    """
    return np.concatenate((np.zeros(np.shape(t[t<t0])[0]), \
                           (A*np.exp(-1/tau *(t-t0)) +y0)[t>=t0]))

#
def sinFun(t, A, f, phi, t0, y0):
    """
    Define sinusoidal function with amplitude <A>, frequency <f>,
    phase <phi>, offset <y0> on a time axis <t> (np.array)
    """
    return np.concatenate((np.zeros(np.shape(t[t<t0])[0]), \
                           (A *np.sin(2*np.pi*f*(t-t0) +phi) +y0)[t>=t0]))

#
def sinDivX(t, A, f, t0, y0):
    """
    Define sin(x)/x function A* sin(x) /x +y0
    where x = 2*pi*f*(t-t0), and A is the amplitude 
    """
    x = 2*np.pi*f*(t-t0)
    return np.concatenate((np.zeros(np.shape(t[t<t0])[0]), \
                           (A *np.sin(x) /x +y0)[t>=t0]))

#
def erfFun(t, A, SD, t0, y0):
    """
    Define an error function which is a convolution of a Gaussian
    function with a sigma/ standard deviation of SD
    and a step function centered at t0.
    The amplitude of the function is A. The offset is y0
    Returns values on the grid defined by t
    """
    return A/2*(1 +erf((t-t0) /(SD*np.sqrt(2)))) +y0

#
def sqrtFun(t, A, t0, y0):
    """
    Define a square root rise function:
    amplitude (A) * sqrt(t-t0) +y0
    Output is zero for all t < t0
    Multiplier M as in M*(t-t0) would be redundant
    (correlates perfectly with A)
    """
    # numpy array .clip sets all t<t0 to zero
    return A *np.sqrt((t-t0).clip(0)) +y0

#
# convolution functions
# kernels followed by respective recommended kernel width
#

#
def gaussCONV(x, SD):
    return np.exp( -1/2*(x/SD)**2)
def gaussCONV_kernel_width(): return 4

#
def lorentzCONV(x, W):
    return 1 /(1 +( x/W/2)**2)
def lorentzCONV_kernel_width(): return 12

#
def voigtCONV(x, SD, W):
    voigt = np.real(wofz((x +1j*(W/2)) /SD /np.sqrt(2)))
    return voigt /np.max(voigt)
def voigtCONV_kernel_width(): return 12

#
def expSymCONV(x, tau):
    return np.exp( -1/tau *np.abs(x))
def expSymCONV_kernel_width(): return 6

#
def expDecayCONV(x, tau):
    return np.concatenate((np.zeros(np.shape(x[x<0])), \
                           expSymCONV(x[x>=0], tau)))
def expDecayCONV_kernel_width(): return 6

#
def expRiseCONV(x, tau):
    return np.concatenate((expSymCONV_kernel_width(x[x<=0], tau), \
                           np.zeros(np.shape(x[x>0]))))
def expRiseCONV_kernel_width(): return 6

#
def boxCONV(x, width):
    return (square(x-np.min(x)/2, duty=width /(2*np.pi)) +1) /2
def boxCONV_kernel_width(): return 1