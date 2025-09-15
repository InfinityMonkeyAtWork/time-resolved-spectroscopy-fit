# 
# individual peak shapes/ spectral components 
# and background functions to generate spectra
# IF YOU ADD A BACKGROUND FUNCTION, ADD IT TO THE CONFIG.PY FILE!
#
import numpy as np
from scipy.special import wofz
#
# INFORMATION (readme)
# - do not use underscores in function or parameter names
# - the combination of these components into a spectrum is handled in
#   the package "spectra.py"
# - individual components should have the form function(x, par) 
#   or <par> can be explicitly written out, i.e. *par
# - the parameters <par> of an individual peak function should be
#   structured like this: <x> (x axis), <A> (amplitude/Area/etc.),
#   <x0> center, <SD> or <F> or <W> representing the peak width,
#   followed by optional other parameters
# - background functions have the structure function(spectrum, par)
#   where spectrum is the sum of all peaks/components
#
# currently uses amplitude in all function definitions in this package!
# [should swap to area for all functions at some point?]
# SD2FWHM = 2*np.sqrt(2*np.log(2))

#
# background function definitions
#

#
def Offset(x, y0, spectrum):
    """
    Create offset for input spectrum [x axis unused sofar]
    """
    return y0* np.ones(np.shape(spectrum)[0])

#
def Shirley(x, pShirley, spectrum):
    """
    Create Shirley background for an input <spectrum> (1D numpy array)
    Input spectrum should have increasing kinetic energy (or decreasing
    binding energy) as an independent variable.
    
    <pShirley> is the factor used to convert area under the <spectrum>
    to intensity of Shirley spectrum that is returned (type np.array)
    <pShirley> is internally (in here) multiplied by 1E-6 as fittings 
    functions may have problems dealing with small values
    
    """
    # serial 
    # #return 1E-6 *pShirley *np.asarray([np.sum(spectrum[i:]) for \
    #                                   i in range(np.shape(spectrum)[0])])
    # vectorized
    return 1E-6 * pShirley * np.cumsum(spectrum[::-1])[::-1]

#
#def ExpBack(x, pExponential, spectrum):
#    """
#    """
#    return np.exp()

#
def LinBack(x, pLinear, spectrum):
    """
    Create linear background for an input <spectrum>, both 1D np.array
    input spectrum should have increasing kinetic energy (or decreasing
    binding energy) as an independent variable
    cut the spectrum to start at the wings of the outermost peaks for 
    this function to work
    
    Note: implement option for start/ stop selector (in energy)
    """  
    background = np.arange(0, np.shape(spectrum)[0], 1)
    return pLinear*background[::-1] + spectrum[-1]

#
# peak shape function definitions 
#

#
def Gauss(x, A, x0, SD):
    """
    Define Gaussian function for every value in the 1D array x,
    where A is the amplitude, x0 is the x axis offset (center), 
    SD is the standard deviation of the Gaussian distribution.
    The maximum value of this function is equal to A
    """
    return A*np.exp(-1/2*((x-x0)/SD)**2)

#
def GaussAsym(x, A, x0, SD, ratio):
    """
    Define asymmetric Gaussian function with SD1 below x0 and
    SD2 above x0. A is the amplitude
    SD1 = SD, SD2 = ratio *SD -> ratio = SD2 /SD1
    """
    # define two Gaussians different SDs
    lo_x0 = Gauss(x, A, x0, SD)
    hi_x0 = Gauss(x, A, x0, SD *ratio)
    # merge the two Gaussians at x=x0
    # they have the same amplitude at x0 by definition
    return np.concatenate([lo_x0[x<x0], hi_x0[x>=x0]], axis=0)

#
def Lorentz(x, A, x0, W):
    """
    Define Lorentzian function with full width at half 
    amplitude of W centered around x0
    The amplitude of the function is A
    Returns values on the grid defined by x
    """
    return A /(1+ ((x-x0) /W*2)**2)

#
def Voigt(x, A, x0, SD, W):
    """
    Define Voigt function (convolution of Gaussian and
    Lorentzian) with x input, Lorentzian FWHM W, Gaussian 
    standard deviation SD, amplitude A, and x axis offset
    of center x0
    """
    # scipy version
    #voigt = np.real(wofz(((x-x0) + 1j*(W/2)) \
    #        /SD /np.sqrt(2))) /SD /np.sqrt(2*np.pi)
    #return A *voigt +y0
    voigt = np.real(wofz(((x-x0) + 1j*(W/2)) /SD /np.sqrt(2)))
    return A *voigt /np.max(voigt)

#
def GLS(x, A, x0, F, m):
    """
    Gaussian-Lorentzian sum form to approximate a Voigt
    from http://www.casaxps.com/help_manual/line_shapes.htm
    <x> is the x axis on which to define the peak (np.array)
    <A> is the amplitude of the peak (maximum value).
    <x0> is the center of the peak on the x axis (float).
    <m> (float, typical value is 0.3) determines the mix of
    Gaussian and Lorentzian weight in this approximation
    of a Voigt line shape. m=0 ->Gaussian, m=1 ->Lorentzian
    <F> is the overall width of the peak (proportional?)
    """
    return A*(1-m) *np.exp(-((x-x0) /F)**2 *4 *np.log(2)) \
            + m/(1+ 4*((x-x0) /F)**2)

#
def GLP(x, A, x0, F, m):
    """
    Gaussian-Lorentzian product form to approximate a Voigt
    from http://www.casaxps.com/help_manual/line_shapes.htm
    <x> is the x axis on which to define the peak (np.array)
    <A> is the amplitude of the peak (maximum value).
    <x0> is the center of the peak on the x axis (float).
    <m> (float, typical value is 0.3) determines the mix of
    Gaussian and Lorentzian weight in this approximation
    of a Voigt line shape. m=0 ->Gaussian, m=1 ->Lorentzian
    <F> is the overall width of the peak (proportional?)
    """
    return A*np.exp(-((x-x0) /F)**2 *4 *np.log(2) *(1-m)) \
            / (1+ 4*m*((x-x0) /F)**2)

#
def DS(x, A, x0, F, alpha):
    """
    Doniac Sunjic is a theory-based asymmetric lineshape.
    See http://www.casaxps.com/help_manual/line_shapes.htm
    and Doniach S. and Sunjic M., J. Phys. 4C31, 285 (1970)
    for more information.
    
    A is not the amplitude. Should it be or will that distort?
    """
    return A *np.cos(np.pi*alpha/2 +(1-alpha)*np.arctan((x-x0)/F)) \
           / (F**2 +(x-x0)**2 )**((1-alpha)/2)

# Moeini, B, Linford, MR, Fairley, N, et al. [Doniach-Sunjic-Shirley (DSS)]
# Definition of a new (DSS) peak shape for fitting asymmetric XPS signals
# Surf Interface Anal. 2022; 54(1): 67-77. doi:10.1002/sia.7021

#
def DSGLS(x, A, x0, F, alpha, m):
    """
    Linear blend between Doniac Sunjic and Voigt-type functions,
    called F function in casaXPS
    (http://www.casaxps.com/help_manual/line_shapes.htm)
    Voigt-type used here is Gaussian-Lorentzian Sum function (GLS)
    
    THE MIXING m (m and 1-m) is between 0 and 100 and controls the 
    amount of DS-ness, but the m in GLS/GLP is between 0 and 1 and 
    controls the Gauss vs Lorentz
    casaXPS also mentions extra convolution with Gaussian
    
    READ MORE
    """
    return A* (m /(F**2 +(x-x0)**2 )**((1-alpha)/2) + \
           (1-m) *GLS(x, A=1, x0=x0, F=F, m=0.3))

#
def DSGLP(x, A, x0, F, alpha, m):
    """
    Linear blend between Doniac Sunjic and Voigt-type functions,
    called F function in casaXPS
    (http://www.casaxps.com/help_manual/line_shapes.htm)
    Voigt-type used here is Gaussian-Lorentzian Product function (GLP)
    
    THE MIXING m (m and 1-m) is between 0 and 100 and controls the
    amount of DS-ness, but the m in GLS/GLP is between 0 and 1 and
    controls the Gauss vs Lorentz
    casaXPS also mentions extra convolution with Gaussian
    
    READ MORE
    """
    return A* (m /(F**2 +(x-x0)**2 )**((1-alpha)/2) + \
           (1-m) *GLP(x, A=1, x0=x0, F=F, m=0.3))

#
# def gauss_fun_int1(x, A, x0, SD):
#     """
#     Define Gaussian function as a probability density function
#     This means the integral over from minus to plus infinity
#     is 1 (if A=1). For every value in the 1D array x the function
#     will return a function value, where A is the amplitude scaling
#     factor, x0 is the x axis offset, SD is the standard deviation
#     of the Gaussian distribution
#     """
#     return A/SD/np.sqrt(2*np.pi)*np.exp(-1/2*((x-x0)/SD)**2)