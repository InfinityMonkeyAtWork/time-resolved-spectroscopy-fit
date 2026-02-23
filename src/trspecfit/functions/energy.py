"""
Spectral component functions for energy-resolved spectroscopy.

Function Conventions
--------------------
**Peak Functions:**
Signature: func(x, par1, par2, ...)
- x: Energy axis (numpy array)
- par1, par2, ...: Function-specific parameters
- Returns: Spectrum as numpy array (same shape as x)

**Background Functions:**
Signature: func(x, par, spectrum)
- x: Energy axis (numpy array)
- par: Background parameter(s)
- spectrum: Current peak sum (numpy array, for backgrounds that depend on peaks)
- Returns: Background spectrum as numpy array (same shape as x)

**Parameter Naming:**
- Use descriptive names without underscores: A, x0, SD, W, F, m, alpha
- A: Amplitude (maximum value of function)
- x0: Peak center/position
- SD: Standard deviation (Gaussian width)
- W: FWHM (Full Width at Half Maximum)
- F: Width parameter (pseudo-Voigt approximations)
- m: Mixing parameter (Gaussian-Lorentzian balance)
- alpha: Asymmetry parameter (Doniach-Sunjic)

Implementation Notes
--------------------
**Amplitude vs Area:**
All functions currently use amplitude normalization (peak height = A).
Future versions may add area normalization options. SD2FWHM = 2*np.sqrt(2*np.log(2))

Adding New Functions
--------------------
To add a new peak or background function:

1. Implement function following conventions above
2. Add to config/functions.py if it's a background
3. Test with realistic spectroscopy parameters
"""

import numpy as np
from scipy.special import wofz


#
def Offset(x, y0, spectrum):
    """
    Constant offset background.
    
    Parameters
    ----------
    x : ndarray
        Energy axis (not used, but required for consistent function signature)
    y0 : float
        Offset value (constant intensity level)
    spectrum : ndarray
        Current peak sum (not used, but required for background function signature)
    
    Returns
    -------
    ndarray
        Constant array of shape len(x) with value y0
    """
    return y0* np.ones(np.shape(spectrum)[0])

#
def Shirley(x, pShirley, spectrum):
    """
    Shirley background for inelastic electron scattering.
    
    Parameters
    ----------
    x : ndarray
        Energy axis (not used in calculation, but required for signature)
    pShirley : float
        Shirley scaling factor (internally multiplied by 1E-6 for numerical stability).
        Controls the strength of the background relative to peak area.
    spectrum : ndarray
        Current peak sum. The Shirley background is computed as cumulative integral
        of this spectrum. **Must have increasing kinetic energy (or decreasing
        binding energy) direction.**
    
    Returns
    -------
    ndarray
        Shirley background spectrum (same shape as spectrum)
    """
    return 1E-6 * pShirley * np.cumsum(spectrum[::-1])[::-1]

#
def LinBack(x, pLinear, spectrum):
    """
    Linear background with positive slope.
    
    Parameters
    ----------
    x : ndarray
        Energy axis (not used in calculation, but required for signature)
    pLinear : float
        Slope of linear background (in intensity per index).
        Positive values create increasing background.
    spectrum : ndarray
        Current peak sum. Used to determine starting point (spectrum[0])
        and length of background array.
    
    Returns
    -------
    ndarray
        Linear background starting at spectrum[0] with slope pLinear
    """  
    background = np.arange(0, np.shape(spectrum)[0], 1)
    return pLinear*background + spectrum[0]

#
def LinBackRev(x, pLinear, spectrum):
    """
    Linear background with negative slope.
    
    Parameters
    ----------
    x : ndarray
        Energy axis (not used in calculation, but required for signature)
    pLinear : float
        Slope magnitude of linear background (in intensity per index).
        Positive values create decreasing background (reversed).
    spectrum : ndarray
        Current peak sum. Used to determine starting point (spectrum[-1])
        and length of background array.
    
    Returns
    -------
    ndarray
        Linear background starting at spectrum[-1] with negative slope
    """  
    background = np.arange(0, np.shape(spectrum)[0], 1)
    return pLinear*background[::-1] + spectrum[-1]

#
# peak shape function definitions 
#

#
def Gauss(x, A, x0, SD):
    """
    Gaussian (normal) distribution peak.
    
    Parameters
    ----------
    x : ndarray
        Energy axis
    A : float
        Peak amplitude (maximum value of function)
    x0 : float
        Peak center position (energy at maximum)
    SD : float
        Standard deviation (Gaussian width parameter).
        Related to FWHM by: FWHM = 2.355 * SD = 2*sqrt(2*ln(2)) * SD
    
    Returns
    -------
    ndarray
        Gaussian peak spectrum
    """
    return A*np.exp(-1/2*((x-x0)/SD)**2)

#
def GaussAsym(x, A, x0, SD, ratio):
    """
    Asymmetric Gaussian peak with different widths on each side.
    
    Parameters
    ----------
    x : ndarray
        Energy axis
    A : float
        Peak amplitude (maximum value at x0)
    x0 : float
        Peak center position (energy at maximum)
    SD : float
        Standard deviation for x < x0 (low energy side)
    ratio : float
        Width ratio: SD2/SD1, where SD2 is width for x >= x0.
        - ratio = 1: Symmetric Gaussian
        - ratio < 1: Narrower on high energy side
        - ratio > 1: Broader on high energy side
    
    Returns
    -------
    ndarray
        Asymmetric Gaussian peak spectrum
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
    Lorentzian (Cauchy) distribution peak.
    
    Parameters
    ----------
    x : ndarray
        Energy axis
    A : float
        Peak amplitude (maximum value at x0)
    x0 : float
        Peak center position
    W : float
        Full width at half maximum (FWHM).
        The width where intensity drops to half the maximum value.
    
    Returns
    -------
    ndarray
        Lorentzian peak spectrum
    """
    return A /(1+ ((x-x0) /W*2)**2)

#
def Voigt(x, A, x0, SD, W):
    """
    Voigt profile (convolution of Gaussian and Lorentzian).
    [Use GLP or GLS (pseudo-Voigt) for ~10x speedup]

    Parameters
    ----------
    x : ndarray
        Energy axis
    A : float
        Peak amplitude (maximum value, approximately, for narrow peaks)
    x0 : float
        Peak center position
    SD : float
        Gaussian standard deviation (instrumental/inhomogeneous width)
    W : float
        Lorentzian FWHM (lifetime/homogeneous width)
    
    Returns
    -------
    ndarray
        Voigt profile spectrum
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
    Gaussian-Lorentzian Sum (pseudo-Voigt) approximation.
    
    Parameters
    ----------
    x : ndarray
        Energy axis
    A : float
        Peak amplitude (maximum value)
    x0 : float
        Peak center position
    F : float
        Peak width parameter (related to FWHM)
    m : float
        Mixing parameter controlling Gaussian/Lorentzian balance:
        - m = 0: Pure Gaussian
        - m = 1: Pure Lorentzian
        - 0 < m < 1: Weighted mixture
        Typical value: m ≈ 0.3
    
    Returns
    -------
    ndarray
        Pseudo-Voigt profile (sum form)
    """
    return A*(1-m) *np.exp(-((x-x0) /F)**2 *4 *np.log(2)) \
            + m/(1+ 4*((x-x0) /F)**2)

#
def GLP(x, A, x0, F, m):
    """
    Gaussian-Lorentzian Product (pseudo-Voigt) approximation.
    
    Parameters
    ----------
    x : ndarray
        Energy axis
    A : float
        Peak amplitude (maximum value)
    x0 : float
        Peak center position
    F : float
        Peak width parameter (related to FWHM)
    m : float
        Mixing parameter controlling Gaussian/Lorentzian character:
        - m = 0: Pure Gaussian
        - m = 1: Pure Lorentzian
        - 0 < m < 1: Hybrid shape
        Typical value: m ≈ 0.3
    
    Returns
    -------
    ndarray
        Pseudo-Voigt profile (product form)
    """
    return A*np.exp(-((x-x0) /F)**2 *4 *np.log(2) *(1-m)) \
            / (1+ 4*m*((x-x0) /F)**2)

#
def DS(x, A, x0, F, alpha):
    """
    Doniach-Sunjic lineshape for metallic systems.
    
    Parameters
    ----------
    x : ndarray
        Energy axis
    A : float
        Amplitude scaling factor (note: NOT the maximum value due to asymmetry)
    x0 : float
        Peak position (approximately the maximum, depends on alpha)
    F : float
        Width parameter (related to FWHM, but complex due to asymmetry)
    alpha : float
        Asymmetry parameter (singularity index):
        - alpha = 0: Lorentzian (no asymmetry)
        - 0 < alpha < 0.3: Typical for metals (e.g., Al: 0.10-0.15)
        - Larger alpha: Stronger asymmetry, more pronounced tail
        - Range: typically 0-0.5 for physical systems
    
    Returns
    -------
    ndarray
        Doniach-Sunjic lineshape
    """
    return A *np.cos(np.pi*alpha/2 +(1-alpha)*np.arctan((x-x0)/F)) \
           / (F**2 +(x-x0)**2 )**((1-alpha)/2)

# Moeini, B, Linford, MR, Fairley, N, et al. [Doniach-Sunjic-Shirley (DSS)]
# Definition of a new (DSS) peak shape for fitting asymmetric XPS signals
# Surf Interface Anal. 2022; 54(1): 67-77. doi:10.1002/sia.7021

# CasaXPS Manual: http://www.casaxps.com/help_manual/line_shapes.htm

# #
# def DSGLS(x, A, x0, F, alpha, m):
#     """
#     Doniach-Sunjic blended with Gaussian-Lorentzian Sum.
    
#     Parameters
#     ----------
#     x : ndarray
#         Energy axis
#     A : float
#         Amplitude scaling factor
#     x0 : float
#         Peak center position
#     F : float
#         Width parameter (shared by DS and GLS components)
#     alpha : float
#         Asymmetry parameter (from DS):
#         - 0 < alpha < 0.5: Metallic asymmetry
#         - alpha = 0: Reduces to pure GLS
#     m : float
#         Mixing parameter (from GLS):
#         - Controls Gaussian/Lorentzian balance in symmetric part
#         - 0 < m < 1 (typical: 0.3)
    
#     Returns
#     -------
#     ndarray
#         Hybrid DS-GLS lineshape
#     """
#     return A* (m /(F**2 +(x-x0)**2 )**((1-alpha)/2) + \
#            (1-m) *GLS(x, A=1, x0=x0, F=F, m=0.3))

# #
# def DSGLP(x, A, x0, F, alpha, m):
#     """
#     Doniach-Sunjic blended with Gaussian-Lorentzian Product.
    
#     Parameters
#     ----------
#     x : ndarray
#         Energy axis
#     A : float
#         Amplitude scaling factor
#     x0 : float
#         Peak center position
#     F : float
#         Width parameter (shared by DS and GLP components)
#     alpha : float
#         Asymmetry parameter (from DS):
#         - 0 < alpha < 0.5: Metallic asymmetry
#         - alpha = 0: Reduces to pure GLP
#     m : float
#         Mixing parameter:
#         - Controls both DS blending and Gaussian/Lorentzian balance
#         - Implementation details unclear (see Notes)
    
#     Returns
#     -------
#     ndarray
#         Hybrid DS-GLP lineshape
#     """
#     return A* (m /(F**2 +(x-x0)**2 )**((1-alpha)/2) + \
#            (1-m) *GLP(x, A=1, x0=x0, F=F, m=0.3))

#
# Area instead of Amplitude
#

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