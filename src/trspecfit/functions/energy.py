"""
Spectral component functions for energy-resolved spectroscopy.

Function Conventions
--------------------
Use CamelCase naming (UpperCamelCase or lowerCamelCase) for function names.

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
def Offset(x: np.ndarray, y0: float, spectrum: np.ndarray) -> np.ndarray:
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

    return np.full_like(spectrum, y0, dtype=float)


#
def Shirley(x: np.ndarray, pShirley: float, spectrum: np.ndarray) -> np.ndarray:
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

    return 1e-6 * pShirley * np.cumsum(spectrum[::-1])[::-1]


#
def LinBack(
    x: np.ndarray, m: float, b: float, xStart: float, xStop: float, spectrum: np.ndarray
) -> np.ndarray:
    """
    Clamped linear background.

    Linear between xStart and xStop, constant outside that range.
    Works for both inclining and declining energy axes — xStart and xStop
    refer to energy values (xStart < xStop required).

    Parameters
    ----------
    x : ndarray
        Energy axis.
    m : float
        Slope (intensity per energy unit).
    b : float
        Y-value at xStart (intercept).
    xStart : float
        Left boundary of linear region (energy units, must be < xStop).
    xStop : float
        Right boundary of linear region (energy units, must be > xStart).
    spectrum : ndarray
        Current peak sum (background calling convention).

    Returns
    -------
    ndarray
        Background: linear between xStart and xStop, clamped constant outside.

    Raises
    ------
    ValueError
        If xStart >= xStop.
    """

    if xStart >= xStop:
        raise ValueError(
            f"LinBack requires xStart < xStop, got xStart={xStart}, xStop={xStop}"
        )
    y = m * (x - xStart) + b
    y_stop = m * (xStop - xStart) + b
    return np.where(x < xStart, b, np.where(x > xStop, y_stop, y))


#
# peak shape function definitions
#


#
def Gauss(x: np.ndarray, A: float, x0: float, SD: float) -> np.ndarray:
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

    return A * np.exp(-1 / 2 * ((x - x0) / SD) ** 2)


#
def GaussAsym(
    x: np.ndarray, A: float, x0: float, SD: float, ratio: float
) -> np.ndarray:
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

    return np.where(x < x0, Gauss(x, A, x0, SD), Gauss(x, A, x0, SD * ratio))


#
def Lorentz(x: np.ndarray, A: float, x0: float, W: float) -> np.ndarray:
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

    return A / (1 + ((x - x0) / W * 2) ** 2)


#
def Voigt(x: np.ndarray, A: float, x0: float, SD: float, W: float) -> np.ndarray:
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

    voigt = np.real(wofz(((x - x0) + 1j * (W / 2)) / SD / np.sqrt(2)))
    return np.asarray(A * voigt / np.max(voigt))


#
def GLS(x: np.ndarray, A: float, x0: float, F: float, m: float) -> np.ndarray:
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

    return np.asarray(
        A
        * (
            (1 - m) * np.exp(-(((x - x0) / F) ** 2) * 4 * np.log(2))
            + m / (1 + 4 * ((x - x0) / F) ** 2)
        )
    )


#
def GLP(x: np.ndarray, A: float, x0: float, F: float, m: float) -> np.ndarray:
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

    return np.asarray(
        A
        * np.exp(-(((x - x0) / F) ** 2) * 4 * np.log(2) * (1 - m))
        / (1 + 4 * m * ((x - x0) / F) ** 2)
    )


#
def DS(x: np.ndarray, A: float, x0: float, F: float, alpha: float) -> np.ndarray:
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

    return (
        A
        * np.cos(np.pi * alpha / 2 + (1 - alpha) * np.arctan((x - x0) / F))
        / (F**2 + (x - x0) ** 2) ** ((1 - alpha) / 2)
    )
