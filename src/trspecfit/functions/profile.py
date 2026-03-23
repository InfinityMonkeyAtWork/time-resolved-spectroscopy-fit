"""
Parameter-profile functions for auxiliary-axis-resolved spectroscopy models.

Function Conventions
--------------------
Use CamelCase naming (UpperCamelCase) for function names (with ``p`` prefix).
Profile functions must be named ``pFunctionName`` (starting with ``p`` for profile)
to distinguish them from energy- and time-domain functions.

**Profile Functions:**
Signature: func(x, par1, par2, ...)
- x: Auxiliary axis (e.g. depth, position, fluence)
- par1, par2, ...: Function-specific parameters
- Returns: Profile values as numpy array (same shape as x)

**Usage Context:**
Profile functions are attached to an energy-model parameter via
``File.add_par_profile(...)``. During evaluation, the profiled parameter is
sampled across the auxiliary axis and the resulting component traces are
uniformly averaged over that axis.

Parameter Naming
----------------
Common parameter names:
- A: Amplitude
- tau: Decay length/constant
- m: Slope
- b: Intercept
- x0: Center position
- SD: Standard deviation (width)

Use descriptive names without underscores.

Adding New Functions
--------------------
To add a new profile function:

1. Implement a function named ``pFunctionName(x, ...)``
2. Return an array matching the shape of ``x``
3. Keep parameters physically meaningful and unit-consistent
"""

import numpy as np


#
def pExpDecay(x: np.ndarray, A: float, tau: float) -> np.ndarray:
    """
    Exponential decay profile.

    Parameters
    ----------
    x : ndarray
        Auxiliary axis (e.g. depth, position).
    A : float
        Amplitude at x=0.
    tau : float
        Decay constant (same units as x).

    Notes
    -----
    Primary use cases: IMFP-weighted depth profiles in XPS,
    Beer-Lambert absorption profiles in pump-probe.
    """

    return A * np.exp(-x / tau)


#
def pLinear(x: np.ndarray, m: float, b: float) -> np.ndarray:
    """
    Linear profile.

    Parameters
    ----------
    x : ndarray
        Auxiliary axis.
    m : float
        Slope.
    b : float
        Intercept.

    Notes
    -----
    Primary use cases: band bending over depth, center offset over position.
    """

    return m * x + b


#
def pGauss(x: np.ndarray, A: float, x0: float, SD: float) -> np.ndarray:
    """
    Gaussian profile.

    Parameters
    ----------
    x : ndarray
        Auxiliary axis.
    A : float
        Peak amplitude.
    x0 : float
        Center position.
    SD : float
        Standard deviation (width).

    Notes
    -----
    Primary use cases: fluence averaging, inhomogeneous broadening.
    """

    return A * np.exp(-0.5 * ((x - x0) / SD) ** 2)
