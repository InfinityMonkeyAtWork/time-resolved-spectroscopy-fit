#
# functions defining parameter profiles
#
# e.g. [inelastic mean free path weighted-] depth-dependent profiles
# of an amplitude of a peak in XPS (X-ray photoelectron spectroscopy)

import numpy as np


#
def exp_decay(x: np.ndarray, A: float, tau: float) -> np.ndarray:
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
def linear(x: np.ndarray, m: float, b: float) -> np.ndarray:
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
def Gauss(x: np.ndarray, A: float, x0: float, w: float) -> np.ndarray:
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
    w : float
        Standard deviation (width).

    Notes
    -----
    Primary use cases: fluence averaging, inhomogeneous broadening.
    """

    return A * np.exp(-0.5 * ((x - x0) / w) ** 2)
