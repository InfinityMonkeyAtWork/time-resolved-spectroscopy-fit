"""
Temporal dynamics functions for time-resolved spectroscopy.

Function Conventions
--------------------
**Dynamics Functions:**
Signature: func(t, par1, par2, ..., t0, y0)
- t: Time axis (numpy array)
- par1, par2, ...: Function-specific parameters
- t0: Time zero (function starts at this time)
- y0: Offset value (baseline)
- Returns: f(t) = 0 for t < t0, dynamics for t >= t0

**Convolution Kernels:**
Signature: funcCONV(t, par1, par2, ...)
- t: Time axis centered at zero (from create_t_kernel)
- par1, par2, ...: Kernel parameters
- Returns: Normalized kernel function
- Must have companion function: funcCONV_kernel_width() returning multiplier

**Time Zero Convention:**
All dynamics functions are zero before t0 and activate at t >= t0.
This reflects physical causality: response begins after excitation.

**Offset Convention:**
Parameter y0 sets the asymptotic value or baseline.
- Decays: approach y0 as t → ∞
- Rises: start from 0, reach y0 + A
- Oscillations: oscillate around y0

**Time Resolution:**
Functions inherit time axis from Dynamics model. Consider:
- Time step size relative to dynamics (dt << tau)
- Time range coverage (include full decay/rise)
- Kernel width appropriate for convolution

Parameter Naming
----------------
Common parameter names:
- A: Amplitude (change in signal)
- tau: Time constant (decay/rise time, 1/e point)
- t0: Time zero (start of dynamics)
- y0: Offset/baseline value
- f: Frequency (for oscillations)
- phi: Phase (for oscillations)
- SD: Standard deviation (for Gaussian kernels)
- W: FWHM (for Lorentzian kernels)

Adding New Functions
--------------------
To add a new dynamics or convolution function:

1. Implement following conventions above
2. Ensure f(t<t0) = 0 for dynamics functions
3. Add kernel_width function for convolution kernels
4. Test with realistic time-resolved data
"""

import numpy as np
from scipy.signal import square
from scipy.special import erf, wofz


#
def none(t):
    """
    Placeholder function to define empty subcycles in a mcp.Dynamics model.

    Used to define empty subcycles in multi-cycle Dynamics models without
    adding any time-dependent behavior. This allows subcycle numbering to
    work correctly when some subcycles should have no dynamics.

    Usage (in model yaml file)
    ----------
    ```
    model_sub2:
      none: {}
    ```

    Parameters
    ----------
    t : ndarray
        Time axis (not used)

    Returns
    -------
    ndarray
        Array of zeros with same shape as t
    """

    # This function should never actually be called:
    # It is caught in mcp.Model.combine() and skipped entirely.
    return np.zeros_like(t)


#
def linFun(t, m, t0, y0):
    """
    Linear dynamics (constant rate of change).

    Parameters
    ----------
    t : ndarray
        Time axis
    m : float
        Slope (rate of change). Units: [signal units]/[time units]
        - m > 0: Linear increase
        - m < 0: Linear decrease
    t0 : float
        Time zero (start of linear change)
    y0 : float
        Offset value at t0 (initial value)

    Returns
    -------
    ndarray
        Linear function: 0 for t<t0, m*(t-t0)+y0 for t>=t0
    """

    return np.concatenate(
        (np.zeros(np.shape(t[t < t0])[0]), (m * (t - t0) + y0)[t >= t0])
    )


#
def expFun(t, A, tau, t0, y0):
    """
    Exponential decay or rise dynamics.

    Parameters
    ----------
    t : ndarray
        Time axis
    A : float
        Amplitude (initial change at t0).
        - A > 0: Decay from y0+A to y0
        - A < 0: Rise from y0 to y0+|A|
    tau : float
        Time constant (1/e time). Units: [time units]
        At t = t0 + tau, signal changes by factor of e (≈2.718)
    t0 : float
        Time zero (start of exponential)
    y0 : float
        Asymptotic value (baseline as t → ∞)

    Returns
    -------
    ndarray
        Exponential: 0 for t<t0, A*exp(-(t-t0)/tau)+y0 for t>=t0
    """

    return np.concatenate(
        (
            np.zeros(np.shape(t[t < t0])[0]),
            (A * np.exp(-1 / tau * (t - t0)) + y0)[t >= t0],
        )
    )


#
def sinFun(t, A, f, phi, t0, y0):
    """
    Sinusoidal oscillations (coherent dynamics).

    Parameters
    ----------
    t : ndarray
        Time axis
    A : float
        Oscillation amplitude (peak-to-peak = 2A)
    f : float
        Frequency in [1/time units]
        Period = 1/f
    phi : float
        Phase offset in radians
        - phi = 0: Sine starts at zero
        - phi = π/2: Starts at maximum (cosine)
        - phi = π: Starts at zero (negative slope)
    t0 : float
        Time zero (start of oscillation)
    y0 : float
        Offset (center line of oscillation)

    Returns
    -------
    ndarray
        Sinusoid: 0 for t<t0, A*sin(2πf(t-t0)+phi)+y0 for t>=t0
    """

    return np.concatenate(
        (
            np.zeros(np.shape(t[t < t0])[0]),
            (A * np.sin(2 * np.pi * f * (t - t0) + phi) + y0)[t >= t0],
        )
    )


#
def sinDivX(t, A, f, t0, y0):
    """
    Damped sinc function: sin(x)/x oscillation.

    Parameters
    ----------
    t : ndarray
        Time axis
    A : float
        Amplitude scaling factor
    f : float
        Frequency in [1/time units]
    t0 : float
        Time zero (start of oscillation)
    y0 : float
        Offset value

    Returns
    -------
    ndarray
        Sinc oscillation: 0 for t<t0, A*sin(2πf(t-t0))/(2πf(t-t0))+y0 for t>=t0
    """

    x = 2 * np.pi * f * (t - t0)
    return np.concatenate(
        (np.zeros(np.shape(t[t < t0])[0]), (A * np.sin(x) / x + y0)[t >= t0])
    )


#
def erfFun(t, A, SD, t0, y0):
    """
    Error function rise (step with Gaussian broadening).
    erfFun ≈ step ⊗ Gaussian(SD)

    Parameters
    ----------
    t : ndarray
        Time axis
    A : float
        Amplitude (total change from initial to final value)
    SD : float
        Standard deviation of Gaussian broadening (rise time ~2.355*SD)
        Smaller SD → sharper rise
    t0 : float
        Center of rise (50% point)
    y0 : float
        Final value (asymptote as t → ∞)

    Returns
    -------
    ndarray
        Error function: A/2 * (1 + erf((t-t0)/(SD*√2))) + y0
    """

    return A / 2 * (1 + erf((t - t0) / (SD * np.sqrt(2)))) + y0


#
def sqrtFun(t, A, t0, y0):
    """
    Square root rise (diffusion dynamics).

    Parameters
    ----------
    t : ndarray
        Time axis
    A : float
        Amplitude scaling factor
    t0 : float
        Time zero (start of diffusion)
    y0 : float
        Offset value

    Returns
    -------
    ndarray
        Square root rise: 0 for t<t0, A*√(t-t0)+y0 for t>=t0
    """

    # numpy array .clip sets all t<t0 to zero
    return A * np.sqrt((t - t0).clip(0)) + y0


#
# convolution functions
# kernels followed by respective recommended kernel width
#


#
def gaussCONV(x, SD):
    """
    Gaussian convolution kernel (instrumental response function).

    Parameters
    ----------
    x : ndarray
        Time axis (typically from Component.create_t_kernel, centered at 0)
    SD : float
        Standard deviation (Gaussian width).
        FWHM = 2.355 * SD = 2*√(2ln2) * SD

    Returns
    -------
    ndarray
        Gaussian kernel (unnormalized, will be normalized in convolution)
    """

    return np.exp(-1 / 2 * (x / SD) ** 2)


def gaussCONV_kernel_width():
    """
    Kernel width multiplier for Gaussian convolution.
    Kernel extends to ±4*SD from center.
    At 4*SD, Gaussian has decayed to exp(-8) ≈ 3×10⁻⁴ of peak value.
    """

    return 4


#
def lorentzCONV(x, W):
    """
    Lorentzian convolution kernel.

    Parameters
    ----------
    x : ndarray
        Time axis (centered at 0)
    W : float
        Full width at half maximum (FWHM) of Lorentzian

    Returns
    -------
    ndarray
        Lorentzian kernel (unnormalized)
    """

    return 1 / (1 + (x / W / 2) ** 2)


def lorentzCONV_kernel_width():
    """Kernel width multiplier for Lorentzian (12×W)."""

    return 12


#
def voigtCONV(x, SD, W):
    """
    Voigt convolution kernel (Gaussian and Lorentzian combined).

    Parameters
    ----------
    x : ndarray
        Time axis (centered at 0)
    SD : float
        Gaussian standard deviation
    W : float
        Lorentzian FWHM

    Returns
    -------
    ndarray
        Voigt kernel (normalized to peak = 1)
    """

    voigt = np.real(wofz((x + 1j * (W / 2)) / SD / np.sqrt(2)))
    return voigt / np.max(voigt)


def voigtCONV_kernel_width():
    """Kernel width multiplier for Voigt (12)."""

    return 12


#
def expSymCONV(x, tau):
    """
    Symmetric exponential kernel (double exponential).
    Exponential decay in both directions from center:
    exp(-|x|/tau)

    Parameters
    ----------
    x : ndarray
        Time axis (centered at 0)
    tau : float
        Decay time constant

    Returns
    -------
    ndarray
        Symmetric exponential kernel
    """

    return np.exp(-1 / tau * np.abs(x))


def expSymCONV_kernel_width():
    """Kernel width multiplier for symmetric exponential (6×tau)."""

    return 6


#
def expDecayCONV(x, tau):
    """
    Causal exponential kernel (one-sided decay).

    Parameters
    ----------
    x : ndarray
        Time axis (centered at 0)
    tau : float
        Decay time constant

    Returns
    -------
    ndarray
        One-sided exponential: 0 for x<0, exp(-x/tau) for x≥0
    """

    return np.concatenate((np.zeros(np.shape(x[x < 0])), expSymCONV(x[x >= 0], tau)))


def expDecayCONV_kernel_width():
    """Kernel width multiplier for decay exponential (6×tau)."""

    return 6


#
def expRiseCONV(x, tau):
    """
    Causal exponential rise kernel.

    Parameters
    ----------
    x : ndarray
        Time axis (centered at 0)
    tau : float
        Rise time constant

    Returns
    -------
    ndarray
        One-sided exponential: exp(x/tau) for x≤0, 0 for x>0
    """

    return np.concatenate((expSymCONV(x[x <= 0], tau), np.zeros(np.shape(x[x > 0]))))


def expRiseCONV_kernel_width():
    """Kernel width multiplier for rise exponential (6×tau)."""

    return 6


#
def boxCONV(x, width):
    """
    Box (rectangular) convolution kernel.

    Parameters
    ----------
    x : ndarray
        Time axis (centered at 0)
    width : float
        Width of rectangular window

    Returns
    -------
    ndarray
        Rectangular function: 1 inside width, 0 outside (with smooth edges)
    """

    return (square(x - np.min(x) / 2, duty=width / (2 * np.pi)) + 1) / 2


def boxCONV_kernel_width():
    """Kernel width multiplier for box (1×width)."""

    return 1
