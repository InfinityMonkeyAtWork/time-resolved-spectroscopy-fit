"""
Utility functions for array operations and data manipulation.

This module provides utilities for:
- Scientific number formatting with consistent width
- Pandas DataFrame item extraction
- Sign change detection with zero handling
- Array padding and convolution for signal processing
- Angular normalization and running averages
"""
import math
from decimal import Decimal
import pandas as pd
import numpy as np
from scipy.signal import convolve

#
# General
#

#
def format_float_scientific(number, exp_digits=4, precision=14):
    """
    Format number in scientific notation with consistent string length.
    
    Unlike numpy.format_float_scientific, this guarantees fixed-width output
    by padding the exponent to a specified number of digits. This is useful
    for fixed-width output files and column alignment.
    
    Parameters
    ----------
    number : float
        Number to format
    exp_digits : int, default=4
        Number of digits in exponent (e.g., 4 gives E+0001)
    precision : int, default=14
        Number of significant figures in mantissa
    
    Returns
    -------
    str
        Formatted number string (e.g., "1.23456789012345E+0001")
    
    Examples
    --------
    >>> format_float_scientific(1234.5, exp_digits=4, precision=6)
    '1.234500E+0003'
    >>> format_float_scientific(0.000123, exp_digits=2, precision=4)
    '1.2300E-04'
    
    Notes
    -----
    numpy.format_float_scientific with trim='k' does not maintain consistent
    width, which this function addresses.
    """
    exp_digits_str = f"{exp_digits:02d}"
    precision_str = f"{precision:02d}"
    
    # Split into mantissa and exponent
    num, exp_raw = f"{Decimal(number):.{precision_str}E}".split('E')
    sign = exp_raw[0]
    # Format exponent with fixed width (excluding sign)
    exp = f"{int(exp_raw[1:]):0{exp_digits_str}d}"
    #
    return f"{num}E{sign}{exp}"

#
def OoM(x):
    """
    Get order of magnitude of a number.
    
    Parameters
    ----------
    x : float
        Input number (positive or negative, non-zero)
    
    Returns
    -------
    int
        Order of magnitude (power of 10)
    
    Examples
    --------
    >>> OoM(137)
    2
    >>> OoM(-137)
    2
    >>> OoM(0.006)
    -3
    >>> OoM(-0.006)
    -3
    
    Notes
    -----
    Rounds down: OoM(9.99) returns 0, not 1.
    Uses absolute value, so sign doesn't affect order of magnitude.
    Raises ValueError for zero since log10(0) is undefined.
    
    Raises
    ------
    ValueError
        If x is zero (order of magnitude undefined)
    """
    if x == 0:
        raise ValueError("Order of magnitude undefined for zero")
    #
    return int(math.floor(math.log10(abs(x))))

#
# Pandas utilities
#

#
def get_item(df, row, col, astype='series'):
    """
    Extract item from pandas DataFrame with flexible row/column selection.
    
    Provides a unified interface for accessing DataFrame elements with
    multiple selection modes for both rows and columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to extract from
    row : int or list
        Row selector:
        - int: Row index (uses iloc)
        - list: [column_name, values] filters rows where df[column_name].isin(values)
          Single items must be passed as lists: row=['col', ['item']]
    col : str or int
        Column selector:
        - str: Column name
        - int: Column index
    astype : {'series', 'float', 'bool'}, default='series'
        Return type for the extracted item
    
    Returns
    -------
    series, float, or bool
        Extracted item in requested format, or -1 if DataFrame is empty
    
    Examples
    --------
    >>> df = pd.DataFrame({'name': ['A', 'B', 'C'], 'value': [1, 2, 3]})
    >>> get_item(df, row=0, col='value', astype='float')
    1.0
    >>> get_item(df, row=['name', ['B', 'C']], col='value', astype='series')
    1    2
    2    3
    Name: value, dtype: int64
    """
    # Check for empty dataframe
    if df.empty:
        return -1
    
    # Row selection
    if isinstance(row, int):
        series = df.iloc[row]
    elif isinstance(row, list):
        series = df.loc[df[row[0]].isin(row[1])]
    
    # Column selection
    if isinstance(col, str):
        item = series[col]
    elif isinstance(col, int):
        item = series[series.columns[col]]
    
    # Type conversion
    if astype == 'series':
        return item
    elif astype == 'float':
        return float(item)
    elif astype == 'bool':
        return bool(item)

#
# NumPy/SciPy array operations
#

#
def sign_change(array, ignore_zeros=True):
    """
    Detect sign changes in an array with proper zero handling.
    
    Standard np.diff(np.sign(array)) approach incorrectly treats zeros as
    sign changes. This function propagates the previous non-zero sign through
    zero values, detecting only "true" sign crossings.
    
    Parameters
    ----------
    array : array_like
        Input array
    ignore_zeros : bool, default=True
        If True, propagate previous non-zero sign through zeros.
        If False, treat zeros as having their own sign (np.sign(0) = 0)
    
    Returns
    -------
    ndarray
        Boolean array where 1 indicates a sign change, 0 otherwise.
        First element is always 0.
    
    Examples
    --------
    >>> sign_change([1, 0, -1], ignore_zeros=True)
    array([0, 0, 1])  # One sign change
    >>> sign_change([1, 0, -1], ignore_zeros=False)
    array([0, 1, 1])  # Two sign changes (incorrect for most use cases)
    
    Notes
    -----
    Useful for detecting zero-crossings in noisy spectroscopy data where
    zeros may be measurement artifacts rather than true crossings.
    """
    asign = np.sign(array)
    
    if ignore_zeros:
        sz = asign == 0
        while sz.any():
            asign[sz] = np.roll(asign, 1)[sz]
            sz = asign == 0
    
    sign_change_arr = ((np.roll(asign, 1) - asign) != 0).astype(int)
    sign_change_arr[0] = 0
    #
    return sign_change_arr

#
def pad_x_y(x, y, x_step, pad_size):
    """
    Pad x and y arrays for convolution with proper edge handling.
    
    Extends arrays by copying edge values, which minimizes edge artifacts
    in convolution operations. The x-axis is extended on a linear grid.
    
    Parameters
    ----------
    x : array_like
        X-axis array (must be linearly spaced)
    y : array_like
        Y-axis array (signal values)
    x_step : float
        Step size of x-axis grid
    pad_size : int or float
        Number of points to pad on each side (converted to int)
    
    Returns
    -------
    x_pad : ndarray
        Padded x-axis array
    y_pad : ndarray
        Padded y-axis array (edge values replicated)
    
    Examples
    --------
    >>> x = np.array([0, 1, 2])
    >>> y = np.array([1, 2, 1])
    >>> x_pad, y_pad = pad_x_y(x, y, x_step=1, pad_size=2)
    >>> x_pad
    array([-2., -1.,  0.,  1.,  2.,  3.,  4.])
    >>> y_pad
    array([1, 1, 1, 2, 1, 1, 1])
    
    Notes
    -----
    Used internally by my_conv for edge handling in convolution operations.
    """
    pad_size = int(pad_size)
    
    # Pad y-axis (replicate edge values)
    y_pad_l = y[0] * np.ones(pad_size)
    y_pad_r = y[-1] * np.ones(pad_size)
    y_pad = np.concatenate((y_pad_l, y, y_pad_r))
    
    # Pad x-axis (extend linearly)
    x_pad_l = np.linspace(x[0] - x_step*pad_size, x[0], pad_size, 
                          endpoint=False)
    x_pad_r = np.linspace(x[-1] + x_step, x[-1] + pad_size*x_step, 
                          pad_size, endpoint=True)
    x_pad = np.concatenate((x_pad_l, x, x_pad_r))
    #
    return x_pad, y_pad

#
def my_conv(x, y, kernel, method='scipy'):
    """
    Convolution with proper edge handling via padding.
    
    Wraps scipy.signal.convolve or numpy.convolve with automatic padding
    to minimize edge artifacts. Used for convolving time dynamics with
    instrument response functions (IRF) in time-resolved spectroscopy.
    
    Parameters
    ----------
    x : array_like
        X-axis (typically time or energy)
    y : array_like
        Y-axis (signal to convolve)
    kernel : array_like
        Convolution kernel (e.g., IRF, smoothing kernel)
    method : {'scipy', 'numpy'}, default='scipy'
        Convolution backend to use
    
    Returns
    -------
    ndarray
        Convolved signal with same length as input y
    
    Examples
    --------
    >>> x = np.linspace(0, 10, 100)
    >>> y = np.sin(x)
    >>> kernel = np.ones(5) / 5  # Boxcar average
    >>> y_smooth = my_conv(x, y, kernel)
    
    Notes
    -----
    - Automatically pads arrays to minimize edge effects
    - Normalizes kernel (divides by sum) for proper amplitude
    - scipy method is generally faster and more robust
    
    See Also
    --------
    pad_x_y : Padding function used internally
    scipy.signal.convolve : Backend convolution implementation
    """
    # Determine padding size from kernel
    pad_size = int(kernel.size / 2)
    
    # Add padding to minimize edge artifacts
    x_pad, y_pad = pad_x_y(x, y, x[1] - x[0], pad_size)
    
    # Compute convolution with normalized kernel
    if method == 'scipy':
        y_conv_pad = convolve(y_pad, kernel, mode='same') / np.sum(kernel)
    elif method == 'numpy':
        y_conv_pad = np.convolve(y_pad, kernel, mode='same') / np.sum(kernel)
    else:
        raise ValueError(f"method must be 'scipy' or 'numpy', got '{method}'")
    
    # Remove padding and return
    return y_conv_pad[pad_size:-pad_size]

#
def phi_norm(phi, norm=2*np.pi):
    """
    Normalize angle to range [0, norm).
    
    Wraps angles to a specified range, handling both positive and negative
    input values. Useful for phase analysis and periodic data.
    
    Parameters
    ----------
    phi : float
        Angle to normalize (radians)
    norm : float, default=2*pi
        Normalization range:
        - 2*pi: Full circle [0, 2π), preserves sine/cosine character
        - pi: Half circle [0, π), forces sine character
    
    Returns
    -------
    float
        Normalized angle in range [0, norm)
    
    Examples
    --------
    >>> phi_norm(3 * np.pi)  # Wrap to [0, 2π)
    3.141592653589793
    >>> phi_norm(-np.pi / 4)  # Negative angle
    5.497787143782138
    >>> phi_norm(3 * np.pi, norm=np.pi)  # Wrap to [0, π)
    0.0
    
    Notes
    -----
    Works for both positive and negative angles. Result is always in [0, norm).
    """
    #
    return phi - norm * math.floor(phi / norm)

#
def running_mean(x, y, N):
    """
    Calculate running (moving) average with proper edge handling.
    
    Computes a moving average using convolution with a boxcar kernel,
    with padding to handle edges properly. Alternative to pandas.rolling().
    
    Parameters
    ----------
    x : array_like
        X-axis (independent variable)
    y : array_like
        Y-axis (signal to smooth)
    N : int
        Window size (number of points to average)
    
    Returns
    -------
    ndarray
        Smoothed signal with same length as input y
    
    Examples
    --------
    >>> x = np.linspace(0, 10, 100)
    >>> y = np.sin(x) + 0.1 * np.random.randn(100)  # Noisy signal
    >>> y_smooth = running_mean(x, y, N=5)
    
    Notes
    -----
    Uses my_conv with boxcar kernel for proper edge handling.
    For more advanced smoothing, consider pandas.rolling() or scipy filters.
    
    References
    ----------
    Performance comparison:
    https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
    - numpy.convolve: slowest
    - cumsum: floating point errors for N > 1E5
    - scipy with padding: fastest and most robust (this implementation)
    
    See Also
    --------
    my_conv : Underlying convolution function
    pandas.DataFrame.rolling : Alternative with more features
    """
    #
    return my_conv(x, y, np.ones(N))