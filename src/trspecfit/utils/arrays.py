"""
Utility functions for array operations and data manipulation.

This module provides utilities for:
- Scientific number formatting with consistent width
- Pandas DataFrame item extraction
- Sign change detection with zero handling
- Array padding and convolution for signal processing
- Angular normalization
- Running averages
"""

import math
from decimal import Decimal
from typing import Literal, cast, overload

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.signal import convolve

#
# General
#

#
def format_float_scientific(
    number: float, exp_digits: int = 4, precision: int = 14
) -> str:
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
def OoM(x: float) -> int:
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
        [rounds down: OoM(9.99) returns 0, not 1]

    Raises
    ------
    ValueError
        If x is zero, since log10(0) is undefined.
    """
    if x == 0:
        raise ValueError("Order of magnitude undefined for zero")
    #
    return int(math.floor(math.log10(abs(x))))

#
# Pandas utilities
#

#
@overload
def get_item(
    df: pd.DataFrame, row: int | list, col: str | int,
    astype: Literal['series'] = ...) -> pd.Series: ...
@overload
def get_item(
    df: pd.DataFrame, row: int | list, col: str | int,
    astype: Literal['float']) -> float: ...
@overload
def get_item(
    df: pd.DataFrame, row: int | list, col: str | int,
    astype: Literal['bool']) -> bool: ...
def get_item(
    df: pd.DataFrame,
    row: int | list,
    col: str | int,
    astype: Literal['series', 'float', 'bool'] = 'series'
) -> pd.Series | float | bool:
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
    pd.Series, float, or bool
        Extracted item in requested format.

    Raises
    ------
    ValueError
        If df is empty.

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
    if df.empty:
        raise ValueError("get_item called on an empty DataFrame")

    # Row selection
    if isinstance(row, int):
        series = df.iloc[row]
    elif isinstance(row, list):
        series = df.loc[df[row[0]].isin(row[1])]
    else:
        raise TypeError("row must be int or list")

    # Column selection
    if isinstance(col, str):
        item = series[col]
    elif isinstance(col, int):
        item = series[series.columns[col]]
    else:
        raise TypeError("col must be str or int")

    # Type conversion
    if astype == 'series':
        return item
    elif astype == 'float':
        return float(item)
    elif astype == 'bool':
        return bool(item)
    else:
        raise ValueError(f"astype must be 'series', 'float', or 'bool', got '{astype}'")

#
# NumPy/SciPy array operations
#

#
def sign_change(array: ArrayLike, ignore_zeros: bool = True) -> NDArray[np.int_]:
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
    return cast(NDArray[np.int_], sign_change_arr)

#
def pad_x_y(
    x: ArrayLike,
    y: ArrayLike,
    x_step: float,
    pad_size: int | float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
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
    >>> y = np.array([1, 3, 6])
    >>> x_pad, y_pad = pad_x_y(x, y, x_step=1, pad_size=2)
    >>> x_pad
    array([-2., -1.,  0.,  1.,  2.,  3.,  4.])
    >>> y_pad
    array([1, 1, 1, 3, 6, 6, 6])
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    pad_size = int(pad_size)
    
    # Pad y-axis (replicate edge values)
    y_pad_l = y_arr[0] * np.ones(pad_size)
    y_pad_r = y_arr[-1] * np.ones(pad_size)
    y_pad = np.concatenate((y_pad_l, y_arr, y_pad_r))
    
    # Pad x-axis (extend linearly)
    x_pad_l = np.linspace(x_arr[0] - x_step*pad_size, x_arr[0], pad_size, 
                          endpoint=False)
    x_pad_r = np.linspace(x_arr[-1] + x_step, x_arr[-1] + pad_size*x_step, 
                          pad_size, endpoint=True)
    x_pad = np.concatenate((x_pad_l, x_arr, x_pad_r))
    #
    return x_pad, y_pad

#
def my_conv(
    x: ArrayLike,
    y: ArrayLike,
    kernel: ArrayLike,
    method: Literal['scipy', 'numpy'] = 'scipy'
) -> NDArray[np.float64]:
    """
    Convolution with proper edge handling via padding.
    
    Wraps scipy.signal.convolve or numpy.convolve with automatic padding
    to minimize edge artifacts. Normalizes kernel (divides by sum).
    
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
    """
    # Determine padding size from kernel
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    kernel_arr = np.asarray(kernel, dtype=float)
    pad_size = int(kernel_arr.size / 2)
    
    # Add padding to minimize edge artifacts
    x_pad, y_pad = pad_x_y(x_arr, y_arr, float(x_arr[1] - x_arr[0]), pad_size)
    
    # Compute convolution with normalized kernel
    if method == 'scipy':
        y_conv_pad = np.asarray(
            convolve(y_pad, kernel_arr, mode='same') / np.sum(kernel_arr),
            dtype=float,
        )
    elif method == 'numpy':
        y_conv_pad = np.asarray(
            np.convolve(y_pad, kernel_arr, mode='same') / np.sum(kernel_arr),
            dtype=float,
        )
    else:
        raise ValueError(f"Unknown method '{method}'")
    
    # Remove padding and return
    return cast(NDArray[np.float64], y_conv_pad[pad_size:-pad_size])

#
def phi_norm(phi: float, norm: float = 2*np.pi) -> float:
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
    """
    #
    return phi - norm * math.floor(phi / norm)

#
def running_mean(x: ArrayLike, y: ArrayLike, N: int) -> NDArray[np.float64]:
    """
    Calculate running (moving) average with proper edge handling.
    
    Computes a moving average using convolution with a boxcar kernel,
    with padding to handle edges properly.
    For more advanced smoothing, consider pandas.rolling() or scipy filters.

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
    
    Notes
    ----------
    Performance comparison:
    https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
    - numpy.convolve: slowest
    - cumsum: floating point errors for N > 1E5
    - scipy with padding: fastest and most robust (this implementation)
    """
    #
    return my_conv(x, y, np.ones(N))
