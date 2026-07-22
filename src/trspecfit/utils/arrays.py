"""
Utility functions for array operations and data manipulation.

This module provides utilities for:
- Scientific number formatting with consistent width
- Pandas DataFrame item extraction
- Time-axis index resolution
- Sign change detection with zero handling
- Array padding and convolution for signal processing
- Angular normalization
- Running averages
"""

import math
from decimal import Decimal
from typing import Literal, NamedTuple, cast, overload

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
    num, exp_raw = f"{Decimal(number):.{precision_str}E}".split("E")
    sign = exp_raw[0]
    # Format exponent with fixed width (excluding sign)
    exp = f"{int(exp_raw[1:]):0{exp_digits_str}d}"

    return f"{num}E{sign}{exp}"


#
def oom(x: float) -> int:
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
        [rounds down: oom(9.99) returns 0, not 1]

    Raises
    ------
    ValueError
        If x is zero, since log10(0) is undefined.
    """

    if x == 0:
        raise ValueError("Order of magnitude undefined for zero")

    return math.floor(math.log10(abs(x)))


#
# Pandas utilities
#


#
@overload
def get_item(
    df: pd.DataFrame, row: int | list, col: str | int, astype: Literal["series"] = ...
) -> pd.Series: ...
@overload
def get_item(
    df: pd.DataFrame, row: int | list, col: str | int, astype: Literal["float"]
) -> float: ...
@overload
def get_item(
    df: pd.DataFrame, row: int | list, col: str | int, astype: Literal["bool"]
) -> bool: ...
def get_item(
    df: pd.DataFrame,
    row: int | list,
    col: str | int,
    astype: Literal["series", "float", "bool"] = "series",
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
    if astype == "series":
        return item
    if astype == "float":
        return float(item)
    if astype == "bool":
        return bool(item)
    raise ValueError(f"astype must be 'series', 'float', or 'bool', got '{astype}'")


#
# NumPy/SciPy array operations
#


#
def resolve_time_selection(
    time: ArrayLike,
    t_start: float,
    t_stop: float,
    *,
    time_type: str = "abs",
) -> list[int]:
    """
    Convert time bounds to validated ``[ind_start, ind_stop)`` slice indices.

    For a single time point pass ``t_start == t_stop``. Both bounds are
    inclusive in the input; the returned stop is exclusive.

    Parameters
    ----------
    time : array_like
        Time axis to resolve against.
    t_start, t_stop : float
        Time bounds (absolute values or indices, see ``time_type``).
    time_type : {'abs', 'ind'}, default='abs'
        'abs': absolute time stamps. 'ind': time array indices.

    Returns
    -------
    list[int]
        ``[ind_start, ind_stop)``.

    Raises
    ------
    ValueError
        If the result is out of range or empty, or ``time_type`` is
        unrecognized.
    """

    time_arr = np.asarray(time)
    n = len(time_arr)
    if time_type == "abs":
        if t_start == t_stop:
            ind_start = int(np.searchsorted(time_arr, t_start, side="left"))
            ind_stop = ind_start + 1
        else:
            ind_start = int(np.searchsorted(time_arr, t_start, side="left"))
            ind_stop = int(np.searchsorted(time_arr, t_stop, side="right"))
    elif time_type == "ind":
        ind_start = int(t_start)
        ind_stop = int(t_stop) + 1 if t_start == t_stop else int(t_stop + 1)
    else:
        raise ValueError(f"Unknown time_type '{time_type}'. Expected 'abs' or 'ind'.")
    if ind_start >= ind_stop or ind_start >= n or ind_stop <= 0:
        raise ValueError(
            f"Time selection resolves to an empty or out-of-range slice "
            f"[{ind_start}:{ind_stop}). "
            f"Time axis has {n} points [{time_arr[0]}, {time_arr[-1]}]."
        )
    ind_start = max(ind_start, 0)
    ind_stop = min(ind_stop, n)
    return [ind_start, ind_stop]


#
def sign_change(array: ArrayLike, *, ignore_zeros: bool = True) -> NDArray[np.int_]:
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

    # all-zero input has no sign to propagate (and the loop below would
    # never terminate); the roll-difference then correctly yields no changes
    if ignore_zeros and asign.any():
        sz = asign == 0
        while sz.any():
            asign[sz] = np.roll(asign, 1)[sz]
            sz = asign == 0

    sign_change_arr = ((np.roll(asign, 1) - asign) != 0).astype(int)
    sign_change_arr[0] = 0

    return cast("NDArray[np.int_]", sign_change_arr)


#
def pad_x_y(
    x: ArrayLike, y: ArrayLike, x_step: float, pad_size: float
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
    x_pad_l = np.linspace(
        x_arr[0] - x_step * pad_size, x_arr[0], pad_size, endpoint=False
    )
    x_pad_r = np.linspace(
        x_arr[-1] + x_step, x_arr[-1] + pad_size * x_step, pad_size, endpoint=True
    )
    x_pad = np.concatenate((x_pad_l, x_arr, x_pad_r))

    return x_pad, y_pad


#
#
class ConvOperator(NamedTuple):
    """Theta-independent part of the kernel-matrix convolution.

    Built once per time axis by ``conv_matrix_operator``; consumed per
    kernel-parameter set by ``conv_matrix_apply``. The kernel function
    is evaluated on ``dt_unique`` only (kernels are elementwise, and dt
    values repeat heavily on uniform or piecewise-uniform axes), then
    gathered back into the full matrix via ``gather_idx``.

    ``dt_left``/``dt_right`` are the abscissae at which per-kernel
    edge-mass companions evaluate the exact exterior kernel mass
    (edge-value padding semantics; see ``conv_matrix_apply``).
    """

    dt_unique: NDArray[np.float64]  # (n_unique,) sorted unique t_i - t_j
    gather_idx: NDArray[np.intp]  # (n_t, n_t) index into dt_unique
    quad_weights: NDArray[np.float64]  # (n_t,) trapezoid weights
    dt_left: NDArray[np.float64]  # (n_t,) t - t[0], >= 0
    dt_right: NDArray[np.float64]  # (n_t,) t - t[-1], <= 0


#
def conv_matrix_operator(t: ArrayLike) -> ConvOperator:
    """
    Build the theta-independent part of the kernel-matrix convolution.

    The convolution ``y_conv = K @ y`` with
    ``K[i, j] = g(t_i - t_j; theta) * w_j`` (rows normalized) is exact
    quadrature on any monotonic axis — the axis enters through the
    ``dt`` matrix and the weights, so non-uniform sampling is handled
    correctly. Everything that does not depend on the kernel parameters
    is built here, once per axis.

    Kernel mass falling outside the data window is handled analytically:
    the signal is assumed constant beyond the window (edge-value
    padding), so the exterior contribution of row ``i`` is
    ``y[0] * M_L(i) + y[-1] * M_R(i)`` where ``M_L(i)`` is the exact
    integral of the kernel body over ``(-inf, t[0]]`` (an upper-tail
    mass at ``dt_left[i] = t_i - t[0]``) and ``M_R(i)`` the integral
    over ``[t[-1], inf)`` (a lower-tail mass at
    ``dt_right[i] = t_i - t[-1]``). The masses are computed by
    per-kernel companions (``CONV_EDGE_MASS`` in ``functions/time.py``)
    and passed to ``conv_matrix_apply`` — no ghost points, no support
    truncation for any kernel width.

    The ``(n_t, n_t)`` dt matrix is stored deduplicated: kernel
    functions evaluate on ``dt_unique`` (typically ``O(n_t)`` values on
    uniform or piecewise-uniform axes instead of ``O(n_t^2)``) and
    ``conv_matrix_apply`` gathers the values back by index. This is a
    pure optimization — the gathered matrix is bit-identical to direct
    evaluation on the full dt matrix.

    Parameters
    ----------
    t : array_like
        Strictly increasing time axis, at least 2 samples.

    Returns
    -------
    ConvOperator
        ``(dt_unique, gather_idx, quad_weights, dt_left, dt_right)``.
    """

    t_arr = np.asarray(t, dtype=np.float64)
    if t_arr.ndim != 1 or t_arr.size < 2:
        raise ValueError(
            "Convolution requires a 1D time axis with at least 2 points, "
            f"got shape {t_arr.shape}"
        )
    if not np.all(np.isfinite(t_arr)):
        raise ValueError("Convolution time axis contains non-finite values")
    if np.any(np.diff(t_arr) <= 0):
        raise ValueError(
            "Convolution time axis must be strictly increasing; "
            "sort the axis (and data) before fitting."
        )

    # True trapezoid weights: half-cells at both window edges (the
    # exterior beyond the half-cells is covered exactly by the analytic
    # edge masses; np.gradient would give full end cells and
    # double-count against them).
    steps = np.diff(t_arr)
    quad_weights = np.empty_like(t_arr)
    quad_weights[0] = steps[0] / 2
    quad_weights[-1] = steps[-1] / 2
    quad_weights[1:-1] = (t_arr[2:] - t_arr[:-2]) / 2

    dt = t_arr[:, np.newaxis] - t_arr[np.newaxis, :]
    dt_unique, inverse = np.unique(dt, return_inverse=True)
    gather_idx = inverse.reshape(dt.shape).astype(np.intp, copy=False)
    dt_left = t_arr - t_arr[0]
    dt_right = t_arr - t_arr[-1]
    return ConvOperator(dt_unique, gather_idx, quad_weights, dt_left, dt_right)


#
def conv_matrix_apply(
    operator: ConvOperator,
    kernel_values: NDArray[np.float64],
    edge_mass_left: NDArray[np.float64],
    edge_mass_right: NDArray[np.float64],
    y: ArrayLike,
) -> NDArray[np.float64]:
    """
    Apply the kernel-matrix convolution for one set of kernel parameters.

    Consumes the kernel evaluated elementwise on ``operator.dt_unique``
    plus the analytic exterior masses evaluated at ``operator.dt_left``
    / ``operator.dt_right`` (edge-value padding semantics: the signal
    is assumed constant beyond the window). Each row is normalized so a
    constant signal is preserved exactly.

    Parameters
    ----------
    operator : ConvOperator
        Precomputed operator from ``conv_matrix_operator``.
    kernel_values : ndarray
        ``(n_unique,)`` kernel evaluated on ``operator.dt_unique``
        (unnormalized).
    edge_mass_left, edge_mass_right : ndarray
        ``(n_t,)`` exterior kernel masses per row, from the kernel's
        ``CONV_EDGE_MASS`` companion (same normalization as the body).
    y : array_like
        ``(n_t,)`` signal to convolve.

    Returns
    -------
    ndarray
        Convolved signal, same length as ``y``.
    """

    y_arr = np.asarray(y, dtype=np.float64)
    n_t = operator.gather_idx.shape[0]
    if (
        y_arr.shape != (n_t,)
        or edge_mass_left.shape != (n_t,)
        or edge_mass_right.shape != (n_t,)
        or kernel_values.shape != operator.dt_unique.shape
    ):
        raise ValueError(
            "Convolution inputs have wrong shapes: kernel_values "
            f"{kernel_values.shape} (expected {operator.dt_unique.shape}), "
            f"edge masses {edge_mass_left.shape} / {edge_mass_right.shape} "
            f"and signal y {y_arr.shape} (all expected ({n_t},))."
        )
    # Signed errors can cancel inside row sums, so validate sign and
    # finiteness of the raw ingredients (O(n) against the O(n^2) matmul).
    if np.any(kernel_values < 0) or not np.all(np.isfinite(kernel_values)):
        raise ValueError(
            "Convolution kernel values must be finite and nonnegative; "
            "the kernel parameters are likely invalid (NaN or negative "
            "width)."
        )
    if (
        np.any(edge_mass_left < 0)
        or np.any(edge_mass_right < 0)
        or not np.all(np.isfinite(edge_mass_left))
        or not np.all(np.isfinite(edge_mass_right))
    ):
        raise ValueError(
            "Convolution edge masses must be finite and nonnegative; "
            "the kernel's edge-mass companion or its parameters are "
            "likely invalid."
        )

    interior = kernel_values[operator.gather_idx] * operator.quad_weights
    row_sums = interior.sum(axis=1) + edge_mass_left + edge_mass_right
    if np.any(row_sums <= 0):
        raise ValueError(
            "Convolution kernel matrix has non-positive row sums; the "
            "kernel parameters are likely invalid (zero width)."
        )
    y_conv = interior @ y_arr + edge_mass_left * y_arr[0]
    y_conv += edge_mass_right * y_arr[-1]
    return cast("NDArray[np.float64]", y_conv / row_sums)


#
def my_conv(
    x: ArrayLike,
    y: ArrayLike,
    kernel: ArrayLike,
    method: Literal["scipy", "numpy"] = "scipy",
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
    if x_arr.size < 2:
        raise ValueError(
            "my_conv requires at least 2 x samples to determine the "
            f"step size, got {x_arr.size}"
        )
    pad_size = int(kernel_arr.size / 2)

    # Add padding to minimize edge artifacts (y only; the x grid is not
    # needed for the convolution itself)
    y_pad = np.pad(y_arr, pad_size, mode="edge")

    # Normalize the kernel (cheaper than dividing the padded signal)
    kernel_sum = np.sum(kernel_arr)
    if kernel_sum == 0 or not np.isfinite(kernel_sum):
        raise ValueError(
            f"my_conv kernel sums to {kernel_sum}; cannot normalize. "
            "The kernel likely collapsed below the axis step "
            "(width parameter too small) or contains non-finite values."
        )
    kernel_norm = kernel_arr / kernel_sum

    # Compute convolution with normalized kernel
    if method == "scipy":
        y_conv_pad = np.asarray(convolve(y_pad, kernel_norm, mode="same"), dtype=float)
    elif method == "numpy":
        y_conv_pad = np.asarray(
            np.convolve(y_pad, kernel_norm, mode="same"), dtype=float
        )
    else:
        raise ValueError(f"Unknown method '{method}'")

    # Remove padding and return
    if pad_size == 0:
        return cast("NDArray[np.float64]", y_conv_pad)
    return cast("NDArray[np.float64]", y_conv_pad[pad_size:-pad_size])


#
def phi_norm(phi: float, norm: float = 2 * np.pi) -> float:
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

    return phi - norm * math.floor(phi / norm)


#
def running_mean(x: ArrayLike, y: ArrayLike, n: int) -> NDArray[np.float64]:
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
    n : int
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
    - cumsum: floating point errors for n > 1E5
    - scipy with padding: fastest and most robust (this implementation)
    """

    return my_conv(x, y, np.ones(n))
