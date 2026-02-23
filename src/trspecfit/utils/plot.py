"""
Plotting utilities for trspecfit.

This module provides matplotlib-based plotting functions for:
- 1D spectroscopy data (energy- or time-resolved)
- 2D spectroscopy data (time- and energy-resolved)
- Image display and grid layouts
- Matplotlib helper utilities for axis formatting
"""

import pathlib
from collections.abc import Sequence
from typing import Any, Union

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray

from trspecfit.config.plot import PlotConfig
from trspecfit.utils.arrays import OoM

PathLike = Union[str, pathlib.Path]

#
# Image display utilities
#

#
def load_plot(path: PathLike, dpi_fig: int = 75) -> None:
    """
    Load and display a saved figure as an image.
    
    Displays a saved plot file without axes or borders, useful for
    showing previously generated figures in notebooks or reports.

    Parameters
    ----------
    path : str or Path
        Path to the image file to load
    dpi_fig : int, default=75
        Display DPI (actual DPI multiplied by 1.25)
    """
    # 1.25x factor accounts for typical whitespace/margins in saved figures
    fig, ax = plt.subplots(1, 1, dpi=1.25*dpi_fig)
    img = mpimg.imread(path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

#
def load_plot_grid(paths: Sequence[PathLike], columns: int = 3, fig_width: float = 16,
                   debug: bool = False) -> None:
    """
    Load and display multiple images in a grid layout.
    
    Convenience wrapper for load_plot + plot_grid to display multiple
    saved figures together for comparison.
    
    Parameters
    ----------
    paths : list of str
        List of paths to image files
    columns : int, default=3
        Number of columns in grid
    fig_width : float, default=16
        Total figure width in inches
    debug : bool, default=False
        Print layout info if debug is True
    """
    images = [plt.imread(path) for path in paths]
    plot_grid(images, columns, fig_width, debug)

#
def plot_grid(images: Sequence[NDArray[np.generic]], columns: int = 3,
              fig_width: float = 16, debug: bool = False) -> None:
    """
    Display multiple images in a grid layout.
    
    Arranges images in a grid with automatic height calculation to maintain
    aspect ratios. Useful for comparing multiple plots side-by-side.
    
    Parameters
    ----------
    images : list of ndarray
        List of image arrays (e.g., from plt.imread or mpimg.imread)
    columns : int, default=3
        Number of columns in grid
    fig_width : float, default=16
        Total figure width in inches
    debug : bool, default=0
        If True, print layout calculations (rows, aspect ratio, height)
    
    Notes
    -----
    - Assumes all images have the same aspect ratio
    - Uses first image dimensions to calculate figure height
    - Axes are turned off for clean presentation
    """
    rows = np.ceil(len(images) / columns).astype(int)
    
    # Calculate figure height to maintain aspect ratio
    img_shape = np.shape(images[0])
    ratio = img_shape[1] / img_shape[0]  # width/height
    fig_height = fig_width * rows / (ratio * columns)
    
    if debug >= 1:
        print(f'rows {rows}')
        print(f'image shape {img_shape}')
        print(f'aspect ratio {ratio}')
        print(f'figure height {fig_height}')
    
    # Create grid
    _, axs = plt.subplots(rows, columns, figsize=(fig_width, fig_height))
    axs = axs.flatten()
    
    # Display images
    for image, ax in zip(images, axs):
        ax.imshow(image)
        ax.set_axis_off()
    
    # Hide unused subplots
    for ax in axs[len(images):]:
        ax.set_axis_off()
    
    plt.subplots_adjust(hspace=0, wspace=0.05)
    plt.show()

#
# Main plotting functions
#

#
def plot_2D(
    data: ArrayLike,
    x: ArrayLike | None = None,
    y: ArrayLike | None = None,
    config: PlotConfig | None = None,
    **kwargs: Any
) -> None:
    """
    Plot 2D spectroscopy data as a color map.
    
    Creates a pseudocolor plot (heatmap) for 2D time- and energy-resolved
    spectroscopy data with extensive customization options.
    
    Parameters
    ----------
    data : 2D array
        Data to plot as heatmap (shape: [y_points, x_points])
    x : array-like, optional
        X-axis (energy) coordinates. If None, uses column indices.
    y : array-like, optional
        Y-axis (time) coordinates. If None, uses row indices.
    config : PlotConfig, optional
        Configuration object with plot settings. If None, uses defaults.
    **kwargs : dict
        Override any config attributes for this specific plot.
        
        Common options:
        - x_label, y_label, title : Axis labels and title
        - x_lim, y_lim : Axis display limits (coordinate values, not indices)
        - z_lim : Color scale limits [min, max] or [0, 'max'] for auto-max
        - x_dir, y_dir : 'def' or 'rev' for axis direction
        - x_type, y_type : 'lin' or 'log' for axis scale
        - z_colormap : Colormap name (default 'viridis')
        - z_colorbar : 'ver' or 'hor' for colorbar orientation
        - data_slice : [[x_start, x_stop], [y_start, y_stop]] for slicing by index
        - vlines, hlines : List of coordinates for reference lines
        - ticksize : Font size for tick labels
        - dpi_plot, dpi_save : Display and save resolution
        - save_img : 0 (display), 1 (save+display), -1 (save only)
        - save_path : Path for saving figure
    
    Examples
    --------
    >>> # Basic plot
    >>> plot_2D(data, x=energy, y=time)
    
    >>> # With configuration
    >>> config = PlotConfig.from_project(project)
    >>> plot_2D(data, x, y, config=config)
    
    >>> # Slice data and set color scale
    >>> plot_2D(data, x, y, config=config,
    ...         data_slice=[[10, 100], [5, 50]],
    ...         z_lim=[0, 100])
    
    >>> # Reversed energy axis with reference lines
    >>> plot_2D(data, x, y, config=config,
    ...         x_dir='rev',
    ...         vlines=[85.0, 87.5],
    ...         hlines=[0, 100])
    
    Notes
    -----
    - data_slice uses INDEX-based slicing, not coordinate values
    - x_lim/y_lim control display zoom using COORDINATE values
    - Color scale can be [min, max], [0, 'max'] for auto-max, or None for auto-both
    - Reference lines (vlines/hlines) use coordinate values, not indices
    """
    # Use default config if none provided
    if config is None:
        config = PlotConfig()

    data_arr = np.asarray(data)
    if data_arr.ndim != 2:
        raise ValueError("data must be a 2D array")
    x_arr = None if x is None else np.asarray(x)
    y_arr = None if y is None else np.asarray(y)
    
    # Extract settings from config, allowing kwargs to override
    x_label = kwargs.get('x_label', config.x_label)
    y_label = kwargs.get('y_label', config.y_label)
    title = kwargs.get('title', config.title)
    x_dir = kwargs.get('x_dir', config.x_dir)
    x_type = kwargs.get('x_type', config.x_type)
    y_dir = kwargs.get('y_dir', config.y_dir)
    y_type = kwargs.get('y_type', config.y_type)
    x_lim = kwargs.get('x_lim', config.x_lim)
    y_lim = kwargs.get('y_lim', config.y_lim)
    z_lim = kwargs.get('z_lim', config.z_lim)
    dpi_plot = kwargs.get('dpi_plot', config.dpi_plot)
    dpi_save = kwargs.get('dpi_save', config.dpi_save)
    z_colormap = kwargs.get('z_colormap', config.z_colormap)
    z_colorbar = kwargs.get('z_colorbar', config.z_colorbar)
    vlines = kwargs.get('vlines', config.vlines)
    hlines = kwargs.get('hlines', config.hlines)
    ticksize = kwargs.get('ticksize', config.ticksize)
    save_img = kwargs.get('save_img', 0)
    save_path = kwargs.get('save_path', '')
    
    # Data slicing
    data_slice = kwargs.get('data_slice', config.data_slice)
    fig_size = kwargs.get('fig_size', [])
    
    # Slice data if requested
    data_plt = data_arr
    x_plt = x_arr
    y_plt = y_arr
    
    if data_slice is not None:
        x_slice = data_slice[0] if len(data_slice) > 0 else []
        y_slice = data_slice[1] if len(data_slice) > 1 else []
        
        if len(x_slice) > 0 and len(y_slice) > 0:
            data_plt = data_arr[y_slice[0]:y_slice[1], x_slice[0]:x_slice[1]]
            if x_arr is not None:
                x_plt = x_arr[x_slice[0]:x_slice[1]]
            if y_arr is not None:
                y_plt = y_arr[y_slice[0]:y_slice[1]]
        elif len(y_slice) > 0:  # y only
            data_plt = data_arr[y_slice[0]:y_slice[1], :]
            if y_arr is not None:
                y_plt = y_arr[y_slice[0]:y_slice[1]]
        elif len(x_slice) > 0:  # x only
            data_plt = data_arr[:, x_slice[0]:x_slice[1]]
            if x_arr is not None:
                x_plt = x_arr[x_slice[0]:x_slice[1]]
    
    # Determine z-axis (color) range
    if z_lim is None:
        min2D = np.min(data_plt)
        max2D = np.max(data_plt)
        scale_txt = 'autoscale min. and max. z (color)'
    elif isinstance(z_lim, list) and len(z_lim) == 2 and z_lim[1] == 'max':
        min2D = z_lim[0]
        max2D = np.max(data_plt)
        scale_txt = f'autoscale max. z (color) [min={z_lim[0]}]'
    else:
        min2D = z_lim[0]
        max2D = z_lim[1]
        scale_txt = 'user defined z scale (color)'
    
    # Create default axes if not provided
    if x_plt is None:
        x_plt = np.arange(0, np.shape(data_plt)[1], 1)
    if y_plt is None:
        y_plt = np.arange(0, np.shape(data_plt)[0], 1)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, dpi=dpi_plot)
    if len(fig_size) != 0:
        fig.set_size_inches(fig_size[0], fig_size[1], forward=True)
    
    # Title
    plot_title = title
    if plot_title:
        plot_title += '\n'
    plot_title += f'{scale_txt}\nsize 2D data set: {np.shape(data_plt)}'
    plt.title(plot_title, loc='left', fontsize=10)
    
    # Set axis labels
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    
    # Plot data
    plt.pcolormesh(x_plt, y_plt, data_plt,
                   cmap=z_colormap,
                   vmin=min2D, vmax=max2D,
                   shading='nearest')
    
    # Colorbar
    if z_colorbar == 'ver':
        cbar = plt.colorbar(orientation='vertical')
    elif z_colorbar == 'hor':
        cbar = plt.colorbar(orientation='horizontal')
    else:
        cbar = None
    
    # Set tick label font size
    if ticksize is not None:
        ax.tick_params(axis='both', which='major', labelsize=ticksize)
        if cbar is not None:
            cbar.ax.tick_params(labelsize=ticksize)
    
    # Axis settings
    if x_type == 'log':
        ax.set_xscale('log')
    if x_lim is not None:
        ax.set_xlim(x_lim[0], x_lim[1])
    if x_dir == 'rev':
        plt.gca().invert_xaxis()
    if y_type == 'log':
        ax.set_yscale('log')
    if y_lim is not None:
        ax.set_ylim(y_lim[0], y_lim[1])
    if y_dir == 'rev':
        plt.gca().invert_yaxis()
    
    # Reference lines
    if hlines is not None:
        plt.hlines(y=np.asarray(hlines),
                   xmin=np.min(x_plt), xmax=np.max(x_plt),
                   color='#000000', linestyle=':')
    
    if vlines is not None:
        plt.vlines(x=np.asarray(vlines),
                   ymin=np.min(y_plt), ymax=np.max(y_plt),
                   color='#000000', linestyle=':')
    
    # Save figure
    if abs(save_img) == 1:
        img_save(save_path, dpi_save)
    
    # Show/close plot
    if save_img >= 0:
        plt.show()
    else:
        plt.close()

#
def plot_1D(
    data: Sequence[ArrayLike] | ArrayLike,
    x: ArrayLike | list[ArrayLike] | None = None,
    config: PlotConfig | None = None,
    **kwargs: Any
) -> None:
    """
    Plot 1D spectroscopy data with extensive customization.
    
    Creates line plots for 1D energy-resolved or time-resolved spectroscopy
    data with support for multiple traces, styling, and normalization.
    
    Parameters
    ----------
    data : list of arrays or 2D array
        Data to plot. Either a list of 1D arrays or a 2D array where
        each row is a separate trace.
    x : array-like or list of arrays, optional
        X-axis data. Can be a single array (used for all traces) or a list
        of arrays (one per trace). If None, uses indices.
    config : PlotConfig, optional
        Configuration object with plot settings. If None, uses defaults.
    **kwargs : dict
        Override any config attributes for this specific plot.
        
        Common options:
        - x_label, y_label, title : Axis labels and title
        - x_lim, y_lim : Axis display limits
        - x_dir, y_dir : 'def' or 'rev' for axis direction
        - x_type, y_type : 'lin' or 'log' for axis scale
        - colors : List of colors for each trace
        - linestyles : List of line styles ('-', '--', ':', etc.)
        - linewidths : List of line widths
        - markers : List of marker styles ('o', 's', '^', etc.)
        - markersizes : List of marker sizes
        - legend : List of legend labels
        - waterfall : Y-offset between traces for waterfall display
        - y_norm : 0 (raw data) or 1 (normalize each trace to [0, 1])
        - y_scale : List of scaling factors for each trace
        - vlines, hlines : List of coordinates for reference lines
        - ticksize : Font size for tick labels
        - dpi_plot, dpi_save : Display and save resolution
        - save_img : 0 (display), 1 (save+display), -1 (save only)
        - save_path : Path for saving figure
    
    Examples
    --------
    >>> # Simple plot
    >>> plot_1D([data1, data2], x=energy)
    
    >>> # With project configuration
    >>> config = PlotConfig.from_project(project)
    >>> plot_1D(data, x, config=config)
    
    >>> # Waterfall plot with custom styling
    >>> plot_1D([trace1, trace2, trace3], x=time,
    ...         waterfall=0.5,
    ...         colors=['red', 'blue', 'green'],
    ...         legend=['Early', 'Mid', 'Late'])
    
    >>> # Normalized traces with reversed x-axis
    >>> plot_1D(data, x=energy, config=config,
    ...         y_norm=1, x_dir='rev',
    ...         vlines=[85.0, 87.5])
    
    Notes
    -----
    - waterfall parameter adds vertical offset between traces
    - y_norm=1 normalizes each trace independently to [0, 1]
    - y_scale allows scaling individual traces (e.g., [1, 0.5, 2])
    - If data is 2D array, each row is treated as a separate trace
    """
    # Use default config if none provided
    if config is None:
        config = PlotConfig()

    if isinstance(data, Sequence) and not isinstance(data, np.ndarray):
        data_series = [np.asarray(trace, dtype=float) for trace in data]
    else:
        data_arr = np.asarray(data, dtype=float)
        if data_arr.ndim == 1:
            data_series = [data_arr]
        elif data_arr.ndim == 2:
            data_series = [data_arr[i, :] for i in range(data_arr.shape[0])]
        else:
            raise ValueError("data must be a 1D/2D array or a sequence of 1D arrays")
    
    # Extract settings from config, allowing kwargs to override
    x_label = kwargs.get('x_label', config.x_label)
    y_label = kwargs.get('y_label', config.y_label)
    title = kwargs.get('title', config.title)
    x_dir = kwargs.get('x_dir', config.x_dir)
    x_type = kwargs.get('x_type', config.x_type)
    y_dir = kwargs.get('y_dir', config.y_dir)
    y_type = kwargs.get('y_type', config.y_type)
    x_lim = kwargs.get('x_lim', config.x_lim)
    y_lim = kwargs.get('y_lim', config.y_lim)
    dpi_plot = kwargs.get('dpi_plot', config.dpi_plot)
    dpi_save = kwargs.get('dpi_save', config.dpi_save)
    waterfall = kwargs.get('waterfall', config.waterfall)
    y_norm = kwargs.get('y_norm', config.y_norm)
    ticksize = kwargs.get('ticksize', config.ticksize)
    save_img = kwargs.get('save_img', 0)
    save_path = kwargs.get('save_path', '')
    
    # Get style settings with smart defaults
    colors = kwargs.get('colors', config.colors)
    linestyles = kwargs.get('linestyles', config.linestyles)
    linewidths = kwargs.get('linewidths', config.linewidths)
    markers = kwargs.get('markers', config.markers)
    markersizes = kwargs.get('markersizes', config.markersizes)
    legend = kwargs.get('legend', config.legend)
    vlines = kwargs.get('vlines', config.vlines)
    hlines = kwargs.get('hlines', config.hlines)
    y_scale = kwargs.get('y_scale', config.y_scale)
    
    # Determine number of plots
    N_plots = len(data_series)
    
    # Create default values if not provided
    if linestyles is None:
        linestyles = N_plots * ['-']
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if linewidths is None:
        linewidths = N_plots * [1.5]
    if markers is None:
        markers = N_plots * [None]
    if markersizes is None:
        markersizes = N_plots * [6]
    if x is None:
        x_common = np.arange(0, data_series[0].shape[0], 1)
        x_list: list[NDArray[np.float64]] | None = None
    elif isinstance(x, list):
        x_common = None
        x_list = [np.asarray(xi, dtype=float) for xi in x]
    else:
        x_common = np.asarray(x, dtype=float)
        x_list = None
    if y_scale is None:
        y_scale_arr = np.ones(N_plots, dtype=float)
    else:
        y_scale_arr = np.asarray(y_scale, dtype=float)
    if legend is None:
        legend = [i + 1 for i in range(N_plots)]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, dpi=dpi_plot)
    
    # Title
    plot_title = title
    if y_norm == 1:
        plot_title += '\n(all data normalized to baseline 0 and amplitude 1 [each])'
    plt.title(plot_title, loc='left', fontsize=10)
    
    # Plot each dataset
    for i in range(N_plots):
        x_plot = x_list[i] if x_list is not None else x_common
        if x_plot is None:
            raise ValueError("x axis could not be determined")
        y_data = data_series[i]
        
        # Normalize if requested
        if y_norm == 1:
            y_plot = (y_data - np.min(y_data)) / (np.max(y_data - np.min(y_data)))
            y_plot = y_plot + i * waterfall
        else:
            y_plot = y_scale_arr[i] * y_data + i * waterfall
        
        # Plot
        label = f'{y_scale_arr[i]}*{legend[i]}' if y_scale_arr[i] != 1 else str(legend[i])
        ax.plot(x_plot, y_plot,
                ls=linestyles[i],
                c=colors[i % len(colors)],
                lw=linewidths[i],
                marker=markers[i],
                ms=markersizes[i],
                label=label)
    
    # Set axis labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    # Reference lines
    if hlines is not None:
        if x_list is not None:
            x_minmax = [
                np.min([np.min(x_list[i]) for i in range(N_plots)]),
                np.max([np.max(x_list[i]) for i in range(N_plots)]),
            ]
        else:
            if x_common is None:
                raise ValueError("x axis could not be determined")
            x_minmax = [np.min(x_common), np.max(x_common)]
        plt.hlines(y=np.asarray(hlines),
                   xmin=x_minmax[0], xmax=x_minmax[1],
                   color='#808080', linestyle=':')
    
    if vlines is not None:
        if y_norm == 1:
            y_minmax = [0, 1]
        else:
            y_minmax = [
                np.min([np.min(y_scale_arr[i] * data_series[i]) for i in range(N_plots)]),
                np.max([np.max(y_scale_arr[i] * data_series[i]) for i in range(N_plots)]),
            ]
        plt.vlines(x=np.asarray(vlines),
                   ymin=y_minmax[0], ymax=y_minmax[1],
                   color='#808080', linestyle='--')
    
    # Axis settings
    if x_type == 'log':
        ax.set_xscale('log')
    if x_lim is not None:
        ax.set_xlim(x_lim[0], x_lim[1])
    if x_dir == 'rev':
        plt.gca().invert_xaxis()
    if y_type == 'log':
        ax.set_yscale('log')
    if y_lim is not None:
        ax.set_ylim(y_lim[0], y_lim[1])
    if y_dir == 'rev':
        plt.gca().invert_yaxis()
    
    # Tick size
    if ticksize is not None:
        ax.tick_params(axis='both', which='major', labelsize=ticksize)
    
    # Legend
    plt.legend(bbox_to_anchor=(1, 1))
    
    # Save
    if abs(save_img) == 1:
        img_save(save_path, dpi_save)
    
    # Show/close
    if save_img >= 0:
        plt.show()
    else:
        plt.close()

#
def img_save(save_path: PathLike, dpi: int = 300) -> None:
    """
    Save current matplotlib figure with sensible defaults.
    
    Wrapper around plt.savefig with tight bounding box to minimize whitespace,
    small padding (0.05 inches), white background, auto edge color.
    
    Parameters
    ----------
    save_path : str or Path
        Output file path (extension determines format: .png, .pdf, .svg, etc.)
    dpi : int, default=300
        Resolution in dots per inch
    """
    plt.savefig(save_path,
                dpi=dpi,
                bbox_inches='tight',
                pad_inches=0.05,
                facecolor='white',
                edgecolor='auto')

#
# Matplotlib helper functions
#

#
def major_locator_input(x: ArrayLike) -> float:
    """
    Calculate major tick spacing for matplotlib axis.
    
    Determines appropriate major tick spacing based on the order of magnitude
    of the data range. Useful for programmatically setting axis ticks.
    
    Parameters
    ----------
    x : array-like
        Data array for which to calculate tick spacing
    
    Returns
    -------
    float
        Major tick spacing (power of 10)
    """
    #
    x_max = float(np.max(np.asarray(x, dtype=float)))
    return float(10 ** OoM(x_max))

#
def minor_locator_input(x: ArrayLike) -> float:
    """
    Calculate minor tick spacing for matplotlib axis.
    
    Determines appropriate minor tick spacing as 1/10 of the major tick
    spacing, based on the order of magnitude of the data range.
    
    Parameters
    ----------
    x : array-like
        Data array for which to calculate tick spacing
    
    Returns
    -------
    float
        Minor tick spacing (1/10 of major spacing)
    """
    #
    x_max = float(np.max(np.asarray(x, dtype=float)))
    return float(10 ** (OoM(x_max) - 1))

#
def major_formatter_input(x: ArrayLike) -> str:
    """
    Generate format string for matplotlib axis tick labels.
    
    Creates appropriate numeric format string based on the order of magnitude
    of the data, ensuring readable tick labels without unnecessary precision.
    
    Parameters
    ----------
    x : array-like
        Data array for which to generate format string
    
    Returns
    -------
    str
        Format string for matplotlib tick labels (e.g., '%0.2f', '%4.0f')
    """
    axis_OoM = OoM(np.max(x))
    
    if axis_OoM < 0:
        return f'%0.{abs(axis_OoM)}f'
    elif axis_OoM == 0:
        return '%0.0f'
    else:  # axis_OoM > 0
        return f'%{axis_OoM + 1}.0f'
