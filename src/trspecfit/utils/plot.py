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
from typing import Any

import matplotlib.axes
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray

from trspecfit.config.plot import PlotConfig
from trspecfit.utils.arrays import oom

type PathLike = str | pathlib.Path

#
# Image display utilities
#


#
def load_plot(path: PathLike, dpi_fig: int = 75, *, save_img: int = 0) -> None:
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
    save_img : {-1, 0, 1}, default=0
        -1 save only, 0 display only, 1 save and display.
    """

    # 1.25x factor accounts for typical whitespace/margins in saved figures
    fig, _ax = plt.subplots(1, 1, dpi=1.25 * dpi_fig)
    img = mpimg.imread(path)
    plt.imshow(img)
    plt.axis("off")
    _finalize_plot(fig, save_img)


#
def load_plot_grid(
    paths: Sequence[PathLike],
    columns: int = 3,
    fig_width: float = 16,
    *,
    show_info: bool = False,
    save_img: int = 0,
    save_path: PathLike = "",
    dpi_save: int = 300,
) -> None:
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
    show_info : bool, default=False
        Print layout info.
    save_img : {-1, 0, 1}, default=0
        -1 save only, 0 display only, 1 save and display.
    save_path : str or Path, default=""
        File path for saving.
    dpi_save : int, default=300
        DPI for saved image.
    """

    images = [plt.imread(path) for path in paths]
    plot_grid(
        images,
        columns,
        fig_width,
        show_info=show_info,
        save_img=save_img,
        save_path=save_path,
        dpi_save=dpi_save,
    )


#
def plot_grid(
    images: Sequence[NDArray[np.generic]],
    columns: int = 3,
    fig_width: float = 16,
    *,
    show_info: bool = False,
    save_img: int = 0,
    save_path: PathLike = "",
    dpi_save: int = 300,
) -> None:
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
    show_info : bool, default=False
        If True, print layout calculations (rows, aspect ratio, height).
    save_img : {-1, 0, 1}, default=0
        -1 save only, 0 display only, 1 save and display.
    save_path : str or Path, default=""
        File path for saving.
    dpi_save : int, default=300
        DPI for saved image.

    Notes
    -----
    - Assumes all images have the same aspect ratio
    - Uses first image dimensions to calculate figure height
    - Axes are turned off for clean presentation
    """

    rows = np.ceil(len(images) / columns).astype(int)

    # Calculate figure height to maintain aspect ratio
    img_shape = images[0].shape
    ratio = img_shape[1] / img_shape[0]  # width/height
    fig_height = fig_width * rows / (ratio * columns)

    if show_info:
        print(f"rows {rows}")
        print(f"image shape {img_shape}")
        print(f"aspect ratio {ratio}")
        print(f"figure height {fig_height}")

    # Create grid
    fig, axs = plt.subplots(rows, columns, figsize=(fig_width, fig_height))
    axs = axs.flatten()

    # Display images
    for image, ax in zip(images, axs, strict=False):
        ax.imshow(image)
        ax.set_axis_off()

    # Hide unused subplots
    for ax in axs[len(images) :]:
        ax.set_axis_off()

    plt.subplots_adjust(hspace=0, wspace=0.05)
    _finalize_plot(fig, save_img, save_path, dpi_save)


#
def plot_2d_grid(
    datasets: Sequence[ArrayLike],
    *,
    x: ArrayLike | None = None,
    y: ArrayLike | None = None,
    titles: Sequence[str] | None = None,
    config: "PlotConfig | None" = None,
    columns: int | None = None,
    vlines: Sequence[Sequence[float]] | None = None,
    hlines: Sequence[Sequence[float]] | None = None,
    save_img: int = 0,
    save_path: PathLike = "",
    dpi_save: int = 300,
) -> None:
    """
    Plot multiple 2D datasets in a grid layout.

    Parameters
    ----------
    datasets : list of 2D arrays
        Data arrays to plot (each shape [n_time, n_energy]).
    x : array-like, optional
        Shared energy axis for all panels.
    y : array-like, optional
        Shared time axis for all panels.
    titles : list of str, optional
        Title for each panel. If None, panels are untitled.
    config : PlotConfig, optional
        Plot configuration (colormap, axis directions, labels, panel size,
        reference-line styling).
    columns : int, optional
        Number of grid columns. If None, auto-selected from the panel count.
    vlines : list of list of float, optional
        Per-panel vertical reference lines. Each element is a list of
        x-coordinates for that panel (or empty list for none).
    hlines : list of list of float, optional
        Per-panel horizontal reference lines.
    save_img : {-1, 0, 1}, default=0
        -1 save only, 0 display only, 1 save and display.
    save_path : str or Path, default=""
        File path for saving.
    dpi_save : int, default=300
        DPI for saved image.
    """

    from trspecfit.config.plot import PlotConfig as _PlotConfig

    if config is None:
        config = _PlotConfig()

    n = len(datasets)
    if n == 0:
        return

    if columns is not None:
        cols = columns
    # Auto-select columns: 2 for <=4, 3 for <=9, 4 for <=16, 5 above
    elif n <= 4:
        cols = 2
    elif n <= 9:
        cols = 3
    elif n <= 16:
        cols = 4
    else:
        cols = 5

    rows = int(np.ceil(n / cols))

    z_colormap = config.z_colormap or "viridis"
    x_arr = None if x is None else np.asarray(x)
    y_arr = None if y is None else np.asarray(y)

    panel_w, panel_h = config.panel_size
    fig, axs = plt.subplots(
        rows,
        cols,
        figsize=(panel_w * cols, panel_h * rows),
        squeeze=False,
        dpi=config.dpi_plot or 100,
    )

    for idx, (data_raw, ax) in enumerate(zip(datasets, axs.flatten(), strict=False)):
        data_arr = np.asarray(data_raw)
        xp = x_arr if x_arr is not None else np.arange(data_arr.shape[1])
        yp = y_arr if y_arr is not None else np.arange(data_arr.shape[0])

        im = ax.pcolormesh(xp, yp, data_arr, cmap=z_colormap, shading="nearest")
        fig.colorbar(im, ax=ax, pad=0.02)

        if titles is not None and idx < len(titles):
            ax.set_title(titles[idx], fontsize=10)

        # Axis labels only on edges
        if idx % cols == 0:
            ax.set_ylabel(config.y_label or "")
        if idx >= (rows - 1) * cols:
            ax.set_xlabel(config.x_label or "")

        _apply_axis_settings(
            ax,
            config.x_type,
            config.x_dir,
            config.y_type,
            config.y_dir,
            config.x_lim,
            config.y_lim,
        )
        if config.ticksize is not None:
            ax.tick_params(labelsize=config.ticksize)

        # Reference lines
        if vlines is not None and idx < len(vlines) and vlines[idx]:
            ax.vlines(
                x=np.asarray(vlines[idx]),
                ymin=np.min(yp),
                ymax=np.max(yp),
                color=config.refline_color,
                linestyle=config.refline_style,
            )
        if hlines is not None and idx < len(hlines) and hlines[idx]:
            ax.hlines(
                y=np.asarray(hlines[idx]),
                xmin=np.min(xp),
                xmax=np.max(xp),
                color=config.refline_color,
                linestyle=config.refline_style,
            )

    # Hide unused subplots
    for ax in axs.flatten()[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    _finalize_plot(fig, save_img, save_path, dpi_save)


#
# Main plotting functions
#


#
def plot_2d(
    data: ArrayLike,
    x: ArrayLike | None = None,
    y: ArrayLike | None = None,
    config: PlotConfig | None = None,
    **kwargs: Any,
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
        - z_type : 'lin' or 'log' for color scale type
        - z_colorbar : 'ver' or 'hor' for colorbar orientation
        - data_slice : [[x_start, x_stop], [y_start, y_stop]] for slicing by index
        - vlines, hlines : List of coordinates for reference lines
        - refline_color, refline_style : Reference-line color and line style
        - ticksize : Font size for tick labels
        - dpi_plot, dpi_save : Display and save resolution
        - save_img : 0 (display), 1 (save+display), -1 (save only)
        - save_path : Path for saving figure

    Examples
    --------
    >>> # Basic plot
    >>> plot_2d(data, x=energy, y=time)

    >>> # With configuration
    >>> config = project.plot_config
    >>> plot_2d(data, x, y, config=config)

    >>> # Slice data and set color scale
    >>> plot_2d(data, x, y, config=config,
    ...         data_slice=[[10, 100], [5, 50]],
    ...         z_lim=[0, 100])

    >>> # Reversed energy axis with reference lines
    >>> plot_2d(data, x, y, config=config,
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
    x_label = kwargs.get("x_label", config.x_label)
    y_label = kwargs.get("y_label", config.y_label)
    title = kwargs.get("title", config.title)
    x_dir = kwargs.get("x_dir", config.x_dir)
    x_type = kwargs.get("x_type", config.x_type)
    y_dir = kwargs.get("y_dir", config.y_dir)
    y_type = kwargs.get("y_type", config.y_type)
    x_lim = kwargs.get("x_lim", config.x_lim)
    y_lim = kwargs.get("y_lim", config.y_lim)
    z_lim = kwargs.get("z_lim", config.z_lim)
    dpi_plot = kwargs.get("dpi_plot", config.dpi_plot)
    dpi_save = kwargs.get("dpi_save", config.dpi_save)
    z_colormap = kwargs.get("z_colormap", config.z_colormap)
    z_colorbar = kwargs.get("z_colorbar", config.z_colorbar)
    z_type = kwargs.get("z_type", config.z_type)
    vlines = kwargs.get("vlines", config.vlines)
    hlines = kwargs.get("hlines", config.hlines)
    refline_color = kwargs.get("refline_color", config.refline_color)
    refline_style = kwargs.get("refline_style", config.refline_style)
    ticksize = kwargs.get("ticksize", config.ticksize)
    save_img = kwargs.get("save_img", 0)
    save_path = kwargs.get("save_path", "")

    # Data slicing
    data_slice = kwargs.get("data_slice", config.data_slice)
    fig_size = kwargs.get("fig_size", [])

    # Slice data if requested
    data_plt = data_arr
    x_plt = x_arr
    y_plt = y_arr

    if data_slice is not None:
        x_slice = data_slice[0] if len(data_slice) > 0 else []
        y_slice = data_slice[1] if len(data_slice) > 1 else []

        if len(x_slice) > 0 and len(y_slice) > 0:
            data_plt = data_arr[y_slice[0] : y_slice[1], x_slice[0] : x_slice[1]]
            if x_arr is not None:
                x_plt = x_arr[x_slice[0] : x_slice[1]]
            if y_arr is not None:
                y_plt = y_arr[y_slice[0] : y_slice[1]]
        elif len(y_slice) > 0:  # y only
            data_plt = data_arr[y_slice[0] : y_slice[1], :]
            if y_arr is not None:
                y_plt = y_arr[y_slice[0] : y_slice[1]]
        elif len(x_slice) > 0:  # x only
            data_plt = data_arr[:, x_slice[0] : x_slice[1]]
            if x_arr is not None:
                x_plt = x_arr[x_slice[0] : x_slice[1]]

    # Determine z-axis (color) range
    if z_lim is None:
        min_2d = np.min(data_plt)
        max_2d = np.max(data_plt)
        scale_txt = "autoscale min. and max. z (color)"
    elif isinstance(z_lim, list) and len(z_lim) == 2 and z_lim[1] == "max":
        min_2d = z_lim[0]
        max_2d = np.max(data_plt)
        scale_txt = f"autoscale max. z (color) [min={z_lim[0]}]"
    else:
        min_2d = z_lim[0]
        max_2d = z_lim[1]
        scale_txt = "user defined z scale (color)"

    # Create default axes if not provided
    if x_plt is None:
        x_plt = np.arange(data_plt.shape[1])
    if y_plt is None:
        y_plt = np.arange(data_plt.shape[0])

    # Create figure
    fig, ax = plt.subplots(1, 1, dpi=dpi_plot)
    if len(fig_size) != 0:
        fig.set_size_inches(fig_size[0], fig_size[1], forward=True)

    # Title
    plot_title = title
    if plot_title:
        plot_title += "\n"
    plot_title += f"{scale_txt}\nsize 2D data set: {data_plt.shape}"
    plt.title(plot_title, loc="left", fontsize=10)

    # Set axis labels
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    # Plot data
    if z_type == "log":
        norm = mcolors.LogNorm(vmin=min_2d, vmax=max_2d)
        plt.pcolormesh(
            x_plt, y_plt, data_plt, cmap=z_colormap, norm=norm, shading="nearest"
        )
    else:
        plt.pcolormesh(
            x_plt,
            y_plt,
            data_plt,
            cmap=z_colormap,
            vmin=min_2d,
            vmax=max_2d,
            shading="nearest",
        )

    # Colorbar
    if z_colorbar == "ver":
        cbar = plt.colorbar(orientation="vertical")
    elif z_colorbar == "hor":
        cbar = plt.colorbar(orientation="horizontal")
    else:
        cbar = None

    # Set tick label font size
    if ticksize is not None:
        ax.tick_params(axis="both", which="major", labelsize=ticksize)
        if cbar is not None:
            cbar.ax.tick_params(labelsize=ticksize)

    # Axis settings
    _apply_axis_settings(ax, x_type, x_dir, y_type, y_dir, x_lim, y_lim)

    # Reference lines
    if hlines is not None:
        plt.hlines(
            y=np.asarray(hlines),
            xmin=np.min(x_plt),
            xmax=np.max(x_plt),
            color=refline_color,
            linestyle=refline_style,
        )

    if vlines is not None:
        plt.vlines(
            x=np.asarray(vlines),
            ymin=np.min(y_plt),
            ymax=np.max(y_plt),
            color=refline_color,
            linestyle=refline_style,
        )

    # Save/show/close
    _finalize_plot(fig, save_img, save_path, dpi_save)


#
def plot_1d(
    data: Sequence[ArrayLike] | ArrayLike,
    x: ArrayLike | list[ArrayLike] | None = None,
    config: PlotConfig | None = None,
    **kwargs: Any,
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
        - alphas : List of opacity values (0–1) for each trace
        - legend : List of legend labels
        - waterfall : Y-offset between traces for waterfall display
        - y_norm : 0 (raw data) or 1 (normalize each trace to [0, 1])
        - y_scale : List of scaling factors for each trace
        - vlines, hlines : List of coordinates for reference lines
        - refline_color, refline_style : Reference-line color and line style
        - ticksize : Font size for tick labels
        - dpi_plot, dpi_save : Display and save resolution
        - save_img : 0 (display), 1 (save+display), -1 (save only)
        - save_path : Path for saving figure

    Examples
    --------
    >>> # Simple plot
    >>> plot_1d([data1, data2], x=energy)

    >>> # With project configuration
    >>> config = project.plot_config
    >>> plot_1d(data, x, config=config)

    >>> # Waterfall plot with custom styling
    >>> plot_1d([trace1, trace2, trace3], x=time,
    ...         waterfall=0.5,
    ...         colors=['red', 'blue', 'green'],
    ...         legend=['Early', 'Mid', 'Late'])

    >>> # Normalized traces with reversed x-axis
    >>> plot_1d(data, x=energy, config=config,
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
    x_label = kwargs.get("x_label", config.x_label)
    y_label = kwargs.get("y_label", config.y_label)
    title = kwargs.get("title", config.title)
    x_dir = kwargs.get("x_dir", config.x_dir)
    x_type = kwargs.get("x_type", config.x_type)
    y_dir = kwargs.get("y_dir", config.y_dir)
    y_type = kwargs.get("y_type", config.y_type)
    x_lim = kwargs.get("x_lim", config.x_lim)
    y_lim = kwargs.get("y_lim", config.y_lim)
    dpi_plot = kwargs.get("dpi_plot", config.dpi_plot)
    dpi_save = kwargs.get("dpi_save", config.dpi_save)
    waterfall = kwargs.get("waterfall", config.waterfall)
    y_norm = kwargs.get("y_norm", config.y_norm)
    ticksize = kwargs.get("ticksize", config.ticksize)
    save_img = kwargs.get("save_img", 0)
    save_path = kwargs.get("save_path", "")

    # Get style settings with smart defaults
    colors = kwargs.get("colors", config.colors)
    linestyles = kwargs.get("linestyles", config.linestyles)
    linewidths = kwargs.get("linewidths", config.linewidths)
    markers = kwargs.get("markers", config.markers)
    markersizes = kwargs.get("markersizes", config.markersizes)
    alphas = kwargs.get("alphas", config.alphas)
    legend = kwargs.get("legend", config.legend)
    vlines = kwargs.get("vlines", config.vlines)
    hlines = kwargs.get("hlines", config.hlines)
    refline_color = kwargs.get("refline_color", config.refline_color)
    refline_style = kwargs.get("refline_style", config.refline_style)
    y_scale = kwargs.get("y_scale", config.y_scale)

    # Determine number of plots
    n_plots = len(data_series)

    # Create default values if not provided
    if linestyles is None:
        linestyles = n_plots * ["-"]
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if linewidths is None:
        linewidths = n_plots * [1.5]
    if markers is None:
        markers = n_plots * [None]
    if markersizes is None:
        markersizes = n_plots * [6]
    if alphas is None:
        alphas = n_plots * [1.0]
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
        y_scale_arr = np.ones(n_plots, dtype=float)
    else:
        y_scale_arr = np.asarray(y_scale, dtype=float)
    if legend is None:
        legend = [i + 1 for i in range(n_plots)]

    # Create figure
    fig, ax = plt.subplots(1, 1, dpi=dpi_plot)

    # Title
    plot_title = title
    if y_norm == 1:
        plot_title += "\n(all data normalized to baseline 0 and amplitude 1 [each])"
    plt.title(plot_title, loc="left", fontsize=10)

    # Plot each dataset
    for i in range(n_plots):
        x_plot = x_list[i] if x_list is not None else x_common
        if x_plot is None:
            raise ValueError("x axis could not be determined")
        y_data = data_series[i]

        # Normalize if requested; a constant trace has zero amplitude and
        # maps to baseline 0 (dividing by its zero range would yield NaNs)
        if y_norm == 1:
            y_range = np.max(y_data) - np.min(y_data)
            if y_range == 0:
                y_plot = np.zeros_like(y_data, dtype=float)
            else:
                y_plot = (y_data - np.min(y_data)) / y_range
            y_plot = y_plot + i * waterfall
        else:
            y_plot = y_scale_arr[i] * y_data + i * waterfall

        # Plot
        label = (
            f"{y_scale_arr[i]}*{legend[i]}" if y_scale_arr[i] != 1 else str(legend[i])
        )
        ax.plot(
            x_plot,
            y_plot,
            ls=linestyles[i],
            c=colors[i % len(colors)],
            lw=linewidths[i],
            marker=markers[i],
            ms=markersizes[i],
            alpha=alphas[i],
            label=label,
        )

    # Set axis labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Reference lines
    if hlines is not None:
        if x_list is not None:
            x_minmax = [
                np.min([np.min(x_list[i]) for i in range(n_plots)]),
                np.max([np.max(x_list[i]) for i in range(n_plots)]),
            ]
        else:
            if x_common is None:
                raise ValueError("x axis could not be determined")
            x_minmax = [np.min(x_common), np.max(x_common)]
        plt.hlines(
            y=np.asarray(hlines),
            xmin=x_minmax[0],
            xmax=x_minmax[1],
            color=refline_color,
            linestyle=refline_style,
        )

    if vlines is not None:
        if y_norm == 1:
            y_minmax = [0, 1]
        else:
            y_minmax = [
                np.min(
                    [np.min(y_scale_arr[i] * data_series[i]) for i in range(n_plots)]
                ),
                np.max(
                    [np.max(y_scale_arr[i] * data_series[i]) for i in range(n_plots)]
                ),
            ]
        plt.vlines(
            x=np.asarray(vlines),
            ymin=y_minmax[0],
            ymax=y_minmax[1],
            color=refline_color,
            linestyle=refline_style,
        )

    # Axis settings
    _apply_axis_settings(ax, x_type, x_dir, y_type, y_dir, x_lim, y_lim)

    # Tick size
    if ticksize is not None:
        ax.tick_params(axis="both", which="major", labelsize=ticksize)

    # Legend
    plt.legend(bbox_to_anchor=(1, 1))

    # Save/show/close
    _finalize_plot(fig, save_img, save_path, dpi_save)


#
def plot_fit_panel_1d(
    *,
    x: NDArray[Any],
    observed: NDArray[Any],
    fit: NDArray[Any],
    components: NDArray[Any] | None = None,
    component_names: Sequence[str] | None = None,
    fit_ini: NDArray[Any] | None = None,
    show_init: bool = True,
    roi: list[int] | None = None,
    title: str = "",
    x_label: str = "energy",
    x_dir: str = "def",
    show_plot: bool = True,
    save_path: PathLike | None = None,
    config: PlotConfig | None = None,
    **kwargs: Any,
) -> Any:
    """
    Observed + fit (with components, when given) over a residual panel.

    Array-in/figure-out renderer for a single persisted (or live-derived)
    1D fit result — used both for a whole baseline/spectrum/2d fit and for
    one Slice-by-Slice slice. ``components`` (shape
    ``(n_components, x.size)``) draws the per-component decomposition
    plus a black sum line; without it, ``fit`` is drawn alone. ``fit_ini``
    draws the dotted-gold initial-guess overlay when ``show_init``. ``roi``
    draws dashed boundary lines at the given ``[start, stop)`` index
    window (for full-range reconstructions where ``x`` spans more than
    the fit window). NaN entries in any array leave a gap rather than a
    fabricated value. Styling (intensity label ``z_label``, ``dpi_plot``,
    ``dpi_save``) resolves from ``config`` with per-call ``**kwargs``
    overrides; ``x_label`` / ``x_dir`` stay explicit because they are
    data-dependent (energy axis vs index fallback).

    Returns the built ``Figure`` regardless of ``show_plot``/``save_path``
    — callers that only display or save still get it back (e.g. for
    direct test inspection after a suppressed/closed display).
    """

    if config is None:
        config = PlotConfig()
    y_label = kwargs.get("z_label", config.z_label)
    x_type = kwargs.get("x_type", config.x_type)
    y_type = kwargs.get("y_type", config.y_type)
    x_lim = kwargs.get("x_lim", config.x_lim)
    y_lim = kwargs.get("y_lim", config.y_lim)
    dpi_plot = kwargs.get("dpi_plot", config.dpi_plot)
    dpi_save = kwargs.get("dpi_save", config.dpi_save)

    obs = np.asarray(observed).ravel()
    fit_arr = np.asarray(fit).ravel()
    fig, (ax_fit, ax_res) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(6.0, 5.0),
        height_ratios=[3, 1],
        dpi=dpi_plot,
    )
    ax_fit.plot(x, obs, "k.", ms=3, label="observed")
    if show_init and fit_ini is not None:
        ax_fit.plot(
            x,
            np.asarray(fit_ini).ravel(),
            color="#FFD700",
            linestyle=":",
            linewidth=2,
            label="initial guess",
        )
    if components is not None:
        colors = list(
            plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])
        )
        names = component_names or [
            f"component {i}" for i in range(components.shape[0])
        ]
        for p, (peak, name) in enumerate(zip(components, names, strict=True)):
            color = colors[p % len(colors)]
            ax_fit.plot(x, peak, color=color, linestyle="-", linewidth=2, label=name)
            ax_fit.fill_between(x, 0, peak, facecolor=color, alpha=0.5)
        ax_fit.plot(x, fit_arr, "-", lw=1.5, color="#000000", label="fit")
    else:
        ax_fit.plot(x, fit_arr, "-", lw=1.5, label="fit")
    ax_fit.set_ylabel(y_label)
    ax_fit.legend(fontsize="small")
    ax_fit.set_title(title)
    ax_res.plot(x, obs - fit_arr, "-", lw=1.0)
    ax_res.axhline(0, color="gray", lw=0.5)
    ax_res.set_xlabel(x_label)
    ax_res.set_ylabel("residual")
    if roi is not None and len(roi) == 2 and x.size:
        x_start = x[roi[0]]
        x_end = x[roi[1] - 1] if roi[1] > 0 else x[-1]
        for ax in (ax_fit, ax_res):
            for x_pos in (x_start, x_end):
                ax.axvline(
                    x_pos,
                    color=config.refline_color,
                    linestyle=config.refline_style,
                )
    # x settings propagate to ax_res via sharex; y settings style the
    # intensity panel only (the residual keeps its own auto scale)
    _apply_axis_settings(
        ax_fit, x_type, x_dir, y_type, y_dir=None, x_lim=x_lim, y_lim=y_lim
    )
    fig.tight_layout()

    save_img = (
        -2
        if save_path is None and not show_plot
        else _save_img_flag(save=save_path is not None, show=show_plot)
    )
    _finalize_plot(fig, save_img, save_path or "", dpi_save)
    return fig


#
def plot_fit_overlay_1d(
    x: ArrayLike,
    observed: ArrayLike,
    *,
    fit: ArrayLike | None = None,
    components: Sequence[ArrayLike] | None = None,
    init: ArrayLike | None = None,
    legend: Sequence[str] | None = None,
    title: str = "",
    fit_lim: list[int] | None = None,
    config: PlotConfig | None = None,
    **kwargs: Any,
) -> None:
    """
    Single-panel 1D fit overlay: data, curves, and scaled inline residual.

    The pre-fit inspection style (``File.describe_model``): observed data,
    an optional initial-guess curve (dotted gold), optional per-component
    curves (colored + filled), an optional total curve (black), and the
    residual scaled by ``res_mult`` drawn in the same axes. The residual
    is ``observed - fit``, falling back to ``observed - init`` when no
    total curve is given; at least one of ``fit`` / ``init`` is required.
    All curves arrive as plain arrays — model evaluation is the caller's
    job (``fitlib.eval_model_curves_1d``).

    Parameters
    ----------
    x : array
        X-axis data (energy or time).
    observed : array
        Measured data.
    fit : array, optional
        Total model curve ("final fit", black).
    components : sequence of array, optional
        Per-component curves (colored, with fill).
    init : array, optional
        Initial-guess curve (dotted gold).
    legend : sequence of str, optional
        Component labels; auto-numbered when omitted.
    title : str, default=''
        Plot title (left-aligned).
    fit_lim : list of int, optional
        Fit-limit indices ``[start, stop)`` drawn as dashed vlines.
    config : PlotConfig, optional
        Plot configuration object. If None, uses defaults.
    **kwargs : dict
        Override config attributes for this plot: x_label, z_label, x_lim,
        y_lim, x_dir, x_type, y_type, res_mult, save_img, save_path,
        dpi_plot, dpi_save.
    """

    if config is None:
        config = PlotConfig()

    x_label = kwargs.get("x_label", config.x_label)
    y_label = kwargs.get("z_label", config.z_label)  # y is Intensity in 1D plot
    x_dir = kwargs.get("x_dir", config.x_dir)
    x_type = kwargs.get("x_type", config.x_type)
    y_type = kwargs.get("y_type", config.y_type)
    x_lim = kwargs.get("x_lim", config.x_lim)
    y_lim = kwargs.get("y_lim", config.y_lim)
    dpi_plot = kwargs.get("dpi_plot", config.dpi_plot)
    dpi_save = kwargs.get("dpi_save", config.dpi_save)
    res_mult = kwargs.get("res_mult", config.res_mult)
    save_img = kwargs.get("save_img", 0)
    save_path = kwargs.get("save_path", "")

    # validate before any figure exists, so a bad call leaks nothing
    if fit is None and init is None:
        raise ValueError("plot_fit_overlay_1d needs fit= or init= to draw a residual.")

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(observed, dtype=float)

    colors: list[str] = list(
        plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])
    )

    fig, ax = plt.subplots(1, 1, dpi=dpi_plot)

    plt.plot(x_arr, y_arr, color=colors[0], linewidth=2, label="data")

    if init is not None:
        plt.plot(
            x_arr,
            np.asarray(init, dtype=float),
            color="#FFD700",
            linestyle=":",
            linewidth=2,
            label="initial guess",
        )

    if components is not None:
        for p, peak in enumerate(components):
            label = legend[p] if legend and p < len(legend) else f"component {p}"
            color_idx = (p + 1) % len(colors)
            peak_arr = np.asarray(peak, dtype=float)
            plt.plot(
                x_arr,
                peak_arr,
                color=colors[color_idx],
                linestyle="-",
                linewidth=2,
                label=label,
            )
            ax.fill_between(x_arr, 0, peak_arr, facecolor=colors[color_idx], alpha=0.5)

    if fit is not None:
        fit_arr = np.asarray(fit, dtype=float)
        plt.plot(
            x_arr,
            fit_arr,
            color="#000000",
            linestyle="-",
            linewidth=1,
            label="final fit",
        )
        res = y_arr - fit_arr
    else:
        assert init is not None  # type guard
        res = y_arr - np.asarray(init, dtype=float)

    plt.plot(
        x_arr,
        res * res_mult,
        color="#808080",
        linestyle="-",
        linewidth=2,
        label=f"{res_mult}*residual",
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.title(title, loc="left", fontsize=10)

    _apply_axis_settings(
        ax, x_type, x_dir, y_type, y_dir=None, x_lim=x_lim, y_lim=y_lim
    )

    # Draw zero line
    if x_lim is not None:
        ax.hlines(y=0, xmin=x_lim[0], xmax=x_lim[1], color="#A9A9A9", linestyle=":")
    else:
        ax.hlines(
            y=0, xmin=np.min(x_arr), xmax=np.max(x_arr), color="#A9A9A9", linestyle=":"
        )

    # Draw vertical lines showing fit limits
    if fit_lim is not None and len(fit_lim) == 2:
        x_start = x_arr[fit_lim[0]]
        x_end = x_arr[fit_lim[1] - 1] if fit_lim[1] > 0 else x_arr[-1]
        ax.vlines(
            x=[x_start, x_end],
            ymin=np.min(res),
            ymax=np.max(y_arr),
            colors=config.refline_color,
            linestyle=config.refline_style,
        )

    plt.legend(bbox_to_anchor=(1.35, 1))

    _finalize_plot(fig, save_img, save_path, dpi_save)


#
def plot_fit_res_2d(
    data: np.ndarray,
    fit: np.ndarray,
    x: ArrayLike | None = None,
    y: ArrayLike | None = None,
    config: PlotConfig | None = None,
    *,
    title: str = "",
    **kwargs: Any,
) -> None:
    """
    Plot 2D fit results: data, fit, and residual maps.

    Creates a three-panel visualization showing measured data, fitted data,
    and residual (data - fit) as 2D color maps. Use to improve fit by switching
    component type, changing number of components, etc.

    Parameters
    ----------
    data : 2D array
        Measured data (shape: [n_time, n_energy])
    fit : 2D array
        Fitted data (same shape as data)
    x : array-like, optional
        X-axis (energy) coordinates. If None, uses column indices.
    y : array-like, optional
        Y-axis (time) coordinates. If None, uses row indices.
    config : PlotConfig, optional
        Plot configuration object. If None, uses defaults.
    title : str, default=''
        Figure-level title (e.g. file/model identification). Empty means no
        suptitle, matching prior behavior.
    **kwargs : dict
        Override config attributes for this plot.

        Common options:

        - x_label, y_label : Axis labels (z_label used for colorbar title)
        - x_lim, y_lim : Fit limit indices ``[left, right]`` or ``[start, stop]``.
          Used for both slicing residual and drawing limit lines
        - z_lim_top : Color scale ``[min, max]`` for data and fit panels.
          Synchronized scale enables direct comparison
        - z_lim_res : Color scale ``[min, max]`` for residual panel.
          If None, symmetric around 0 so the diverging colormap's
          midpoint marks zero residual
        - z_colormap : Colormap name for data/fit panels (default 'viridis')
        - z_colormap_res : Diverging colormap name for the residual panel
          (default 'RdBu_r')
        - x_dir, y_dir : 'def' or 'rev' for axis direction
        - x_type, y_type : 'lin' or 'log' for axis scale
        - save_img : 0 (display), 1 (save+display), -1 (save only)
        - save_path : Directory path (file saved as '2D_data_fit_res.png')
    """

    if config is None:
        config = PlotConfig()

    # Extract settings from config
    x_label = kwargs.get("x_label", config.x_label)
    y_label = kwargs.get("y_label", config.y_label)
    z_colormap = kwargs.get("z_colormap", config.z_colormap)
    z_colormap_res = kwargs.get("z_colormap_res", config.z_colormap_res)
    x_dir = kwargs.get("x_dir", config.x_dir)
    x_type = kwargs.get("x_type", config.x_type)
    y_dir = kwargs.get("y_dir", config.y_dir)
    y_type = kwargs.get("y_type", config.y_type)
    z_label = kwargs.get("z_label", config.z_label)
    dpi_plot = kwargs.get("dpi_plot", config.dpi_plot)
    dpi_save = kwargs.get("dpi_save", config.dpi_save)
    save_img = kwargs.get("save_img", 0)
    save_path = kwargs.get("save_path", "")

    # Fit limit indices
    x_lim = kwargs.get("x_lim")
    y_lim = kwargs.get("y_lim")

    # Color scale limits
    z_lim_top = kwargs.get("z_lim_top")  # Shared for data and fit
    z_lim_res = kwargs.get("z_lim_res")  # Independent for residual

    # Calculate residual
    res = data - fit

    # Cut residual according to x_lim and y_lim for statistics
    if x_lim is not None and y_lim is not None:
        res_cut = res[y_lim[0] : y_lim[1], x_lim[0] : x_lim[1]]
    elif x_lim is not None:
        res_cut = res[:, x_lim[0] : x_lim[1]]
    elif y_lim is not None:
        res_cut = res[y_lim[0] : y_lim[1], :]
    else:
        res_cut = res

    res_sum = np.sum(np.abs(res_cut))
    res_dim = res_cut.shape

    # Create default axes if not provided
    if x is None:
        x_arr = np.arange(data.shape[1], dtype=float)
    else:
        x_arr = np.asarray(x, dtype=float)
    if y is None:
        y_arr = np.arange(data.shape[0], dtype=float)
    else:
        y_arr = np.asarray(y, dtype=float)

    # Determine color scale ranges
    # Data and fit share the same scale for comparison
    if z_lim_top is None:
        # nanmin/nanmax: full_range mode's fit array carries NaN outside
        # the fit window; identical to min/max when no NaN is present.
        range_dat_fit = [
            min(np.min(data), np.nanmin(fit)),
            max(np.max(data), np.nanmax(fit)),
        ]
    else:
        range_dat_fit = z_lim_top

    # Residual has independent scale, symmetric around 0 by default so the
    # diverging colormap's midpoint marks zero residual
    if z_lim_res is None:
        range_res = _symmetric_range([res_cut])
    else:
        range_res = z_lim_res

    # Create figure layout
    fig, axs = plt.subplot_mosaic(
        [["left", "right"], ["bottom", "bottom"], ["bottom", "bottom"]],
        constrained_layout=True,
        figsize=(9, 12),
        dpi=dpi_plot,
    )
    if title:
        fig.suptitle(title)

    # Data panel (uses shared scale)
    axs["left"].pcolormesh(
        x_arr,
        y_arr,
        data,
        cmap=z_colormap,
        vmin=range_dat_fit[0],
        vmax=range_dat_fit[1],
        shading="nearest",
    )
    axs["left"].set_title(
        "Data [min: "
        + str(f"{np.min(data):.3E}")
        + ", max: "
        + str(f"{np.max(data):.3E}")
        + "]"
    )

    # Fit panel (uses shared scale)
    axs["right"].pcolormesh(
        x_arr,
        y_arr,
        fit,
        cmap=z_colormap,
        vmin=range_dat_fit[0],
        vmax=range_dat_fit[1],
        shading="nearest",
    )
    axs["right"].set_title(
        "Fit [min: "
        + str(f"{np.nanmin(fit):.3E}")
        + ", max: "
        + str(f"{np.nanmax(fit):.3E}")
        + "]"
    )

    # Residual panel (independent scale)
    pc_res = axs["bottom"].pcolormesh(
        x_arr,
        y_arr,
        res,
        cmap=z_colormap_res,
        vmin=range_res[0],
        vmax=range_res[1],
        shading="nearest",
    )
    axs["bottom"].set_title(
        "Residual (Data-Fit) [min: "
        + str(f"{np.min(res_cut):.3E}")
        + ", max: "
        + str(f"{np.max(res_cut):.3E}")
        + "]"
        + "\n"
        + "total residual (sum within fit-limit lines): "
        + str(f"{res_sum:.3E}")
        + "\n"
        + "per spectrum: "
        + str(f"{res_sum / res_dim[0]:.3E}")
        + ", per pixel: "
        + str(f"{res_sum / res_dim[0] / res_dim[1]:.3E}")
    )

    # Colorbar only on residual map
    fig.colorbar(pc_res, orientation="vertical", label=z_label)

    # Labels only on residual map
    axs["bottom"].set_ylabel(y_label)
    axs["bottom"].set_xlabel(x_label)

    # Draw horizontal and vertical lines showing fit limits
    if y_lim is not None:
        axs["bottom"].axhline(
            y=float(y_arr[y_lim[0]]),
            xmin=0,
            xmax=1,
            color=config.refline_color,
            linestyle=config.refline_style,
        )
        axs["bottom"].axhline(
            y=float(y_arr[y_lim[1] - 1]),
            xmin=0,
            xmax=1,
            color=config.refline_color,
            linestyle=config.refline_style,
        )
    if x_lim is not None:
        axs["bottom"].axvline(
            x=float(x_arr[x_lim[0]]),
            ymin=0,
            ymax=1,
            color=config.refline_color,
            linestyle=config.refline_style,
        )
        axs["bottom"].axvline(
            x=float(x_arr[x_lim[1] - 1]),
            ymin=0,
            ymax=1,
            color=config.refline_color,
            linestyle=config.refline_style,
        )

    # Apply axis settings to all three plots
    for a in axs.values():
        _apply_axis_settings(a, x_type, x_dir, y_type, y_dir)

    # Save/show/close
    _finalize_plot(
        fig, save_img, pathlib.Path(save_path) / "2D_data_fit_res.png", dpi_save
    )


#
def plot_par_series(
    df: Any,
    x: ArrayLike | None = None,
    config: PlotConfig | None = None,
    save_img: int | list[int] = 0,
    save_path: PathLike = "",
) -> None:
    """
    Plot fit parameters individually as functions of time/index.

    Creates separate plots for each parameter column in the DataFrame,
    showing how parameters evolve over time (from Slice-by-Slice fitting).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with parameters as columns. Typically from results_to_df().
        Each row represents one fitted slice/time point.
    x : array-like, optional
        X-axis (time) values for plotting. If None, uses row indices.
    config : PlotConfig, optional
        Plot configuration object. If None, uses defaults.
    save_img : int or list, default=0
        Save/display control for each plot:

        - int: Apply same setting to all parameters

          - 0: Display only
          - 1: Display and save
          - -1: Save only (no display)

        - list: One element per parameter (per row in df).
          Allows selective saving (e.g., save only varied parameters)

    save_path : str or Path, default=''
        Directory path for saving plots.
        Each plot saved as: save_path/{parameter_name}.png
        Directory created if doesn't exist.
    """

    # Use default config if none provided
    if config is None:
        config = PlotConfig()

    # if save_img is passed as int, make array of length = number of parameters
    if isinstance(save_img, int):
        save_img_list = len(df.columns) * [save_img]
    else:
        save_img_list = save_img

    # plot all parameters as function of time
    for c, col in enumerate(df.columns):
        plot_1d(
            data=[df[col]],
            x=x,
            config=config,
            title=col,
            x_dir="def",
            x_type=config.y_type,
            x_label=config.y_label,
            y_label=col,
            save_img=save_img_list[c],
            save_path=pathlib.Path(save_path) / col,
        )


#
def plot_mcmc_diagnostics(
    mcmc: Any, *, show_plot: bool, config: PlotConfig | None = None
) -> None:
    """
    Walker-acceptance and corner figures from an ``MCMCResult``.

    The single rendering primitive behind ``FitResults.plot_mcmc``
    (per-file slots) and ``plot_joint_mcmc`` (joint records). The
    acceptance panel is skipped when ``acceptance_fraction`` is ``None``.
    Both figures render at ``config.dpi_plot``.
    """

    import corner

    if config is None:
        config = PlotConfig()
    if mcmc.acceptance_fraction is not None:
        fig_walker, ax = plt.subplots(1, 1, dpi=config.dpi_plot)
        ax.plot(mcmc.acceptance_fraction, "o")
        ax.set_xlabel("Walker number")
        ax.set_ylabel("Acceptance fraction")
        _finalize_plot(fig_walker, 0 if show_plot else -2)
    if not mcmc.flatchain.empty:
        var_names = list(mcmc.flatchain.columns)
        truths = None
        if not mcmc.table.empty:
            best = dict(
                zip(
                    mcmc.table.iloc[:, 0],
                    mcmc.table["best fit"],
                    strict=True,
                )
            )
            truths = [best.get(name) for name in var_names]
        fig_corner = plt.figure(figsize=(10, 10), dpi=config.dpi_plot)
        corner.corner(
            mcmc.flatchain,
            labels=var_names,
            truths=truths,
            fig=fig_corner,
        )
        _finalize_plot(fig_corner, 0 if show_plot else -2)


#
def plot_residual_panels_1d(
    panels: Sequence[dict[str, Any]],
    *,
    suptitle: str,
    show_plot: bool,
    figsize: tuple[float, float] | None = None,
    config: PlotConfig | None = None,
) -> Any:
    """
    Side-by-side 1D residual columns: top row observed + fit, bottom residual.

    Each panel dict carries plain arrays/strings — keys ``x``, ``x_label``,
    ``observed``, ``fit``, ``title``. Slot selection and axis lookup stay
    with the caller; ``config.z_label`` names the intensity axis. Returns
    the Figure.
    """

    if config is None:
        config = PlotConfig()
    n = len(panels)
    fig, axs = plt.subplots(
        2,
        n,
        figsize=figsize or (4.0 * max(n, 1), 5.0),
        squeeze=False,
        sharex="col",
        dpi=config.dpi_plot,
    )
    for col, panel in enumerate(panels):
        x, obs, fit = panel["x"], panel["observed"], panel["fit"]
        axs[0, col].plot(x, obs, "k.", ms=3, label="observed")
        axs[0, col].plot(x, fit, "-", lw=1.5, label="fit")
        axs[0, col].set_title(panel["title"])
        axs[0, col].legend(fontsize="small")
        axs[1, col].plot(x, obs - fit, "-", lw=1.0)
        axs[1, col].axhline(0, color="gray", lw=0.5)
        axs[1, col].set_xlabel(panel["x_label"])
        # config-driven x direction/scale (index axes stand in for energy)
        _apply_axis_settings(axs[1, col], config.x_type, config.x_dir)
        if col == 0:
            axs[0, col].set_ylabel(config.z_label)
            axs[1, col].set_ylabel("residual")
    fig.suptitle(suptitle)
    fig.tight_layout()
    _finalize_plot(fig, 0 if show_plot else -2)
    return fig


#
def plot_residual_maps_2d(
    panels: Sequence[dict[str, Any]],
    *,
    suptitle: str,
    show_plot: bool,
    figsize: tuple[float, float] | None = None,
    config: PlotConfig | None = None,
) -> Any:
    """
    Side-by-side residual heatmaps on one shared diverging scale.

    Each panel dict carries plain arrays/strings — keys ``residual`` (2D
    array), ``extent`` (imshow extent tuple or None for index axes),
    ``x_label``, ``y_label``, ``title``. The shared symmetric color scale
    is computed here from all residuals; the diverging colormap comes from
    ``config.z_colormap_res`` (matching ``plot_fit_res_2d``'s residual
    panel). Returns the Figure.
    """

    if config is None:
        config = PlotConfig()
    vmin, vmax = _symmetric_range([panel["residual"] for panel in panels])

    n = len(panels)
    fig, axs = plt.subplots(
        1,
        n,
        figsize=figsize or (5.0 * max(n, 1), 4.0),
        squeeze=False,
        dpi=config.dpi_plot,
    )
    im = None
    for col, panel in enumerate(panels):
        im = axs[0, col].imshow(
            panel["residual"],
            aspect="auto",
            cmap=config.z_colormap_res,
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            extent=panel["extent"],
        )
        axs[0, col].set_title(panel["title"])
        axs[0, col].set_xlabel(panel["x_label"])
        _apply_axis_settings(
            axs[0, col], config.x_type, config.x_dir, config.y_type, config.y_dir
        )
        if col == 0:
            axs[0, col].set_ylabel(panel["y_label"])
    if im is not None:
        fig.colorbar(im, ax=axs[0, :].tolist(), shrink=0.85)
    fig.suptitle(suptitle)
    _finalize_plot(fig, 0 if show_plot else -2)
    return fig


#
def use_headless_backend() -> None:
    """
    Force matplotlib's non-interactive Agg backend.

    For worker processes (Slice-by-Slice executors) that must never try
    to open a display; ``force=True`` covers workers that inherited an
    interactive backend.
    """

    import matplotlib

    matplotlib.use("Agg", force=True)


#
def _apply_axis_settings(
    ax: matplotlib.axes.Axes,
    x_type: str | None = None,
    x_dir: str | None = None,
    y_type: str | None = None,
    y_dir: str | None = None,
    x_lim: Sequence[float] | None = None,
    y_lim: Sequence[float] | None = None,
) -> None:
    """Apply scale, limits, and direction settings to a matplotlib axes."""

    if x_type == "log":
        ax.set_xscale("log")
    if x_lim is not None:
        ax.set_xlim(x_lim[0], x_lim[1])
    if x_dir == "rev" and not ax.xaxis_inverted():
        ax.invert_xaxis()
    elif x_dir != "rev" and ax.xaxis_inverted():
        ax.invert_xaxis()
    if y_type == "log":
        ax.set_yscale("log")
    if y_lim is not None:
        ax.set_ylim(y_lim[0], y_lim[1])
    if y_dir == "rev" and not ax.yaxis_inverted():
        ax.invert_yaxis()
    elif y_dir != "rev" and ax.yaxis_inverted():
        ax.invert_yaxis()


#
def _symmetric_range(arrays: Sequence[NDArray[Any]]) -> list[float]:
    """Symmetric range around zero from the arrays' NaN-aware max magnitude.

    Falls back to ``[-1, 1]`` for empty/all-zero input so a diverging
    colormap keeps its midpoint at zero residual.
    """

    amp = 0.0
    for arr in arrays:
        if arr.size:
            local = float(np.nanmax(np.abs(arr)))
            if np.isfinite(local) and local > amp:
                amp = local
    if amp == 0.0:
        amp = 1.0
    return [-amp, amp]


#
def _save_img_flag(*, save: bool, show: bool) -> int:
    """Map already-decided ``save`` / ``show`` booleans onto the legacy
    ``save_img`` int used by :func:`_finalize_plot`.

    +1 = save+show, -1 = save+close, 0 = show only. Callers are expected
    to skip the plot helper entirely when both flags are False, so this
    function never returns -2.
    """

    if save and show:
        return 1
    if save:
        return -1
    return 0


def _finalize_plot(
    fig: Any, save_img: int, save_path: PathLike = "", dpi_save: int = 300
) -> None:
    """Save, show, or close the given figure — never pyplot's implicit one.

    The single figure-lifecycle owner: every renderer in this module ends
    here. When saving, uses tight bounding box, 0.05-inch padding, white
    facecolor, and auto edgecolor (same defaults as ``img_save``).
    """

    if abs(save_img) == 1:
        img_save(save_path, dpi_save, fig=fig)
    if save_img >= 0:
        plt.show()
    else:
        plt.close(fig)


#
def img_save(save_path: PathLike, dpi: int = 300, *, fig: Any = None) -> None:
    """
    Save a matplotlib figure with sensible defaults.

    Wrapper around ``Figure.savefig`` with tight bounding box to minimize
    whitespace, small padding (0.05 inches), white background, auto edge
    color. Creates the parent directory if it does not exist.

    Parameters
    ----------
    save_path : str or Path
        Output file path (extension determines format: .png, .pdf, .svg, etc.)
    dpi : int, default=300
        Resolution in dots per inch
    fig : matplotlib.figure.Figure, optional
        Figure to save. Default: pyplot's current figure (back-compat for
        direct user calls; the module's renderers always pass theirs).
    """

    if fig is None:
        fig = plt.gcf()
    pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        save_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white",
        edgecolor="auto",
    )


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

    x_max = float(np.max(np.asarray(x, dtype=float)))
    return float(10 ** oom(x_max))


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

    x_max = float(np.max(np.asarray(x, dtype=float)))
    return float(10 ** (oom(x_max) - 1))


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

    axis_oom = oom(np.max(np.asarray(x, dtype=float)))

    if axis_oom < 0:
        return f"%0.{abs(axis_oom)}f"
    if axis_oom == 0:
        return "%0.0f"
    # axis_oom > 0
    return f"%{axis_oom + 1}.0f"
