"""
Package for all plot functions and helper functions
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import numpy as np
from trspecfit.config.plot import PlotConfig

#
def load_plot(path, dpi_fig=75):
    """
    Standard function to load and display a saved figure 
    (factor 1.25*dpi accounts for a typical whitespace/ margin)
    """
    fig, ax = plt.subplots(1, 1, dpi=1.25*dpi_fig) # create figure
    img = mpimg.imread(path) # load image
    plt.imshow(img) # display image
    plt.axis('off') # get rid of axis (and whitespace)
    plt.show()
    #
    return None

#
def load_plot_grid(paths, columns=3, fig_width=16, debug=0):
    """
    Load images (one each per path in <paths>) and plot on a grid that
    is <columns> wide
    """
    images = [plt.imread(path) for path in paths] # read
    plot_grid(images, columns, fig_width, debug) # plot
    #
    return None

#
def plot_grid(images, columns=3, fig_width=16, debug=0):
    """
    Plot images on a grid that is <columns> wide
    """
    rows = math.ceil(len(images)/columns)
    # get right image height
    # (assumes all images have the same width to height ratio)
    img_shape = np.shape(images[0])
    ratio = img_shape[1]/img_shape[0]
    fig_height = fig_width *rows /(ratio*columns)
    
    if debug >= 1:
        print(f'rows {rows}')
        print(f'image shape {img_shape}')
        print(f'ratio {ratio}')
        print(f'figure height {fig_height}')
    #
    _, axs = plt.subplots(math.ceil(len(images)/columns),
                          columns,
                          figsize=(fig_width, fig_height)
                          )
    axs = axs.flatten()
    for image, ax in zip(images, axs):
        ax.imshow(image)
        ax.set_axis_off()
    plt.subplots_adjust(hspace=0, wspace=0.05)
    #plt.subplots_adjust(top=0.1, bottom=0, left=0.10, right=0.95,
    #                    hspace=0.25, wspace=0.35)
    plt.show()
    #
    return None

#
def plot_2D(data, x=None, y=None, config=None, **kwargs):
    """
    Plot 2D data as a color map.
    
    Parameters
    ----------
    data : 2D array
        Data to plot as a heatmap
    x : array-like, optional
        X-axis (horizontal) coordinates. If None, uses column indices.
    y : array-like, optional
        Y-axis (vertical) coordinates. If None, uses row indices.
    config : PlotConfig, optional
        Configuration object with plot settings. If None, uses defaults.
    **kwargs : dict
        Override any config attributes for this specific plot.
        Common options: x_label, y_label, title, x_lim, y_lim, z_lim, 
        x_dir, x_type, y_dir, y_type, z_colormap, z_colorbar, data_slice, 
        vlines, hlines, dpi_plot, dpi_save, save_img, save_path
    
    Notes
    -----
    data_slice controls which portion of data array to plot (by index):
        data_slice = [[x_start, x_stop], [y_start, y_stop]]
    
    z_lim controls color scale limits:
        z_lim = [min, max]  (manual limits)
        z_lim = [0, 'max']  (auto max with min=0)
        z_lim = None        (auto scale both)
    
    x_lim/y_lim control axis display range (zoom) using actual coordinate values.
    
    Examples
    --------
    >>> # Simple plot with defaults
    >>> plot_2D(data, x=energy, y=time)
    
    >>> # Slice data and set color scale
    >>> plot_2D(data, x, y, config=config, 
    ...         data_slice=[[10, 100], [5, 50]], z_lim=[0, 100])
    
    >>> # Zoom axis display without slicing data
    >>> plot_2D(data, x, y, config=config, x_lim=[80, 90], y_lim=[0, 50])
    """
    # Use default config if none provided
    if config is None:
        config = PlotConfig()
    
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
    data_plt = data
    x_plt = x
    y_plt = y
    
    if data_slice is not None:
        x_slice = data_slice[0] if len(data_slice) > 0 else []
        y_slice = data_slice[1] if len(data_slice) > 1 else []
        
        if len(x_slice) > 0 and len(y_slice) > 0:
            data_plt = data[y_slice[0]:y_slice[1], x_slice[0]:x_slice[1]]
            if x is not None:
                x_plt = x[x_slice[0]:x_slice[1]]
            if y is not None:
                y_plt = y[y_slice[0]:y_slice[1]]
        elif len(y_slice) > 0:  # y only
            data_plt = data[y_slice[0]:y_slice[1], :]
            if y is not None:
                y_plt = y[y_slice[0]:y_slice[1]]
        elif len(x_slice) > 0:  # x only
            data_plt = data[:, x_slice[0]:x_slice[1]]
            if x is not None:
                x_plt = x[x_slice[0]:x_slice[1]]
    
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
    plt.pcolormesh(x_plt,
                   y_plt,
                   data_plt,
                   cmap=z_colormap,
                   vmin=min2D,
                   vmax=max2D, 
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
    
    # Horizontal lines
    if hlines is not None:
        plt.hlines(y=np.asarray(hlines),
                   xmin=np.min(x_plt),
                   xmax=np.max(x_plt),
                   color='#000000',
                   linestyle=':'
                   )
    
    # Vertical lines
    if vlines is not None:
        plt.vlines(x=np.asarray(vlines),
                   ymin=np.min(y_plt),
                   ymax=np.max(y_plt),
                   color='#000000',
                   linestyle=':'
                   )
    
    # Save figure
    if abs(save_img) == 1:
        img_save(save_path, dpi_save)
    
    # Show/close plot
    if save_img >= 0:
        plt.show()
    else:
        plt.close()

#
def plot_1D(data, x=None, config=None, **kwargs):
    """
    Plot 1D data with extensive customization options.
    
    Parameters
    ----------
    data : list of arrays or 2D array
        Data to plot. Either a list of 1D arrays or a 2D numpy array where
        each row is a separate plot.
    x : array-like or list of arrays, optional
        X-axis data. Can be a single array (used for all plots) or a list
        of arrays (one per plot). If None, uses indices.
    config : PlotConfig, optional
        Configuration object with plot settings. If None, uses defaults.
    **kwargs : dict
        Override any config attributes for this specific plot.
        Common options: x_label, y_label, title, x_lim, y_lim, x_dir, x_type,
        colors, linestyles, legend, dpi_plot, dpi_save, save_img, save_path
    
    Examples
    --------
    >>> # Simple plot with defaults
    >>> plot_1D([data1, data2], x=energy)
    
    >>> # With configuration from project
    >>> config = PlotConfig.from_project(project)
    >>> plot_1D(data, x, config=config)
    
    >>> # Override specific settings
    >>> plot_1D(data, x, config=config, x_label='Custom Label', x_dir='rev')
    
    >>> # Save plot
    >>> plot_1D(data, x, config=config, save_img=1, save_path='plot.png')
    """
    # Use default config if none provided
    if config is None:
        config = PlotConfig()
    
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
    if isinstance(data, list):
        N_plots = len(data)
    else:
        N_plots = np.shape(data)[0]
    
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
        x = np.arange(0, np.shape(data)[1], 1)
    if y_scale is None:
        y_scale = np.ones(N_plots)
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
        if isinstance(x, list):
            x_plot = x[i]
        else:
            x_plot = x
        
        # Normalize if requested
        if y_norm == 1:
            y_plot = (np.asarray(data[i]) - np.min(data[i])) / \
                     (np.max(np.asarray(data[i]) - np.min(data[i]))) + i * waterfall
        else:
            y_plot = y_scale[i] * np.asarray(data[i]) + i * waterfall
        
        # Plot
        label = f'{y_scale[i]}*{legend[i]}' if y_scale[i] != 1 else str(legend[i])
        ax.plot(x_plot,
                y_plot,
                ls=linestyles[i],
                c=colors[i % len(colors)],
                lw=linewidths[i],
                marker=markers[i],
                ms=markersizes[i],
                label=label
                )
    
    # Set axis labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    # Horizontal lines
    if hlines is not None:
        if isinstance(x, list):
            x_minmax = [np.min([np.min(x[i]) for i in range(N_plots)]),
                       np.max([np.max(x[i]) for i in range(N_plots)])]
        else:
            x_minmax = [np.min(x), np.max(x)]
        plt.hlines(y=np.asarray(hlines),
                   xmin=x_minmax[0],
                   xmax=x_minmax[1],
                   color='#808080',
                   linestyle=':'
                   )
    
    # Vertical lines
    if vlines is not None:
        if y_norm == 1:
            y_minmax = [0, 1]
        else:
            y_minmax = [np.min([np.min(y_scale[i] * np.asarray(data[i]))
                               for i in range(N_plots)]),
                       np.max([np.max(y_scale[i] * np.asarray(data[i]))
                               for i in range(N_plots)])]
        plt.vlines(x=np.asarray(vlines),
                   ymin=y_minmax[0],
                   ymax=y_minmax[1],
                   color='#808080',
                   linestyle='--'
                   )
    
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
def img_save(save_path, dpi=300):
    """
    standard plt.savefig function wrapper with reasonable parameters
    """
    plt.savefig(save_path,
                dpi=dpi,
                bbox_inches='tight',
                pad_inches=0.05,
                facecolor='white',
                edgecolor='auto'
                )

#
# matplotlib helper functions
#
def major_locator_input(x):
    """
    get the input for major_locator for plotting with matplotlib
    """  
    return 10**OoM(np.max(x))

#
def minor_locator_input(x):
    """
    get the input for minor_locator for plotting with matplotlib
    """ 
    return 10**(OoM(np.max(x)) -1)

#
def major_formatter_input(x):
    """
    get the input for major_formatter for plotting with matplotlib
    """ 
    axis_OoM = OoM(np.max(x))
    if axis_OoM < 0:
        return '%0.' +str(abs(axis_OoM)) +'f'
    elif axis_OoM == 0:
        return '%0.0f'
    elif axis_OoM > 0:
        return '%' +str(axis_OoM +1) +'.0f'

#
# misc.
#
def OoM(x):
    """
    get the order of magnitude (OoM) of a number x
    Examples: 0.006 -> -3; 2.717 -> 0; 137 -> 2
    rounds down, in the sense that OoM(9.99) will return 0
    However, a delta of <=1E-15 to the boundary will get rounded up
    So OoM(9.999999999999999) = 1 [computer memory thing?]
    """
    return int(math.floor(math.log10(x)))

#
def list2str(list_input):
    str_out = ''
    for list_item in list_input:
        str_out += str(list_item)+'; '
    return str_out[:-2]