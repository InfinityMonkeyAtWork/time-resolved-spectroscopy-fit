#
# package for all plot functions and helper functions
#
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import seaborn as sns
import math
import numpy as np
#import os
#
# replace empty list defaults in plot_1D and plot_2D (?)
#

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
    _, axs = plt.subplots(math.ceil(len(images)/columns), columns,
                          figsize=(fig_width, fig_height))
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
# general 2D plot function
def plot_2D(data, title='', x=[], y=[], xlabel='x axis', ylabel='y axis',
            colormap='RdBu', colorbar='ver', xtype='lin', ytype='lin',
            xdir='', ydir='', hlines=[], vlines=[], ticksize=None,
            ranges=[[],[],[]], dpi_plot=100, fig_size=[],
            save_img=0, save_path='', dpi_save=300):
    """
    plot <data> as a function of the independent variables <x> and <y>
    <cmap> is the colormap
    <cbar> is the colorbar orientation ('ver' or 'hor')
    <save_img> = 0(no) or 1(yes), <save_path> is the full image path
    ranges so far are index elements (not absolute values)
    ranges = [x_range, y_range, z_range] are each a list [start, stop]
    (x: col, y: row)
    additional options available for <z_range>:
    "[0, 'max']" [add more here]
    """
    # define data to plot
    if len(ranges[0])!=0 and len(ranges[1])!=0:
        data_plt = data[ranges[1][0]:ranges[1][1], ranges[0][0]:ranges[0][1]]
        if len(x)!=0: x = x[ranges[0][0]:ranges[0][1]]
        if len(y)!=0: y = y[ranges[1][0]:ranges[1][1]]
    elif len(ranges[1])!=0: # y only
        data_plt = data[ranges[1][0]:ranges[1][1], :]
        if len(y)!=0: y = y[ranges[1][0]:ranges[1][1]]
    elif len(ranges[0])!=0: # x only
        data_plt = data[:, ranges[0][0]:ranges[0][1]]
        if len(x)!=0: x = x[ranges[0][0]:ranges[0][1]]
    else:
        data_plt = data
        
    # z (color coded) range: auto/ user-defined/ mix?
    if ranges[2] == []:
        min2D = np.min(data_plt); max2D=np.max(data_plt)
        scale_txt = 'autoscale min. and max. z (color)'
    elif ranges[2] == [0, 'max']:
        min2D = 0; max2D=np.max(data_plt)
        scale_txt = 'autoscale max. z (color) [min=0]'
    else:
        min2D = ranges[2][0]; max2D = ranges[2][1]
        scale_txt = 'user defined z scale (color)'
    
    # if no x and/or y axis is passed, create one
    if len(x)==0: x = np.arange(0, np.shape(data_plt)[1], 1)
    if len(y)==0: y = np.arange(0, np.shape(data_plt)[0], 1)
    
    # create figure
    fig, ax = plt.subplots(1, 1, dpi=dpi_plot)
    if len(fig_size)!=0: fig.set_size_inches(fig_size[0], fig_size[1], forward=True)
    plt.title(title + '\n' + scale_txt + '\n' + 'size 2D data set: ' + \
              str(np.shape(data_plt)), loc='left', fontsize=10)
    if xlabel != None: ax.set_xlabel(xlabel)
    if ylabel != None: ax.set_ylabel(ylabel)
    plt.pcolormesh(x, y, data_plt, cmap=colormap,
                   vmin=min2D, vmax=max2D, shading='nearest')
    if colorbar == 'ver':
        cbar = plt.colorbar(orientation = 'vertical')
    elif colorbar == 'hor':
        cbar = plt.colorbar(orientation = 'horizontal')
    
    # set label font size for major tick lables
    if ticksize != None:
        ax.tick_params(axis='both', which='major', labelsize=ticksize)
        cbar.ax.tick_params(labelsize=ticksize)
    
    # define x and y direction, log/linear, or limits (if passed)
    if xtype == 'log': ax.set_xscale('log')
    if xdir == 'rev': plt.gca().invert_xaxis()
    if ytype == 'log': ax.set_yscale('log')
    if ydir == 'rev': plt.gca().invert_yaxis()
    
    # draw horizontal and vertical line(s)
    if len(hlines) != 0:
        plt.hlines(y=np.asarray(hlines), xmin=np.min(x), xmax=np.max(x),
                   color='#000000', linestyle=':')
    if len(vlines) != 0:
        plt.vlines(x=np.asarray(vlines), ymin=np.min(y), ymax=np.max(y),
                   color='#000000', linestyle=':')
        
    # save figure
    if abs(save_img) == 1: img_save(save_path, dpi_save)
    #
    if save_img >=0 : plt.show()
    else: plt.close()

#
# general 1D plot function [for a list of individual 1D plots]
def plot_1D(data, x=[], title='', xlabel='x axis', ylabel='y axis',
            linestyles=[], colors=[], linewidths=[], markers=[], markersizes=[],
            legend=[], xtype='lin', ytype='lin', xdir='', ydir='', waterfall=0,
            xlim=[], ylim=[], yscale=[], ynorm=0, hlines=[], vlines=[],
            ticksize=None, save_img=0, save_path='', dpi_save=300, dpi_plot=100):
    """
    data: either rows of a 2D numpy array or a list of 1D arrays
    x: numpy array with length equal to number of columns of data
    or length of each item of list (that is data)
    xlim/ ylim: [lower limit, upper limit]
    waterfall (option) allows for y offset between neighboring data
    ynorm -> rename into "minmax" (0 baseline, 1 max amplitude)
    yscale is a list of one scaling factor for each scan
    save_img = 0/1; save_path is the full path (plus file name and
    extension); dpi = dots per inch
    """
    # define number of plots
    if isinstance(data, list): N_plots = len(data) # list
    else: N_plots = np.shape(data)[0] # numpy array
    
    fig, ax = plt.subplots(1, 1, dpi=dpi_plot) # create figure
    
    # create default values if user does not pass input
    if len(linestyles) == 0: linestyles = N_plots*['-']
    if len(colors) == 0: colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if len(linewidths) == 0: linewidths = N_plots*[1.5]
    if len(markers) == 0: markers = N_plots*[None]
    if len(markersizes) == 0: markersizes = N_plots*[6]
    if len(x) == 0: x = np.arange(0, np.shape(data)[1], 1)
    if len(yscale) == 0: yscale = np.ones(N_plots) # scaling factors
    if len(legend) == 0: legend = [i+1 for i in range(N_plots)] # starting at 1
    
    # title
    if ynorm==1: 
        title += '\n(all data normalized to baseline 0 and amplitude 1 [each])'
    plt.title(title, loc='left', fontsize=10)
    
    # plot components one by one (normalized or scaling with yscale)
    for i in range(N_plots):

        if isinstance(x, list): # x axis can be a list of x axis ...
            x_plot = x[i]
        else: # ... or a np.array (same x axis is used for all plots)
            x_plot = x
        
        # normalize to a baseline of 0 and amplitude of 1
        if ynorm==1:
            ax.plot(x_plot,
                    (np.asarray(data[i])-np.min(data[i]))/ \
                    (np.max(np.asarray(data[i])-np.min(data[i]))) +i*waterfall,
                    ls=linestyles[i], c=colors[i%len(colors)], lw=linewidths[i],
                    marker=markers[i], ms=markersizes[i], label=str(legend[i]))
        # plot data as is OR scale data to a "scaling factor" yscale
        else:
            ax.plot(x_plot, yscale[i]*np.asarray(data[i]) +i*waterfall, 
                    ls=linestyles[i], c=colors[i%len(colors)], lw=linewidths[i],
                    marker=markers[i], ms=markersizes[i],
                    label = (f'{yscale[i]}*{legend[i]}' if yscale[i]!=1
                             else str(legend[i])))
    
    # set axis labels
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    
    # draw horizontal line(s)
    if len(hlines) != 0:
        # find min and max for horizontal lines
        if isinstance(x, list): # x axis can be a list of x axis ...
            x_minmax = [np.min([np.min(x[i]) for i in range(N_plots)]), \
                        np.max([np.max(x[i]) for i in range(N_plots)])]
        else: # ... or a np.array (same x axis is used for all plots)
            x_minmax = [np.min(x), np.max(x)]
        plt.hlines(y=np.asarray(hlines), xmin=x_minmax[0], xmax=x_minmax[1],
                   color='#808080', linestyle=':')
    # draw vertical line(s)
    if len(vlines) != 0:
        # find min and max for vertical lines
        if ynorm == 1:
            y_minmax = [0, 1]
        else:
            y_minmax = [np.min([np.min(yscale[i]*np.asarray(data[i])) \
                                for i in range(N_plots)]),
                        np.max([np.max(yscale[i]*np.asarray(data[i])) \
                                for i in range(N_plots)])]
        plt.vlines(x=np.asarray(vlines), ymin=y_minmax[0], ymax=y_minmax[1],
                   color='#808080', linestyle='--')
        
    # define x and y direction, log/linear, or limits (if passed)
    if xtype == 'log': ax.set_xscale('log')
    if len(xlim)!=0: ax.set_xlim(xlim[0], xlim[1])
    if xdir == 'rev': plt.gca().invert_xaxis()
    if ytype == 'log': ax.set_yscale('log')
    if len(ylim)!=0: ax.set_ylim(ylim[0], ylim[1])
    if ydir == 'rev': plt.gca().invert_yaxis()
    # set label font size for major tick lables
    if ticksize != None:
        ax.tick_params(axis='both', which='major', labelsize=ticksize)
    plt.legend(bbox_to_anchor = (1,1)) # legend
    # save figure
    if abs(save_img) == 1: img_save(save_path, dpi_save)
    # show/ close plot
    if save_img >=0 : plt.show()
    else: plt.close()

#
def img_save(save_path, dpi=300):
    """
    standard plt.savefig function wrapper with reasonable parameters
    """
    plt.savefig(save_path, dpi=dpi, bbox_inches = 'tight', \
                pad_inches=0.05, facecolor='white', edgecolor='auto')

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
#
#
#