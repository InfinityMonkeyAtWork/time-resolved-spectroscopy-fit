"""
Configuration of trspecfit plotting functions
"""
from dataclasses import dataclass
from typing import Optional, List, Tuple

# Plot configuration hierarchy:
#
# Project (defaults from YAML)
#    ↓
# File (can customize persistently per file)
#    ↓
# Model (inherits from File, no customization)
#    ↓
# plot call (can override temporarily for one plot)

@dataclass
class PlotConfig:
    """
    Configuration for plot appearance and behavior.
    
    This class bundles all plot-related settings that can be passed to
    plotting functions. It can be created standalone or from a Project instance.
    
    Attributes
    ----------
    x_label : str
        X-axis label (energy for 2D, energy or time for 1D)
    y_label : str
        Y-axis label (time for 2D, intensity for 1D)
    z_label : str
        Z-axis/colorbar label (intensity for 2D plots)
    title : str
        Plot title
    x_dir : str
        X-axis direction: 'def' (default) or 'rev' (reversed)
    x_type : str
        X-axis scale: 'lin' (linear) or 'log' (logarithmic)
    y_dir : str
        Y-axis direction: 'def' (default) or 'rev' (reversed)
    y_type : str
        Y-axis scale: 'lin' (linear) or 'log' (logarithmic)
    x_lim : Optional[Tuple[float, float]]
        X-axis limits (min, max)
    y_lim : Optional[Tuple[float, float]]
        Y-axis limits (min, max)
    z_lim : Optional[Tuple[float, float]]
        Color scale limits for 2D plots (min, max)
    dpi_plot : int
        DPI for displaying plots
    dpi_save : int
        DPI for saving plots
    z_colormap : str
        Colormap name for 2D plots
    data_slice : Optional[List[List[int]]]
        Data slicing indices for 2D plots: [[x_start, x_stop], [y_start, y_stop]]
    colors : Optional[List[str]]
        List of colors for line plots
    linestyles : Optional[List[str]]
        List of line styles
    linewidths : Optional[List[float]]
        List of line widths
    markers : Optional[List[str]]
        List of marker styles
    markersizes : Optional[List[float]]
        List of marker sizes
    legend : Optional[List[str]]
        List of legend labels
    waterfall : float
        Y-offset between plots for waterfall display
    vlines : Optional[List[float]]
        X-coordinates for vertical lines
    hlines : Optional[List[float]]
        Y-coordinates for horizontal lines
    ticksize : Optional[float]
        Font size for tick labels
 
    Examples
    --------
    Create a configuration with custom settings:
    
    >>> config = PlotConfig(x_label='Energy (eV)', x_dir='rev', dpi_plot=150)
    >>> plot_1D(data, x, config=config)
    
    Create from Project settings:
    
    >>> project = Project(path='...', config_file='project.yaml')
    >>> config = PlotConfig.from_project(project)
    >>> plot_1D(data, x, config=config)
    
    Override Project settings:
    
    >>> config = PlotConfig.from_project(project, x_label='Binding Energy (eV)')
    >>> plot_2D(data, x, y, config=config)
    
    Override config for a specific plot:
    
    >>> config = PlotConfig.from_project(project)
    >>> plot_1D(data, x, config=config, x_dir='rev', linewidth=2)
    
    Create multiple configs from one project:
    
    >>> project = Project(path='...')
    >>> default_config = PlotConfig.from_project(project)
    >>> pub_config = PlotConfig.from_project(project, dpi_save=600)
    >>> talk_config = PlotConfig.from_project(project, dpi_plot=150)
    """
    
    # Axis labels
    x_label: str = 'x axis'
    y_label: str = 'y axis'
    z_label: str = 'z axis'
    title: str = ''
    
    # Axis behavior
    x_dir: str = 'def'
    x_type: str = 'lin'
    y_dir: str = 'def'
    y_type: str = 'lin'
    
    # Axis limits
    x_lim: Optional[Tuple[float, float]] = None
    y_lim: Optional[Tuple[float, float]] = None
    
    # Display settings
    dpi_plot: int = 100
    dpi_save: int = 300
    
    # Residual multiplier for 1D fit plots
    res_mult: float = 5

    # 2D plot settings
    z_colormap: str = 'viridis'
    z_colorbar: str = 'ver'  # 'ver' or 'hor'
    # 2D data handling
    data_slice: Optional[List[List[int]]] = None
    z_lim: Optional[Tuple[float, float]] = None
    
    # Line plot styles
    colors: Optional[List[str]] = None
    linestyles: Optional[List[str]] = None
    linewidths: Optional[List[float]] = None
    markers: Optional[List[str]] = None
    markersizes: Optional[List[float]] = None
    
    # Plot annotations
    legend: Optional[List[str]] = None
    waterfall: float = 0
    vlines: Optional[List[float]] = None
    hlines: Optional[List[float]] = None
    ticksize: Optional[float] = None
    
    # Normalization
    y_norm: int = 0  # 0: no normalization, 1: normalize to [0,1]
    y_scale: Optional[List[float]] = None
    
    @classmethod
    def from_project(cls, project, **overrides):
        """
        Create PlotConfig from Project settings.
        
        Parameters
        ----------
        project : Project
            Project instance with plot settings
        **overrides : dict
            Any parameters to override from project defaults
            
        Returns
        -------
        PlotConfig
            Configuration object with settings from project
            
        Examples
        --------
        >>> project = Project(path='...')
        >>> config = PlotConfig.from_project(project)
        >>> config = PlotConfig.from_project(project, x_label='Custom Label')
        """
        config_dict = {
            'x_label': project.e_label,
            'y_label': project.t_label,
            'z_label': project.z_label,
            'x_dir': project.x_dir,
            'x_type': project.x_type,
            'y_dir': project.y_dir,
            'y_type': project.y_type,
            'dpi_plot': project.dpi_plt,
            'dpi_save': project.dpi_save,
            'z_colormap': project.z_colormap,
            'z_colorbar': project.z_colorbar,
            'res_mult': project.res_mult,
        }
        
        # Apply any overrides
        config_dict.update(overrides)
        
        return cls(**config_dict)
    
    #
    def update(self, **kwargs):
        """
        Update configuration attributes.
        
        Parameters
        ----------
        **kwargs : dict
            Attributes to update
            
        Returns
        -------
        PlotConfig
            Self (for chaining)
            
        Examples
        --------
        >>> config = PlotConfig()
        >>> config.update(x_label='Energy', dpi_plot=150)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"PlotConfig has no attribute '{key}'")
        return self
    
    #
    def copy(self, **overrides):
        """
        Create a copy of this config with optional overrides.
        
        Parameters
        ----------
        **overrides : dict
            Attributes to override in the copy
            
        Returns
        -------
        PlotConfig
            New configuration object
            
        Examples
        --------
        >>> config = PlotConfig(x_label='Energy')
        >>> config2 = config.copy(y_label='Intensity')
        """
        import copy as cp
        new_config = cp.copy(self)
        new_config.update(**overrides)
        return new_config