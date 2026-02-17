"""
Spectrum generation functions for fitting.

This module provides the interface between trspecfit's model/component/parameter
(mcp) system and the fitting routines. It contains functions that generate
spectral data from model parameters during optimization.

The functions here are called by the fitting engine (fitlib.residual_fun) on
every iteration to compute the current model prediction, which is then compared
to experimental data.

Key Concepts
------------
- fit_model_mcp: Default spectrum generator using mcp.Model system
- Custom generators: Users can define alternative spectrum functions
  and specify them via Project.spec_fun_str
  ["x, par, plot_ind, args" is the typical fit function structure]

Architecture
------------
The fitting workflow is:
1. Optimizer proposes new parameter values
2. fitlib.residual_fun calls spectrum function (this module)
3. Spectrum function generates model prediction
4. Residual = data - model is computed and returned to optimizer
"""

from trspecfit.mcp import Model
from IPython.display import display 
import numpy as np
from typing import Sequence, Union

#
def fit_model_mcp(
    x: Union[Sequence[float], np.ndarray],
    par: Union[Sequence[float], np.ndarray],
    plot_ind: bool,
    model: Model,
    dim: int,
    debug: bool
) -> Union[np.ndarray, list[np.ndarray]]:
    """
    Generate spectrum from mcp.Model for fitting or visualization.
    
    This is the default spectrum generation function used by trspecfit. It
    updates model parameters, evaluates the model, and returns either the
    complete spectrum or individual component spectra.
    
    Parameters
    ----------
    x : array-like
        Independent variable axis (energy or time). Not directly used here
        as model contains its own axes, but required for fitting interface
        compatibility.

    par : list or array-like
        Parameter values in same order as model.par_names. These are the
        current values proposed by the optimizer during fitting.
    plot_ind : bool
        Component return mode:

        - False: Return sum of all components (used during fitting)
        - True: Return list of individual component spectra (for visualization)

    model : mcp.Model
        Model instance containing components and parameter structure.
        Modified in-place to reflect current parameter values.
    dim : int
        Dimensionality of spectrum to generate:

        - 1: Generate 1D spectrum (energy-resolved or time-resolved)
        - 2: Generate 2D spectrum (time- and energy-resolved)

    debug : bool
        If True, print parameter values and detailed model info to console.
        Useful for debugging optimization issues.
    
    Returns
    -------
    ndarray or list of ndarray
        Generated spectrum or spectra:
        - If dim=1 and plot_ind=False: 1D array (sum of components)
        - If dim=1 and plot_ind=True: List of 1D arrays (individual components)
        - If dim=2: 2D array (time × energy), regardless of plot_ind
    
    Examples
    --------
    >>> # During fitting (1D)
    >>> spectrum = fit_model_mcp(energy, par_values, False, model, 1, False)
    >>> residual = data - spectrum
    
    >>> # For visualization (1D, individual components)
    >>> components = fit_model_mcp(energy, par_values, True, model, 1, False)
    >>> for i, comp in enumerate(components):
    ...     plt.plot(energy, comp, label=f'Component {i}')
    
    >>> # During fitting (2D)
    >>> spectrum_2D = fit_model_mcp(energy, par_values, False, model, 2, False)
    >>> residual_2D = data_2D - spectrum_2D
    
    Notes
    -----
    **Function Signature:**
    The signature follows the standard form [x, par, plot_ind, args] required
    by fitlib.residual_fun. The 'args' tuple contains (model, dim, debug).
    
    **Parameter Update:**
    This function updates model.lmfit_pars in-place via model.update_value().
    The model retains these values after the function returns.
    
    **2D Behavior:**
    For 2D models, plot_ind is ignored and the full 2D spectrum is always
    returned. Individual component plotting for 2D is typically done by
    examining time slices.
    
    **Performance:**
    2D spectrum generation can be slow for large grids or complex models
    with many time-dependent parameters. Consider:
    - Reducing time/energy grid density during initial fits
    - Using fit_SliceBySlice for quasi-independent time points
    - Implementing parallel evaluation (model.create_value2D_parallel)
    """
    par_values: Union[list[float], np.ndarray]
    if isinstance(par, np.ndarray):
        par_values = par
    else:
        par_values = list(par)
    model.update_value(new_par_values=par_values)  # Update lmfit parameters
    
    if debug: 
        display(model.lmfit_pars)
        model.print_all_pars(detail=1)
    
    # Create energy- (and time-)resolved spectrum/data
    if dim == 1:  # 1D
        if plot_ind:  # Return individual components
            model.create_value1D(store1D=1)
            return model.component_spectra
        else:  # Return sum of all components
            model.create_value1D()
            if model.value1D is None:
                raise RuntimeError("Model evaluation did not produce value1D")
            return model.value1D
        
    elif dim == 2:  # 2D
        model.create_value2D()
        if model.value2D is None:
            raise RuntimeError("Model evaluation did not produce value2D")
        return model.value2D
    else:
        raise ValueError(f"Unsupported dim={dim}; expected 1 or 2")
