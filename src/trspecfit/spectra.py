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
  ["x, par, plot_sum, args" is the typical fit function structure]

Architecture
------------
The fitting workflow is:
1. Optimizer proposes new parameter values
2. fitlib.residual_fun calls spectrum function (this module)
3. Spectrum function generates model prediction
4. Residual = data - model is computed and returned to optimizer
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from trspecfit.eval_2d import evaluate_2d
from trspecfit.graph_ir import ScheduledPlan2D
from trspecfit.mcp import Model


#
def fit_model_mcp(
    x: Sequence[float] | np.ndarray,
    par: Sequence[float] | np.ndarray,
    plot_sum: bool,
    model: Model,
    dim: int,
) -> np.ndarray | list[np.ndarray]:
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
        Parameter values in same order as model.parameter_names. These are the
        current values proposed by the optimizer during fitting.
    plot_sum : bool
        Component return mode:

        - True: Return sum of all components (used during fitting)
        - False: Return list of individual component spectra (for visualization)

    model : mcp.Model
        Model instance containing components and parameter structure.
        Modified in-place to reflect current parameter values.
    dim : int
        Dimensionality of spectrum to generate:

        - 1: Generate 1D spectrum (energy-resolved or time-resolved)
        - 2: Generate 2D spectrum (time- and energy-resolved)

    Returns
    -------
    ndarray or list of ndarray
        Generated spectrum or spectra:
        - If dim=1 and plot_sum=True: 1D array (sum of components)
        - If dim=1 and plot_sum=False: List of 1D arrays (individual components)
        - If dim=2: 2D array (time x energy), regardless of plot_sum

    Examples
    --------
    >>> # During fitting (1D)
    >>> spectrum = fit_model_mcp(energy, par_values, True, model, 1)
    >>> residual = data - spectrum

    >>> # For visualization (1D, individual components)
    >>> components = fit_model_mcp(energy, par_values, False, model, 1)
    >>> for i, comp in enumerate(components):
    ...     plt.plot(energy, comp, label=f'Component {i}')

    >>> # During fitting (2D)
    >>> spectrum_2d = fit_model_mcp(energy, par_values, True, model, 2)
    >>> residual_2d = data_2d - spectrum_2d

    Notes
    -----
    **Function Signature:**
    The signature follows the standard form [x, par, plot_sum, args] required
    by fitlib.residual_fun. The 'args' tuple contains (model, dim).

    **Parameter Update:**
    This function updates model.lmfit_pars in-place via model.update_value().
    The model retains these values after the function returns.

    **2D Behavior:**
    For 2D models, plot_sum is ignored and the full 2D spectrum is always
    returned. Individual component plotting for 2D is typically done by
    examining time slices.

    **Performance:**
    2D spectrum generation can be slow for large grids or complex models
    with many time-dependent parameters. Consider:
    - Reducing time/energy grid density during initial fits
    - Using fit_slice_by_slice for quasi-independent time points
    - Implementing parallel evaluation (model.create_value_2d_parallel)
    """

    par_values: list[float] | np.ndarray
    if isinstance(par, np.ndarray):
        par_values = par
    else:
        par_values = list(par)
    model.update_value(new_par_values=par_values)  # Update lmfit parameters

    # Create energy- (and time-)resolved spectrum/data
    if dim == 1:  # 1D
        if not plot_sum:  # Return individual components
            model.create_value_1d(store_1d=1)
            return model.component_spectra
        # Return sum of all components
        model.create_value_1d()
        if model.value_1d is None:
            raise RuntimeError("Model evaluation did not produce value_1d")
        return model.value_1d

    if dim == 2:  # 2D
        model.create_value_2d()
        if model.value_2d is None:
            raise RuntimeError("Model evaluation did not produce value_2d")
        return model.value_2d
    raise ValueError(f"Unsupported dim={dim}; expected 1 or 2")


#
def fit_model_gir(
    x: Sequence[float] | np.ndarray,
    par: Sequence[float] | np.ndarray,
    plot_sum: bool,
    *args: Any,
) -> np.ndarray | list[np.ndarray]:
    """Generate spectrum using the compiled GIR backend when available.

    When the first element of *args* is a :class:`ScheduledPlan2D` the
    compiled 2-D evaluator is used.  Otherwise the call is forwarded to
    :func:`fit_model_mcp` (1-D fits, or 2-D models that could not be
    lowered).

    Parameters
    ----------
    x : array-like
        Independent variable axis (energy or time).
    par : array-like
        Full parameter vector (all params, fixed + varying).
    plot_sum : bool
        Component return mode (forwarded to ``fit_model_mcp`` on fallback).
    *args
        Either ``(plan, theta_indices)`` for the compiled path, or
        ``(model, dim)`` for the interpreter fallback.
    """

    if isinstance(args[0], ScheduledPlan2D):
        plan: ScheduledPlan2D = args[0]
        theta_indices: np.ndarray = args[1]
        par_arr = np.asarray(par, dtype=np.float64)
        theta = par_arr[theta_indices]
        return evaluate_2d(plan, theta)
    # 1D or non-lowerable 2D fallback
    return fit_model_mcp(x, par, plot_sum, *args)


#
def fit_model_compare(
    x: Sequence[float] | np.ndarray,
    par: Sequence[float] | np.ndarray,
    plot_sum: bool,
    *args: Any,
) -> np.ndarray | list[np.ndarray]:
    """Run both GIR and interpreter paths, compare results.

    When the first element of *args* is a :class:`ScheduledPlan2D` the
    compiled path is executed and its output compared against the
    interpreter via ``np.testing.assert_allclose``.  On fallback the
    interpreter is called directly.

    Parameters
    ----------
    x : array-like
        Independent variable axis (energy or time).
    par : array-like
        Full parameter vector (all params, fixed + varying).
    plot_sum : bool
        Component return mode.
    *args
        ``(plan, theta_indices, model, dim)`` for the comparison path,
        or ``(model, dim)`` for interpreter-only fallback.
    """

    if isinstance(args[0], ScheduledPlan2D):
        plan: ScheduledPlan2D = args[0]
        theta_indices: np.ndarray = args[1]
        model: Model = args[2]
        dim: int = args[3]
        fast = fit_model_gir(x, par, plot_sum, plan, theta_indices)
        slow = fit_model_mcp(x, par, plot_sum, model, dim)
        np.testing.assert_allclose(fast, slow, rtol=1e-10, atol=1e-10)
        return fast
    return fit_model_mcp(x, par, plot_sum, *args)


#
def fit_project_mcp(
    x: Sequence[float] | np.ndarray,
    par: Sequence[float] | np.ndarray,
    plot_sum: bool,
    project_fit_info: dict[str, Any],
    dim: int,
) -> np.ndarray:
    """
    Generate concatenated spectra from multiple files for project-level fitting.

    Distributes combined optimizer parameters to individual file models,
    evaluates each model, slices to each file's fit region, and returns
    one concatenated array for residual computation.

    Parameters
    ----------
    x : array-like
        Unused (kept for fit-function signature compatibility).
    par : list or array-like
        Combined parameter values proposed by the optimizer, ordered to
        match ``project_fit_info["par_names"]``.
    plot_sum : bool
        Unused (kept for signature compatibility). Always returns sum.
    project_fit_info : dict
        Fitting context built by ``Project._build_fit_params()``:

        - ``"mapping"``: list of ``(project_name, file_idx, local_name)``
        - ``"files"``: list of File objects (limits read from ``f.e_lim``/``f.t_lim``)
        - ``"models"``: list of Model objects (one per file)
        - ``"par_names"``: list of combined parameter names

    dim : int
        Must be 2 (project-level fitting is 2D only).

    Returns
    -------
    ndarray
        Concatenated (flattened) fit arrays from all files.
    """

    par_values = list(par) if not isinstance(par, np.ndarray) else par

    mapping = project_fit_info["mapping"]
    files = project_fit_info["files"]
    models = project_fit_info["models"]
    par_names = project_fit_info["par_names"]

    # Build name→value lookup from the combined parameter vector
    par_lookup: dict[str, float] = {}
    for i, name in enumerate(par_names):
        par_lookup[name] = float(par_values[i])

    # Distribute values to each file's model
    for project_name, file_idx, local_name in mapping:
        model = models[file_idx]
        if local_name in model.lmfit_pars:
            model.lmfit_pars[local_name].value = par_lookup[project_name]

    # Evaluate each file and collect sliced results
    slices: list[np.ndarray] = []
    for i, (model, f) in enumerate(zip(models, files, strict=True)):
        model.create_value_2d()
        if model.value_2d is None:
            raise RuntimeError(
                f"Model evaluation for file {i} did not produce value_2d"
            )
        fit_2d = model.value_2d
        # Apply per-file slicing (read limits from file directly)
        if f.e_lim and f.t_lim:
            fit_2d = fit_2d[f.t_lim[0] : f.t_lim[1], f.e_lim[0] : f.e_lim[1]]
        elif f.e_lim:
            fit_2d = fit_2d[:, f.e_lim[0] : f.e_lim[1]]
        elif f.t_lim:
            fit_2d = fit_2d[f.t_lim[0] : f.t_lim[1], :]
        slices.append(fit_2d.flatten())

    return np.concatenate(slices)
