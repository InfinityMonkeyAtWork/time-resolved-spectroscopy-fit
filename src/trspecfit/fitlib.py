"""
1D and 2D peak fitting functions based on lmfit.

This module provides the complete fitting workflow for spectroscopy data:
- Residual calculation for optimization
- Fitting with lmfit (including global + local optimization)
- Confidence interval estimation (lmfit.conf_interval)
- MCMC sampling via lmfit.emcee for uncertainty quantification
- Result visualization and export

The fitting functions here work with the MCP (Model/Component/Parameter) system
from trspecfit.mcp to provide a flexible, component-based fitting framework.

Key Functions
-------------
residual_fun : Compute residual for optimizer
fit_wrapper : Main fitting function with CI and MCMC support
eval_model_curves_1d : Evaluate model curves for the ``utils.plot`` renderers
"""

import copy
import math
import multiprocessing
import pathlib
import time
from collections.abc import Callable, Sequence
from typing import Any, cast

import lmfit
import numpy as np
import pandas as pd
from IPython.display import display
from lmfit.minimizer import MinimizerResult
from numpy.typing import ArrayLike

from trspecfit import spectra
from trspecfit.config.plot import PlotConfig
from trspecfit.utils import lmfit as ulmfit
from trspecfit.utils import spawn as uspawn

# Define a type alias for file paths
type PathLike = str | pathlib.Path


#
def _result_params(result: MinimizerResult) -> lmfit.parameter.Parameters:
    """Return typed lmfit parameters from a MinimizerResult-like object."""

    params = getattr(result, "params", None)
    if not isinstance(params, lmfit.parameter.Parameters):
        raise TypeError(
            f"Expected lmfit.Parameters in result.params, got {type(params).__name__}"
        )
    return params


#
def _result_errorbars(result: MinimizerResult) -> bool:
    """Safely read MinimizerResult.errorbars with fallback."""

    return bool(getattr(result, "errorbars", False))


#
def compute_fit_metrics(
    *,
    observed: np.ndarray,
    fit: np.ndarray,
    n_free_pars: int | None,
    sigma_eff: float | None = None,
) -> dict[str, float]:
    """
    Compute fit-quality metrics from observed and fitted arrays.

    Always emits the raw (unweighted) diagnostics ``chi2_raw`` and
    ``chi2_red_raw`` — these match lmfit's ``MinimizerResult.chisqr / .redchi``
    for unweighted fits. When ``sigma_eff`` is provided, also emits the
    σ-calibrated ``chi2`` and ``chi2_red`` (``≈ 1`` for a fit at the noise
    floor); without ``sigma_eff`` both calibrated values are ``NaN``.
    ``r2``, ``aic``, ``bic`` are unaffected by σ (R² is dimensionless;
    AIC/BIC depend on raw χ² but their *differences* are invariant under
    constant rescaling).

    Parameters
    ----------
    observed : ndarray
        Data view that was fit against (e.g. ``data_base`` for baseline,
        cropped ``data`` for sbs/2d). Any shape; the array is flattened.
    fit : ndarray
        Model evaluated at final parameters. Must broadcast to ``observed``.
    n_free_pars : int or None
        Number of varying (non-fixed, non-expression) parameters in the fit.
        Used as ``nvarys`` in the AIC/BIC and reduced-χ² formulas. ``None``
        means the parameter count is undefined for this data view — the
        per-file projections of a project-level joint fit, where the joint
        count does not decompose by file — and every count-dependent metric
        (``chi2_red_raw``, ``chi2_red``, ``aic``, ``bic``) is ``NaN``.
    sigma_eff : float, optional
        Effective noise σ on the fit's data view (per-pixel for SbS/2D,
        ``σ_pixel / √N_avg`` for baseline). When ``None`` / ``NaN`` /
        non-positive, the calibrated ``chi2``/``chi2_red`` fields are
        ``NaN``. Caller is responsible for any view-specific scaling.

    Returns
    -------
    dict
        ``{"chi2_raw", "chi2_red_raw", "chi2", "chi2_red", "r2", "aic",
        "bic"}``. ``chi2_red_raw``, ``aic``, ``bic`` are ``NaN`` when
        ``n_free_pars`` is ``None``, ``ndata <= n_free_pars``, or
        ``chi2_raw == 0`` (degenerate fits); ``chi2`` / ``chi2_red`` are
        additionally ``NaN`` when ``sigma_eff`` is missing or invalid.
    """

    residual = np.asarray(observed) - np.asarray(fit)
    ndata = residual.size
    chi2_raw = float(np.sum(residual**2))

    obs_flat = np.asarray(observed).ravel()
    ss_tot = float(np.sum((obs_flat - obs_flat.mean()) ** 2))
    r2 = float("nan") if ss_tot == 0.0 else 1.0 - chi2_raw / ss_tot

    if n_free_pars is None:
        chi2_red_raw = float("nan")
        aic = float("nan")
        bic = float("nan")
    else:
        dof = ndata - n_free_pars
        chi2_red_raw = chi2_raw / dof if dof > 0 else float("nan")

        if chi2_raw > 0 and ndata > 0:
            log_chi2_per_n = math.log(chi2_raw / ndata)
            aic = ndata * log_chi2_per_n + 2 * n_free_pars
            bic = ndata * log_chi2_per_n + math.log(ndata) * n_free_pars
        else:
            aic = float("nan")
            bic = float("nan")

    if sigma_eff is None or not np.isfinite(sigma_eff) or sigma_eff <= 0:
        chi2 = float("nan")
        chi2_red = float("nan")
    else:
        sigma_sq = float(sigma_eff) ** 2
        chi2 = chi2_raw / sigma_sq
        chi2_red = (
            chi2_red_raw / sigma_sq if np.isfinite(chi2_red_raw) else float("nan")
        )

    return {
        "chi2_raw": chi2_raw,
        "chi2_red_raw": chi2_red_raw,
        "chi2": chi2,
        "chi2_red": chi2_red,
        "r2": r2,
        "aic": aic,
        "bic": bic,
    }


#
def _fit_window_slices(
    ndim: int, e_lim: list[int] | None, t_lim: list[int] | None
) -> tuple[slice, ...]:
    """Build array slices selecting the user-defined fit window.

    Empty or None limits select the full axis. 1D data is indexed as
    [energy]; 2D data as [time, energy].
    """

    e_slice = slice(e_lim[0], e_lim[1]) if e_lim else slice(None)
    if ndim == 1:
        return (e_slice,)
    if ndim == 2:
        t_slice = slice(t_lim[0], t_lim[1]) if t_lim else slice(None)
        return (t_slice, e_slice)
    raise ValueError("data must be 1D or 2D")


#
def residual_fun(
    par: Any,
    x: ArrayLike,
    data: np.ndarray,
    fit_fun_str: str,
    unpack: int = 0,
    e_lim: list[int] | None = None,
    t_lim: list[int] | None = None,
    res_type: str = "lmfit",
    args: Sequence[Any] | None = None,
) -> np.ndarray | float:
    """
    Compute residual (data - fit) for optimization and analysis.

    This function is called repeatedly by optimizers to evaluate how well
    current parameters fit the data. It supports both 1D and 2D fitting,
    with options for limiting the fitting region and returning different
    residual representations.

    Parameters
    ----------
    par : list or lmfit.Parameters
        Current parameter values. If lmfit.Parameters, values are extracted.
        Order must match the fit function's parameter expectations.
    x : ndarray
        Independent variable axis (energy or time). Passed to fit function
        but may not be used if function has its own axes.
    data : ndarray (1D or 2D)
        Experimental data to fit. Shape determines dimensionality:

        - 1D: [n_energy] for energy-resolved fits
        - 2D: [n_time, n_energy] for time- and energy-resolved fits

    fit_fun_str : str
        Name of fit function in ``trspecfit.spectra``
        (e.g., ``'fit_model_mcp'``, ``'fit_model_gir'``)
    unpack : {0, 1}, default=0
        Parameter passing mode:

        - 0: Pass parameters as list: ``fit_fun(x, par, ...)``
        - 1: Unpack parameters: ``fit_fun(x, *par, ...)``

    e_lim : list of int, default=[]
        Energy axis limits [start, stop) for residual calculation.
        Uses slice notation: data[e_lim[0]:e_lim[1]]
        Empty list uses full energy range.
    t_lim : list of int, default=[]
        Time axis limits [start, stop) for residual calculation.
        Uses slice notation: data[t_lim[0]:t_lim[1]]
        Empty list uses full time range.
    res_type : {'lmfit', 'RSS', 'abs', 'res', 'fit'}, default='lmfit'
        Return type:

        - 'lmfit': Residual array (1D) or flattened residual (2D) for lmfit
        - 'RSS': Residual sum of squares (scalar, for scipy.optimize)
        - 'abs': Sum of absolute residuals (scalar, L1 norm)
        - 'res': Raw residual array (data - fit)
        - 'fit': Return fit itself instead of residual

    args : tuple, default=()
        Additional arguments for fit function, passed via ``*args``

    Returns
    -------
    ndarray or float
        Return type depends on res_type:
        - 'lmfit': ndarray (1D for 1D data, flattened for 2D data)
        - 'RSS', 'abs': float (scalar metric)
        - 'res': ndarray (same shape as data)
        - 'fit': ndarray (same shape as data)
    """

    if e_lim is None:
        e_lim = []
    if t_lim is None:
        t_lim = []
    if args is None:
        args = ()

    # define the fit function
    fit_fun = getattr(spectra, fit_fun_str)

    # if the minimizer calling this is from the lmfit package, then
    # extract the value from their lmfit.Parameter() (dictionary)
    # or if list of [value, vary(, min, max)] transition to val list
    par = ulmfit.par_extract(par, return_type="list")

    # compute the fit curve [plot_sum has to be hardcoded as True/1 here]
    if unpack == 1:
        fit = fit_fun(x, *par, True, *args)
    elif unpack == 0:
        fit = fit_fun(x, par, True, *args)
    else:
        raise ValueError("unpack must be 0 or 1")
    fit_arr = np.asarray(fit)
    data_arr = np.asarray(data)

    # select user-defined region to consider for residual computation
    window = _fit_window_slices(data_arr.ndim, e_lim, t_lim)
    residual = data_arr[window] - fit_arr[window]

    # type of residual to return
    if res_type == "RSS":
        return float(np.sum(residual**2))
    if res_type == "abs":
        return float(np.sum(np.abs(residual)))
    if res_type == "res":
        return np.asarray(residual)
    if res_type == "lmfit":
        if len(data_arr.shape) == 1:  # 1D data
            return np.asarray(residual)
        if len(data_arr.shape) == 2:  # 2D data
            return np.asarray(residual).flatten()
    elif res_type == "fit":
        return fit_arr
    raise ValueError(f"Unknown res_type '{res_type}'")


#
def jacobian_fun(
    par: Any,
    x: ArrayLike,
    data: np.ndarray,
    fit_fun_str: str,
    unpack: int = 0,
    e_lim: list[int] | None = None,
    t_lim: list[int] | None = None,
    res_type: str = "lmfit",
    args: Sequence[Any] | None = None,
) -> np.ndarray:
    """Analytic Jacobian of :func:`residual_fun` for lmfit's ``Dfun``.

    The signature mirrors ``residual_fun`` because lmfit calls the
    Jacobian with the same ``fcn_args``.  Requires the
    ``fit_model_jax`` dispatch convention:
    ``args = (evaluator, jacobian, theta_indices, model, dim)`` with
    *jacobian* from ``eval_jax.make_jacobian_2d_jax``.

    Returns
    -------
    ndarray
        ``d(residual)/d(varying params)``, shape
        ``(n_residuals, n_varys)``, columns in lmfit varying-parameter
        order (``col_deriv=0``).  Residual is ``data - fit``, so this
        is the negated model Jacobian over the fit window.
    """

    if e_lim is None:
        e_lim = []
    if t_lim is None:
        t_lim = []
    if args is None or not callable(args[1]):
        raise ValueError(
            "jacobian_fun requires the fit_model_jax dispatch args "
            "(evaluator, jacobian, theta_indices, model, dim)."
        )
    jacobian = args[1]
    theta_indices: np.ndarray = args[2]
    model = args[3]

    par_values = np.asarray(
        ulmfit.par_extract(par, return_type="list"), dtype=np.float64
    )
    # (n_time, n_energy, n_opt)
    jac = np.asarray(jacobian(par_values[theta_indices]), dtype=np.float64)

    window = _fit_window_slices(2, e_lim, t_lim)
    n_opt = jac.shape[-1]
    d_res = -jac[window].reshape(-1, n_opt)

    # Column order: plan opt order -> lmfit varying-parameter order.
    opt_names = [model.parameter_names[int(i)] for i in theta_indices]
    var_names = [name for name in par if par[name].vary]
    if sorted(opt_names) != sorted(var_names):
        raise RuntimeError(
            "JAX Jacobian column mismatch: plan optimizer parameters "
            f"{opt_names} do not match lmfit varying parameters {var_names}."
        )
    columns = [opt_names.index(name) for name in var_names]
    return d_res[:, columns]


#
def jacobian_fun_project(
    par: Any,
    x: ArrayLike,
    data: np.ndarray,
    fit_fun_str: str,
    unpack: int = 0,
    e_lim: list[int] | None = None,
    t_lim: list[int] | None = None,
    res_type: str = "lmfit",
    args: Sequence[Any] | None = None,
) -> np.ndarray:
    """Analytic joint Jacobian for project-level fits (lmfit ``Dfun``).

    Project counterpart of :func:`jacobian_fun`. Requires the
    ``fit_project_jax`` dispatch convention:
    ``args = (evaluator, jacobian, theta_c_indices, var_names, dim)``
    with *jacobian* from ``eval_jax.make_project_jacobian_2d_jax``.
    Per-file fit windows are applied inside the fused jacobian, so no
    window slicing happens here (``e_lim``/``t_lim`` are empty for
    project fits).

    Returns
    -------
    ndarray
        ``d(residual)/d(varying params)``, shape
        ``(n_residuals_total, n_varys)``, columns in lmfit
        varying-parameter order (``col_deriv=0``). Residual is
        ``data - fit``, so this is the negated fused model Jacobian.
    """

    if args is None or not callable(args[1]):
        raise ValueError(
            "jacobian_fun_project requires the fit_project_jax dispatch "
            "args (evaluator, jacobian, theta_c_indices, var_names, dim)."
        )
    jacobian = args[1]
    theta_c_indices: np.ndarray = args[2]
    theta_names: list[str] = list(args[3])

    par_values = np.asarray(
        ulmfit.par_extract(par, return_type="list"), dtype=np.float64
    )
    # (n_residuals_total, n_opt), columns in theta_c order
    jac = np.asarray(jacobian(par_values[theta_c_indices]), dtype=np.float64)
    d_res = -jac

    # Column order: theta_c order -> lmfit varying-parameter order.
    var_names = [name for name in par if par[name].vary]
    if sorted(theta_names) != sorted(var_names):
        raise RuntimeError(
            "Project JAX Jacobian column mismatch: combined optimizer "
            f"parameters {theta_names} do not match lmfit varying "
            f"parameters {var_names}."
        )
    columns = [theta_names.index(name) for name in var_names]
    return d_res[:, columns]


#
def time_display(
    t_start: float, print_str: str = "", *, return_delta_seconds: bool = False
) -> float | None:
    """
    Display elapsed time in human-readable format.

    Computes time elapsed since t_start and displays it with appropriate
    units (seconds, minutes, hours, or days). Useful for benchmarking
    fitting operations.

    Parameters
    ----------
    t_start : float
        Start time from time.time()
    print_str : str, default=''
        Prefix string for display (e.g., 'Fitting completed in: ')
    return_delta_seconds : bool, default=False
        If True, also return elapsed seconds as float

    Returns
    -------
    None or float
        None normally, or elapsed seconds if return_delta_seconds=True

    Examples
    --------
    >>> import time
    >>> t0 = time.time()
    >>> # ... do some work ...
    >>> time_display(t0, 'Operation completed in: ')
    Operation completed in: 01:23.456(mm:ss.ms)
    """

    t_stop = time.time()
    seconds = t_stop - t_start

    str_format = "ss.ms"
    minutes, seconds = divmod(seconds, 60)
    delta_format = f"{seconds:06.3f}"

    if minutes > 0:
        str_format = "mm:" + str_format
        hours, minutes = divmod(minutes, 60)
        delta_format = f"{math.floor(minutes):02d}:{delta_format}"

        if hours > 0:
            str_format = "hh:" + str_format
            days, hours = divmod(hours, 24)
            delta_format = f"{math.floor(hours):02d}" + ":" + delta_format

            if days > 0:
                str_format = "ddd:" + str_format
                delta_format = f"{math.floor(days):03d}" + ":" + delta_format

    print(print_str + delta_format + f"({str_format})")
    #
    if return_delta_seconds:
        return seconds
    return None


#
# error estimation helper functions
#


#
def sigma_dict() -> dict[str, float]:
    """
    Get percentage of distribution within N-sigma intervals.

    Returns dictionary mapping sigma values (as strings) to the percentage
    of a normal distribution contained within ±N sigma of the mean.

    Returns
    -------
    dict
        Keys: Sigma values as strings ('0.5', '1.0', ..., '5.0')
        Values: Percentage of distribution (float)
    """

    return {
        "0.5": 38.2924922548026,
        "1.0": 68.2689492137086,
        "1.5": 86.6385597462284,
        "2.0": 95.4499736103642,
        "2.5": 98.7580669348448,
        "3.0": 99.7300203936740,
        "3.5": 99.9534741841929,
        "4.0": 99.9936657516334,
        "4.5": 99.9993204653751,
        "5.0": 99.9999426696856,
    }


#
def sigma_start_stop_percent(sigma_list: Sequence[float]) -> list[list[float]]:
    """
    Calculate percentile bounds for symmetric confidence intervals.

    For each sigma level, computes the lower and upper percentiles that
    define a symmetric interval containing the specified percentage of
    a normal distribution.

    Parameters
    ----------
    sigma_list : list of float
        Sigma values (e.g., [1, 2, 3] for 1σ, 2σ, 3σ intervals).
        Values must be in range [0.5, 5.0] in 0.5 increments.

    Returns
    -------
    list of [float, float]
        Percentile bounds for each sigma level.
        Each element: [lower_percentile, upper_percentile]
        Returns empty list if any sigma value not supported.

    Examples
    --------
    >>> # 1σ, 2σ, 3σ intervals
    >>> bounds = sigma_start_stop_percent([1.0, 2.0, 3.0])
    >>> print(bounds)
    [[15.87, 84.13], [2.28, 97.72], [0.14, 99.86]]

    >>> # Use with numpy.percentile on MCMC samples
    >>> import numpy as np
    >>> samples = np.random.normal(0, 1, 10000)
    >>> bounds_1sigma = sigma_start_stop_percent([1.0])[0]
    >>> ci = np.percentile(samples, bounds_1sigma)
    >>> print(f"1σ interval: [{ci[0]:.2f}, {ci[1]:.2f}]")
    1σ interval: [-1.01, 1.02]
    """

    borders_pc: list[list[float]] = []  # low/high borders in percent
    for sigma in sigma_list:
        a2a_total_raw = sigma_dict().get(f"{sigma:.1f}", "sigma value not supported")
        if isinstance(a2a_total_raw, str):
            print(a2a_total_raw)
            borders_pc = []
        else:
            a2a_total = float(a2a_total_raw)
            a_exclude_to_a_total = 100 - a2a_total
            borders_pc.append(
                [a_exclude_to_a_total / 2, 100 - a_exclude_to_a_total / 2]
            )

    return borders_pc


#
# wrapper for lmfit fit, confidence interval, and lmfit.emcee functions
#
def fit_wrapper(
    const: tuple[Any, ...],
    args: tuple[Any, ...],
    par_names: list[str],
    par: Any,
    stages: int,
    ci_sigmas: list[float] | None = None,
    try_ci: int = 1,
    mc_settings: ulmfit.MC | None = None,
    fit_alg_1: str = "Nelder",
    fit_alg_2: str = "leastsq",
    jac_fun: Callable[..., np.ndarray] | None = None,
    seed: int | None = None,
    show_output: int = 0,
) -> ulmfit.FitOutput:
    """
    Comprehensive fitting wrapper with optimization, CI, and MCMC.

    This is the main fitting function in trspecfit. It handles:
    - Single or two-stage optimization
    - Confidence interval estimation via lmfit.conf_interval
    - MCMC sampling via lmfit.emcee
    - Result visualization (never writes to disk)

    Two-stage fitting (stages=2) is recommended for robust optimization:
    first finds global minimum with Nelder-Mead, then refines locally with
    leastsq for accurate error bars.

    Parameters
    ----------
    const : tuple
        Constants for residual_fun:
        (x, data, function_str, unpack, e_lim, t_lim)
    args : tuple
        Arguments for fit function (passed to residual_fun):
        Typically (model, dim) for MCP models
    par_names : list of str
        Parameter names in order (for display and export)
    par : lmfit.Parameters or list
        Initial parameter guess:

        - lmfit.Parameters: Use directly
        - list: Convert to lmfit.Parameters using par_names.
          Each element: ``[value, vary, min, max]`` or ``['expression']``

    stages : {1, 2}
        Number of optimization stages:

        - 1: Single optimization with ``fit_alg_1``
        - 2: Two-stage fit (``fit_alg_1`` then ``fit_alg_2``)

    ci_sigmas : list of int or float, default=[1,2,3]
        Confidence levels in σ units for CI and MCMC quantile tables
        (e.g., [1,2,3] for 1σ, 2σ, 3σ). Not the data noise σ — that is
        ``File.set_sigma()`` / ``sigma_data``.
    try_ci : {0, 1}, default=1
        Confidence interval estimation:

        - 0: Skip CI calculation
        - 1: Calculate CI if error bars available (result.errorbars=True)

    mc_settings : ulmfit.MC, default=ulmfit.MC()
        MCMC configuration (``use_mc``: 0 skip, 1 always, 2 if CI fails).
        Knobs left at None are resolved from the optimizer result into a
        copy that is returned as ``FitOutput.mc_settings``; the object
        passed in is never mutated. See ``ulmfit.MC`` for every knob.
    fit_alg_1 : str, default='Nelder'
        First/only optimization method. Common options:

        - 'Nelder': Nelder-Mead (robust, no gradients, slower)
        - 'powell': Powell (robust, no gradients)
        - 'leastsq': Levenberg-Marquardt (fast, needs good initial guess)

        See lmfit documentation for full list
    fit_alg_2 : str, default='leastsq'
        Second optimization method (stages=2 only).
        Typically 'leastsq' for accurate local optimization and error bars.
    jac_fun : callable, optional
        Analytic Jacobian with the same signature as ``residual_fun``
        (e.g. :func:`jacobian_fun` on the JAX backend).  Passed to
        lmfit as ``Dfun`` for stages whose method is ``'leastsq'``;
        ignored for gradient-free methods.
    seed : int, optional
        Optimizer RNG seed, forwarded to the ``fit_alg_1`` stage only —
        the two-stage contract designates stage 2 as deterministic
        refinement (``fit_alg_2`` stays free-form; a stochastic second
        stage stays unseeded and is surfaced by the save-time collision
        rules). Makes a stochastic method (``differential_evolution``
        etc.) reproducible and, when supplied, enters fit identity: a
        seeded run is a distinct configuration from an unseeded one. No
        capability table is maintained — a seed on a method that cannot
        consume it surfaces as the library's own ``TypeError``, which is
        correct behavior (fit_archive_principles.md). Distinct from
        SbS's ``seed_source``/``seed_values`` knobs, which choose
        initial *parameter values*, not the RNG state.
    show_output : {0, 1}, default=0
        Output mode:

        - 0: Silent / programmatic / API mode -- no prints
        - 1: Interactive / notebook / UI mode -- print timing, fit
          results, confidence intervals, and the MCMC report. Builds no
          figures itself; callers that want the MCMC diagnostic plots
          call ``FitResults.plot_mcmc()`` (each of
          ``File.fit_baseline``/``fit_spectrum``/``fit_2d`` does so
          right after appending the fit slot).

    Returns
    -------
    ulmfit.FitOutput
        Typed result container with fields:

        - **par_ini** (*lmfit.Parameters*) -- Initial parameter guess.
        - **par_fin** (*lmfit.MinimizerResult*) -- Final fit result
          from lmfit.minimize.
        - **conf_ci** (*pd.DataFrame*) -- Confidence intervals from
          lmfit.conf_interval. Columns: ``['par[v]/sigma[>]', '-3σ',
          '-2σ', '-1σ', 'best', '+1σ', '+2σ', '+3σ']``.
          Empty DataFrame if CI not calculated/failed.
        - **emcee_fin** (*lmfit.MinimizerResult or None*) -- MCMC result
          from lmfit.emcee. None if MCMC not used.
        - **emcee_ci** (*pd.DataFrame*) -- MCMC confidence intervals
          from quantiles of flatchain. Same column structure as conf_ci;
          one row per sampled parameter (varying model params + the
          ``__lnsigma`` nuisance), fixed parameters excluded.
          Empty DataFrame if MCMC not used.

    Examples
    --------
    >>> # Basic single-stage fit
    >>> const = (energy, spectrum, spectra, 'fit_model_mcp', 0, [], [])
    >>> args = (model, 1, False)
    >>> results = fit_wrapper(
    ...     const=const,
    ...     args=args,
    ...     par_names=model.parameter_names,
    ...     par=model.lmfit_pars,
    ...     stages=1,
    ...     show_output=1
    ... )
    >>> results.par_fin.params  # optimized parameters

    >>> # Two-stage fit with confidence intervals
    >>> results = fit_wrapper(
    ...     const=const,
    ...     args=args,
    ...     par_names=model.parameter_names,
    ...     par=model.lmfit_pars,
    ...     stages=2,
    ...     try_ci=1,
    ...     ci_sigmas=[1, 2, 3],
    ...     show_output=1
    ... )

    >>> # Fit with MCMC for uncertainty quantification
    >>> mc = ulmfit.MC(use_mc=1, steps=5000, nwalkers=50)
    >>> results = fit_wrapper(
    ...     const=const,
    ...     args=args,
    ...     par_names=model.parameter_names,
    ...     par=model.lmfit_pars,
    ...     stages=2,
    ...     try_ci=1,
    ...     mc_settings=mc,
    ...     show_output=1
    ... )

    Notes
    -----
    **Two-Stage Fitting:**
    Recommended approach for robust results:
    1. Nelder-Mead finds global minimum (no gradient needed)
    2. Levenberg-Marquardt refines locally (fast, accurate errors)

    This combination avoids local minima while providing reliable error estimates.

    **Error Estimation Methods:**
    - Confidence intervals (CI): Profile likelihood method, exact but slow
    - MCMC: Samples parameter space, handles correlations, provides full distributions
    - Use MCMC for complex models or when CI fails

    **MCMC Diagnostics:**
    When using MCMC, check:
    - Acceptance ratios: Should be 0.2-0.5
    - Corner plot: Should show well-defined peaks
    - Chain length: Increase steps if distributions look noisy

    fit_wrapper itself builds no figures for these -- call
    FitResults.plot_mcmc() (or File.plot_mcmc()) to render them, from a
    live session or a loaded archive alike.

    **Performance Tips:**
    - Use stages=1 for quick fits during model development
    - Use stages=2 for final/publication fits
    - MCMC is slow (minutes for complex models) but provides best uncertainties
    """

    if ci_sigmas is None:
        ci_sigmas = [1.0, 2.0, 3.0]
    if mc_settings is None:
        mc_settings = ulmfit.MC()

    if stages not in (1, 2):
        raise ValueError(f"stages must be 1 or 2, got {stages}")

    # Fail fast on NaN/Inf inside the fit window: lmfit raises a generic
    # error that blames "input data or the objective/model function",
    # leaving the user to figure out which. Non-finite data outside the
    # e_lim/t_lim window never reaches the residual and stays legal.
    data_arr = np.asarray(const[1], dtype=float)
    window = data_arr[_fit_window_slices(data_arr.ndim, const[4], const[5])]
    n_bad = int(np.size(window) - np.count_nonzero(np.isfinite(window)))
    if n_bad > 0:
        raise ValueError(
            f"Data contains {n_bad} non-finite value(s) (NaN/Inf) inside "
            "the fit window. Clean the data or exclude the affected "
            "region with set_fit_limits() before fitting."
        )

    # construct the lmfit parameters if necessary
    if isinstance(par, lmfit.parameter.Parameters):
        par_ini = copy.deepcopy(par)
    else:
        par_ini = ulmfit.par_construct(par_names=par_names, par_info=par)

    if show_output >= 1:
        t_0 = time.time()  # start time

    # construct lmfit minimizer
    mini = lmfit.Minimizer(residual_fun, par_ini, fcn_args=(*const, "lmfit", args))

    # analytic Jacobian: only lmfit's leastsq accepts a Dfun. The optimizer
    # seed goes to the stage-1 method only — the two-stage contract
    # designates stage 2 as deterministic refinement (fit_alg_2 stays
    # free-form; a stochastic second stage stays unseeded).
    def _method_kws(method: str, *, stage_1: bool = True) -> dict[str, Any]:
        kws: dict[str, Any] = {}
        if jac_fun is not None and method == "leastsq":
            kws.update({"Dfun": jac_fun, "col_deriv": 0})
        if seed is not None and stage_1:
            kws["seed"] = int(seed)
        return kws

    # perform fit(s)
    if show_output >= 1:
        t_ini = time.time()
        print(f"\nTime initialize: {t_ini - t_0} s")
    #
    if stages == 1:  # one fit only
        par_fin = mini.minimize(method=fit_alg_1, **_method_kws(fit_alg_1))
        par_fin_params = _result_params(par_fin)
        if show_output >= 1:
            print(f"\nResults fit (method={fit_alg_1}): ")
            lmfit.report_fit(par_fin_params)
            t_fit = time.time()
            print(f"Time fit: {t_fit - t_ini} s")
    #
    if stages == 2:  # find global minimum + local optimization
        par_fin_gm = mini.minimize(method=fit_alg_1, **_method_kws(fit_alg_1))
        par_fin_gm_params = _result_params(par_fin_gm)
        if show_output >= 1:
            print(f"\nResults global minumum fit (method={fit_alg_1}): ")
            lmfit.report_fit(par_fin_gm_params)
            t_fit0 = time.time()
            print(f"Time fit (global minimum): {t_fit0 - t_ini} s")
        #
        par_fin = mini.minimize(
            method=fit_alg_2,
            params=par_fin_gm_params,
            **_method_kws(fit_alg_2, stage_1=False),
        )
        par_fin_params = _result_params(par_fin)
        # Stage 2 starts from stage 1's output, and lmfit's prepare_fit()
        # unconditionally resets init_value = value at the start of every
        # stage — restore the true pre-fit seed before it's printed below
        # or reaches any consumer of FitOutput.par_fin (report_fit's own
        # "(init = ...)" annotation would otherwise show stage 1's output,
        # contradicting the persisted archive).
        ulmfit.restore_true_init_values(par_fin_params, par_ini)
        if show_output >= 1:
            print(f"\nResults local optimization fit (method={fit_alg_2}): ")
            lmfit.report_fit(par_fin_params)
            t_fit = time.time()
            print(f"Time fit (local optimization): {t_fit - t_fit0} s")

    # confidence intervals

    # define column headers for the confidence interval dataframes
    # (conf_interval and emcee)
    ci_cols = (
        ["par[v]/sigma[>]"]
        + ["-" + str(sigma) for sigma in ci_sigmas[::-1]]
        + ["best fit"]
        + ["+" + str(sigma) for sigma in ci_sigmas]
    )

    run_mcmc = mc_settings.use_mc == 1
    mc_run: ulmfit.MC | None = None
    # conf_interval (https://lmfit.github.io/lmfit-py/confidence.html)
    if try_ci == 1:
        if _result_errorbars(par_fin):
            ci_fin, _trace_fin = lmfit.conf_interval(
                mini, par_fin, sigmas=ci_sigmas, trace=True
            )
            if show_output >= 1:
                print()
                lmfit.printfuncs.report_ci(ci_fin)
            # convert ci_fin to standard CI dataframe
            conf_ci = ulmfit.conf_interval_to_df(ci_fin, ci_cols)
        else:
            conf_ci = pd.DataFrame()
            if show_output >= 1:
                print("\nNo successful error bar determination via conf_interval")
            if mc_settings.use_mc == 2:
                # conf_interval didn't work -> fall back to lmfit.emcee()
                run_mcmc = True
    elif try_ci == 0:
        conf_ci = pd.DataFrame()

    # lmfit.emcee() [not a fit, it is a way to sample the parameter space!]
    if run_mcmc:
        t_emcee0 = time.time()
        # deepcopy first: __lnsigma is an MCMC sampling construct, not a model
        # parameter. _result_params returns the live par_fin.params (stored as
        # FitOutput.par_fin and consumed downstream as the model-only fit
        # result), so adding __lnsigma in place would leak it into every
        # consumer of that result (display, get_parameters, SbS tables).
        # emcee gets the copy.
        par_fin_params = copy.deepcopy(_result_params(par_fin))
        # Resolve the derivable MC knobs from the optimizer result into a
        # copy: sigma from the RMS residual (the MLE sigma of the unweighted
        # Gaussian model on the data view this fit saw), walkers from the
        # sampled dimension count. The caller's MC is left untouched.
        n_dim = int(par_fin.nvarys) + (0 if mc_settings.is_weighted else 1)
        sigma_fit = float(np.sqrt(par_fin.chisqr / par_fin.ndata))
        mc_run = mc_settings.resolve(sigma_fit=sigma_fit, n_dim=n_dim)
        if not mc_run.is_weighted:
            # __lnsigma only enters lmfit's log-probability for unweighted
            # sampling; adding it to a weighted run would sample a flat,
            # likelihood-free direction and report a meaningless posterior.
            assert mc_run.sigma_ini is not None  # type guard
            assert mc_run.sigma_min is not None  # type guard
            assert mc_run.sigma_max is not None  # type guard
            par_fin_params.add(
                "__lnsigma",
                value=np.log(mc_run.sigma_ini),
                min=np.log(mc_run.sigma_min),
                max=np.log(mc_run.sigma_max),
            )
        if show_output >= 1:
            print(
                "\nProgress of lmfit.emcee confidence interval determination\n"
                "(based on Markov chain Monte Carlo parameter space sampling):"
            )
        # burn necessary if starting point not close to max(probability distribution)
        # i.e. not close to the optimized parameter set, so burn=0 is ok here!
        emcee_kwargs: dict[str, Any] = {
            "params": par_fin_params,
            "steps": mc_run.steps,
            "nwalkers": mc_run.nwalkers,
            "burn": mc_run.burn,
            "thin": mc_run.thin,
            "ntemps": mc_run.ntemps,
            "is_weighted": mc_run.is_weighted,
            "seed": mc_run.seed,
            "progress": show_output >= 1,
        }
        if isinstance(mc_run.workers, int) and mc_run.workers > 1:
            # lmfit would build a default-context Pool, which fork()s on
            # Linux < 3.14 — deadlock-prone in multithreaded processes.
            # Supply a spawn-backed pool instead (lmfit hands any object
            # with .map to emcee), matching the slice-by-slice executor.
            # sanitized_spawn_main keeps the spawn workers from re-running a
            # non-.py __main__ (e.g. a notebook executed via %run), the same
            # guard the SbS executor uses.
            ctx = multiprocessing.get_context("spawn")
            with (
                uspawn.sanitized_spawn_main(),
                ctx.Pool(mc_run.workers) as pool,
            ):
                # lmfit annotates workers as int but accepts pool-likes
                emcee_fin = mini.emcee(workers=cast("int", pool), **emcee_kwargs)
        else:
            emcee_fin = mini.emcee(workers=mc_run.workers, **emcee_kwargs)
        emcee_fin_params = _result_params(emcee_fin)
        emcee_flatchain = cast(
            "pd.DataFrame", getattr(emcee_fin, "flatchain", pd.DataFrame())
        )
        emcee_var_names = cast("list[str]", getattr(emcee_fin, "var_names", []))
        # lmfit.emcee() results
        if show_output >= 1:
            print("\nResults lmfit.emcee() confidence interval determination:")
            lmfit.report_fit(emcee_fin_params)
            t_emcee1 = time.time()
            print(f"Time lmfit.emcee: {t_emcee1 - t_emcee0} s")
            # Diagnostics figures (walker acceptance, corner plot) are no
            # longer built here: fit_baseline/fit_spectrum/fit_2d each call
            # self.plot_mcmc(...) after the slot is appended, the same single
            # code path FitResults.plot_mcmc uses to re-render from a loaded
            # archive -- one implementation instead of two that could drift.
        # get percentage borders to categorize emcee.flatchain data
        sigma_borders = sigma_start_stop_percent(ci_sigmas)
        # one row per sampled parameter (varying model params + the __lnsigma
        # noise-scale nuisance). Fixed parameters have no posterior, so they
        # get no row — mirroring lmfit.conf_interval, which only profiles
        # varying parameters. (Emitting a placeholder row here would surface
        # as real-looking quantiles through get_mcmc().table and the saved
        # archive's mcmc.ci.)
        sampled_names = [
            par_name
            for par_name in [*par_names, "__lnsigma"]
            if par_name in emcee_var_names
        ]
        emcee_ci_list = []  # initialize results
        for par_name in sampled_names:
            emcee_par_ci: list[Any] = [par_name]  # initialize results for parameter
            for sigma_b in sigma_borders:
                # get cutoff values that meet this sigma threshold (+/-)
                quantiles = np.percentile(emcee_flatchain[par_name], sigma_b)
                # lower threshold (0 is par_name)
                emcee_par_ci.insert(1, quantiles[0])
                # upper threshold
                emcee_par_ci.insert(len(emcee_par_ci), quantiles[1])
            # append this line to list containing all parameters
            emcee_ci_list.append(emcee_par_ci)
        # convert confidence interval cutoffs to a dataframe and add the
        # "best fit result" in the middle (aligned to sampled_names order)
        emcee_ci = pd.DataFrame(data=emcee_ci_list)
        best_fit = emcee_fin_params.valuesdict()
        emcee_ci.insert(
            loc=len(ci_sigmas) + 1,
            column="bla",
            value=[best_fit[par_name] for par_name in sampled_names],
        )
        emcee_ci.columns = ci_cols
        if show_output >= 1:
            print(display(emcee_ci))
    else:  # use_mc 0, or 2 with a working conf_interval
        emcee_fin = None
        emcee_ci = pd.DataFrame()

    # cast: lmfit sets result attributes dynamically, so its returns are
    # opaque to type checkers — TypedMinimizerResult declares what we read
    return ulmfit.FitOutput(
        par_ini=par_ini,
        par_fin=cast("ulmfit.TypedMinimizerResult", par_fin),
        conf_ci=conf_ci,
        emcee_fin=cast("ulmfit.TypedMinimizerResult | None", emcee_fin),
        emcee_ci=emcee_ci,
        mc_settings=mc_run,
    )


#
# plotting fit results for Slice-by-Slice methods
#


#
def results_to_df(
    results: list[ulmfit.FitOutput],
    x: ArrayLike | None = None,
    index: ArrayLike | None = None,
    config: PlotConfig | None = None,
) -> pd.DataFrame:
    """
    Convert Slice-by-Slice fit results to a DataFrame.

    Pure conversion: transforms a list of fit results (from slice-by-slice
    fitting) into a pandas DataFrame with time/index as rows and parameters
    as columns. Saving and plotting are the caller's responsibility
    (``df.to_csv`` / ``utils.plot.plot_par_series``).

    Parameters
    ----------
    results : list of ulmfit.FitOutput
        Fit results from fit_wrapper, one per time slice.
    x : array-like, optional
        Time axis values. If provided, included as column in DataFrame.
    index : array-like, optional
        Index values (e.g., slice numbers). If provided, included as column.
    config : PlotConfig, optional
        Supplies the label of the ``x`` column (``config.y_label``).
        If None, uses defaults.

    Returns
    -------
    pd.DataFrame
        Results with structure:
        - Columns: ['index', config.y_label, param1, param2, ...]
        - Rows: One per fitted slice
        - Values: Optimized parameter values
    """

    # Use default config if none provided
    if config is None:
        config = PlotConfig()

    # transform lmfit_wrapper results to dataframe
    df = ulmfit.list_of_par_to_df(results)

    # insert x (time) and index data if passed
    if x is not None:
        df.insert(0, config.y_label, x)
    if index is not None:
        df.insert(0, "index", index)

    return df


#
def results_to_fit_2d(
    results: list[ulmfit.FitOutput] | pd.DataFrame,
    const: tuple[Any, ...],
    args: tuple[Any, ...],
    parameter_names: list[str] | None = None,
) -> np.ndarray:
    """
    Reconstruct 2D fit spectrum from Slice-by-Slice fit results.

    Pure reconstruction: takes individual 1D fit results (one per time
    slice) and stacks them into a complete 2D fit array. This allows
    visualization and comparison with the measured 2D data for
    Slice-by-Slice fitting. Saving is the caller's responsibility
    (``np.savetxt``); note that completed SbS fits already persist this
    array as ``SavedFitSlot.fit``.

    Parameters
    ----------
    results : list of ulmfit.FitOutput or pd.DataFrame
        Fit results, either:

        - list: ``fit_wrapper`` output for each slice.
        - pd.DataFrame: From results_to_df() with parameters as columns

    parameter_names : list of str, optional
        For DataFrame results: select and order these columns as the
        parameter vector before evaluation. Pass when the DataFrame may
        carry extra non-parameter columns (e.g. the index/time columns in
        ``results_to_df`` output); extra columns are otherwise passed to
        the fit function as parameters. If None, all columns are used in
        DataFrame order. Ignored for list results.

    const : tuple
        Constants for residual_fun:
        (x, data, function_str, unpack, e_lim, t_lim)
        Used to evaluate fit function at each time point.
    args : tuple
        Arguments for fit function (model, dim).
        Passed to residual_fun for spectrum generation.

    Returns
    -------
    ndarray
        2D fit array (shape: [n_time, n_energy])
        Each row is the fitted spectrum for one time slice
    """

    (
        x_const,
        data_const,
        fit_fun_const,
        unpack_const,
        e_lim_const,
        t_lim_const,
    ) = const

    # Select/order parameter columns; raises KeyError on missing names
    if isinstance(results, pd.DataFrame) and parameter_names is not None:
        results = results.loc[:, parameter_names]

    lst = []  # intialize
    for i in range(len(results)):
        # list of lmfit_wrapper fit results
        if isinstance(results, list):
            lst.append(
                residual_fun(
                    results[i].par_fin.params,
                    x_const,
                    np.asarray(data_const),
                    fit_fun_const,
                    unpack=cast("int", unpack_const),
                    e_lim=cast("list[int]", e_lim_const),
                    t_lim=cast("list[int]", t_lim_const),
                    res_type="fit",
                    args=args,
                )
            )
        # pandas dataframe containing parameters as columns
        elif isinstance(results, pd.DataFrame):
            lst.append(
                residual_fun(
                    results.iloc[i].values,
                    x_const,
                    np.asarray(data_const),
                    fit_fun_const,
                    unpack=cast("int", unpack_const),
                    e_lim=cast("list[int]", e_lim_const),
                    t_lim=cast("list[int]", t_lim_const),
                    res_type="fit",
                    args=args,
                )
            )
    return np.asarray(lst)


#
# Model-curve evaluation for display
#


#
def eval_model_curves_1d(
    x: ArrayLike,
    fit_fun_str: str,
    par: Any,
    args: tuple[Any, ...] = (),
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Evaluate a 1D model's total and per-component curves for display.

    Thin dispatch onto ``trspecfit.spectra`` (``fit_fun_str`` names the
    evaluator, e.g. ``'fit_model_mcp'`` / ``'fit_model_gir'``): parameter
    values are extracted from ``par`` (lmfit ``Parameters`` or
    ``MinimizerResult``) and the model is evaluated once summed and once
    per component. Rendering is ``utils.plot``'s job — this helper keeps
    model evaluation out of the renderers.
    """

    fit_fun = getattr(spectra, fit_fun_str)
    x_arr = np.asarray(x, dtype=float)
    par_vals = ulmfit.par_extract(par, return_type="list")
    total = fit_fun(x_arr, par_vals, True, *args)
    components = fit_fun(x_arr, par_vals, False, *args)
    return np.asarray(total), [np.asarray(c) for c in components]
