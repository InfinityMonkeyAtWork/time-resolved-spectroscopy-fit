"""
Helper functions for lmfit parameter handling and result management.

This module provides utilities for:
- Creating and constructing lmfit.Parameter objects
- Extracting parameter values from various lmfit objects
- Converting lmfit results to pandas DataFrames for analysis
- Managing MCMC sampling configuration
- Compatibility with scipy.optimize workflows
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, overload

import lmfit
import numpy as np
import pandas as pd
from lmfit.minimizer import MinimizerResult

if TYPE_CHECKING:
    #
    #
    class TypedMinimizerResult(MinimizerResult):
        """Annotation-only view of ``lmfit.minimizer.MinimizerResult``.

        lmfit sets result attributes dynamically (``setattr`` in
        ``__init__`` / ``minimize``), so type checkers see none of them.
        This subclass declares the ones trspecfit reads; at runtime it IS
        ``MinimizerResult`` (see the ``else`` branch).
        """

        params: lmfit.Parameters
        method: str
        success: bool
        errorbars: bool
        nvarys: int
        ndata: int
        nfree: int
        chisqr: float
        redchi: float
        covar: np.ndarray | None
        var_names: list[str]
        # set by Minimizer.emcee only
        flatchain: pd.DataFrame
        acceptance_fraction: np.ndarray
else:
    TypedMinimizerResult = MinimizerResult

#
# lmfit parameter creation and extraction
#


# Valid string values for the vary field in YAML parameter specs
VARY_LEVELS = {"project", "file", "static"}


#
def _vary_to_bool(vary: bool | str) -> bool:
    """Convert vary specification to bool for lmfit.

    Maps ``"project"`` and ``"file"`` to ``True`` (optimized),
    ``"static"`` to ``False`` (fixed). Bool values pass through.
    """

    if isinstance(vary, bool):
        return vary
    if vary in ("project", "file"):
        return True
    if vary == "static":
        return False
    raise ValueError(
        f"Invalid vary value: {vary!r}. "
        f"Must be True, False, 'project', 'file', or 'static'."
    )


#
def vary_to_level(vary: bool | str) -> str:
    """Convert vary specification to a canonical level string.

    Maps ``True`` → ``"file"``, ``False`` → ``"static"``.
    String values ``"project"``/``"file"``/``"static"`` pass through.
    """

    if vary is True:
        return "file"
    if vary is False:
        return "static"
    if isinstance(vary, str) and vary in VARY_LEVELS:
        return vary
    raise ValueError(
        f"Invalid vary value: {vary!r}. "
        f"Must be True, False, 'project', 'file', or 'static'."
    )


#
def par_create(
    par_name: str,
    par_info: list[Any],
    prefix: str = "",
    suffix: str = "",
) -> lmfit.Parameter:
    """
    Create lmfit.Parameter object with optional name modifiers.

    Convenience wrapper for creating lmfit parameters that handles both
    standard parameters (value, vary, min, max) and expression-based
    parameters with automatic name prefix/suffix support.

    Parameters
    ----------
    par_name : str
        Base parameter name
    par_info : list
        Parameter specification, either:
        - [value, vary, min, max] for standard parameter
        - [value, vary] for unbound fit parameter
        - [expr_string] for expression-based parameter
    prefix : str, default=''
        String to prepend to parameter name
    suffix : str, default=''
        String to append to parameter name

    Returns
    -------
    lmfit.Parameter
        Configured parameter object
    """

    # Assemble parameter name
    par_str = prefix + par_name + suffix

    # Create lmfit.Parameter object
    lmf_par = lmfit.Parameter(par_str)

    # Standard parameter: [value, vary, min, max]
    if len(par_info) == 4:
        value, vary, pmin, pmax = par_info
        vary = _vary_to_bool(vary)
        lmf_par.set(value, vary, pmin, pmax)
    # Unbound fit parameter: [value, vary]
    elif len(par_info) == 2:
        vary = _vary_to_bool(par_info[1])
        lmf_par.set(par_info[0], vary, -np.inf, np.inf)
    # Expression parameter: [expr_string]
    elif len(par_info) == 1:
        try:
            lmf_par.set(expr=par_info[0])
        except Exception as e:  # noqa: BLE001
            warnings.warn(
                f"Exception while adding expression {par_info[0]} "
                f"to parameter {par_str}: {e}",
                stacklevel=2,
            )

    return lmf_par


# Type alias for input types accepted by par_extract
type _ParExtractInput = (
    lmfit.Parameters | MinimizerResult | list[float] | dict[str, list[Any]] | np.ndarray
)


@overload
def par_extract(
    lmfit_pars: _ParExtractInput,
    return_type: Literal["list"] = ...,
) -> list[float]: ...
@overload
def par_extract(
    lmfit_pars: _ParExtractInput,
    return_type: Literal["par.x"],
) -> par_dummy: ...
def par_extract(
    lmfit_pars: _ParExtractInput, return_type: Literal["list", "par.x"] = "list"
) -> list[float] | par_dummy:
    """
    Extract parameter values from lmfit objects.

    Converts various lmfit parameter representations into a simple list
    of values or scipy-compatible format. Handles Parameters objects,
    MinimizerResult objects, lists, dicts, and numpy arrays.

    Parameters
    ----------
    lmfit_pars : lmfit.Parameters, lmfit.MinimizerResult, list, dict, or ndarray
        Parameter source to extract from:
        - lmfit.Parameters: Extract current values
        - lmfit.MinimizerResult: Extract optimized values
        - list: Pass through directly (list of values)
        - dict: Extract first element of each value
          (format: {name: [val, vary, min, max]})
        - ndarray: Convert to list
    return_type : {'list', 'par.x'}, default='list'
        Output format:
        - 'list': Return Python list of values
        - 'par.x': Return par_dummy object with .x attribute (scipy compatible)

    Returns
    -------
    list or par_dummy
        Parameter values in requested format

    Examples
    --------
    >>> # From lmfit.Parameters
    >>> params = lmfit.Parameters()
    >>> params.add('a', value=1.5)
    >>> params.add('b', value=2.0)
    >>> par_extract(params)
    [1.5, 2.0]

    >>> # From fit result
    >>> result = minimize(residual, params, ...)
    >>> par_extract(result)
    [1.523, 1.987]

    >>> # From list (passthrough)
    >>> par_extract([1.5, 2.0, 3.0])
    [1.5, 2.0, 3.0]

    >>> # From dict (initial guess format)
    >>> par_dict = {'a': [1.5, True, 0, 5], 'b': [2.0, True, 0, 10]}
    >>> par_extract(par_dict)
    [1.5, 2.0]

    >>> # scipy-compatible format
    >>> par_obj = par_extract(params, return_type='par.x')
    >>> par_obj.x
    [1.5, 2.0]
    """

    # lmfit.Parameters object
    if isinstance(lmfit_pars, lmfit.parameter.Parameters):
        pars_dict = lmfit_pars.valuesdict()
        pars = [v for k, v in pars_dict.items()]

    # List of values (passthrough)
    elif isinstance(lmfit_pars, list):
        pars = lmfit_pars

    # Initial guess dictionary: {name: [value, vary, min, max]}
    elif isinstance(lmfit_pars, dict):
        pars = [v[0] for k, v in lmfit_pars.items()]

    # Numpy array
    elif isinstance(lmfit_pars, np.ndarray):
        pars = lmfit_pars.tolist()

    # lmfit.MinimizerResult object
    elif isinstance(lmfit_pars, MinimizerResult):
        result_params = getattr(lmfit_pars, "params", None)
        if not isinstance(result_params, lmfit.parameter.Parameters):
            raise TypeError(
                "par_extract: MinimizerResult.params is missing or has unexpected type."
            )
        pars_dict = result_params.valuesdict()
        pars = [v for _, v in pars_dict.items()]

    else:
        raise TypeError(
            f"par_extract: unsupported type {type(lmfit_pars).__name__}. "
            f"Expected Parameters, MinimizerResult, list, dict, or ndarray."
        )

    # Return in requested format
    if return_type == "list":
        return pars
    if return_type == "par.x":
        pars_scipy = par_dummy()
        pars_scipy.x = pars
        return pars_scipy
    raise ValueError(f"return_type must be 'list' or 'par.x', got '{return_type}'")


#
def par_construct(par_names: list[str], par_info: list[list[Any]]) -> lmfit.Parameters:
    """
    Construct lmfit.Parameters object from lists.

    Batch version of par_create that builds a complete Parameters object
    from parallel lists of names and parameter specifications.

    Parameters
    ----------
    par_names : list of str
        Parameter names
    par_info : list of list
        Parameter specifications, one per name. Each element is either:
        - [value, vary, min, max] for standard parameter
        - [value, vary] for unbound fit parameter
        - [expr_string] for expression-based parameter

    Returns
    -------
    lmfit.Parameters
        Complete Parameters object with all parameters added
    """

    # Initialize Parameters object
    lmf_pars = lmfit.Parameters()

    # Add parameters one by one
    for par_name, p_info in zip(par_names, par_info, strict=True):
        if len(p_info) == 4:  # [value, vary, min, max]
            value, vary, pmin, pmax = p_info
            lmf_pars.add(par_name, value, _vary_to_bool(vary), pmin, pmax)
        elif len(p_info) == 2:  # [value, vary]
            lmf_pars.add(par_name, p_info[0], _vary_to_bool(p_info[1]), -np.inf, np.inf)
        elif len(p_info) == 1:  # [expr]
            lmf_pars.add(par_name, expr=p_info[0])

    return lmf_pars


#
# Result conversion to pandas DataFrames
#


#
def conf_interval_to_df(ci: dict[str, Any], ci_cols: list[str]) -> pd.DataFrame:
    """
    Convert lmfit confidence interval results to pandas DataFrame.

    Transforms the nested dictionary structure returned by lmfit.conf_interval
    into a tabular DataFrame format suitable for display and saving.
    Each row contains parameter name followed by values at different sigma levels.
    The confidence interval values represent parameter bounds at each sigma level.

    Parameters
    ----------
    ci : dict
        Confidence interval results from lmfit.conf_interval.
        Structure: {param_name: [(sigma, value), ...]}
    ci_cols : list of str
        Column headers for the output DataFrame.
        Typically: ['par[v]/sigma[>]', '-3', '-2', '-1', 'best fit', '+1', '+2', '+3']

    Returns
    -------
    pd.DataFrame
        DataFrame with rows=parameters, columns=sigma levels

    Examples
    --------
    >>> # After running confidence interval calculation
    >>> ci, trace = lmfit.conf_interval(minimizer, result, sigmas=[1,2,3], trace=True)
    >>> ci_cols = ['parameter', '-3', '-2', '-1', 'best', '+1', '+2', '+3']
    >>> df = conf_interval_to_df(ci, ci_cols)
    >>> df.to_csv('confidence_intervals.csv', index=False)
    """

    conf_ci_list = []

    for param_name, values in ci.items():
        conf_par_ci = [param_name]  # Start with parameter name

        # Extract parameter values at each sigma level
        # values is list of (sigma_percentage, param_value) tuples
        conf_par_ci.extend(val[1] for val in values)  # val[1] is the parameter value

        conf_ci_list.append(conf_par_ci)

    return pd.DataFrame(data=conf_ci_list, columns=ci_cols)


#
def par_to_df(
    lmfit_params: lmfit.Parameters,
    col_type: Literal["ini", "min"] | list[str],
    par_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Convert lmfit.Parameters object to pandas DataFrame.

    Extracts parameter information into tabular format for easy display,
    analysis, and saving. Supports different column sets for initial
    guesses vs. fit results.

    Parameters
    ----------
    lmfit_params : lmfit.Parameters
        Parameters object to convert. For fit results, pass ``result.params``.
    col_type : {'ini', 'min'} or list of str
        Column selection:
        - 'ini': Initial guess columns ['name', 'value', 'vary', 'min', 'max', 'expr']
        - 'min': Fit result columns ['name', 'value', 'stderr', 'init_value',
          'min', 'max', 'vary', 'expr']
        - list: Custom list of attribute names to extract
    par_names : list of str, optional
        Subset of parameter names to include. If None, includes all parameters.

    Returns
    -------
    pd.DataFrame
        DataFrame with rows=parameters, columns=attributes

    Examples
    --------
    >>> # Initial guess parameters
    >>> params = lmfit.Parameters()
    >>> params.add('amplitude', value=10, vary=True, min=0, max=100)
    >>> df = par_to_df(params, col_type='ini')
    >>> df.to_csv('initial_parameters.csv', index=False)

    >>> # Fit results
    >>> result = minimize(residual, params, ...)
    >>> df = par_to_df(result.params, col_type='min')
    >>> print(df[['name', 'value', 'stderr']])

    >>> # Custom columns
    >>> df = par_to_df(params, col_type=['name', 'value', 'vary'])

    >>> # Subset of parameters
    >>> df = par_to_df(params, col_type='ini', par_names=['amplitude', 'center'])

    Notes
    -----
    Relative error (value/stderr*100) not included but easily computed from output.
    """

    # Select all parameters if none specified
    if par_names is None:
        par_names = list(lmfit_params.keys())

    # Define columns based on type
    if col_type == "ini":
        cols = ["name", "value", "vary", "min", "max", "expr"]
    elif col_type == "min":
        cols = ["name", "value", "stderr", "init_value", "min", "max", "vary", "expr"]
    else:
        cols = col_type  # list[str] custom columns

    # Extract parameter attributes
    par_info_list = []
    for par_name in par_names:
        par_info = [getattr(lmfit_params.get(par_name), col) for col in cols]
        par_info_list.append(par_info)

    return pd.DataFrame(data=par_info_list, columns=cols)


#
def restore_true_init_values(
    result_params: lmfit.Parameters, par_ini: lmfit.Parameters
) -> None:
    """
    Correct ``result_params``' ``init_value`` in place to the true
    pre-fit seed (``par_ini``).

    Two-stage fitting (``fitlib.fit_wrapper`` stages=2) starts its second
    stage from stage-1's output, and lmfit's ``prepare_fit`` unconditionally
    resets ``Parameter.init_value = Parameter.value`` at the start of every
    stage — so a two-stage result's ``init_value`` reflects stage-1's
    output, not the true original seed, unless corrected here.

    Parameters
    ----------
    result_params : lmfit.Parameters
        A completed fit's ``result.params`` (mutated in place).
    par_ini : lmfit.Parameters
        The true pre-fit seed (``FitOutput.par_ini``).
    """

    for name, par in result_params.items():
        if name in par_ini:
            par.init_value = par_ini[name].value


#
def correl_to_df(lmfit_params: lmfit.Parameters) -> pd.DataFrame:
    """
    Build the varying-parameter correlation matrix from lmfit Parameters.

    Parameters
    ----------
    lmfit_params : lmfit.Parameters
        Parameters from a completed fit (pass ``result.params``).

    Returns
    -------
    pd.DataFrame
        Square matrix indexed by the varying parameter names: 1.0 on the
        diagonal, lmfit's pairwise correlations off-diagonal (0.0 where a
        pair is uncorrelated or the optimizer reported no covariance).
    """

    names = [n for n in lmfit_params if lmfit_params[n].vary]
    mat = pd.DataFrame(np.eye(len(names)), index=names, columns=names, dtype=float)
    for n in names:
        for other, corr in (lmfit_params[n].correl or {}).items():
            if other in mat.columns:
                mat.loc[n, other] = corr
    return mat


#
def correl_from_result(result: Any) -> pd.DataFrame | None:
    """
    Correlation matrix of a completed fit, or ``None`` without covariance.

    Wraps ``correl_to_df`` behind the covariance guard shared by every slot
    capture path: when the optimizer produced no covariance matrix (e.g.
    Nelder without numdifftools, or the minimal per-file result of a
    project-level joint fit), returns ``None`` rather than an identity
    matrix that would misreport "no covariance" as "uncorrelated".

    Parameters
    ----------
    result : lmfit.minimizer.MinimizerResult
        A completed fit result (``FitOutput.par_fin``).

    Returns
    -------
    pd.DataFrame or None
        See ``correl_to_df``; ``None`` when ``result.covar`` is absent.
    """

    if getattr(result, "covar", None) is None:
        return None
    return correl_to_df(result.params)


#
def list_of_par_to_df(results: list[FitOutput]) -> pd.DataFrame:
    """
    Extract parameter values from multiple fit results into DataFrame.

    Collects optimized parameter values from a list of fit results
    (e.g., from slice-by-slice fitting) and organizes them in a DataFrame
    with rows=fits and columns=parameters.
    Assumes all fits have the same parameter names (typical for slice-by-slice).
    Parameter names are extracted from the first result.

    Parameters
    ----------
    results : list of FitOutput
        Fit results from ``fitlib.fit_wrapper``, one per fit; each
        ``par_fin`` holds the lmfit.MinimizerResult with a ``.params``
        attribute.

    Returns
    -------
    pd.DataFrame
        DataFrame with rows=individual fits, columns=parameter values

    Examples
    --------
    >>> # After slice-by-slice fitting
    >>> results_list = []
    >>> for spectrum in data_2d:
    ...     result = fit_wrapper(spectrum, ...)
    ...     results_list.append(result)
    >>>
    >>> df = list_of_par_to_df(results_list)
    >>> df.columns
    Index(['amplitude', 'center', 'width', ...], dtype='object')
    >>>
    >>> # Plot parameter evolution
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(df['center'])
    >>> plt.xlabel('Slice number')
    >>> plt.ylabel('Peak center')
    """

    # Extract parameter values from each result
    param_values_list = [par_extract(result.par_fin.params) for result in results]

    # Get parameter names from first result (all should be identical)
    param_names = list(results[0].par_fin.params.valuesdict())

    return pd.DataFrame(param_values_list, columns=param_names)


#
def list_of_par_stderr_to_df(results: list[FitOutput]) -> pd.DataFrame:
    """
    Extract per-fit parameter stderr into a DataFrame (NaN where absent).

    Companion to :func:`list_of_par_to_df` with the same shape contract
    (rows=fits, columns=parameters): collects each fit's per-parameter
    standard errors instead of the optimized values. lmfit reports
    ``stderr=None`` when the optimizer produced no covariance; those cells
    become ``NaN`` so the frame stays numeric.

    Parameters
    ----------
    results : list of FitOutput
        Fit results from ``fitlib.fit_wrapper``; each ``par_fin`` holds
        the lmfit.MinimizerResult with a ``.params`` attribute.

    Returns
    -------
    pd.DataFrame
        DataFrame with rows=individual fits, columns=parameter stderr.
    """

    param_names = list(results[0].par_fin.params.keys())
    rows = []
    for result in results:
        params = result.par_fin.params
        rows.append(
            [
                float(params[name].stderr)
                if params[name].stderr is not None
                else np.nan
                for name in param_names
            ]
        )
    return pd.DataFrame(rows, columns=param_names)


#
def list_of_par_ini_to_df(results: list[FitOutput]) -> pd.DataFrame:
    """
    Extract per-fit true initial-guess values into a DataFrame.

    Companion to :func:`list_of_par_stderr_to_df` with the same shape
    contract (rows=fits, columns=parameters): collects each fit's true
    pre-fit seed (``FitOutput.par_ini``) rather than the optimized value.
    Reads the seed directly, so it is unaffected by
    ``fitlib.fit_wrapper``'s two-stage ``init_value`` correction.

    Parameters
    ----------
    results : list of FitOutput
        Fit results from ``fitlib.fit_wrapper``; each ``par_ini`` holds
        the pre-fit ``lmfit.Parameters`` seed (never ``None`` for
        per-slice SbS results, the only caller of this function).

    Returns
    -------
    pd.DataFrame
        DataFrame with rows=individual fits, columns=parameter init values.
    """

    param_names = list(results[0].par_fin.params.keys())
    rows = []
    for result in results:
        par_ini = result.par_ini
        assert par_ini is not None  # type guard
        rows.append([par_ini[name].value for name in param_names])
    return pd.DataFrame(rows, columns=param_names)


#
# Configuration and compatibility classes
#


#
#
@dataclass(frozen=True)
class FitOutput:
    """
    Typed result of one ``fitlib.fit_wrapper`` optimization run.

    Internal container replacing the historical raw five-element list
    ``[par_ini, par_fin, conf_ci, emcee_fin, emcee_ci]``. Stored on
    ``mcp.Model.result`` and, per slice, in ``File.results_sbs``; the
    authoritative persisted record remains ``SavedFitSlot``.

    Attributes
    ----------
    par_ini : lmfit.Parameters or None
        Initial parameter guess (deep copy, untouched by the fit). None
        on the project-level joint-fit path, where per-file results are
        projections of one joint optimization and no per-file initial
        guess exists.
    par_fin : lmfit.minimizer.MinimizerResult
        Final fit result from ``lmfit.minimize`` (annotated as
        ``TypedMinimizerResult`` for static attribute access). On the
        project-level joint-fit path this is a minimal
        ``MinimizerResult`` carrying only ``params`` / ``method`` /
        ``nvarys``.
    conf_ci : pd.DataFrame
        Confidence intervals from ``lmfit.conf_interval`` (columns
        ``['par[v]/sigma[>]', '-3.0', ..., 'best fit', ..., '+3.0']``).
        Empty if CI was skipped or failed.
    emcee_fin : lmfit.minimizer.MinimizerResult or None
        MCMC sampling result from ``lmfit.emcee``. None if MCMC not used.
    emcee_ci : pd.DataFrame
        MCMC confidence intervals (quantiles of the flatchain, same
        column structure as ``conf_ci``). Empty if MCMC not used.
    mc_settings : MC or None
        The MCMC settings as they ran — the caller's ``MC`` with every
        derivable knob resolved from the optimizer result (see
        ``MC.resolve``). None if MCMC not used.
    """

    par_ini: lmfit.Parameters | None
    par_fin: TypedMinimizerResult
    conf_ci: pd.DataFrame
    emcee_fin: TypedMinimizerResult | None
    emcee_ci: pd.DataFrame
    mc_settings: MC | None = None


#
def _validate_sigma(
    sigma_ini: float | None,
    sigma_min: float | None,
    sigma_max: float | None,
    *,
    derived: bool = False,
) -> None:
    """
    Check the noise-scale knobs that are set; ``None`` means "derive later".

    ``derived`` marks ``sigma_ini`` as the fit-derived start, so the error
    names the value and the way out instead of blaming a kwarg the caller
    never passed.
    """

    for name, value in (
        ("sigma_ini", sigma_ini),
        ("sigma_min", sigma_min),
        ("sigma_max", sigma_max),
    ):
        if value is not None and not (np.isfinite(value) and value > 0):
            raise ValueError(f"{name} must be a positive finite number, got {value}")
    if sigma_min is not None and sigma_max is not None and not sigma_min < sigma_max:
        raise ValueError("sigma_min must be < sigma_max")
    if sigma_ini is None:
        return
    below = sigma_min is not None and sigma_ini < sigma_min
    above = sigma_max is not None and sigma_ini > sigma_max
    if below or above:
        if derived:
            raise ValueError(
                f"sigma_ini derived from the fit ({sigma_ini:.4g}, the RMS "
                f"residual) lies outside the explicit bounds "
                f"[sigma_min, sigma_max] = [{sigma_min}, {sigma_max}]; pass "
                "sigma_ini explicitly or widen/drop the bounds"
            )
        raise ValueError("sigma_ini must lie within [sigma_min, sigma_max]")


#
#
class MC:
    """
    Configuration for lmfit.emcee MCMC sampling.

    Holds the settings ``fitlib.fit_wrapper`` passes to
    ``lmfit.Minimizer.emcee``. Knobs left at ``None`` are derived from the
    optimizer result at fit time by :meth:`resolve`, which returns a fully
    explicit copy: that copy is what ran, is returned as
    ``FitOutput.mc_settings`` and is recorded in the fit slot's provenance.
    A fit never mutates the object it was given.

    Parameters
    ----------
    use_mc : int, default=0
        MCMC usage flag:
        - 0: Don't use MCMC
        - 1: Always use MCMC
        - 2: Use MCMC if conf_interval fails
    steps : int, default=5000
        Number of MCMC steps per walker.
    nwalkers : int or None, default=None
        Number of MCMC walkers. None derives ``2 * n_dim`` with a floor of
        20, where ``n_dim`` counts the varying parameters plus the
        ``__lnsigma`` nuisance for unweighted sampling. An explicit value
        below ``2 * n_dim`` raises at fit time: emcee's stretch move needs at
        least that many.
    burn : int, default=0
        Leading steps discarded from every walker. The chain starts at the
        optimizer's solution, so 0 is adequate; raise it when the walkers
        start far from the optimum.
    thin : int, default=1
        Keep every ``thin``-th step. 1 keeps the whole chain; larger values
        reduce the autocorrelation between retained samples.
    ntemps : int, default=1
        Number of temperatures for parallel tempering.
    workers : int, default=1
        Number of parallel workers (1 = serial). Workers > 1 run the
        sampling in a spawn-backed process pool (safe in multithreaded
        processes, unlike fork), costing ~1-2 s of pool startup per fit.
    is_weighted : bool, default=False
        Whether the residual is already in units of sigma. False samples
        the noise scale as the ``__lnsigma`` nuisance parameter; True trusts
        the residual's scale, has no nuisance, and ignores the sigma knobs.
    sigma_ini : float or None, default=None
        Start of the sampled noise scale (data units). None derives the
        fit's RMS residual, ``sqrt(chisqr / ndata)`` of the optimizer result:
        the maximum-likelihood sigma of the unweighted Gaussian model on the
        data view the fit saw. Started far off the true scale, the walkers
        spend the chain hunting for it and the posterior widths are
        unreliable.
    sigma_min, sigma_max : float or None, default=None
        Bounds of the sampled noise scale (data units). None derives two
        decades either side of the start. A derived start outside explicit
        bounds raises at fit time.
    seed : int or None, default=None
        Seed for the sampler's random state (initial walker spread and
        proposals), forwarded to ``lmfit.Minimizer.emcee(seed=)``. Makes a
        chain reproducible; the optimizer result does not depend on it
        (``fit_wrapper(seed=)`` seeds the stage-1 optimizer instead).

    Attributes
    ----------
    Same names as the parameters. On the copy returned by :meth:`resolve`
    every derivable knob is explicit, except that the ``sigma_*`` knobs stay
    None when ``is_weighted`` is True.

    Examples
    --------
    >>> # everything derived from the fit
    >>> result = fit_wrapper(..., mc_settings=MC(use_mc=1, steps=10000))
    >>> result.mc_settings.nwalkers  # what ran

    >>> # reproducible chain, explicit walkers
    >>> mc_config = MC(use_mc=1, steps=5000, nwalkers=50, seed=7)

    Notes
    -----
    - workers > 1 enables parallel sampling in a spawn-backed process pool
      (not supported on the JAX evaluator path)

    See Also
    --------
    lmfit.emcee : lmfit's MCMC wrapper
    emcee : Underlying MCMC library
    """

    #
    def __init__(
        self,
        *,
        use_mc: int = 0,
        steps: int = 5000,
        nwalkers: int | None = None,
        burn: int = 0,
        thin: int = 1,
        ntemps: int = 1,
        workers: int = 1,
        is_weighted: bool = False,
        sigma_ini: float | None = None,
        sigma_min: float | None = None,
        sigma_max: float | None = None,
        seed: int | None = None,
    ) -> None:
        if use_mc not in (0, 1, 2):
            raise ValueError(
                f"use_mc must be 0 (off), 1 (always) or 2 (if CI fails), got {use_mc}"
            )
        if nwalkers is not None and nwalkers < 2:
            raise ValueError(f"nwalkers must be >= 2 or None (derived), got {nwalkers}")
        if seed is not None and (isinstance(seed, bool) or seed < 0):
            raise ValueError(f"seed must be a non-negative int or None, got {seed!r}")
        _validate_sigma(sigma_ini, sigma_min, sigma_max)
        self.use_mc = use_mc
        self.steps = steps
        self.nwalkers = nwalkers
        self.burn = burn
        self.thin = thin
        self.ntemps = ntemps
        self.workers = workers
        self.is_weighted = is_weighted
        self.sigma_ini = sigma_ini
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.seed = seed

    #
    def resolve(self, *, sigma_fit: float, n_dim: int) -> MC:
        """
        Fill every derivable knob from the optimizer result; return a copy.

        Parameters
        ----------
        sigma_fit : float
            RMS residual of the optimizer result, ``sqrt(chisqr / ndata)``,
            in data units: the start of the sampled noise scale when
            ``sigma_ini`` is None.
        n_dim : int
            Sampled dimensions: varying parameters plus one for ``__lnsigma``
            when ``is_weighted`` is False.

        Raises
        ------
        ValueError
            Derived start outside explicit sigma bounds; explicit ``nwalkers``
            below emcee's minimum of ``2 * n_dim``; a non-positive
            ``sigma_fit`` when the start must be derived.
        """

        sigma_ini: float | None = None
        sigma_min: float | None = None
        sigma_max: float | None = None
        if not self.is_weighted:
            if self.sigma_ini is None:
                if not (np.isfinite(sigma_fit) and sigma_fit > 0):
                    raise ValueError(
                        f"cannot derive sigma_ini: the fit's RMS residual is "
                        f"{sigma_fit} (noiseless data?); pass sigma_ini explicitly"
                    )
                start, derived = float(sigma_fit), True
            else:
                start, derived = self.sigma_ini, False
            sigma_ini = start
            sigma_min = start / 100.0 if self.sigma_min is None else self.sigma_min
            sigma_max = start * 100.0 if self.sigma_max is None else self.sigma_max
            _validate_sigma(sigma_ini, sigma_min, sigma_max, derived=derived)
        if self.nwalkers is None:
            nwalkers = max(20, 2 * n_dim)
        elif self.nwalkers < 2 * n_dim:
            raise ValueError(
                f"nwalkers={self.nwalkers} is below emcee's minimum of "
                f"2 * n_dim = {2 * n_dim} ({n_dim} sampled dimensions: the "
                "varying parameters"
                + ("" if self.is_weighted else " plus __lnsigma")
                + "); raise it or pass nwalkers=None to derive it"
            )
        else:
            nwalkers = self.nwalkers
        return MC(
            use_mc=self.use_mc,
            steps=self.steps,
            nwalkers=nwalkers,
            burn=self.burn,
            thin=self.thin,
            ntemps=self.ntemps,
            workers=self.workers,
            is_weighted=self.is_weighted,
            sigma_ini=sigma_ini,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            seed=self.seed,
        )

    #
    def __repr__(self) -> str:
        return (
            f"MC(use_mc={self.use_mc}, steps={self.steps}, "
            f"nwalkers={self.nwalkers}, sigma_ini={self.sigma_ini}, "
            f"seed={self.seed})"
        )


#
#
@dataclass(frozen=True)
class MCMCResult:
    """MCMC outputs for a single optimization (counterpart to ``MC`` settings).

    A read-only bundle built by ``fit_io.mcmc_result_from_payload`` from the
    persisted mcmc payload — one type shared by the per-file slot path
    (``FitResults.get_mcmc`` / the ``File.get_mcmc`` sugar) and the
    project-level joint path (``JointFitResult.mcmc``).

    Attributes
    ----------
    table : pandas.DataFrame
        Posterior quantile table, one row per sampled parameter — the
        varying model parameters plus the ``__lnsigma`` noise-scale
        nuisance row. Fixed parameters have no posterior and are excluded.
    flatchain : pandas.DataFrame
        Flattened MCMC chain, one column per sampled parameter.
    acceptance_fraction : numpy.ndarray | None
        Per-walker acceptance fraction (healthy range ≈ 0.2–0.5); ``None``
        when absent from the stored payload.
    lnsigma : float | None
        Final value of the ``__lnsigma`` nuisance parameter — a single
        noise scale over the sampled residual (for a joint fit: the
        concatenated residual, never a per-file σ). ``None`` when the
        sampling was weighted (no nuisance parameter).
    """

    table: pd.DataFrame
    flatchain: pd.DataFrame
    acceptance_fraction: np.ndarray | None
    lnsigma: float | None = None


#
#
class par_dummy:
    """
    Dummy parameter object for scipy.optimize compatibility.

    Mimics the structure of scipy.optimize.minimize result objects to allow
    uniform handling of initial guesses and fit results. Useful for displaying
    initial parameter guesses using the same code that handles fit results.

    Attributes
    ----------
    final_simplex : None
        Final simplex (placeholder)
    fun : None
        Objective function value (placeholder)
    message : None
        Optimization message (placeholder)
    nfev : None
        Number of function evaluations (placeholder)
    nit : None
        Number of iterations (placeholder)
    status : None
        Optimization status (placeholder)
    success : bool
        Optimization success flag (always True for dummy)
    x : None or array
        Parameter values (set by par_extract when return_type='par.x')

    Examples
    --------
    >>> # Create dummy result for initial guess
    >>> params_init = par_extract(initial_params, return_type='par.x')
    >>> params_init.x
    [10, 5.0, 1.0]

    >>> # Can now use same plotting code for initial guess and fit result
    >>> plot_parameters(params_init)  # Initial guess
    >>> result = minimize(...)
    >>> plot_parameters(result)        # Fit result
    """

    #
    def __init__(self) -> None:
        self.final_simplex = None
        self.fun = None
        self.message = None
        self.nfev = None
        self.nit = None
        self.status = None
        self.success = True
        self.x: list[Any] | None = None
