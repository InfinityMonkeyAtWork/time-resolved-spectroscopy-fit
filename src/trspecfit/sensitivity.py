"""
Fisher-information sensitivity limits and photon-budget estimates.

For photon-counting data the precision attainable on any model parameter is
bounded below by the Cramer-Rao inequality. Because the bound depends only on
the model shape and the photon budget — not on measured data — it can be
evaluated before an experiment runs, which makes it useful for beam-time
planning and for deciding whether a target effect is detectable at all.

[function "fisher_matrix"]
Poisson Fisher information matrix over a model's free parameters.

[function "crb"]
Cramer-Rao lower bounds on parameter uncertainties for a photon budget.

[function "counts_required"]
Invert the bound: photons needed to reach a target parameter precision.

[function "sensitivity_report"]
Tabulate bounds, correlation penalties and relative precision.

Notes
-----
**Count convention.** ``counts`` always means the *total* expected photon
count over the evaluated window, background included. For data from
``Simulator(detection='photon_counting', counts_per_delay=...)`` that is
``counts_per_delay`` in 1D, where the sampler scales the spectrum to that
total; in 2D it normalises the *mean* row total instead, so the window carries
``counts_per_delay * n_time``. Counts in the peaks alone are reported
separately as ``counts_in_peaks``; confusing the two is the most common way to
misread these numbers.

**Exact scaling.** The Fisher information is linear in the photon budget for a
fixed model shape, so every bound scales as ``counts**-0.5`` exactly. Bounds at
one budget therefore determine bounds at every other budget with no refit, and
:func:`counts_required` inverts the relation in closed form rather than
searching.

**What the bound does and does not cover.** It is the best any unbiased
estimator can do given Poisson statistics. It says nothing about a fitter that
weights its residuals sub-optimally, and such a fit goes wrong in two separate
ways. Its *scatter* over repeated measurements is the smaller effect: a
least-squares fit with unweighted residuals is no longer the maximum-likelihood
estimator, yet on a peak-on-background control its scatter stayed within 25%
of the bound. Its *quoted* ``stderr`` is the larger one: lmfit rescales the
covariance by ``redchi``, spreading a single variance over a window whose
counting noise peaks where the signal does, so on the same control the error
bars came out 1.4-1.8 times too small on the peak parameters and too large on
the background offset (``tests/test_noise_crb.py::TestSeedScatterControl``).
Comparing measured scatter against these numbers separates "not enough
photons" from "wrong objective function".

Declaring the noise model with ``File.set_noise('poisson', scale=...)`` removes
both: the fit then minimises the Poisson deviance, which is the
maximum-likelihood objective, so its scatter reaches the bound and its quoted
``stderr`` equals it (verified against :func:`fisher_matrix` in
``tests/test_noise_crb.py``).

**Detector noise.** Pure Poisson statistics are assumed, which holds for true
single-event counting. Analog detection (e.g. MCP stack into a phosphor and
CCD) carries an excess-noise factor F > 1 that reduces the effective count to
roughly ``counts / F``; pass the reduced value for such data.

**API tier.** Every function here takes an ``mcp.Model`` — ``file.model_active``
after ``File.load_model``, promoted to ``dim=2`` by ``add_time_dependence`` — so
they sit in the advanced tier next to ``trspecfit.mcp``, not in the user API
(``Project``, ``File``, ``Simulator``, ``PlotConfig``). See ``docs/stability.md``:
the stability commitment for that tier is not stated yet.
"""

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd

from trspecfit import mcp
from trspecfit.config import functions as fcts_config

# Relative condition-number ceiling above which the Fisher matrix is treated
# as singular and inverted with a pseudo-inverse instead.
_COND_LIMIT = 1e12


#
def _free_parameter_names(model: mcp.Model) -> list[str]:
    """
    Names of parameters the measurement can actually constrain.

    Fixed parameters carry no uncertainty, and expression parameters have no
    freedom of their own — their precision follows from whatever they
    reference. Both are excluded.
    """

    return [
        name
        for name in model.parameter_names
        if model.lmfit_pars[name].expr is None and model.lmfit_pars[name].vary
    ]


#
def _model_values(model: mcp.Model) -> np.ndarray:
    """
    Evaluate the model over its full axes and return a flat float array.

    Dispatches on ``model.dim`` so 1D energy models and 2D time-and-energy
    models share one code path; the flattening is harmless because the Fisher
    sum runs over all bins regardless of their layout.
    """

    if model.dim == 2:
        model.create_value_2d()
        values = model.value_2d
    else:
        values = model.create_value_1d(return_1d=1)

    if values is None:
        raise ValueError(
            "Model evaluation produced no values; check that model.energy "
            "(and model.time for dim=2) are set."
        )

    return np.asarray(values, dtype=float).ravel()


#
def _as_intensity(values: np.ndarray, context: str) -> np.ndarray:
    """
    Validate that model values can be read as a Poisson intensity.

    A Poisson rate cannot be negative. Small excursions below zero are
    numerical and get clipped; a genuinely negative model is a modelling error
    that would silently corrupt the information matrix, so it raises.
    """

    scale = float(np.max(np.abs(values))) if values.size else 0.0
    if scale > 0.0 and float(np.min(values)) < -1e-9 * scale:
        raise ValueError(
            f"{context} contains negative intensities "
            f"(min={float(np.min(values)):.4g}), which cannot be a photon "
            "rate. Check background components and parameter bounds."
        )
    return np.asarray(np.clip(values, 0.0, None), dtype=float)


#
def _counts_in_peaks(model: mcp.Model, total_signal: float, counts: float) -> float:
    """
    Expected counts contributed by non-background components.

    Reported alongside the bounds because sensitivity is driven by photons in
    the peaks, while ``counts`` is a whole-window budget; the ratio is what a
    user needs to translate between the two.
    """

    background = set(fcts_config.background_functions())
    peak_signal = 0.0
    for component in model.components:
        if component.fct_str in background:
            continue
        value = component.value()
        peak_signal += float(np.sum(np.abs(np.asarray(value, dtype=float))))

    if total_signal <= 0.0:
        return 0.0
    return counts * peak_signal / total_signal


#
def fisher_matrix(
    model: mcp.Model,
    counts: float,
    par_names: Sequence[str] | None = None,
    *,
    rel_step: float = 1e-5,
) -> tuple[np.ndarray, list[str], dict[str, float]]:
    """
    Poisson Fisher information matrix over a model's free parameters.

    Builds ``I_jk = sum_bins (1/lambda) (dlambda/dtheta_j)(dlambda/dtheta_k)``
    with ``lambda`` the expected counts per bin, using central finite
    differences on the model evaluator. Works with any lineshape, background or
    dynamics the model supports, because the derivatives are taken numerically
    rather than analytically.

    Parameters
    ----------
    model : mcp.Model
        Model with axes set (``model.energy``, plus ``model.time`` for
        ``dim=2``). Parameter values define the operating point at which the
        information is evaluated; the model is left unmodified.
    counts : float
        Total expected photon count over the evaluated window, background
        included. Must be positive.
    par_names : sequence of str, optional
        Parameters the matrix should span. Defaults to every free parameter
        (``vary=True`` and no expression). Passing fixed or expression
        parameters is an error, since they have no information of their own.

        Restricting the span is equivalent to declaring the excluded
        parameters **exactly known**, which yields optimistic bounds. For
        realistic bounds on a subset, span everything and select afterwards —
        which is what :func:`crb` does.
    rel_step : float, default=1e-5
        Relative finite-difference step. Each parameter uses
        ``max(abs(value), 1.0) * rel_step``.

    Returns
    -------
    info : ndarray
        Fisher information matrix, shape ``(n_par, n_par)``, ordered as
        ``par_names``. Units are inverse squared parameter units.
    par_names : list of str
        The parameters actually included, in matrix order.
    meta : dict
        ``total_signal`` (summed model value in model units), ``gain``
        (model units per count), ``counts_in_peaks``, ``n_bins`` (bins with
        non-negligible intensity) and ``condition_number``.

    Raises
    ------
    ValueError
        If ``counts`` is not positive, if no free parameters remain, if a
        requested parameter is fixed or expression-defined, or if the model
        evaluates to a negative intensity.

    Notes
    -----
    The information is linear in ``counts``, so the matrix at one budget can
    be rescaled to any other by simple multiplication.
    """

    if counts <= 0:
        raise ValueError(f"counts must be positive, got {counts}.")

    free = _free_parameter_names(model)
    if par_names is None:
        names = free
    else:
        names = list(par_names)
        unknown = [n for n in names if n not in model.parameter_names]
        if unknown:
            raise ValueError(f"Unknown parameter(s): {unknown}")
        constrained = [n for n in names if n not in free]
        if constrained:
            raise ValueError(
                f"Parameter(s) {constrained} are fixed or expression-defined "
                "and carry no information of their own. Their precision "
                "follows from the parameters they reference."
            )
    if not names:
        raise ValueError("Model has no free parameters; nothing to compute bounds for.")

    baseline = _as_intensity(_model_values(model), "Model")
    total_signal = float(np.sum(baseline))
    if total_signal <= 0.0:
        raise ValueError("Model evaluates to zero everywhere; cannot normalise.")

    # Model units per count: converts the model onto an absolute count scale
    # without requiring the user to express amplitudes in counts.
    gain = total_signal / counts
    lam = baseline / gain

    jacobian = np.empty((len(names), lam.size), dtype=float)
    for row, name in enumerate(names):
        par = model.lmfit_pars[name]
        value = float(par.value)
        step = max(abs(value), 1.0) * rel_step
        try:
            par.value = value + step
            up = _as_intensity(_model_values(model), f"Model at {name}+h") / gain
            par.value = value - step
            down = _as_intensity(_model_values(model), f"Model at {name}-h") / gain
        finally:
            par.value = value
        jacobian[row] = (up - down) / (2.0 * step)

    # Bins with no expected counts carry no information and would divide by
    # zero; excluding them is exact, not an approximation.
    keep = lam > 1e-12
    if not np.any(keep):
        raise ValueError("No bins with non-negligible intensity.")
    weighted = jacobian[:, keep] / lam[keep]
    info = weighted @ jacobian[:, keep].T

    # Restore the exact operating point: finite differencing rewrote parameter
    # values, and lmfit expressions may have been re-evaluated along the way.
    _model_values(model)

    meta = {
        "total_signal": total_signal,
        "gain": gain,
        "counts_in_peaks": _counts_in_peaks(model, total_signal, counts),
        "n_bins": float(np.count_nonzero(keep)),
        "condition_number": float(np.linalg.cond(info)),
    }
    return info, names, meta


#
def _invert(info: np.ndarray, names: Sequence[str]) -> np.ndarray:
    """
    Covariance from a Fisher matrix, tolerating degenerate directions.

    A singular Fisher matrix is a statement about the measurement, not a
    failure: some parameter combination is unidentifiable from this model and
    window. Rather than raising, fall back to a pseudo-inverse and warn, so the
    identifiable parameters still get usable bounds.
    """

    condition = np.linalg.cond(info)
    if condition < _COND_LIMIT:
        try:
            return np.linalg.inv(info)
        except np.linalg.LinAlgError:
            pass

    warnings.warn(
        f"Fisher matrix is near-singular (condition number {condition:.3g}): "
        f"some combination of {list(names)} is not identifiable from this "
        "model and window. Using a pseudo-inverse; affected bounds are "
        "unreliable or infinite.",
        stacklevel=3,
    )
    return np.linalg.pinv(info)


#
def crb(
    model: mcp.Model,
    counts: float,
    par_names: Sequence[str] | None = None,
    *,
    rel_step: float = 1e-5,
    marginal: bool = True,
) -> dict[str, float]:
    """
    Cramer-Rao lower bounds on parameter uncertainties for a photon budget.

    The smallest standard deviation any unbiased estimator can achieve on each
    parameter, given Poisson statistics and ``counts`` photons.

    Parameters
    ----------
    model : mcp.Model
        Model with axes set. Left unmodified.
    counts : float
        Total expected photon count over the window, background included.
    par_names : sequence of str, optional
        Which bounds to return. Defaults to all free parameters. This selects
        the *output* only: the information matrix always spans every free
        parameter, so asking for one parameter still accounts for covariance
        with the rest of the model.
    rel_step : float, default=1e-5
        Relative finite-difference step for the Jacobian.
    marginal : bool, default=True
        If True, bounds account for covariance with the other free parameters
        (the realistic case: everything is fitted at once). If False, each
        bound assumes every other parameter is known exactly — optimistic, but
        useful for isolating how much correlation costs.

    Returns
    -------
    dict
        Parameter name to bound, in the parameter's own units. Unidentifiable
        parameters map to ``inf``.

    Examples
    --------
    >>> # Smallest resolvable binding-energy shift with 10^4 photons
    >>> bounds = crb(model, counts=1e4)
    >>> print(f"{bounds['GLP_01_x0'] * 1e3:.1f} meV")

    >>> # How much does fitting width and amplitude cost the shift precision?
    >>> tied = crb(model, counts=1e4, marginal=False)
    >>> penalty = bounds['GLP_01_x0'] / tied['GLP_01_x0']

    See Also
    --------
    counts_required : Invert the bound for a target precision.
    sensitivity_report : Tabulated summary over all free parameters.
    """

    # Always span the full free set: a bound on one parameter that silently
    # assumed the others were known would be optimistic, sometimes by a large
    # factor. Selection happens after inversion.
    info, names, _ = fisher_matrix(model, counts, None, rel_step=rel_step)

    if marginal:
        variances = np.diag(_invert(info, names))
    else:
        with np.errstate(divide="ignore"):
            variances = 1.0 / np.diag(info)

    bounds: dict[str, float] = {}
    for name, variance in zip(names, variances, strict=True):
        bounds[name] = float(np.sqrt(variance)) if variance > 0 else float("inf")

    if par_names is None:
        return bounds

    requested = list(par_names)
    missing = [n for n in requested if n not in bounds]
    if missing:
        constrained = [n for n in missing if n in model.parameter_names]
        if constrained:
            raise ValueError(
                f"Parameter(s) {constrained} are fixed or expression-defined "
                "and carry no information of their own. Their precision "
                "follows from the parameters they reference."
            )
        raise ValueError(f"Unknown parameter(s): {missing}")
    return {name: bounds[name] for name in requested}


#
def counts_required(
    model: mcp.Model,
    par_name: str,
    target_sigma: float,
    *,
    counts_ref: float = 1.0e4,
    rel_step: float = 1e-5,
    marginal: bool = True,
) -> float:
    """
    Photons needed to reach a target precision on one parameter.

    Exploits the exact ``sigma ~ counts**-0.5`` scaling: one Fisher evaluation
    at ``counts_ref`` determines the requirement in closed form, with no
    iteration.

    Parameters
    ----------
    model : mcp.Model
        Model with axes set. Left unmodified.
    par_name : str
        Free parameter to reach the target on.
    target_sigma : float
        Desired standard deviation, in the parameter's own units. Must be
        positive.
    counts_ref : float, default=1e4
        Reference budget at which the information is evaluated. The result is
        independent of this choice; it only sets the numerical scale.
    rel_step : float, default=1e-5
        Relative finite-difference step for the Jacobian.
    marginal : bool, default=True
        Whether to account for covariance with the other free parameters.

    Returns
    -------
    float
        Total expected photon count over the window, background included.
        ``inf`` if the parameter is unidentifiable at any budget.

    Examples
    --------
    >>> # Photons for 5 meV shift sensitivity, and the beam time it implies
    >>> n = counts_required(model, 'GLP_01_x0', target_sigma=0.005)
    >>> seconds = n / count_rate_hz

    Notes
    -----
    The scaling holds only while the model *shape* is fixed. Changing the
    window, the lineshape or the background fraction changes the information
    per photon and requires a fresh evaluation.
    """

    if target_sigma <= 0:
        raise ValueError(f"target_sigma must be positive, got {target_sigma}.")

    reference = crb(
        model,
        counts_ref,
        [par_name],
        rel_step=rel_step,
        marginal=marginal,
    )[par_name]

    if not np.isfinite(reference):
        return float("inf")
    return float(counts_ref * (reference / target_sigma) ** 2)


#
def sensitivity_report(
    model: mcp.Model,
    counts: float,
    par_names: Sequence[str] | None = None,
    *,
    rel_step: float = 1e-5,
) -> pd.DataFrame:
    """
    Tabulate bounds, correlation penalties and relative precision.

    One row per free parameter, ordered as the model declares them.

    Parameters
    ----------
    model : mcp.Model
        Model with axes set. Left unmodified.
    counts : float
        Total expected photon count over the window, background included.
    par_names : sequence of str, optional
        Parameters to include. Defaults to all free parameters.
    rel_step : float, default=1e-5
        Relative finite-difference step for the Jacobian.

    Returns
    -------
    pd.DataFrame
        Columns:

        - **name** -- parameter name
        - **value** -- operating point the bound was evaluated at
        - **sigma** -- marginal Cramer-Rao bound (the realistic one)
        - **sigma_uncorrelated** -- bound with all other parameters held known
        - **correlation_penalty** -- ``sigma / sigma_uncorrelated``; how much
          is lost to covariance with the rest of the model. Values far above 1
          flag a parameter whose precision is limited by degeneracy rather
          than by photon count, which more photons will not fix.
        - **rel_precision** -- ``sigma / abs(value)``, or NaN at ``value == 0``

        The frame carries ``counts``, ``counts_in_peaks``, ``n_bins`` and
        ``condition_number`` in ``DataFrame.attrs``.

    Examples
    --------
    >>> df = sensitivity_report(model, counts=1e4)
    >>> df.sort_values('correlation_penalty', ascending=False).head()
    """

    info, names, meta = fisher_matrix(model, counts, par_names, rel_step=rel_step)

    covariance = _invert(info, names)
    marginal_var = np.diag(covariance)
    with np.errstate(divide="ignore"):
        independent_var = 1.0 / np.diag(info)

    def _sigma(variance: float) -> float:
        return float(np.sqrt(variance)) if variance > 0 else float("inf")

    rows = []
    for index, name in enumerate(names):
        value = float(model.lmfit_pars[name].value)
        sigma = _sigma(marginal_var[index])
        sigma_free = _sigma(independent_var[index])
        rows.append(
            {
                "name": name,
                "value": value,
                "sigma": sigma,
                "sigma_uncorrelated": sigma_free,
                "correlation_penalty": (
                    sigma / sigma_free
                    if np.isfinite(sigma_free) and sigma_free > 0
                    else float("inf")
                ),
                "rel_precision": (sigma / abs(value) if value != 0 else float("nan")),
            }
        )

    frame = pd.DataFrame(rows)
    frame.attrs["counts"] = float(counts)
    frame.attrs["counts_in_peaks"] = meta["counts_in_peaks"]
    frame.attrs["n_bins"] = meta["n_bins"]
    frame.attrs["condition_number"] = meta["condition_number"]
    return frame
