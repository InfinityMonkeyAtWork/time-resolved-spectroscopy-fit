"""Noise-weighted fits checked against the Poisson Cramer-Rao bound.

``trspecfit.sensitivity`` builds the Poisson Fisher information from the model
shape and a photon budget alone — it never sees a residual, a Jacobian or an
optimizer. That makes it an independent reference for the weighting the fit
applies: at ``d = m`` the Gauss-Newton matrix of the deviance residual *is* the
Fisher matrix, so ``JᵀJ`` and ``fisher_matrix`` must agree entry for entry, and
on sampled data the quoted ``stderr`` must land on the bound.

``counts`` is the total expected count over the *evaluated window*, so every
test here leaves the fit window at the full axes: ``fisher_matrix`` evaluates
the model over ``model.energy`` (and ``model.time``), and only an unrestricted
``e_lim`` / ``t_lim`` makes the two sums run over the same bins.
"""

import copy

import matplotlib
import numpy as np
import pytest
from _utils import (
    capture_fit_wrapper,
    extract_truth_pars,
    make_project,
    residual_jacobian_fd,
)

matplotlib.use("Agg")

from trspecfit import File, Simulator, fitlib, sensitivity  # noqa: E402

_ENERGY = np.linspace(82.0, 88.0, 41)
_TIME = np.linspace(-2.0, 10.0, 17)
_ENERGY_YAML = "models/eval_2d_energy.yaml"
_TIME_YAML = "models/file_time.yaml"
_MODEL = "offset_only"
_DYNAMICS = "MonoExpPos"

# Peak well above a flat background, amplitude decaying over the time axis:
# the shape the 2026-09-15 control ran on. The YAML values leave the peak only
# a few times the background, which flattens the counting-noise contrast the
# unweighted fit is supposed to get wrong.
_TRUTH = {"Offset_y0": 1.0, "GLP_01_A": 16.0, "GLP_01_A_expFun_01_A": 4.0}

# Counts per data unit for the deterministic tests. Both sides of the identity
# are linear in it, so any positive value works; a non-round one keeps a
# dropped factor from hiding.
_SCALE = 37.5

# Parameters that live on the peak. Their information sits in the few bins
# where the counting noise is highest, which is exactly where an unweighted
# residual misprices it.
_PEAK_PARS = (
    "GLP_01_A",
    "GLP_01_A_expFun_01_A",
    "GLP_01_A_expFun_01_tau",
    "GLP_01_x0",
)


#
def _peak_file(*, name, data=None, backend="fit_model_gir"):
    """File with the peak-on-background model on the shared axes."""

    project = make_project(name=name, spec_fun_str=backend)
    file = File(
        parent_project=project,
        name=name,
        data=data,
        energy=_ENERGY.copy(),
        time=_TIME.copy(),
    )
    file.load_model(model_yaml=_ENERGY_YAML, model_info=_MODEL)
    return file


#
def _add_dynamics(file):
    """Give the peak amplitude its exponential time dependence."""

    file.add_time_dependence(
        target_model=_MODEL,
        target_parameter="GLP_01_A",
        dynamics_yaml=_TIME_YAML,
        dynamics_model=[_DYNAMICS],
    )


#
def _set_truth(model):
    """Move *model* to the truth operating point and return it."""

    for name, value in _TRUTH.items():
        if name in model.parameter_names:
            model.lmfit_pars[name].value = value
    return model


#
def _truth_model(*, dynamics):
    """Truth model on the shared axes, with or without the dynamics."""

    file = _peak_file(name=f"truth_{'2d' if dynamics else '1d'}")
    if dynamics:
        _add_dynamics(file)
    return _set_truth(file.model_active)


#
def _par_at_truth(par, model):
    """Copy of the optimizer's parameters, reset to the truth values."""

    at_truth = copy.deepcopy(par)
    for name, value in extract_truth_pars(model).items():
        at_truth[name].value = value
    return at_truth


#
def _data_matched_const(call, par):
    """``const`` with the data replaced by this backend's own model curve.

    ``d = m`` has to hold bit for bit. The deviance takes the square root of a
    difference that cancels to nothing at the operating point, so a model
    agreeing to the last digit still scores a residual of order
    ``sqrt(2 * scale * eps * m)`` — eight orders above the difference itself.
    """

    fit = np.asarray(fitlib.residual_fun(par, *call["const"], "fit", call["args"]))
    return (call["const"][0], fit, *call["const"][2:])


#
def _information_deviation(observed, expected):
    """``|A - B|`` scaled by ``sqrt(B_ii * B_jj)``, entry by entry.

    A relative comparison is meaningless on the off-diagonal entries of an
    almost orthogonal parameter pair, which pass through zero; scaling by the
    diagonals puts every entry of the matrix on the same footing.
    """

    diagonal = np.sqrt(np.outer(np.diag(expected), np.diag(expected)))
    return np.abs(observed - expected) / diagonal


#
def _fisher_at_truth(model, *, counts, par_names):
    """Fisher matrix over every free parameter, reordered as *par_names*."""

    info, names, _meta = sensitivity.fisher_matrix(model, counts)
    # passing par_names to fisher_matrix would restrict the span, which
    # declares the excluded parameters exactly known; the fit's varying set
    # and the model's free set have to be the same set for the comparison
    assert sorted(names) == sorted(par_names)
    order = [names.index(name) for name in par_names]
    return info[np.ix_(order, order)]


#
def _photon_counting_data(model, *, counts_per_delay, seed):
    """Poisson-sampled 2D data and the ``set_noise`` scale that produced it."""

    sim = Simulator(
        model=model,
        detection="photon_counting",
        counts_per_delay=counts_per_delay,
        seed=seed,
    )
    clean, noisy, _noise = sim.simulate_2d()
    # what _sample_photons_2d applied: the mean row carries counts_per_delay
    scale = counts_per_delay / float(np.mean(np.sum(np.abs(clean), axis=1)))
    return noisy, scale


#
def _fit_2d(data, *, name, scale=None):
    """Fit the 2D model to *data*; ``scale=None`` leaves the fit unweighted."""

    file = _peak_file(name=name, data=data)
    file.define_baseline(0, 4, time_type="ind", show_plot=False)
    file.fit_baseline(model_name=_MODEL, stages=1, fit_alg_1="leastsq", try_ci=0)
    _add_dynamics(file)
    if scale is not None:
        file.set_noise("poisson", scale=scale)
    file.fit_2d(_MODEL, stages=1, fit_alg_1="leastsq", try_ci=0)
    assert file.model_2d is not None  # type guard
    assert file.model_2d.result is not None  # type guard
    return file.model_2d.result.par_fin.params


#
#
class TestGaussNewtonEqualsFisher:
    """On noiseless data the weighted Jacobian carries the Fisher information.

    At ``d = m`` the deviance residual has slope ``-sqrt(scale / m)``, so
    ``JᵀJ = sum_bins (scale / m) (dm/dtheta)(dm/dtheta)``, which is the Poisson
    Fisher matrix at ``lambda = scale * m``. Passing ``counts = scale * sum(m)``
    makes ``sensitivity``'s internal gain exactly ``1 / scale``, so the two
    sides are the same sum and any disagreement is numerical.
    """

    #
    def test_spectrum_fit_information_matches_the_bound(self, monkeypatch):
        """1D: lmfit differences ``residual_fun`` itself on this path.

        There is no analytic ``Dfun`` for a 1D fit, so the matrix lmfit builds
        its covariance from is the finite-difference Jacobian of the weighted
        residual — the object compared here.
        """

        truth = _truth_model(dynamics=False)
        curve = np.asarray(truth.create_value_1d(return_1d=1), dtype=float)

        file = _peak_file(name="crb_1d", data=np.tile(curve, (_TIME.size, 1)))
        _set_truth(file.model_active)
        file.set_noise("poisson", scale=_SCALE)
        calls = capture_fit_wrapper(monkeypatch)
        file.fit_spectrum(
            _MODEL,
            time_point=0,
            time_type="ind",
            stages=1,
            fit_alg_1="leastsq",
            try_ci=0,
            show_plot=False,
        )
        call = calls[0]
        assert call["noise"].kind == "poisson"
        assert call["noise"].scale == pytest.approx(_SCALE)

        par = _par_at_truth(call["par"], truth)
        const = _data_matched_const(call, par)
        residual = fitlib.residual_fun(
            par, *const, "lmfit", call["args"], noise=call["noise"]
        )
        np.testing.assert_array_equal(residual, np.zeros_like(residual))

        jac = residual_jacobian_fd(
            par, const=const, args=call["args"], noise=call["noise"], rel_step=1e-5
        )
        var_names = [name for name in par if par[name].vary]
        info = _fisher_at_truth(
            truth, counts=_SCALE * float(np.sum(curve)), par_names=var_names
        )

        # measured 1.9e-6 on the diagonal, 4.0e-5 over the whole matrix: the
        # off-diagonal floor is finite-difference noise on a residual that is
        # identically zero at the operating point, not a disagreement
        np.testing.assert_allclose(np.diag(jac.T @ jac), np.diag(info), rtol=2e-5)
        assert _information_deviation(jac.T @ jac, info).max() < 2e-4

    #
    def test_2d_fit_information_matches_the_bound(self, monkeypatch):
        """2D: the analytic ``jacobian_fun`` lmfit uses as ``Dfun``."""

        pytest.importorskip("jax")
        truth = _truth_model(dynamics=True)
        truth.create_value_2d()
        clean = np.asarray(truth.value_2d, dtype=float)

        file = _peak_file(name="crb_2d", data=clean.copy(), backend="fit_model_jax")
        file.define_baseline(0, 4, time_type="ind", show_plot=False)
        file.fit_baseline(model_name=_MODEL, stages=1, fit_alg_1="leastsq", try_ci=0)
        _add_dynamics(file)
        _set_truth(file.model_active)
        file.set_noise("poisson", scale=_SCALE)
        calls = capture_fit_wrapper(monkeypatch)
        file.fit_2d(_MODEL, stages=1, fit_alg_1="leastsq", try_ci=0)
        call = calls[0]
        assert call["jac_fun"] is fitlib.jacobian_fun

        par = _par_at_truth(call["par"], truth)
        const = _data_matched_const(call, par)
        residual = fitlib.residual_fun(
            par, *const, "lmfit", call["args"], noise=call["noise"]
        )
        np.testing.assert_array_equal(residual, np.zeros_like(residual))

        jac = call["jac_fun"](par, *const, "lmfit", call["args"], noise=call["noise"])
        var_names = [name for name in par if par[name].vary]
        info = _fisher_at_truth(
            truth, counts=_SCALE * float(np.sum(clean)), par_names=var_names
        )

        # measured 4.4e-7, on the peak position: what is left is the
        # truncation error of sensitivity's own central differences, the
        # analytic Jacobian having none
        assert _information_deviation(jac.T @ jac, info).max() < 2e-6


#
#
class TestStderrAgainstTheBound:
    """A weighted fit of sampled data quotes the bound as its uncertainty."""

    #
    def test_poisson_stderr_matches_the_crb(self):
        """At 1.4e6 counts the fit is asymptotic and ``stderr`` is the CRB.

        ``scale_covar=False`` under a declared noise model leaves the
        covariance as ``(JᵀJ)^-1`` in likelihood units, so this checks the
        whole chain — deviance, Jacobian factor, covariance — against a number
        computed without any of them. One seed: ``stderr`` is evaluated at
        the fitted point, so it scatters seed to seed (a few percent here,
        most on ``tau``); the population statement is the seed mean in
        ``TestSeedScatterControl``.
        """

        truth = _truth_model(dynamics=True)
        truth.create_value_2d()
        clean = np.asarray(truth.value_2d, dtype=float)
        noisy, scale = _photon_counting_data(truth, counts_per_delay=80000, seed=3)

        params = _fit_2d(noisy, name="crb_stderr", scale=scale)
        crb = sensitivity.crb(truth, scale * float(np.sum(clean)))

        for name, bound in crb.items():
            stderr = params[name].stderr
            assert stderr is not None  # type guard
            assert stderr == pytest.approx(bound, rel=0.10), name


#
#
@pytest.mark.slow
class TestSeedScatterControl:
    """Seed scatter against quoted ``stderr``, weighted and unweighted.

    The 2026-09-15 control found the unweighted fit quoting error bars ~1.6x
    too small for peak parameters on photon-counting data, because lmfit's
    redchi-scaled covariance spreads one variance over a window whose counting
    noise is far from uniform. The declared Poisson model is the fix, and the
    scatter over seeds is the only thing that can confirm it.
    """

    #
    def test_scatter_over_stderr_lands_on_one_only_under_poisson(self):
        truth = _truth_model(dynamics=True)
        truth_values = extract_truth_pars(truth)
        truth.create_value_2d()
        clean = np.asarray(truth.value_2d, dtype=float)

        names = [name for name in truth.lmfit_pars if truth.lmfit_pars[name].vary]

        n_seeds = 256
        values = {"unknown": [], "poisson": []}
        errors = {"unknown": [], "poisson": []}
        for seed in range(n_seeds):
            noisy, scale = _photon_counting_data(
                truth, counts_per_delay=20000, seed=seed
            )
            for kind, declared in (("unknown", None), ("poisson", scale)):
                params = _fit_2d(noisy, name=f"{kind}_{seed}", scale=declared)
                assert [name for name in params if params[name].vary] == names
                assert all(params[name].stderr is not None for name in names)
                values[kind].append([params[name].value for name in names])
                errors[kind].append([params[name].stderr for name in names])

        ratio = {
            kind: dict(
                zip(
                    names,
                    np.std(values[kind], axis=0, ddof=1)
                    / np.mean(errors[kind], axis=0),
                    strict=True,
                )
            )
            for kind in values
        }

        # the standard deviation of n_seeds samples is itself uncertain by
        # 1 / sqrt(2 * n_seeds) = 4.4%; the band is 3.4 times that
        for name, value in ratio["poisson"].items():
            assert 0.85 < value < 1.15, f"{name}: {value:.3f}"

        # the seed mean of stderr is the population version of the one-seed
        # spot check above; measured 1.000-1.009 against the bound, the
        # standard error of that mean is 0.5% on tau and below 0.2% elsewhere
        # (scale is the same for every seed: the clean signal sets it)
        crb = sensitivity.crb(truth, scale * float(np.sum(clean)))
        mean_stderr = dict(zip(names, np.mean(errors["poisson"], axis=0), strict=True))
        for name, value in mean_stderr.items():
            assert value / crb[name] == pytest.approx(1.0, rel=0.02), name

        # unweighted: too small on the peak (measured 1.39 to 1.79) and too
        # large on the background (0.60), which is one misweighting, not a
        # missing overall factor. Both bounds belong to this geometry: the
        # contrast between peak and background sets how far the counting
        # noise is from uniform, and with it how wrong the redchi scaling is
        for name in _PEAK_PARS:
            assert ratio["unknown"][name] > 1.25, (
                f"{name}: {ratio['unknown'][name]:.3f}"
            )
        assert ratio["unknown"]["Offset_y0"] < 0.85

        # error bars that cover the truth: no parameter drifts off it
        bias = (
            np.mean(values["poisson"], axis=0) - [truth_values[n] for n in names]
        ) / (np.std(values["poisson"], axis=0, ddof=1) / np.sqrt(n_seeds))
        assert np.max(np.abs(bias)) < 3.0, dict(zip(names, bias, strict=True))
