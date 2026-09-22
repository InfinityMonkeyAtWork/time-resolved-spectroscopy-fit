"""Declared noise models: residual, Jacobian factor, view reductions, wiring.

The deviance branches (``d = 0``, ``d = m``, below the model floor) are pinned
against closed forms, and every derivative is cross-checked against central
finite differences of the residual it claims to differentiate -- including the
full residual Jacobian through ``fitlib.jacobian_fun``, which is the sign check
for the ``-dm/dtheta`` convention of the unweighted Jacobian.
"""

import copy
import warnings

import matplotlib
import numpy as np
import pytest
from _utils import make_project
from scipy.special import kl_div

matplotlib.use("Agg")

from trspecfit import File, fitlib  # noqa: E402
from trspecfit.utils.noise import (  # noqa: E402
    EPS_COUNTS,
    NoiseModel,
    SegmentedNoise,
    apply_segments,
    jacobian_factor_segments,
)

pytest.importorskip("jax")

from trspecfit.eval_jax import (  # noqa: E402
    make_evaluator_2d_jax,
    make_jacobian_2d_jax,
)
from trspecfit.graph_ir import build_graph, schedule_2d  # noqa: E402

_ENERGY_YAML = "models/file_energy.yaml"
_MODEL_INFO = "single_glp"


#
def _make_1d_glp_file():
    """File with a loaded 1D GLP model (public API)."""

    project = make_project(name="noise")
    file = File(parent_project=project, energy=np.linspace(80, 90, 101))
    file.load_model(model_yaml=_ENERGY_YAML, model_info=_MODEL_INFO)
    assert file.model_active is not None  # type guard
    return file


#
def _make_2d_jax_args():
    """2D GLP model plus the JAX dispatch args, as ``File.fit_2d`` builds them."""

    project = make_project(name="noise")
    file = File(parent_project=project)
    file.energy = np.linspace(80, 90, 21)
    file.time = np.linspace(-5, 20, 7)
    file.load_model(model_yaml=_ENERGY_YAML, model_info=_MODEL_INFO)
    model = file.model_active
    assert model is not None  # type guard
    plan = schedule_2d(build_graph(model))
    name_to_idx = {name: i for i, name in enumerate(model.parameter_names)}
    theta_indices = np.array(
        [name_to_idx[name] for name in plan.opt_param_names], dtype=np.intp
    )
    args = (
        make_evaluator_2d_jax(plan),
        make_jacobian_2d_jax(plan),
        theta_indices,
        model,
        2,
    )
    return file, model, args


#
def _evaluate(par, file, data, fit_fun_str, args):
    """Model prediction over the full data grid."""

    return np.asarray(
        fitlib.residual_fun(par, file.energy, data, fit_fun_str, 0, [], [], "fit", args)
    )


#
def _residual_jacobian_fd(par, *, file, data, fit_fun_str, args, e_lim, t_lim, noise):
    """Central-difference residual Jacobian, columns in lmfit vary order."""

    columns = []
    for name in [name for name in par if par[name].vary]:
        perturbed = copy.deepcopy(par)
        value = perturbed[name].value
        step = 1e-6 * max(1.0, abs(value))
        perturbed[name].value = value + step
        res_plus = np.asarray(
            fitlib.residual_fun(
                perturbed,
                file.energy,
                data,
                fit_fun_str,
                0,
                e_lim,
                t_lim,
                "lmfit",
                args,
                noise=noise,
            )
        )
        perturbed[name].value = value - step
        res_minus = np.asarray(
            fitlib.residual_fun(
                perturbed,
                file.energy,
                data,
                fit_fun_str,
                0,
                e_lim,
                t_lim,
                "lmfit",
                args,
                noise=noise,
            )
        )
        columns.append((res_plus - res_minus) / (2 * step))
    return np.stack(columns, axis=1)


#
def _factor_fd(noise, d, m, *, step):
    """Central difference of ``apply`` with respect to the model."""

    return (noise.apply(d, m + step) - noise.apply(d, m - step)) / (2 * step)


# ---------------------------------------------------------------------------
# Deviance limits
# ---------------------------------------------------------------------------


#
def test_poisson_residual_at_zero_data_is_sqrt_law():
    """``d = 0`` reduces the deviance to ``-sqrt(2 * scale * m)``."""

    noise = NoiseModel(kind="poisson", scale=2.5)
    m = np.array([0.4, 3.0, 17.0])
    expected = -np.sqrt(2.0 * 2.5 * m)
    np.testing.assert_allclose(noise.apply(np.zeros(3), m), expected, rtol=1e-12)


#
def test_poisson_residual_vanishes_at_perfect_model():
    """``d = m`` gives exactly zero, so noiseless data cost nothing."""

    noise = NoiseModel(kind="poisson", scale=3.0)
    d = np.array([0.0, 0.7, 12.0, 250.0])
    assert np.all(noise.apply(d, d) == 0.0)


#
def test_poisson_sum_of_squares_equals_deviance():
    """Σ r² is the deviance ``2 * scale * Σ kl_div(d, m)``."""

    scale = 1.7
    noise = NoiseModel(kind="poisson", scale=scale)
    d = np.array([0.0, 1.0, 4.0, 9.0, 31.0])
    m = np.array([0.6, 1.4, 3.1, 11.0, 28.5])
    residual = noise.apply(d, m)
    expected = 2.0 * scale * float(np.sum(kl_div(d, m)))
    assert float(np.sum(residual**2)) == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# Floor policy
# ---------------------------------------------------------------------------


#
def test_poisson_residual_below_floor_is_linear_with_floor_slope():
    """For ``d > 0`` the residual continues linearly below the model floor."""

    scale = 4.0
    noise = NoiseModel(kind="poisson", scale=scale)
    m_floor = EPS_COUNTS / scale
    d = np.array([2.0, 2.0])
    m_below = np.array([0.4 * m_floor, 0.1 * m_floor])

    slope = float(noise.jacobian_factor(np.array([2.0]), np.array([m_floor]))[0])
    res_floor = float(noise.apply(np.array([2.0]), np.array([m_floor]))[0])
    expected = res_floor + slope * (m_below - m_floor)

    np.testing.assert_allclose(noise.apply(d, m_below), expected, rtol=1e-12)
    assert slope < 0  # a zero slope would stall Dfun below the floor


#
def test_poisson_residual_at_zero_data_plateaus_for_negative_model():
    """``d = 0`` scores ``m <= 0`` like ``m = 0``, without warnings."""

    noise = NoiseModel(kind="poisson", scale=2.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        residual = noise.apply(np.zeros(3), np.array([-5.0, -1e-9, 0.0]))
        factor = noise.jacobian_factor(np.zeros(3), np.array([-5.0, -1e-9, 0.0]))
    assert np.all(residual == 0.0)
    assert np.all(np.isfinite(factor))
    # the factor is taken at the floor, so the gradient stays finite and
    # pushes a negative model back up
    m_floor = EPS_COUNTS / 2.0
    np.testing.assert_allclose(factor, -np.sqrt(2.0 / (2.0 * m_floor)), rtol=1e-12)


# ---------------------------------------------------------------------------
# Gaussian broadcasting and view reductions
# ---------------------------------------------------------------------------


#
def test_poisson_residual_and_factor_propagate_nan_data():
    """NaN data stay NaN instead of scoring like a zero-count bin."""

    noise = NoiseModel(kind="poisson", scale=2.0)
    d = np.array([np.nan, 0.0, 3.0])
    m = np.array([1.5, 1.5, 1.5])
    residual = noise.apply(d, m)
    factor = noise.jacobian_factor(d, m)
    assert np.isnan(residual[0]) and np.isnan(factor[0])
    assert np.all(np.isfinite(residual[1:])) and np.all(np.isfinite(factor[1:]))


#
def test_gaussian_residual_scalar_and_per_point_sigma():
    """Scalar σ broadcasts; a per-point σ divides element-wise."""

    d = np.array([1.0, 2.0, 3.0])
    m = np.array([0.5, 2.5, 1.0])
    np.testing.assert_allclose(
        NoiseModel(kind="gaussian", sigma=0.5).apply(d, m), (d - m) / 0.5
    )
    sigma = np.array([0.5, 1.0, 2.0])
    per_point = NoiseModel(kind="gaussian", sigma=sigma)
    np.testing.assert_allclose(per_point.apply(d, m), (d - m) / sigma)
    np.testing.assert_allclose(per_point.jacobian_factor(d, m), -1.0 / sigma)


#
def test_unknown_residual_is_unweighted():
    """``unknown`` reproduces ``data - fit`` and a factor of -1."""

    noise = NoiseModel(kind="unknown")
    d = np.array([1.0, 2.0])
    m = np.array([0.25, 3.0])
    assert not noise.is_weighted
    np.testing.assert_allclose(noise.apply(d, m), d - m)
    np.testing.assert_allclose(noise.jacobian_factor(d, m), [-1.0, -1.0])


#
def test_for_view_baseline_constant_sigma():
    """A baseline over n slices divides a constant σ by √n."""

    reduced = NoiseModel(kind="gaussian", sigma=0.6).for_view(average=9)
    assert reduced.sigma == pytest.approx(0.2)


#
def test_for_view_baseline_per_point_sigma():
    """A baseline over n slices gives ``sqrt(Σ σ²) / n`` pixel-wise."""

    sigma = np.array([[1.0, 2.0], [4.0, 4.0], [8.0, 1.0]])
    reduced = NoiseModel(kind="gaussian", sigma=sigma).for_view(
        rows=slice(0, 3), average=3
    )
    expected = np.sqrt(np.sum(sigma**2, axis=0)) / 3
    np.testing.assert_allclose(reduced.sigma, expected)


#
def test_for_view_baseline_poisson_scale():
    """Averaging n slices of Poisson data multiplies the scale by n."""

    reduced = NoiseModel(kind="poisson", scale=2.5).for_view(average=4)
    assert reduced.scale == pytest.approx(10.0)
    assert reduced.sigma is None


#
def test_for_view_slice_and_window_align_sigma():
    """A row index drops the time axis; the energy window slices the last axis."""

    sigma = np.arange(12.0).reshape(3, 4) + 1.0
    noise = NoiseModel(kind="gaussian", sigma=sigma)

    single = noise.for_view(rows=1, e_window=slice(1, 3))
    np.testing.assert_allclose(single.sigma, sigma[1, 1:3])

    window = noise.for_view(rows=slice(0, 2), e_window=slice(2, 4))
    np.testing.assert_allclose(window.sigma, sigma[0:2, 2:4])

    assert NoiseModel(kind="poisson", scale=3.0).for_view(rows=1).scale == 3.0


#
def test_for_view_rejects_average_mismatched_with_sigma_rows():
    """Averaging n slices needs exactly n rows of per-point σ."""

    noise = NoiseModel(kind="gaussian", sigma=np.ones((4, 5)))
    with pytest.raises(ValueError, match="rows"):
        noise.for_view(rows=slice(0, 2), average=3)


# ---------------------------------------------------------------------------
# Jacobian factor
# ---------------------------------------------------------------------------


#
def test_poisson_factor_matches_finite_differences():
    """The closed-form factor differentiates the deviance residual."""

    noise = NoiseModel(kind="poisson", scale=2.0)
    d = np.array([0.0, 1.0, 5.0, 40.0, 3.0])
    m = np.array([0.3, 2.2, 4.2, 37.5, 9.0])
    factor = noise.jacobian_factor(d, m)
    fd = _factor_fd(noise, d, m, step=1e-7)
    np.testing.assert_allclose(factor, fd, rtol=1e-6)
    assert np.all(factor < 0)


#
def test_poisson_factor_limit_branch_equals_analytic_limit():
    """At ``d = m`` the factor takes the analytic limit ``-sqrt(scale / m)``."""

    scale = 2.5
    noise = NoiseModel(kind="poisson", scale=scale)
    m = np.array([0.5, 4.0, 90.0])
    np.testing.assert_allclose(
        noise.jacobian_factor(m, m), -np.sqrt(scale / m), rtol=1e-12
    )
    # the branch is a neighborhood, not a single point
    np.testing.assert_allclose(
        noise.jacobian_factor(m, m * (1 + 1e-9)),
        -np.sqrt(scale / (m * (1 + 1e-9))),
        rtol=1e-12,
    )


#
def test_poisson_factor_below_floor_is_the_floor_slope():
    """Below the floor the factor freezes at its value at the floor."""

    scale = 4.0
    noise = NoiseModel(kind="poisson", scale=scale)
    m_floor = EPS_COUNTS / scale
    d = np.array([2.0, 2.0])
    at_floor = float(noise.jacobian_factor(np.array([2.0]), np.array([m_floor]))[0])
    np.testing.assert_allclose(
        noise.jacobian_factor(d, np.array([0.5 * m_floor, 0.01 * m_floor])),
        [at_floor, at_floor],
        rtol=1e-12,
    )
    # and it is the slope the residual actually has down there
    fd = _factor_fd(noise, d, np.array([0.5 * m_floor, 0.2 * m_floor]), step=1e-8)
    np.testing.assert_allclose(fd, [at_floor, at_floor], rtol=1e-6)


# ---------------------------------------------------------------------------
# Full residual Jacobian through fitlib
# ---------------------------------------------------------------------------


#
def test_residual_jacobian_poisson_matches_finite_differences():
    """``jacobian_fun`` with Poisson noise differentiates ``residual_fun``."""

    file, model, args = _make_2d_jax_args()
    e_lim, t_lim = [8, 14], [1, 6]
    scale = 20.0
    clean = _evaluate(model.lmfit_pars, file, np.zeros((7, 21)), "fit_model_jax", args)
    data = np.random.default_rng(7).poisson(np.maximum(clean, 0.0) * scale) / scale
    noise = NoiseModel(kind="poisson", scale=scale)

    jac = fitlib.jacobian_fun(
        model.lmfit_pars,
        file.energy,
        data,
        "fit_model_jax",
        0,
        e_lim,
        t_lim,
        "lmfit",
        args,
        noise=noise,
    )
    fd = _residual_jacobian_fd(
        model.lmfit_pars,
        file=file,
        data=data,
        fit_fun_str="fit_model_jax",
        args=args,
        e_lim=e_lim,
        t_lim=t_lim,
        noise=noise,
    )
    assert jac.shape == fd.shape
    np.testing.assert_allclose(jac, fd, rtol=1e-5, atol=1e-8)


#
def test_residual_jacobian_gaussian_matches_finite_differences():
    """Per-point σ weights the Jacobian rows by ``1 / σ``, sign included."""

    file, model, args = _make_2d_jax_args()
    e_lim, t_lim = [8, 14], [1, 6]
    clean = _evaluate(model.lmfit_pars, file, np.zeros((7, 21)), "fit_model_jax", args)
    rng = np.random.default_rng(11)
    sigma_full = 0.05 + 0.1 * rng.random((7, 21))
    data = clean + sigma_full * rng.standard_normal((7, 21))
    noise = NoiseModel(kind="gaussian", sigma=sigma_full).for_view(
        rows=slice(t_lim[0], t_lim[1]), e_window=slice(e_lim[0], e_lim[1])
    )

    jac = fitlib.jacobian_fun(
        model.lmfit_pars,
        file.energy,
        data,
        "fit_model_jax",
        0,
        e_lim,
        t_lim,
        "lmfit",
        args,
        noise=noise,
    )
    fd = _residual_jacobian_fd(
        model.lmfit_pars,
        file=file,
        data=data,
        fit_fun_str="fit_model_jax",
        args=args,
        e_lim=e_lim,
        t_lim=t_lim,
        noise=noise,
    )
    np.testing.assert_allclose(jac, fd, rtol=1e-5, atol=1e-8)

    # weighting is exactly a per-row division of the unweighted Jacobian
    jac_raw = fitlib.jacobian_fun(
        model.lmfit_pars,
        file.energy,
        data,
        "fit_model_jax",
        0,
        e_lim,
        t_lim,
        "lmfit",
        args,
    )
    sigma_view = np.asarray(noise.sigma).reshape(-1, 1)
    np.testing.assert_allclose(jac, jac_raw / sigma_view, rtol=1e-12)


#
def test_residual_fun_unknown_noise_is_the_unweighted_residual():
    """``None`` and an ``unknown`` model leave the residual untouched."""

    file, model, args = _make_2d_jax_args()
    data = np.ones((7, 21))
    plain = fitlib.residual_fun(
        model.lmfit_pars, file.energy, data, "fit_model_jax", 0, [], [], "lmfit", args
    )
    declared = fitlib.residual_fun(
        model.lmfit_pars,
        file.energy,
        data,
        "fit_model_jax",
        0,
        [],
        [],
        "lmfit",
        args,
        noise=NoiseModel(kind="unknown"),
    )
    np.testing.assert_array_equal(np.asarray(plain), np.asarray(declared))


# ---------------------------------------------------------------------------
# Segments (joint fits)
# ---------------------------------------------------------------------------


#
def test_segments_concatenate_per_segment_results():
    """Each segment is scored by its own model, then concatenated."""

    gaussian = NoiseModel(kind="gaussian", sigma=np.array([[0.5, 2.0], [1.0, 4.0]]))
    poisson = NoiseModel(kind="poisson", scale=3.0)
    d_gauss = np.array([[1.0, 2.0], [3.0, 4.0]])
    m_gauss = np.array([[0.5, 2.5], [3.5, 1.0]])
    d_pois = np.array([0.0, 5.0, 11.0])
    m_pois = np.array([0.4, 4.2, 12.0])
    data = np.concatenate([d_gauss.ravel(), d_pois])
    model = np.concatenate([m_gauss.ravel(), m_pois])

    segments = SegmentedNoise(models=(gaussian, poisson), lengths=(4, 3))
    expected = np.concatenate(
        [
            (d_gauss - m_gauss).ravel() / np.ravel(np.asarray(gaussian.sigma)),
            poisson.apply(d_pois, m_pois),
        ]
    )
    np.testing.assert_allclose(segments.apply(data, model), expected, rtol=1e-12)
    np.testing.assert_allclose(
        apply_segments((gaussian, poisson), data, model, lengths=(4, 3)),
        expected,
        rtol=1e-12,
    )

    expected_factor = np.concatenate(
        [
            -1.0 / np.ravel(np.asarray(gaussian.sigma)),
            poisson.jacobian_factor(d_pois, m_pois),
        ]
    )
    np.testing.assert_allclose(
        segments.jacobian_factor(data, model), expected_factor, rtol=1e-12
    )
    np.testing.assert_allclose(
        jacobian_factor_segments((gaussian, poisson), data, model, lengths=(4, 3)),
        expected_factor,
        rtol=1e-12,
    )
    assert segments.is_weighted


#
def test_segments_reject_unknown_mixed_with_weighted():
    """A joint residual cannot carry two different units."""

    models = (NoiseModel(kind="unknown"), NoiseModel(kind="poisson", scale=1.0))
    with pytest.raises(ValueError, match="cannot mix"):
        SegmentedNoise(models=models, lengths=(2, 2))
    with pytest.raises(ValueError, match="cannot mix"):
        apply_segments(models, np.ones(4), np.ones(4), lengths=(2, 2))


#
def test_segments_reject_length_mismatch():
    """Segment lengths must cover the concatenated vector exactly."""

    models = (NoiseModel(kind="poisson", scale=1.0),) * 2
    with pytest.raises(ValueError, match="sum to"):
        apply_segments(models, np.ones(4), np.ones(4), lengths=(2, 3))


# ---------------------------------------------------------------------------
# fit_wrapper wiring
# ---------------------------------------------------------------------------


#
def test_fit_wrapper_poisson_recovers_truth():
    """A weighted one-stage leastsq fit runs end to end and finds the truth."""

    file = _make_1d_glp_file()
    model = file.model_active
    args = (model, 1)
    truth = {name: model.lmfit_pars[name].value for name in model.parameter_names}
    clean = _evaluate(model.lmfit_pars, file, np.zeros(101), "fit_model_mcp", args)
    scale = 50.0
    data = np.random.default_rng(3).poisson(np.maximum(clean, 0.0) * scale) / scale

    par_ini = copy.deepcopy(model.lmfit_pars)
    par_ini["GLP_01_A"].value = 16.0
    par_ini["GLP_01_x0"].value = 84.6
    out = fitlib.fit_wrapper(
        const=(file.energy, data, "fit_model_mcp", 0, [], []),
        args=args,
        par_names=model.parameter_names,
        par=par_ini,
        stages=1,
        try_ci=0,
        fit_alg_1="leastsq",
        noise=NoiseModel(kind="poisson", scale=scale),
    )

    for name, true_value in truth.items():
        fitted = out.par_fin.params[name].value
        assert abs(fitted - true_value) < 0.1 * abs(true_value), name
        assert out.par_fin.params[name].stderr > 0


#
def test_fit_wrapper_weighted_covariance_is_not_redchi_scaled():
    """``scale_covar=False`` under declared noise: σ sets the error bars.

    A constant σ cannot change the optimum, so the fitted values are
    identical to the unweighted fit -- but the quoted ``stderr`` is
    ``σ·sqrt(C_ii)`` instead of lmfit's ``sqrt(redchi·C_ii)``. With
    ``scale_covar`` left on, the two would agree exactly.
    """

    file = _make_1d_glp_file()
    model = file.model_active
    args = (model, 1)
    clean = _evaluate(model.lmfit_pars, file, np.zeros(101), "fit_model_mcp", args)
    rng = np.random.default_rng(5)
    sigma = 0.35
    data = clean + sigma * rng.standard_normal(101)

    par_ini = copy.deepcopy(model.lmfit_pars)
    par_ini["GLP_01_A"].value = 16.0
    const = (file.energy, data, "fit_model_mcp", 0, [], [])
    common = {
        "const": const,
        "args": args,
        "par_names": model.parameter_names,
        "stages": 1,
        "try_ci": 0,
        "fit_alg_1": "leastsq",
    }
    weighted = fitlib.fit_wrapper(
        par=copy.deepcopy(par_ini),
        noise=NoiseModel(kind="gaussian", sigma=sigma),
        **common,
    )
    unweighted = fitlib.fit_wrapper(par=copy.deepcopy(par_ini), **common)

    ratio = sigma / np.sqrt(unweighted.par_fin.redchi)
    assert not np.isclose(ratio, 1.0, rtol=1e-3)
    for name in model.parameter_names:
        assert weighted.par_fin.params[name].value == pytest.approx(
            unweighted.par_fin.params[name].value, rel=1e-8
        )
        assert weighted.par_fin.params[name].stderr == pytest.approx(
            unweighted.par_fin.params[name].stderr * ratio, rel=1e-6
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


#
def test_poisson_rejects_negative_data():
    """Counts cannot be negative; the error points at the Gaussian model."""

    noise = NoiseModel(kind="poisson", scale=1.0)
    noise.validate_data(np.array([0.0, 3.0, np.nan]))  # NaN outside a window is legal
    with pytest.raises(ValueError, match="dark-subtracted"):
        noise.validate_data(np.array([1.0, -0.2]))
    NoiseModel(kind="gaussian", sigma=1.0).validate_data(np.array([-1.0, 2.0]))


#
def test_noise_model_rejects_mismatched_knobs():
    """Each kind takes exactly its own knob."""

    with pytest.raises(ValueError, match="sigma belongs to gaussian"):
        NoiseModel(kind="poisson", sigma=0.5)
    with pytest.raises(ValueError, match="scale belongs to poisson"):
        NoiseModel(kind="gaussian", sigma=0.5, scale=2.0)
    with pytest.raises(ValueError, match="requires sigma"):
        NoiseModel(kind="gaussian")
    with pytest.raises(ValueError, match="neither sigma nor scale"):
        NoiseModel(kind="unknown", sigma=0.5)
    with pytest.raises(ValueError, match="noise kind"):
        NoiseModel(kind="gauss")  # type: ignore[arg-type]


#
def test_noise_model_rejects_non_positive_sigma_and_scale():
    """σ must be finite and positive everywhere; so must the Poisson scale."""

    with pytest.raises(ValueError, match="finite and positive"):
        NoiseModel(kind="gaussian", sigma=0.0)
    with pytest.raises(ValueError, match="finite and positive"):
        NoiseModel(kind="gaussian", sigma=np.array([1.0, -2.0]))
    with pytest.raises(ValueError, match="finite and positive"):
        NoiseModel(kind="gaussian", sigma=np.array([1.0, np.nan]))
    with pytest.raises(ValueError, match="finite positive"):
        NoiseModel(kind="poisson", scale=0.0)


#
def test_sigma_array_is_copied_and_frozen():
    """A per-point σ cannot be mutated through the caller's array."""

    sigma = np.array([1.0, 2.0])
    noise = NoiseModel(kind="gaussian", sigma=sigma)
    sigma[0] = 99.0
    np.testing.assert_allclose(np.asarray(noise.sigma), [1.0, 2.0])
    with pytest.raises(ValueError):
        np.asarray(noise.sigma)[0] = 5.0
