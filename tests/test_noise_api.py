"""Declared noise on the File/Project API: declaration, defaults, reductions.

``File.set_noise`` is the single entry point; ``noise_type`` / ``sigma_data``
/ ``sigma_type`` / ``noise_scale`` are read-only views of it. The reductions
each fit applies (baseline mean, ``time_range`` mean, one slice, the 2D
window, one segment per file) are pinned at the ``fitlib.fit_wrapper``
boundary, where the view noise is handed to the optimizer — that is the last
point at which σ and the fitted data view must still line up element for
element.
"""

import copy

import matplotlib
import numpy as np
import pytest
from _utils import make_project

matplotlib.use("Agg")

from trspecfit import File, Project, fitlib  # noqa: E402
from trspecfit.utils.noise import SegmentedNoise  # noqa: E402

_ENERGY_YAML = "models/file_energy.yaml"
_TIME_YAML = "models/file_time.yaml"
_MODEL = "single_glp"
_PROJECT_ENERGY_YAML = "models/project_energy.yaml"
_PROJECT_TIME_YAML = "models/project_time.yaml"


#
def _peak_data(*, n_time=6, n_energy=24, amplitude=20.0, noise_level=0.0, seed=7):
    """``(data, energy, time)`` of a strictly positive peak, rising in time."""

    energy = np.linspace(83.0, 87.0, n_energy)
    time = np.linspace(0.0, 5.0, n_time)
    peak = amplitude * np.exp(-0.5 * ((energy - 85.0) / 0.6) ** 2) + 0.5
    data = peak[None, :] * (1.0 + 0.05 * np.arange(n_time))[:, None]
    if noise_level > 0:
        rng = np.random.default_rng(seed)
        data = data + noise_level * rng.standard_normal(data.shape)
    return data, energy, time


#
def _fit_ready_file(*, project=None, name="noise", n_time=6, noise_level=0.0):
    """File with positive 2D data, a GLP model, a baseline and an e_lim window."""

    project = make_project(name="noise_api") if project is None else project
    data, energy, time = _peak_data(n_time=n_time, noise_level=noise_level)
    file = File(parent_project=project, name=name, data=data, energy=energy, time=time)
    file.load_model(model_yaml=_ENERGY_YAML, model_info=_MODEL)
    file.define_baseline(0, 2, time_type="ind", show_plot=False)
    file.set_fit_limits([84.0, 86.0], show_plot=False)
    return file


#
def _plain_file(*, data=None, project=None, name="plain"):
    """File without a model; ``data=None`` leaves the file dataless."""

    project = make_project(name="noise_api") if project is None else project
    if data is None:
        return File(parent_project=project, name=name)
    return File(
        parent_project=project,
        name=name,
        data=data,
        energy=np.linspace(83.0, 87.0, data.shape[-1]),
        time=np.linspace(0.0, 5.0, data.shape[0]) if data.ndim == 2 else None,
    )


#
def _capture_fit_wrapper(monkeypatch):
    """Record every ``fitlib.fit_wrapper`` call and run the real one."""

    calls: list[dict] = []
    real = fitlib.fit_wrapper

    def _recording(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(fitlib, "fit_wrapper", _recording)
    return calls


#
def _e_slice(file):
    """The ``e_lim`` window as the residual applies it."""

    return slice(file.e_lim[0], file.e_lim[1])


#
def _residual_jacobian_fd(par, *, const, args, noise):
    """Central-difference residual Jacobian, columns in lmfit vary order."""

    columns = []
    for name in [name for name in par if par[name].vary]:
        perturbed = copy.deepcopy(par)
        value = perturbed[name].value
        step = 1e-6 * max(1.0, abs(value))
        perturbed[name].value = value + step
        res_plus = np.asarray(
            fitlib.residual_fun(perturbed, *const, "lmfit", args, noise=noise)
        )
        perturbed[name].value = value - step
        res_minus = np.asarray(
            fitlib.residual_fun(perturbed, *const, "lmfit", args, noise=noise)
        )
        columns.append((res_plus - res_minus) / (2 * step))
    return np.stack(columns, axis=1)


#
#
class TestSetNoiseValidation:
    """``set_noise`` rejects every declaration it cannot honor."""

    #
    def test_poisson_on_negative_data_points_at_gaussian(self):
        file = _plain_file(data=np.array([[1.0, -0.5], [2.0, 3.0]]))
        with pytest.raises(ValueError, match="non-negative data"):
            file.set_noise("poisson")
        # the rejected declaration leaves the previous model in place
        assert file.noise_type == "unknown"

    #
    def test_sigma_with_poisson_raises(self):
        file = _plain_file(data=np.ones((2, 3)))
        with pytest.raises(ValueError, match="poisson noise takes scale"):
            file.set_noise("poisson", sigma=0.5)

    #
    def test_scale_with_gaussian_raises(self):
        file = _plain_file(data=np.ones((2, 3)))
        with pytest.raises(ValueError, match="gaussian noise takes sigma"):
            file.set_noise("gaussian", scale=2.0)

    #
    def test_unknown_kind_raises(self):
        file = _plain_file(data=np.ones((2, 3)))
        with pytest.raises(ValueError, match="noise kind must be one of"):
            file.set_noise("laplace")

    #
    def test_sigma_shape_that_cannot_broadcast_names_both_shapes(self):
        file = _plain_file(data=np.ones((4, 6)))
        with pytest.raises(ValueError, match=r"\(5,\).*\(4, 6\)"):
            file.set_noise("gaussian", sigma=np.ones(5))

    #
    def test_per_point_sigma_broadcasts_a_row_to_the_data_shape(self):
        file = _plain_file(data=np.ones((4, 6)))
        file.set_noise("gaussian", sigma=np.linspace(0.1, 0.6, 6))
        sigma = file.noise.sigma
        assert isinstance(sigma, np.ndarray)  # type guard
        assert sigma.shape == (4, 6)
        # copied and frozen: the declaration does not alias user input
        assert not sigma.flags.writeable

    #
    def test_poisson_before_data_raises(self):
        file = _plain_file()
        with pytest.raises(ValueError, match="load the data first"):
            file.set_noise("poisson")

    #
    def test_poisson_after_data_validates(self):
        file = _plain_file(data=np.ones((3, 4)))
        file.set_noise("poisson", scale=4.0)
        assert file.noise_type == "poisson"
        assert file.noise_scale == pytest.approx(4.0)

    #
    def test_per_point_sigma_before_data_raises(self):
        file = _plain_file()
        with pytest.raises(ValueError, match="load the data first"):
            file.set_noise("gaussian", sigma=np.ones(5))

    #
    def test_constant_sigma_needs_no_data(self):
        file = _plain_file()
        file.set_noise("gaussian", sigma=0.25)
        assert file.sigma_data == pytest.approx(0.25)


#
#
class TestNoiseProperties:
    """``noise_type`` / ``sigma_data`` / ``sigma_type`` / ``noise_scale``."""

    #
    def test_unknown_is_the_default(self):
        file = _plain_file(data=np.ones((3, 4)))
        assert file.noise_type == "unknown"
        assert np.isnan(file.sigma_data)
        assert file.sigma_type == "constant"
        assert np.isnan(file.noise_scale)

    #
    def test_constant_gaussian(self):
        file = _plain_file(data=np.ones((3, 4)))
        file.set_noise("gaussian", sigma=0.4)
        assert file.noise_type == "gaussian"
        assert file.sigma_data == pytest.approx(0.4)
        assert file.sigma_type == "constant"
        assert np.isnan(file.noise_scale)

    #
    def test_per_point_gaussian_reports_nan_sigma_data(self):
        file = _plain_file(data=np.ones((3, 4)))
        file.set_noise("gaussian", sigma=np.full((3, 4), 0.4))
        assert file.noise_type == "gaussian"
        assert file.sigma_type == "per_point"
        assert np.isnan(file.sigma_data)

    #
    def test_poisson(self):
        file = _plain_file(data=np.ones((3, 4)))
        file.set_noise("poisson", scale=2.5)
        assert file.noise_type == "poisson"
        assert file.noise_scale == pytest.approx(2.5)
        assert np.isnan(file.sigma_data)
        assert file.sigma_type == "constant"

    #
    def test_poisson_scale_defaults_to_one(self):
        file = _plain_file(data=np.ones((3, 4)))
        file.set_noise("poisson")
        assert file.noise_scale == pytest.approx(1.0)

    #
    @pytest.mark.parametrize(
        "attr", ["noise_type", "sigma_data", "sigma_type", "noise_scale"]
    )
    def test_properties_are_read_only(self, attr):
        file = _plain_file(data=np.ones((3, 4)))
        with pytest.raises(AttributeError):
            setattr(file, attr, "gaussian")

    #
    def test_set_sigma_delegates_to_set_noise(self):
        file = _plain_file(data=np.ones((3, 4)))
        previous = file.set_sigma(0.3)
        assert previous is None
        assert (file.noise.kind, file.noise.sigma, file.noise.scale) == (
            "gaussian",
            pytest.approx(0.3),
            None,
        )
        previous = file.set_sigma(None)
        assert previous == pytest.approx(0.3)
        assert (file.noise.kind, file.noise.sigma, file.noise.scale) == (
            "unknown",
            None,
            None,
        )

    #
    def test_set_sigma_rejects_a_contradicting_noise_type(self):
        file = _plain_file(data=np.ones((3, 4)))
        with pytest.raises(ValueError, match="contradicts"):
            file.set_sigma(0.3, noise_type="unknown")

    #
    def test_set_sigma_rejects_per_point_sigma_type(self):
        file = _plain_file(data=np.ones((3, 4)))
        with pytest.raises(ValueError, match="per-point"):
            file.set_sigma(0.3, sigma_type="per_point")


#
#
class TestProjectNoiseDefaults:
    """Project-level defaults reach every File built under them."""

    #
    def test_poisson_default_propagates_and_validates(self, tmp_path):
        config = tmp_path / "project.yaml"
        config.write_text("show_output: 0\nnoise_type: poisson\nnoise_scale: 3.0\n")
        project = Project(path=tmp_path, config_file="project.yaml")
        file = _plain_file(data=np.ones((3, 4)), project=project, name="ok")
        assert file.noise_type == "poisson"
        assert file.noise_scale == pytest.approx(3.0)
        with pytest.raises(ValueError, match="non-negative data"):
            _plain_file(data=-np.ones((3, 4)), project=project, name="bad")

    #
    def test_poisson_default_from_attributes(self):
        project = make_project(name="noise_defaults")
        project.noise_type = "poisson"
        file = _plain_file(data=np.ones((3, 4)), project=project)
        assert file.noise_type == "poisson"
        assert file.noise_scale == pytest.approx(1.0)

    #
    def test_gaussian_default_without_sigma_raises(self):
        project = make_project(name="noise_defaults")
        project.noise_type = "gaussian"
        with pytest.raises(ValueError, match="finite positive sigma_data"):
            _plain_file(data=np.ones((3, 4)), project=project)

    #
    def test_gaussian_default_with_sigma_propagates(self, tmp_path):
        config = tmp_path / "project.yaml"
        config.write_text("show_output: 0\nnoise_type: gaussian\nsigma_data: 0.7\n")
        project = Project(path=tmp_path, config_file="project.yaml")
        file = _plain_file(data=np.ones((3, 4)), project=project)
        assert file.noise_type == "gaussian"
        assert file.sigma_data == pytest.approx(0.7)

    #
    def test_sigma_without_gaussian_raises_at_config_load(self, tmp_path):
        config = tmp_path / "project.yaml"
        config.write_text("show_output: 0\nsigma_data: 0.7\n")
        with pytest.raises(ValueError, match="sigma_data is set but noise_type"):
            Project(path=tmp_path, config_file="project.yaml")

    #
    def test_noise_scale_without_poisson_raises(self):
        project = make_project(name="noise_defaults")
        project.noise_scale = 2.0
        with pytest.raises(ValueError, match="noise_scale belongs to"):
            _plain_file(data=np.ones((3, 4)), project=project)


#
#
class TestCorrectionsUnderDeclaredNoise:
    """Corrections refuse to run while a weighted model is declared."""

    #
    @staticmethod
    def _corrections(file):
        n_energy = file.energy.shape[0]
        return [
            ("subtract_dark", lambda: file.subtract_dark(np.full(n_energy, 0.1))),
            ("calibrate_data", lambda: file.calibrate_data(np.full(n_energy, 2.0))),
            ("reset_dark", file.reset_dark),
            ("reset_calibration", file.reset_calibration),
        ]

    #
    @pytest.mark.parametrize(
        "declare",
        [
            lambda f: f.set_noise("gaussian", sigma=0.3),
            lambda f: f.set_noise("poisson", scale=2.0),
        ],
    )
    def test_all_four_raise_and_work_again_after_unknown(self, declare):
        file = _plain_file(data=np.ones((4, 6)))
        declare(file)
        for name, correction in self._corrections(file):
            with pytest.raises(ValueError, match=f"{name}.*set_noise"):
                correction()
        # data untouched by the refused corrections
        np.testing.assert_allclose(file.data, np.ones((4, 6)))

        file.set_noise("unknown")
        for _name, correction in self._corrections(file):
            correction()
        np.testing.assert_allclose(file.data, np.ones((4, 6)))


#
#
class TestViewReductions:
    """σ handed to the optimizer matches the data view the fit residual sees."""

    #
    def test_unknown_passes_no_noise(self, monkeypatch):
        file = _fit_ready_file()
        calls = _capture_fit_wrapper(monkeypatch)
        file.fit_baseline(model_name=_MODEL, stages=1, try_ci=0)
        assert len(calls) == 1
        assert calls[0]["noise"] is None

    #
    def test_baseline_constant_sigma_scales_with_sqrt_n(self, monkeypatch):
        file = _fit_ready_file()
        file.set_sigma(0.4)
        n_avg = file.base_t_ind[1] - file.base_t_ind[0]
        assert n_avg > 1
        calls = _capture_fit_wrapper(monkeypatch)
        file.fit_baseline(model_name=_MODEL, stages=1, try_ci=0)
        noise = calls[0]["noise"]
        assert noise.kind == "gaussian"
        assert noise.sigma == pytest.approx(0.4 / np.sqrt(n_avg))

    #
    def test_baseline_per_point_sigma_adds_in_quadrature(self, monkeypatch):
        file = _fit_ready_file()
        rng = np.random.default_rng(3)
        sigma_full = 0.2 + 0.3 * rng.random(file.data.shape)
        file.set_noise("gaussian", sigma=sigma_full)
        lo, hi = file.base_t_ind
        n_avg = hi - lo
        calls = _capture_fit_wrapper(monkeypatch)
        file.fit_baseline(model_name=_MODEL, stages=1, try_ci=0)
        expected = (
            np.sqrt(np.sum(sigma_full[lo:hi, _e_slice(file)] ** 2, axis=0)) / n_avg
        )
        np.testing.assert_allclose(calls[0]["noise"].sigma, expected)

    #
    def test_baseline_poisson_scale_grows_with_n(self, monkeypatch):
        file = _fit_ready_file()
        file.set_noise("poisson", scale=2.0)
        n_avg = file.base_t_ind[1] - file.base_t_ind[0]
        calls = _capture_fit_wrapper(monkeypatch)
        file.fit_baseline(model_name=_MODEL, stages=1, try_ci=0)
        noise = calls[0]["noise"]
        assert noise.kind == "poisson"
        assert noise.scale == pytest.approx(2.0 * n_avg)

    #
    def test_spectrum_time_range_reduces_like_the_baseline(self, monkeypatch):
        file = _fit_ready_file()
        file.set_sigma(0.4)
        calls = _capture_fit_wrapper(monkeypatch)
        file.fit_spectrum(
            _MODEL,
            time_range=(0, 2),
            time_type="ind",
            stages=1,
            try_ci=0,
            show_plot=False,
        )
        n_avg = file.spec_t_ind[1] - file.spec_t_ind[0]
        assert n_avg > 1
        assert calls[0]["noise"].sigma == pytest.approx(0.4 / np.sqrt(n_avg))

    #
    def test_spectrum_time_point_takes_that_row(self, monkeypatch):
        file = _fit_ready_file()
        rng = np.random.default_rng(5)
        sigma_full = 0.2 + 0.3 * rng.random(file.data.shape)
        file.set_noise("gaussian", sigma=sigma_full)
        calls = _capture_fit_wrapper(monkeypatch)
        file.fit_spectrum(
            _MODEL, time_point=3, time_type="ind", stages=1, try_ci=0, show_plot=False
        )
        np.testing.assert_allclose(
            calls[0]["noise"].sigma, sigma_full[3, _e_slice(file)]
        )

    #
    def test_slice_by_slice_takes_one_row_per_slice(self, monkeypatch):
        file = _fit_ready_file(n_time=4)
        file.p.spec_fun_str = "fit_model_mcp"
        rng = np.random.default_rng(9)
        sigma_full = 0.2 + 0.3 * rng.random(file.data.shape)
        file.set_noise("gaussian", sigma=sigma_full)
        calls = _capture_fit_wrapper(monkeypatch)
        file.fit_slice_by_slice(
            _MODEL, n_workers=1, seed_source="model", seed_adapt=None, try_ci=0
        )
        assert len(calls) == file.data.shape[0]
        for s_i, call in enumerate(calls):
            np.testing.assert_allclose(
                call["noise"].sigma, sigma_full[s_i, _e_slice(file)]
            )

    #
    @pytest.mark.slow
    def test_slice_by_slice_parallel_matches_serial(self):
        """The per-slice σ reaches worker processes too.

        A per-point σ varies within a row, so it moves the minimum: equal
        fitted values across the serial and parallel paths mean the same
        weighting was applied in both.
        """

        fitted = {}
        for n_workers in (1, 2):
            file = _fit_ready_file(name=f"sbs_{n_workers}", n_time=4)
            file.p.spec_fun_str = "fit_model_mcp"
            rng = np.random.default_rng(9)
            sigma_full = 0.2 + 0.3 * rng.random(file.data.shape)
            file.set_noise("gaussian", sigma=sigma_full)
            file.fit_slice_by_slice(
                _MODEL,
                n_workers=n_workers,
                seed_source="model",
                seed_adapt=None,
                try_ci=0,
            )
            fitted[n_workers] = [
                [par.value for par in out.par_fin.params.values()]
                for out in file.results_sbs
            ]
        np.testing.assert_allclose(fitted[1], fitted[2], rtol=1e-6)

    #
    def test_fit_2d_takes_the_2d_window(self, monkeypatch):
        file = _fit_ready_file(n_time=6)
        file.fit_baseline(model_name=_MODEL, stages=1, try_ci=0)
        file.add_time_dependence(
            target_model=_MODEL,
            target_parameter="GLP_01_A",
            dynamics_yaml=_TIME_YAML,
            dynamics_model=["MonoExpPos"],
        )
        file.set_fit_limits([84.0, 86.0], time_limits=[1.0, 4.0], show_plot=False)
        rng = np.random.default_rng(13)
        sigma_full = 0.2 + 0.3 * rng.random(file.data.shape)
        file.set_noise("gaussian", sigma=sigma_full)
        calls = _capture_fit_wrapper(monkeypatch)
        file.fit_2d(_MODEL, stages=1, try_ci=0)
        t_slice = slice(file.t_lim[0], file.t_lim[1])
        expected = sigma_full[t_slice, _e_slice(file)]
        assert expected.ndim == 2
        np.testing.assert_allclose(calls[0]["noise"].sigma, expected)

    #
    def test_baseline_set_by_hand_refuses_weighted_noise(self):
        data, _, _ = _peak_data()
        file = _plain_file(data=data)
        file.load_model(model_yaml=_ENERGY_YAML, model_info=_MODEL)
        file.data_base = data.mean(axis=0)
        file.set_fit_limits([84.0, 86.0], show_plot=False)
        file.set_sigma(0.4)
        with pytest.raises(ValueError, match="define_baseline"):
            file.fit_baseline(model_name=_MODEL, stages=1, try_ci=0)
        file.set_noise("unknown")
        file.fit_baseline(model_name=_MODEL, stages=1, try_ci=0)


#
#
class TestJointNoise:
    """``Project.fit_2d`` carries one segment per file, in ``Project.files`` order."""

    #
    @staticmethod
    def _joint_project(*, spec_fun_str="fit_model_gir"):
        """Two-file project ready for a joint 2D fit.

        The files are registered in the order ``b_file``, ``a_file`` and get
        different time windows, so a segment list built in sorted-name order
        instead of ``Project.files`` order fails on the lengths.
        """

        project = make_project(name="noise_joint", spec_fun_str=spec_fun_str)
        windows = {"b_file": [0.0, 5.0], "a_file": [1.0, 4.0]}
        for name, t_window in windows.items():
            data, energy, time = _peak_data(n_time=8, n_energy=16)
            file = File(
                parent_project=project,
                name=name,
                data=data,
                energy=energy,
                time=time,
            )
            file.load_model(
                model_yaml=_PROJECT_ENERGY_YAML, model_info="project_glp_base"
            )
            file.define_baseline(0, 1, time_type="ind", show_plot=False)
            file.set_fit_limits([84.0, 86.0], time_limits=t_window, show_plot=False)
            file.fit_baseline(model_name="project_glp_base", stages=1, try_ci=0)
            file.load_model(model_yaml=_PROJECT_ENERGY_YAML, model_info="project_glp")
            file.add_time_dependence(
                target_model="project_glp",
                target_parameter="GLP_01_x0",
                dynamics_yaml=_PROJECT_TIME_YAML,
                dynamics_model=["MonoExpProject"],
            )
        return project

    #
    @staticmethod
    def _window_sizes(project):
        return [
            int(f.data[fitlib._fit_window_slices(2, f.e_lim, f.t_lim)].size)
            for f in project.files
        ]

    #
    def test_mixed_unknown_and_weighted_raises_before_fitting(self, monkeypatch):
        project = self._joint_project()
        project.files[0].set_sigma(0.3)
        calls = _capture_fit_wrapper(monkeypatch)
        with pytest.raises(ValueError, match="cannot mix 'unknown'"):
            project.fit_2d(model_name="project_glp", stages=1, try_ci=0)
        assert calls == []

    #
    def test_gaussian_plus_poisson_runs_with_per_file_segments(self, monkeypatch):
        project = self._joint_project()
        project.files[0].set_sigma(0.3)
        project.files[1].set_noise("poisson", scale=2.0)
        calls = _capture_fit_wrapper(monkeypatch)
        project.fit_2d(model_name="project_glp", stages=1, try_ci=0)
        noise = calls[0]["noise"]
        assert isinstance(noise, SegmentedNoise)
        assert list(noise.lengths) == self._window_sizes(project)
        assert [model.kind for model in noise.models] == ["gaussian", "poisson"]

    #
    def test_segmented_jacobian_matches_finite_differences(self, monkeypatch):
        pytest.importorskip("jax")
        project = self._joint_project(spec_fun_str="fit_model_jax")
        project.files[0].set_sigma(0.3)
        project.files[1].set_noise("poisson", scale=2.0)
        calls = _capture_fit_wrapper(monkeypatch)
        project.fit_2d(model_name="project_glp", stages=1, try_ci=0)
        call = calls[0]
        assert call["jac_fun"] is fitlib.jacobian_fun_project
        jac = call["jac_fun"](
            call["par"], *call["const"], "lmfit", call["args"], noise=call["noise"]
        )
        fd = _residual_jacobian_fd(
            call["par"], const=call["const"], args=call["args"], noise=call["noise"]
        )
        assert jac.shape == fd.shape
        np.testing.assert_allclose(jac, fd, rtol=1e-5, atol=1e-8)


#
#
class TestWeightedUncertainty:
    """A declared σ sets the quoted uncertainty, the fitted values do not move."""

    #
    def test_doubling_sigma_doubles_stderr(self):
        values = []
        errors = []
        for sigma in (0.2, 0.4):
            file = _fit_ready_file(name=f"sigma_{sigma}", noise_level=0.05)
            file.set_sigma(sigma)
            file.fit_spectrum(
                _MODEL,
                time_point=2,
                time_type="ind",
                stages=1,
                fit_alg_1="leastsq",
                try_ci=0,
                show_plot=False,
            )
            params = file.model_spec.result.par_fin.params
            varying = [name for name in params if params[name].vary]
            values.append([params[name].value for name in varying])
            errors.append([params[name].stderr for name in varying])

        np.testing.assert_allclose(values[0], values[1], rtol=1e-6)
        assert all(err is not None for err in errors[0] + errors[1])
        np.testing.assert_allclose(
            np.asarray(errors[1]) / np.asarray(errors[0]), 2.0, rtol=1e-4
        )
