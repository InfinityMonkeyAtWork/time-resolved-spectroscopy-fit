"""Tests for File.fit_spectrum() — fit 1D model at a selected time point/range."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from _utils import make_project, simulate_clean

from trspecfit import File, FitResults


#
def _make_truth_file(project):
    """Create file with single GLP peak + exponential dynamics on amplitude."""

    energy = np.linspace(83, 87, 30)
    time = np.linspace(-2, 10, 24)

    file = File(parent_project=project, name="truth", energy=energy, time=time)

    file.load_model(
        model_yaml="models/file_energy.yaml",
        model_info="single_glp",
    )
    file.add_time_dependence(
        target_model="single_glp",
        target_parameter="GLP_01_A",
        dynamics_yaml="models/file_time.yaml",
        dynamics_model=["MonoExpPos"],
    )
    return file


#
def _make_fit_file_1d(project, *, name="fit_1d"):
    """A 1D file holding the truth model's first (pre-trigger) spectrum."""

    truth_file = _make_truth_file(project)
    clean = simulate_clean(truth_file.model_active)
    file = File(
        parent_project=project,
        name=name,
        data=clean[0].copy(),
        energy=truth_file.energy.copy(),
    )
    file.load_model(
        model_yaml="models/file_energy.yaml",
        model_info="single_glp",
    )
    return file


#
def _make_fit_file(project, data, energy, time):
    """Create a fresh file loaded with simulated data, ready to fit."""

    file = File(
        parent_project=project,
        name="fit",
        data=data,
        energy=energy.copy(),
        time=time.copy(),
    )

    file.load_model(
        model_yaml="models/file_energy.yaml",
        model_info="single_glp",
    )
    return file


#
#
class TestFitSpectrumErrors:
    """Validation errors for fit_spectrum()."""

    #
    def test_1d_file_with_time_selection_raises(self):
        """A 1D file has no time axis to select from."""

        project = make_project(name="fit_spectrum")
        file = _make_fit_file_1d(project)
        with pytest.raises(ValueError, match="no time axis to select"):
            file.fit_spectrum("single_glp", time_point=0.0)
        with pytest.raises(ValueError, match="no time axis to select"):
            file.fit_spectrum("single_glp", time_range=(0.0, 1.0))

    #
    def test_no_time_selection_raises(self):
        """fit_spectrum raises ValueError if neither time_point nor time_range given."""

        project = make_project(name="fit_spectrum")
        energy = np.linspace(83, 87, 30)
        time = np.linspace(-2, 10, 24)
        data = np.random.default_rng(42).normal(size=(len(time), len(energy)))
        file = File(
            parent_project=project,
            name="err_no_time",
            data=data,
            energy=energy,
            time=time,
        )
        file.load_model(
            model_yaml="models/file_energy.yaml",
            model_info="single_glp",
        )
        with pytest.raises(ValueError, match="time_point or time_range"):
            file.fit_spectrum("single_glp")

    #
    def test_both_time_point_and_range_raises(self):
        """fit_spectrum raises ValueError if both time_point and time_range given."""

        project = make_project(name="fit_spectrum")
        energy = np.linspace(83, 87, 30)
        time = np.linspace(-2, 10, 24)
        data = np.random.default_rng(42).normal(size=(len(time), len(energy)))
        file = File(
            parent_project=project,
            name="err_both",
            data=data,
            energy=energy,
            time=time,
        )
        file.load_model(
            model_yaml="models/file_energy.yaml",
            model_info="single_glp",
        )
        with pytest.raises(ValueError, match="mutually exclusive"):
            file.fit_spectrum("single_glp", time_point=1.0, time_range=(0.0, 2.0))

    #
    def test_2d_model_raises(self):
        """fit_spectrum raises ValueError for a model with time dependence (dim=2)."""

        project = make_project(name="fit_spectrum")
        truth_file = _make_truth_file(project)
        clean = simulate_clean(truth_file.model_active)

        fit_file = _make_fit_file(project, clean, truth_file.energy, truth_file.time)
        fit_file.add_time_dependence(
            target_model="single_glp",
            target_parameter="GLP_01_A",
            dynamics_yaml="models/file_time.yaml",
            dynamics_model=["MonoExpPos"],
        )
        with pytest.raises(ValueError, match="dim=2"):
            fit_file.fit_spectrum("single_glp", time_point=5.0)

    #
    def test_time_point_out_of_range_raises(self):
        """fit_spectrum raises ValueError for a time_point beyond the time axis."""

        project = make_project(name="fit_spectrum")
        truth_file = _make_truth_file(project)
        clean = simulate_clean(truth_file.model_active)

        fit_file = _make_fit_file(project, clean, truth_file.energy, truth_file.time)
        with pytest.raises(ValueError, match="out-of-range"):
            fit_file.fit_spectrum("single_glp", time_point=999.0)

    #
    def test_time_point_ind_out_of_range_raises(self):
        """fit_spectrum raises ValueError for an index beyond the time axis."""

        project = make_project(name="fit_spectrum")
        truth_file = _make_truth_file(project)
        clean = simulate_clean(truth_file.model_active)

        fit_file = _make_fit_file(project, clean, truth_file.energy, truth_file.time)
        with pytest.raises(ValueError, match="out-of-range"):
            fit_file.fit_spectrum("single_glp", time_point=100, time_type="ind")

    #
    def test_reversed_time_range_raises(self):
        """fit_spectrum raises ValueError for a reversed time_range (start > stop)."""

        project = make_project(name="fit_spectrum")
        truth_file = _make_truth_file(project)
        clean = simulate_clean(truth_file.model_active)

        fit_file = _make_fit_file(project, clean, truth_file.energy, truth_file.time)
        with pytest.raises(ValueError, match="empty or out-of-range"):
            fit_file.fit_spectrum("single_glp", time_range=(8.0, 2.0))


#
#
class TestFitSpectrum1D:
    """A 1D file is a single spectrum: fit_spectrum fits it as is."""

    #
    def test_1d_file_fits_directly(self):
        """No time selection; the slot records none and the fit is exact."""

        project = make_project(name="fit_spectrum")
        file = _make_fit_file_1d(project)
        file.fit_spectrum("single_glp", stages=2, try_ci=0, show_plot=False)

        assert file.model_spec is not None  # type guard
        assert file.model_spec.result is not None
        assert file.spec_t_ind == []
        assert file.spec_t_abs == []
        slot = project.results.get(
            file="fit_1d", model="single_glp", fit_type="spectrum"
        )
        assert slot.selection["time_point"] is None
        assert slot.selection["time_range"] is None
        np.testing.assert_allclose(slot.fit, slot.observed, rtol=1e-3, atol=1e-6)
        params = file.get_parameters(fit_type="spectrum")
        assert not params.empty

    #
    def test_1d_file_respects_fit_limits(self):
        """The energy window crops the recorded spectrum like any other fit."""

        project = make_project(name="fit_spectrum")
        file = _make_fit_file_1d(project)
        file.set_fit_limits([84.0, 86.0], show_plot=False)
        file.fit_spectrum("single_glp", stages=1, try_ci=0, show_plot=False)

        slot = project.results.get(
            file="fit_1d", model="single_glp", fit_type="spectrum"
        )
        assert slot.observed.shape[0] == file.e_lim[1] - file.e_lim[0]
        assert slot.observed.shape[0] < len(file.energy)

    #
    def test_1d_file_under_declared_noise(self):
        """A per-point σ weights the 1D fit without any row selection."""

        project = make_project(name="fit_spectrum")
        file = _make_fit_file_1d(project)
        file.set_noise("gaussian", sigma=np.full(len(file.energy), 0.1))
        file.fit_spectrum("single_glp", stages=1, try_ci=0, show_plot=False)

        assert file.model_spec is not None  # type guard
        assert file.model_spec.result is not None  # type guard
        fin = file.model_spec.result.par_fin.params
        varying = [name for name in fin if fin[name].vary]
        assert varying
        assert all(np.isfinite(fin[name].stderr) for name in varying)
        slot = project.results.get(
            file="fit_1d", model="single_glp", fit_type="spectrum"
        )
        assert slot.sigma is not None  # type guard
        assert slot.sigma.shape == slot.observed.shape
        assert "chi2" in slot.metrics

    #
    def test_1d_file_round_trips_through_the_archive(self, tmp_path):
        """A 1D spectrum slot saves and loads like a 2D one."""

        project = make_project(name="fit_spectrum")
        file = _make_fit_file_1d(project)
        file.fit_spectrum("single_glp", stages=1, try_ci=0, show_plot=False)
        archive_path = tmp_path / "spectrum_1d.fit.h5"
        project.save_fits(archive_path, show_output=0)

        loaded = FitResults.load(archive_path)
        assert len(loaded) == 1
        pd.testing.assert_frame_equal(
            loaded.get_parameters(
                file="fit_1d", model="single_glp", fit_type="spectrum"
            ),
            project.results.get_parameters(
                file="fit_1d", model="single_glp", fit_type="spectrum"
            ),
        )
        slot = loaded.get(file="fit_1d", model="single_glp", fit_type="spectrum")
        assert slot.selection["time_point"] is None
        assert slot.selection["time_range"] is None


#
#
class TestFitSpectrumTimePoint:
    """Fit individual spectrum at a single time point."""

    #
    @pytest.mark.slow
    def test_time_point_abs_recovery(self):
        """Fit at a time_point (abs) recovers the 1D spectrum parameters."""

        project = make_project(name="fit_spectrum")
        truth_file = _make_truth_file(project)
        clean = simulate_clean(truth_file.model_active)

        fit_file = _make_fit_file(project, clean, truth_file.energy, truth_file.time)
        fit_file.fit_spectrum(
            "single_glp",
            time_point=5.0,
            stages=2,
            try_ci=0,
            show_plot=False,
        )

        assert fit_file.model_spec is not None
        assert fit_file.data_spec is not None
        assert len(fit_file.spec_t_abs) == 2
        assert len(fit_file.spec_t_ind) == 2
        # single time point: both bounds should be equal
        assert fit_file.spec_t_abs[0] == fit_file.spec_t_abs[1]

    #
    @pytest.mark.slow
    def test_time_point_ind(self):
        """Fit at a time_point using index addressing."""

        project = make_project(name="fit_spectrum")
        truth_file = _make_truth_file(project)
        clean = simulate_clean(truth_file.model_active)

        fit_file = _make_fit_file(project, clean, truth_file.energy, truth_file.time)
        fit_file.fit_spectrum(
            "single_glp",
            time_point=10,
            time_type="ind",
            stages=1,
            try_ci=0,
            show_plot=False,
        )

        assert fit_file.model_spec is not None
        assert fit_file.spec_t_ind == [10, 11]


#
#
class TestFitSpectrumTimeRange:
    """Fit individual spectrum averaged over a time range."""

    #
    @pytest.mark.slow
    def test_time_range_abs(self):
        """Fit averaged spectrum over a time range (abs)."""

        project = make_project(name="fit_spectrum")
        truth_file = _make_truth_file(project)
        clean = simulate_clean(truth_file.model_active)

        fit_file = _make_fit_file(project, clean, truth_file.energy, truth_file.time)
        fit_file.fit_spectrum(
            "single_glp",
            time_range=(2.0, 6.0),
            stages=2,
            try_ci=0,
            show_plot=False,
        )

        assert fit_file.model_spec is not None
        assert fit_file.data_spec is not None
        # bounds should be close to requested range (within one grid step)
        grid_step = np.diff(truth_file.time).mean()
        assert fit_file.spec_t_abs[0] <= 2.0 + grid_step
        assert fit_file.spec_t_abs[1] >= 6.0 - grid_step

    #
    @pytest.mark.slow
    def test_time_range_ind(self):
        """Fit averaged spectrum over a time range using indices."""

        project = make_project(name="fit_spectrum")
        truth_file = _make_truth_file(project)
        clean = simulate_clean(truth_file.model_active)

        fit_file = _make_fit_file(project, clean, truth_file.energy, truth_file.time)
        fit_file.fit_spectrum(
            "single_glp",
            time_range=(5, 10),
            time_type="ind",
            stages=1,
            try_ci=0,
            show_plot=False,
        )

        assert fit_file.model_spec is not None
        assert fit_file.spec_t_ind == [5, 11]

    #
    @pytest.mark.slow
    def test_data_spec_matches_manual_average(self):
        """Extracted spectrum matches manual np.mean over the same range."""

        project = make_project(name="fit_spectrum")
        truth_file = _make_truth_file(project)
        clean = simulate_clean(truth_file.model_active)

        fit_file = _make_fit_file(project, clean, truth_file.energy, truth_file.time)
        fit_file.fit_spectrum(
            "single_glp",
            time_range=(3, 7),
            time_type="ind",
            stages=1,
            try_ci=0,
            show_plot=False,
        )

        expected = np.mean(clean[3:8, :], axis=0)
        np.testing.assert_array_equal(fit_file.data_spec, expected)
