"""Tests for File.fit_spectrum() — fit 1D model at a selected time point/range."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from trspecfit import File, Project, Simulator

pytestmark = pytest.mark.slow


#
def _make_project():
    """Create a silent project pointing to tests/ for YAML access."""

    project = Project(path="tests", name="fit_spectrum")
    project.show_output = 0
    return project


#
def _make_truth_file(project):
    """Create file with single GLP peak + exponential dynamics on amplitude."""

    energy = np.linspace(83, 87, 30)
    time = np.linspace(-2, 10, 24)

    file = File(parent_project=project, name="truth")
    file.energy = energy
    file.time = time
    file.dim = 2

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
def _extract_par_dict(model):
    """Return {name: value} for all non-expression parameters."""

    return {
        name: model.lmfit_pars[name].value
        for name in model.parameter_names
        if model.lmfit_pars[name].expr is None
    }


#
def _simulate_clean(truth_file):
    """Generate noiseless 2D data from the truth model."""

    sim = Simulator(
        model=truth_file.model_active,
        detection="analog",
        noise_level=0.0,
        noise_type="none",
        seed=42,
    )
    clean, _, _ = sim.simulate_2d()
    return clean


#
#
class TestFitSpectrumErrors:
    """Validation errors for fit_spectrum()."""

    #
    def test_1d_data_raises(self):
        """fit_spectrum raises ValueError for 1D data."""

        project = _make_project()
        file = File(
            parent_project=project,
            name="err_1d",
            data=np.ones(50),
            energy=np.linspace(80, 90, 50),
        )
        file.load_model(
            model_yaml="models/file_energy.yaml",
            model_info="single_glp",
        )
        with pytest.raises(ValueError, match="2D data"):
            file.fit_spectrum("single_glp", time_point=0.0)

    #
    def test_no_time_selection_raises(self):
        """fit_spectrum raises ValueError if neither time_point nor time_range given."""

        project = _make_project()
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

        project = _make_project()
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

        project = _make_project()
        truth_file = _make_truth_file(project)
        clean = _simulate_clean(truth_file)

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

        project = _make_project()
        truth_file = _make_truth_file(project)
        clean = _simulate_clean(truth_file)

        fit_file = _make_fit_file(project, clean, truth_file.energy, truth_file.time)
        with pytest.raises(ValueError, match="out-of-range"):
            fit_file.fit_spectrum("single_glp", time_point=999.0)

    #
    def test_time_point_ind_out_of_range_raises(self):
        """fit_spectrum raises ValueError for an index beyond the time axis."""

        project = _make_project()
        truth_file = _make_truth_file(project)
        clean = _simulate_clean(truth_file)

        fit_file = _make_fit_file(project, clean, truth_file.energy, truth_file.time)
        with pytest.raises(ValueError, match="out-of-range"):
            fit_file.fit_spectrum("single_glp", time_point=100, time_type="ind")

    #
    def test_reversed_time_range_raises(self):
        """fit_spectrum raises ValueError for a reversed time_range (start > stop)."""

        project = _make_project()
        truth_file = _make_truth_file(project)
        clean = _simulate_clean(truth_file)

        fit_file = _make_fit_file(project, clean, truth_file.energy, truth_file.time)
        with pytest.raises(ValueError, match="empty or out-of-range"):
            fit_file.fit_spectrum("single_glp", time_range=(8.0, 2.0))


#
#
class TestFitSpectrumTimePoint:
    """Fit individual spectrum at a single time point."""

    #
    def test_time_point_abs_recovery(self):
        """Fit at a time_point (abs) recovers the 1D spectrum parameters."""

        project = _make_project()
        truth_file = _make_truth_file(project)
        clean = _simulate_clean(truth_file)

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
    def test_time_point_ind(self):
        """Fit at a time_point using index addressing."""

        project = _make_project()
        truth_file = _make_truth_file(project)
        clean = _simulate_clean(truth_file)

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
    def test_time_range_abs(self):
        """Fit averaged spectrum over a time range (abs)."""

        project = _make_project()
        truth_file = _make_truth_file(project)
        clean = _simulate_clean(truth_file)

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
    def test_time_range_ind(self):
        """Fit averaged spectrum over a time range using indices."""

        project = _make_project()
        truth_file = _make_truth_file(project)
        clean = _simulate_clean(truth_file)

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
    def test_data_spec_matches_manual_average(self):
        """Extracted spectrum matches manual np.mean over the same range."""

        project = _make_project()
        truth_file = _make_truth_file(project)
        clean = _simulate_clean(truth_file)

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
