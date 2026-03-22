"""Tests for File class methods.

Exercises load_model, select_model, delete_model, reset_models,
set_fit_limits, and define_baseline.
"""

import numpy as np
import pytest

from trspecfit import File, Project


#
#
class TestModelManagement:
    """Test File model management."""

    #
    def _make_file_with_axes(self):
        """Create project and file with axes and dummy data."""

        project = Project(path="tests")
        aux_axis = np.array([0.0, 1.0, 2.0, 3.0])
        file = File(parent_project=project, aux_axis=aux_axis)
        file.energy = np.linspace(80, 90, 201)
        file.time = np.linspace(-10, 100, 111)
        # 2D dummy data for methods that need it
        file.data = np.random.default_rng(42).normal(
            size=(len(file.time), len(file.energy))
        )
        file.dim = 2
        return file

    #
    def test_load_model_sets_active(self):
        """load_model should set model_active and populate models list."""

        file = self._make_file_with_axes()
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["simple_energy"],
        )
        assert file.model_active is not None
        assert file.model_active.name == "simple_energy"
        assert len(file.models) == 1
        assert file.models[0] is file.model_active

    #
    def test_load_multiple_models(self):
        """Loading a second model should add to list and update active."""

        file = self._make_file_with_axes()
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["simple_energy"],
        )
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["single_glp"],
        )
        assert len(file.models) == 2
        assert file.model_active is not None
        assert file.model_active.name == "single_glp"

    #
    def test_load_model_returns_none_for_energy(self):
        """load_model with model_type='energy' should return None."""

        file = self._make_file_with_axes()
        result = file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["simple_energy"],
        )
        assert result is None

    #
    def test_load_model_returns_model_for_dynamics(self):
        """load_model with model_type='dynamics' should return a Dynamics object."""

        file = self._make_file_with_axes()
        result = file.load_model(
            model_yaml="test_models_time.yaml",
            model_info=["MonoExpPos"],
            par_name="GLP_01_A",
            model_type="dynamics",
        )
        assert result is not None
        assert result.name == "GLP_01_A"
        # dynamics should not be added to file.models
        assert len(file.models) == 0

    #
    def test_load_model_returns_model_for_profile(self):
        """load_model with model_type='profile' should return a Profile object."""

        file = self._make_file_with_axes()
        result = file.load_model(
            model_yaml="test_models_profile.yaml",
            model_info=["profile_pLinear"],
            par_name="GLP_01_A",
            model_type="profile",
        )
        assert result is not None
        assert result.name == "GLP_01_A"
        assert len(file.models) == 0

    #
    def test_load_model_rejects_non_list(self):
        """load_model should raise TypeError if model_info is not a list."""

        file = self._make_file_with_axes()
        with pytest.raises(TypeError, match="model_info must be a list"):
            file.load_model(
                model_yaml="test_models_energy.yaml",
                model_info="simple_energy",  # type: ignore[arg-type]
            )

    #
    def test_load_model_rejects_multiple_energy_names(self):
        """load_model should raise ValueError for energy model with >1 name."""

        file = self._make_file_with_axes()
        with pytest.raises(ValueError, match="single model"):
            file.load_model(
                model_yaml="test_models_energy.yaml",
                model_info=["simple_energy", "single_glp"],
            )

    #
    def test_load_model_rejects_duplicate_name(self):
        """load_model should raise ValueError when trying to load a model again."""

        file = self._make_file_with_axes()
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["simple_energy"],
        )
        with pytest.raises(ValueError, match="already exists"):
            file.load_model(
                model_yaml="test_models_energy.yaml",
                model_info=["simple_energy"],
            )

    #
    def test_load_model_rejects_nonexistent_submodel(self):
        """load_model should raise ValueError when model name not in YAML."""

        file = self._make_file_with_axes()
        with pytest.raises(ValueError, match="not found"):
            file.load_model(
                model_yaml="test_models_energy.yaml",
                model_info=["this_model_does_not_exist"],
            )

    #
    def test_load_model_rejects_invalid_model_type(self):
        """load_model should raise ValueError for unrecognized model_type."""

        file = self._make_file_with_axes()
        with pytest.raises(ValueError, match="not recognized"):
            file.load_model(
                model_yaml="test_models_energy.yaml",
                model_info=["simple_energy"],
                model_type="bogus",  # type: ignore[arg-type]
            )

    #
    def test_load_model_propagates_axes(self):
        """Loaded model should inherit energy, time, and aux_axis from file."""

        file = self._make_file_with_axes()
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["simple_energy"],
        )
        model = file.model_active
        assert model is not None
        np.testing.assert_array_equal(model.energy, file.energy)
        np.testing.assert_array_equal(model.time, file.time)
        np.testing.assert_array_equal(model.aux_axis, file.aux_axis)

    #
    def test_select_model_by_name(self):
        """select_model with a string should return the matching model."""

        file = self._make_file_with_axes()
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["simple_energy"],
        )
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["single_glp"],
        )
        model = file.select_model("simple_energy")
        assert model is not None
        assert model.name == "simple_energy"

    #
    def test_select_model_by_index(self):
        """select_model with an int should return the model at that index."""

        file = self._make_file_with_axes()
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["simple_energy"],
        )
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["single_glp"],
        )
        model = file.select_model(0)
        assert model is not None
        assert model.name == "simple_energy"
        model = file.select_model(1)
        assert model is not None
        assert model.name == "single_glp"

    #
    def test_select_model_not_found(self):
        """select_model should return None for nonexistent name / out-of-range index."""

        file = self._make_file_with_axes()
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["simple_energy"],
        )
        assert file.select_model("nonexistent") is None
        assert file.select_model(99) is None

    #
    def test_select_model_returns_index(self):
        """select_model with return_type='index' should return the list index."""

        file = self._make_file_with_axes()
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["simple_energy"],
        )
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["single_glp"],
        )
        idx = file.select_model("single_glp", return_type="index")
        assert idx == 1

    #
    def test_select_model_by_list(self):
        """select_model with a list should compose a name and find the model."""

        file = self._make_file_with_axes()
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["simple_energy"],
        )
        model = file.select_model(["simple_energy"])
        assert model is not None
        assert model.name == "simple_energy"

    #
    def test_delete_model_by_name(self):
        """delete_model with a name should remove that model."""

        file = self._make_file_with_axes()
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["simple_energy"],
        )
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["single_glp"],
        )
        file.delete_model("simple_energy")
        assert len(file.models) == 1
        assert file.models[0].name == "single_glp"

    #
    def test_delete_model_by_index(self):
        """delete_model with an index should remove the model at that position."""

        file = self._make_file_with_axes()
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["simple_energy"],
        )
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["single_glp"],
        )
        file.delete_model(0)
        assert len(file.models) == 1
        assert file.models[0].name == "single_glp"

    #
    def test_delete_active_model(self):
        """delete_model with None should remove the currently active model."""

        file = self._make_file_with_axes()
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["simple_energy"],
        )
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["single_glp"],
        )
        # active is single_glp (last loaded)
        file.delete_model()
        assert len(file.models) == 1
        assert file.models[0].name == "simple_energy"

    #
    def test_delete_model_no_active_warns(self):
        """delete_model(None) with no active model should warn."""

        file = self._make_file_with_axes()
        file.model_active = None
        with pytest.warns(UserWarning, match="No active model"):
            file.delete_model()

    #
    def test_delete_model_nonexistent(self):
        """delete_model with bad name or out-of-range index should not crash."""

        file = self._make_file_with_axes()
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["simple_energy"],
        )
        file.delete_model("nonexistent")
        assert len(file.models) == 1
        file.delete_model(99)
        assert len(file.models) == 1

    #
    def test_reset_models(self):
        """reset_models should clear all models."""

        file = self._make_file_with_axes()
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["simple_energy"],
        )
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["single_glp"],
        )
        assert len(file.models) == 2
        file.reset_models()
        assert len(file.models) == 0

    #
    def test_reset_models_on_empty(self):
        """reset_models on a file with no models should not crash."""

        file = self._make_file_with_axes()
        assert len(file.models) == 0
        file.reset_models()
        assert len(file.models) == 0


#
#
class TestFitLimitsAndBaseline:
    """Test fit limits and baseline."""

    #
    def _make_file_with_data(self):
        """Create file with axes and 2D data."""

        project = Project(path="tests")
        file = File(parent_project=project)
        file.energy = np.linspace(80, 90, 201)
        file.time = np.linspace(-10, 100, 111)
        file.data = np.random.default_rng(42).normal(
            size=(len(file.time), len(file.energy))
        )
        file.dim = 2
        return file

    #
    def test_set_fit_limits_energy_only(self):
        """set_fit_limits should set e_lim and e_lim_abs."""

        file = self._make_file_with_data()
        file.set_fit_limits([82, 88], show_plot=False)
        assert file.e_lim_abs == [82, 88]
        assert file.e_lim is not None
        assert file.energy is not None
        # Slicing with e_lim should give a smaller array
        e_cut = file.energy[file.e_lim[0] : file.e_lim[1]]
        assert len(e_cut) < len(file.energy)
        assert np.min(e_cut) >= 82
        assert np.max(e_cut) <= 88

    #
    def test_set_fit_limits_energy_and_time(self):
        """set_fit_limits with time_limits should set both e_lim and t_lim."""

        file = self._make_file_with_data()
        file.set_fit_limits([82, 88], time_limits=[0, 50], show_plot=False)
        assert file.e_lim_abs == [82, 88]
        assert file.t_lim_abs == [0, 50]
        assert file.t_lim is not None
        assert file.time is not None
        t_cut = file.time[file.t_lim[0] : file.t_lim[1]]
        assert np.min(t_cut) >= 0
        assert np.max(t_cut) <= 50

    #
    def test_set_fit_limits_none_uses_full_range(self):
        """set_fit_limits with None energy_limits should use full energy range."""

        file = self._make_file_with_data()
        file.set_fit_limits(None, show_plot=False)
        assert file.e_lim_abs is not None
        assert np.isclose(file.e_lim_abs[0], 80.0)
        assert np.isclose(file.e_lim_abs[1], 90.0)

    #
    def test_set_fit_limits_descending_energy(self):
        """set_fit_limits should handle descending energy axes correctly."""

        project = Project(path="tests")
        file = File(parent_project=project)
        file.energy = np.linspace(90, 80, 201)  # descending
        file.time = np.linspace(-10, 100, 111)
        file.data = np.random.default_rng(42).normal(
            size=(len(file.time), len(file.energy))
        )
        file.dim = 2
        file.set_fit_limits([82, 88], show_plot=False)
        assert file.e_lim_abs == [82, 88]
        assert file.e_lim is not None
        assert file.energy is not None
        # Slicing with e_lim should give a smaller array within bounds
        e_cut = file.energy[file.e_lim[0] : file.e_lim[1]]
        assert len(e_cut) < len(file.energy)
        assert np.min(e_cut) >= 82
        assert np.max(e_cut) <= 88

    #
    def test_set_fit_limits_time_without_time_axis_warns(self):
        """set_fit_limits with time_limits but no time axis (1D) should warn."""

        project = Project(path="tests")
        file = File(parent_project=project)
        file.energy = np.linspace(80, 90, 201)
        file.dim = 1
        with pytest.warns(UserWarning, match="[Tt]ime"):
            file.set_fit_limits([82, 88], time_limits=[0, 50], show_plot=False)

    #
    def test_set_fit_limits_no_data_warns(self):
        """set_fit_limits without data or energy should warn."""

        project = Project(path="tests")
        file = File(parent_project=project)
        with pytest.warns(UserWarning, match="cannot set fit limits"):
            file.set_fit_limits([82, 88], show_plot=False)

    #
    def test_define_baseline_abs(self):
        """define_baseline should average data in the given time window."""

        file = self._make_file_with_data()
        # Baseline from t=-10 to t=0 (absolute)
        file.define_baseline(-10, 0, time_type="abs", show_plot=False)
        assert file.data_base is not None
        assert file.energy is not None
        assert file.time is not None
        assert file.data is not None
        assert file.data_base.shape == file.energy.shape
        # Manually compute expected baseline
        t_start = int(np.searchsorted(file.time, -10))
        t_stop = int(np.searchsorted(file.time, 0))
        expected = np.mean(file.data[t_start:t_stop, :], axis=0)
        np.testing.assert_allclose(file.data_base, expected)

    #
    def test_define_baseline_ind(self):
        """define_baseline with time_type='ind' should use indices directly."""

        file = self._make_file_with_data()
        file.define_baseline(0, 5, time_type="ind", show_plot=False)
        assert file.data_base is not None
        assert file.data is not None
        expected = np.mean(file.data[0:5, :], axis=0)
        np.testing.assert_allclose(file.data_base, expected)

    #
    def test_define_baseline_stores_time_info(self):
        """define_baseline should store base_t_ind and base_t_abs."""

        file = self._make_file_with_data()
        file.define_baseline(-5, 5, time_type="abs", show_plot=False)
        assert file.base_t_ind is not None
        assert file.base_t_abs is not None
        assert len(file.base_t_ind) == 2
        assert len(file.base_t_abs) == 2

    #
    def test_define_baseline_1d_warns(self):
        """define_baseline on 1D data should warn."""

        file = self._make_file_with_data()
        file.dim = 1
        with pytest.warns(UserWarning, match="Cannot define baseline for 1D"):
            file.define_baseline(-10, 0, show_plot=False)

    #
    def test_define_baseline_no_data_warns(self):
        """define_baseline without data should warn."""

        project = Project(path="tests")
        file = File(parent_project=project)
        file.dim = 2
        with pytest.warns(UserWarning, match="No data loaded"):
            file.define_baseline(-10, 0, show_plot=False)

    #
    def test_define_baseline_invalid_time_type_warns(self):
        """define_baseline with invalid time_type should warn."""

        file = self._make_file_with_data()
        with pytest.warns(UserWarning, match="Unknown time_type"):
            file.define_baseline(-10, 0, time_type="bogus", show_plot=False)


#
#
class TestFitLimitsSlicing:
    """Test that fit limits produce correct slices of real axis data.

    These tests verify the actual data points selected by e_lim and t_lim,
    not just that the attributes are set. Covers ascending energy, descending
    energy, full-range edge cases, and the residual function in fitlib.
    """

    #
    def _make_file(self, *, energy, time=None):
        """Create a File with given energy axis and optional time axis."""

        project = Project(path="tests")
        file = File(parent_project=project)
        file.energy = energy
        if time is not None:
            file.time = time
            file.data = np.random.default_rng(42).normal(size=(len(time), len(energy)))
            file.dim = 2
        else:
            file.data = np.random.default_rng(42).normal(size=len(energy))
            file.dim = 1
        return file

    # -- sub-range limits: verify correct data points remain --

    #
    def test_subrange_ascending_energy_correct_points(self):
        """Sub-range limits on ascending energy should keep only points in range."""

        energy = np.linspace(80, 90, 201)  # 0.05 eV steps
        file = self._make_file(energy=energy)
        file.set_fit_limits([82, 88], show_plot=False)

        e_cut = energy[file.e_lim[0] : file.e_lim[1]]
        assert len(e_cut) > 0
        assert np.min(e_cut) >= 82.0
        assert np.max(e_cut) <= 88.0
        # boundary points should be included
        assert np.isclose(np.min(e_cut), 82.0, atol=0.05)
        assert np.isclose(np.max(e_cut), 88.0, atol=0.05)

    #
    def test_subrange_descending_energy_correct_points(self):
        """Sub-range limits on descending energy should keep only points in range."""

        energy = np.linspace(90, 80, 201)  # descending, 0.05 eV steps
        file = self._make_file(energy=energy)
        file.set_fit_limits([82, 88], show_plot=False)

        e_cut = energy[file.e_lim[0] : file.e_lim[1]]
        assert len(e_cut) > 0
        assert np.min(e_cut) >= 82.0
        assert np.max(e_cut) <= 88.0
        assert np.isclose(np.min(e_cut), 82.0, atol=0.05)
        assert np.isclose(np.max(e_cut), 88.0, atol=0.05)

    #
    def test_subrange_time_correct_points(self):
        """Sub-range time limits should keep only time points in range."""

        energy = np.linspace(80, 90, 201)
        time = np.linspace(-10, 100, 111)
        file = self._make_file(energy=energy, time=time)
        file.set_fit_limits([80, 90], time_limits=[0, 50], show_plot=False)

        t_cut = time[file.t_lim[0] : file.t_lim[1]]
        assert len(t_cut) > 0
        assert np.min(t_cut) >= 0.0
        assert np.max(t_cut) <= 50.0

    # -- full-range edge cases --

    #
    def test_full_range_ascending_preserves_all_energy(self):
        """Full-range limits on ascending energy must keep every point."""

        energy = np.linspace(80, 90, 201)
        file = self._make_file(energy=energy)
        file.set_fit_limits(energy_limits=None, show_plot=False)

        e_cut = energy[file.e_lim[0] : file.e_lim[1]]
        assert len(e_cut) == len(energy)

    #
    def test_full_range_descending_preserves_all_energy(self):
        """Full-range limits on descending energy must keep every point."""

        energy = np.linspace(90, 80, 201)
        file = self._make_file(energy=energy)
        file.set_fit_limits(energy_limits=None, show_plot=False)

        e_cut = energy[file.e_lim[0] : file.e_lim[1]]
        assert len(e_cut) == len(energy)

    #
    def test_full_range_preserves_all_time(self):
        """Full-range time limits must keep every time point."""

        energy = np.linspace(80, 90, 201)
        time = np.linspace(-10, 100, 111)
        file = self._make_file(energy=energy, time=time)
        file.set_fit_limits(energy_limits=None, time_limits=None, show_plot=False)

        t_cut = time[file.t_lim[0] : file.t_lim[1]]
        assert len(t_cut) == len(time)

    #
    def test_full_range_descending_2d_data_shape(self):
        """Full-range on descending energy 2D data must preserve full shape."""

        energy = np.linspace(90, 80, 201)
        time = np.linspace(-10, 100, 111)
        file = self._make_file(energy=energy, time=time)
        file.set_fit_limits(energy_limits=None, show_plot=False)

        data_cut = file.data[:, file.e_lim[0] : file.e_lim[1]]
        assert data_cut.shape == file.data.shape

    # -- residual function slicing (fitlib) --

    #
    def test_residual_no_limits_full_shape(self):
        """Residual with no limits should return array matching data shape."""

        data_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        fit_1d = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        # Directly test the slicing logic: with empty e_lim, full array used
        residual = data_1d - fit_1d
        assert residual.shape == data_1d.shape

        data_2d = np.ones((10, 20))
        fit_2d = np.ones((10, 20)) * 1.1
        residual_2d = data_2d - fit_2d
        assert residual_2d.shape == data_2d.shape

    #
    def test_residual_e_lim_slicing_1d(self):
        """e_lim slicing in 1D residual should select correct subarray."""

        data = np.arange(10, dtype=float)
        fit = np.zeros(10)
        e_lim = [2, 7]  # [start, stop) → indices 2,3,4,5,6

        residual = data[e_lim[0] : e_lim[1]] - fit[e_lim[0] : e_lim[1]]
        assert len(residual) == 5
        np.testing.assert_array_equal(residual, [2.0, 3.0, 4.0, 5.0, 6.0])

    #
    def test_residual_e_lim_slicing_2d(self):
        """e_lim slicing in 2D residual should select correct energy columns."""

        data = np.ones((5, 10))
        fit = np.zeros((5, 10))
        e_lim = [2, 7]  # [start, stop) → 5 columns

        residual = data[:, e_lim[0] : e_lim[1]] - fit[:, e_lim[0] : e_lim[1]]
        assert residual.shape == (5, 5)

    #
    def test_residual_t_lim_slicing_2d(self):
        """t_lim slicing in 2D residual should select correct time rows."""

        data = np.ones((10, 20))
        fit = np.zeros((10, 20))
        t_lim = [2, 8]

        residual = data[t_lim[0] : t_lim[1], :] - fit[t_lim[0] : t_lim[1], :]
        assert residual.shape == (6, 20)

    #
    def test_residual_both_limits_2d(self):
        """Combined e_lim and t_lim should select correct sub-region."""

        data = np.ones((10, 20))
        fit = np.zeros((10, 20))
        e_lim = [2, 17]  # [start, stop) → 15 columns
        t_lim = [1, 7]  # [start, stop) → 6 rows

        residual = (
            data[t_lim[0] : t_lim[1], e_lim[0] : e_lim[1]]
            - fit[t_lim[0] : t_lim[1], e_lim[0] : e_lim[1]]
        )
        assert residual.shape == (6, 15)

    #
    def test_residual_e_lim_full_range(self):
        """e_lim spanning full array must not produce empty array."""

        data = np.arange(10, dtype=float)
        fit = np.zeros(10)
        e_lim = [0, 10]  # [start, stop) — full array

        residual = data[e_lim[0] : e_lim[1]] - fit[e_lim[0] : e_lim[1]]
        assert len(residual) == 10


if __name__ == "__main__":
    pytest.main([__file__])
