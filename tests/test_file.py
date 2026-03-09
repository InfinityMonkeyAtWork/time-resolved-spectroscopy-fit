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
    def setUp(self):
        """Setup function to create project and file with axes and dummy data."""
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
        file = self.setUp()
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["simple_energy"],
            debug=False,
        )
        assert file.model_active is not None
        assert file.model_active.name == "simple_energy"
        assert len(file.models) == 1
        assert file.models[0] is file.model_active

    #
    def test_load_multiple_models(self):
        """Loading a second model should add to list and update active."""
        file = self.setUp()
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["simple_energy"],
            debug=False,
        )
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["single_glp"],
            debug=False,
        )
        assert len(file.models) == 2
        assert file.model_active is not None
        assert file.model_active.name == "single_glp"

    #
    def test_load_model_returns_none_for_energy(self):
        """load_model with model_type='energy' should return None."""
        file = self.setUp()
        result = file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["simple_energy"],
            debug=False,
        )
        assert result is None

    #
    def test_load_model_returns_model_for_dynamics(self):
        """load_model with model_type='dynamics' should return a Dynamics object."""
        file = self.setUp()
        result = file.load_model(
            model_yaml="test_models_time.yaml",
            model_info=["MonoExpPos"],
            par_name="GLP_01_A",
            model_type="dynamics",
            debug=False,
        )
        assert result is not None
        assert result.name == "GLP_01_A"
        # dynamics should not be added to file.models
        assert len(file.models) == 0

    #
    def test_load_model_returns_model_for_profile(self):
        """load_model with model_type='profile' should return a Profile object."""
        file = self.setUp()
        result = file.load_model(
            model_yaml="test_models_profile.yaml",
            model_info=["profile_linear"],
            par_name="GLP_01_A",
            model_type="profile",
            debug=False,
        )
        assert result is not None
        assert result.name == "GLP_01_A"
        assert len(file.models) == 0

    #
    def test_load_model_rejects_non_list(self):
        """load_model should raise TypeError if model_info is not a list."""
        file = self.setUp()
        with pytest.raises(TypeError, match="model_info must be a list"):
            file.load_model(
                model_yaml="test_models_energy.yaml",
                model_info="simple_energy",  # type: ignore[arg-type]
                debug=False,
            )

    #
    def test_load_model_rejects_multiple_energy_names(self):
        """load_model should raise ValueError for energy model with >1 name."""
        file = self.setUp()
        with pytest.raises(ValueError, match="single model"):
            file.load_model(
                model_yaml="test_models_energy.yaml",
                model_info=["simple_energy", "single_glp"],
                debug=False,
            )

    #
    def test_load_model_rejects_duplicate_name(self):
        """load_model should raise ValueError when trying to load a model again."""
        file = self.setUp()
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["simple_energy"],
            debug=False,
        )
        with pytest.raises(ValueError, match="already exists"):
            file.load_model(
                model_yaml="test_models_energy.yaml",
                model_info=["simple_energy"],
                debug=False,
            )

    #
    def test_load_model_rejects_nonexistent_submodel(self):
        """load_model should raise ValueError when model name not in YAML."""
        file = self.setUp()
        with pytest.raises(ValueError, match="not found"):
            file.load_model(
                model_yaml="test_models_energy.yaml",
                model_info=["this_model_does_not_exist"],
                debug=False,
            )

    #
    def test_load_model_rejects_invalid_model_type(self):
        """load_model should raise ValueError for unrecognized model_type."""
        file = self.setUp()
        with pytest.raises(ValueError, match="not recognized"):
            file.load_model(
                model_yaml="test_models_energy.yaml",
                model_info=["simple_energy"],
                model_type="bogus",  # type: ignore[arg-type]
                debug=False,
            )

    #
    def test_load_model_propagates_axes(self):
        """Loaded model should inherit energy, time, and aux_axis from file."""
        file = self.setUp()
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["simple_energy"],
            debug=False,
        )
        model = file.model_active
        assert model is not None
        np.testing.assert_array_equal(model.energy, file.energy)
        np.testing.assert_array_equal(model.time, file.time)
        np.testing.assert_array_equal(model.aux_axis, file.aux_axis)

    #
    def test_select_model_by_name(self):
        """select_model with a string should return the matching model."""
        file = self.setUp()
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["simple_energy"],
            debug=False,
        )
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["single_glp"],
            debug=False,
        )
        model = file.select_model("simple_energy")
        assert model is not None
        assert model.name == "simple_energy"

    #
    def test_select_model_by_index(self):
        """select_model with an int should return the model at that index."""
        file = self.setUp()
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["simple_energy"],
            debug=False,
        )
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["single_glp"],
            debug=False,
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
        file = self.setUp()
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["simple_energy"],
            debug=False,
        )
        assert file.select_model("nonexistent") is None
        assert file.select_model(99) is None

    #
    def test_select_model_returns_index(self):
        """select_model with return_type='index' should return the list index."""
        file = self.setUp()
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["simple_energy"],
            debug=False,
        )
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["single_glp"],
            debug=False,
        )
        idx = file.select_model("single_glp", return_type="index")
        assert idx == 1

    #
    def test_select_model_by_list(self):
        """select_model with a list should compose a name and find the model."""
        file = self.setUp()
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["simple_energy"],
            debug=False,
        )
        model = file.select_model(["simple_energy"])
        assert model is not None
        assert model.name == "simple_energy"

    #
    def test_delete_model_by_name(self):
        """delete_model with a name should remove that model."""
        file = self.setUp()
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["simple_energy"],
            debug=False,
        )
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["single_glp"],
            debug=False,
        )
        file.delete_model("simple_energy")
        assert len(file.models) == 1
        assert file.models[0].name == "single_glp"

    #
    def test_delete_model_by_index(self):
        """delete_model with an index should remove the model at that position."""
        file = self.setUp()
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["simple_energy"],
            debug=False,
        )
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["single_glp"],
            debug=False,
        )
        file.delete_model(0)
        assert len(file.models) == 1
        assert file.models[0].name == "single_glp"

    #
    def test_delete_active_model(self):
        """delete_model with None should remove the currently active model."""
        file = self.setUp()
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["simple_energy"],
            debug=False,
        )
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["single_glp"],
            debug=False,
        )
        # active is single_glp (last loaded)
        file.delete_model()
        assert len(file.models) == 1
        assert file.models[0].name == "simple_energy"

    #
    def test_delete_model_no_active_warns(self):
        """delete_model(None) with no active model should warn."""
        file = self.setUp()
        file.model_active = None
        with pytest.warns(UserWarning, match="No active model"):
            file.delete_model()

    #
    def test_delete_model_nonexistent(self):
        """delete_model with bad name or out-of-range index should not crash."""
        file = self.setUp()
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["simple_energy"],
            debug=False,
        )
        file.delete_model("nonexistent")
        assert len(file.models) == 1
        file.delete_model(99)
        assert len(file.models) == 1

    #
    def test_reset_models(self):
        """reset_models should clear all models."""
        file = self.setUp()
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["simple_energy"],
            debug=False,
        )
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["single_glp"],
            debug=False,
        )
        assert len(file.models) == 2
        file.reset_models()
        assert len(file.models) == 0

    #
    def test_reset_models_on_empty(self):
        """reset_models on a file with no models should not crash."""
        file = self.setUp()
        assert len(file.models) == 0
        file.reset_models()
        assert len(file.models) == 0


#
#
class TestFitLimitsAndBaseline:
    """Test fit limits and baseline."""

    #
    def setUp(self):
        """Setup function to create file with axes and 2D data."""
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
        file = self.setUp()
        file.set_fit_limits([82, 88], show_plot=False)
        assert file.e_lim_abs == [82, 88]
        assert file.e_lim is not None
        assert file.energy is not None
        # Slicing with e_lim should give a smaller array
        e_cut = file.energy[file.e_lim[0] : -file.e_lim[1]]
        assert len(e_cut) < len(file.energy)
        assert np.min(e_cut) >= 82
        assert np.max(e_cut) <= 88

    #
    def test_set_fit_limits_energy_and_time(self):
        """set_fit_limits with time_limits should set both e_lim and t_lim."""
        file = self.setUp()
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
        file = self.setUp()
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
        e_cut = file.energy[file.e_lim[0] : -file.e_lim[1]]
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
        file = self.setUp()
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
        file = self.setUp()
        file.define_baseline(0, 5, time_type="ind", show_plot=False)
        assert file.data_base is not None
        assert file.data is not None
        expected = np.mean(file.data[0:5, :], axis=0)
        np.testing.assert_allclose(file.data_base, expected)

    #
    def test_define_baseline_stores_time_info(self):
        """define_baseline should store base_t_ind and base_t_abs."""
        file = self.setUp()
        file.define_baseline(-5, 5, time_type="abs", show_plot=False)
        assert file.base_t_ind is not None
        assert file.base_t_abs is not None
        assert len(file.base_t_ind) == 2
        assert len(file.base_t_abs) == 2

    #
    def test_define_baseline_1d_warns(self):
        """define_baseline on 1D data should warn."""
        file = self.setUp()
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
        file = self.setUp()
        with pytest.warns(UserWarning, match="Unknown time_type"):
            file.define_baseline(-10, 0, time_type="bogus", show_plot=False)


if __name__ == "__main__":
    pytest.main([__file__])
