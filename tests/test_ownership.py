"""
Behavioral probes for the ownership contract, rules 1 to 3
(docs/design/api_ownership_contract.md; check 21 of docs/ai/code-review.md).

Reading the code shows what a method does; these probes show what the File
does afterwards: a caller mutates across the ownership boundary, or assigns
an owned attribute, and the File's state is what the contract says.
"""

import numpy as np
import pytest
from _utils import make_project

from trspecfit import File


#
def make_2d_file(*, name: str = "owned") -> tuple[File, dict[str, np.ndarray]]:
    """A 2D File built from caller arrays that the caller keeps and may edit."""

    project = make_project()
    arrays = {
        "data": np.arange(12.0).reshape(3, 4) + 10.0,
        "energy": np.array([80.0, 81.0, 82.0, 83.0]),
        "time": np.array([-1.0, 0.0, 1.0]),
        "aux_axis": np.array([0.0, 0.5]),
    }
    file = File(parent_project=project, name=name, **arrays)
    return file, arrays


#
#
class TestInputsAreOwnedAtConstruction:
    """Rule 1: inputs are copied, frozen, and replaced only by a new File."""

    #
    def test_caller_edits_after_construction_do_not_reach_the_file(self):
        file, arrays = make_2d_file()
        before = {k: v.copy() for k, v in arrays.items()}
        for arr in arrays.values():
            arr[...] = -999.0
        np.testing.assert_array_equal(file.data, before["data"])
        np.testing.assert_array_equal(file.data_raw, before["data"])
        np.testing.assert_array_equal(file.energy, before["energy"])
        np.testing.assert_array_equal(file.time, before["time"])
        np.testing.assert_array_equal(file.aux_axis, before["aux_axis"])

    #
    @pytest.mark.parametrize("attr", ["data", "data_raw", "energy", "time", "aux_axis"])
    def test_handed_out_arrays_are_read_only(self, attr):
        file, _ = make_2d_file()
        with pytest.raises(ValueError, match="read-only"):
            getattr(file, attr)[0] = 0.0

    #
    def test_synthesized_axes_are_read_only_too(self):
        project = make_project()
        file = File(parent_project=project, data=np.ones((2, 3)))
        with pytest.raises(ValueError, match="read-only"):
            file.energy[0] = 5.0
        with pytest.raises(ValueError, match="read-only"):
            file.time[0] = 5.0

    #
    def test_models_share_the_frozen_axes(self):
        """Models hold the file's axes by reference; frozen arrays make that safe."""

        file, _ = make_2d_file()
        file.load_model(
            model_yaml="models/file_energy.yaml", model_info="simple_energy"
        )
        model = file.model_active
        assert model is not None  # type guard
        assert model.energy is file.energy
        with pytest.raises(ValueError, match="read-only"):
            model.energy[0] = 0.0


#
#
class TestOwnedAttributesRefuseAssignment:
    """Rule 8 precedent: refuse at the point of misuse and name the route."""

    #
    @pytest.mark.parametrize(
        ("attr", "route"),
        [
            ("data", "correction methods"),
            ("data_raw", "construct a new File"),
            ("dim", "construct a new File"),
            ("energy", "construct a new File"),
            ("time", "construct a new File"),
            ("aux_axis", "construct a new File"),
            ("dark", "subtract_dark()"),
            ("calibration", "calibrate_data()"),
            ("data_base", "define_baseline()"),
            ("base_t_ind", "define_baseline()"),
            ("base_t_abs", "define_baseline()"),
            ("e_lim", "set_fit_limits()"),
            ("e_lim_abs", "set_fit_limits()"),
            ("t_lim", "set_fit_limits()"),
            ("t_lim_abs", "set_fit_limits()"),
            ("noise", "set_noise()"),
        ],
    )
    def test_assignment_raises_and_names_the_route(self, attr, route):
        file, _ = make_2d_file()
        before = getattr(file, attr)
        with pytest.raises(AttributeError, match=f"File.{attr} is owned") as info:
            setattr(file, attr, None)
        assert route in str(info.value)
        assert getattr(file, attr) is before

    #
    @pytest.mark.parametrize("attr", ["e_lim", "e_lim_abs", "t_lim", "t_lim_abs"])
    def test_fit_windows_cannot_be_edited_in_place(self, attr):
        file, _ = make_2d_file()
        file.set_fit_limits([81.0, 82.0], time_limits=[0.0, 1.0], show_plot=False)
        window = getattr(file, attr)
        assert len(window) == 2
        with pytest.raises(TypeError):
            window[0] = 0  # type: ignore[index]

    #
    def test_baseline_window_cannot_be_edited_in_place(self):
        file, _ = make_2d_file()
        file.define_baseline(-1.0, 0.0, show_plot=False)
        assert file.base_t_ind == (0, 2)
        assert file.base_t_abs == (-1.0, 0.0)
        with pytest.raises(TypeError):
            file.base_t_ind[0] = 1  # type: ignore[index]
        with pytest.raises(ValueError, match="read-only"):
            file.data_base[0] = 0.0


#
#
class TestCorrectionsAreOperationsOnOwnedArrays:
    """Rule 2: the File copies the correction it is given; data is derived."""

    #
    def test_caller_edits_after_subtract_dark_do_not_reach_the_file(self):
        file, arrays = make_2d_file()
        dark = np.array([1.0, 2.0, 3.0, 4.0])
        file.subtract_dark(dark)
        expected = arrays["data"] - dark
        dark[...] = 100.0
        np.testing.assert_array_equal(file.dark, [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(file.data, expected)
        with pytest.raises(ValueError, match="read-only"):
            file.dark[0] = 0.0

    #
    def test_caller_edits_after_calibrate_data_do_not_reach_the_file(self):
        file, arrays = make_2d_file()
        calibration = np.array([1.0, 2.0, 4.0, 8.0])
        file.calibrate_data(calibration)
        expected = arrays["data"] / calibration
        calibration[...] = 1.0
        np.testing.assert_array_equal(file.calibration, [1.0, 2.0, 4.0, 8.0])
        np.testing.assert_array_equal(file.data, expected)

    #
    def test_corrected_data_and_baseline_are_recomputed_read_only(self):
        file, arrays = make_2d_file()
        file.define_baseline(-1.0, 0.0, show_plot=False)
        file.subtract_dark(np.ones(4))
        np.testing.assert_array_equal(
            file.data_base, (arrays["data"] - 1.0)[0:2].mean(axis=0)
        )
        with pytest.raises(ValueError, match="read-only"):
            file.data[0, 0] = 0.0
        np.testing.assert_array_equal(file.data_raw, arrays["data"])
