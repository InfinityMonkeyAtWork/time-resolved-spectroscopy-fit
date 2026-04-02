"""Tests for File data correction methods.

Exercises subtract_dark, calibrate_data, reset_dark, reset_calibration,
and the _apply_corrections pipeline.
"""

import numpy as np
import pytest

from trspecfit import File, Project


def make_energy_axis(n=50):
    return np.linspace(80, 90, n)


def make_time_axis(n=20):
    return np.linspace(-5, 50, n)


#
def _guard(f: File) -> None:
    """Narrow ``| None`` attributes for Pylance after creating a data-bearing File."""

    assert f.data is not None  # type guard
    assert f.data_raw is not None  # type guard
    assert f.energy is not None  # type guard
    assert f.dark is not None  # type guard
    assert f.calibration is not None  # type guard


#
#
class TestSubtractDark:
    """Test dark/background subtraction."""

    #
    def _make_1d_file(self):
        energy = make_energy_axis()
        data = np.ones(energy.shape[0]) * 10.0
        return File(parent_project=Project(path=None), data=data, energy=energy)

    #
    def _make_2d_file(self):
        energy = make_energy_axis()
        time = make_time_axis()
        data = np.ones((time.shape[0], energy.shape[0])) * 10.0
        return File(
            parent_project=Project(path=None),
            data=data,
            energy=energy,
            time=time,
        )

    #
    def test_1d_subtract_dark(self):
        """Dark subtraction on 1D data."""

        f = self._make_1d_file()
        _guard(f)
        dark = np.ones(f.energy.shape[0]) * 3.0
        f.subtract_dark(dark)
        np.testing.assert_allclose(f.data, 7.0)
        np.testing.assert_allclose(f.data_raw, 10.0)

    #
    def test_2d_subtract_dark(self):
        """Dark subtraction broadcasts across time slices."""

        f = self._make_2d_file()
        _guard(f)
        dark = np.ones(f.energy.shape[0]) * 2.0
        f.subtract_dark(dark)
        np.testing.assert_allclose(f.data, 8.0)
        np.testing.assert_allclose(f.data_raw, 10.0)

    #
    def test_subtract_dark_replaces_previous(self):
        """Calling subtract_dark twice replaces, does not stack."""

        f = self._make_1d_file()
        _guard(f)
        f.subtract_dark(np.ones(f.energy.shape[0]) * 3.0)
        f.subtract_dark(np.ones(f.energy.shape[0]) * 5.0)
        np.testing.assert_allclose(f.data, 5.0)

    #
    def test_subtract_dark_wrong_shape(self):
        """Wrong shape raises ValueError."""

        f = self._make_1d_file()
        with pytest.raises(ValueError, match="1D with length"):
            f.subtract_dark(np.ones(10))

    #
    def test_subtract_dark_2d_array(self):
        """2D dark array raises ValueError."""

        f = self._make_1d_file()
        _guard(f)
        with pytest.raises(ValueError, match="1D with length"):
            f.subtract_dark(np.ones((2, f.energy.shape[0])))

    #
    def test_subtract_dark_no_data(self):
        """No data raises ValueError."""

        f = File(parent_project=Project(path=None))
        with pytest.raises(ValueError, match="No data loaded"):
            f.subtract_dark(np.ones(10))


#
#
class TestCalibrateData:
    """Test sensitivity calibration."""

    #
    def _make_1d_file(self):
        energy = make_energy_axis()
        data = np.ones(energy.shape[0]) * 10.0
        return File(parent_project=Project(path=None), data=data, energy=energy)

    #
    def _make_2d_file(self):
        energy = make_energy_axis()
        time = make_time_axis()
        data = np.ones((time.shape[0], energy.shape[0])) * 10.0
        return File(
            parent_project=Project(path=None),
            data=data,
            energy=energy,
            time=time,
        )

    #
    def test_1d_calibrate(self):
        """Calibration divides 1D data."""

        f = self._make_1d_file()
        _guard(f)
        cal = np.ones(f.energy.shape[0]) * 2.0
        f.calibrate_data(cal)
        np.testing.assert_allclose(f.data, 5.0)
        np.testing.assert_allclose(f.data_raw, 10.0)

    #
    def test_2d_calibrate(self):
        """Calibration broadcasts across time slices."""

        f = self._make_2d_file()
        _guard(f)
        cal = np.ones(f.energy.shape[0]) * 5.0
        f.calibrate_data(cal)
        np.testing.assert_allclose(f.data, 2.0)

    #
    def test_calibrate_replaces_previous(self):
        """Calling calibrate_data twice replaces, does not stack."""

        f = self._make_1d_file()
        _guard(f)
        f.calibrate_data(np.ones(f.energy.shape[0]) * 2.0)
        f.calibrate_data(np.ones(f.energy.shape[0]) * 5.0)
        np.testing.assert_allclose(f.data, 2.0)

    #
    def test_calibrate_zeros_rejected(self):
        """Zeros in calibration raise ValueError."""

        f = self._make_1d_file()
        _guard(f)
        cal = np.ones(f.energy.shape[0])
        cal[10] = 0.0
        with pytest.raises(ValueError, match="zeros"):
            f.calibrate_data(cal)

    #
    def test_calibrate_wrong_shape(self):
        """Wrong shape raises ValueError."""

        f = self._make_1d_file()
        with pytest.raises(ValueError, match="1D with length"):
            f.calibrate_data(np.ones(10))

    #
    def test_calibrate_no_data(self):
        """No data raises ValueError."""

        f = File(parent_project=Project(path=None))
        with pytest.raises(ValueError, match="No data loaded"):
            f.calibrate_data(np.ones(10))


#
#
class TestCombinedCorrections:
    """Test dark + calibration applied together."""

    #
    def _make_1d_file(self):
        energy = make_energy_axis()
        data = np.ones(energy.shape[0]) * 10.0
        return File(parent_project=Project(path=None), data=data, energy=energy)

    #
    def _make_2d_file(self):
        energy = make_energy_axis()
        time = make_time_axis()
        data = np.ones((time.shape[0], energy.shape[0])) * 10.0
        return File(
            parent_project=Project(path=None),
            data=data,
            energy=energy,
            time=time,
        )

    #
    def test_dark_then_calibrate(self):
        """(data_raw - dark) / calibration applied in correct order."""

        f = self._make_1d_file()
        _guard(f)
        dark = np.ones(f.energy.shape[0]) * 2.0
        cal = np.ones(f.energy.shape[0]) * 4.0
        f.subtract_dark(dark)
        f.calibrate_data(cal)
        # (10 - 2) / 4 = 2.0
        np.testing.assert_allclose(f.data, 2.0)

    #
    def test_calibrate_then_dark(self):
        """Order of method calls irrelevant — formula is always (raw - dark) / cal."""

        f = self._make_1d_file()
        _guard(f)
        dark = np.ones(f.energy.shape[0]) * 2.0
        cal = np.ones(f.energy.shape[0]) * 4.0
        f.calibrate_data(cal)
        f.subtract_dark(dark)
        # still (10 - 2) / 4 = 2.0
        np.testing.assert_allclose(f.data, 2.0)

    #
    def test_data_raw_never_changes(self):
        """data_raw remains unchanged through all corrections."""

        f = self._make_2d_file()
        _guard(f)
        original = f.data_raw.copy()
        f.subtract_dark(np.ones(f.energy.shape[0]) * 3.0)
        f.calibrate_data(np.ones(f.energy.shape[0]) * 2.0)
        np.testing.assert_array_equal(f.data_raw, original)


#
#
class TestResets:
    """Test reset_dark and reset_calibration."""

    #
    def _make_1d_file(self):
        energy = make_energy_axis()
        data = np.ones(energy.shape[0]) * 10.0
        return File(parent_project=Project(path=None), data=data, energy=energy)

    #
    def test_reset_dark(self):
        """reset_dark restores data to raw / calibration."""

        f = self._make_1d_file()
        _guard(f)
        cal = np.ones(f.energy.shape[0]) * 2.0
        f.subtract_dark(np.ones(f.energy.shape[0]) * 3.0)
        f.calibrate_data(cal)
        # (10 - 3) / 2 = 3.5
        np.testing.assert_allclose(f.data, 3.5)
        f.reset_dark()
        # (10 - 0) / 2 = 5.0
        np.testing.assert_allclose(f.data, 5.0)

    #
    def test_reset_calibration(self):
        """reset_calibration restores data to raw - dark."""

        f = self._make_1d_file()
        _guard(f)
        f.subtract_dark(np.ones(f.energy.shape[0]) * 3.0)
        f.calibrate_data(np.ones(f.energy.shape[0]) * 2.0)
        f.reset_calibration()
        # (10 - 3) / 1 = 7.0
        np.testing.assert_allclose(f.data, 7.0)

    #
    def test_reset_both(self):
        """Resetting both restores original data."""

        f = self._make_1d_file()
        _guard(f)
        f.subtract_dark(np.ones(f.energy.shape[0]) * 3.0)
        f.calibrate_data(np.ones(f.energy.shape[0]) * 2.0)
        f.reset_dark()
        f.reset_calibration()
        np.testing.assert_allclose(f.data, f.data_raw)

    #
    def test_reset_no_data(self):
        """Reset on empty file raises ValueError."""

        f = File(parent_project=Project(path=None))
        with pytest.raises(ValueError, match="No data loaded"):
            f.reset_dark()
        with pytest.raises(ValueError, match="No data loaded"):
            f.reset_calibration()


#
#
class TestBaselineRecomputation:
    """Test that data_base is recomputed after corrections."""

    #
    def _make_2d_file(self):
        energy = make_energy_axis()
        time = make_time_axis()
        rng = np.random.default_rng(42)
        data = rng.normal(loc=100.0, scale=1.0, size=(time.shape[0], energy.shape[0]))
        return File(
            parent_project=Project(path=None),
            data=data,
            energy=energy,
            time=time,
        )

    #
    def test_baseline_recomputed_after_dark(self):
        """define_baseline result updates when dark is applied."""

        f = self._make_2d_file()
        _guard(f)
        assert f.time is not None  # type guard
        f.define_baseline(f.time[0], f.time[5], show_plot=False)
        assert f.data_base is not None  # type guard
        baseline_before = f.data_base.copy()
        dark = np.ones(f.energy.shape[0]) * 10.0
        f.subtract_dark(dark)
        # baseline should shift down by 10
        np.testing.assert_allclose(f.data_base, baseline_before - 10.0)

    #
    def test_baseline_recomputed_after_calibrate(self):
        """define_baseline result updates when calibration is applied."""

        f = self._make_2d_file()
        _guard(f)
        assert f.time is not None  # type guard
        f.define_baseline(f.time[0], f.time[5], show_plot=False)
        assert f.data_base is not None  # type guard
        baseline_before = f.data_base.copy()
        cal = np.ones(f.energy.shape[0]) * 2.0
        f.calibrate_data(cal)
        np.testing.assert_allclose(f.data_base, baseline_before / 2.0)


#
#
class TestDefaults:
    """Test that default dark/calibration are identity operations."""

    #
    def test_defaults_are_identity(self):
        """Default dark=zeros and calibration=ones leave data unchanged."""

        energy = make_energy_axis()
        data = np.arange(energy.shape[0], dtype=float)
        f = File(parent_project=Project(path=None), data=data, energy=energy)
        _guard(f)
        np.testing.assert_array_equal(f.data, f.data_raw)
        np.testing.assert_array_equal(f.dark, np.zeros(energy.shape[0]))
        np.testing.assert_array_equal(f.calibration, np.ones(energy.shape[0]))

    #
    def test_no_data_defaults_none(self):
        """Bare File() has None for data_raw, dark, calibration."""

        f = File(parent_project=Project(path=None))
        assert f.data_raw is None
        assert f.dark is None
        assert f.calibration is None
