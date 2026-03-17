"""Unit tests for energy (spectral) functions.

Each function is tested against known analytical values:
- Value at center (x=x0)
- Symmetry where applicable
- Limiting cases (m=0 → Gaussian, m=1 → Lorentzian, etc.)
- Known relationships between parameters
"""

import numpy as np
import pytest

from trspecfit.functions.energy import (
    DS,
    GLP,
    GLS,
    Gauss,
    GaussAsym,
    LinBack,
    Lorentz,
    Offset,
    Shirley,
    Voigt,
)


#
def make_energy_axis():
    """Standard energy axis for peak tests."""

    return np.linspace(-10, 10, 2001)


#
def setUpSpectrum(x):
    """Simple test spectrum (single Gaussian) for background functions."""

    return 10 * np.exp(-0.5 * (x / 2) ** 2)


#
#
class TestOffset:
    #
    def test_constant_value(self):
        x = make_energy_axis()
        spectrum = setUpSpectrum(x)
        result = Offset(x, y0=3.0, spectrum=spectrum)
        assert result.shape == spectrum.shape
        np.testing.assert_allclose(result, 3.0)

    #
    def test_zero_offset(self):
        x = make_energy_axis()
        spectrum = setUpSpectrum(x)
        result = Offset(x, y0=0.0, spectrum=spectrum)
        np.testing.assert_allclose(result, 0.0)

    #
    def test_negative_offset(self):
        x = make_energy_axis()
        spectrum = setUpSpectrum(x)
        result = Offset(x, y0=-2.5, spectrum=spectrum)
        np.testing.assert_allclose(result, -2.5)


#
#
class TestShirley:
    #
    def test_monotonic_decreasing(self):
        """Shirley background should be monotonically non-increasing."""

        x = make_energy_axis()
        spectrum = setUpSpectrum(x)
        result = Shirley(x, pShirley=400, spectrum=spectrum)
        assert result.shape == spectrum.shape
        assert np.all(np.diff(result) <= 1e-12)

    #
    def test_zero_scaling(self):
        x = make_energy_axis()
        spectrum = setUpSpectrum(x)
        result = Shirley(x, pShirley=0, spectrum=spectrum)
        np.testing.assert_allclose(result, 0.0)

    #
    def test_scales_linearly(self):
        x = make_energy_axis()
        spectrum = setUpSpectrum(x)
        r1 = Shirley(x, pShirley=100, spectrum=spectrum)
        r2 = Shirley(x, pShirley=200, spectrum=spectrum)
        np.testing.assert_allclose(r2, 2 * r1)


#
#
class TestLinBack:
    """Clamped linear background: linear between xStart/xStop, constant outside."""

    #
    def test_zero_slope_is_constant(self):
        x = make_energy_axis()
        spectrum = setUpSpectrum(x)
        result = LinBack(x, m=0.0, b=5.0, xStart=-5, xStop=5, spectrum=spectrum)
        np.testing.assert_allclose(result, 5.0)

    #
    def test_positive_slope(self):
        x = make_energy_axis()
        spectrum = setUpSpectrum(x)
        result = LinBack(x, m=2.0, b=1.0, xStart=-5, xStop=5, spectrum=spectrum)
        idx = np.argmin(np.abs(x - 0.0))
        # At x=0: m*(0 - (-5)) + b = 2*5 + 1 = 11
        assert result[idx] == pytest.approx(11.0, abs=0.02)

    #
    def test_negative_slope(self):
        x = make_energy_axis()
        spectrum = setUpSpectrum(x)
        result = LinBack(x, m=-1.0, b=10.0, xStart=-5, xStop=5, spectrum=spectrum)
        idx = np.argmin(np.abs(x - 0.0))
        # At x=0: -1*(0 - (-5)) + 10 = -5 + 10 = 5
        assert result[idx] == pytest.approx(5.0, abs=0.02)

    #
    def test_clamped_below_xStart(self):
        x = make_energy_axis()
        spectrum = setUpSpectrum(x)
        result = LinBack(x, m=2.0, b=1.0, xStart=-3, xStop=3, spectrum=spectrum)
        # All points below xStart should equal b
        mask = x < -3
        np.testing.assert_allclose(result[mask], 1.0)

    #
    def test_clamped_above_xStop(self):
        x = make_energy_axis()
        spectrum = setUpSpectrum(x)
        result = LinBack(x, m=2.0, b=1.0, xStart=-3, xStop=3, spectrum=spectrum)
        # All points above xStop should equal m*(xStop - xStart) + b = 2*6 + 1 = 13
        mask = x > 3
        np.testing.assert_allclose(result[mask], 13.0)

    #
    def test_full_range_no_clamping(self):
        x = make_energy_axis()
        spectrum = setUpSpectrum(x)
        result = LinBack(x, m=0.5, b=0.0, xStart=x[0], xStop=x[-1], spectrum=spectrum)
        expected = 0.5 * (x - x[0])
        np.testing.assert_allclose(result, expected, atol=1e-12)

    #
    def test_xStart_ge_xStop_raises(self):
        x = make_energy_axis()
        spectrum = setUpSpectrum(x)
        with pytest.raises(ValueError, match="xStart < xStop"):
            LinBack(x, m=1.0, b=0.0, xStart=5, xStop=5, spectrum=spectrum)
        with pytest.raises(ValueError, match="xStart < xStop"):
            LinBack(x, m=1.0, b=0.0, xStart=5, xStop=3, spectrum=spectrum)

    #
    def test_declining_axis(self):
        """Works correctly when energy axis is declining."""

        x = np.linspace(10, -10, 2001)
        spectrum = 10 * np.exp(-0.5 * (x / 2) ** 2)
        result = LinBack(x, m=1.0, b=0.0, xStart=-3, xStop=3, spectrum=spectrum)
        # At x=0: m*(0 - (-3)) + 0 = 3
        idx = np.argmin(np.abs(x - 0.0))
        assert result[idx] == pytest.approx(3.0, abs=0.02)
        # Below xStart (x < -3): clamped at b = 0
        np.testing.assert_allclose(result[x < -3], 0.0)
        # Above xStop (x > 3): clamped at m*(3-(-3)) + 0 = 6
        np.testing.assert_allclose(result[x > 3], 6.0)


#
#
class TestGauss:
    #
    def test_peak_at_center(self):
        x = make_energy_axis()
        result = Gauss(x, A=5.0, x0=0.0, SD=1.0)
        assert result[len(x) // 2] == pytest.approx(5.0, rel=1e-6)

    #
    def test_symmetry(self):
        x = make_energy_axis()
        result = Gauss(x, A=1.0, x0=0.0, SD=1.0)
        np.testing.assert_allclose(result, result[::-1], atol=1e-12)

    #
    def test_fwhm(self):
        """FWHM = 2*sqrt(2*ln2) * SD ≈ 2.3548 * SD."""

        x = make_energy_axis()
        SD = 2.0
        result = Gauss(x, A=1.0, x0=0.0, SD=SD)
        half_max_indices = np.where(result >= 0.5)[0]
        fwhm_measured = x[half_max_indices[-1]] - x[half_max_indices[0]]
        fwhm_expected = 2 * np.sqrt(2 * np.log(2)) * SD
        assert fwhm_measured == pytest.approx(fwhm_expected, abs=0.02)

    #
    def test_offset_center(self):
        x = make_energy_axis()
        result = Gauss(x, A=3.0, x0=2.0, SD=1.0)
        peak_idx = np.argmax(result)
        assert x[peak_idx] == pytest.approx(2.0, abs=0.01)

    #
    def test_value_at_one_sigma(self):
        """At x = x0 ± SD, value should be A * exp(-0.5) ≈ 0.6065*A."""

        x = make_energy_axis()
        SD = 1.0
        result = Gauss(x, A=1.0, x0=0.0, SD=SD)
        idx = np.argmin(np.abs(x - SD))
        assert result[idx] == pytest.approx(np.exp(-0.5), abs=1e-3)


#
#
class TestGaussAsym:
    #
    def test_symmetric_when_ratio_1(self):
        x = make_energy_axis()
        sym = Gauss(x, A=1.0, x0=0.0, SD=1.5)
        asym = GaussAsym(x, A=1.0, x0=0.0, SD=1.5, ratio=1.0)
        np.testing.assert_allclose(asym, sym, atol=1e-12)

    #
    def test_peak_at_center(self):
        x = make_energy_axis()
        result = GaussAsym(x, A=5.0, x0=0.0, SD=1.0, ratio=2.0)
        peak_idx = np.argmax(result)
        assert x[peak_idx] == pytest.approx(0.0, abs=0.01)
        assert result[peak_idx] == pytest.approx(5.0, rel=1e-6)

    #
    def test_asymmetry_direction(self):
        """Ratio > 1 means broader on high-energy side."""

        x = make_energy_axis()
        result = GaussAsym(x, A=1.0, x0=0.0, SD=1.0, ratio=2.0)
        center = len(x) // 2
        left_area = np.sum(result[:center])
        right_area = np.sum(result[center:])
        assert right_area > left_area


#
#
class TestLorentz:
    #
    def test_peak_at_center(self):
        x = make_energy_axis()
        result = Lorentz(x, A=7.0, x0=0.0, W=2.0)
        assert result[len(x) // 2] == pytest.approx(7.0, rel=1e-6)

    #
    def test_symmetry(self):
        x = make_energy_axis()
        result = Lorentz(x, A=1.0, x0=0.0, W=2.0)
        np.testing.assert_allclose(result, result[::-1], atol=1e-12)

    #
    def test_half_max_at_W_over_2(self):
        """At x = x0 ± W/2, value should be A/2."""

        x = make_energy_axis()
        W = 2.0
        result = Lorentz(x, A=1.0, x0=0.0, W=W)
        idx = np.argmin(np.abs(x - W / 2))
        assert result[idx] == pytest.approx(0.5, abs=1e-3)

    #
    def test_heavier_tails_than_gauss(self):
        """Lorentzian should have more intensity in the tails than Gaussian."""

        x = make_energy_axis()
        lorentz = Lorentz(x, A=1.0, x0=0.0, W=2.0)
        gauss = Gauss(x, A=1.0, x0=0.0, SD=2.0 / 2.3548)
        # Compare at 3*FWHM from center
        tail_idx = np.argmin(np.abs(x - 6.0))
        assert lorentz[tail_idx] > gauss[tail_idx]


#
#
class TestVoigt:
    #
    def test_peak_at_center(self):
        x = make_energy_axis()
        result = Voigt(x, A=4.0, x0=0.0, SD=1.0, W=1.0)
        peak_idx = np.argmax(result)
        assert x[peak_idx] == pytest.approx(0.0, abs=0.01)
        assert result[peak_idx] == pytest.approx(4.0, rel=1e-3)

    #
    def test_symmetry(self):
        x = make_energy_axis()
        result = Voigt(x, A=1.0, x0=0.0, SD=1.0, W=1.0)
        np.testing.assert_allclose(result, result[::-1], atol=1e-10)

    #
    def test_wider_than_components(self):
        """Voigt FWHM > max(Gaussian FWHM, Lorentzian FWHM)."""

        x = make_energy_axis()
        SD, W = 1.0, 1.0
        voigt = Voigt(x, A=1.0, x0=0.0, SD=SD, W=W)
        half_max = np.where(voigt >= 0.5)[0]
        voigt_fwhm = x[half_max[-1]] - x[half_max[0]]
        gauss_fwhm = 2.3548 * SD
        assert voigt_fwhm > gauss_fwhm
        assert voigt_fwhm > W


#
#
class TestGLS:
    #
    def test_peak_at_center(self):
        x = make_energy_axis()
        result = GLS(x, A=3.0, x0=0.0, F=1.0, m=0.3)
        peak_idx = np.argmax(result)
        assert x[peak_idx] == pytest.approx(0.0, abs=0.01)

    #
    def test_pure_gaussian_m0(self):
        """m=0 should give pure Gaussian shape."""

        x = make_energy_axis()
        result = GLS(x, A=5.0, x0=0.0, F=1.0, m=0.0)
        expected = 5.0 * np.exp(-((x / 1.0) ** 2) * 4 * np.log(2))
        np.testing.assert_allclose(result, expected, atol=1e-12)

    #
    def test_peak_value_at_center_m0(self):
        """At x=x0 with m=0, value should be A."""

        x = make_energy_axis()
        result = GLS(x, A=5.0, x0=0.0, F=1.0, m=0.0)
        assert result[len(x) // 2] == pytest.approx(5.0, rel=1e-6)

    #
    def test_symmetry(self):
        x = make_energy_axis()
        result = GLS(x, A=1.0, x0=0.0, F=1.5, m=0.5)
        np.testing.assert_allclose(result, result[::-1], atol=1e-12)


#
#
class TestGLP:
    #
    def test_peak_at_center(self):
        x = make_energy_axis()
        result = GLP(x, A=3.0, x0=0.0, F=1.0, m=0.3)
        assert result[len(x) // 2] == pytest.approx(3.0, rel=1e-6)

    #
    def test_pure_gaussian_m0(self):
        """m=0 should give pure Gaussian shape."""

        x = make_energy_axis()
        result = GLP(x, A=5.0, x0=0.0, F=1.0, m=0.0)
        expected = 5.0 * np.exp(-((x / 1.0) ** 2) * 4 * np.log(2))
        np.testing.assert_allclose(result, expected, atol=1e-12)

    #
    def test_pure_lorentzian_m1(self):
        """m=1 should give pure Lorentzian shape."""

        x = make_energy_axis()
        result = GLP(x, A=5.0, x0=0.0, F=1.0, m=1.0)
        expected = 5.0 / (1 + 4 * (x / 1.0) ** 2)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    #
    def test_symmetry(self):
        x = make_energy_axis()
        result = GLP(x, A=1.0, x0=0.0, F=1.5, m=0.5)
        np.testing.assert_allclose(result, result[::-1], atol=1e-12)

    #
    def test_offset_center(self):
        x = make_energy_axis()
        result = GLP(x, A=2.0, x0=3.0, F=1.0, m=0.3)
        peak_idx = np.argmax(result)
        assert x[peak_idx] == pytest.approx(3.0, abs=0.01)


#
#
class TestDS:
    #
    def test_lorentzian_at_alpha0(self):
        """Alpha=0 should reduce to a Lorentzian-like shape."""

        x = make_energy_axis()
        result = DS(x, A=1.0, x0=0.0, F=1.0, alpha=0.0)
        lorentz = 1.0 / (1.0 + (x / 1.0) ** 2)
        np.testing.assert_allclose(result, lorentz, atol=1e-12)

    #
    def test_asymmetry(self):
        """Nonzero alpha produces an asymmetric lineshape."""

        x = make_energy_axis()
        result = DS(x, A=1.0, x0=0.0, F=1.0, alpha=0.15)
        center = len(x) // 2
        # DS tail is on the high-binding-energy (low-KE) side = left side
        # The peak is slightly shifted, so compare integrated weight
        assert not np.allclose(result[:center], result[::-1][:center], atol=1e-3)

    #
    def test_peak_near_center(self):
        """Peak should be near x0 for small alpha."""

        x = make_energy_axis()
        result = DS(x, A=1.0, x0=0.0, F=1.0, alpha=0.1)
        peak_idx = np.argmax(result)
        assert x[peak_idx] == pytest.approx(0.0, abs=0.2)


if __name__ == "__main__":
    pytest.main([__file__])
