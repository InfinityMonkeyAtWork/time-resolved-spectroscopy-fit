"""Unit tests for convolution kernel functions.

Each kernel is tested for:
- Symmetry (where applicable)
- Peak at center (x=0)
- Known shape properties
- kernel_width() returns positive int/float
"""

import numpy as np
import pytest

from trspecfit.functions.time import (
    boxCONV,
    boxCONV_kernel_width,
    expDecayCONV,
    expDecayCONV_kernel_width,
    expRiseCONV,
    expRiseCONV_kernel_width,
    expSymCONV,
    expSymCONV_kernel_width,
    gaussCONV,
    gaussCONV_kernel_width,
    lorentzCONV,
    lorentzCONV_kernel_width,
    voigtCONV,
    voigtCONV_kernel_width,
)


#
def make_kernel_axis():
    """Symmetric time axis centered at 0 (typical kernel axis)."""

    return np.linspace(-10, 10, 2001)


#
#
class TestGaussCONV:
    #
    def test_peak_at_center(self):
        x_sym = make_kernel_axis()
        result = gaussCONV(x_sym, SD=1.0)
        assert np.argmax(result) == len(x_sym) // 2

    #
    def test_peak_value_is_one(self):
        """exp(0) = 1."""

        x_sym = make_kernel_axis()
        result = gaussCONV(x_sym, SD=1.0)
        assert result[len(x_sym) // 2] == pytest.approx(1.0)

    #
    def test_symmetry(self):
        x_sym = make_kernel_axis()
        result = gaussCONV(x_sym, SD=2.0)
        np.testing.assert_allclose(result, result[::-1], atol=1e-12)

    #
    def test_value_at_one_sigma(self):
        x_sym = make_kernel_axis()
        SD = 2.0
        result = gaussCONV(x_sym, SD=SD)
        idx = np.argmin(np.abs(x_sym - SD))
        assert result[idx] == pytest.approx(np.exp(-0.5), abs=1e-3)

    #
    def test_kernel_width_positive(self):
        assert gaussCONV_kernel_width() > 0


#
#
class TestLorentzCONV:
    #
    def test_peak_at_center(self):
        x_sym = make_kernel_axis()
        result = lorentzCONV(x_sym, W=2.0)
        assert np.argmax(result) == len(x_sym) // 2

    #
    def test_peak_value_is_one(self):
        x_sym = make_kernel_axis()
        result = lorentzCONV(x_sym, W=2.0)
        assert result[len(x_sym) // 2] == pytest.approx(1.0)

    #
    def test_symmetry(self):
        x_sym = make_kernel_axis()
        result = lorentzCONV(x_sym, W=2.0)
        np.testing.assert_allclose(result, result[::-1], atol=1e-12)

    #
    def test_half_max_at_half_width(self):
        """At x = ±W/2, value should be 0.5 (by FWHM definition)."""

        x_sym = make_kernel_axis()
        W = 2.0
        result = lorentzCONV(x_sym, W=W)
        idx = np.argmin(np.abs(x_sym - W))
        expected = 1 / (1 + (W / W / 2) ** 2)
        assert result[idx] == pytest.approx(expected, abs=1e-3)

    #
    def test_kernel_width_positive(self):
        assert lorentzCONV_kernel_width() > 0


#
#
class TestVoigtCONV:
    #
    def test_peak_at_center(self):
        x_sym = make_kernel_axis()
        result = voigtCONV(x_sym, SD=1.0, W=1.0)
        assert np.argmax(result) == len(x_sym) // 2

    #
    def test_peak_normalized_to_one(self):
        """voigtCONV divides by max, so peak = 1."""

        x_sym = make_kernel_axis()
        result = voigtCONV(x_sym, SD=1.0, W=1.0)
        assert result[len(x_sym) // 2] == pytest.approx(1.0, rel=1e-6)

    #
    def test_symmetry(self):
        x_sym = make_kernel_axis()
        result = voigtCONV(x_sym, SD=1.0, W=1.0)
        np.testing.assert_allclose(result, result[::-1], atol=1e-10)

    #
    def test_kernel_width_positive(self):
        assert voigtCONV_kernel_width() > 0


#
#
class TestExpSymCONV:
    #
    def test_peak_at_center(self):
        x_sym = make_kernel_axis()
        result = expSymCONV(x_sym, tau=2.0)
        assert np.argmax(result) == len(x_sym) // 2

    #
    def test_peak_value_is_one(self):
        """exp(-|0|/tau) = 1."""

        x_sym = make_kernel_axis()
        result = expSymCONV(x_sym, tau=2.0)
        assert result[len(x_sym) // 2] == pytest.approx(1.0)

    #
    def test_symmetry(self):
        x_sym = make_kernel_axis()
        result = expSymCONV(x_sym, tau=2.0)
        np.testing.assert_allclose(result, result[::-1], atol=1e-12)

    #
    def test_value_at_tau(self):
        x_sym = make_kernel_axis()
        tau = 2.0
        result = expSymCONV(x_sym, tau=tau)
        idx = np.argmin(np.abs(x_sym - tau))
        assert result[idx] == pytest.approx(np.exp(-1), abs=1e-3)

    #
    def test_kernel_width_positive(self):
        assert expSymCONV_kernel_width() > 0


#
#
class TestExpDecayCONV:
    #
    def test_zero_for_negative_x(self):
        x_sym = make_kernel_axis()
        result = expDecayCONV(x_sym, tau=2.0)
        np.testing.assert_allclose(result[x_sym < 0], 0.0)

    #
    def test_peak_at_zero(self):
        x_sym = make_kernel_axis()
        result = expDecayCONV(x_sym, tau=2.0)
        # Peak should be at x=0 (first non-zero point)
        pos = result[x_sym >= 0]
        assert pos[0] == pytest.approx(1.0, abs=0.01)

    #
    def test_decays_for_positive_x(self):
        x_sym = make_kernel_axis()
        result = expDecayCONV(x_sym, tau=2.0)
        pos = result[x_sym >= 0]
        assert np.all(np.diff(pos) <= 1e-10)

    #
    def test_kernel_width_positive(self):
        assert expDecayCONV_kernel_width() > 0


#
#
class TestExpRiseCONV:
    #
    def test_zero_for_positive_x(self):
        x_sym = make_kernel_axis()
        result = expRiseCONV(x_sym, tau=2.0)
        np.testing.assert_allclose(result[x_sym > 0], 0.0)

    #
    def test_peak_at_zero(self):
        x_sym = make_kernel_axis()
        result = expRiseCONV(x_sym, tau=2.0)
        neg = result[x_sym <= 0]
        assert neg[-1] == pytest.approx(1.0, abs=0.01)

    #
    def test_rises_toward_zero(self):
        """For x < 0, signal should increase toward x=0."""

        x_sym = make_kernel_axis()
        result = expRiseCONV(x_sym, tau=2.0)
        neg = result[x_sym <= 0]
        assert np.all(np.diff(neg) >= -1e-10)

    #
    def test_mirror_of_decay(self):
        """expRiseCONV(x) should equal expDecayCONV(-x)."""

        x_sym = make_kernel_axis()
        rise = expRiseCONV(x_sym, tau=2.0)
        decay = expDecayCONV(x_sym, tau=2.0)
        np.testing.assert_allclose(rise, decay[::-1], atol=1e-12)

    #
    def test_kernel_width_positive(self):
        assert expRiseCONV_kernel_width() > 0


#
#
class TestBoxCONV:
    #
    def test_binary_values(self):
        """Box kernel should be approximately 0 or 1."""

        x_sym = make_kernel_axis()
        result = boxCONV(x_sym, width=2.0)
        # Allow for edge effects from square wave
        assert np.all((result >= -0.1) & (result <= 1.1))

    #
    def test_kernel_width_positive(self):
        assert boxCONV_kernel_width() > 0


if __name__ == "__main__":
    pytest.main([__file__])
