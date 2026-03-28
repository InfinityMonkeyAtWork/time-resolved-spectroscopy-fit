"""Unit tests for profile (auxiliary-axis) functions.

Each function is tested against known analytical values.
"""

import numpy as np
import pytest

from trspecfit.functions.profile import pExpDecay, pGauss, pLinear


#
def make_aux_axis():
    """Standard auxiliary axis (e.g. depth in nm)."""

    return np.linspace(0, 10, 1001)


#
#
class TestPExpDecay:
    #
    def test_value_at_zero(self):
        x = make_aux_axis()
        result = pExpDecay(x, A=5.0, tau=2.0)
        assert result[0] == pytest.approx(5.0, rel=1e-10)

    #
    def test_value_at_one_tau(self):
        x = make_aux_axis()
        tau = 2.0
        result = pExpDecay(x, A=1.0, tau=tau)
        idx = np.argmin(np.abs(x - tau))
        assert result[idx] == pytest.approx(np.exp(-1), abs=1e-3)

    #
    def test_decays_to_zero(self):
        x = make_aux_axis()
        result = pExpDecay(x, A=1.0, tau=0.5)
        assert result[-1] == pytest.approx(0.0, abs=1e-6)

    #
    def test_monotonic_decrease(self):
        x = make_aux_axis()
        result = pExpDecay(x, A=1.0, tau=2.0)
        assert np.all(np.diff(result) <= 1e-12)

    #
    def test_zero_amplitude(self):
        x = make_aux_axis()
        result = pExpDecay(x, A=0.0, tau=2.0)
        np.testing.assert_array_equal(result, 0.0)

    #
    def test_scales_with_amplitude(self):
        x = make_aux_axis()
        r1 = pExpDecay(x, A=1.0, tau=2.0)
        r2 = pExpDecay(x, A=3.0, tau=2.0)
        np.testing.assert_allclose(r2, 3.0 * r1, atol=1e-12)


#
#
class TestPLinear:
    #
    def test_intercept(self):
        x = make_aux_axis()
        result = pLinear(x, m=2.0, b=3.0)
        assert result[0] == pytest.approx(3.0, rel=1e-10)

    #
    def test_slope(self):
        x = make_aux_axis()
        result = pLinear(x, m=2.0, b=0.0)
        idx = np.argmin(np.abs(x - 5.0))
        assert result[idx] == pytest.approx(10.0, abs=0.02)

    #
    def test_zero_slope(self):
        x = make_aux_axis()
        result = pLinear(x, m=0.0, b=7.0)
        np.testing.assert_allclose(result, 7.0)

    #
    def test_negative_slope(self):
        x = make_aux_axis()
        result = pLinear(x, m=-1.0, b=10.0)
        assert np.all(np.diff(result) < 0)

    #
    def test_known_endpoint(self):
        """At x=10: m*10 + b = 2*10 + 1 = 21."""

        x = make_aux_axis()
        result = pLinear(x, m=2.0, b=1.0)
        assert result[-1] == pytest.approx(21.0, abs=0.02)


#
#
class TestPGauss:
    #
    def test_peak_at_center(self):
        x = np.linspace(-5, 15, 2001)
        result = pGauss(x, A=4.0, x0=5.0, SD=2.0)
        peak_idx = np.argmax(result)
        assert x[peak_idx] == pytest.approx(5.0, abs=0.01)
        assert result[peak_idx] == pytest.approx(4.0, rel=1e-6)

    #
    def test_symmetry(self):
        x = np.linspace(-10, 10, 2001)
        result = pGauss(x, A=1.0, x0=0.0, SD=2.0)
        np.testing.assert_allclose(result, result[::-1], atol=1e-12)

    #
    def test_value_at_one_sigma(self):
        x = np.linspace(-10, 10, 2001)
        SD = 2.0
        result = pGauss(x, A=1.0, x0=0.0, SD=SD)
        idx = np.argmin(np.abs(x - SD))
        assert result[idx] == pytest.approx(np.exp(-0.5), abs=1e-3)

    #
    def test_zero_amplitude(self):
        x = np.linspace(-10, 10, 2001)
        result = pGauss(x, A=0.0, x0=0.0, SD=2.0)
        np.testing.assert_array_equal(result, 0.0)

    #
    def test_monotonic_from_center(self):
        """Decreasing on both sides away from center."""

        x = np.linspace(-10, 10, 2001)
        result = pGauss(x, A=1.0, x0=0.0, SD=2.0)
        center = len(x) // 2
        # Left side: increasing toward center
        assert np.all(np.diff(result[:center]) >= -1e-12)
        # Right side: decreasing from center
        assert np.all(np.diff(result[center:]) <= 1e-12)

    #
    def test_near_zero_far_from_center(self):
        x = np.linspace(-10, 10, 2001)
        result = pGauss(x, A=1.0, x0=0.0, SD=0.5)
        assert result[0] == pytest.approx(0.0, abs=1e-10)
        assert result[-1] == pytest.approx(0.0, abs=1e-10)


if __name__ == "__main__":
    pytest.main([__file__])
