"""Unit tests for convolution kernel functions.

Each kernel is tested for:
- Symmetry (where applicable)
- Peak at center (x=0)
- Known shape properties

Edge-mass companions (CONV_EDGE_MASS) are validated against numerical
integration of the kernel bodies, and both registries are checked for
completeness so a kernel can never ship without its companion.
"""

import numpy as np
import pytest
from scipy.integrate import quad

from trspecfit.functions import time as fcts_time
from trspecfit.functions.time import (
    CONV_EDGE_MASS,
    boxCONV,
    expDecayCONV,
    expRiseCONV,
    expSymCONV,
    gaussCONV,
)

# one representative parameter set per kernel, used by the companion
# consistency test; extend when adding a kernel
_KERNEL_TEST_PARAMS = {
    "gaussCONV": (1.7,),
    "expSymCONV": (2.3,),
    "expDecayCONV": (2.3,),
    "expRiseCONV": (2.3,),
    "boxCONV": (3.0,),
}


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
    def test_half_max_at_half_width(self):
        """FWHM = 2.355 * SD. At x = ±FWHM/2, value should be 0.5."""

        x_sym = make_kernel_axis()
        SD = 2.0
        result = gaussCONV(x_sym, SD=SD)
        fwhm_half = np.sqrt(2 * np.log(2)) * SD  # FWHM/2
        idx = np.argmin(np.abs(x_sym - fwhm_half))
        assert result[idx] == pytest.approx(0.5, abs=0.01)

    #
    def test_decays_monotonically(self):
        """Should decay monotonically away from center on both sides."""

        x_sym = make_kernel_axis()
        result = gaussCONV(x_sym, SD=2.0)
        center = len(x_sym) // 2
        # Right side: decreasing
        assert np.all(np.diff(result[center:]) <= 1e-12)
        # Left side: increasing toward center
        assert np.all(np.diff(result[:center]) >= -1e-12)


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
    def test_half_max_at_half_width(self):
        """At x = ±tau*ln(2), value should be 0.5 (half-max of exp(-|x|/tau))."""

        x_sym = make_kernel_axis()
        tau = 2.0
        result = expSymCONV(x_sym, tau=tau)
        half_width = tau * np.log(2)
        idx = np.argmin(np.abs(x_sym - half_width))
        assert result[idx] == pytest.approx(0.5, abs=0.01)

    #
    def test_decays_monotonically(self):
        """Should decay monotonically away from center on both sides."""

        x_sym = make_kernel_axis()
        result = expSymCONV(x_sym, tau=2.0)
        center = len(x_sym) // 2
        assert np.all(np.diff(result[center:]) <= 1e-12)
        assert np.all(np.diff(result[:center]) >= -1e-12)


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
    def test_half_max_at_tau_ln2(self):
        """At x = tau*ln(2), value should be 0.5."""

        x_sym = make_kernel_axis()
        tau = 2.0
        result = expDecayCONV(x_sym, tau=tau)
        half_life = tau * np.log(2)
        idx = np.argmin(np.abs(x_sym - half_life))
        assert result[idx] == pytest.approx(0.5, abs=0.01)


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
    def test_half_max_at_tau_ln2(self):
        """At x = -tau*ln(2), value should be 0.5."""

        x_sym = make_kernel_axis()
        tau = 2.0
        result = expRiseCONV(x_sym, tau=tau)
        half_life = -tau * np.log(2)
        idx = np.argmin(np.abs(x_sym - half_life))
        assert result[idx] == pytest.approx(0.5, abs=0.01)

    #
    def test_mirror_of_decay(self):
        """expRiseCONV(x) should equal expDecayCONV(-x)."""

        x_sym = make_kernel_axis()
        rise = expRiseCONV(x_sym, tau=2.0)
        decay = expDecayCONV(x_sym, tau=2.0)
        np.testing.assert_allclose(rise, decay[::-1], atol=1e-12)


#
#
class TestBoxCONV:
    #
    def test_peak_at_center(self):
        x_sym = make_kernel_axis()
        result = boxCONV(x_sym, width=2.0)
        assert result[len(x_sym) // 2] == pytest.approx(1.0)

    #
    def test_peak_value_is_one(self):
        x_sym = make_kernel_axis()
        result = boxCONV(x_sym, width=2.0)
        assert np.max(result) == pytest.approx(1.0)

    #
    def test_symmetry(self):
        x_sym = make_kernel_axis()
        result = boxCONV(x_sym, width=2.0)
        np.testing.assert_allclose(result, result[::-1], atol=1e-12)

    #
    def test_values_are_exactly_zero_or_one(self):
        """Box kernel values should be exactly 0.0 or 1.0."""

        x_sym = make_kernel_axis()
        result = boxCONV(x_sym, width=2.0)
        assert np.all((result == 0.0) | (result == 1.0))

    #
    def test_wider_width_more_ones(self):
        """Larger width should produce a wider box."""

        x_sym = make_kernel_axis()
        narrow = boxCONV(x_sym, width=1.0)
        wide = boxCONV(x_sym, width=3.0)
        assert np.sum(wide > 0.5) > np.sum(narrow > 0.5)


#
#
class TestConvEdgeMassCompanions:
    """Edge-mass companions integrate the exact kernel bodies."""

    #
    @pytest.mark.parametrize("kernel_name", sorted(CONV_EDGE_MASS))
    def test_masses_match_numerical_integration(self, kernel_name):
        """Companion masses equal quad integrals of the kernel body.

        M_L(a) must equal the body's integral over [a, inf) and M_R(b)
        the integral over (-inf, b] — the exterior masses used for
        edge-value padding. Guards any future companion against a
        normalization or left/right mismatch with its kernel body.
        """

        body = getattr(fcts_time, kernel_name)
        params = _KERNEL_TEST_PARAMS[kernel_name]
        companion = CONV_EDGE_MASS[kernel_name]
        # generous finite integration bounds: all kernels are negligible
        # beyond +-50 for the test parameter values
        bound = 50.0
        dt_left = np.array([0.0, 0.3, 2.0, 7.0])
        dt_right = -dt_left
        mass_left, mass_right = companion(dt_left, dt_right, *params)
        for i, a in enumerate(dt_left):
            expected, _err = quad(
                lambda x: float(body(np.array([x]), *params)[0]), a, bound
            )
            np.testing.assert_allclose(mass_left[i], expected, rtol=1e-6, atol=1e-12)
        for i, b in enumerate(dt_right):
            expected, _err = quad(
                lambda x: float(body(np.array([x]), *params)[0]), -bound, b
            )
            np.testing.assert_allclose(mass_right[i], expected, rtol=1e-6, atol=1e-12)

    #
    @pytest.mark.parametrize("kernel_name", sorted(CONV_EDGE_MASS))
    @pytest.mark.parametrize("bad_value", [0.0, -1.0, np.nan, np.inf])
    def test_nonpositive_params_rejected(self, kernel_name, bad_value):
        """Companions reject nonpositive/non-finite parameters loudly.

        Runtime backstop for expression-driven kernel parameters, which
        bypass the model-load bound validation: boxCONV(width=0) would
        otherwise pass every downstream check and silently become the
        identity operator (the dt=0 diagonal keeps row sums positive).
        """

        companion = CONV_EDGE_MASS[kernel_name]
        dt_left = np.array([0.0, 1.0, 4.0])
        dt_right = -dt_left
        params = (bad_value, *_KERNEL_TEST_PARAMS[kernel_name][1:])
        with pytest.raises(ValueError, match="strictly positive"):
            companion(dt_left, dt_right, *params)

    #
    def test_every_kernel_has_companion(self):
        """Registry completeness: kernels and companions stay in lockstep.

        Every discoverable *CONV function needs a CONV_EDGE_MASS entry
        (mcp path), every entry must reference an existing kernel, and
        the GIR dispatch tables must cover the same kernel set.
        """

        from trspecfit.eval_2d import CONV_EDGE_MASS_DISPATCH, CONV_KERNEL_DISPATCH

        conv_kernels = {
            name
            for name in dir(fcts_time)
            if name.endswith("CONV") and not name.startswith("_")
        }
        assert conv_kernels == set(CONV_EDGE_MASS)
        assert set(CONV_KERNEL_DISPATCH) == set(CONV_EDGE_MASS_DISPATCH)

    #
    def test_test_params_cover_all_companions(self):
        """_KERNEL_TEST_PARAMS stays in sync with the registry."""

        assert set(_KERNEL_TEST_PARAMS) == set(CONV_EDGE_MASS)


if __name__ == "__main__":
    pytest.main([__file__])
