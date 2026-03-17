"""Unit tests for time dynamics functions.

Each function is tested for:
- Causality: f(t) = 0 for t < t0
- Known values at specific time points
- Limiting/special cases
"""

import numpy as np
import pytest

from trspecfit.functions.time import (
    erfFun,
    expFun,
    linFun,
    none,
    sinDivX,
    sinFun,
    sqrtFun,
)


#
def setUp():
    """Standard time axis spanning before and after t0."""

    return np.linspace(-5, 50, 5501)


#
#
class TestNone:
    #
    def test_returns_zeros(self):
        t = setUp()
        result = none(t)
        np.testing.assert_allclose(result, 0.0)
        assert result.shape == t.shape


#
#
class TestLinFun:
    #
    def test_zero_before_t0(self):
        t = setUp()
        result = linFun(t, m=2.0, t0=0.0, y0=1.0)
        np.testing.assert_allclose(result[t < 0], 0.0)

    #
    def test_value_at_t0(self):
        t = setUp()
        result = linFun(t, m=2.0, t0=0.0, y0=1.0)
        idx = np.argmin(np.abs(t - 0.0))
        assert result[idx] == pytest.approx(1.0, abs=0.01)

    #
    def test_slope(self):
        t = setUp()
        m = 3.0
        result = linFun(t, m=m, t0=0.0, y0=0.0)
        # At t=10, value should be m*10 = 30
        idx = np.argmin(np.abs(t - 10.0))
        assert result[idx] == pytest.approx(30.0, abs=0.1)

    #
    def test_offset_t0(self):
        t = setUp()
        result = linFun(t, m=1.0, t0=5.0, y0=2.0)
        # Zero before t0=5
        np.testing.assert_allclose(result[t < 5.0], 0.0)
        # At t=10: m*(10-5) + y0 = 7
        idx = np.argmin(np.abs(t - 10.0))
        assert result[idx] == pytest.approx(7.0, abs=0.1)


#
#
class TestExpFun:
    #
    def test_zero_before_t0(self):
        t = setUp()
        result = expFun(t, A=1.0, tau=5.0, t0=0.0, y0=0.0)
        np.testing.assert_allclose(result[t < 0], 0.0)

    #
    def test_value_at_t0(self):
        t = setUp()
        result = expFun(t, A=3.0, tau=5.0, t0=0.0, y0=1.0)
        idx = np.argmin(np.abs(t - 0.0))
        assert result[idx] == pytest.approx(4.0, abs=0.01)  # A + y0

    #
    def test_decay_to_y0(self):
        """At t >> tau, value approaches y0."""

        t = setUp()
        result = expFun(t, A=5.0, tau=2.0, t0=0.0, y0=1.0)
        assert result[-1] == pytest.approx(1.0, abs=1e-6)

    #
    def test_value_at_one_tau(self):
        """At t = t0 + tau, value = A*exp(-1) + y0."""

        t = setUp()
        tau = 5.0
        result = expFun(t, A=1.0, tau=tau, t0=0.0, y0=0.0)
        idx = np.argmin(np.abs(t - tau))
        assert result[idx] == pytest.approx(np.exp(-1), abs=1e-3)

    #
    def test_negative_amplitude_rise(self):
        """A < 0 gives a rise from y0+A toward y0."""

        t = setUp()
        result = expFun(t, A=-2.0, tau=5.0, t0=0.0, y0=0.0)
        idx_t0 = np.argmin(np.abs(t - 0.0))
        assert result[idx_t0] == pytest.approx(-2.0, abs=0.01)
        assert result[-1] == pytest.approx(0.0, abs=1e-3)


#
#
class TestSinFun:
    #
    def test_zero_before_t0(self):
        t = setUp()
        result = sinFun(t, A=1.0, f=0.5, phi=0.0, t0=0.0, y0=0.0)
        np.testing.assert_allclose(result[t < 0], 0.0)

    #
    def test_frequency(self):
        """Period = 1/f. At t = t0 + 1/f, sine completes one cycle."""

        t = setUp()
        f = 0.5
        result = sinFun(t, A=1.0, f=f, phi=0.0, t0=0.0, y0=0.0)
        # At t = 1/(4f), should be at maximum (A)
        idx = np.argmin(np.abs(t - 1 / (4 * f)))
        assert result[idx] == pytest.approx(1.0, abs=0.02)

    #
    def test_phase_shift(self):
        """Phi = pi/2 turns sin into cos (starts at maximum)."""

        t = setUp()
        result = sinFun(t, A=1.0, f=0.5, phi=np.pi / 2, t0=0.0, y0=0.0)
        idx_t0 = np.argmin(np.abs(t - 0.0))
        assert result[idx_t0] == pytest.approx(1.0, abs=0.02)

    #
    def test_offset(self):
        """y0 shifts the oscillation center."""

        t = setUp()
        result = sinFun(t, A=1.0, f=0.5, phi=0.0, t0=0.0, y0=3.0)
        # Mean of oscillation should be ~y0 over full cycles
        # At t0, sin=0 so value = y0
        idx_t0 = np.argmin(np.abs(t - 0.0))
        assert result[idx_t0] == pytest.approx(3.0, abs=0.02)


#
#
class TestSinDivX:
    #
    def test_zero_before_t0(self):
        t = setUp()
        result = sinDivX(t, A=1.0, f=0.5, t0=0.0, y0=0.0)
        np.testing.assert_allclose(result[t < 0], 0.0)

    #
    def test_decaying_envelope(self):
        """Amplitude should decrease over time (sinc envelope)."""

        t = setUp()
        result = sinDivX(t, A=1.0, f=0.5, t0=0.0, y0=0.0)
        active = result[t > 0.5]  # skip near t0 where sinc diverges
        peaks = np.abs(active[1:-1])[
            (active[1:-1] > active[:-2]) & (active[1:-1] > active[2:])
        ]
        if len(peaks) >= 2:
            assert peaks[-1] < peaks[0]

    #
    def test_value_at_t0(self):
        """At t=t0, sinc(0)=1 so value should be A + y0."""

        t = setUp()
        A = 1.5
        y0 = 0.2
        result = sinDivX(t, A=A, f=0.5, t0=0.0, y0=y0)
        idx_t0 = np.argmin(np.abs(t - 0.0))
        assert result[idx_t0] == pytest.approx(A + y0, abs=0.01)

    #
    def test_first_zero_location(self):
        """First sinc zero after t0 occurs at t = t0 + 1/(2f)."""

        t = setUp()
        f = 0.5
        t0 = 0.0
        y0 = 0.3
        result = sinDivX(t, A=1.0, f=f, t0=t0, y0=y0)
        t_zero = t0 + 1.0 / (2.0 * f)
        idx_zero = np.argmin(np.abs(t - t_zero))
        assert result[idx_zero] == pytest.approx(y0, abs=0.01)


#
#
class TestErfFun:
    #
    def test_midpoint_value(self):
        """At t = t0, erf(0) = 0, so value = A/2 + y0."""

        t = setUp()
        result = erfFun(t, A=4.0, SD=1.0, t0=10.0, y0=1.0)
        idx = np.argmin(np.abs(t - 10.0))
        assert result[idx] == pytest.approx(3.0, abs=0.01)  # 4/2 + 1

    #
    def test_asymptotic_low(self):
        """For t << t0, erf → -1, so value → y0."""

        t = setUp()
        result = erfFun(t, A=4.0, SD=1.0, t0=10.0, y0=1.0)
        assert result[0] == pytest.approx(1.0, abs=1e-6)

    #
    def test_asymptotic_high(self):
        """For t >> t0, erf → 1, so value → A + y0."""

        t = setUp()
        result = erfFun(t, A=4.0, SD=1.0, t0=10.0, y0=1.0)
        assert result[-1] == pytest.approx(5.0, abs=1e-3)

    #
    def test_monotonic_increase(self):
        """Error function rise should be monotonically increasing."""

        t = setUp()
        result = erfFun(t, A=2.0, SD=1.0, t0=10.0, y0=0.0)
        assert np.all(np.diff(result) >= -1e-12)

    #
    def test_note_no_hard_t0_cutoff(self):
        """erfFun does NOT have a hard t0 cutoff — it's a smooth sigmoid.
        Value at t << t0 approaches y0 but is never exactly zero."""

        t = setUp()
        result = erfFun(t, A=4.0, SD=1.0, t0=10.0, y0=0.0)
        assert result[0] == pytest.approx(0.0, abs=1e-6)
        # Slight non-zero values near t0 are expected
        idx_before = np.argmin(np.abs(t - 8.0))  # 2 SD before
        assert result[idx_before] > 0.0


#
#
class TestSqrtFun:
    #
    def test_zero_before_t0(self):
        """
        sqrtFun uses .clip(0) so t<t0 gives y0 (not zero)
        Note sqrt(0)=0 so result = A*0 + y0).
        """

        t = setUp()
        result = sqrtFun(t, A=1.0, t0=0.0, y0=0.0)
        np.testing.assert_allclose(result[t < 0], 0.0, atol=1e-12)

    #
    def test_value_at_t0(self):
        t = setUp()
        result = sqrtFun(t, A=2.0, t0=0.0, y0=3.0)
        idx = np.argmin(np.abs(t - 0.0))
        assert result[idx] == pytest.approx(3.0, abs=0.01)  # A*sqrt(0) + y0

    #
    def test_known_value(self):
        """At t=4 with t0=0: A*sqrt(4) + y0 = 2A + y0."""

        t = setUp()
        result = sqrtFun(t, A=3.0, t0=0.0, y0=1.0)
        idx = np.argmin(np.abs(t - 4.0))
        assert result[idx] == pytest.approx(7.0, abs=0.1)  # 3*2 + 1

    #
    def test_monotonic_after_t0(self):
        t = setUp()
        result = sqrtFun(t, A=1.0, t0=0.0, y0=0.0)
        active = result[t >= 0]
        assert np.all(np.diff(active) >= -1e-12)

    #
    def test_offset_y0(self):
        """y0 shifts everything (including before t0 via clip behavior)."""

        t = setUp()
        result = sqrtFun(t, A=1.0, t0=0.0, y0=5.0)
        # Before t0: A*sqrt(0) + y0 = y0
        np.testing.assert_allclose(result[t <= 0], 5.0, atol=0.01)


if __name__ == "__main__":
    pytest.main([__file__])
