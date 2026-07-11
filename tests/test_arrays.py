"""Tests for trspecfit.utils.arrays — running_mean, conv_kernel_support,
sign_change, my_conv."""

import numpy as np
import pytest

from trspecfit.utils.arrays import (
    conv_kernel_support,
    my_conv,
    running_mean,
    sign_change,
)


#
#
class TestMyConv:
    """Tests for my_conv padded convolution."""

    #
    def test_delta_kernel_is_identity(self):
        """A single-sample kernel returns the input signal unchanged."""

        x = np.linspace(0, 10, 21)
        y = np.sin(x)
        np.testing.assert_allclose(my_conv(x, y, np.array([1.0])), y)

    #
    def test_normalization(self):
        """Kernel normalization preserves the level of a constant signal."""

        x = np.linspace(0, 10, 21)
        y = np.full(21, 3.0)
        kernel = np.array([1.0, 2.0, 1.0])
        np.testing.assert_allclose(my_conv(x, y, kernel), y)

    #
    def test_single_element_x_raises(self):
        """Degenerate x axis raises a clear error instead of IndexError.

        Regression: x_arr[1] - x_arr[0] had no length guard.
        """

        with pytest.raises(ValueError, match="at least 2 x samples"):
            my_conv(np.array([0.0]), np.array([1.0]), np.array([1.0]))

    #
    def test_zero_sum_kernel_raises(self):
        """A kernel that sums to zero raises instead of yielding NaN/inf."""

        x = np.linspace(0, 10, 21)
        y = np.sin(x)
        with pytest.raises(ValueError, match="cannot normalize"):
            my_conv(x, y, np.zeros(5))

    #
    def test_nonfinite_kernel_raises(self):
        """A kernel with non-finite entries raises instead of propagating."""

        x = np.linspace(0, 10, 21)
        y = np.sin(x)
        with pytest.raises(ValueError, match="cannot normalize"):
            my_conv(x, y, np.array([1.0, np.nan, 1.0]))


#
#
class TestSignChange:
    """Tests for sign_change zero-crossing detection."""

    #
    def test_all_zero_input_returns_no_changes(self):
        """All-zero input terminates and reports no sign changes.

        Regression: the zero-propagation loop never terminated on
        all-zero input (np.roll can't introduce a non-zero sign).
        """

        result = sign_change(np.zeros(5), ignore_zeros=True)
        np.testing.assert_array_equal(result, np.zeros(5, dtype=int))

    #
    def test_zero_crossing_with_ignore_zeros(self):
        """Zeros between opposite signs count as one crossing."""

        np.testing.assert_array_equal(
            sign_change([1, 0, -1], ignore_zeros=True), [0, 0, 1]
        )

    #
    def test_zero_crossing_without_ignore_zeros(self):
        """ignore_zeros=False treats zero as its own sign."""

        np.testing.assert_array_equal(
            sign_change([1, 0, -1], ignore_zeros=False), [0, 1, 1]
        )

    #
    def test_no_crossing(self):
        """Same-sign input reports no changes."""

        np.testing.assert_array_equal(
            sign_change([1, 2, 0, 3], ignore_zeros=True), [0, 0, 0, 0]
        )


#
#
class TestConvKernelSupport:
    """Tests for the convolution kernel support builder."""

    #
    def test_symmetric_and_odd_length(self):
        """Support is symmetric around 0 with an odd number of samples."""

        axis = conv_kernel_support(3.7, 0.5)
        assert axis.size % 2 == 1
        np.testing.assert_allclose(axis, -axis[::-1])
        assert axis[axis.size // 2] == 0.0

    #
    def test_covers_t_range(self):
        """Support extends to at least ±t_range."""

        axis = conv_kernel_support(3.7, 0.5)
        assert axis.max() >= 3.7
        assert axis.min() <= -3.7

    #
    def test_exact_multiple(self):
        """t_range on the grid gives endpoints exactly at ±t_range."""

        axis = conv_kernel_support(4.0, 0.5)
        np.testing.assert_allclose(axis[0], -4.0)
        np.testing.assert_allclose(axis[-1], 4.0)
        assert axis.size == 17

    #
    def test_minimum_support(self):
        """t_range below one step still yields a 3-sample support."""

        axis = conv_kernel_support(0.01, 0.5)
        np.testing.assert_allclose(axis, [-0.5, 0.0, 0.5])

    #
    def test_grows_with_t_range(self):
        """Doubling t_range widens the support accordingly."""

        narrow = conv_kernel_support(2.0, 0.5)
        wide = conv_kernel_support(4.0, 0.5)
        assert wide.size > narrow.size
        assert wide.max() >= 2 * narrow.max() - 0.5

    #
    def test_nonfinite_range_raises(self):
        """Non-finite t_range raises ValueError."""

        with pytest.raises(ValueError, match="not finite"):
            conv_kernel_support(np.nan, 0.5)
        with pytest.raises(ValueError, match="not finite"):
            conv_kernel_support(np.inf, 0.5)

    #
    def test_bad_step_raises(self):
        """Non-positive or non-finite t_step raises ValueError."""

        with pytest.raises(ValueError, match="positive"):
            conv_kernel_support(1.0, 0.0)
        with pytest.raises(ValueError, match="positive"):
            conv_kernel_support(1.0, -0.5)


#
#
class TestRunningMean:
    """Tests for running_mean smoothing function."""

    #
    def test_constant_signal_unchanged(self):
        """Running mean of a constant signal returns the same constant."""

        x = np.arange(50, dtype=float)
        y = np.full(50, 7.0)
        result = running_mean(x, y, n=5)
        np.testing.assert_allclose(result, 7.0)

    #
    def test_window_one_is_identity(self):
        """Window size 1 returns the original signal."""

        x = np.arange(10, dtype=float)
        y = np.array([1.0, 3.0, 5.0, 2.0, 8.0, 0.0, 4.0, 6.0, 9.0, 7.0])
        result = running_mean(x, y, n=1)
        np.testing.assert_allclose(result, y)

    #
    def test_known_boxcar_average(self):
        """Interior points match hand-computed 3-point average."""

        x = np.arange(5, dtype=float)
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = running_mean(x, y, n=3)
        # interior points: (1+2+3)/3=2, (2+3+4)/3=3, (3+4+5)/3=4
        np.testing.assert_allclose(result[1:4], [2.0, 3.0, 4.0])

    #
    def test_output_length_matches_input(self):
        """Output has same length as input regardless of window size."""

        x = np.arange(20, dtype=float)
        y = np.random.default_rng(42).random(20)
        for n in [1, 3, 5, 7]:
            result = running_mean(x, y, n=n)
            assert result.shape == y.shape

    #
    def test_smoothing_reduces_variance(self):
        """Smoothing a noisy signal should reduce variance."""

        rng = np.random.default_rng(42)
        x = np.arange(100, dtype=float)
        y = np.sin(x / 10) + rng.normal(scale=0.5, size=100)
        smoothed = running_mean(x, y, n=7)
        assert np.var(smoothed) < np.var(y)
