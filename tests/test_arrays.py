"""Tests for trspecfit.utils.arrays — running_mean."""

import numpy as np

from trspecfit.utils.arrays import running_mean


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
