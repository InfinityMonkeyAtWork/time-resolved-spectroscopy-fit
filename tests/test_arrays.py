"""Tests for trspecfit.utils.arrays — running_mean, kernel-matrix
convolution, sign_change, my_conv, resolve_time_selection."""

import numpy as np
import pytest
from scipy.stats import norm

from trspecfit.functions.time import (
    CONV_EDGE_MASS,
    erfFun,
    expDecayCONV,
    expFun,
    expRiseCONV,
    gaussCONV,
)
from trspecfit.utils.arrays import (
    conv_matrix_apply,
    conv_matrix_operator,
    my_conv,
    resolve_time_selection,
    running_mean,
    sign_change,
)


#
def _conv_matrix(t, kernel_fct, *kernel_pars, y):
    """Build the operator for t and convolve y with the given kernel."""

    operator = conv_matrix_operator(t)
    kernel_values = kernel_fct(operator.dt_unique, *kernel_pars)
    edge_mass = CONV_EDGE_MASS[kernel_fct.__name__]
    mass_left, mass_right = edge_mass(operator.dt_left, operator.dt_right, *kernel_pars)
    return conv_matrix_apply(operator, kernel_values, mass_left, mass_right, y)


#
def _dense_reference(t, y_fct, kernel_fct, *kernel_pars, t_step=0.005, margin=80.0):
    """Continuous-convolution reference on a dense uniform grid.

    Evaluates the signal analytically on a fine grid extending well past
    both ends of t (so edge effects cannot reach the window), convolves
    with the normalized kernel, and interpolates back onto t.
    """

    t_fine = np.arange(t[0] - margin, t[-1] + margin, t_step)
    y_fine = y_fct(t_fine)
    kernel = kernel_fct(t_fine - t_fine[t_fine.size // 2], *kernel_pars)
    kernel = kernel / kernel.sum()
    y_conv_fine = np.convolve(y_fine, kernel, mode="same")
    return np.interp(t, t_fine, y_conv_fine)


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
class TestConvMatrixOperator:
    """Tests for the theta-independent operator builder."""

    #
    def test_shapes(self):
        """Operator arrays are interior-only: (n, n) matrix, (n,) vectors."""

        t = np.linspace(0, 10, 21)
        operator = conv_matrix_operator(t)
        assert operator.gather_idx.shape == (t.size, t.size)
        assert operator.quad_weights.shape == (t.size,)
        assert operator.dt_left.shape == (t.size,)
        assert operator.dt_right.shape == (t.size,)

    #
    def test_dt_matrix_reconstructs(self):
        """Gathered dt values reconstruct to t_i - t_j."""

        t = np.array([0.0, 1.0, 3.0, 7.0])
        operator = conv_matrix_operator(t)
        dt = operator.dt_unique[operator.gather_idx]
        np.testing.assert_allclose(dt, t[:, None] - t[None, :])

    #
    def test_edge_mass_abscissae(self):
        """dt_left/dt_right are the offsets from the window edges."""

        t = np.array([0.0, 1.0, 3.0, 7.0])
        operator = conv_matrix_operator(t)
        np.testing.assert_allclose(operator.dt_left, t - t[0])
        np.testing.assert_allclose(operator.dt_right, t - t[-1])
        assert np.all(operator.dt_left >= 0)
        assert np.all(operator.dt_right <= 0)

    #
    def test_dt_unique_deduplicates_uniform_axis(self):
        """On a uniform axis dt values collapse to O(n) unique entries."""

        t = np.arange(0, 10, 0.5)
        operator = conv_matrix_operator(t)
        # all row/column offsets on a uniform grid: 2*n_t - 1 values
        assert operator.dt_unique.size == 2 * t.size - 1
        assert operator.dt_unique.size < operator.gather_idx.size

    #
    def test_trapezoid_weights(self):
        """True trapezoid weights: half-cells at both window edges.

        The exterior beyond the edge half-cells is covered exactly by
        the analytic edge masses; full end cells (np.gradient) would
        double-count against them.
        """

        t = np.arange(0, 10, 0.5)
        operator = conv_matrix_operator(t)
        np.testing.assert_allclose(operator.quad_weights[1:-1], 0.5)
        np.testing.assert_allclose(operator.quad_weights[[0, -1]], 0.25)
        # non-uniform: interior weights are the centered half-spans
        t_nu = np.array([0.0, 1.0, 3.0, 7.0])
        op_nu = conv_matrix_operator(t_nu)
        np.testing.assert_allclose(op_nu.quad_weights, [0.5, 1.5, 3.0, 2.0])

    #
    def test_single_point_axis_raises(self):
        """A 1-point axis raises a clear error."""

        with pytest.raises(ValueError, match="at least 2 points"):
            conv_matrix_operator(np.array([0.0]))

    #
    def test_non_monotonic_axis_raises(self):
        """A non-increasing axis raises instead of silently misconvolving."""

        with pytest.raises(ValueError, match="strictly increasing"):
            conv_matrix_operator(np.array([0.0, 1.0, 0.5]))
        with pytest.raises(ValueError, match="strictly increasing"):
            conv_matrix_operator(np.array([0.0, 1.0, 1.0, 2.0]))

    #
    def test_nonfinite_axis_raises(self):
        """A NaN in the axis raises a clear error."""

        with pytest.raises(ValueError, match="non-finite"):
            conv_matrix_operator(np.array([0.0, np.nan, 1.0]))


#
#
class TestConvMatrixApply:
    """Numerical behavior of the kernel-matrix convolution."""

    #
    def test_constant_preserved_on_nonuniform_axis(self):
        """Row normalization preserves a constant signal exactly."""

        t = np.concatenate([np.arange(-5, 2, 0.25), np.arange(2, 20, 1.0)])
        y = np.full(t.size, 3.3)
        result = _conv_matrix(t, gaussCONV, 1.2, y=y)
        np.testing.assert_allclose(result, 3.3, rtol=1e-12)

    #
    def test_uniform_axis_matches_my_conv(self):
        """Uniform axis reproduces the padded-1D-kernel path.

        Tolerance covers the old path's finite kernel support (±4*SD
        for gaussCONV truncates ~3e-4 of the kernel mass).
        """

        t = np.arange(-20, 100, 0.5)
        SD = 0.8
        y = expFun(t, 1.0, 5.0, 0.0)
        result = _conv_matrix(t, gaussCONV, SD, y=y)
        n = int(np.ceil(4 * SD / 0.5))
        support = np.arange(-n, n + 1) * 0.5
        old = my_conv(t, y, gaussCONV(support, SD))
        np.testing.assert_allclose(result, old, atol=1e-4)

    #
    def test_matches_dense_reference_on_nonuniform_axis(self):
        """Non-uniform axis matches the continuous convolution.

        Smooth signal (erfFun) and smooth kernel, so quadrature error is
        the only deviation — measured ~1e-3 relative on this axis. The
        sample-index convolution it replaced was far worse on discontinuous
        signals over non-uniform axes (up to ~7% of trace max on example
        21's stepped axis).
        """

        t = np.concatenate([np.arange(-15, 5, 0.25), np.arange(5, 60, 1.0)])
        SD = 1.5

        def signal(t_axis):
            return erfFun(t_axis, 5.0, 2.0, 0.0)

        result = _conv_matrix(t, gaussCONV, SD, y=signal(t))
        reference = _dense_reference(t, signal, gaussCONV, SD)
        scale = np.abs(reference).max()
        assert np.abs(result - reference).max() / scale < 5e-3

    #
    def test_asymmetric_kernel_direction(self):
        """Causal kernel (expDecayCONV) delays the signal, not advances it.

        Pins the sign convention K[i, j] = g(t_i - t_j): a causal kernel
        (nonzero for positive time differences) draws from the past.
        Compared against the old padded-1D path, which shares the
        sampled-kernel semantics at the dt=0 discontinuity.
        """

        tau = 2.0
        t = np.arange(-10, 40, 0.5)
        y = expFun(t, 1.0, 8.0, 0.0)
        result = _conv_matrix(t, expDecayCONV, tau, y=y)
        n = int(np.ceil(6 * tau / 0.5))
        support = np.arange(-n, n + 1) * 0.5
        old = my_conv(t, y, expDecayCONV(support, tau))
        # ±6*tau truncates exp(-6) ≈ 0.25% of the old kernel's mass
        np.testing.assert_allclose(result, old, atol=5e-3)
        # the convolved response lags the raw signal on the rising edge
        assert result[t == 1.0] < y[t == 1.0]
        # and is causal: nothing before the t0=0 onset
        np.testing.assert_allclose(result[t < 0.0], 0.0, atol=1e-12)

    #
    def test_edge_mass_accumulation(self):
        """Kernel mass beyond the window folds onto the edge samples.

        A saturated signal (constant near both ends) must stay flat at
        the edges — the edge-padding assumption — instead of decaying
        toward zero as pure row-truncation would produce.
        """

        t = np.arange(0, 20, 0.5)
        y = np.full(t.size, 2.0)
        y[20:] = 5.0  # step in the middle, constant at both ends
        result = _conv_matrix(t, gaussCONV, 1.0, y=y)
        np.testing.assert_allclose(result[0], 2.0, rtol=1e-6)
        np.testing.assert_allclose(result[-1], 5.0, rtol=1e-6)

    #
    def test_broad_kernel_fine_start_axis_edges_exact(self):
        """Broad kernel on a fine-start axis: edge masses stay exact.

        Regression for the ghost-point edge scheme this replaced: its
        coverage was only n_t * boundary_step per side (here ~1.3 time
        units against SD = 5), so most of the exterior kernel mass was
        silently discarded and renormalized away. The analytic edge
        masses have no coverage limit. Reference: identical interior
        trapezoid discretization + exterior masses from an independent
        implementation (scipy.stats.norm survival function).
        """

        t = np.concatenate([np.arange(0.0, 1.0, 0.01), np.arange(1.0, 31.0, 1.0)])
        SD = 5.0
        y = np.linspace(2.0, 4.0, t.size)
        result = _conv_matrix(t, gaussCONV, SD, y=y)

        operator = conv_matrix_operator(t)
        interior = gaussCONV(t[:, None] - t[None, :], SD) * operator.quad_weights
        scale = SD * np.sqrt(2 * np.pi)  # peak-1 body integral
        mass_left = scale * norm.sf((t - t[0]) / SD)
        mass_right = scale * norm.cdf((t - t[-1]) / SD)
        row_sums = interior.sum(axis=1) + mass_left + mass_right
        expected = (interior @ y + mass_left * y[0] + mass_right * y[-1]) / row_sums
        np.testing.assert_allclose(result, expected, rtol=1e-9)

    #
    def test_asymmetric_kernels_broad_edge_behavior(self):
        """One-sided kernels with window-scale tau: edges stay exact.

        Left/right swap detector: a causal kernel (expDecayCONV) draws
        only from the past, so the first sample sees nothing but
        edge-padded y[0] and must return it exactly; the anti-causal
        mirror (expRiseCONV) must return y[-1] at the last sample. With
        tau comparable to the window, most of each row's mass is
        exterior, so swapped left/right mass formulas fail loudly.
        """

        t = np.arange(0.0, 10.5, 0.5)
        tau = 5.0
        y = np.where(t < 5.0, 1.0, 3.0)

        result_causal = _conv_matrix(t, expDecayCONV, tau, y=y)
        np.testing.assert_allclose(result_causal[0], y[0], rtol=1e-12)
        assert result_causal[-1] < y[-1]  # lags behind the step

        result_anti = _conv_matrix(t, expRiseCONV, tau, y=y)
        np.testing.assert_allclose(result_anti[-1], y[-1], rtol=1e-12)
        assert result_anti[0] > y[0]  # anticipates the step

    #
    def test_narrow_kernel_degrades_to_identity(self):
        """A kernel far narrower than the step returns the input signal.

        The dt=0 diagonal keeps every row sum positive, so sub-step
        kernels degrade gracefully instead of raising (the old 1D path
        errored out when the sampled kernel summed to zero).
        """

        t = np.arange(0, 10, 1.0)
        y = np.sin(t)
        result = _conv_matrix(t, gaussCONV, 1e-6, y=y)
        np.testing.assert_allclose(result, y, rtol=1e-12)

    #
    def test_nan_kernel_params_raise(self):
        """Non-finite kernel parameters raise instead of propagating NaN.

        The edge-mass companion validates the parameters first; the
        apply-side finite/nonnegative check on the kernel values remains
        as a backstop for direct calls.
        """

        t = np.arange(0, 10, 1.0)
        y = np.sin(t)
        with pytest.raises(ValueError, match="strictly positive"):
            _conv_matrix(t, gaussCONV, np.nan, y=y)
        operator = conv_matrix_operator(t)
        bad_kernel = np.full(operator.dt_unique.shape, np.nan)
        good_mass = np.full(t.size, 0.1)
        with pytest.raises(ValueError, match="finite and nonnegative"):
            conv_matrix_apply(operator, bad_kernel, good_mass, good_mass, y)

    #
    def test_negative_edge_mass_rejected(self):
        """Signed mass errors cannot cancel inside row sums silently."""

        t = np.arange(0, 10, 1.0)
        y = np.sin(t)
        operator = conv_matrix_operator(t)
        kernel_values = gaussCONV(operator.dt_unique, 1.0)
        bad_mass = np.full(t.size, -0.1)
        good_mass = np.full(t.size, 0.1)
        with pytest.raises(ValueError, match="edge masses"):
            conv_matrix_apply(operator, kernel_values, bad_mass, good_mass, y)

    #
    def test_wrong_shape_inputs_rejected(self):
        """Mismatched kernel/mass shapes raise instead of broadcasting."""

        t = np.arange(0, 10, 1.0)
        y = np.sin(t)
        operator = conv_matrix_operator(t)
        kernel_values = gaussCONV(operator.dt_unique, 1.0)
        mass = np.full(t.size - 1, 0.1)  # wrong length
        with pytest.raises(ValueError, match="wrong shapes"):
            conv_matrix_apply(operator, kernel_values, mass, mass, y)

    #
    def test_wrong_shape_signal_rejected(self):
        """A column-vector y raises instead of broadcasting to (n_t, n_t)."""

        t = np.arange(0, 10, 1.0)
        y_col = np.sin(t)[:, None]
        operator = conv_matrix_operator(t)
        kernel_values = gaussCONV(operator.dt_unique, 1.0)
        mass = np.full(t.size, 0.1)
        with pytest.raises(ValueError, match="wrong shapes"):
            conv_matrix_apply(operator, kernel_values, mass, mass, y_col)


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


#
class TestResolveTimeSelection:
    """resolve_time_selection — extracted from File._resolve_time_selection
    so archive-side reconstruction can resolve a raw time selection with no
    live File (see FitResults._full_observed_for)."""

    #
    def test_abs_single_point(self):
        time = np.linspace(-2, 10, 24)
        assert resolve_time_selection(time, 1.5, 1.5, time_type="abs") == [
            int(np.searchsorted(time, 1.5)),
            int(np.searchsorted(time, 1.5)) + 1,
        ]

    #
    def test_abs_range(self):
        time = np.linspace(-2, 10, 24)
        ind = resolve_time_selection(time, 0.0, 2.0, time_type="abs")
        assert ind[0] < ind[1]
        assert time[ind[0]] >= 0.0
        assert time[ind[1] - 1] <= 2.0

    #
    def test_ind_single_point(self):
        time = np.linspace(-2, 10, 24)
        assert resolve_time_selection(time, 5, 5, time_type="ind") == [5, 6]

    #
    def test_ind_range(self):
        time = np.linspace(-2, 10, 24)
        assert resolve_time_selection(time, 5, 8, time_type="ind") == [5, 9]

    #
    def test_unknown_time_type_raises(self):
        time = np.linspace(-2, 10, 24)
        with pytest.raises(ValueError, match="Unknown time_type"):
            resolve_time_selection(time, 0, 1, time_type="bogus")

    #
    def test_out_of_range_raises(self):
        time = np.linspace(-2, 10, 24)
        with pytest.raises(ValueError, match="empty or out-of-range"):
            resolve_time_selection(time, 100, 100, time_type="ind")
