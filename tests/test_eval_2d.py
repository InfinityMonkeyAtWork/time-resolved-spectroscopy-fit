"""Tests for 2D component dispatch functions.

All component functions live in ``functions.energy`` as the single source
of truth.  Tests here verify 2D shape correctness and row-wise parameter
variation for all dispatched functions (peak functions broadcast
naturally, backgrounds have unified 1D/2D signatures).
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
)


def _energy_2d(start=280.0, stop=290.0, n=50):
    """Energy axis shaped (1, n_energy)."""

    return np.linspace(start, stop, n)[np.newaxis, :]


def _col(values):
    """Make (n_time, 1) column from a list."""

    return np.array(values, dtype=np.float64)[:, np.newaxis]


#
#
class TestGauss2D:
    def test_1d_match(self):
        energy = _energy_2d()
        A, x0, SD = 5.0, 285.0, 1.2
        result = Gauss(energy, _col([A]), _col([x0]), _col([SD]))
        expected = Gauss(energy[0], A, x0, SD)
        np.testing.assert_allclose(result[0], expected)

    def test_row_wise_variation(self):
        """x0 varies across time -- each row should match its own 1D call."""

        energy = _energy_2d()
        n_time = 4
        A_vals = [3.0, 5.0, 7.0, 2.0]
        x0_vals = [283.0, 284.5, 286.0, 288.0]
        SD_vals = [1.0, 1.2, 0.8, 1.5]

        result = Gauss(energy, _col(A_vals), _col(x0_vals), _col(SD_vals))
        assert result.shape == (n_time, energy.shape[1])

        for t in range(n_time):
            expected = Gauss(energy[0], A_vals[t], x0_vals[t], SD_vals[t])
            np.testing.assert_allclose(result[t], expected, rtol=1e-12)


#
#
class TestGaussAsym2D:
    def test_1d_match(self):
        energy = _energy_2d()
        A, x0, SD, ratio = 4.0, 285.0, 1.0, 1.5
        result = GaussAsym(energy, _col([A]), _col([x0]), _col([SD]), _col([ratio]))
        expected = GaussAsym(energy[0], A, x0, SD, ratio)
        np.testing.assert_allclose(result[0], expected)

    def test_row_wise_variation(self):
        """ratio and x0 vary across time."""

        energy = _energy_2d()
        n_time = 3
        A_vals = [4.0, 6.0, 3.0]
        x0_vals = [283.0, 285.0, 287.0]
        SD_vals = [1.0, 1.2, 0.9]
        ratio_vals = [0.8, 1.0, 1.5]

        result = GaussAsym(
            energy,
            _col(A_vals),
            _col(x0_vals),
            _col(SD_vals),
            _col(ratio_vals),
        )
        assert result.shape == (n_time, energy.shape[1])

        for t in range(n_time):
            expected = GaussAsym(
                energy[0], A_vals[t], x0_vals[t], SD_vals[t], ratio_vals[t]
            )
            np.testing.assert_allclose(result[t], expected, rtol=1e-12)


#
#
class TestLorentz2D:
    def test_1d_match(self):
        energy = _energy_2d()
        A, x0, W = 3.0, 285.0, 2.0
        result = Lorentz(energy, _col([A]), _col([x0]), _col([W]))
        expected = Lorentz(energy[0], A, x0, W)
        np.testing.assert_allclose(result[0], expected)

    def test_row_wise_variation(self):
        energy = _energy_2d()
        n_time = 3
        x0_vals = [283.0, 285.0, 287.0]

        result = Lorentz(energy, _col([3.0] * 3), _col(x0_vals), _col([2.0] * 3))
        for t in range(n_time):
            expected = Lorentz(energy[0], 3.0, x0_vals[t], 2.0)
            np.testing.assert_allclose(result[t], expected, rtol=1e-12)


#
#
class TestGLS2D:
    def test_1d_match(self):
        energy = _energy_2d()
        A, x0, F, m = 5.0, 285.0, 1.5, 0.3
        result = GLS(energy, _col([A]), _col([x0]), _col([F]), _col([m]))
        expected = GLS(energy[0], A, x0, F, m)
        np.testing.assert_allclose(result[0], expected, rtol=1e-12)

    def test_row_wise_variation(self):
        energy = _energy_2d()
        n_time = 3
        m_vals = [0.0, 0.5, 1.0]  # pure Gauss, mixed, pure Lorentz

        result = GLS(
            energy,
            _col([5.0] * 3),
            _col([285.0] * 3),
            _col([1.5] * 3),
            _col(m_vals),
        )
        for t in range(n_time):
            expected = GLS(energy[0], 5.0, 285.0, 1.5, m_vals[t])
            np.testing.assert_allclose(result[t], expected, rtol=1e-12)


#
#
class TestGLP2D:
    def test_1d_match(self):
        energy = _energy_2d()
        A, x0, F, m = 5.0, 285.0, 1.5, 0.3
        result = GLP(energy, _col([A]), _col([x0]), _col([F]), _col([m]))
        expected = GLP(energy[0], A, x0, F, m)
        np.testing.assert_allclose(result[0], expected, rtol=1e-12)

    def test_row_wise_variation(self):
        energy = _energy_2d()
        n_time = 4
        x0_vals = [283.0, 284.5, 286.0, 288.0]
        F_vals = [1.0, 1.5, 2.0, 0.8]

        result = GLP(
            energy,
            _col([5.0] * 4),
            _col(x0_vals),
            _col(F_vals),
            _col([0.3] * 4),
        )
        for t in range(n_time):
            expected = GLP(energy[0], 5.0, x0_vals[t], F_vals[t], 0.3)
            np.testing.assert_allclose(result[t], expected, rtol=1e-12)


#
#
class TestDS2D:
    def test_1d_match(self):
        energy = _energy_2d()
        A, x0, F, alpha = 5.0, 285.0, 1.5, 0.15
        result = DS(energy, _col([A]), _col([x0]), _col([F]), _col([alpha]))
        expected = DS(energy[0], A, x0, F, alpha)
        np.testing.assert_allclose(result[0], expected, rtol=1e-12)

    def test_row_wise_variation(self):
        energy = _energy_2d()
        n_time = 3
        alpha_vals = [0.0, 0.15, 0.3]

        result = DS(
            energy,
            _col([5.0] * 3),
            _col([285.0] * 3),
            _col([1.5] * 3),
            _col(alpha_vals),
        )
        for t in range(n_time):
            expected = DS(energy[0], 5.0, 285.0, 1.5, alpha_vals[t])
            np.testing.assert_allclose(result[t], expected, rtol=1e-12)


#
#
class TestOffset2D:
    def test_1d_match(self):
        energy = _energy_2d()
        result = Offset(energy, _col([2.5]))
        expected = Offset(energy[0], 2.5)
        np.testing.assert_allclose(result[0], expected)

    def test_row_wise_variation(self):
        energy = _energy_2d()
        y0_vals = [1.0, 3.0, 0.5]
        result = Offset(energy, _col(y0_vals))
        assert result.shape == (3, energy.shape[1])

        for t in range(3):
            np.testing.assert_allclose(result[t], y0_vals[t])

    def test_result_is_writable(self):
        """Offset result must be a writable copy, not a read-only view."""

        energy = _energy_2d()
        result = Offset(energy, _col([1.0]))
        result[0, 0] = 999.0  # should not raise


#
#
class TestLinBack2D:
    def test_1d_match(self):
        energy = _energy_2d()
        m, b, xStart, xStop = 0.5, 1.0, 282.0, 288.0
        result = LinBack(energy, _col([m]), _col([b]), _col([xStart]), _col([xStop]))
        expected = LinBack(energy[0], m, b, xStart, xStop)
        np.testing.assert_allclose(result[0], expected)

    def test_row_wise_xstart_xstop_variation(self):
        """xStart/xStop vary across time -- clamp boundaries shift per row."""

        energy = _energy_2d()
        n_time = 3
        m_vals = [0.5, 0.5, 0.5]
        b_vals = [1.0, 1.0, 1.0]
        xStart_vals = [281.0, 283.0, 285.0]
        xStop_vals = [286.0, 288.0, 289.0]

        result = LinBack(
            energy,
            _col(m_vals),
            _col(b_vals),
            _col(xStart_vals),
            _col(xStop_vals),
        )
        for t in range(n_time):
            expected = LinBack(
                energy[0],
                m_vals[t],
                b_vals[t],
                xStart_vals[t],
                xStop_vals[t],
            )
            np.testing.assert_allclose(result[t], expected, rtol=1e-12)

    def test_row_wise_slope_variation(self):
        """m varies across time."""

        energy = _energy_2d()
        n_time = 3
        m_vals = [-0.5, 0.0, 1.0]

        result = LinBack(
            energy,
            _col(m_vals),
            _col([1.0] * 3),
            _col([282.0] * 3),
            _col([288.0] * 3),
        )
        for t in range(n_time):
            expected = LinBack(energy[0], m_vals[t], 1.0, 282.0, 288.0)
            np.testing.assert_allclose(result[t], expected, rtol=1e-12)


#
#
class TestShirley2D:
    def test_1d_match(self):
        energy = _energy_2d()
        # Build a realistic peak sum (single GLP peak)
        spectrum_1d = GLP(energy[0], 5.0, 285.0, 1.5, 0.3)
        spectrum_2d = spectrum_1d[np.newaxis, :]  # (1, n_energy)
        pShirley = 1e-3

        result = Shirley(energy, _col([pShirley]), spectrum_2d)
        expected = Shirley(energy[0], pShirley, spectrum_1d)
        np.testing.assert_allclose(result[0], expected, rtol=1e-12)

    def test_row_wise_different_spectra(self):
        """Each row has a different peak spectrum -- catches axis=0 cumsum bug."""

        energy = _energy_2d()
        n_time = 3
        # Different peaks at different positions per time step
        x0_vals = [283.0, 285.0, 287.0]
        spectra = np.array([GLP(energy[0], 5.0, x0, 1.5, 0.3) for x0 in x0_vals])
        assert spectra.shape == (n_time, energy.shape[1])

        pShirley_vals = [1e-3, 2e-3, 5e-4]
        result = Shirley(energy, _col(pShirley_vals), spectra)

        for t in range(n_time):
            expected = Shirley(energy[0], pShirley_vals[t], spectra[t])
            np.testing.assert_allclose(result[t], expected, rtol=1e-12)

    def test_cumsum_direction(self):
        """Shirley cumsum must be along energy (axis=-1), not time (axis=0)."""

        energy = _energy_2d(n=10)
        # Row 0: peak at left; Row 1: peak at right
        spectrum = np.zeros((2, 10))
        spectrum[0, 1] = 1.0  # peak near left edge
        spectrum[1, 8] = 1.0  # peak near right edge

        result = Shirley(energy, _col([1.0, 1.0]), spectrum)

        # Row 0: cumsum from right -> value at index 0 should be 1.0
        assert result[0, 0] == pytest.approx(1.0)
        assert result[0, 9] == pytest.approx(0.0)

        # Row 1: cumsum from right -> value at index 0 should be 1.0
        assert result[1, 0] == pytest.approx(1.0)
        assert result[1, 9] == pytest.approx(0.0)

        # Rows should differ in where the step occurs
        assert not np.allclose(result[0], result[1])
