"""Fit-entry validation: non-finite data and degenerate axes.

Covers the public File fit entry points with NaN/Inf-contaminated data
(inside vs outside the fit window) and single-element energy/time axes.
The non-finite check lives at the fit_wrapper choke point, so one
entry point per dimensionality is exercised rather than every method.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from _utils import make_project, simulate_clean

from trspecfit import File


#
def _simulate_truth(project):
    """Simulate clean 2D data (single GLP + MonoExpPos on its amplitude)."""

    energy = np.linspace(83, 87, 30)
    time = np.linspace(-2, 10, 24)

    file = File(parent_project=project, name="truth")
    file.energy = energy
    file.time = time
    file.dim = 2
    file.load_model(
        model_yaml="models/file_energy.yaml",
        model_info="single_glp",
    )
    file.add_time_dependence(
        target_model="single_glp",
        target_parameter="GLP_01_A",
        dynamics_yaml="models/file_time.yaml",
        dynamics_model=["MonoExpPos"],
    )
    assert file.model_active is not None  # type guard
    return simulate_clean(file.model_active), energy, time


#
def _make_fit_file(project, data, energy, time):
    """Create a file loaded with data and the 1D single_glp model."""

    file = File(
        parent_project=project,
        name="fit",
        data=data,
        energy=energy.copy(),
        time=time.copy(),
    )
    file.load_model(
        model_yaml="models/file_energy.yaml",
        model_info="single_glp",
    )
    return file


#
#
class TestNonFiniteData:
    """NaN/Inf in data fail fast with a clear error at the fit entry."""

    #
    def test_nan_in_fit_window_raises(self):
        """fit_spectrum on data with a NaN in the fit window raises.

        Regression guard on the message: lmfit's own error blames "input
        data or the output of your objective/model function", leaving the
        user to figure out which one is broken.
        """

        project = make_project(name="nan_window")
        data, energy, time = _simulate_truth(project)
        data[5, 10] = np.nan

        fit_file = _make_fit_file(project, data, energy, time)
        with pytest.raises(ValueError, match="non-finite"):
            fit_file.fit_spectrum(
                "single_glp", time_point=float(time[5]), show_plot=False, try_ci=0
            )

    #
    def test_inf_in_fit_window_raises(self):
        """fit_spectrum treats Inf the same as NaN."""

        project = make_project(name="inf_window")
        data, energy, time = _simulate_truth(project)
        data[5, 10] = np.inf

        fit_file = _make_fit_file(project, data, energy, time)
        with pytest.raises(ValueError, match="non-finite"):
            fit_file.fit_spectrum(
                "single_glp", time_point=float(time[5]), show_plot=False, try_ci=0
            )

    #
    def test_nan_outside_fit_window_is_allowed(self):
        """A NaN excluded by set_fit_limits never reaches the residual.

        Cutting a contaminated detector region out of the fit window is a
        supported workflow; validation must only inspect the window.
        """

        project = make_project(name="nan_outside")
        data, energy, time = _simulate_truth(project)
        data[5, 0] = np.nan  # energy[0] = 83.0, below the limit chosen next

        fit_file = _make_fit_file(project, data, energy, time)
        fit_file.set_fit_limits([83.5, 87.0], show_plot=False)
        fit_file.fit_spectrum(
            "single_glp", time_point=float(time[5]), show_plot=False, try_ci=0
        )

        result = fit_file.model_spec.result
        assert result is not None  # type guard
        assert result.par_fin.success
        assert np.isfinite(result.par_fin.chisqr)

    #
    def test_nan_in_fit_window_raises_2d(self):
        """fit_2d validates the (t_lim, e_lim) window of the 2D data."""

        project = make_project(name="nan_2d")
        data, energy, time = _simulate_truth(project)
        data[15, 10] = np.nan

        fit_file = _make_fit_file(project, data, energy, time)
        fit_file.define_baseline(-2, -1, show_plot=False)
        fit_file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
        fit_file.add_time_dependence(
            target_model="single_glp",
            target_parameter="GLP_01_A",
            dynamics_yaml="models/file_time.yaml",
            dynamics_model=["MonoExpPos"],
        )
        with pytest.raises(ValueError, match="non-finite"):
            fit_file.fit_2d("single_glp", stages=1, try_ci=0)


#
#
class TestDegenerateAxes:
    """Single-element energy/time axes through the public fit pipeline."""

    #
    def test_single_element_energy_axis_fits(self):
        """fit_spectrum on a 1-point energy axis completes.

        Documents current behavior: the pipeline tolerates the degenerate
        axis and Nelder-Mead returns a result. With more free parameters
        than data points the fit is underdetermined (negative nfree), and
        judging that is left to the user.
        """

        project = make_project(name="one_energy")
        data, energy, time = _simulate_truth(project)

        fit_file = _make_fit_file(project, data[:, :1], energy[:1], time)
        fit_file.fit_spectrum(
            "single_glp", time_point=float(time[5]), show_plot=False, try_ci=0
        )

        result = fit_file.model_spec.result
        assert result is not None  # type guard
        assert result.par_fin.nfree < 0  # underdetermined, documented not endorsed

    #
    def test_single_element_time_axis_fit_2d(self):
        """fit_2d on a 1-point time axis completes for conv-free dynamics.

        Convolution models raise at model construction (kernel step size
        needs 2 points); plain dynamics stay evaluable at a single time.
        """

        project = make_project(name="one_time")
        data, energy, time = _simulate_truth(project)

        fit_file = _make_fit_file(project, data[:1, :], energy, time[:1])
        fit_file.define_baseline(0, 0, time_type="ind", show_plot=False)
        fit_file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
        fit_file.add_time_dependence(
            target_model="single_glp",
            target_parameter="GLP_01_A",
            dynamics_yaml="models/file_time.yaml",
            dynamics_model=["MonoExpPos"],
        )
        fit_file.fit_2d("single_glp", stages=1, try_ci=0)

        result = fit_file.model_2d.result
        assert result is not None  # type guard
        assert result.par_fin.success
