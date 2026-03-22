"""Round-trip tests: simulate data with known parameters, fit, recover.

Uses small grids (30 energy × 30 time points) to keep tests fast.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np

from trspecfit import File, Project, Simulator


#
def _make_project():
    """Create a silent project pointing to tests/ for YAML access."""

    project = Project(path="tests", name="roundtrip")
    project.show_info = 0
    return project


#
def _make_truth_file(project):
    """Create file with single GLP peak + exponential dynamics on amplitude."""

    energy = np.linspace(83, 87, 30)
    time = np.linspace(-2, 10, 24)

    file = File(parent_project=project)
    file.energy = energy
    file.time = time
    file.dim = 2

    file.load_model(
        model_yaml="test_models_energy.yaml",
        model_info=["single_glp"],
    )
    file.add_time_dependence(
        model_yaml="test_models_time.yaml",
        model_info=["MonoExpPos"],
        par_name="GLP_01_A",
    )
    return file


#
def _make_fit_file(project, data, energy, time):
    """Create a fresh file loaded with simulated data, ready to fit."""

    file = File(
        parent_project=project,
        data=data,
        energy=energy.copy(),
        time=time.copy(),
    )

    file.load_model(
        model_yaml="test_models_energy.yaml",
        model_info=["single_glp"],
    )
    file.add_time_dependence(
        model_yaml="test_models_time.yaml",
        model_info=["MonoExpPos"],
        par_name="GLP_01_A",
    )

    file.define_baseline(time_start=0, time_stop=3, time_type="ind", show_plot=False)
    return file


#
def _extract_par_dict(model):
    """Return {name: value} for all non-expression parameters."""

    return {
        name: model.lmfit_pars[name].value
        for name in model.par_names
        if model.lmfit_pars[name].expr is None
    }


#
#
class TestRoundTripClean:
    """Fit noiseless data — parameters should be recovered exactly."""

    #
    def test_clean_recovery(self):
        """Simulate clean 2D data, fit, assert parameter recovery."""

        project = _make_project()
        truth_file = _make_truth_file(project)
        truth_pars = _extract_par_dict(truth_file.model_active)

        sim = Simulator(
            model=truth_file.model_active,
            detection="analog",
            noise_level=0.0,
            noise_type="none",
            seed=42,
        )
        clean, _, _ = sim.simulate_2D()

        fit_file = _make_fit_file(project, clean, truth_file.energy, truth_file.time)

        fit_file.fit_baseline(
            model_name="single_glp",
            fit=2,
            try_CI=0,
        )
        fit_file.fit_2Dmodel(
            model_name="single_glp",
            fit=2,
            try_CI=0,
        )

        fitted_pars = fit_file.model_2D.result[1].params
        for name, true_val in truth_pars.items():
            fit_val = fitted_pars[name].value
            assert np.isclose(true_val, fit_val, rtol=1e-2, atol=1e-6), (
                f"{name}: true={true_val:.6f}, fit={fit_val:.6f}"
            )


#
#
class TestRoundTripNoisy:
    """Fit noisy data — parameters should be recovered within tolerance."""

    #
    def test_noisy_recovery(self):
        """Simulate noisy 2D data, fit, assert recovery within 5%."""

        project = _make_project()
        truth_file = _make_truth_file(project)
        truth_pars = _extract_par_dict(truth_file.model_active)

        sim = Simulator(
            model=truth_file.model_active,
            detection="analog",
            noise_level=0.01,
            noise_type="gaussian",
            seed=42,
        )
        _, noisy, _ = sim.simulate_2D()

        fit_file = _make_fit_file(project, noisy, truth_file.energy, truth_file.time)

        fit_file.fit_baseline(
            model_name="single_glp",
            fit=2,
            try_CI=0,
        )
        fit_file.fit_2Dmodel(
            model_name="single_glp",
            fit=1,
            try_CI=0,
        )

        fitted_pars = fit_file.model_2D.result[1].params
        for name, true_val in truth_pars.items():
            if abs(true_val) < 1e-6:
                continue  # skip fixed-at-zero params (t0, y0)
            fit_val = fitted_pars[name].value
            rel_err = abs(fit_val - true_val) / abs(true_val)
            assert rel_err < 0.05, (
                f"{name}: true={true_val:.4f}, fit={fit_val:.4f}, rel_err={rel_err:.1%}"
            )
