"""Round-trip tests for parameter profiles: simulate, fit, recover.

Clean tests (plumbing): noiseless data, machine-precision recovery.
Noisy tests: Gaussian noise, recovery within 5% relative error.

Test 1: Two profiles on same component (pLinear on x0 + pExpDecay on A),
        no dynamics — baseline-only fit.
Test 2: Two profiles + dynamics (expFun on pExpDecay A) — full 2D fit.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from trspecfit import File, Project, Simulator


#
def _make_project():
    """Create a silent project pointing to tests/ for YAML access."""

    project = Project(path="tests", name="roundtrip_profile")
    project.show_output = 0
    return project


#
def _make_energy_axis():
    return np.linspace(81, 89, 50)


#
def _make_time_axis():
    return np.linspace(-2, 10, 24)


#
def _make_aux_axis():
    return np.linspace(0, 8, 20)


#
def _extract_par_dict(model):
    """Return {name: value} for all non-expression parameters."""

    return {
        name: model.lmfit_pars[name].value
        for name in model.parameter_names
        if model.lmfit_pars[name].expr is None
    }


# ---- truth file builders ----


#
def _make_truth_two_profiles(project):
    """Gauss + pLinear on x0 + pExpDecay on A (two profiles, no dynamics)."""

    file = File(parent_project=project, name="truth", aux_axis=_make_aux_axis())
    file.energy = _make_energy_axis()
    file.time = _make_time_axis()
    file.dim = 2

    file.load_model(
        model_yaml="models/file_energy.yaml",
        model_info="single_gauss",
    )
    file.add_par_profile(
        target_model="single_gauss",
        target_parameter="Gauss_01_x0",
        profile_yaml="models/file_profile.yaml",
        profile_model=["roundtrip_pLinear_x0"],
    )
    file.add_par_profile(
        target_model="single_gauss",
        target_parameter="Gauss_01_A",
        profile_yaml="models/file_profile.yaml",
        profile_model=["roundtrip_pExpDecay_A"],
    )
    return file


#
def _make_truth_profile_dynamics(project):
    """Gauss + two profiles + expFun dynamics on pExpDecay A."""

    file = File(parent_project=project, name="truth", aux_axis=_make_aux_axis())
    file.energy = _make_energy_axis()
    file.time = _make_time_axis()
    file.dim = 2

    file.load_model(
        model_yaml="models/file_energy.yaml",
        model_info="single_gauss",
    )
    file.add_par_profile(
        target_model="single_gauss",
        target_parameter="Gauss_01_x0",
        profile_yaml="models/file_profile.yaml",
        profile_model=["roundtrip_pLinear_x0"],
    )
    file.add_par_profile(
        target_model="single_gauss",
        target_parameter="Gauss_01_A",
        profile_yaml="models/file_profile.yaml",
        profile_model=["roundtrip_pExpDecay_A"],
    )
    file.add_time_dependence(
        target_model="single_gauss",
        target_parameter="Gauss_01_A_pExpDecay_01_A",
        dynamics_yaml="models/file_time.yaml",
        dynamics_model=["MonoExpPosStrong"],
    )
    return file


# ---- fit file builders ----


#
def _make_fit_two_profiles(project, data, energy, time, aux):
    """Fresh file with two profiles, loaded with simulated data."""

    file = File(
        parent_project=project,
        name="fit",
        data=data,
        energy=energy.copy(),
        time=time.copy(),
        aux_axis=aux.copy(),
    )
    file.load_model(
        model_yaml="models/file_energy.yaml",
        model_info="single_gauss",
    )
    file.add_par_profile(
        target_model="single_gauss",
        target_parameter="Gauss_01_x0",
        profile_yaml="models/file_profile.yaml",
        profile_model=["roundtrip_pLinear_x0"],
    )
    file.add_par_profile(
        target_model="single_gauss",
        target_parameter="Gauss_01_A",
        profile_yaml="models/file_profile.yaml",
        profile_model=["roundtrip_pExpDecay_A"],
    )
    file.define_baseline(time_start=0, time_stop=3, time_type="ind", show_plot=False)
    return file


#
def _make_fit_profile_dynamics(project, data, energy, time, aux):
    """Fresh file with two profiles, ready for baseline fit."""

    file = File(
        parent_project=project,
        name="fit",
        data=data,
        energy=energy.copy(),
        time=time.copy(),
        aux_axis=aux.copy(),
    )
    file.load_model(
        model_yaml="models/file_energy.yaml",
        model_info="single_gauss",
    )
    file.add_par_profile(
        target_model="single_gauss",
        target_parameter="Gauss_01_x0",
        profile_yaml="models/file_profile.yaml",
        profile_model=["roundtrip_pLinear_x0"],
    )
    file.add_par_profile(
        target_model="single_gauss",
        target_parameter="Gauss_01_A",
        profile_yaml="models/file_profile.yaml",
        profile_model=["roundtrip_pExpDecay_A"],
    )
    file.define_baseline(time_start=0, time_stop=3, time_type="ind", show_plot=False)
    return file


#
def _add_fit_profile_dynamics(file):
    """Add dynamics after baseline fitting the profile-only model."""

    file.add_time_dependence(
        target_model="single_gauss",
        target_parameter="Gauss_01_A_pExpDecay_01_A",
        dynamics_yaml="models/file_time.yaml",
        dynamics_model=["MonoExpPosStrong"],
    )


#
#
class TestRoundTripTwoProfiles:
    """Two profiles on same component, no dynamics — baseline fit recovery."""

    #
    @pytest.mark.slow
    def test_clean_recovery(self):
        """Simulate clean 2D data with two profiles, fit baseline, recover."""

        project = _make_project()
        truth_file = _make_truth_two_profiles(project)
        truth_pars = _extract_par_dict(truth_file.model_active)

        sim = Simulator(
            model=truth_file.model_active,
            detection="analog",
            noise_level=0.0,
            noise_type="none",
            seed=42,
        )
        clean, _, _ = sim.simulate_2d()

        fit_file = _make_fit_two_profiles(
            project, clean, truth_file.energy, truth_file.time, truth_file.aux_axis
        )
        fit_file.fit_baseline(
            model_name="single_gauss",
            stages=2,
            try_ci=0,
        )

        fitted_pars = fit_file.model_base.result[1].params
        for name, true_val in truth_pars.items():
            fit_val = fitted_pars[name].value
            assert np.isclose(true_val, fit_val, rtol=1e-10, atol=1e-12), (
                f"{name}: true={true_val:.6f}, fit={fit_val:.6f}"
            )


#
#
class TestRoundTripProfileDynamics:
    """Two profiles + dynamics on pExpDecay A — full 2D fit recovery."""

    #
    @pytest.mark.slow
    def test_clean_recovery(self):
        """Simulate clean 2D data with profile + dynamics, fit, recover."""

        project = _make_project()
        truth_file = _make_truth_profile_dynamics(project)
        truth_pars = _extract_par_dict(truth_file.model_active)

        sim = Simulator(
            model=truth_file.model_active,
            detection="analog",
            noise_level=0.0,
            noise_type="none",
            seed=42,
        )
        clean, _, _ = sim.simulate_2d()

        fit_file = _make_fit_profile_dynamics(
            project, clean, truth_file.energy, truth_file.time, truth_file.aux_axis
        )
        fit_file.fit_baseline(
            model_name="single_gauss",
            stages=2,
            try_ci=0,
        )
        _add_fit_profile_dynamics(fit_file)
        fit_file.fit_2d(
            model_name="single_gauss",
            stages=2,
            try_ci=0,
        )

        fitted_pars = fit_file.model_2d.result[1].params
        for name, true_val in truth_pars.items():
            fit_val = fitted_pars[name].value
            assert np.isclose(true_val, fit_val, rtol=1e-10, atol=1e-12), (
                f"{name}: true={true_val:.6f}, fit={fit_val:.6f}"
            )


#
#
class TestRoundTripTwoProfilesNoisy:
    """Two profiles, no dynamics — noisy baseline fit recovery within 5%."""

    #
    @pytest.mark.slow
    def test_noisy_recovery(self):
        """Simulate noisy 2D data with two profiles, fit baseline, recover."""

        project = _make_project()
        truth_file = _make_truth_two_profiles(project)
        truth_pars = _extract_par_dict(truth_file.model_active)

        sim = Simulator(
            model=truth_file.model_active,
            detection="analog",
            noise_level=0.01,
            noise_type="gaussian",
            seed=42,
        )
        _, noisy, _ = sim.simulate_2d()

        fit_file = _make_fit_two_profiles(
            project, noisy, truth_file.energy, truth_file.time, truth_file.aux_axis
        )
        fit_file.fit_baseline(
            model_name="single_gauss",
            stages=2,
            try_ci=0,
        )

        fitted_pars = fit_file.model_base.result[1].params
        for name, true_val in truth_pars.items():
            if abs(true_val) < 1e-6:
                continue  # skip fixed-at-zero params
            fit_val = fitted_pars[name].value
            rel_err = abs(fit_val - true_val) / abs(true_val)
            assert rel_err < 0.05, (
                f"{name}: true={true_val:.4f}, fit={fit_val:.4f}, rel_err={rel_err:.1%}"
            )


#
#
class TestRoundTripProfileDynamicsNoisy:
    """Two profiles + dynamics on pExpDecay A — noisy 2D fit recovery."""

    #
    @pytest.mark.slow
    def test_noisy_recovery(self):
        """Simulate noisy 2D data with profile + dynamics, fit, recover."""

        project = _make_project()
        truth_file = _make_truth_profile_dynamics(project)
        truth_pars = _extract_par_dict(truth_file.model_active)

        sim = Simulator(
            model=truth_file.model_active,
            detection="analog",
            noise_level=0.01,
            noise_type="gaussian",
            seed=42,
        )
        _, noisy, _ = sim.simulate_2d()

        fit_file = _make_fit_profile_dynamics(
            project, noisy, truth_file.energy, truth_file.time, truth_file.aux_axis
        )
        fit_file.fit_baseline(
            model_name="single_gauss",
            stages=2,
            try_ci=0,
        )
        _add_fit_profile_dynamics(fit_file)
        fit_file.fit_2d(
            model_name="single_gauss",
            stages=1,
            try_ci=0,
        )

        fitted_pars = fit_file.model_2d.result[1].params
        for name, true_val in truth_pars.items():
            if abs(true_val) < 1e-6:
                continue  # skip fixed-at-zero params
            fit_val = fitted_pars[name].value
            rel_err = abs(fit_val - true_val) / abs(true_val)
            assert rel_err < 0.05, (
                f"{name}: true={true_val:.4f}, fit={fit_val:.4f}, rel_err={rel_err:.1%}"
            )
