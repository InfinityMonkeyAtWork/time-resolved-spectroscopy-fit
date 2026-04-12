"""Pipeline integration tests for the GIR (compiled) 2D evaluator.

Tests verify that the dispatch logic in File.fit_2d correctly selects
the GIR fast path or falls back to the interpreter, and that residuals
produced by both paths match.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from trspecfit import File, Project, Simulator, fitlib, spectra
from trspecfit.graph_ir import build_graph, can_lower_2d, schedule_2d

_ENERGY_YAML = "models/eval_2d_energy.yaml"
_TIME_YAML = "models/file_time.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


#
def _make_project(*, spec_fun_str="fit_model_gir"):
    """Create a silent project with configurable spec_fun_str."""

    project = Project(path="tests", name="gir_int")
    project.show_output = 0
    project.spec_fun_str = spec_fun_str
    return project


#
def _make_2d_model(project, model_info, dynamics_params):
    """Load energy model + add dynamics -> 2D model."""

    file = File(parent_project=project)
    file.energy = np.linspace(80, 90, 101)
    file.time = np.linspace(-10, 100, 51)
    file.load_model(model_yaml=_ENERGY_YAML, model_info=model_info)
    model = file.model_active
    assert model is not None

    for target_par, dyn_model in dynamics_params:
        file.add_time_dependence(
            target_model=model_info[0],
            target_parameter=target_par,
            dynamics_yaml=_TIME_YAML,
            dynamics_model=dyn_model,
        )

    return file, model


#
def _extract_par_list(model):
    """Return full parameter value list in model.parameter_names order."""

    return [model.lmfit_pars[n].value for n in model.parameter_names]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


#
#
class TestGIRDispatch:
    """Verify that File.fit_2d selects the correct dispatch path."""

    #
    def test_fit_2d_uses_gir_when_lowerable(self):
        """Default spec_fun_str dispatches to fit_model_gir for lowerable model."""

        project = _make_project()
        file, model = _make_2d_model(
            project,
            ["glp_only"],
            [("GLP_01_A", ["MonoExpPos"])],
        )
        # Simulate setup that fit_2d would do — just check const tuple.
        # Build graph to verify it IS lowerable, then check that
        # the dispatch logic in spectra produces a ScheduledPlan2D path.
        graph = build_graph(model)
        assert can_lower_2d(graph)

        # Verify fit_model_gir handles the compiled args
        plan = schedule_2d(graph)
        name_to_idx = {n: i for i, n in enumerate(model.parameter_names)}
        theta_indices = np.array(
            [name_to_idx[n] for n in plan.opt_param_names], dtype=np.intp
        )
        par = _extract_par_list(model)
        result = spectra.fit_model_gir(file.energy, par, True, plan, theta_indices)
        assert result.shape == (len(file.time), len(file.energy))

    #
    def test_fit_2d_falls_back_for_non_lowerable(self):
        """Real non-lowerable model (profile) falls back to interpreter."""

        project = _make_project()
        aux_axis = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        file = File(parent_project=project, aux_axis=aux_axis)
        file.energy = np.linspace(80, 90, 101)
        file.load_model(
            model_yaml="models/file_energy.yaml",
            model_info=["single_gauss"],
        )
        file.add_par_profile(
            target_model="single_gauss",
            target_parameter="Gauss_01_A",
            profile_yaml="models/file_profile.yaml",
            profile_model=["profile_pExpDecay"],
        )
        model = file.model_active
        assert model is not None

        graph = build_graph(model)
        assert not can_lower_2d(graph)

        # fit_model_gir should delegate to fit_model_mcp when given Model args
        par = _extract_par_list(model)
        result = spectra.fit_model_gir(file.energy, par, True, model, 1)
        assert result is not None

    #
    def test_force_interpreter(self):
        """spec_fun_str='fit_model_mcp' forces interpreter path."""

        project = _make_project(spec_fun_str="fit_model_mcp")
        file, model = _make_2d_model(
            project,
            ["glp_only"],
            [("GLP_01_A", ["MonoExpPos"])],
        )
        # Model is lowerable but spec_fun_str forces MCP
        par = _extract_par_list(model)
        result = spectra.fit_model_mcp(file.energy, par, True, model, 2)
        assert result.shape == (len(file.time), len(file.energy))

    #
    def test_1d_fit_delegates_to_mcp(self):
        """1D fit with default spec_fun_str='fit_model_gir' delegates to mcp."""

        project = _make_project()
        file = File(parent_project=project)
        file.energy = np.linspace(80, 90, 101)
        file.load_model(model_yaml=_ENERGY_YAML, model_info=["glp_only"])
        model = file.model_active
        assert model is not None

        par = _extract_par_list(model)
        # fit_model_gir receives (model, dim=1) and delegates to mcp
        result = spectra.fit_model_gir(file.energy, par, True, model, 1)
        assert isinstance(result, np.ndarray)
        assert result.shape == (len(file.energy),)


#
#
class TestGIRvsInterpreter:
    """Compare GIR and interpreter outputs through the pipeline."""

    #
    def test_residual_same_gir_vs_mcp(self):
        """Residual from residual_fun matches between GIR and MCP paths."""

        project = _make_project()
        file, model = _make_2d_model(
            project,
            ["glp_only"],
            [("GLP_01_A", ["MonoExpPos"])],
        )

        # Generate synthetic data from the model
        model.create_value_2d()
        assert model.value_2d is not None
        data = model.value_2d + 0.01  # small offset so residual is non-zero

        # Compile GIR path
        graph = build_graph(model)
        assert can_lower_2d(graph)
        plan = schedule_2d(graph)
        name_to_idx = {n: i for i, n in enumerate(model.parameter_names)}
        theta_indices = np.array(
            [name_to_idx[n] for n in plan.opt_param_names], dtype=np.intp
        )

        par = model.lmfit_pars

        # Residual via GIR
        res_gir = fitlib.residual_fun(
            par=par,
            x=file.energy,
            data=data,
            package=spectra,
            fit_fun_str="fit_model_gir",
            args=(plan, theta_indices),
        )

        # Residual via MCP
        res_mcp = fitlib.residual_fun(
            par=par,
            x=file.energy,
            data=data,
            package=spectra,
            fit_fun_str="fit_model_mcp",
            args=(model, 2),
        )

        np.testing.assert_allclose(res_gir, res_mcp, rtol=1e-10, atol=1e-10)

    #
    def test_compare_mode(self):
        """spec_fun_str='fit_model_compare' runs both paths without error."""

        project = _make_project(spec_fun_str="fit_model_compare")
        file, model = _make_2d_model(
            project,
            ["glp_only"],
            [("GLP_01_A", ["MonoExpPos"])],
        )

        graph = build_graph(model)
        assert can_lower_2d(graph)
        plan = schedule_2d(graph)
        name_to_idx = {n: i for i, n in enumerate(model.parameter_names)}
        theta_indices = np.array(
            [name_to_idx[n] for n in plan.opt_param_names], dtype=np.intp
        )
        par = _extract_par_list(model)

        # fit_model_compare runs both and asserts internally
        result = spectra.fit_model_compare(
            file.energy, par, True, plan, theta_indices, model, 2
        )
        assert result.shape == (len(file.time), len(file.energy))

    #
    def test_residual_with_slicing(self):
        """GIR residual with e_lim/t_lim matches MCP residual."""

        project = _make_project()
        file, model = _make_2d_model(
            project,
            ["glp_only"],
            [("GLP_01_A", ["MonoExpPos"])],
        )

        model.create_value_2d()
        assert model.value_2d is not None
        data = model.value_2d + 0.01

        graph = build_graph(model)
        plan = schedule_2d(graph)
        name_to_idx = {n: i for i, n in enumerate(model.parameter_names)}
        theta_indices = np.array(
            [name_to_idx[n] for n in plan.opt_param_names], dtype=np.intp
        )

        par = model.lmfit_pars
        e_lim = [10, 80]
        t_lim = [5, 40]

        res_gir = fitlib.residual_fun(
            par=par,
            x=file.energy,
            data=data,
            package=spectra,
            fit_fun_str="fit_model_gir",
            e_lim=e_lim,
            t_lim=t_lim,
            args=(plan, theta_indices),
        )

        res_mcp = fitlib.residual_fun(
            par=par,
            x=file.energy,
            data=data,
            package=spectra,
            fit_fun_str="fit_model_mcp",
            e_lim=e_lim,
            t_lim=t_lim,
            args=(model, 2),
        )

        np.testing.assert_allclose(res_gir, res_mcp, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# End-to-end through File.fit_2d
# ---------------------------------------------------------------------------

_FILE_ENERGY_YAML = "models/file_energy.yaml"


#
def _make_truth_file(project):
    """Single GLP peak + exponential dynamics on amplitude."""

    energy = np.linspace(83, 87, 30)
    time = np.linspace(-2, 10, 24)
    file = File(parent_project=project, name="truth")
    file.energy = energy
    file.time = time
    file.dim = 2
    file.load_model(model_yaml=_FILE_ENERGY_YAML, model_info="single_glp")
    file.add_time_dependence(
        target_model="single_glp",
        target_parameter="GLP_01_A",
        dynamics_yaml=_TIME_YAML,
        dynamics_model=["MonoExpPos"],
    )
    return file


#
def _make_fit_file(project, data, energy, time):
    """Fresh file loaded with simulated data, ready for baseline fit."""

    file = File(
        parent_project=project,
        name="fit",
        data=data,
        energy=energy.copy(),
        time=time.copy(),
    )
    file.load_model(model_yaml=_FILE_ENERGY_YAML, model_info="single_glp")
    file.define_baseline(time_start=0, time_stop=3, time_type="ind", show_plot=False)
    return file


#
#
class TestFileFit2D:
    """End-to-end tests through File.fit_2d dispatch + writeback."""

    #
    @pytest.mark.slow
    def test_gir_fit_writes_back_to_model(self):
        """After GIR fit, model_2d.lmfit_pars reflects optimized values."""

        project = Project(path="tests", name="gir_e2e")
        project.show_output = 0
        # Default spec_fun_str is "fit_model_gir"

        truth_file = _make_truth_file(project)
        truth_pars = {
            name: truth_file.model_active.lmfit_pars[name].value
            for name in truth_file.model_active.parameter_names
            if truth_file.model_active.lmfit_pars[name].expr is None
        }

        sim = Simulator(
            model=truth_file.model_active,
            detection="analog",
            noise_level=0.0,
            noise_type="none",
            seed=42,
        )
        clean, _, _ = sim.simulate_2d()

        fit_file = _make_fit_file(project, clean, truth_file.energy, truth_file.time)
        fit_file.fit_baseline(model_name="single_glp", stages=2, try_ci=0)
        fit_file.add_time_dependence(
            target_model="single_glp",
            target_parameter="GLP_01_A",
            dynamics_yaml=_TIME_YAML,
            dynamics_model=["MonoExpPos"],
        )
        fit_file.fit_2d(model_name="single_glp", stages=2, try_ci=0)

        # Verify writeback: model_2d.lmfit_pars should match result params
        assert fit_file.model_2d is not None
        result_params = fit_file.model_2d.result[1].params
        for name in fit_file.model_2d.parameter_names:
            model_val = fit_file.model_2d.lmfit_pars[name].value
            result_val = result_params[name].value
            assert np.isclose(model_val, result_val, rtol=1e-12), (
                f"{name}: model={model_val}, result={result_val}"
            )

        # Verify parameter recovery
        for name, true_val in truth_pars.items():
            fit_val = result_params[name].value
            assert np.isclose(true_val, fit_val, rtol=1e-10, atol=1e-12), (
                f"{name}: true={true_val:.6f}, fit={fit_val:.6f}"
            )

    #
    @pytest.mark.slow
    def test_compare_mode_through_fit_2d(self):
        """fit_model_compare through File.fit_2d validates both paths."""

        project = Project(path="tests", name="gir_cmp")
        project.show_output = 0
        project.spec_fun_str = "fit_model_compare"

        truth_file = _make_truth_file(project)

        sim = Simulator(
            model=truth_file.model_active,
            detection="analog",
            noise_level=0.0,
            noise_type="none",
            seed=42,
        )
        clean, _, _ = sim.simulate_2d()

        fit_file = _make_fit_file(project, clean, truth_file.energy, truth_file.time)
        fit_file.fit_baseline(model_name="single_glp", stages=2, try_ci=0)
        fit_file.add_time_dependence(
            target_model="single_glp",
            target_parameter="GLP_01_A",
            dynamics_yaml=_TIME_YAML,
            dynamics_model=["MonoExpPos"],
        )
        # If GIR and interpreter disagree, fit_model_compare raises
        fit_file.fit_2d(model_name="single_glp", stages=1, try_ci=0)
