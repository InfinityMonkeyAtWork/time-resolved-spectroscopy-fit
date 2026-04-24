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
from trspecfit.graph_ir import (
    ScheduledPlan1D,
    build_graph,
    can_lower_1d,
    can_lower_2d,
    schedule_1d,
    schedule_2d,
)

_ENERGY_YAML = "models/eval_2d_energy.yaml"
_TIME_YAML = "models/file_time.yaml"
_PROFILE_YAML = "models/file_profile.yaml"


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
def _make_2d_model(project, model_info, dynamics_params, *, frequency=None):
    """Load energy model + add dynamics -> 2D model.

    ``frequency`` (optional) is forwarded to every ``add_time_dependence``
    call, enabling subcycle-aware fixtures.
    """

    file = File(parent_project=project)
    file.energy = np.linspace(80, 90, 101)
    file.time = np.linspace(-10, 100, 51)
    file.load_model(model_yaml=_ENERGY_YAML, model_info=model_info)
    model = file.model_active
    assert model is not None

    for target_par, dyn_model in dynamics_params:
        kwargs = {
            "target_model": model_info[0],
            "target_parameter": target_par,
            "dynamics_yaml": _TIME_YAML,
            "dynamics_model": dyn_model,
        }
        if frequency is not None:
            kwargs["frequency"] = frequency
        file.add_time_dependence(**kwargs)

    return file, model


#
def _make_1d_profile_model(project, model_info, profiles):
    """Load a 1D energy model and attach parameter profiles."""

    file = File(
        parent_project=project,
        energy=np.linspace(83, 87, 121),
        aux_axis=np.linspace(0, 4, 5),
    )
    file.load_model(model_yaml="models/file_energy.yaml", model_info=model_info)
    for target_parameter, profile_model in profiles:
        file.add_par_profile(
            target_model=model_info[0],
            target_parameter=target_parameter,
            profile_yaml=_PROFILE_YAML,
            profile_model=profile_model,
        )
    model = file.model_active
    assert model is not None
    return file, model


#
def _make_2d_profile_model(project, model_info, dynamics_params, profiles):
    """Load energy model + add dynamics + profiles -> 2D profiled model."""

    file = File(
        parent_project=project,
        energy=np.linspace(83, 87, 61),
        time=np.linspace(-10, 100, 31),
        aux_axis=np.linspace(0, 4, 5),
    )
    file.load_model(model_yaml="models/file_energy.yaml", model_info=model_info)

    for target_par, dyn_model in dynamics_params:
        file.add_time_dependence(
            target_model=model_info[0],
            target_parameter=target_par,
            dynamics_yaml=_TIME_YAML,
            dynamics_model=dyn_model,
        )

    for target_parameter, profile_model in profiles:
        file.add_par_profile(
            target_model=model_info[0],
            target_parameter=target_parameter,
            profile_yaml=_PROFILE_YAML,
            profile_model=profile_model,
        )

    model = file.model_active
    assert model is not None
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
        result = spectra.fit_model_gir(
            file.energy, par, True, plan, theta_indices, model, 2
        )
        assert result.shape == (len(file.time), len(file.energy))

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
    def test_fit_2d_uses_gir_for_irf(self):
        """2D IRF models are lowered by the GIR backend."""

        project = _make_project()
        file, model = _make_2d_model(
            project,
            ["glp_only"],
            [("GLP_01_A", ["MonoExpPosIRF"])],
        )
        graph = build_graph(model)
        assert can_lower_2d(graph)

        plan = schedule_2d(graph)
        name_to_idx = {n: i for i, n in enumerate(model.parameter_names)}
        theta_indices = np.array(
            [name_to_idx[n] for n in plan.opt_param_names], dtype=np.intp
        )
        par = _extract_par_list(model)
        result = spectra.fit_model_gir(
            file.energy, par, True, plan, theta_indices, model, 2
        )
        assert result.shape == (len(file.time), len(file.energy))

    #
    def test_1d_fit_uses_gir_when_lowerable(self):
        """1D lowerable model dispatches through fit_model_gir fast path."""

        project = _make_project()
        file = File(parent_project=project, energy=np.linspace(80, 90, 101))
        file.load_model(model_yaml=_ENERGY_YAML, model_info=["glp_only"])
        model = file.model_active
        assert model is not None

        graph = build_graph(model)
        assert can_lower_1d(graph)

        plan = schedule_1d(graph)
        name_to_idx = {n: i for i, n in enumerate(model.parameter_names)}
        theta_indices = np.array(
            [name_to_idx[n] for n in plan.opt_param_names], dtype=np.intp
        )

        par = _extract_par_list(model)
        result = spectra.fit_model_gir(
            file.energy, par, True, plan, theta_indices, model, 1
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (len(file.energy),)

    #
    def test_1d_profile_fit_uses_gir_when_lowerable(self):
        """Profile-only 1D models now dispatch through the compiled fast path."""

        project = _make_project()
        file, model = _make_1d_profile_model(
            project,
            ["single_gauss"],
            [("Gauss_01_A", ["profile_pExpDecay"])],
        )

        graph = build_graph(model)
        assert can_lower_1d(graph)

        plan = schedule_1d(graph)
        name_to_idx = {n: i for i, n in enumerate(model.parameter_names)}
        theta_indices = np.array(
            [name_to_idx[n] for n in plan.opt_param_names], dtype=np.intp
        )

        par = _extract_par_list(model)
        result = spectra.fit_model_gir(
            file.energy, par, True, plan, theta_indices, model, 1
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (len(file.energy),)

    #
    def test_2d_profile_fit_uses_gir_when_lowerable(self):
        """2D profiled model dispatches through the compiled fast path."""

        project = _make_project()
        file, model = _make_2d_profile_model(
            project,
            ["single_gauss"],
            [("Gauss_01_x0", ["MonoExpPos"])],
            [("Gauss_01_A", ["profile_pExpDecay"])],
        )

        graph = build_graph(model)
        assert can_lower_2d(graph)

        plan = schedule_2d(graph)
        name_to_idx = {n: i for i, n in enumerate(model.parameter_names)}
        theta_indices = np.array(
            [name_to_idx[n] for n in plan.opt_param_names], dtype=np.intp
        )

        par = _extract_par_list(model)
        result = spectra.fit_model_gir(
            file.energy, par, True, plan, theta_indices, model, 2
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (len(file.time), len(file.energy))

    #
    def test_1d_plot_sum_false_falls_back(self):
        """1D with plot_sum=False falls back to MCP for component extraction."""

        project = _make_project()
        file = File(parent_project=project, energy=np.linspace(80, 90, 101))
        file.load_model(model_yaml=_ENERGY_YAML, model_info=["offset_only"])
        model = file.model_active
        assert model is not None

        graph = build_graph(model)
        plan = schedule_1d(graph)
        name_to_idx = {n: i for i, n in enumerate(model.parameter_names)}
        theta_indices = np.array(
            [name_to_idx[n] for n in plan.opt_param_names], dtype=np.intp
        )

        par = _extract_par_list(model)
        result = spectra.fit_model_gir(
            file.energy, par, False, plan, theta_indices, model, 1
        )
        # plot_sum=False returns list of component spectra
        assert isinstance(result, list)

    #
    def test_1d_mcp_fallback_when_non_lowerable(self):
        """1D non-lowerable model falls back to interpreter."""

        project = _make_project()
        file = File(parent_project=project, energy=np.linspace(80, 90, 101))
        file.load_model(model_yaml=_ENERGY_YAML, model_info=["glp_only"])
        model = file.model_active
        assert model is not None

        par = _extract_par_list(model)
        # Pass (model, dim=1) — no plan → delegates to MCP
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
            fit_fun_str="fit_model_gir",
            args=(plan, theta_indices, model, 2),
        )

        # Residual via MCP
        res_mcp = fitlib.residual_fun(
            par=par,
            x=file.energy,
            data=data,
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
            fit_fun_str="fit_model_gir",
            e_lim=e_lim,
            t_lim=t_lim,
            args=(plan, theta_indices, model, 2),
        )

        res_mcp = fitlib.residual_fun(
            par=par,
            x=file.energy,
            data=data,
            fit_fun_str="fit_model_mcp",
            e_lim=e_lim,
            t_lim=t_lim,
            args=(model, 2),
        )

        np.testing.assert_allclose(res_gir, res_mcp, rtol=1e-10, atol=1e-10)

    #
    @pytest.mark.parametrize(
        "dyn_model",
        [
            "MonoExpPosIRF",
            "MonoExpPosLorentzIRF",
            "MonoExpPosVoigtIRF",
            "MonoExpPosExpSymIRF",
            "MonoExpPosExpDecayIRF",
            "MonoExpPosExpRiseIRF",
            "MonoExpPosBoxIRF",
        ],
    )
    def test_residual_same_gir_vs_mcp_irf(self, dyn_model):
        """residual_fun parity for IRF / CONVOLUTION dynamics across all
        lowerable kernel functions.
        """

        project = _make_project()
        file, model = _make_2d_model(
            project,
            ["glp_only"],
            [("GLP_01_A", [dyn_model])],
        )

        model.create_value_2d()
        assert model.value_2d is not None
        data = model.value_2d + 0.01

        graph = build_graph(model)
        assert can_lower_2d(graph)
        plan = schedule_2d(graph)
        name_to_idx = {n: i for i, n in enumerate(model.parameter_names)}
        theta_indices = np.array(
            [name_to_idx[n] for n in plan.opt_param_names], dtype=np.intp
        )

        par = model.lmfit_pars

        res_gir = fitlib.residual_fun(
            par=par,
            x=file.energy,
            data=data,
            fit_fun_str="fit_model_gir",
            args=(plan, theta_indices, model, 2),
        )

        res_mcp = fitlib.residual_fun(
            par=par,
            x=file.energy,
            data=data,
            fit_fun_str="fit_model_mcp",
            args=(model, 2),
        )

        np.testing.assert_allclose(res_gir, res_mcp, rtol=1e-10, atol=1e-10)

    #
    def test_compare_mode_irf(self):
        """fit_model_compare exercises both paths for an IRF 2D fit."""

        project = _make_project(spec_fun_str="fit_model_compare")
        file, model = _make_2d_model(
            project,
            ["glp_only"],
            [("GLP_01_A", ["MonoExpPosIRF"])],
        )

        graph = build_graph(model)
        assert can_lower_2d(graph)
        plan = schedule_2d(graph)
        name_to_idx = {n: i for i, n in enumerate(model.parameter_names)}
        theta_indices = np.array(
            [name_to_idx[n] for n in plan.opt_param_names], dtype=np.intp
        )
        par = _extract_par_list(model)

        # fit_model_compare asserts GIR == MCP internally at rtol=atol=1e-10.
        result = spectra.fit_model_compare(
            file.energy, par, True, plan, theta_indices, model, 2
        )
        assert result.shape == (len(file.time), len(file.energy))

    #
    def test_subcycle_residual_gir_vs_mcp(self):
        """Subcycle dynamics residual matches between GIR and MCP."""

        project = _make_project()
        file, model = _make_2d_model(
            project,
            ["glp_only"],
            [("GLP_01_A", ["ModelNone", "MonoExpNeg", "MonoExpPosExpr"])],
            frequency=10,
        )

        model.create_value_2d()
        assert model.value_2d is not None
        data = model.value_2d + 0.01

        graph = build_graph(model)
        assert can_lower_2d(graph)
        plan = schedule_2d(graph)
        name_to_idx = {n: i for i, n in enumerate(model.parameter_names)}
        theta_indices = np.array(
            [name_to_idx[n] for n in plan.opt_param_names], dtype=np.intp
        )
        par = model.lmfit_pars

        res_gir = fitlib.residual_fun(
            par=par,
            x=file.energy,
            data=data,
            fit_fun_str="fit_model_gir",
            args=(plan, theta_indices, model, 2),
        )
        res_mcp = fitlib.residual_fun(
            par=par,
            x=file.energy,
            data=data,
            fit_fun_str="fit_model_mcp",
            args=(model, 2),
        )
        np.testing.assert_allclose(res_gir, res_mcp, rtol=1e-10, atol=1e-10)

    #
    def test_subcycle_compare_mode(self):
        """fit_model_compare runs GIR and MCP on a subcycle model without error."""

        project = _make_project(spec_fun_str="fit_model_compare")
        file, model = _make_2d_model(
            project,
            ["glp_only"],
            [("GLP_01_A", ["ModelNone", "MonoExpNeg"])],
            frequency=10,
        )

        graph = build_graph(model)
        assert can_lower_2d(graph)
        plan = schedule_2d(graph)
        name_to_idx = {n: i for i, n in enumerate(model.parameter_names)}
        theta_indices = np.array(
            [name_to_idx[n] for n in plan.opt_param_names], dtype=np.intp
        )
        par = _extract_par_list(model)

        result = spectra.fit_model_compare(
            file.energy, par, True, plan, theta_indices, model, 2
        )
        assert result.shape == (len(file.time), len(file.energy))


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


# ---------------------------------------------------------------------------
# 1D GIR vs interpreter residual parity
# ---------------------------------------------------------------------------


#
#
class TestGIR1DvsInterpreter:
    """Compare 1D GIR and interpreter outputs through the pipeline."""

    #
    def test_residual_same_gir_vs_mcp_1d(self):
        """1D residual from residual_fun matches between GIR and MCP paths."""

        project = _make_project()
        file = File(parent_project=project, energy=np.linspace(80, 90, 101))
        file.load_model(model_yaml=_ENERGY_YAML, model_info=["glp_only"])
        model = file.model_active
        assert model is not None

        # Generate synthetic data
        model.create_value_1d()
        assert model.value_1d is not None
        data = model.value_1d + 0.01

        # Compile GIR 1D path
        graph = build_graph(model)
        assert can_lower_1d(graph)
        plan = schedule_1d(graph)
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
            fit_fun_str="fit_model_gir",
            args=(plan, theta_indices, model, 1),
        )

        # Residual via MCP
        res_mcp = fitlib.residual_fun(
            par=par,
            x=file.energy,
            data=data,
            fit_fun_str="fit_model_mcp",
            args=(model, 1),
        )

        np.testing.assert_allclose(res_gir, res_mcp, rtol=1e-10, atol=1e-10)

    #
    def test_residual_with_e_lim_1d(self):
        """1D GIR residual with e_lim matches MCP residual."""

        project = _make_project()
        file = File(parent_project=project, energy=np.linspace(80, 90, 101))
        file.load_model(model_yaml=_ENERGY_YAML, model_info=["offset_only"])
        model = file.model_active
        assert model is not None

        model.create_value_1d()
        data = model.value_1d + 0.01

        graph = build_graph(model)
        plan = schedule_1d(graph)
        name_to_idx = {n: i for i, n in enumerate(model.parameter_names)}
        theta_indices = np.array(
            [name_to_idx[n] for n in plan.opt_param_names], dtype=np.intp
        )

        par = model.lmfit_pars
        e_lim = [10, 80]

        res_gir = fitlib.residual_fun(
            par=par,
            x=file.energy,
            data=data,
            fit_fun_str="fit_model_gir",
            e_lim=e_lim,
            args=(plan, theta_indices, model, 1),
        )

        res_mcp = fitlib.residual_fun(
            par=par,
            x=file.energy,
            data=data,
            fit_fun_str="fit_model_mcp",
            e_lim=e_lim,
            args=(model, 1),
        )

        np.testing.assert_allclose(res_gir, res_mcp, rtol=1e-10, atol=1e-10)

    #
    def test_residual_same_gir_vs_mcp_profile_1d(self):
        """Profile-aware 1D residuals match between GIR and MCP paths."""

        project = _make_project()
        file, model = _make_1d_profile_model(
            project,
            ["two_glp_expr_amplitude"],
            [("GLP_01_A", ["profile_pLinear"])],
        )

        model.create_value_1d()
        assert model.value_1d is not None
        data = model.value_1d + 0.01

        graph = build_graph(model)
        assert can_lower_1d(graph)
        plan = schedule_1d(graph)
        name_to_idx = {n: i for i, n in enumerate(model.parameter_names)}
        theta_indices = np.array(
            [name_to_idx[n] for n in plan.opt_param_names], dtype=np.intp
        )

        par = model.lmfit_pars
        res_gir = fitlib.residual_fun(
            par=par,
            x=file.energy,
            data=data,
            fit_fun_str="fit_model_gir",
            args=(plan, theta_indices, model, 1),
        )
        res_mcp = fitlib.residual_fun(
            par=par,
            x=file.energy,
            data=data,
            fit_fun_str="fit_model_mcp",
            args=(model, 1),
        )

        np.testing.assert_allclose(res_gir, res_mcp, rtol=1e-10, atol=1e-10)

    #
    def test_compare_mode_1d(self):
        """1D fit_model_compare runs both paths without error."""

        project = _make_project(spec_fun_str="fit_model_compare")
        file = File(parent_project=project, energy=np.linspace(80, 90, 101))
        file.load_model(model_yaml=_ENERGY_YAML, model_info=["glp_expression"])
        model = file.model_active
        assert model is not None

        graph = build_graph(model)
        assert can_lower_1d(graph)
        plan = schedule_1d(graph)
        name_to_idx = {n: i for i, n in enumerate(model.parameter_names)}
        theta_indices = np.array(
            [name_to_idx[n] for n in plan.opt_param_names], dtype=np.intp
        )
        par = _extract_par_list(model)

        result = spectra.fit_model_compare(
            file.energy, par, True, plan, theta_indices, model, 1
        )
        assert result.shape == (len(file.energy),)

    #
    def test_compare_mode_profile_1d(self):
        """Profile-aware 1D models run through compare mode without mismatch."""

        project = _make_project(spec_fun_str="fit_model_compare")
        file, model = _make_1d_profile_model(
            project,
            ["single_gauss"],
            [("Gauss_01_A", ["profile_pExpDecay"])],
        )

        graph = build_graph(model)
        assert can_lower_1d(graph)
        plan = schedule_1d(graph)
        name_to_idx = {n: i for i, n in enumerate(model.parameter_names)}
        theta_indices = np.array(
            [name_to_idx[n] for n in plan.opt_param_names], dtype=np.intp
        )
        par = _extract_par_list(model)

        result = spectra.fit_model_compare(
            file.energy, par, True, plan, theta_indices, model, 1
        )
        assert result.shape == (len(file.energy),)

    #
    def test_residual_same_gir_vs_mcp_profile_2d(self):
        """2D profile residuals match between GIR and MCP paths."""

        project = _make_project()
        file, model = _make_2d_profile_model(
            project,
            ["single_gauss"],
            [("Gauss_01_x0", ["MonoExpPos"])],
            [("Gauss_01_A", ["profile_pExpDecay"])],
        )

        model.create_value_2d()
        assert model.value_2d is not None
        data = model.value_2d + 0.01

        graph = build_graph(model)
        assert can_lower_2d(graph)
        plan = schedule_2d(graph)
        name_to_idx = {n: i for i, n in enumerate(model.parameter_names)}
        theta_indices = np.array(
            [name_to_idx[n] for n in plan.opt_param_names], dtype=np.intp
        )

        par = model.lmfit_pars
        res_gir = fitlib.residual_fun(
            par=par,
            x=file.energy,
            data=data,
            fit_fun_str="fit_model_gir",
            args=(plan, theta_indices, model, 2),
        )
        res_mcp = fitlib.residual_fun(
            par=par,
            x=file.energy,
            data=data,
            fit_fun_str="fit_model_mcp",
            args=(model, 2),
        )

        np.testing.assert_allclose(res_gir, res_mcp, rtol=1e-10, atol=1e-10)

    #
    def test_compare_mode_profile_2d(self):
        """2D profiled models run through compare mode without mismatch."""

        project = _make_project(spec_fun_str="fit_model_compare")
        file, model = _make_2d_profile_model(
            project,
            ["single_gauss"],
            [("Gauss_01_x0", ["MonoExpPos"])],
            [("Gauss_01_A", ["profile_pExpDecay"])],
        )

        graph = build_graph(model)
        assert can_lower_2d(graph)
        plan = schedule_2d(graph)
        name_to_idx = {n: i for i, n in enumerate(model.parameter_names)}
        theta_indices = np.array(
            [name_to_idx[n] for n in plan.opt_param_names], dtype=np.intp
        )
        par = _extract_par_list(model)

        result = spectra.fit_model_compare(
            file.energy, par, True, plan, theta_indices, model, 2
        )
        assert result.shape == (len(file.time), len(file.energy))


# ---------------------------------------------------------------------------
# End-to-end through File.fit_baseline / File.fit_spectrum (1D)
# ---------------------------------------------------------------------------


#
def _make_1d_truth_file(project):
    """Single GLP peak truth model for 1D baseline fitting."""

    energy = np.linspace(83, 87, 50)
    time = np.linspace(-2, 10, 12)
    file = File(parent_project=project, name="truth_1d", energy=energy, time=time)
    file.dim = 2
    file.load_model(model_yaml=_FILE_ENERGY_YAML, model_info="single_glp")
    return file


#
def _make_1d_fit_file(project, data_2d, energy, time):
    """Fresh file with 2D data, baseline defined, ready for fit_baseline."""

    file = File(
        parent_project=project,
        name="fit_1d",
        data=data_2d,
        energy=energy.copy(),
        time=time.copy(),
    )
    file.load_model(model_yaml=_FILE_ENERGY_YAML, model_info="single_glp")
    file.define_baseline(time_start=0, time_stop=3, time_type="ind", show_plot=False)
    return file


#
#
class TestFileFitBaseline:
    """End-to-end tests through File.fit_baseline dispatch + writeback."""

    #
    def test_1d_dispatch_args_lower_on_2d_file(self):
        """1D workflow args compile on a 2D File with a plain energy model."""

        project = _make_project()
        energy = np.linspace(83, 87, 50)
        time = np.linspace(-2, 10, 12)
        fit_file = File(
            parent_project=project,
            name="fit_1d_lowerable",
            data=np.zeros((len(time), len(energy))),
            energy=energy.copy(),
            time=time.copy(),
        )
        fit_file.load_model(model_yaml=_FILE_ENERGY_YAML, model_info="single_glp")

        args = fit_file._build_1d_dispatch_args(fit_file.model_active, "fit_model_gir")
        # Contract: (ScheduledPlan1D, theta_indices, original model, dim==1).
        assert len(args) == 4
        assert isinstance(args[0], ScheduledPlan1D)
        assert isinstance(args[1], np.ndarray)
        assert args[1].dtype == np.intp
        assert len(args[1]) == len(args[0].opt_param_names)
        assert args[2] is fit_file.model_active
        assert args[3] == 1

    #
    @pytest.mark.slow
    def test_gir_baseline_writes_back(self):
        """After GIR baseline fit, model_base.lmfit_pars reflects results."""

        project = Project(path="tests", name="gir_base")
        project.show_output = 0

        truth = _make_1d_truth_file(project)
        truth_pars = {
            name: truth.model_active.lmfit_pars[name].value
            for name in truth.model_active.parameter_names
            if truth.model_active.lmfit_pars[name].expr is None
        }

        # Tile 1D truth spectrum into 2D data (constant across time)
        truth.model_active.create_value_1d()
        assert truth.model_active.value_1d is not None
        spectrum_1d = truth.model_active.value_1d.copy()
        data_2d = np.tile(spectrum_1d, (len(truth.time), 1))

        fit_file = _make_1d_fit_file(project, data_2d, truth.energy, truth.time)
        fit_file.fit_baseline(model_name="single_glp", stages=2, try_ci=0)

        # Verify writeback
        assert fit_file.model_base is not None
        result_params = fit_file.model_base.result[1].params
        for name in fit_file.model_base.parameter_names:
            model_val = fit_file.model_base.lmfit_pars[name].value
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
    def test_compare_mode_through_baseline(self):
        """fit_model_compare through File.fit_baseline validates both paths."""

        project = Project(path="tests", name="gir_base_cmp")
        project.show_output = 0
        project.spec_fun_str = "fit_model_compare"

        truth = _make_1d_truth_file(project)
        truth.model_active.create_value_1d()
        spectrum_1d = truth.model_active.value_1d.copy()
        data_2d = np.tile(spectrum_1d, (len(truth.time), 1))

        fit_file = _make_1d_fit_file(project, data_2d, truth.energy, truth.time)
        # If GIR and interpreter disagree, fit_model_compare raises
        fit_file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)


#
#
class TestFileFitSpectrum:
    """End-to-end tests through File.fit_spectrum dispatch + writeback."""

    #
    @pytest.mark.slow
    def test_compare_mode_through_fit_spectrum(self):
        """fit_model_compare through File.fit_spectrum validates both paths."""

        project = Project(path="tests", name="gir_spec_cmp")
        project.show_output = 0
        project.spec_fun_str = "fit_model_compare"

        # Build 2D data: tile a 1D spectrum across time so every slice
        # is identical and recovery is deterministic.
        energy = np.linspace(83, 87, 50)
        time = np.linspace(-2, 10, 12)
        truth_file = File(parent_project=project, name="truth_spec", energy=energy)
        truth_file.load_model(model_yaml=_FILE_ENERGY_YAML, model_info="single_glp")
        truth_file.model_active.create_value_1d()
        spectrum_1d = truth_file.model_active.value_1d.copy()
        data_2d = np.tile(spectrum_1d, (len(time), 1))

        fit_file = File(
            parent_project=project,
            name="fit_spec",
            data=data_2d,
            energy=energy.copy(),
            time=time.copy(),
        )
        fit_file.load_model(model_yaml=_FILE_ENERGY_YAML, model_info="single_glp")
        # If GIR and interpreter disagree, fit_model_compare raises
        fit_file.fit_spectrum(
            model_name="single_glp",
            time_point=5.0,
            stages=1,
            show_plot=False,
            try_ci=0,
        )


#
#
class TestFileFitSliceBySlice:
    """End-to-end tests through File.fit_slice_by_slice dispatch."""

    #
    @pytest.mark.slow
    def test_compare_mode_through_fit_slice_by_slice(self):
        """fit_slice_by_slice uses the 1D GIR path when the model lowers."""

        project = Project(path="tests", name="gir_sbs_cmp")
        project.show_output = 0
        project.spec_fun_str = "fit_model_compare"
        # Fit 3 slices so we exercise the hoisted-args reuse across the loop
        # and the multi-row save_sbs_fit reconstruction path.
        project.first_n_spec_only = 2

        truth = _make_1d_truth_file(project)
        truth.model_active.create_value_1d()
        spectrum_1d = truth.model_active.value_1d.copy()
        data_2d = np.tile(spectrum_1d, (len(truth.time), 1))

        fit_file = _make_1d_fit_file(project, data_2d, truth.energy, truth.time)
        fit_file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)

        # If GIR and interpreter disagree, fit_model_compare raises.
        fit_file.fit_slice_by_slice(model_name="single_glp", stages=1, try_ci=0)

        assert fit_file.model_sbs is not None
        assert fit_file.model_sbs.args is not None
        assert len(fit_file.model_sbs.args) == 4
        assert isinstance(fit_file.model_sbs.args[0], ScheduledPlan1D)
        assert len(fit_file.results_sbs) == 3
