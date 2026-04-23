"""Numerical validation: evaluate_2d vs interpreter for every OpKind.

For each model we:
1. Build the OOP model (load YAML + optionally add dynamics).
2. Compile with build_graph / schedule_2d.
3. Extract theta from model.lmfit_pars in plan.opt_param_names order.
4. Compare evaluate_2d(plan, theta) against model.create_value_2d().
5. Perturb theta and repeat to catch ordering bugs.
"""

import numpy as np
import pytest

from trspecfit import File, Project
from trspecfit.eval_2d import evaluate_2d
from trspecfit.graph_ir import (
    OpKind,
    build_graph,
    can_lower_2d,
    schedule_2d,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENERGY_YAML = "models/eval_2d_energy.yaml"
_TIME_YAML = "models/file_time.yaml"


#
def _make_energy_model(model_info):
    """Load a 1D energy model from eval_2d_energy.yaml."""

    project = Project(path="tests")
    file = File(parent_project=project)
    file.energy = np.linspace(80, 90, 101)
    file.load_model(model_yaml=_ENERGY_YAML, model_info=model_info)
    model = file.model_active
    assert model is not None
    return file, model


#
def _make_2d_model(model_info, dynamics_params, *, frequency=None, time=None):
    """Load energy model + add dynamics -> 2D model.

    ``dynamics_params`` is a list of ``(target_par, dyn_model)`` tuples
    passed to ``File.add_time_dependence``.  ``frequency`` (optional) is
    forwarded to every call so subcycle-aware fixtures can set it in one
    place; ``time`` (optional) overrides the default time axis.
    """

    project = Project(path="tests")
    file = File(parent_project=project)
    file.energy = np.linspace(80, 90, 101)
    file.time = np.linspace(-10, 100, 51) if time is None else time
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
def _extract_theta(plan, model):
    """Extract theta vector from model.lmfit_pars in plan order."""

    return np.array(
        [model.lmfit_pars[name].value for name in plan.opt_param_names],
        dtype=np.float64,
    )


#
def _compare_evaluator_vs_interpreter(model, plan, *, rtol=1e-10):
    """Run evaluate_2d and interpreter, assert they match."""

    theta = _extract_theta(plan, model)
    fast = evaluate_2d(plan, theta)
    model.create_value_2d()
    slow = model.value_2d
    np.testing.assert_allclose(fast, slow, rtol=rtol, atol=1e-10)
    return theta


#
def _perturb_theta(plan, model, theta, indices, deltas):
    """Perturb theta at given indices, update model, and compare."""

    theta_new = theta.copy()
    for idx, delta in zip(indices, deltas, strict=True):
        theta_new[idx] += delta
        # Also update model.lmfit_pars for interpreter comparison
        name = plan.opt_param_names[idx]
        model.lmfit_pars[name].value = float(theta_new[idx])

    fast = evaluate_2d(plan, theta_new)
    model.create_value_2d()
    slow = model.value_2d
    np.testing.assert_allclose(fast, slow, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# Static 2D models (no dynamics -- all params constant over time)
# ---------------------------------------------------------------------------


#
#
class TestStaticModels:
    """Each OpKind with all-static parameters (no time dependence)."""

    def _run_static(self, model_info):
        """Shared pattern: load, compile, compare at initial + perturbed theta."""

        file, model = _make_2d_model(model_info, [])
        graph = build_graph(model)
        assert can_lower_2d(graph)
        plan = schedule_2d(graph)
        theta = _compare_evaluator_vs_interpreter(model, plan)

        # Perturb at least two params at once to catch order bugs.
        # Use 1% relative perturbation to stay within lmfit bounds.
        n_opt = len(plan.opt_param_names)
        if n_opt >= 2:
            d0 = 0.01 * abs(theta[0]) if theta[0] != 0 else 0.01
            d1 = -0.01 * abs(theta[1]) if theta[1] != 0 else -0.01
            _perturb_theta(plan, model, theta, [0, 1], [d0, d1])

    def test_gauss(self):
        self._run_static(["gauss_only"])

    def test_gauss_asym(self):
        self._run_static(["gauss_asym_only"])

    def test_lorentz(self):
        self._run_static(["lorentz_only"])

    def test_voigt(self):
        self._run_static(["voigt_only"])

    def test_gls(self):
        self._run_static(["gls_only"])

    def test_glp(self):
        self._run_static(["glp_only"])

    def test_ds(self):
        self._run_static(["ds_only"])

    def test_offset_peak(self):
        self._run_static(["offset_only"])

    def test_linback_peak(self):
        self._run_static(["linback_peak"])

    def test_shirley_peak(self):
        self._run_static(["shirley_peak"])

    def test_expression(self):
        self._run_static(["glp_expression"])


# ---------------------------------------------------------------------------
# Time-dependent 2D models (dynamics on one or more params)
# ---------------------------------------------------------------------------


#
#
class TestDynamicModels:
    """Models with time-dependent parameters via dynamics subgraphs."""

    def test_gauss_time_dep_amplitude(self):
        """Gauss with A time-dependent via expFun."""

        _file, model = _make_2d_model(
            ["gauss_only"],
            [("Gauss_01_A", ["MonoExpPos"])],
        )
        graph = build_graph(model)
        assert can_lower_2d(graph)
        plan = schedule_2d(graph)
        theta = _compare_evaluator_vs_interpreter(model, plan)

        # Perturb A and dynamics tau simultaneously
        A_idx = plan.opt_param_names.index("Gauss_01_A")
        tau_idx = plan.opt_param_names.index("Gauss_01_A_expFun_01_tau")
        _perturb_theta(plan, model, theta, [A_idx, tau_idx], [2.0, 1.0])

    def test_lorentz_time_dep_position(self):
        """Lorentz with x0 time-dependent."""

        _file, model = _make_2d_model(
            ["lorentz_only"],
            [("Lorentz_01_x0", ["MonoExpPos"])],
        )
        graph = build_graph(model)
        assert can_lower_2d(graph)
        plan = schedule_2d(graph)
        theta = _compare_evaluator_vs_interpreter(model, plan)

        x0_idx = plan.opt_param_names.index("Lorentz_01_x0")
        A_idx = plan.opt_param_names.index("Lorentz_01_A")
        _perturb_theta(plan, model, theta, [x0_idx, A_idx], [0.5, 3.0])

    def test_voigt_time_dep_amplitude(self):
        """Voigt with A time-dependent."""

        _file, model = _make_2d_model(
            ["voigt_only"],
            [("Voigt_01_A", ["MonoExpPos"])],
        )
        graph = build_graph(model)
        assert can_lower_2d(graph)
        plan = schedule_2d(graph)
        theta = _compare_evaluator_vs_interpreter(model, plan)

        A_idx = plan.opt_param_names.index("Voigt_01_A")
        W_idx = plan.opt_param_names.index("Voigt_01_W")
        _perturb_theta(plan, model, theta, [A_idx, W_idx], [2.0, 0.2])

    def test_glp_time_dep_amplitude(self):
        """GLP with A time-dependent -- the standard test case."""

        _file, model = _make_2d_model(
            ["glp_only"],
            [("GLP_01_A", ["MonoExpPos"])],
        )
        graph = build_graph(model)
        assert can_lower_2d(graph)
        plan = schedule_2d(graph)
        theta = _compare_evaluator_vs_interpreter(model, plan)

        A_idx = plan.opt_param_names.index("GLP_01_A")
        F_idx = plan.opt_param_names.index("GLP_01_F")
        _perturb_theta(plan, model, theta, [A_idx, F_idx], [5.0, 0.3])

    def test_gls_time_dep_mixing(self):
        """GLS with mixing parameter m time-dependent."""

        _file, model = _make_2d_model(
            ["gls_only"],
            [("GLS_01_m", ["MonoExpPos"])],
        )
        graph = build_graph(model)
        assert can_lower_2d(graph)
        plan = schedule_2d(graph)
        _compare_evaluator_vs_interpreter(model, plan)

    def test_ds_time_dep_amplitude(self):
        """DS with A time-dependent."""

        _file, model = _make_2d_model(
            ["ds_only"],
            [("DS_01_A", ["MonoExpPos"])],
        )
        graph = build_graph(model)
        assert can_lower_2d(graph)
        plan = schedule_2d(graph)
        _compare_evaluator_vs_interpreter(model, plan)

    def test_gauss_asym_time_dep_ratio(self):
        """GaussAsym with ratio time-dependent."""

        _file, model = _make_2d_model(
            ["gauss_asym_only"],
            [("GaussAsym_01_ratio", ["MonoExpPos"])],
        )
        graph = build_graph(model)
        assert can_lower_2d(graph)
        plan = schedule_2d(graph)
        _compare_evaluator_vs_interpreter(model, plan)

    def test_shirley_time_dep_peak(self):
        """Shirley + GLP where GLP amplitude is time-dependent."""

        _file, model = _make_2d_model(
            ["shirley_peak"],
            [("GLP_01_A", ["MonoExpPos"])],
        )
        graph = build_graph(model)
        assert can_lower_2d(graph)
        plan = schedule_2d(graph)
        _compare_evaluator_vs_interpreter(model, plan)

    def test_offset_time_dep(self):
        """Offset y0 time-dependent + GLP."""

        _file, model = _make_2d_model(
            ["offset_only"],
            [("Offset_y0", ["MonoExpPos"])],
        )
        graph = build_graph(model)
        assert can_lower_2d(graph)
        plan = schedule_2d(graph)
        _compare_evaluator_vs_interpreter(model, plan)

    def test_expression_with_dynamics(self):
        """Expression model where the base param (GLP_01_A) is time-dependent.

        GLP_02_A = 3/4 * GLP_01_A should update when GLP_01_A changes over time.
        """

        _file, model = _make_2d_model(
            ["glp_expression"],
            [("GLP_01_A", ["MonoExpPos"])],
        )
        graph = build_graph(model)
        assert can_lower_2d(graph)
        plan = schedule_2d(graph)
        theta = _compare_evaluator_vs_interpreter(model, plan)

        # Perturb GLP_01_A and Offset_y0 simultaneously
        A_idx = plan.opt_param_names.index("GLP_01_A")
        y0_idx = plan.opt_param_names.index("Offset_y0")
        _perturb_theta(plan, model, theta, [A_idx, y0_idx], [3.0, 0.5])

    def test_biexp_shared_t0(self):
        """Bi-exponential dynamics with expFun_02_t0 = "expFun_01_t0".

        The expression must be resolved before the second dynamics trace
        is computed. Without interleaved resolution order, the compiled
        path uses stale initial values after theta changes.
        """

        _file, model = _make_2d_model(
            ["glp_only"],
            [("GLP_01_A", ["BiExpSharedT0"])],
        )
        graph = build_graph(model)
        assert can_lower_2d(graph)
        plan = schedule_2d(graph)
        theta = _compare_evaluator_vs_interpreter(model, plan)

        # Perturb two dynamics params to expose ordering bugs
        A1_idx = plan.opt_param_names.index("GLP_01_A_expFun_01_A")
        tau2_idx = plan.opt_param_names.index("GLP_01_A_expFun_02_tau")
        _perturb_theta(plan, model, theta, [A1_idx, tau2_idx], [1.0, 5.0])


# ---------------------------------------------------------------------------
# Theta contract
# ---------------------------------------------------------------------------


#
#
class TestThetaContract:
    """Verify evaluate_2d rejects wrong-length theta."""

    def test_wrong_length_raises(self):
        _file, model = _make_2d_model(["glp_only"], [])
        graph = build_graph(model)
        plan = schedule_2d(graph)

        with pytest.raises(ValueError, match="theta length"):
            evaluate_2d(plan, np.array([1.0, 2.0]))  # wrong length

    def test_empty_theta_if_no_free_params(self):
        """If all params are fixed, theta should be empty."""

        _file, model = _make_2d_model(["gauss_only"], [])
        # Fix all parameters
        for par in model.lmfit_pars.values():
            par.vary = False

        graph = build_graph(model)
        plan = schedule_2d(graph)
        assert len(plan.opt_indices) == 0

        result = evaluate_2d(plan, np.array([], dtype=np.float64))
        assert result.shape == (51, 101)


# ---------------------------------------------------------------------------
# Static component caching
# ---------------------------------------------------------------------------


#
#
class TestStaticComponentCaching:
    """Verify constant components are precomputed in the compiled plan."""

    def test_fixed_offset_is_cached(self):
        """Offset with fixed params is cached and skipped in the hot path."""

        _file, model = _make_2d_model(["cached_offset_peak"], [])
        graph = build_graph(model)
        assert can_lower_2d(graph)
        plan = schedule_2d(graph)

        offset_indices = [
            i for i in range(plan.n_ops) if plan.op_kinds[i] == int(OpKind.OFFSET)
        ]
        assert len(offset_indices) == 1
        assert plan.op_is_constant[offset_indices[0]]
        assert np.any(plan.cached_result)
        np.testing.assert_allclose(plan.cached_peak_sum, 0.0, atol=1e-12)

        theta = _compare_evaluator_vs_interpreter(model, plan)
        A_idx = plan.opt_param_names.index("GLP_01_A")
        _perturb_theta(plan, model, theta, [A_idx], [2.0])

    def test_fixed_peak_populates_cached_peak_sum(self):
        """A fixed peak still contributes to Shirley via cached peak_sum."""

        _file, model = _make_2d_model(["cached_shirley_peak"], [])
        graph = build_graph(model)
        assert can_lower_2d(graph)
        plan = schedule_2d(graph)

        glp_indices = [
            i for i in range(plan.n_ops) if plan.op_kinds[i] == int(OpKind.GLP)
        ]
        shirley_indices = [
            i for i in range(plan.n_ops) if plan.op_kinds[i] == int(OpKind.SHIRLEY)
        ]
        assert len(glp_indices) == 1
        assert len(shirley_indices) == 1
        assert plan.op_is_constant[glp_indices[0]]
        assert not plan.op_is_constant[shirley_indices[0]]
        assert np.any(plan.cached_peak_sum)

        theta = _compare_evaluator_vs_interpreter(model, plan)
        p_idx = plan.opt_param_names.index("Shirley_pShirley")
        _perturb_theta(plan, model, theta, [p_idx], [1.0e-4])


# ---------------------------------------------------------------------------
# Profile models (2D with aux-axis-varying parameters)
# ---------------------------------------------------------------------------

_FILE_ENERGY_YAML = "models/file_energy.yaml"
_PROFILE_YAML = "models/file_profile.yaml"


#
def _make_2d_profile_model(
    model_info,
    dynamics_params,
    profiles,
    *,
    energy=None,
    time=None,
    aux_axis=None,
    model_yaml=None,
):
    """Load energy model + add dynamics + profiles -> 2D profiled model."""

    if energy is None:
        energy = np.linspace(83, 87, 61)
    if time is None:
        time = np.linspace(-10, 100, 31)
    if aux_axis is None:
        aux_axis = np.linspace(0, 4, 5)
    if model_yaml is None:
        model_yaml = _FILE_ENERGY_YAML

    project = Project(path="tests")
    file = File(parent_project=project, energy=energy, time=time, aux_axis=aux_axis)
    file.load_model(model_yaml=model_yaml, model_info=model_info)
    model = file.model_active
    assert model is not None

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

    return file, model


#
#
class TestProfileModels:
    """2D models with aux-axis-varying parameters (profiles)."""

    #
    def test_profiled_amplitude(self):
        """Single profiled parameter on a Gauss peak with dynamics."""

        _file, model = _make_2d_profile_model(
            ["single_gauss"],
            [("Gauss_01_x0", ["MonoExpPos"])],
            [("Gauss_01_A", ["profile_pExpDecay"])],
        )
        graph = build_graph(model)
        assert can_lower_2d(graph)
        plan = schedule_2d(graph)
        theta = _compare_evaluator_vs_interpreter(model, plan)

        A_idx = plan.opt_param_names.index("Gauss_01_A_pExpDecay_01_A")
        tau_idx = plan.opt_param_names.index("Gauss_01_A_pExpDecay_01_tau")
        _perturb_theta(plan, model, theta, [A_idx, tau_idx], [1.5, -0.15])

    #
    def test_profiled_amplitude_no_dynamics(self):
        """Profile on a 2D model where no param has dynamics (all static over time)."""

        _file, model = _make_2d_profile_model(
            ["single_gauss"],
            [],
            [("Gauss_01_A", ["profile_pExpDecay"])],
        )
        graph = build_graph(model)
        assert can_lower_2d(graph)
        plan = schedule_2d(graph)
        _compare_evaluator_vs_interpreter(model, plan)

    #
    def test_two_profiles_same_component(self):
        """Multiple profiled params (x0 and A) on same component in 2D."""

        _file, model = _make_2d_profile_model(
            ["single_gauss"],
            [("Gauss_01_SD", ["MonoExpPos"])],
            [
                ("Gauss_01_x0", ["roundtrip_pLinear_x0"]),
                ("Gauss_01_A", ["roundtrip_pExpDecay_A"]),
            ],
        )
        graph = build_graph(model)
        assert can_lower_2d(graph)
        plan = schedule_2d(graph)
        theta = _compare_evaluator_vs_interpreter(model, plan)

        amp_idx = plan.opt_param_names.index("Gauss_01_A_pExpDecay_01_A")
        slope_idx = plan.opt_param_names.index("Gauss_01_x0_pLinear_01_m")
        _perturb_theta(plan, model, theta, [amp_idx, slope_idx], [0.75, -0.05])

    #
    def test_profile_dependent_expression(self):
        """Expression referencing profiled param propagates in 2D."""

        _file, model = _make_2d_profile_model(
            ["two_glp_expr_amplitude"],
            [("GLP_01_x0", ["MonoExpPos"])],
            [("GLP_01_A", ["profile_pLinear"])],
        )
        graph = build_graph(model)
        assert can_lower_2d(graph)
        plan = schedule_2d(graph)
        theta = _compare_evaluator_vs_interpreter(model, plan)

        amp_idx = plan.opt_param_names.index("GLP_01_A")
        slope_idx = plan.opt_param_names.index("GLP_01_A_pLinear_01_m")
        _perturb_theta(plan, model, theta, [amp_idx, slope_idx], [1.0, 0.08])

    #
    def test_profile_expr_with_time_dep_ref(self):
        """Per-sample expression referencing both a profiled param and a
        time-dependent param binds the resolved trace, not the stale base.
        """

        _file, model = _make_2d_profile_model(
            ["two_glp_mixed_profile_dynamics"],
            [("GLP_01_x0", ["MonoExpPos"])],
            [("GLP_01_A", ["profile_pLinear"])],
        )
        graph = build_graph(model)
        assert can_lower_2d(graph)
        plan = schedule_2d(graph)
        theta = _compare_evaluator_vs_interpreter(model, plan)

        # Perturb the dynamics param that the expression reads
        x0_dyn_A_idx = plan.opt_param_names.index("GLP_01_x0_expFun_01_A")
        slope_idx = plan.opt_param_names.index("GLP_01_A_pLinear_01_m")
        _perturb_theta(plan, model, theta, [x0_dyn_A_idx, slope_idx], [0.3, 0.05])

    #
    def test_profiled_shirley(self):
        """Profiled spectrum-fed op (Shirley) stays in parity."""

        _file, model = _make_2d_profile_model(
            ["shirley_peak"],
            [("GLP_01_A", ["MonoExpPos"])],
            [("Shirley_pShirley", ["profile_pLinear"])],
            model_yaml=_ENERGY_YAML,
            energy=np.linspace(80, 90, 101),
        )
        model.lmfit_pars["Shirley_pShirley_pLinear_01_m"].value = 1.0e-5

        graph = build_graph(model)
        assert can_lower_2d(graph)
        plan = schedule_2d(graph)
        _compare_evaluator_vs_interpreter(model, plan)

    #
    def test_profile_with_time_dep_profile_params(self):
        """Profile function params themselves have dynamics (the hard case).

        The profile pExpDecay amplitude is time-dependent via expFun,
        so at each time step the profile shape changes.
        Profile must be added first, then dynamics on the profile param.
        """

        project = Project(path="tests")
        file = File(
            parent_project=project,
            energy=np.linspace(83, 87, 61),
            time=np.linspace(-10, 100, 31),
            aux_axis=np.linspace(0, 4, 5),
        )
        file.load_model(
            model_yaml=_FILE_ENERGY_YAML,
            model_info=["single_gauss"],
        )
        # First: attach profile
        file.add_par_profile(
            target_model="single_gauss",
            target_parameter="Gauss_01_A",
            profile_yaml=_PROFILE_YAML,
            profile_model=["profile_pExpDecay"],
        )
        # Then: add dynamics to the profile's own amplitude param
        file.add_time_dependence(
            target_model="single_gauss",
            target_parameter="Gauss_01_A_pExpDecay_01_A",
            dynamics_yaml=_TIME_YAML,
            dynamics_model=["MonoExpPosStrong"],
        )
        model = file.model_active
        assert model is not None  # type guard

        graph = build_graph(model)
        assert can_lower_2d(graph)
        plan = schedule_2d(graph)
        theta = _compare_evaluator_vs_interpreter(model, plan)

        # Perturb the dynamics param that feeds into the profile
        dyn_A_idx = plan.opt_param_names.index("Gauss_01_A_pExpDecay_01_A_expFun_01_A")
        dyn_tau_idx = plan.opt_param_names.index(
            "Gauss_01_A_pExpDecay_01_A_expFun_01_tau"
        )
        _perturb_theta(plan, model, theta, [dyn_A_idx, dyn_tau_idx], [0.5, 0.5])


# ---------------------------------------------------------------------------
# Resolved-trace convolution (IRF)
# ---------------------------------------------------------------------------


#
#
class TestDynamicsConvolution:
    """Lowered IRF dynamics: CONVOLUTION is compiled into a kind=2 step.

    Covers plan encoding (conv program layout, frozen support, resolution
    ordering) and numerical parity against ``Model.create_value_2d()``.
    The IRF path rewrites a resolved trace row in place via
    ``my_conv`` -- the same code MCP calls -- so tolerance matches the
    other OpKind parity tests.
    """

    #
    def _make_irf_plan(self):
        file, model = _make_2d_model(
            ["glp_only"],
            [("GLP_01_A", ["MonoExpPosIRF"])],
        )
        graph = build_graph(model)
        assert can_lower_2d(graph)
        plan = schedule_2d(graph)
        return plan, graph, model

    #
    def test_irf_parity_at_initial_theta(self):
        """evaluate_2d == interpreter at the initial theta for an IRF fit."""

        plan, _graph, model = self._make_irf_plan()
        _compare_evaluator_vs_interpreter(model, plan)

    #
    def test_irf_parity_under_perturbation(self):
        """evaluate_2d == interpreter after perturbing kernel / dynamics params."""

        plan, _graph, model = self._make_irf_plan()
        theta = _compare_evaluator_vs_interpreter(model, plan)
        sd_idx = plan.opt_param_names.index("GLP_01_A_gaussCONV_SD")
        tau_idx = plan.opt_param_names.index("GLP_01_A_expFun_01_tau")
        _perturb_theta(plan, model, theta, [sd_idx, tau_idx], [0.03, 0.5])

    #
    def test_conv_program_populated(self):
        """IRF plan emits exactly one conv step with well-formed CSR."""

        plan, _graph, _model = self._make_irf_plan()

        assert plan.n_conv_steps == 1
        assert plan.conv_target_rows.shape == (1,)
        assert plan.conv_func_ids.shape == (1,)
        assert plan.conv_param_indptr.shape == (2,)
        assert plan.conv_param_indptr[0] == 0
        n_kernel_params = int(plan.conv_param_indptr[1])
        assert n_kernel_params >= 1  # at least SD
        assert plan.conv_param_rows.shape == (n_kernel_params,)
        assert plan.conv_support_indptr.shape == (2,)
        assert plan.conv_support_indptr[0] == 0
        n_support = int(plan.conv_support_indptr[1])
        assert plan.conv_support_values.shape == (n_support,)

    #
    def test_conv_target_row_valid(self):
        """Conv target row is a valid trace-matrix index."""

        plan, _graph, _model = self._make_irf_plan()
        target = int(plan.conv_target_rows[0])
        assert 0 <= target < plan.n_params

    #
    def test_conv_param_rows_valid(self):
        """Every kernel-param row is a valid trace-matrix index."""

        plan, _graph, _model = self._make_irf_plan()
        for row in plan.conv_param_rows:
            assert 0 <= int(row) < plan.n_params

    #
    def test_conv_support_frozen_from_kernel_time(self):
        """Plan's conv support matches the CONVOLUTION node's kernel_time array."""

        plan, graph, _model = self._make_irf_plan()
        from trspecfit.graph_ir import NodeKind

        conv_nodes = [n for n in graph.nodes if n.kind == NodeKind.CONVOLUTION]
        assert len(conv_nodes) == 1
        expected = conv_nodes[0].arrays["kernel_time"]
        start = int(plan.conv_support_indptr[0])
        end = int(plan.conv_support_indptr[1])
        np.testing.assert_array_equal(plan.conv_support_values[start:end], expected)

    #
    def test_conv_step_runs_after_dynamics(self):
        """kind=2 conv step appears after the kind=0 dyn_group for its target."""

        plan, _graph, _model = self._make_irf_plan()
        kinds = plan.resolution_kinds.tolist()
        conv_positions = [i for i, k in enumerate(kinds) if k == 2]
        assert len(conv_positions) == 1
        conv_pos = conv_positions[0]
        conv_idx = int(plan.resolution_indices[conv_pos])
        target_row = int(plan.conv_target_rows[conv_idx])

        dyn_before = False
        for i in range(conv_pos):
            if kinds[i] == 0:
                dyn_idx = int(plan.resolution_indices[i])
                if int(plan.dyn_group_target_row[dyn_idx]) == target_row:
                    dyn_before = True
                    break
        assert dyn_before, (
            "kind=2 conv step must follow the kind=0 dyn_group "
            "that populates its target row"
        )

    #
    def test_non_irf_has_empty_conv_program(self):
        """Regression guard: non-IRF 2D models still emit an empty conv program."""

        file, model = _make_2d_model(
            ["glp_only"],
            [("GLP_01_A", ["MonoExpPos"])],
        )
        graph = build_graph(model)
        plan = schedule_2d(graph)

        assert plan.n_conv_steps == 0
        assert plan.conv_target_rows.shape == (0,)
        assert plan.conv_func_ids.shape == (0,)
        assert plan.conv_param_rows.shape == (0,)
        assert plan.conv_support_values.shape == (0,)
        assert 2 not in plan.resolution_kinds.tolist()


# ---------------------------------------------------------------------------
# Subcycle dynamics
# ---------------------------------------------------------------------------


#
#
class TestSubcycleDynamics:
    """Multi-cycle dynamics: SUBCYCLE_REMAP/MASK nodes are compiled into
    per-substep ``dyn_sub_time_axes`` / ``dyn_sub_masks`` schedule arrays.

    The evaluator applies them as
    ``func(dyn_sub_time_axes[s], *pars) * dyn_sub_masks[s]``, which must
    match MCP's ``fct(time_norm, ...) * time_n_sub``.
    """

    #
    def _make_two_subcycle_plan(self):
        file, model = _make_2d_model(
            ["glp_only"],
            [("GLP_01_A", ["ModelNone", "MonoExpNeg"])],
            frequency=10,
        )
        graph = build_graph(model)
        assert can_lower_2d(graph)
        plan = schedule_2d(graph)
        return plan, graph, model

    #
    def _make_three_subcycle_plan(self):
        """Three-subcycle model with cross-subcycle expression refs.

        ``MonoExpPosExpr`` references ``MonoExpNeg``'s params via
        expressions (``"-expFun_01_A"``, ``"expFun_01_tau"``), so this
        exercises the expression/subcycle interaction.
        """

        file, model = _make_2d_model(
            ["glp_only"],
            [("GLP_01_A", ["ModelNone", "MonoExpNeg", "MonoExpPosExpr"])],
            frequency=10,
        )
        graph = build_graph(model)
        assert can_lower_2d(graph)
        plan = schedule_2d(graph)
        return plan, graph, model

    #
    def test_two_subcycle_parity_at_initial_theta(self):
        """evaluate_2d == interpreter at initial theta for two-subcycle fit."""

        plan, _graph, model = self._make_two_subcycle_plan()
        _compare_evaluator_vs_interpreter(model, plan)

    #
    def test_two_subcycle_parity_under_perturbation(self):
        """Perturb subcycle-1 tau / amplitude and check parity holds."""

        plan, _graph, model = self._make_two_subcycle_plan()
        theta = _compare_evaluator_vs_interpreter(model, plan)
        A_idx = plan.opt_param_names.index("GLP_01_A_expFun_01_A")
        tau_idx = plan.opt_param_names.index("GLP_01_A_expFun_01_tau")
        _perturb_theta(plan, model, theta, [A_idx, tau_idx], [-0.3, 0.8])

    #
    def test_three_subcycle_with_cross_expression_parity(self):
        """Cross-subcycle expression refs (MonoExpPosExpr) match MCP."""

        plan, _graph, model = self._make_three_subcycle_plan()
        _compare_evaluator_vs_interpreter(model, plan)

    #
    def test_three_subcycle_cross_expression_under_perturbation(self):
        """Perturb subcycle-1 params; subcycle-2 expression deps must follow."""

        plan, _graph, model = self._make_three_subcycle_plan()
        theta = _compare_evaluator_vs_interpreter(model, plan)
        A_idx = plan.opt_param_names.index("GLP_01_A_expFun_01_A")
        tau_idx = plan.opt_param_names.index("GLP_01_A_expFun_01_tau")
        _perturb_theta(plan, model, theta, [A_idx, tau_idx], [-0.2, 1.5])

    #
    def test_subcycle_plan_has_nontrivial_axes_and_masks(self):
        """Plan's per-substep arrays carry non-default values where expected."""

        plan, _graph, _model = self._make_two_subcycle_plan()
        assert plan.dyn_sub_time_axes.shape == (
            plan.dyn_sub_func_id.shape[0],
            plan.n_time,
        )
        assert plan.dyn_sub_masks.shape == plan.dyn_sub_time_axes.shape
        # At least one substep must use a non-default time axis (time_norm
        # differs from plan.time) and a non-default mask (not all-ones).
        any_norm = any(
            not np.array_equal(plan.dyn_sub_time_axes[s], plan.time)
            for s in range(plan.dyn_sub_time_axes.shape[0])
        )
        any_mask = any(
            not np.all(plan.dyn_sub_masks[s] == 1.0)
            for s in range(plan.dyn_sub_masks.shape[0])
        )
        assert any_norm, "subcycle substep should use time_norm, not plan.time"
        assert any_mask, "subcycle substep should have a <1.0 mask entry"
        # Mask values are binary (0 or 1).
        assert set(np.unique(plan.dyn_sub_masks).tolist()).issubset({0.0, 1.0})

    #
    def test_non_subcycle_plan_has_default_axes_and_ones_mask(self):
        """Regression guard: single-cycle dynamics keep defaults."""

        file, model = _make_2d_model(
            ["glp_only"],
            [("GLP_01_A", ["MonoExpPos"])],
        )
        graph = build_graph(model)
        plan = schedule_2d(graph)
        for s in range(plan.dyn_sub_time_axes.shape[0]):
            np.testing.assert_array_equal(plan.dyn_sub_time_axes[s], plan.time)
            np.testing.assert_array_equal(plan.dyn_sub_masks[s], np.ones(plan.n_time))

    #
    def test_mixed_irf_and_subcycle_parity(self):
        """Global IRF on one par + subcycle dynamics on another: both lower.

        Confirms that a resolved-trace IRF (``subcycle=0`` convolution
        path) coexists with subcycle-aware dynamics in the same model.
        """

        project = Project(path="tests")
        file = File(parent_project=project)
        file.energy = np.linspace(80, 90, 101)
        file.time = np.linspace(-10, 100, 51)
        file.load_model(model_yaml=_ENERGY_YAML, model_info=["offset_only"])
        model = file.model_active
        assert model is not None

        file.add_time_dependence(
            target_model="offset_only",
            target_parameter="GLP_01_A",
            dynamics_yaml=_TIME_YAML,
            dynamics_model=["ModelNone", "MonoExpNeg"],
            frequency=10,
        )
        file.add_time_dependence(
            target_model="offset_only",
            target_parameter="Offset_y0",
            dynamics_yaml=_TIME_YAML,
            dynamics_model=["MonoExpPosIRF"],
        )

        graph = build_graph(model)
        assert can_lower_2d(graph)
        plan = schedule_2d(graph)
        # Resolved-trace IRF: exactly one conv step.  Subcycle substeps:
        # at least one mask row < 1.0 somewhere.
        assert plan.n_conv_steps == 1
        assert any(
            not np.all(plan.dyn_sub_masks[s] == 1.0)
            for s in range(plan.dyn_sub_masks.shape[0])
        )
        _compare_evaluator_vs_interpreter(model, plan)
