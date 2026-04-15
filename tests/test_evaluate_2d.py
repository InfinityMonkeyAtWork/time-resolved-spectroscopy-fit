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
from trspecfit.graph_ir import OpKind, build_graph, can_lower_2d, schedule_2d

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
def _make_2d_model(model_info, dynamics_params):
    """Load energy model + add dynamics -> 2D model."""

    project = Project(path="tests")
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
