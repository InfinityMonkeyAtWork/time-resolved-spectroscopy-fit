"""Numerical validation: evaluate_1d vs interpreter for every OpKind.

For each model we:
1. Build the OOP model (load YAML, energy-only).
2. Compile with build_graph / schedule_1d.
3. Extract theta from model.lmfit_pars in plan.opt_param_names order.
4. Compare evaluate_1d(plan, theta) against model.create_value_1d().
5. Perturb theta and repeat to catch ordering bugs.
"""

import numpy as np
from _utils import make_project

from trspecfit import File
from trspecfit.eval_1d import evaluate_1d
from trspecfit.graph_ir import build_graph, can_lower_1d, schedule_1d

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENERGY_YAML = "models/eval_2d_energy.yaml"
_FILE_ENERGY_YAML = "models/file_energy.yaml"
_PROFILE_YAML = "models/file_profile.yaml"


#
def _make_energy_model(model_info):
    """Load a 1D energy model from eval_2d_energy.yaml."""

    project = make_project()
    file = File(parent_project=project, energy=np.linspace(80, 90, 101))
    file.load_model(model_yaml=_ENERGY_YAML, model_info=model_info)
    model = file.model_active
    assert model is not None
    return file, model


#
def _make_profile_energy_model(
    model_info,
    profiles,
    *,
    energy=None,
    aux_axis=None,
    model_yaml=None,
):
    """Load a 1D energy model and attach one or more parameter profiles."""

    if energy is None:
        energy = np.linspace(83, 87, 121)
    if aux_axis is None:
        aux_axis = np.linspace(0, 4, 5)
    if model_yaml is None:
        model_yaml = _FILE_ENERGY_YAML

    project = make_project()
    file = File(parent_project=project, energy=energy, aux_axis=aux_axis)
    file.load_model(model_yaml=model_yaml, model_info=model_info)
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
def _extract_theta(plan, model):
    """Extract theta vector from model.lmfit_pars in plan order."""

    return np.array(
        [model.lmfit_pars[name].value for name in plan.opt_param_names],
        dtype=np.float64,
    )


#
def _compare_evaluator_vs_interpreter(model, plan, *, rtol=1e-10):
    """Run evaluate_1d and interpreter, assert they match."""

    theta = _extract_theta(plan, model)
    fast = evaluate_1d(plan, theta)
    model.create_value_1d()
    slow = model.value_1d
    np.testing.assert_allclose(fast, slow, rtol=rtol, atol=1e-10)
    return theta


#
def _perturb_theta(plan, model, theta, indices, deltas):
    """Perturb theta at given indices, update model, and compare."""

    theta_new = theta.copy()
    for idx, delta in zip(indices, deltas, strict=True):
        theta_new[idx] += delta
        name = plan.opt_param_names[idx]
        model.lmfit_pars[name].value = float(theta_new[idx])

    fast = evaluate_1d(plan, theta_new)
    model.create_value_1d()
    slow = model.value_1d
    np.testing.assert_allclose(fast, slow, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# Static 1D models (one per OpKind)
# ---------------------------------------------------------------------------


#
#
class TestStaticModels:
    """Each OpKind as a 1D energy-only model."""

    def _run_static(self, model_info):
        """Shared: load, compile, compare at initial + perturbed theta."""

        file, model = _make_energy_model(model_info)
        graph = build_graph(model)
        assert can_lower_1d(graph)
        plan = schedule_1d(graph)
        theta = _compare_evaluator_vs_interpreter(model, plan)

        # Perturb with 1% relative change to stay within lmfit bounds.
        n_opt = len(plan.opt_param_names)
        if n_opt >= 2:
            d0 = 0.01 * abs(theta[0]) if theta[0] != 0 else 0.01
            d1 = -0.01 * abs(theta[1]) if theta[1] != 0 else -0.01
            _perturb_theta(plan, model, theta, [0, 1], [d0, d1])

    #
    def test_gauss(self):
        self._run_static(["gauss_only"])

    #
    def test_gauss_asym(self):
        self._run_static(["gauss_asym_only"])

    #
    def test_lorentz(self):
        self._run_static(["lorentz_only"])

    #
    def test_voigt(self):
        self._run_static(["voigt_only"])

    #
    def test_gls(self):
        self._run_static(["gls_only"])

    #
    def test_glp(self):
        self._run_static(["glp_only"])

    #
    def test_ds(self):
        self._run_static(["ds_only"])

    #
    def test_offset_peak(self):
        self._run_static(["offset_only"])

    #
    def test_linback_peak(self):
        self._run_static(["linback_peak"])

    #
    def test_shirley_peak(self):
        self._run_static(["shirley_peak"])


# ---------------------------------------------------------------------------
# Static caching
# ---------------------------------------------------------------------------


#
#
class TestCaching:
    """Models with constant components that should be pre-cached."""

    #
    def test_cached_offset(self):
        """Offset with y0=False should be pre-cached."""

        file, model = _make_energy_model(["cached_offset_peak"])
        graph = build_graph(model)
        assert can_lower_1d(graph)
        plan = schedule_1d(graph)

        # Verify caching flag
        has_constant = any(plan.op_is_constant)
        assert has_constant, "Expected at least one constant op"

        _compare_evaluator_vs_interpreter(model, plan)

    #
    def test_cached_shirley(self):
        """Shirley with all-static peak should still work."""

        file, model = _make_energy_model(["cached_shirley_peak"])
        graph = build_graph(model)
        plan = schedule_1d(graph)
        _compare_evaluator_vs_interpreter(model, plan)


# ---------------------------------------------------------------------------
# Expression models
# ---------------------------------------------------------------------------


#
#
class TestExpressionModels:
    """Models with expression-linked parameters."""

    #
    def test_glp_expression(self):
        """Two GLP peaks with A/x0/F/m linked by expressions."""

        file, model = _make_energy_model(["glp_expression"])
        graph = build_graph(model)
        assert can_lower_1d(graph)
        plan = schedule_1d(graph)
        theta = _compare_evaluator_vs_interpreter(model, plan)

        # Perturb primary peak params — expressions should propagate.
        # Use 1% relative change to stay within lmfit bounds.
        if len(theta) >= 2:
            d0 = 0.01 * abs(theta[0]) if theta[0] != 0 else 0.01
            d1 = -0.01 * abs(theta[1]) if theta[1] != 0 else -0.01
            _perturb_theta(plan, model, theta, [0, 1], [d0, d1])

    #
    def test_expression_values_propagate(self):
        """Verify that expression-linked params update correctly."""

        file, model = _make_energy_model(["glp_expression"])
        graph = build_graph(model)
        plan = schedule_1d(graph)

        theta = _extract_theta(plan, model)
        fast = evaluate_1d(plan, theta)

        # Perturb GLP_01_A (stay within bounds [5, 25])
        A_idx = list(plan.opt_param_names).index("GLP_01_A")
        theta_new = theta.copy()
        theta_new[A_idx] = 22.0
        model.lmfit_pars["GLP_01_A"].value = 22.0

        fast_new = evaluate_1d(plan, theta_new)
        model.create_value_1d()
        slow_new = model.value_1d
        np.testing.assert_allclose(fast_new, slow_new, rtol=1e-10, atol=1e-10)

        # Spectra should differ after perturbation
        assert not np.allclose(fast, fast_new)


# ---------------------------------------------------------------------------
# Profile models
# ---------------------------------------------------------------------------


#
#
class TestProfileModels:
    """Models with aux-axis-varying parameters."""

    #
    def test_profiled_amplitude(self):
        """Single profiled parameter matches interpreter at baseline + perturbation."""

        _file, model = _make_profile_energy_model(
            ["single_gauss"],
            [("Gauss_01_A", ["profile_pExpDecay"])],
        )
        graph = build_graph(model)
        assert can_lower_1d(graph)
        plan = schedule_1d(graph)
        theta = _compare_evaluator_vs_interpreter(model, plan)

        A_idx = list(plan.opt_param_names).index("Gauss_01_A_pExpDecay_01_A")
        tau_idx = list(plan.opt_param_names).index("Gauss_01_A_pExpDecay_01_tau")
        _perturb_theta(plan, model, theta, [A_idx, tau_idx], [1.5, -0.15])

    #
    def test_two_profiles_same_component(self):
        """Multiple profiled params on one component compile and stay in parity."""

        _file, model = _make_profile_energy_model(
            ["single_gauss"],
            [
                ("Gauss_01_x0", ["roundtrip_pLinear_x0"]),
                ("Gauss_01_A", ["roundtrip_pExpDecay_A"]),
            ],
        )
        graph = build_graph(model)
        assert can_lower_1d(graph)
        plan = schedule_1d(graph)
        theta = _compare_evaluator_vs_interpreter(model, plan)

        amp_idx = list(plan.opt_param_names).index("Gauss_01_A_pExpDecay_01_A")
        slope_idx = list(plan.opt_param_names).index("Gauss_01_x0_pLinear_01_m")
        _perturb_theta(plan, model, theta, [amp_idx, slope_idx], [0.75, -0.05])

    #
    def test_profile_dependent_expression(self):
        """Expressions that reference profiled params stay in parity."""

        _file, model = _make_profile_energy_model(
            ["two_glp_expr_amplitude"],
            [("GLP_01_A", ["profile_pLinear"])],
        )
        graph = build_graph(model)
        assert can_lower_1d(graph)
        plan = schedule_1d(graph)
        theta = _compare_evaluator_vs_interpreter(model, plan)

        amp_idx = list(plan.opt_param_names).index("GLP_01_A")
        slope_idx = list(plan.opt_param_names).index("GLP_01_A_pLinear_01_m")
        _perturb_theta(plan, model, theta, [amp_idx, slope_idx], [1.0, 0.08])

    #
    def test_profiled_shirley(self):
        """Profiled spectrum-fed ops stay in parity with the interpreter."""

        _file, model = _make_profile_energy_model(
            ["shirley_peak"],
            [("Shirley_pShirley", ["profile_pLinear"])],
            model_yaml=_ENERGY_YAML,
            energy=np.linspace(80, 90, 101),
        )
        model.lmfit_pars["Shirley_pShirley_pLinear_01_m"].value = 1.0e-5

        graph = build_graph(model)
        assert can_lower_1d(graph)
        plan = schedule_1d(graph)
        _compare_evaluator_vs_interpreter(model, plan)


# ---------------------------------------------------------------------------
# can_lower_1d gate
# ---------------------------------------------------------------------------


#
#
class TestCanLower1D:
    """Verify the can_lower_1d gate."""

    #
    def test_energy_only_model_is_lowerable(self):
        file, model = _make_energy_model(["glp_only"])
        graph = build_graph(model)
        assert can_lower_1d(graph)

    #
    def test_2d_model_is_not_lowerable_1d(self):
        """A model with dynamics should not pass can_lower_1d."""

        project = make_project()
        file = File(
            parent_project=project,
            energy=np.linspace(80, 90, 101),
            time=np.linspace(-10, 100, 51),
        )
        file.load_model(model_yaml=_ENERGY_YAML, model_info=["glp_only"])
        file.add_time_dependence(
            target_model="glp_only",
            target_parameter="GLP_01_A",
            dynamics_yaml="models/file_time.yaml",
            dynamics_model=["MonoExpPos"],
        )
        model = file.model_active
        assert model is not None
        graph = build_graph(model)
        assert not can_lower_1d(graph)

    #
    def test_profile_model_is_lowerable_1d(self):
        """Profile-only 1D models should now pass can_lower_1d."""

        _file, model = _make_profile_energy_model(
            ["single_gauss"],
            [("Gauss_01_A", ["profile_pExpDecay"])],
        )
        graph = build_graph(model)
        assert can_lower_1d(graph)
