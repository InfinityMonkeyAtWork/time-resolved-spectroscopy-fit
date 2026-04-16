"""Numerical validation: evaluate_1d vs interpreter for every OpKind.

For each model we:
1. Build the OOP model (load YAML, energy-only).
2. Compile with build_graph / schedule_1d.
3. Extract theta from model.lmfit_pars in plan.opt_param_names order.
4. Compare evaluate_1d(plan, theta) against model.create_value_1d().
5. Perturb theta and repeat to catch ordering bugs.
"""

import numpy as np

from trspecfit import File, Project
from trspecfit.eval_1d import evaluate_1d
from trspecfit.graph_ir import build_graph, can_lower_1d, schedule_1d

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENERGY_YAML = "models/eval_2d_energy.yaml"


#
def _make_energy_model(model_info):
    """Load a 1D energy model from eval_2d_energy.yaml."""

    project = Project(path="tests")
    file = File(parent_project=project, energy=np.linspace(80, 90, 101))
    file.load_model(model_yaml=_ENERGY_YAML, model_info=model_info)
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

        project = Project(path="tests")
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
    def test_profile_model_is_not_lowerable_1d(self):
        """Profile models should not pass can_lower_1d."""

        project = Project(path="tests")
        file = File(
            parent_project=project,
            energy=np.linspace(80, 90, 101),
            aux_axis=np.array([0.0, 1.0, 2.0]),
        )
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
        assert not can_lower_1d(graph)
