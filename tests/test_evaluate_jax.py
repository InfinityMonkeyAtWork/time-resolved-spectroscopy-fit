"""JAX evaluator backend: gate behavior and parity vs the NumPy GIR path.

Parity target is ``evaluate_2d`` (the compiled NumPy evaluator), not the
interpreter — the JAX backend's contract is bit-level agreement with the
reference backend it may eventually replace (float64 mode, tight rtol).

Skips entirely when jax is not installed (optional ``[jax]`` extra).
"""

import numpy as np
import pytest

pytest.importorskip("jax")

from test_evaluate_2d import (  # noqa: E402
    _extract_theta,
    _make_2d_model,
    _make_2d_profile_model,
    _make_energy_model,
)

from trspecfit.eval_2d import evaluate_2d  # noqa: E402
from trspecfit.eval_jax import make_evaluator_2d_jax  # noqa: E402
from trspecfit.graph_ir import (  # noqa: E402
    build_graph,
    can_lower_2d,
    can_lower_jax_2d,
    schedule_2d,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


#
def _make_plan(model_info, dynamics_params, **kwargs):
    """Build model + graph + plan; graph must pass the JAX gate."""

    _file, model = _make_2d_model(model_info, dynamics_params, **kwargs)
    graph = build_graph(model)
    assert can_lower_jax_2d(graph)
    plan = schedule_2d(graph)
    return plan, model


#
def _assert_parity(plan, model, *, rtol=1e-12, atol=1e-12):
    """Compare the jitted JAX evaluator against evaluate_2d.

    Checks the initial theta and a perturbed theta through the same
    compiled evaluator (exercises jit reuse, not just first-trace
    correctness).
    """

    evaluate_jax = make_evaluator_2d_jax(plan)
    theta = _extract_theta(plan, model)

    ref = evaluate_2d(plan, theta)
    got = evaluate_jax(theta)
    np.testing.assert_allclose(got, ref, rtol=rtol, atol=atol)

    theta_new = theta * 1.05 + 0.01
    ref_new = evaluate_2d(plan, theta_new)
    got_new = evaluate_jax(theta_new)
    np.testing.assert_allclose(got_new, ref_new, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Gate behavior
# ---------------------------------------------------------------------------


#
#
class TestJaxGate:
    """can_lower_jax_2d accepts the Phase B slice, rejects the rest."""

    #
    def test_static_model_passes(self):
        _file, model = _make_2d_model(["cached_shirley_peak"], [])
        assert can_lower_jax_2d(build_graph(model))

    #
    def test_dynamics_model_passes(self):
        _file, model = _make_2d_model(["glp_only"], [("GLP_01_A", ["MonoExpPos"])])
        assert can_lower_jax_2d(build_graph(model))

    #
    def test_voigt_rejected_but_numpy_lowerable(self):
        """Voigt fails the JAX gate yet stays on the compiled NumPy path."""

        _file, model = _make_2d_model(["voigt_only"], [])
        graph = build_graph(model)
        assert can_lower_2d(graph)
        assert not can_lower_jax_2d(graph)

    #
    def test_convolution_rejected_but_numpy_lowerable(self):
        _file, model = _make_2d_model(["glp_only"], [("GLP_01_A", ["MonoExpPosIRF"])])
        graph = build_graph(model)
        assert can_lower_2d(graph)
        assert not can_lower_jax_2d(graph)

    #
    def test_subcycle_rejected_but_numpy_lowerable(self):
        _file, model = _make_2d_model(
            ["glp_only"],
            [("GLP_01_A", ["ModelNone", "MonoExpNeg"])],
            frequency=10,
        )
        graph = build_graph(model)
        assert can_lower_2d(graph)
        assert not can_lower_jax_2d(graph)

    #
    def test_profile_rejected_but_numpy_lowerable(self):
        _file, model = _make_2d_profile_model(
            ["single_gauss"],
            [("Gauss_01_x0", ["MonoExpPos"])],
            [("Gauss_01_A", ["profile_pExpDecay"])],
        )
        graph = build_graph(model)
        assert can_lower_2d(graph)
        assert not can_lower_jax_2d(graph)

    #
    def test_1d_graph_rejected(self):
        _file, model = _make_energy_model(["glp_only"])
        assert not can_lower_jax_2d(build_graph(model))


# ---------------------------------------------------------------------------
# Parity: static component ops
# ---------------------------------------------------------------------------


#
#
class TestJaxParityStatic:
    """Every supported OpKind, all-static parameters."""

    _STATIC_MODELS = [
        ["gauss_only"],
        ["gauss_asym_only"],
        ["lorentz_only"],
        ["gls_only"],
        ["glp_only"],
        ["ds_only"],
        ["offset_only"],
        ["linback_peak"],
        ["shirley_peak"],
    ]

    #
    @pytest.mark.parametrize(
        "model_info", _STATIC_MODELS, ids=[m[0] for m in _STATIC_MODELS]
    )
    def test_static_parity(self, model_info):
        plan, model = _make_plan(model_info, [])
        _assert_parity(plan, model)

    #
    def test_constant_component_caching(self):
        """Constant ops are served from the cached arrays, not re-evaluated."""

        plan, model = _make_plan(["cached_shirley_peak"], [])
        _assert_parity(plan, model)


# ---------------------------------------------------------------------------
# Parity: dynamics groups
# ---------------------------------------------------------------------------


#
#
class TestJaxParityDynamics:
    """Every DynFuncKind through a dynamics group on GLP_01_A."""

    _DYN_MODELS = [
        "MonoExpPos",
        "MonoExpNeg",
        "MonoStep",
        "MonoSin",
        "MonoLin",
        "MonoSinDivX",
        "MonoErf",
        "MonoSqrt",
    ]

    #
    @pytest.mark.parametrize("dyn_model", _DYN_MODELS)
    def test_single_dynamics_parity(self, dyn_model):
        plan, model = _make_plan(["glp_only"], [("GLP_01_A", [dyn_model])])
        _assert_parity(plan, model)

    #
    def test_multi_substep_group(self):
        """Bi-exponential: two substeps summed into one dynamics group.

        The second expFun's t0 is an expression (``expFun_01_t0``), so
        this also covers expression-valued dynamics parameters.
        """

        plan, model = _make_plan(["glp_only"], [("GLP_01_A", ["BiExpSharedT0"])])
        _assert_parity(plan, model)

    #
    def test_two_dynamic_parameters(self):
        """Independent dynamics groups on two parameters."""

        plan, model = _make_plan(
            ["glp_only"],
            [("GLP_01_A", ["MonoExpPos"]), ("GLP_01_x0", ["MonoSin"])],
        )
        _assert_parity(plan, model)


# ---------------------------------------------------------------------------
# Parity: arithmetic expressions
# ---------------------------------------------------------------------------


#
#
class TestJaxParityExpressions:
    """Compiled RPN expressions, static and interleaved with dynamics."""

    #
    def test_static_expression(self):
        plan, model = _make_plan(["glp_expression"], [])
        _assert_parity(plan, model)

    #
    def test_expression_reads_resolved_trace(self):
        """Expression referencing a time-dependent (resolved) parameter."""

        plan, model = _make_plan(
            ["glp_expression"],
            [("GLP_01_A", ["MonoExpPos"])],
        )
        _assert_parity(plan, model)


# ---------------------------------------------------------------------------
# Plan-level rejection and theta contract
# ---------------------------------------------------------------------------


#
#
class TestJaxEvaluatorErrors:
    """make_evaluator_2d_jax raises on unsupported plans and bad theta."""

    #
    def test_voigt_plan_raises(self):
        _file, model = _make_2d_model(["voigt_only"], [])
        plan = schedule_2d(build_graph(model))
        with pytest.raises(ValueError, match="VOIGT"):
            make_evaluator_2d_jax(plan)

    #
    def test_convolution_plan_raises(self):
        _file, model = _make_2d_model(["glp_only"], [("GLP_01_A", ["MonoExpPosIRF"])])
        plan = schedule_2d(build_graph(model))
        with pytest.raises(ValueError, match="convolution"):
            make_evaluator_2d_jax(plan)

    #
    def test_profile_plan_raises(self):
        _file, model = _make_2d_profile_model(
            ["single_gauss"],
            [("Gauss_01_x0", ["MonoExpPos"])],
            [("Gauss_01_A", ["profile_pExpDecay"])],
        )
        plan = schedule_2d(build_graph(model))
        with pytest.raises(ValueError, match="profile"):
            make_evaluator_2d_jax(plan)

    #
    def test_theta_length_mismatch_raises(self):
        plan, model = _make_plan(["glp_only"], [])
        evaluate_jax = make_evaluator_2d_jax(plan)
        theta = _extract_theta(plan, model)
        with pytest.raises(ValueError, match="theta shape"):
            evaluate_jax(np.append(theta, 1.0))
