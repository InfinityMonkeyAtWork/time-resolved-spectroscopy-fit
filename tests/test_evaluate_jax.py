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
    """can_lower_jax_2d covers the lowered 2D surface, rejects 1D/MCP-only."""

    _COVERED = [
        (["cached_shirley_peak"], [], {}),
        (["glp_only"], [("GLP_01_A", ["MonoExpPos"])], {}),
        (["voigt_only"], [], {}),
        (["glp_only"], [("GLP_01_A", ["MonoExpPosIRF"])], {}),
        (
            ["glp_only"],
            [("GLP_01_A", ["ModelNone", "MonoExpNeg"])],
            {"frequency": 10},
        ),
    ]

    #
    @pytest.mark.parametrize(
        "model_info,dynamics,kwargs",
        _COVERED,
        ids=["static", "dynamics", "voigt", "convolution", "subcycle"],
    )
    def test_lowered_surface_passes(self, model_info, dynamics, kwargs):
        _file, model = _make_2d_model(model_info, dynamics, **kwargs)
        graph = build_graph(model)
        assert can_lower_2d(graph)
        assert can_lower_jax_2d(graph)

    #
    def test_profile_model_passes(self):
        _file, model = _make_2d_profile_model(
            ["single_gauss"],
            [("Gauss_01_x0", ["MonoExpPos"])],
            [("Gauss_01_A", ["profile_pExpDecay"])],
        )
        graph = build_graph(model)
        assert can_lower_2d(graph)
        assert can_lower_jax_2d(graph)

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
        ["voigt_only"],
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
# Parity: subcycle dynamics
# ---------------------------------------------------------------------------


#
#
class TestJaxParitySubcycles:
    """Subcycle time axes / masks (compiled schedule data) in parity."""

    #
    def test_two_subcycles(self):
        plan, model = _make_plan(
            ["glp_only"],
            [("GLP_01_A", ["ModelNone", "MonoExpNeg"])],
            frequency=10,
        )
        _assert_parity(plan, model)

    #
    def test_three_subcycles_with_expressions(self):
        """Cross-subcycle expression refs (MonoExpPosExpr <- MonoExpNeg)."""

        plan, model = _make_plan(
            ["glp_only"],
            [("GLP_01_A", ["ModelNone", "MonoExpNeg", "MonoExpPosExpr"])],
            frequency=10,
        )
        _assert_parity(plan, model)


# ---------------------------------------------------------------------------
# Parity: profile-varying parameters
# ---------------------------------------------------------------------------


#
def _make_profile_plan(model_info, dynamics_params, profiles, **kwargs):
    _file, model = _make_2d_profile_model(
        model_info, dynamics_params, profiles, **kwargs
    )
    graph = build_graph(model)
    assert can_lower_jax_2d(graph)
    plan = schedule_2d(graph)
    return plan, model


#
#
class TestJaxParityProfiles:
    """Profiled amplitude (linear) and position (nonlinear), exprs, Shirley."""

    #
    def test_profiled_amplitude_with_dynamics(self):
        plan, model = _make_profile_plan(
            ["single_gauss"],
            [("Gauss_01_x0", ["MonoExpPos"])],
            [("Gauss_01_A", ["profile_pExpDecay"])],
        )
        _assert_parity(plan, model)

    #
    def test_two_profiles_same_component(self):
        """Profiled x0 (nonlinear) + profiled A on one component."""

        plan, model = _make_profile_plan(
            ["single_gauss"],
            [("Gauss_01_SD", ["MonoExpPos"])],
            [
                ("Gauss_01_x0", ["roundtrip_pLinear_x0"]),
                ("Gauss_01_A", ["roundtrip_pExpDecay_A"]),
            ],
        )
        _assert_parity(plan, model)

    #
    def test_profile_expr_with_time_dep_ref(self):
        """Per-sample expression reading a profiled and a resolved param."""

        plan, model = _make_profile_plan(
            ["two_glp_mixed_profile_dynamics"],
            [("GLP_01_x0", ["MonoExpPos"])],
            [("GLP_01_A", ["profile_pLinear"])],
        )
        _assert_parity(plan, model)

    #
    def test_profiled_shirley(self):
        """Profiled spectrum-fed op (Shirley background)."""

        plan, model = _make_profile_plan(
            ["shirley_peak"],
            [("GLP_01_A", ["MonoExpPos"])],
            [("Shirley_pShirley", ["profile_pLinear"])],
            model_yaml="models/eval_2d_energy.yaml",
            energy=np.linspace(80, 90, 101),
        )
        model.lmfit_pars["Shirley_pShirley_pLinear_01_m"].value = 1.0e-5
        _assert_parity(plan, model)


# ---------------------------------------------------------------------------
# Parity: resolved-trace convolution (kernel-matrix)
# ---------------------------------------------------------------------------


#
#
class TestJaxParityConvolution:
    """Every lowered conv kernel, plus a chained double IRF."""

    _IRF_MODELS = [
        "MonoExpPosIRF",
        "MonoExpPosExpSymIRF",
        "MonoExpPosExpDecayIRF",
        "MonoExpPosExpRiseIRF",
        "MonoExpPosBoxIRF",
    ]

    #
    @pytest.mark.parametrize("dyn_model", _IRF_MODELS)
    def test_kernel_parity(self, dyn_model):
        plan, model = _make_plan(["glp_only"], [("GLP_01_A", [dyn_model])])
        _assert_parity(plan, model)

    #
    def test_chained_convolution(self):
        """Two CONVOLUTION nodes on one trace, applied in order."""

        plan, model = _make_plan(["glp_only"], [("GLP_01_A", ["MonoExpPosDoubleIRF"])])
        _assert_parity(plan, model)


# ---------------------------------------------------------------------------
# wofz approximation accuracy
# ---------------------------------------------------------------------------


#
#
class TestWofzAccuracy:
    """Weideman wofz matches scipy over the physical Voigt domain."""

    #
    def test_against_scipy(self):
        from scipy.special import wofz as scipy_wofz

        from trspecfit.eval_jax import _wofz

        # z = (dx + i W/2) / (SD sqrt(2)): wide dx range, W/SD from
        # Lorentzian-dominated to Gaussian-dominated
        dx = np.linspace(-200.0, 200.0, 2001)
        for im in [1e-3, 1e-1, 1.0, 10.0, 100.0]:
            z = dx + 1j * im
            got = np.asarray(_wofz(z))
            ref = scipy_wofz(z)
            np.testing.assert_allclose(got.real, ref.real, rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------------------
# Plan-level rejection and theta contract
# ---------------------------------------------------------------------------


#
#
class TestJaxEvaluatorErrors:
    """make_evaluator_2d_jax rejects bad theta shapes."""

    #
    def test_theta_length_mismatch_raises(self):
        plan, model = _make_plan(["glp_only"], [])
        evaluate_jax = make_evaluator_2d_jax(plan)
        theta = _extract_theta(plan, model)
        with pytest.raises(ValueError, match="theta shape"):
            evaluate_jax(np.append(theta, 1.0))
