"""JAX evaluator backend: gate, Jacobian, wofz accuracy, and e2e fit.

Broad evaluator parity is NOT here: every evaluator-vs-interpreter
comparison in ``test_evaluate_2d.py`` also asserts JAX parity (via
``_assert_jax_parity``), so the full 2D matrix — including regression
fixtures — covers the JAX backend without duplication.  This module
keeps only JAX-specific coverage: the capability gate, the analytic
Jacobian, the Weideman wofz approximation, the chained-convolution
fixture (absent from the 2D matrix), error contracts, and an
end-to-end ``File.fit_2d`` run on ``spec_fun_str="fit_model_jax"``.

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
from trspecfit.eval_jax import (  # noqa: E402
    make_evaluator_2d_jax,
    make_jacobian_2d_jax,
    make_project_evaluator_2d_jax,
    make_project_jacobian_2d_jax,
)
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
# Profile-model plan helper (used by the Jacobian tests)
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


# ---------------------------------------------------------------------------
# Parity: chained convolution (fixture absent from the 2D matrix)
# ---------------------------------------------------------------------------


#
#
class TestJaxParityChainedConvolution:
    """MonoExpPosDoubleIRF is not exercised by test_evaluate_2d."""

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
            got = np.asarray(_wofz(z), dtype=np.complex128)
            ref = scipy_wofz(z)
            np.testing.assert_allclose(got.real, ref.real, rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------------------
# Analytic Jacobian vs central finite differences of evaluate_2d
# ---------------------------------------------------------------------------


#
def _assert_jacobian_matches_fd(plan, model):
    """Compare make_jacobian_2d_jax against central differences.

    Finite differences run through ``evaluate_2d`` (the NumPy
    reference), so this cross-validates the Jacobian against the other
    backend, not against the JAX evaluator differentiating itself.
    Central differences carry O(h^2) truncation error scaling with the
    derivative magnitude and O(eps_machine/h) cancellation error
    scaling with the *model* magnitude, so the bound uses both scales
    (elementwise rtol on near-zero entries is meaningless).
    """

    jacobian = make_jacobian_2d_jax(plan)
    theta = _extract_theta(plan, model)
    jac = jacobian(theta)
    assert jac.shape == (len(plan.time), len(plan.energy), len(theta))

    f_scale = np.max(np.abs(evaluate_2d(plan, theta)))
    for i in range(len(theta)):
        h = 1e-6 * max(1.0, abs(theta[i]))
        theta_plus = theta.copy()
        theta_plus[i] += h
        theta_minus = theta.copy()
        theta_minus[i] -= h
        fd = (evaluate_2d(plan, theta_plus) - evaluate_2d(plan, theta_minus)) / (2 * h)
        tol = 1e-5 * np.max(np.abs(fd)) + 1e-8 * f_scale
        max_err = np.max(np.abs(jac[:, :, i] - fd))
        assert max_err <= tol, (
            f"Jacobian column {i} ({plan.opt_param_names[i]}): "
            f"max |analytic - fd| = {max_err:.3e} exceeds {tol:.3e}"
        )


#
#
class TestJaxJacobian:
    """Analytic Jacobian across the lowered feature surface."""

    #
    def test_dynamics_and_expression(self):
        plan, model = _make_plan(["glp_expression"], [("GLP_01_A", ["MonoExpPos"])])
        _assert_jacobian_matches_fd(plan, model)

    #
    def test_convolution(self):
        plan, model = _make_plan(["glp_only"], [("GLP_01_A", ["MonoExpPosIRF"])])
        _assert_jacobian_matches_fd(plan, model)

    #
    def test_subcycles(self):
        plan, model = _make_plan(
            ["glp_only"],
            [("GLP_01_A", ["ModelNone", "MonoExpNeg"])],
            frequency=10,
        )
        _assert_jacobian_matches_fd(plan, model)

    #
    def test_profiled_parameter(self):
        plan, model = _make_profile_plan(
            ["single_gauss"],
            [("Gauss_01_x0", ["MonoExpPos"])],
            [("Gauss_01_A", ["profile_pExpDecay"])],
        )
        _assert_jacobian_matches_fd(plan, model)

    #
    def test_voigt(self):
        """Differentiates through the complex Weideman wofz."""

        plan, model = _make_plan(["voigt_only"], [])
        _assert_jacobian_matches_fd(plan, model)

    #
    def test_sqrt_with_varying_onset(self):
        """Regression: sqrt(clip) gave NaN derivatives pre-onset when
        t0 varies (inf * 0 through the clip chain rule), poisoning the
        whole Jacobian and failing leastsq with lmfit's non-finite
        error."""

        plan, model = _make_plan(["glp_only"], [("GLP_01_A", ["MonoSqrtVaryT0"])])
        _assert_parity(plan, model)
        jacobian = make_jacobian_2d_jax(plan)
        jac = jacobian(_extract_theta(plan, model))
        assert np.all(np.isfinite(jac))
        _assert_jacobian_matches_fd(plan, model)


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


# ---------------------------------------------------------------------------
# End-to-end: File.fit_2d on the JAX backend with analytic Jacobian
# ---------------------------------------------------------------------------


#
#
class TestJaxFit2D:
    """spec_fun_str='fit_model_jax' fits through the public API."""

    #
    @pytest.mark.slow
    def test_fit_recovers_truth_with_analytic_jacobian(self):
        """Two-stage fit (Nelder + leastsq/Dfun) recovers clean-data truth."""

        from _utils import extract_truth_pars, make_project, simulate_clean

        from trspecfit import File

        project = make_project(name="jax_e2e", spec_fun_str="fit_model_jax")

        energy = np.linspace(83, 87, 30)
        time = np.linspace(-2, 10, 24)
        truth_file = File(parent_project=project, name="truth")
        truth_file.energy = energy
        truth_file.time = time
        truth_file.dim = 2
        truth_file.load_model(
            model_yaml="models/file_energy.yaml", model_info="single_glp"
        )
        truth_file.add_time_dependence(
            target_model="single_glp",
            target_parameter="GLP_01_A",
            dynamics_yaml="models/file_time.yaml",
            dynamics_model=["MonoExpPos"],
        )
        truth_pars = extract_truth_pars(truth_file.model_active)
        clean = simulate_clean(truth_file.model_active)

        fit_file = File(
            parent_project=project,
            name="fit",
            data=clean,
            energy=energy.copy(),
            time=time.copy(),
        )
        fit_file.load_model(
            model_yaml="models/file_energy.yaml", model_info="single_glp"
        )
        fit_file.define_baseline(
            time_start=0, time_stop=3, time_type="ind", show_plot=False
        )
        fit_file.fit_baseline(model_name="single_glp", stages=2, try_ci=0)
        fit_file.add_time_dependence(
            target_model="single_glp",
            target_parameter="GLP_01_A",
            dynamics_yaml="models/file_time.yaml",
            dynamics_model=["MonoExpPos"],
        )
        fit_file.fit_2d(model_name="single_glp", stages=2, try_ci=0)

        assert fit_file.model_2d is not None  # type guard
        result_params = fit_file.model_2d.result[1].params
        for name, true_val in truth_pars.items():
            fit_val = result_params[name].value
            assert np.isclose(true_val, fit_val, rtol=1e-8, atol=1e-10), (
                f"{name}: true={true_val:.6f}, fit={fit_val:.6f}"
            )


# ---------------------------------------------------------------------------
# Project-level fused evaluator / Jacobian
# ---------------------------------------------------------------------------


#
def _make_project_pair():
    """Two identical-structure plans sharing theta_c slot 0.

    Mimics a 2-file project fit: opt param 0 is project-vary (one
    combined slot feeding both plans), the rest are file-vary. File 0
    gets a proper fit window, file 1 is unwindowed.
    """

    plan_a, model_a = _make_plan(["glp_expression"], [("GLP_01_A", ["MonoExpPos"])])
    plan_b, model_b = _make_plan(["glp_expression"], [("GLP_01_A", ["MonoExpPos"])])
    theta_a = _extract_theta(plan_a, model_a)
    theta_b = theta_a * 1.1 + 0.02
    n = len(theta_a)
    theta_c = np.concatenate([theta_a, theta_b[1:]])
    gather_a = np.arange(n, dtype=np.intp)
    gather_b = np.concatenate([[0], np.arange(n, 2 * n - 1)]).astype(np.intp)
    windows = [
        (slice(2, len(plan_a.time) - 3), slice(5, len(plan_a.energy) - 7)),
        (slice(None), slice(None)),
    ]
    return [plan_a, plan_b], [gather_a, gather_b], windows, theta_c


#
def _fused_reference(plans, gathers, windows, theta_c):
    """NumPy reference: per-plan evaluate_2d, windowed, flattened, concat."""

    pieces = [
        evaluate_2d(plan, theta_c[gather])[window].ravel()
        for plan, gather, window in zip(plans, gathers, windows, strict=True)
    ]
    return np.concatenate(pieces)


#
#
class TestProjectFused:
    """Fused multi-file evaluator and joint Jacobian."""

    #
    def test_evaluator_parity(self):
        """Fused output matches per-file NumPy reference (incl. jit reuse)."""

        plans, gathers, windows, theta_c = _make_project_pair()
        fused = make_project_evaluator_2d_jax(
            plans, plan_gathers=gathers, windows=windows, n_theta=len(theta_c)
        )

        ref = _fused_reference(plans, gathers, windows, theta_c)
        got = fused(theta_c)
        assert got.ndim == 1
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)

        theta_new = theta_c * 1.05 + 0.01
        ref_new = _fused_reference(plans, gathers, windows, theta_new)
        np.testing.assert_allclose(fused(theta_new), ref_new, rtol=1e-12, atol=1e-12)

    #
    def test_jacobian_matches_fd(self):
        """Joint Jacobian vs central differences of the NumPy reference."""

        plans, gathers, windows, theta_c = _make_project_pair()
        jacobian = make_project_jacobian_2d_jax(
            plans, plan_gathers=gathers, windows=windows, n_theta=len(theta_c)
        )
        jac = jacobian(theta_c)
        n_res = len(_fused_reference(plans, gathers, windows, theta_c))
        assert jac.shape == (n_res, len(theta_c))

        f_scale = np.max(np.abs(_fused_reference(plans, gathers, windows, theta_c)))
        for i in range(len(theta_c)):
            h = 1e-6 * max(1.0, abs(theta_c[i]))
            theta_plus = theta_c.copy()
            theta_plus[i] += h
            theta_minus = theta_c.copy()
            theta_minus[i] -= h
            fd = (
                _fused_reference(plans, gathers, windows, theta_plus)
                - _fused_reference(plans, gathers, windows, theta_minus)
            ) / (2 * h)
            tol = 1e-5 * np.max(np.abs(fd)) + 1e-8 * f_scale
            max_err = np.max(np.abs(jac[:, i] - fd))
            assert max_err <= tol, (
                f"Joint Jacobian column {i}: "
                f"max |analytic - fd| = {max_err:.3e} exceeds {tol:.3e}"
            )

    #
    def test_shared_column_spans_both_files(self):
        """The shared slot's column is nonzero in both files' row blocks."""

        plans, gathers, windows, theta_c = _make_project_pair()
        jacobian = make_project_jacobian_2d_jax(
            plans, plan_gathers=gathers, windows=windows, n_theta=len(theta_c)
        )
        jac = jacobian(theta_c)

        n_rows_a = evaluate_2d(plans[0], theta_c[gathers[0]])[windows[0]].size
        shared_col = jac[:, 0]
        assert np.any(shared_col[:n_rows_a] != 0.0)
        assert np.any(shared_col[n_rows_a:] != 0.0)
        # file-vary slots touch only their own file's rows
        n_opt = len(gathers[0])
        file_a_col = jac[:, 1]  # slot 1: file A only
        file_b_col = jac[:, n_opt]  # first file-B-only slot
        assert np.all(file_a_col[n_rows_a:] == 0.0)
        assert np.all(file_b_col[:n_rows_a] == 0.0)

    #
    def test_gather_length_mismatch_raises(self):
        plans, gathers, windows, theta_c = _make_project_pair()
        bad = [gathers[0][:-1], gathers[1]]
        with pytest.raises(ValueError, match="expected"):
            make_project_evaluator_2d_jax(
                plans, plan_gathers=bad, windows=windows, n_theta=len(theta_c)
            )

    #
    def test_gather_out_of_range_raises(self):
        plans, gathers, windows, theta_c = _make_project_pair()
        bad = [gathers[0], gathers[1].copy()]
        bad[1][0] = len(theta_c)
        with pytest.raises(ValueError, match="outside"):
            make_project_evaluator_2d_jax(
                plans, plan_gathers=bad, windows=windows, n_theta=len(theta_c)
            )

    #
    def test_theta_length_mismatch_raises(self):
        plans, gathers, windows, theta_c = _make_project_pair()
        fused = make_project_evaluator_2d_jax(
            plans, plan_gathers=gathers, windows=windows, n_theta=len(theta_c)
        )
        with pytest.raises(ValueError, match="does not match"):
            fused(theta_c[:-1])
