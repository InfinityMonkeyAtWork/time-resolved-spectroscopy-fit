"""Evaluation-focused MCP tests.

These tests complement parser tests by exercising model evaluation paths
that should catch regressions in dependent/time-dependent parameter handling.
"""

import numpy as np
import pytest

from trspecfit import File, Project
from trspecfit.functions.energy import GLP
from trspecfit.functions.profile import pLinear


#
#
class TestEvaluation:
    """Test model evaluation."""

    #
    def _make_file_with_model(self, model_info):
        """Create project, file, and load model."""

        project = Project(path="tests")
        file = File(parent_project=project)
        file.energy = np.linspace(80, 90, 201)
        file.time = np.linspace(-10, 100, 111)
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=model_info,
        )
        model = file.model_active
        assert model is not None, "Model loading failed in setup"
        return file, model

    #
    def _make_file_with_profile_model(self, model_info):
        """Create project, file with aux_axis, and load model."""

        project = Project(path="tests")
        aux_axis = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        file = File(parent_project=project, aux_axis=aux_axis)
        file.energy = np.linspace(80, 90, 201)
        file.time = np.linspace(-10, 100, 111)
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=model_info,
        )
        model = file.model_active
        assert model is not None, "Model loading failed in setup"
        return file, model

    #
    def _par(self, model, name: str):
        """Return parameter object by full name using the model API."""

        ci, pi = model.find_par_by_name(name)
        assert ci is not None and pi is not None, f"Parameter '{name}' not found"
        return model.components[ci].pars[pi]

    #
    def test_eval_energy_expression_value1d(self):
        """Dependent expression parameters should evaluate correctly in 1D."""

        file, model = self._make_file_with_model(["energy_expression"])

        value_1d = model.create_value1D(return_1d=1)
        assert value_1d is not None
        assert file.energy is not None
        assert value_1d.shape == file.energy.shape
        assert np.isfinite(value_1d).all()

        p_x0_1 = self._par(model, "GLP_01_x0")
        p_x0_2 = self._par(model, "GLP_02_x0")

        v1 = p_x0_1.value(t_ind=0)
        v2 = p_x0_2.value(t_ind=0)
        assert v1 is not None
        assert v2 is not None
        assert np.isclose(v2, v1 + 3.6)

    #
    def test_eval_time_dependent_expression_value2d(self):
        """Expressions depending on time-dependent parameters should evaluate in 2D."""

        file, model = self._make_file_with_model(["energy_expression"])
        file.add_time_dependence(
            model_yaml="test_models_time.yaml",
            model_info=["MonoExpPosIRF"],
            par_name="GLP_01_x0",
        )

        assert model.dim == 2

        p_x0_1 = self._par(model, "GLP_01_x0")
        p_x0_2 = self._par(model, "GLP_02_x0")

        v1_t0 = p_x0_1.value(t_ind=0)
        v2_t0 = p_x0_2.value(t_ind=0)
        assert v1_t0 is not None
        assert v2_t0 is not None
        assert np.isclose(v2_t0, v1_t0 + 3.6)

        model.create_value2D()
        assert model.value2D is not None
        assert file.time is not None
        assert file.energy is not None
        assert model.value2D.shape == (len(file.time), len(file.energy))
        assert np.isfinite(model.value2D).all()

    #
    def test_eval_expression_fan_out(self):
        """Fan-out: GLP_02 and GLP_03 both reference GLP_01 directly."""

        file, model = self._make_file_with_model(["expression_fan_out"])

        # Evaluate 1D spectrum
        value_1d = model.create_value1D(return_1d=1)
        assert value_1d is not None
        assert np.isfinite(value_1d).all()

        # Check expression values: A
        A1 = self._par(model, "GLP_01_A").value(t_ind=0)
        A2 = self._par(model, "GLP_02_A").value(t_ind=0)
        A3 = self._par(model, "GLP_03_A").value(t_ind=0)
        assert np.isclose(A2, A1 * 0.5)
        assert np.isclose(A3, A1 * 0.25)

        # Check expression values: x0
        x01 = self._par(model, "GLP_01_x0").value(t_ind=0)
        x02 = self._par(model, "GLP_02_x0").value(t_ind=0)
        x03 = self._par(model, "GLP_03_x0").value(t_ind=0)
        assert np.isclose(x02, x01 + 2.0)
        assert np.isclose(x03, x01 + 4.0)

    #
    def test_eval_expression_chain(self):
        """Chain: GLP_01 → GLP_02 → GLP_03 (each references the previous)."""

        file, model = self._make_file_with_model(["expression_chain"])

        # Evaluate 1D spectrum
        value_1d = model.create_value1D(return_1d=1)
        assert value_1d is not None
        assert np.isfinite(value_1d).all()

        # Check chain resolves: A
        A1 = self._par(model, "GLP_01_A").value(t_ind=0)
        A2 = self._par(model, "GLP_02_A").value(t_ind=0)
        A3 = self._par(model, "GLP_03_A").value(t_ind=0)
        # GLP_02_A = GLP_01_A * 0.5
        assert np.isclose(A2, A1 * 0.5)
        # GLP_03_A = GLP_02_A * 0.5 = GLP_01_A * 0.25
        assert np.isclose(A3, A2 * 0.5)
        assert np.isclose(A3, A1 * 0.25)

        # Check chain resolves: x0
        x01 = self._par(model, "GLP_01_x0").value(t_ind=0)
        x02 = self._par(model, "GLP_02_x0").value(t_ind=0)
        x03 = self._par(model, "GLP_03_x0").value(t_ind=0)
        # GLP_02_x0 = GLP_01_x0 + 2.0
        assert np.isclose(x02, x01 + 2.0)
        # GLP_03_x0 = GLP_02_x0 + 2.0 = GLP_01_x0 + 4.0
        assert np.isclose(x03, x02 + 2.0)
        assert np.isclose(x03, x01 + 4.0)

    #
    def test_eval_multiple_time_dependent_pars(self):
        """Two pars on the same model with different dynamics models."""

        file, model = self._make_file_with_model(["simple_energy"])

        # Attach MonoExpPosIRF to GLP_01_A
        file.add_time_dependence(
            model_yaml="test_models_time.yaml",
            model_info=["MonoExpPosIRF"],
            par_name="GLP_01_A",
        )
        # Attach MonoExpNeg to GLP_01_x0
        file.add_time_dependence(
            model_yaml="test_models_time.yaml",
            model_info=["MonoExpNeg"],
            par_name="GLP_01_x0",
        )

        assert model.dim == 2

        # Both pars should be time-dependent
        p_A = self._par(model, "GLP_01_A")
        p_x0 = self._par(model, "GLP_01_x0")
        assert p_A.t_vary is True
        assert p_x0.t_vary is True

        # Values should differ: t_ind=0 (t=-10, before t0) vs t_ind=15 (t=5, ~2*tau)
        A_early = p_A.value(t_ind=0)
        A_mid = p_A.value(t_ind=15)
        x0_early = p_x0.value(t_ind=0)
        x0_mid = p_x0.value(t_ind=15)
        assert A_early is not None and A_mid is not None
        assert x0_early is not None and x0_mid is not None
        # MonoExpPos drives A up, MonoExpNeg drives x0 down — both should change
        assert not np.isclose(A_early, A_mid)
        assert not np.isclose(x0_early, x0_mid)

        # 2D evaluation should succeed
        model.create_value2D()
        assert model.value2D is not None
        assert file.time is not None
        assert file.energy is not None
        assert model.value2D.shape == (len(file.time), len(file.energy))
        assert np.isfinite(model.value2D).all()

    #
    def test_eval_profile_averaging(self):
        """Profile averaging should equal GLP at mean effective amplitude.

        GLP(x, A, x0, F, m) = A * shape(x, x0, F, m), so averaging N traces
        with different A_i is the same as one trace at mean(A_i).

        Setup: single GLP (A=20), pLinear profile (m=-0.5, b=0) on A,
        aux_axis = [0, 1, 2, 3, 4].
        Profile values: pLinear([0,1,2,3,4], -0.5, 0) = [0, -0.5, -1, -1.5, -2]
        Effective A: [20, 19.5, 19, 18.5, 18] → mean = 19
        Expected: GLP(energy, 19, 85, 1, 0.3)
        """
        file, model = self._make_file_with_profile_model(["single_glp"])

        # Attach pLinear profile (m=-0.5, b=0) to GLP_01_A
        file.add_par_profile(
            model_yaml="test_models_profile.yaml",
            model_info=["profile_pLinear"],
            par_name="GLP_01_A",
        )

        p_A = self._par(model, "GLP_01_A")
        assert p_A.p_vary is True

        # Evaluate via the model
        value_1d = model.create_value1D(return_1d=1)
        assert value_1d is not None

        # Analytical expectation: GLP at mean effective A
        # profile = pLinear([0,1,2,3,4], m=-0.5, b=0) = [0, -0.5, -1, -1.5, -2]
        # mean(A_eff) = 20 + mean([0, -0.5, -1, -1.5, -2]) = 20 + (-1) = 19
        expected = GLP(file.energy, 19.0, 85.0, 1.0, 0.3)
        np.testing.assert_allclose(value_1d, expected, rtol=1e-10)

    #
    def test_eval_time_dependent_profile_par(self):
        """Profile with time-dependent slope should produce different spectra over time.

        Same linear-in-A trick as test_eval_profile_averaging, but now the
        profile slope m is time-dependent via MonoExpPos dynamics.

        At t < t0 (t_ind=0, t=-10): dynamics = 0, so m = -0.5 (base).
        At t = t0 (t_ind=10, t=0): dynamics kicks in, m changes.

        Since GLP is linear in A, the averaged spectrum at any time equals
        GLP(energy, base_A + mean(profile(t)), x0, F, m_glp).
        We read the profile's value1D at each time step to get the exact
        analytical prediction.
        """
        file, model = self._make_file_with_profile_model(["single_glp"])

        # Attach pLinear profile (m=-0.5, b=0) to GLP_01_A
        file.add_par_profile(
            model_yaml="test_models_profile.yaml",
            model_info=["profile_pLinear"],
            par_name="GLP_01_A",
        )
        # Make the profile slope time-dependent
        file.add_time_dependence(
            model_yaml="test_models_time.yaml",
            model_info=["MonoExpPos"],
            par_name="GLP_01_A_pLinear_01_m",
        )

        assert model.dim == 2
        p_A = self._par(model, "GLP_01_A")
        assert p_A.p_vary is True
        profile = p_A.p_model
        assert profile is not None

        # --- t_ind=0 (t=-10, before t0): dynamics = 0, static profile ---
        spec_early = model.create_value1D(t_ind=0, return_1d=1)
        assert spec_early is not None
        # profile value1D was evaluated at t_ind=0; read it for prediction
        assert profile.value1D is not None
        mean_A_early = 20.0 + np.mean(profile.value1D)
        expected_early = GLP(file.energy, mean_A_early, 85.0, 1.0, 0.3)
        np.testing.assert_allclose(spec_early, expected_early, rtol=1e-10)

        # --- t_ind=10 (t=0, at t0): dynamics nonzero, profile changes ---
        spec_at_t0 = model.create_value1D(t_ind=10, return_1d=1)
        assert spec_at_t0 is not None
        assert profile.value1D is not None
        mean_A_at_t0 = 20.0 + np.mean(profile.value1D)
        expected_at_t0 = GLP(file.energy, mean_A_at_t0, 85.0, 1.0, 0.3)
        np.testing.assert_allclose(spec_at_t0, expected_at_t0, rtol=1e-10)

        # The two spectra should differ (dynamics changed the profile slope)
        assert not np.allclose(spec_early, spec_at_t0)

        # 2D evaluation should succeed
        model.create_value2D()
        assert model.value2D is not None
        assert model.value2D.shape == (len(file.time), len(file.energy))
        assert np.isfinite(model.value2D).all()

    #
    def test_eval_expression_inherits_profile(self):
        """Expression referencing a profiled par should inherit depth variation.

        Setup: two GLPs, GLP_02_A = GLP_01_A * 0.5.
        pLinear profile (m=-0.5, b=0) attached to GLP_01_A.
        aux_axis = [0, 1, 2, 3, 4].

        Profile offsets: pLinear([0,1,2,3,4], -0.5, 0) = [0, -0.5, -1, -1.5, -2]

        At each aux point i:
          A1_eff[i] = 20 + offset[i]
          A2_eff[i] = A1_eff[i] * 0.5   (expression must see profiled value)

        Because GLP is linear in A, averaged spectrum should equal:
          GLP(energy, mean(A1_eff), 85, 1, 0.3)
          + GLP(energy, mean(A2_eff), 87, 1, 0.3)

        If the bug is present, GLP_02 ignores the profile and uses the base
        value (20 * 0.5 = 10) at every aux point, giving the wrong average.
        """
        file, model = self._make_file_with_profile_model(["two_glp_expr_amplitude"])

        # Attach pLinear profile (m=-0.5, b=0) to GLP_01_A
        file.add_par_profile(
            model_yaml="test_models_profile.yaml",
            model_info=["profile_pLinear"],
            par_name="GLP_01_A",
        )

        p_A1 = self._par(model, "GLP_01_A")
        assert p_A1.p_vary is True

        # Evaluate
        value_1d = model.create_value1D(return_1d=1)
        assert value_1d is not None

        # Analytical expectation
        offsets = pLinear(model.aux_axis, -0.5, 0.0)
        A1_eff = 20.0 + offsets  # [20, 19.5, 19, 18.5, 18]
        A2_eff = A1_eff * 0.5  # [10, 9.75, 9.5, 9.25, 9] — expression tracks profile

        expected = GLP(file.energy, np.mean(A1_eff), 85.0, 1.0, 0.3) + GLP(
            file.energy, np.mean(A2_eff), 87.0, 1.0, 0.3
        )
        np.testing.assert_allclose(value_1d, expected, rtol=1e-10)

    #
    def test_eval_expression_chain_profile_raises(self):
        """Transitive chain A3->A2->A1 with profile on A1 must raise.

        expression_chain: GLP_02_A=GLP_01_A*0.5, GLP_03_A=GLP_02_A*0.5
        Adding a profile to GLP_01_A creates a transitive chain through
        GLP_02_A that is not supported — users should reference the base
        p_vary parameter directly.
        """

        file, model = self._make_file_with_profile_model(["expression_chain"])

        with pytest.raises(ValueError, match="indirectly references"):
            file.add_par_profile(
                model_yaml="test_models_profile.yaml",
                model_info=["profile_pLinear"],
                par_name="GLP_01_A",
            )
