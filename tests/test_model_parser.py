"""
Test parsing of models from YAML files passed by user
"""

import numpy as np
import pytest

# local imports
from trspecfit import File, Project
from trspecfit.utils.parsing import ModelValidationError


#
#
class TestEnergyParsing:
    """Test parsing of 1D energy models."""

    #
    def setUp(self, model_info):
        """Setup function to create project, file, and load model"""
        project = Project(path="tests")
        file = File(parent_project=project)
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=model_info,
            debug=False,
        )
        model = file.model_active
        assert model is not None, "Model loading failed in setup"
        return model

    #
    def test_simple_energy_model(self):
        """
        Test simple energy model with unbound parameter [value, vary] and
        standard parameter [value, vary, min, max] formats
        """

        # import the model
        model = self.setUp(["simple_energy"])

        # check the model
        assert model.name == "simple_energy"
        assert model.dim == 1
        assert len(model.components) == 4

        # check the components
        # Offset (should not be numbered)
        assert model.components[0].fct_str == "Offset"
        assert model.components[0].comp_name == "Offset"
        assert model.components[0].par_dict["y0"] == [2, True, 0, 5]

        # Shirley (should not be numbered) - now uses [value, vary] format
        assert model.components[1].fct_str == "Shirley"
        assert model.components[1].comp_name == "Shirley"
        assert model.components[1].par_dict["pShirley"] == [400, False]
        # Check that lmfit parameter was created with unbounded min/max
        shirley_par = model.lmfit_pars["Shirley_pShirley"]
        assert shirley_par.value == 400
        assert not shirley_par.vary
        assert shirley_par.min == -np.inf
        assert shirley_par.max == np.inf

        # GLP_01 (should be numbered)
        assert model.components[2].fct_str == "GLP"
        assert model.components[2].comp_name == "GLP_01"
        assert model.components[2].N == 1
        assert model.components[2].par_dict["A"] == [20, True, 5, 25]
        assert model.components[2].par_dict["x0"] == [84.5, True, 82, 88]
        assert model.components[2].par_dict["F"] == [1.0, True, 0.75, 2.5]
        assert model.components[2].par_dict["m"] == [0.3, True, 0, 1]

        # GLP_02 (should be numbered) - x0 now uses [value, vary] format
        assert model.components[3].fct_str == "GLP"
        assert model.components[3].comp_name == "GLP_02"
        assert model.components[3].N == 2
        assert model.components[3].par_dict["A"] == [17, True, 5, 25]
        assert model.components[3].par_dict["x0"] == [88.1, True]
        # Check that lmfit parameter was created with unbounded min/max
        x0_par = model.lmfit_pars["GLP_02_x0"]
        assert x0_par.value == 88.1
        assert x0_par.vary
        assert x0_par.min == -np.inf
        assert x0_par.max == np.inf
        assert model.components[3].par_dict["F"] == [1.0, True, 0.75, 2.5]
        assert model.components[3].par_dict["m"] == [0.3, True, 0, 1]

    #
    def test_energy_expression_model(self):
        """Test energy parameters with expressions"""

        # import the model
        model = self.setUp(["energy_expression"])

        # check the model
        assert model.name == "energy_expression"
        assert model.dim == 1
        assert len(model.components) == 4

        # check components
        # Offset
        assert model.components[0].fct_str == "Offset"
        assert model.components[0].comp_name == "Offset"
        assert "y0" in model.components[0].par_dict

        # Shirley
        assert model.components[1].fct_str == "Shirley"
        assert model.components[1].comp_name == "Shirley"
        assert model.components[1].par_dict["pShirley"] == [
            400,
            True,
            1.0e-6,
            1.0e3,
        ]

        # GLP_01
        assert model.components[2].fct_str == "GLP"
        assert model.components[2].comp_name == "GLP_01"
        assert model.components[2].N == 1
        assert model.components[2].par_dict["A"] == [20, True, 5, 25]
        assert model.components[2].par_dict["x0"] == [84.5, True, 82, 88]
        assert model.components[2].par_dict["F"] == [1.0, True, 0.75, 2.5]
        assert model.components[2].par_dict["m"] == [0.3, True, 0, 1]

        # GLP_02
        assert model.components[3].fct_str == "GLP"
        assert model.components[3].comp_name == "GLP_02"
        assert model.components[3].N == 2
        assert model.components[3].par_dict["A"] == ["3/4*GLP_01_A"]
        assert model.components[3].par_dict["x0"] == ["GLP_01_x0 +3.6"]
        assert model.components[3].par_dict["F"] == ["GLP_01_F"]
        assert model.components[3].par_dict["m"] == ["GLP_01_m"]

    #
    def test_energy_expression_fwd_ref_model(self):
        """Test energy parameters with forward reference expressions"""
        model = self.setUp(["energy_expression_forward_reference"])
        assert model.name == "energy_expression_forward_reference"
        assert model.dim == 1
        assert len(model.components) == 4

        # check components
        # Offset
        assert model.components[0].fct_str == "Offset"
        assert model.components[0].comp_name == "Offset"
        assert "y0" in model.components[0].par_dict

        # Shirley
        assert model.components[1].fct_str == "Shirley"
        assert model.components[1].comp_name == "Shirley"
        assert model.components[1].par_dict["pShirley"] == [
            400,
            True,
            1.0e-6,
            1.0e3,
        ]

        # GLP_01
        assert model.components[2].fct_str == "GLP"
        assert model.components[2].comp_name == "GLP_01"
        assert model.components[2].N == 1
        assert model.components[2].par_dict["A"] == ["3/4*GLP_02_A"]
        assert model.components[2].par_dict["x0"] == ["GLP_02_x0 +3.6"]
        assert model.components[2].par_dict["F"] == ["GLP_02_F"]
        assert model.components[2].par_dict["m"] == ["GLP_02_m"]

        # GLP_02
        assert model.components[3].fct_str == "GLP"
        assert model.components[3].comp_name == "GLP_02"
        assert model.components[3].N == 2
        assert model.components[3].par_dict["A"] == [20, True, 5, 25]
        assert model.components[3].par_dict["x0"] == [84.5, True, 82, 88]
        assert model.components[3].par_dict["F"] == [1.0, True, 0.75, 2.5]
        assert model.components[3].par_dict["m"] == [0.3, True, 0, 1]


#
#
class TestTimeParsing:
    """Test parsing of 1D time models (mcp.Dynamics)."""

    #
    def setUp(self, model_info):
        """Setup function to create project, file, and load model"""
        project = Project(path="tests")
        file = File(parent_project=project)
        file.time = np.linspace(-10, 100, 111)  # needed for time-dependent models
        model = file.load_model(
            model_yaml="test_models_time.yaml",
            model_info=model_info,
            par_name="parTEST",  # this is the name of the time-dependent parameter
            debug=True,
            model_type="dynamics",
        )
        assert model is not None, "Model loading failed in setup"
        return model

    #
    def test_simple_time_model(self):
        """Test simple time model"""

        # import the model
        model = self.setUp(["MonoExpPos"])

        # check the model
        assert model.name == "parTEST"
        assert model.dim == 1
        assert len(model.components) == 1

        # check the component
        assert model.components[0].fct_str == "expFun"
        assert model.components[0].comp_name == "expFun_01"
        assert model.components[0].par_dict["A"] == [1, True, 0, 5]
        assert model.components[0].par_dict["tau"] == [2.5, True, 1, 10]
        assert model.components[0].par_dict["t0"] == [0, False, 0, 1]
        assert model.components[0].par_dict["y0"] == [0, False, 0, 1]

    #
    def test_IRF_model(self):
        """Test IRF model"""

        # import the model
        model = self.setUp(["MonoExpPosIRF"])

        # check the model
        assert model.name == "parTEST"
        assert model.dim == 1
        assert len(model.components) == 2

        # check the components
        assert model.components[0].fct_str == "gaussCONV"
        assert model.components[0].comp_name == "gaussCONV"
        assert model.components[0].par_dict["SD"] == [5.0e-2, True, 0, 1]

        assert model.components[1].fct_str == "expFun"
        assert model.components[1].comp_name == "expFun_01"
        assert model.components[1].par_dict["A"] == [1, True, 0, 5]
        assert model.components[1].par_dict["tau"] == [2.5, True, 1, 10]
        assert model.components[1].par_dict["t0"] == [0, False, 0, 1]
        assert model.components[1].par_dict["y0"] == [0, False, 0, 1]

    #
    def test_multi_cycle_expression_model(self):
        """Test multi-cycle dynamics with expressions referencing other subcycles."""

        # import the model:
        # ModelNone (all times) + MonoExpNeg (sub1) + MonoExpPosExpr (sub2)
        model = self.setUp(["ModelNone", "MonoExpNeg", "MonoExpPosExpr"])

        # check the model
        assert model.name == "parTEST"
        assert len(model.components) == 3

        # Subcycle 0: ModelNone — empty placeholder
        assert model.components[0].fct_str == "none"
        assert model.components[0].subcycle == 0

        # Subcycle 1: MonoExpNeg — numeric parameters
        assert model.components[1].fct_str == "expFun"
        assert model.components[1].comp_name == "expFun_01"
        assert model.components[1].subcycle == 1
        assert model.components[1].par_dict["A"] == [-1, True, -5, 0]
        assert model.components[1].par_dict["tau"] == [2.5, True, 1, 10]

        # Subcycle 2: MonoExpPosExpr — renumbered to expFun_02, expression parameters
        assert model.components[2].fct_str == "expFun"
        assert model.components[2].comp_name == "expFun_02"
        assert model.components[2].subcycle == 2
        assert model.components[2].par_dict["A"] == ["-expFun_01_A"]
        assert model.components[2].par_dict["tau"] == ["expFun_01_tau"]
        assert model.components[2].par_dict["t0"] == [0, False, 0, 1]
        assert model.components[2].par_dict["y0"] == [0, False, 0, 1]

        # Check lmfit parameters exist with prefixed names
        assert "parTEST_expFun_01_A" in model.lmfit_pars
        assert "parTEST_expFun_01_tau" in model.lmfit_pars
        assert "parTEST_expFun_02_A" in model.lmfit_pars
        assert "parTEST_expFun_02_tau" in model.lmfit_pars

        # Expressions should be auto-prefixed with par_name:
        # "-expFun_01_A" → "-parTEST_expFun_01_A"
        expr_A = model.lmfit_pars["parTEST_expFun_02_A"].expr
        assert expr_A == "-parTEST_expFun_01_A"

        # "expFun_01_tau" → "parTEST_expFun_01_tau"
        expr_tau = model.lmfit_pars["parTEST_expFun_02_tau"].expr
        assert expr_tau == "parTEST_expFun_01_tau"


#
#
class Test2DModelParsing:
    """Test parsing of 2D energy- and time-resolved models."""

    #
    def setUp(self, model_energy, par_name, model_time):
        """Setup function to create project, file, load model, and add dynamics"""
        project = Project(path="tests")
        file = File(parent_project=project)
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=model_energy,
            debug=False,
        )
        file.time = np.linspace(-10, 100, 111)  # needed for time-dependent models
        file.add_time_dependence(
            model_yaml="test_models_time.yaml",
            model_info=model_time,
            par_name=par_name,
        )
        model = file.model_active
        assert model is not None, "Model loading failed in setup"
        return model

    #
    def test_simple_2D_model(self):
        """Add IRF+exp_decay time-dependence to the simple energy model"""
        model = self.setUp(
            model_energy=["simple_energy"],
            par_name="GLP_01_x0",
            model_time=["MonoExpPosIRF"],
        )

        # check the model
        assert model.name == "simple_energy"
        assert model.dim == 2
        assert len(model.components) == 4
        assert model.components[0].fct_str == "Offset"
        assert model.components[0].comp_name == "Offset"
        assert model.components[0].par_dict["y0"] == [2, True, 0, 5]
        assert model.components[1].fct_str == "Shirley"
        assert model.components[1].comp_name == "Shirley"
        # Updated to check [value, vary] format
        assert model.components[1].par_dict["pShirley"] == [400, False]
        # GLP_01
        assert model.components[2].fct_str == "GLP"
        assert model.components[2].comp_name == "GLP_01"
        assert model.components[2].par_dict["A"] == [20, True, 5, 25]
        assert model.components[2].par_dict["x0"] == [84.5, True, 82, 88]
        # x0 is the time-dependent parameter
        td_par_model = model.components[2].pars[1].t_model
        assert td_par_model is not None
        assert td_par_model.components[0].comp_name == "gaussCONV"
        assert td_par_model.components[0].par_dict["SD"] == [5.0e-2, True, 0, 1]
        assert td_par_model.components[1].fct_str == "expFun"
        assert td_par_model.components[1].comp_name == "expFun_01"
        assert td_par_model.components[1].par_dict["A"] == [1, True, 0, 5]
        assert td_par_model.components[1].par_dict["tau"] == [2.5, True, 1, 10]
        assert td_par_model.components[1].par_dict["t0"] == [0, False, 0, 1]
        assert td_par_model.components[1].par_dict["y0"] == [0, False, 0, 1]
        # end of time-dependent parameter model
        assert model.components[2].par_dict["F"] == [1.0, True, 0.75, 2.5]
        assert model.components[2].par_dict["m"] == [0.3, True, 0, 1]
        # GLP_02 - x0 now uses [value, vary] format
        assert model.components[3].fct_str == "GLP"
        assert model.components[3].comp_name == "GLP_02"
        assert model.components[3].par_dict["A"] == [17, True, 5, 25]
        assert model.components[3].par_dict["x0"] == [88.1, True]
        assert model.components[3].par_dict["F"] == [1.0, True, 0.75, 2.5]
        assert model.components[3].par_dict["m"] == [0.3, True, 0, 1]

    #
    def test_time_dependence_on_expression_parameter_raises(self):
        """Adding dynamics to expression-linked parameter should fail."""
        project = Project(path="tests")
        file = File(parent_project=project)
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["energy_expression"],
            debug=False,
        )
        file.time = np.linspace(-10, 100, 111)

        with pytest.raises(
            ValueError, match="Cannot add time dependence to expression parameter"
        ):
            file.add_time_dependence(
                model_yaml="test_models_time.yaml",
                model_info=["MonoExpPosIRF"],
                par_name="GLP_02_x0",
            )


#
#
class TestProfileParsing:
    """Test parsing of profile models (mcp.Profile)."""

    #
    def setUp(self, model_info, par_name):
        """Setup function to create project, file with energy model, and load profile"""
        project = Project(path="tests")
        file = File(parent_project=project)
        file.energy = np.linspace(80, 90, 201)
        file.aux_axis = np.linspace(0, 10, 21)
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["simple_energy"],
            debug=False,
        )
        file.add_par_profile(
            model_yaml="test_models_profile.yaml",
            model_info=model_info,
            par_name=par_name,
        )
        model = file.model_active
        assert model is not None, "Model loading failed in setup"
        return model, file

    #
    def test_exp_decay_profile(self):
        """Test exp_decay profile attached to GLP_01_A"""
        model, file = self.setUp(["profile_exp_decay"], "GLP_01_A")

        # target parameter should have p_vary set
        ci, pi = model.find_par_by_name("GLP_01_A")
        assert ci is not None
        assert pi is not None
        par = model.components[ci].pars[pi]
        assert par.p_vary is True
        assert par.p_model is not None

        # profile model should be a Profile with correct components
        p_mod = par.p_model
        assert len(p_mod.components) == 1
        assert p_mod.components[0].fct_str == "exp_decay"
        assert p_mod.components[0].comp_name == "exp_decay_01"
        assert p_mod.components[0].par_dict["A"] == [200, True, 10, 1000]
        assert p_mod.components[0].par_dict["tau"] == [2.0, True, 0.5, 10.0]

        # profile lmfit par names follow convention: {par_name}_{comp}_{par}
        assert "GLP_01_A_exp_decay_01_A" in model.lmfit_pars
        assert "GLP_01_A_exp_decay_01_tau" in model.lmfit_pars

    #
    def test_linear_profile(self):
        """Test linear profile attached to GLP_01_x0"""
        model, file = self.setUp(["profile_linear"], "GLP_01_x0")

        # target parameter should have p_vary set
        ci, pi = model.find_par_by_name("GLP_01_x0")
        assert ci is not None
        assert pi is not None
        par = model.components[ci].pars[pi]
        assert par.p_vary is True

        # profile model components
        p_mod = par.p_model
        assert p_mod is not None
        assert len(p_mod.components) == 1
        assert p_mod.components[0].fct_str == "linear"
        assert p_mod.components[0].comp_name == "linear_01"
        assert p_mod.components[0].par_dict["m"] == [-0.5, True, -2, 2]
        assert p_mod.components[0].par_dict["b"] == [0.0, False, -1.0, 1.0]

        # lmfit par names
        assert "GLP_01_x0_linear_01_m" in model.lmfit_pars
        assert "GLP_01_x0_linear_01_b" in model.lmfit_pars

    #
    def test_profile_aux_axis_propagated(self):
        """Profile model should carry the aux_axis from File"""
        model, file = self.setUp(["profile_exp_decay"], "GLP_01_A")

        ci, pi = model.find_par_by_name("GLP_01_A")
        assert ci is not None
        assert pi is not None
        par = model.components[ci].pars[pi]
        p_mod = par.p_model
        assert p_mod is not None
        assert p_mod.aux_axis is not None
        np.testing.assert_array_equal(p_mod.aux_axis, file.aux_axis)

    #
    def test_profile_on_expression_parameter_raises(self):
        """Adding a profile to an expression-linked parameter should fail."""
        project = Project(path="tests")
        file = File(parent_project=project)
        file.energy = np.linspace(80, 90, 201)
        file.aux_axis = np.linspace(0, 10, 21)
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["energy_expression"],
            debug=False,
        )

        with pytest.raises(ValueError):
            file.add_par_profile(
                model_yaml="test_models_profile.yaml",
                model_info=["profile_exp_decay"],
                par_name="GLP_02_A",
            )


#
#
class TestYAMLValidationErrors:
    """Test YAML validation errors (malformed model definitions)."""

    #
    def test_wrong_parameter_order_accepted(self):
        """Parameters in non-standard order should still parse correctly."""
        project = Project(path="tests")
        file = File(parent_project=project)
        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["wrong_order"],
        )
        model = file.model_active
        assert model is not None
        # Values should be correctly assigned despite m/F swap in YAML
        assert model.components[1].par_dict["m"] == [0.3, True, 0, 1]
        assert model.components[1].par_dict["F"] == [1.0, True, 0.75, 2.5]

    #
    def test_wrong_parameter_name_raises(self):
        """Unknown parameter name (q instead of m for GLP) should fail validation."""
        project = Project(path="tests")
        file = File(parent_project=project)
        with pytest.raises(ModelValidationError, match="Invalid parameter"):
            file.load_model(
                model_yaml="test_models_energy.yaml",
                model_info=["wrong_parameter_name"],
            )

    #
    def test_nonexistent_model_raises(self):
        """Loading a model name that doesn't exist in the YAML should fail."""
        project = Project(path="tests")
        file = File(parent_project=project)
        with pytest.raises(ValueError, match="not found in"):
            file.load_model(
                model_yaml="test_models_energy.yaml",
                model_info=["this_model_does_not_exist"],
            )

    #
    def test_model_info_not_list_raises(self):
        """Passing a string instead of a list for model_info should fail."""
        project = Project(path="tests")
        file = File(parent_project=project)
        with pytest.raises(TypeError, match="model_info must be a list"):
            file.load_model(
                model_yaml="test_models_energy.yaml",
                model_info="simple_energy",  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    pytest.main([__file__])
