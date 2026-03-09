"""
Test MCP (Model/Component/Parameter) library functionality
"""

import numpy as np
import pytest

from trspecfit.functions import profile as fcts_profile
from trspecfit.functions import time as fcts_time
from trspecfit.mcp import Component, Dynamics, Model, Par, Profile


#
#
class TestMCPModel:
    """Test MCP Model class functionality"""

    #
    def test_model_creation(self):
        """Test basic model creation and initialization"""
        model = Model("test_model")

        assert model.name == "test_model"
        assert model.components == []
        assert model.lmfit_pars is not None
        assert model.dim is None
        assert model.energy is None
        assert model.time is None

    #
    def test_model_with_components(self):
        """Test model with spectral components (Au4f test from notebook)"""
        # Initialize 2D fit model
        mod2D = Model("Au4f_test")
        mod2D.energy = np.arange(75, 95, 0.1)[::-1]
        mod2D.time = np.arange(-500, 2500, 10)

        # Define Shirley background
        c_Shirley = Component("Shirley")
        c_Shirley.add_pars({"pShirley": [2500, True, 1e-6, 1e6]})

        # Define Offset parameters
        c_Offset = Component("Offset")
        c_Offset.add_pars({"y0": [3, True, 0, 5]})

        # Define peak components
        c_peak1 = Component("GLP")
        c_peak1.add_pars(
            {
                "A": [16, True, 5, 25],
                "x0": [84.0, True, 81, 87],
                "F": [1.6, True, 1, 2.5],
                "m": [0.3, False, 0, 1],
            }
        )

        c_peak2 = Component("GLP")
        c_peak2.add_pars(
            {
                "A": [12, True, 1, 20],
                "x0": [87.6, True, 84, 90],
                "F": [1.6, True, 1, 2.5],
                "m": [0.3, False, 0, 1],
            }
        )

        # Add components to model
        mod2D.add_components([c_Offset, c_Shirley, c_peak1, c_peak2])

        # Check model structure
        assert mod2D.name == "Au4f_test"
        assert len(mod2D.components) == 4
        assert mod2D.components[0].fct_str == "Offset"
        assert mod2D.components[1].fct_str == "Shirley"
        assert mod2D.components[2].fct_str == "GLP"
        assert mod2D.components[3].fct_str == "GLP"

    #
    def test_model_parameter_profile(self):
        """Profile model adds p_vary to par and produces averaged spectrum."""
        mod = Model("test_profile")
        mod.energy = np.linspace(80, 90, 100)
        mod.aux_axis = np.linspace(0, 5, 20)

        c_peak = Component("GLP_01")
        c_peak.add_pars(
            {
                "A": [10, True, 1, 20],
                "x0": [85, False],
                "F": [1.5, False],
                "m": [0, False],
            }
        )
        mod.add_components([c_peak])

        p_model = Profile("GLP_01_A")
        p_model.aux_axis = mod.aux_axis
        c_prof = Component("exp_decay_01", fcts_profile)
        c_prof.add_pars({"A": [1.0, False], "tau": [2.0, False]})
        p_model.add_components([c_prof])

        mod.add_profile(p_model)

        a_par = mod.components[0].pars[0]
        assert a_par.p_vary
        assert a_par.p_model is p_model

        val = mod.create_value1D(return1D=1)
        assert val is not None
        assert val.shape == mod.energy.shape
        assert np.isfinite(val).all()
        assert np.any(val > 0)


#
#
class TestMCPComponent:
    """Test MCP Component class functionality"""

    #
    def test_component_creation(self):
        """Test basic component creation"""
        comp = Component("GLP")

        assert comp.fct_str == "GLP"
        assert comp.N is None  # Not numbered initially
        assert comp.par_dict == {}
        assert comp.pars == []

    #
    def test_numbered_component(self):
        """Test numbered component creation"""
        comp = Component("GLP_01")

        assert comp.fct_str == "GLP"
        assert comp.N == 1
        assert comp.comp_name == "GLP_01"

    #
    def test_component_parameter_management(self):
        """Test component parameter addition and management"""
        comp = Component("GLP")
        comp.add_pars(
            {
                "A": [20, True, 5, 25],
                "x0": [84.5, True, 82, 88],
                "F": [1.0, True, 0.75, 2.5],
                "m": [0.3, True, 0, 1],
            }
        )

        assert comp.par_dict["A"] == [20, True, 5, 25]
        assert comp.par_dict["x0"] == [84.5, True, 82, 88]
        assert comp.par_dict["F"] == [1.0, True, 0.75, 2.5]
        assert comp.par_dict["m"] == [0.3, True, 0, 1]

    #
    def test_component_prefix_handling(self):
        """Test component prefix handling when component number changes"""
        comp = Component("GLP")
        comp.add_pars({"A": [20, True, 5, 25], "x0": [84.5, True, 82, 88]})

        # Update component number and name
        comp.N = 8
        comp.comp_name = "GLP_08"
        assert comp.prefix == "GLP_08_"
        assert comp.comp_name == "GLP_08"

    #
    def test_component_creation_with_energy_time(self):
        """Test component creation with energy and time axes"""
        comp = Component("GLP")
        comp.add_pars(
            {
                "A": [20, True, 5, 25],
                "x0": [84.5, True, 82, 88],
                "F": [1.0, True, 0.75, 2.5],
                "m": [0.3, False, 0, 1],
            }
        )

        # Set energy and time axes
        comp.energy = np.linspace(80, 90, 100)
        comp.time = np.linspace(0, 10, 50)

        # Create parameters
        comp.create_pars()

        assert len(comp.pars) == 4
        assert comp.pars[0].name == "GLP_A"
        assert comp.pars[1].name == "GLP_x0"
        assert comp.pars[2].name == "GLP_F"
        assert comp.pars[3].name == "GLP_m"


#
#
class TestMCPParameter:
    """Test MCP Parameter class functionality"""

    #
    def test_parameter_creation(self):
        """Test basic parameter creation"""
        par = Par("test_param")

        assert par.name == "test_param"
        assert par.info == []
        assert not par.t_vary
        assert par.t_model is None

    #
    def test_parameter_with_info(self):
        """Test parameter creation with parameter info"""
        par = Par("test_param", [87.6, True, 84, 90])

        assert par.name == "test_param"
        assert par.info == [87.6, True, 84, 90]

    #
    def test_parameter_creation_with_expression(self):
        """Test parameter creation with expression"""
        par = Par("test_param", ["GLP_01_A * 0.75"])

        assert par.name == "test_param"
        assert par.info == ["GLP_01_A * 0.75"]

    #
    def test_parameter_lmfit_creation(self):
        """Test parameter lmfit object creation"""
        par = Par("test_param", [87.6, True, 84, 90])
        par.create()

        assert par.lmfit_par is not None
        assert "test_param" in par.lmfit_par
        assert par.lmfit_par["test_param"].value == 87.6
        assert par.lmfit_par["test_param"].vary
        assert par.lmfit_par["test_param"].min == 84
        assert par.lmfit_par["test_param"].max == 90


#
#
class TestMCPDynamics:
    """Test MCP Dynamics (time-dependent) functionality"""

    #
    def test_dynamics_model_creation(self):
        """Test creation of time-dependence model"""
        # Initialize dynamics model
        t_mod = Model("GLP_01_x0")
        t_mod.time = np.linspace(-10, 100, 111)

        # Define instrument response function
        c_IRF = Component("gaussCONV", fcts_time)
        c_IRF.add_pars({"SD": [80, True, 0, 1e4]})

        # Define decay components
        c_tD1 = Component("expFun", fcts_time)
        c_tD1.add_pars(
            {
                "A": [2, True, 1, 1e2],
                "tau": [5000, True, 1e3, 1e4],
                "t0": [0, False, 0, 1],
                "y0": [0, False, 0, 1],
            }
        )

        c_tD2 = Component("expFun", fcts_time)
        c_tD2.add_pars(
            {
                "A": [5, True, 1, 1e2],
                "tau": [1250, True, 1e2, 1e3],
                "t0": [0, False, 0, 1],
                "y0": [0, False, 0, 1],
            }
        )

        # Add components to dynamics model
        t_mod.add_components([c_IRF, c_tD1, c_tD2])

        # Check model structure
        assert t_mod.name == "GLP_01_x0"
        assert len(t_mod.components) == 3
        assert t_mod.components[0].fct_str == "gaussCONV"
        assert t_mod.components[1].fct_str == "expFun"
        assert t_mod.components[2].fct_str == "expFun"

    #
    def test_dynamics_parameter_handling(self):
        """Test parameter handling in dynamics models"""
        # Create a simple dynamics model
        t_mod = Model("test_dynamics")
        t_mod.time = np.linspace(0, 10, 100)

        # Add a simple exponential component
        c_exp = Component("expFun", fcts_time)
        c_exp.add_pars(
            {
                "A": [1, True, 0, 5],
                "tau": [2.5, True, 1, 10],
                "t0": [0, False, 0, 1],
                "y0": [0, False, 0, 1],
            }
        )

        t_mod.add_components([c_exp])

        # Check parameter structure
        assert len(t_mod.components) == 1
        assert t_mod.components[0].par_dict["A"] == [1, True, 0, 5]
        assert t_mod.components[0].par_dict["tau"] == [2.5, True, 1, 10]


#
#
class TestMCPIntegration:
    """Test MCP integration with 2D models"""

    #
    def test_2d_model_with_dynamics(self):
        """Test 2D model with time-dependent parameters"""
        # Create 2D model
        mod2D = Model("Au4f_2D")
        mod2D.energy = np.linspace(80, 90, 100)
        mod2D.time = np.linspace(0, 10, 50)

        # Add spectral components - need to use numbered components for dynamics
        c_Offset = Component("Offset")
        c_Offset.add_pars({"y0": [3, True, 0, 5]})

        c_peak = Component("GLP_01")  # Use numbered component name
        c_peak.add_pars(
            {
                "A": [16, True, 5, 25],
                "x0": [84.0, True, 81, 87],
                "F": [1.6, True, 1, 2.5],
                "m": [0.3, False, 0, 1],
            }
        )

        mod2D.add_components([c_Offset, c_peak])

        # Create dynamics model for x0 parameter
        t_mod = Dynamics("GLP_01_x0")
        t_mod.time = mod2D.time

        c_IRF = Component("gaussCONV", fcts_time)
        c_IRF.add_pars({"SD": [80, True, 0, 1e4]})  # Use SD parameter name

        c_exp = Component("expFun", fcts_time)
        c_exp.add_pars(
            {
                "A": [2, True, 1, 1e2],
                "tau": [5000, True, 1e3, 1e4],
                "t0": [0, False, 0, 1],
                "y0": [0, False, 0, 1],
            }
        )

        t_mod.add_components([c_IRF, c_exp])

        # Add dynamics to 2D model
        mod2D.add_dynamics(t_mod)

        # Check integration
        assert mod2D.name == "Au4f_2D"
        assert len(mod2D.components) == 2
        # The x0 parameter should now have time dependence
        x0_param = mod2D.components[1].pars[1]  # x0 is the second parameter
        assert x0_param.t_vary
        assert x0_param.t_model is not None

    #
    def test_parameter_value_updates(self):
        """Test parameter value updates during fitting"""
        # Create a simple model
        model = Model("test_updates")
        model.energy = np.linspace(80, 90, 100)
        model.time = np.linspace(0, 10, 50)

        # Add a component with wider bounds to avoid clipping
        c_peak = Component("GLP")
        c_peak.add_pars(
            {
                "A": [10, True, 5, 20],  # Wider bounds
                "x0": [85, True, 80, 95],  # Wider bounds
                "F": [1.5, True, 1, 5],  # Wider bounds
                "m": [0.3, False, 0, 1],  # This parameter is fixed
            }
        )

        model.add_components([c_peak])

        # Get initial values
        initial_values = [model.lmfit_pars[p].value for p in model.lmfit_pars]

        # Update parameter values - only update varying parameters
        new_values = []
        for i, p in enumerate(model.lmfit_pars):
            if model.lmfit_pars[p].vary:  # Only update varying parameters
                new_val = initial_values[i] + 1
                new_values.append(new_val)
                model.lmfit_pars[p].value = new_val
            else:
                new_values.append(initial_values[i])  # Keep fixed parameters unchanged

        # Check that values were updated correctly
        updated_values = [model.lmfit_pars[p].value for p in model.lmfit_pars]
        assert updated_values == new_values


#
#
class TestMCPNormalization:
    """Test MCP time normalization functionality"""

    #
    def test_time_normalization(self):
        """Test time normalization for multi-cycle dynamics"""
        # Create a dynamics model with frequency

        t_mod = Dynamics("test_normalization")
        t_mod.time = np.linspace(0, 100, 1000)
        t_mod.subcycles = 3

        # Test that we can set frequency
        t_mod.set_frequency(frequency=0.1)
        # Test that normalization attributes are created
        assert hasattr(t_mod, "time_norm")
        assert hasattr(t_mod, "N_sub")
        assert hasattr(t_mod, "N_counter")

        # Test that normalized time is calculated
        if hasattr(t_mod, "time_norm"):
            assert t_mod.time_norm is not None
            assert len(t_mod.time_norm) == len(t_mod.time)

    #
    def test_subcycle_handling(self):
        """Test subcycle handling in components"""
        # Create a component with subcycle
        comp = Component("expFun", fcts_time, comp_subcycle=1)
        comp.subcycle = 1

        assert comp.subcycle == 1
        assert comp.comp_type == "add"  # Default for non-background functions


#
#
class TestMCPProfile:
    """Test Profile model functionality."""

    #
    def _make_model_with_peak(self, aux_axis=None):
        """Helper: energy model with one GLP_01 component."""
        mod = Model("test")
        mod.energy = np.linspace(80, 90, 100)
        mod.aux_axis = aux_axis if aux_axis is not None else np.linspace(0, 5, 20)

        c_peak = Component("GLP_01")
        c_peak.add_pars(
            {
                "A": [10, True, 1, 20],
                "x0": [85, False],
                "F": [1.5, False],
                "m": [0, False],
            }
        )
        mod.add_components([c_peak])
        return mod

    #
    def _make_exp_profile(self, name, aux_axis):
        """Helper: Profile model with a single exp_decay component."""
        p_model = Profile(name)
        p_model.aux_axis = aux_axis
        c_prof = Component("exp_decay_01", fcts_profile)
        c_prof.add_pars({"A": [1.0, False], "tau": [2.0, False]})
        p_model.add_components([c_prof])
        return p_model

    #
    def test_profile_class_creation(self):
        """Profile should inherit from Model and carry parent_model."""
        p = Profile("test_profile")
        assert p.name == "test_profile"
        assert p.parent_model is None
        assert p.aux_axis is None

    #
    def test_add_profile_sets_p_vary(self):
        """add_profile() should set p_vary=True on the target parameter."""
        mod = self._make_model_with_peak()
        p_model = self._make_exp_profile("GLP_01_A", mod.aux_axis)

        mod.add_profile(p_model)

        a_par = mod.components[0].pars[0]
        assert a_par.p_vary is True
        assert a_par.p_model is p_model

    #
    def test_add_profile_propagates_aux_axis(self):
        """add_profile() propagates aux_axis to Profile and its components."""
        mod = self._make_model_with_peak()
        p_model = self._make_exp_profile("GLP_01_A", mod.aux_axis)
        mod.add_profile(p_model)

        assert p_model.aux_axis is not None
        assert mod.aux_axis is not None

        assert np.array_equal(p_model.aux_axis, mod.aux_axis)
        for comp in p_model.components:
            assert comp.aux_axis is not None
            assert np.array_equal(comp.aux_axis, mod.aux_axis)

    #
    def test_add_profile_sets_parent_model(self):
        """add_profile() should set parent_model on the Profile."""
        mod = self._make_model_with_peak()
        p_model = self._make_exp_profile("GLP_01_A", mod.aux_axis)
        mod.add_profile(p_model)
        assert p_model.parent_model is mod

    #
    def test_profile_value1D_initialized(self):
        """Profile.value1D should be set after add_profile."""
        mod = self._make_model_with_peak()
        p_model = self._make_exp_profile("GLP_01_A", mod.aux_axis)
        mod.add_profile(p_model)

        assert p_model.value1D is not None
        assert mod.aux_axis is not None
        assert len(p_model.value1D) == len(mod.aux_axis)

    #
    def test_component_value_averaging(self):
        """Component with p_vary should return uniform average over aux_axis."""
        aux = np.linspace(0, 5, 20)
        mod = self._make_model_with_peak(aux_axis=aux)
        p_model = self._make_exp_profile("GLP_01_A", aux)
        mod.add_profile(p_model)

        # Evaluate: should be finite and match energy shape
        val = mod.create_value1D(return1D=1)
        assert val is not None
        assert mod.energy is not None
        assert val.shape == mod.energy.shape
        assert np.isfinite(val).all()
        assert np.any(val > 0)

    #
    def test_profile_differs_from_no_profile(self):
        """Profile averaging should differ from the base parameter alone."""
        aux = np.linspace(0, 5, 20)
        energy = np.linspace(80, 90, 100)

        # Model without profile: A = 10 (base)
        mod_flat = Model("flat")
        mod_flat.energy = energy
        c1 = Component("GLP_01")
        c1.add_pars(
            {
                "A": [10, True, 1, 20],
                "x0": [85, False],
                "F": [1.5, False],
                "m": [0, False],
            }
        )
        mod_flat.add_components([c1])
        val_flat = mod_flat.create_value1D(return1D=1)

        # Model with profile: base A=0, profile adds exp_decay(depth, A=10, tau=2)
        mod_prof = Model("profiled")
        mod_prof.energy = energy
        mod_prof.aux_axis = aux
        c2 = Component("GLP_01")
        c2.add_pars(
            {"A": [0, False], "x0": [85, False], "F": [1.5, False], "m": [0, False]}
        )
        mod_prof.add_components([c2])
        p_model = self._make_exp_profile("GLP_01_A", aux)
        mod_prof.add_profile(p_model)
        val_prof = mod_prof.create_value1D(return1D=1)

        assert val_flat is not None
        assert val_prof is not None
        # Profiles should NOT equal the flat case (different amplitudes)
        assert not np.allclose(val_flat, val_prof)

    #
    def test_multiple_p_vary_pars_same_component(self):
        """Two p_vary parameters on one component share the same aux_axis loop."""
        aux = np.linspace(0, 5, 15)
        mod = Model("multi_profile")
        mod.energy = np.linspace(80, 90, 100)
        mod.aux_axis = aux

        c_peak = Component("GLP_01")
        c_peak.add_pars(
            {"A": [0, False], "x0": [0, False], "F": [1.5, False], "m": [0, False]}
        )
        mod.add_components([c_peak])

        # Profile on A: exp_decay
        p_A = Profile("GLP_01_A")
        p_A.aux_axis = aux
        c_A = Component("exp_decay_01", fcts_profile)
        c_A.add_pars({"A": [10.0, False], "tau": [2.0, False]})
        p_A.add_components([c_A])
        mod.add_profile(p_A)

        # Profile on x0: linear (band bending)
        p_x0 = Profile("GLP_01_x0")
        p_x0.aux_axis = aux
        c_x0 = Component("linear_01", fcts_profile)
        c_x0.add_pars({"m": [0.1, False], "b": [85.0, False]})
        p_x0.add_components([c_x0])
        mod.add_profile(p_x0)

        val = mod.create_value1D(return1D=1)
        assert val is not None
        assert val.shape == mod.energy.shape
        assert np.isfinite(val).all()

    #
    def test_add_profile_raises_without_aux_axis(self):
        """add_profile() should raise ValueError if aux_axis is not set."""
        mod = Model("no_aux")
        mod.energy = np.linspace(80, 90, 100)
        # No aux_axis set

        c_peak = Component("GLP_01")
        c_peak.add_pars(
            {
                "A": [10, True, 1, 20],
                "x0": [85, False],
                "F": [1.5, False],
                "m": [0, False],
            }
        )
        mod.add_components([c_peak])

        p_model = Profile("GLP_01_A")
        with pytest.raises(ValueError, match="aux_axis"):
            mod.add_profile(p_model)

    #
    def test_add_profile_raises_for_expression_par(self):
        """add_profile() should raise ValueError for expression parameters."""
        mod = Model("expr_model")
        mod.energy = np.linspace(80, 90, 100)
        mod.aux_axis = np.linspace(0, 5, 20)

        c1 = Component("GLP_01")
        c1.add_pars(
            {
                "A": [10, True, 1, 20],
                "x0": [85, False],
                "F": [1.5, False],
                "m": [0, False],
            }
        )
        c2 = Component("GLP_02")
        c2.add_pars(
            {"A": ["GLP_01_A"], "x0": [85, False], "F": [1.5, False], "m": [0, False]}
        )
        mod.add_components([c1, c2])

        p_model = Profile("GLP_02_A")
        with pytest.raises(ValueError, match="expression"):
            mod.add_profile(p_model)

    #
    def test_file_aux_axis_propagation(self):
        """File.aux_axis should propagate to loaded Model via load_model()."""
        from trspecfit import File, Project

        project = Project(path="tests")
        file = File(parent_project=project)
        file.energy = np.linspace(80, 90, 100)
        file.time = np.linspace(-10, 100, 50)
        file.aux_axis = np.linspace(0, 5, 20)

        file.load_model(
            model_yaml="test_models_energy.yaml",
            model_info=["energy_expression"],
        )
        assert file.model_active is not None
        assert file.model_active.aux_axis is not None
        assert np.array_equal(file.model_active.aux_axis, file.aux_axis)

    #
    def test_profile_with_dynamics(self):
        """Profile on an amplitude parameter should work alongside time dynamics."""

        aux = np.linspace(0, 5, 15)
        energy = np.linspace(80, 90, 100)
        time = np.linspace(-10, 50, 60)

        mod = Model("combined")
        mod.energy = energy
        mod.time = time
        mod.aux_axis = aux

        c_peak = Component("GLP_01")
        c_peak.add_pars(
            {"A": [0, False], "x0": [85, False], "F": [1.5, False], "m": [0, False]}
        )
        mod.add_components([c_peak])

        # Add exp_decay profile to A
        p_model = self._make_exp_profile("GLP_01_A", aux)
        mod.add_profile(p_model)

        # Add dynamics to x0
        t_mod = Dynamics("GLP_01_x0")
        t_mod.time = time
        c_exp = Component("expFun", fcts_time)
        c_exp.add_pars(
            {"A": [2, False], "tau": [20, False], "t0": [0, False], "y0": [0, False]}
        )
        t_mod.add_components([c_exp])
        mod.add_dynamics(t_mod)

        # Should evaluate at t_ind=5 without error
        val = mod.create_value1D(t_ind=5, return1D=1)
        assert val is not None
        assert val.shape == energy.shape
        assert np.isfinite(val).all()


if __name__ == "__main__":
    pytest.main([__file__])
