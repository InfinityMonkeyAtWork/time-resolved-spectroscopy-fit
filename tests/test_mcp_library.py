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
        mod_2d = Model("Au4f_test")
        mod_2d.energy = np.arange(75, 95, 0.1)[::-1]
        mod_2d.time = np.arange(-500, 2500, 10)

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
        mod_2d.add_components([c_Offset, c_Shirley, c_peak1, c_peak2])

        # Check model structure
        assert mod_2d.name == "Au4f_test"
        assert len(mod_2d.components) == 4
        assert mod_2d.components[0].fct_str == "Offset"
        assert mod_2d.components[1].fct_str == "Shirley"
        assert mod_2d.components[2].fct_str == "GLP"
        assert mod_2d.components[3].fct_str == "GLP"

    #
    def test_model_parameter_profile(self):
        """Profile model produces averaged spectrum that differs from flat A."""

        mod = Model("test_profile")
        energy = np.linspace(80, 90, 100)
        aux = np.linspace(0, 5, 20)
        mod.energy = energy
        mod.aux_axis = aux

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

        # Spectrum without profile (flat A=10)
        val_flat = mod.create_value_1d(return_1d=1)

        p_model = Profile("GLP_01_A")
        p_model.aux_axis = aux
        c_prof = Component("pExpDecay_01", fcts_profile)
        c_prof.add_pars({"A": [1.0, False], "tau": [2.0, False]})
        p_model.add_components([c_prof])

        mod.add_profile(p_model)

        # Spectrum with profile (A varies over aux_axis via exp decay)
        val_prof = mod.create_value_1d(return_1d=1)
        assert val_prof.shape == energy.shape
        assert np.isfinite(val_prof).all()

        # Profile adds positive values to A, so averaged peak should be larger
        assert np.max(val_prof) > np.max(val_flat)


#
#
class TestMCPComponent:
    """Test MCP Component class functionality"""

    #
    def test_component_creation(self):
        """Test basic component creation"""

        comp = Component("GLP")

        assert comp.fct_str == "GLP"
        assert comp.num is None  # Not numbered initially
        assert comp.par_dict == {}
        assert comp.pars == []

    #
    def test_numbered_component(self):
        """Test numbered component creation"""

        comp = Component("GLP_01")

        assert comp.fct_str == "GLP"
        assert comp.num == 1
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
        comp.num = 8
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

        assert par.lmfit_par is not None  # type guard
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
        c_td1 = Component("expFun", fcts_time)
        c_td1.add_pars(
            {
                "A": [2, True, 1, 1e2],
                "tau": [5000, True, 1e3, 1e4],
                "t0": [0, False, 0, 1],
                "y0": [0, False, 0, 1],
            }
        )

        c_td2 = Component("expFun", fcts_time)
        c_td2.add_pars(
            {
                "A": [5, True, 1, 1e2],
                "tau": [1250, True, 1e2, 1e3],
                "t0": [0, False, 0, 1],
                "y0": [0, False, 0, 1],
            }
        )

        # Add components to dynamics model
        t_mod.add_components([c_IRF, c_td1, c_td2])

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
    def test_voigt_kernel_axis_uses_both_width_parameters(self):
        """Voigt kernel support widens when Lorentzian tails dominate."""

        t_mod = Model("test_voigt_irf")
        t_mod.time = np.arange(-5, 6, 1.0)

        c_irf = Component("voigtCONV", fcts_time)
        c_irf.add_pars(
            {
                "SD": [0.1, True, 0.01, 1],
                "W": [4.0, True, 0.01, 10],
            }
        )

        t_mod.add_components([c_irf])

        # Support should span the larger of 12*SD and 10*W.
        assert c_irf.time is not None
        assert c_irf.time[0] == pytest.approx(-40.0)
        assert c_irf.time[-1] == pytest.approx(40.0)


#
#
class TestMCPIntegration:
    """Test MCP integration with 2D models via the public API."""

    #
    def _make_file(self):
        """Helper: File with energy model loaded via public API."""

        from trspecfit import File, Project

        project = Project(path="tests")
        file = File(parent_project=project)
        file.energy = np.linspace(80, 90, 100)
        file.time = np.linspace(0, 10, 50)
        file.load_model(
            model_yaml="models/file_energy.yaml",
            model_info="simple_energy",
        )
        assert file.model_active is not None  # type guard
        return file

    #
    def test_2d_model_with_dynamics(self):
        """2D model: energy model + time dependence via add_time_dependence."""

        file = self._make_file()
        model = file.model_active

        file.add_time_dependence(
            target_model="simple_energy",
            target_parameter="GLP_01_x0",
            dynamics_yaml="models/file_time.yaml",
            dynamics_model=["MonoExpPosIRF"],
        )

        assert model.dim == 2

        # Spectrum should differ at early vs late times (x0 shifts)
        val_early = model.create_value_1d(t_ind=0, return_1d=1)
        val_late = model.create_value_1d(t_ind=40, return_1d=1)
        assert np.isfinite(val_early).all()
        assert np.isfinite(val_late).all()
        assert not np.allclose(val_early, val_late)

    #
    def test_parameter_value_updates(self):
        """lmfit parameter values can be read and updated after model loading."""

        from trspecfit import File, Project

        project = Project(path="tests")
        file = File(parent_project=project)
        file.energy = np.linspace(80, 90, 100)
        file.time = np.linspace(0, 10, 50)
        file.load_model(
            model_yaml="models/file_energy.yaml",
            model_info="single_glp",
        )
        model = file.model_active

        # Get initial values
        initial_values = [model.lmfit_pars[p].value for p in model.lmfit_pars]

        # Update varying, non-expression parameters within their bounds
        new_values = []
        for i, p in enumerate(model.lmfit_pars):
            par = model.lmfit_pars[p]
            if par.vary and par.expr in (None, ""):
                # Stay within bounds: use midpoint between current value and max
                new_val = (par.value + par.max) / 2
                new_values.append(new_val)
                par.value = new_val
            else:
                new_values.append(initial_values[i])

        updated_values = [model.lmfit_pars[p].value for p in model.lmfit_pars]
        assert updated_values == new_values


#
#
class TestMCPNormalization:
    """Test MCP time normalization functionality."""

    #
    def _normalize(self, time, frequency, subcycles):
        """Helper: run normalize_time and return the three arrays."""

        t_mod = Dynamics("test")
        t_mod.time = np.asarray(time)
        t_mod.frequency = frequency
        t_mod.subcycles = subcycles
        t_mod.normalize_time()
        return t_mod.time_norm, t_mod.n_sub, t_mod.n_counter

    #
    def test_time_normalization(self):
        """Test time normalization for multi-cycle dynamics."""

        t_mod = Dynamics("test_normalization")
        t_mod.time = np.linspace(0, 100, 1000)
        t_mod.subcycles = 3

        t_mod.set_frequency(frequency=0.1)
        assert t_mod.time_norm.shape == t_mod.time.shape
        # All values non-negative
        assert np.all(t_mod.time_norm >= -1e-15)
        # n_sub cycles through 1, 2, 3
        assert np.all(t_mod.n_sub >= 1)
        assert np.all(t_mod.n_sub <= 3)

    #
    def test_two_subcycles_values(self):
        """Test exact values for 2-subcycle normalization (freq=10, subcycles=2)."""

        time = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
        t_norm, n_sub, n_counter = self._normalize(time, frequency=10, subcycles=2)

        # norm = 1/(10*2) = 0.05, so subcycle boundaries at 0, 0.05, 0.10, ...
        assert len(t_norm) == len(time)
        # time_norm resets at each subcycle boundary
        assert np.allclose(t_norm, [0, 0, 0, 0.05, 0, 0, 0.05], atol=1e-12)
        # n_sub cycles through 1 and 2
        assert np.allclose(n_sub, [1, 2, 1, 1, 1, 2, 2])
        # n_counter increments each subcycle
        assert np.allclose(n_counter, [1, 2, 3, 3, 5, 6, 6])

    #
    def test_negative_times_are_zero(self):
        """Negative times produce zero for all output arrays."""

        time = np.array([-0.1, -0.05, -0.001, 0.0, 0.05])
        t_norm, n_sub, n_counter = self._normalize(time, frequency=10, subcycles=2)

        assert np.allclose(t_norm[:3], 0.0)
        assert np.allclose(n_sub[:3], 0.0)
        assert np.allclose(n_counter[:3], 0.0)
        # t=0 is the first subcycle
        assert n_sub[3] == 1.0
        assert n_counter[3] == 1.0

    #
    def test_three_subcycles(self):
        """n_sub cycles through 1, 2, 3 with three subcycles."""

        # norm = 1/(30*3) ≈ 0.0111, so use clean multiples
        freq, nsub = 30, 3
        norm = 1.0 / freq / nsub
        time = np.array([0, norm, 2 * norm, 3 * norm, 4 * norm, 5 * norm])
        t_norm, n_sub, _ = self._normalize(time, frequency=freq, subcycles=nsub)

        assert np.allclose(t_norm, 0.0, atol=1e-12)
        assert np.allclose(n_sub, [1, 2, 3, 1, 2, 3])

    #
    def test_no_repetition(self):
        """frequency=-1 passes time through unchanged."""

        time = np.array([-1.0, 0.0, 1.0, 2.0])
        t_norm, n_sub, n_counter = self._normalize(time, frequency=-1, subcycles=0)

        assert np.allclose(t_norm, time)
        assert np.allclose(n_sub, 0.0)
        assert np.allclose(n_counter, 0.0)

    #
    def test_many_parameter_combinations(self):
        """Sweep freq × subcycles and verify shapes and value ranges."""

        for freq in [5, 50, 1000]:
            for nsub in [2, 3, 5]:
                time = np.linspace(-0.05, 0.5, 200)
                t_norm, n_sub, n_counter = self._normalize(time, freq, nsub)

                assert t_norm.shape == time.shape
                # Negative-time entries are zero
                neg_mask = time < 0
                assert np.allclose(t_norm[neg_mask], 0.0)
                assert np.allclose(n_sub[neg_mask], 0.0)
                # Positive-time n_sub is in [1, nsub]
                pos_mask = time >= 0
                assert np.all(n_sub[pos_mask] >= 1)
                assert np.all(n_sub[pos_mask] <= nsub)
                # time_norm is non-negative
                assert np.all(t_norm[pos_mask] >= -1e-15)

    #
    def test_subcycle_handling(self):
        """Component created with subcycle should retain it."""

        comp = Component("expFun", fcts_time, comp_subcycle=1)

        assert comp.subcycle == 1


#
#
class TestMCPProfile:
    """Test Profile model functionality."""

    #
    def _make_file(self, *, aux_axis=None):
        """Helper: File with single GLP energy model loaded via public API."""

        from trspecfit import File, Project

        project = Project(path="tests")
        file = File(parent_project=project, aux_axis=aux_axis)
        file.energy = np.linspace(80, 90, 100)
        file.time = np.linspace(-10, 50, 60)
        file.load_model(
            model_yaml="models/file_energy.yaml",
            model_info="single_glp",
        )
        assert file.model_active is not None  # type guard
        return file

    #
    def test_profile_class_creation(self):
        """Profile should inherit from Model and carry parent_model."""

        p = Profile("test_profile")
        assert p.name == "test_profile"
        assert p.parent_model is None
        assert p.aux_axis is None

    #
    def test_add_profile_changes_spectrum(self):
        """add_par_profile() should change the evaluated spectrum."""

        # Spectrum without profile
        file_flat = self._make_file()
        val_flat = file_flat.model_active.create_value_1d(return_1d=1)

        # Spectrum with profile on A
        file = self._make_file(aux_axis=np.linspace(0, 5, 20))
        file.add_par_profile(
            target_model="single_glp",
            target_parameter="GLP_01_A",
            profile_yaml="models/file_profile.yaml",
            profile_model=["profile_pExpDecay"],
        )

        val_prof = file.model_active.create_value_1d(return_1d=1)
        assert val_prof.shape == val_flat.shape
        assert np.isfinite(val_prof).all()
        # Profile adds positive values to A, so peak amplitude should increase
        assert np.max(val_prof) > np.max(val_flat)

    #
    def test_add_profile_evaluates_over_aux_axis(self):
        """Profile should produce a value for each point on aux_axis."""

        aux = np.linspace(0, 5, 20)
        file = self._make_file(aux_axis=aux)

        file.add_par_profile(
            target_model="single_glp",
            target_parameter="GLP_01_A",
            profile_yaml="models/file_profile.yaml",
            profile_model=["profile_pExpDecay"],
        )

        ci, pi = file.model_active.find_par_by_name("GLP_01_A")
        p_model = file.model_active.components[ci].pars[pi].p_model
        assert p_model.value_1d is not None  # type guard
        assert len(p_model.value_1d) == len(aux)
        assert np.isfinite(p_model.value_1d).all()

    #
    def test_profile_accesses_parent_energy(self):
        """Profile should use parent model's energy axis for evaluation."""

        aux = np.linspace(0, 5, 20)
        file = self._make_file(aux_axis=aux)
        model = file.model_active

        file.add_par_profile(
            target_model="single_glp",
            target_parameter="GLP_01_A",
            profile_yaml="models/file_profile.yaml",
            profile_model=["profile_pExpDecay"],
        )

        # Full model evaluation should produce spectrum over parent's energy axis
        val = model.create_value_1d(return_1d=1)
        assert val.shape == model.energy.shape
        assert np.isfinite(val).all()
        assert np.max(val) > 0

    #
    def test_profile_value_1d_matches_analytical(self):
        """Profile value_1d should match the analytical pExpDecay curve."""

        aux = np.linspace(0, 5, 20)
        file = self._make_file(aux_axis=aux)

        file.add_par_profile(
            target_model="single_glp",
            target_parameter="GLP_01_A",
            profile_yaml="models/file_profile.yaml",
            profile_model=["profile_pExpDecay"],
        )

        ci, pi = file.model_active.find_par_by_name("GLP_01_A")
        p_model = file.model_active.components[ci].pars[pi].p_model
        # pExpDecay: A * exp(-aux / tau), with A=200, tau=2.0
        expected = 200.0 * np.exp(-aux / 2.0)
        np.testing.assert_allclose(p_model.value_1d, expected, rtol=1e-10)

    #
    def test_component_value_averaging(self):
        """Profiled spectrum should equal GLP at mean effective A."""

        from trspecfit.functions.energy import GLP

        aux = np.linspace(0, 5, 20)
        file = self._make_file(aux_axis=aux)
        model = file.model_active

        file.add_par_profile(
            target_model="single_glp",
            target_parameter="GLP_01_A",
            profile_yaml="models/file_profile.yaml",
            profile_model=["profile_pExpDecay"],
        )

        val = model.create_value_1d(return_1d=1)
        # GLP is linear in A, so average over aux = GLP at mean(A_eff)
        # A_eff = base_A + profile_value (additive)
        profile_vals = 200.0 * np.exp(-aux / 2.0)
        mean_A = np.mean(20.0 + profile_vals)
        expected = GLP(file.energy, A=mean_A, x0=85.0, F=1.0, m=0.3)
        np.testing.assert_allclose(val, expected, rtol=1e-10)

    #
    def test_profile_differs_from_no_profile(self):
        """Profile averaging should differ from the base parameter alone."""

        # Model without profile
        file_flat = self._make_file()
        model_flat = file_flat.model_active
        val_flat = model_flat.create_value_1d(return_1d=1)

        # Model with profile
        file_prof = self._make_file(aux_axis=np.linspace(0, 5, 20))
        file_prof.add_par_profile(
            target_model="single_glp",
            target_parameter="GLP_01_A",
            profile_yaml="models/file_profile.yaml",
            profile_model=["profile_pExpDecay"],
        )
        model_prof = file_prof.model_active
        val_prof = model_prof.create_value_1d(return_1d=1)

        # Profile adds positive values to A, so profiled peak should be larger
        assert np.max(val_prof) > np.max(val_flat)

    #
    def test_multiple_p_vary_pars_same_component(self):
        """Two p_vary parameters on one component share the same aux_axis loop."""

        aux = np.linspace(0, 5, 15)
        file = self._make_file(aux_axis=aux)
        model = file.model_active

        # Profile on A: pExpDecay
        file.add_par_profile(
            target_model="single_glp",
            target_parameter="GLP_01_A",
            profile_yaml="models/file_profile.yaml",
            profile_model=["profile_pExpDecay"],
        )

        # Profile on x0: pLinear
        file.add_par_profile(
            target_model="single_glp",
            target_parameter="GLP_01_x0",
            profile_yaml="models/file_profile.yaml",
            profile_model=["profile_pLinear"],
        )

        val = model.create_value_1d(return_1d=1)
        assert val.shape == file.energy.shape
        assert np.isfinite(val).all()
        # Both profiles active: should differ from single-profile result
        file_single = self._make_file(aux_axis=aux)
        file_single.add_par_profile(
            target_model="single_glp",
            target_parameter="GLP_01_A",
            profile_yaml="models/file_profile.yaml",
            profile_model=["profile_pExpDecay"],
        )
        val_single = file_single.model_active.create_value_1d(return_1d=1)
        assert not np.allclose(val, val_single)

    #
    def test_add_profile_raises_without_aux_axis(self):
        """add_par_profile() should raise when File has no aux_axis."""

        file = self._make_file()  # no aux_axis
        assert file.aux_axis is None

        with pytest.raises((ValueError, AttributeError)):
            file.add_par_profile(
                target_model="single_glp",
                target_parameter="GLP_01_A",
                profile_yaml="models/file_profile.yaml",
                profile_model=["profile_pExpDecay"],
            )

    #
    def test_add_profile_raises_for_expression_par(self):
        """add_par_profile() should raise for expression parameters."""

        from trspecfit import File, Project

        project = Project(path="tests")
        file = File(
            parent_project=project,
            aux_axis=np.linspace(0, 5, 20),
        )
        file.energy = np.linspace(80, 90, 100)
        file.time = np.linspace(-10, 50, 60)
        file.load_model(
            model_yaml="models/file_energy.yaml",
            model_info="two_glp_expr_amplitude",
        )

        # GLP_02_A is an expression ("GLP_01_A * 0.5") — profile should be rejected
        with pytest.raises(ValueError, match="expression"):
            file.add_par_profile(
                target_model="two_glp_expr_amplitude",
                target_parameter="GLP_02_A",
                profile_yaml="models/file_profile.yaml",
                profile_model=["profile_pExpDecay"],
            )

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
            model_yaml="models/file_energy.yaml",
            model_info="energy_expression",
        )
        assert file.model_active is not None  # type guard
        assert file.model_active.aux_axis is not None  # type guard
        assert np.array_equal(file.model_active.aux_axis, file.aux_axis)

    #
    def test_profile_with_dynamics(self):
        """Profile on amplitude should work alongside time dynamics on x0."""

        aux = np.linspace(0, 5, 15)
        file = self._make_file(aux_axis=aux)
        model = file.model_active

        # Add profile to A via public API
        file.add_par_profile(
            target_model="single_glp",
            target_parameter="GLP_01_A",
            profile_yaml="models/file_profile.yaml",
            profile_model=["profile_pExpDecay"],
        )

        # Add dynamics to x0 via public API
        file.add_time_dependence(
            target_model="single_glp",
            target_parameter="GLP_01_x0",
            dynamics_yaml="models/file_time.yaml",
            dynamics_model=["MonoExpPos"],
        )

        assert model.dim == 2

        # Should evaluate and differ across time (dynamics active)
        val_early = model.create_value_1d(t_ind=0, return_1d=1)
        val_late = model.create_value_1d(t_ind=30, return_1d=1)
        assert val_early.shape == file.energy.shape
        assert np.isfinite(val_early).all()
        assert np.isfinite(val_late).all()
        assert not np.allclose(val_early, val_late)


if __name__ == "__main__":
    pytest.main([__file__])
