"""
Test MCP (Model/Component/Parameter) library functionality
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import copy
from trspecfit import Project, File
from trspecfit.mcp import Model, Component, Par
from trspecfit.functions import energy as fcts_energy
from trspecfit.functions import time as fcts_time

#
#
class TestMCPModel:
    """Test MCP Model class functionality"""
    
    #
    def test_model_creation(self):
        """Test basic model creation and initialization"""
        model = Model('test_model')
        
        assert model.name == 'test_model'
        assert model.components == []
        assert model.lmfit_pars is not None
        assert model.dim is None
        assert model.energy is None
        assert model.time is None
    
    #
    def test_model_with_components(self):
        """Test model with spectral components (Au4f test from notebook)"""
        # Initialize 2D fit model
        mod2D = Model('Au4f_test')
        mod2D.energy = np.arange(75, 95, 0.1)[::-1]
        mod2D.time = np.arange(-500, 2500, 10)
        
        # Define Shirley background
        c_Shirley = Component('Shirley')
        c_Shirley.add_pars({'pShirley': [2500, True, 1E-6, 1E6]})
        
        # Define Offset parameters
        c_Offset = Component('Offset')
        c_Offset.add_pars({'y0': [3, True, 0, 5]})
        
        # Define peak components
        c_peak1 = Component('GLP')
        c_peak1.add_pars({
            'A': [16, True, 5, 25],
            'x0': [84.0, True, 81, 87],
            'F': [1.6, True, 1, 2.5],
            'm': [0.3, False, 0, 1]
        })
        
        c_peak2 = Component('GLP')
        c_peak2.add_pars({
            'A': [12, True, 1, 20],
            'x0': [87.6, True, 84, 90],
            'F': [1.6, True, 1, 2.5],
            'm': [0.3, False, 0, 1]
        })
        
        # Add components to model
        mod2D.add_components([c_Offset, c_Shirley, c_peak1, c_peak2])
        
        # Check model structure
        assert mod2D.name == 'Au4f_test'
        assert len(mod2D.components) == 4
        assert mod2D.components[0].fct_str == 'Offset'
        assert mod2D.components[1].fct_str == 'Shirley'
        assert mod2D.components[2].fct_str == 'GLP'
        assert mod2D.components[3].fct_str == 'GLP'
    
    #$% parameter distributions to be implemented 
    # def test_model_parameter_distribution(self):
    #     """Test parameter distribution functionality"""
    #     # Create a simple model for testing
    #     model = Model('test_dist')
    #     model.energy = np.linspace(80, 90, 100)
    #     model.time = np.linspace(0, 10, 50)
        
    #     # Add a GLP component
    #     c_peak = Component('GLP')
    #     c_peak.add_pars({
    #         'A': [10, True, 5, 15],
    #         'x0': [85, True, 80, 90],
    #         'F': [1.5, True, 1, 2],
    #         'm': [0.3, False, 0, 1]
    #     })
    #     model.add_components([c_peak])
        
    #     # Test parameter distribution
    #     # ...
    #     assert bla bla

#
#
class TestMCPComponent:
    """Test MCP Component class functionality"""
    
    #
    def test_component_creation(self):
        """Test basic component creation"""
        comp = Component('GLP')
        
        assert comp.fct_str == 'GLP'
        assert comp.N is None  # Not numbered initially
        assert comp.par_dict == {}
        assert comp.pars == []
    
    #
    def test_numbered_component(self):
        """Test numbered component creation"""
        comp = Component('GLP_01')
        
        assert comp.fct_str == 'GLP'
        assert comp.N == 1
        assert comp.comp_name == 'GLP_01'
    
    #
    def test_component_parameter_management(self):
        """Test component parameter addition and management"""
        comp = Component('GLP')
        comp.add_pars({
            'A': [20, True, 5, 25],
            'x0': [84.5, True, 82, 88],
            'F': [1.0, True, 0.75, 2.5],
            'm': [0.3, True, 0, 1]
        })
        
        assert comp.par_dict['A'] == [20, True, 5, 25]
        assert comp.par_dict['x0'] == [84.5, True, 82, 88]
        assert comp.par_dict['F'] == [1.0, True, 0.75, 2.5]
        assert comp.par_dict['m'] == [0.3, True, 0, 1]
    
    #
    def test_component_prefix_handling(self):
        """Test component prefix handling when component number changes"""
        comp = Component('GLP')
        comp.add_pars({
            'A': [20, True, 5, 25],
            'x0': [84.5, True, 82, 88]
        })
        
        # Update component number and name
        comp.N = 8
        comp.comp_name = 'GLP_08'
        assert comp.prefix == 'GLP_08_'
        assert comp.comp_name == 'GLP_08'
    
    #
    def test_component_creation_with_energy_time(self):
        """Test component creation with energy and time axes"""
        comp = Component('GLP')
        comp.add_pars({
            'A': [20, True, 5, 25],
            'x0': [84.5, True, 82, 88],
            'F': [1.0, True, 0.75, 2.5],
            'm': [0.3, False, 0, 1]
        })
        
        # Set energy and time axes
        comp.energy = np.linspace(80, 90, 100)
        comp.time = np.linspace(0, 10, 50)
        
        # Create parameters
        comp.create_pars()
        
        assert len(comp.pars) == 4
        assert comp.pars[0].name == 'GLP_A'
        assert comp.pars[1].name == 'GLP_x0'
        assert comp.pars[2].name == 'GLP_F'
        assert comp.pars[3].name == 'GLP_m'

#
#
class TestMCPParameter:
    """Test MCP Parameter class functionality"""
    
    #
    def test_parameter_creation(self):
        """Test basic parameter creation"""
        par = Par('test_param')
        
        assert par.name == 'test_param'
        assert par.info == []
        assert par.t_vary == False
        assert par.t_model is not None
    
    #
    def test_parameter_with_info(self):
        """Test parameter creation with parameter info"""
        par = Par('test_param', [87.6, True, 84, 90])
        
        assert par.name == 'test_param'
        assert par.info == [87.6, True, 84, 90]
    
    #
    def test_parameter_creation_with_expression(self):
        """Test parameter creation with expression"""
        par = Par('test_param', ['GLP_01_A * 0.75'])
        
        assert par.name == 'test_param'
        assert par.info == ['GLP_01_A * 0.75']
    
    #
    def test_parameter_lmfit_creation(self):
        """Test parameter lmfit object creation"""
        par = Par('test_param', [87.6, True, 84, 90])
        par.create()
        
        assert par.lmfit_par is not None
        assert 'test_param' in par.lmfit_par
        assert par.lmfit_par['test_param'].value == 87.6
        assert par.lmfit_par['test_param'].vary == True
        assert par.lmfit_par['test_param'].min == 84
        assert par.lmfit_par['test_param'].max == 90

#
#
class TestMCPDynamics:
    """Test MCP Dynamics (time-dependent) functionality"""
    
    #
    def test_dynamics_model_creation(self):
        """Test creation of time-dependence model"""
        # Initialize dynamics model
        t_mod = Model('GLP_01_x0')
        t_mod.time = np.linspace(-10, 100, 111)
        
        # Define instrument response function
        c_IRF = Component('gaussCONV', fcts_time)
        c_IRF.add_pars({'SD': [80, True, 0, 1E4]})
        
        # Define decay components
        c_tD1 = Component('expFun', fcts_time)
        c_tD1.add_pars({
            'A': [2, True, 1, 1E2],
            'tau': [5000, True, 1E3, 1E4],
            't0': [0, False, 0, 1],
            'y0': [0, False, 0, 1]
        })
        
        c_tD2 = Component('expFun', fcts_time)
        c_tD2.add_pars({
            'A': [5, True, 1, 1E2],
            'tau': [1250, True, 1E2, 1E3],
            't0': [0, False, 0, 1],
            'y0': [0, False, 0, 1]
        })
        
        # Add components to dynamics model
        t_mod.add_components([c_IRF, c_tD1, c_tD2])
        
        # Check model structure
        assert t_mod.name == 'GLP_01_x0'
        assert len(t_mod.components) == 3
        assert t_mod.components[0].fct_str == 'gaussCONV'
        assert t_mod.components[1].fct_str == 'expFun'
        assert t_mod.components[2].fct_str == 'expFun'
    
    #
    def test_dynamics_parameter_handling(self):
        """Test parameter handling in dynamics models"""
        # Create a simple dynamics model
        t_mod = Model('test_dynamics')
        t_mod.time = np.linspace(0, 10, 100)
        
        # Add a simple exponential component
        c_exp = Component('expFun', fcts_time)
        c_exp.add_pars({
            'A': [1, True, 0, 5],
            'tau': [2.5, True, 1, 10],
            't0': [0, False, 0, 1],
            'y0': [0, False, 0, 1]
        })
        
        t_mod.add_components([c_exp])
        
        # Check parameter structure
        assert len(t_mod.components) == 1
        assert t_mod.components[0].par_dict['A'] == [1, True, 0, 5]
        assert t_mod.components[0].par_dict['tau'] == [2.5, True, 1, 10]

#
#
class TestMCPIntegration:
    """Test MCP integration with 2D models"""
    
    #
    def test_2d_model_with_dynamics(self):
        """Test 2D model with time-dependent parameters"""
        # Create 2D model
        mod2D = Model('Au4f_2D')
        mod2D.energy = np.linspace(80, 90, 100)
        mod2D.time = np.linspace(0, 10, 50)
        
        # Add spectral components - need to use numbered components for dynamics
        c_Offset = Component('Offset')
        c_Offset.add_pars({'y0': [3, True, 0, 5]})
        
        c_peak = Component('GLP_01')  # Use numbered component name
        c_peak.add_pars({
            'A': [16, True, 5, 25],
            'x0': [84.0, True, 81, 87],
            'F': [1.6, True, 1, 2.5],
            'm': [0.3, False, 0, 1]
        })
        
        mod2D.add_components([c_Offset, c_peak])
        
        # Create dynamics model for x0 parameter
        t_mod = Model('GLP_01_x0')
        t_mod.time = mod2D.time
        
        c_IRF = Component('gaussCONV', fcts_time)
        c_IRF.add_pars({'SD': [80, True, 0, 1E4]})  # Use SD parameter name
        
        c_exp = Component('expFun', fcts_time)
        c_exp.add_pars({
            'A': [2, True, 1, 1E2],
            'tau': [5000, True, 1E3, 1E4],
            't0': [0, False, 0, 1],
            'y0': [0, False, 0, 1]
        })
        
        t_mod.add_components([c_IRF, c_exp])
        
        # Add dynamics to 2D model
        mod2D.add_dynamics(t_mod)
        
        # Check integration
        assert mod2D.name == 'Au4f_2D'
        assert len(mod2D.components) == 2
        # The x0 parameter should now have time dependence
        x0_param = mod2D.components[1].pars[1]  # x0 is the second parameter
        assert x0_param.t_vary == True
        assert x0_param.t_model is not None
    
    #
    def test_parameter_value_updates(self):
        """Test parameter value updates during fitting"""
        # Create a simple model
        model = Model('test_updates')
        model.energy = np.linspace(80, 90, 100)
        model.time = np.linspace(0, 10, 50)
        
        # Add a component with wider bounds to avoid clipping
        c_peak = Component('GLP')
        c_peak.add_pars({
            'A': [10, True, 5, 20],  # Wider bounds
            'x0': [85, True, 80, 95],  # Wider bounds
            'F': [1.5, True, 1, 5],  # Wider bounds
            'm': [0.3, False, 0, 1]  # This parameter is fixed
        })
        
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
class TestEnergyFunctions:
    """Test energy/spectral function functionality"""
    
    #
    def test_energy_function_imports(self):
        """Test that energy functions can be imported and used"""
        # Test that we can access energy functions
        assert hasattr(fcts_energy, 'Offset')
        assert hasattr(fcts_energy, 'Shirley')
        assert hasattr(fcts_energy, 'GLP')
    
    #
    def test_energy_function_evaluation(self):
        """Test energy function evaluation"""
        # Create test energy array
        e = np.arange(150, 180, 0.1)
        
        # Test Offset function - requires spectrum parameter
        spectrum = np.zeros_like(e)
        y_offset = fcts_energy.Offset(e, y0=2, spectrum=spectrum)
        assert len(y_offset) == len(e)
        assert np.allclose(y_offset, 2)
        
        # Test GLP function
        y_glp = fcts_energy.GLP(e, A=20, x0=165, F=1.5, m=0.3)
        assert len(y_glp) == len(e)
        assert np.max(y_glp) > 0  # Should have a peak

#
#
class TestTimeFunctions:
    """Test time/dynamics function functionality"""
    
    #
    def test_time_function_imports(self):
        """Test that time functions can be imported and used"""
        # Test that we can access time functions
        assert hasattr(fcts_time, 'expFun')
        assert hasattr(fcts_time, 'linFun')
        assert hasattr(fcts_time, 'gaussCONV')
    
    #
    def test_time_function_evaluation(self):
        """Test time function evaluation"""
        # Create test time array
        t = np.arange(-50, 200, 0.1)
        
        # Test exponential function
        y_exp = fcts_time.expFun(t, A=12, tau=20, t0=10, y0=3)
        assert len(y_exp) == len(t)
        assert np.max(y_exp) > 0
        
        # Test linear function
        y_lin = fcts_time.linFun(t, m=-5E-4, t0=0, y0=-3)
        assert len(y_lin) == len(t)
        
        # Test Gaussian convolution - uses SD parameter
        y_gauss = fcts_time.gaussCONV(t, SD=5)
        assert len(y_gauss) == len(t)

#
#
class TestMCPNormalization:
    """Test MCP time normalization functionality"""
    
    #
    def test_time_normalization(self):
        """Test time normalization for multi-cycle dynamics"""
        # Create a dynamics model with frequency
        from trspecfit.mcp import Dynamics
        t_mod = Dynamics('test_normalization')
        t_mod.time = np.linspace(0, 100, 1000)
        t_mod.subcycles = 3
        
        # Test that we can set frequency
        t_mod.set_frequency(frequency=0.1, time_unit=0)
        # Test that normalization attributes are created
        assert hasattr(t_mod, 'time_norm')
        assert hasattr(t_mod, 'N_sub')
        assert hasattr(t_mod, 'N_counter')
        
        # Test that normalized time is calculated
        if hasattr(t_mod, 'time_norm'):
            assert len(t_mod.time_norm) == len(t_mod.time)
    
    #
    def test_subcycle_handling(self):
        """Test subcycle handling in components"""
        # Create a component with subcycle
        comp = Component('expFun', fcts_time, comp_subcycle=1)
        comp.subcycle = 1
        
        assert comp.subcycle == 1
        assert comp.comp_type == 'add'  # Default for non-background functions


if __name__ == '__main__':
    pytest.main([__file__])