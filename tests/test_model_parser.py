"""
Test parsing of models from YAML files passed by user
"""

import pytest
import numpy as np
from pathlib import Path
# local imports
from trspecfit import Project, File

#
# test class for 1D energy models
class TestEnergyParsing:
    
    #
    def setUp(self, model_info):
        """Setup function to create project, file, and load model"""
        project = Project(path='tests')
        file = File(parent_project=project)
        file.load_model(model_yaml='test_models_energy.yaml',
                        model_info=[model_info,],
                        debug=False)
        return file
    
    #
    def test_simple_energy_model(self):
        """
        Test simple energy model with unbound parameter [value, vary] and 
        standard parameter [value, vary, min, max] formats
        """
        
        # import the model
        file = self.setUp('simple_energy')
        
        # check the model
        assert file.model_active.name == 'simple_energy'
        assert file.model_active.dim == 1
        assert len(file.model_active.components) == 4

        # check the components
        # Offset (should not be numbered)
        assert file.model_active.components[0].fct_str == 'Offset'
        assert file.model_active.components[0].comp_name == 'Offset'
        assert file.model_active.components[0].par_dict['y0'] == [2, True, 0, 5]
        
        # Shirley (should not be numbered) - now uses [value, vary] format
        assert file.model_active.components[1].fct_str == 'Shirley'
        assert file.model_active.components[1].comp_name == 'Shirley'
        assert file.model_active.components[1].par_dict['pShirley'] == [400, False]
        # Check that lmfit parameter was created with unbounded min/max
        shirley_par = file.model_active.lmfit_pars['Shirley_pShirley']
        assert shirley_par.value == 400
        assert shirley_par.vary == False
        assert shirley_par.min == -np.inf
        assert shirley_par.max == np.inf
        
        # GLP_01 (should be numbered)
        assert file.model_active.components[2].fct_str == 'GLP'
        assert file.model_active.components[2].comp_name == 'GLP_01'
        assert file.model_active.components[2].N == 1
        assert file.model_active.components[2].par_dict['A'] == [20, True, 5, 25]
        assert file.model_active.components[2].par_dict['x0'] == [84.5, True, 82, 88]
        assert file.model_active.components[2].par_dict['F'] == [1.0, True, 0.75, 2.5]
        assert file.model_active.components[2].par_dict['m'] == [0.3, True, 0, 1]
        
        # GLP_02 (should be numbered) - x0 now uses [value, vary] format
        assert file.model_active.components[3].fct_str == 'GLP'
        assert file.model_active.components[3].comp_name == 'GLP_02'
        assert file.model_active.components[3].N == 2
        assert file.model_active.components[3].par_dict['A'] == [17, True, 5, 25]
        assert file.model_active.components[3].par_dict['x0'] == [88.1, True]
        # Check that lmfit parameter was created with unbounded min/max
        x0_par = file.model_active.lmfit_pars['GLP_02_x0']
        assert x0_par.value == 88.1
        assert x0_par.vary == True
        assert x0_par.min == -np.inf
        assert x0_par.max == np.inf
        assert file.model_active.components[3].par_dict['F'] == [1.0, True, 0.75, 2.5]
        assert file.model_active.components[3].par_dict['m'] == [0.3, True, 0, 1]
    
    #
    def test_energy_expression_model(self):
        """Test energy parameters with expressions"""

        # import the model
        file = self.setUp('energy_expression')
        
        # check the model
        assert file.model_active.name == 'energy_expression'
        assert file.model_active.dim == 1
        assert len(file.model_active.components) == 4
        
        # check components
        # Offset
        assert file.model_active.components[0].fct_str == 'Offset'
        assert file.model_active.components[0].comp_name == 'Offset'
        assert 'y0' in file.model_active.components[0].par_dict
        
        # Shirley
        assert file.model_active.components[1].fct_str == 'Shirley'
        assert file.model_active.components[1].comp_name == 'Shirley'
        assert file.model_active.components[1].par_dict['pShirley'] == [400, True, 1.0E-6, 1.0E+3]    
        
        # GLP_01
        assert file.model_active.components[2].fct_str == 'GLP'
        assert file.model_active.components[2].comp_name == 'GLP_01'
        assert file.model_active.components[2].N == 1
        assert file.model_active.components[2].par_dict['A'] == [20, True, 5, 25]
        assert file.model_active.components[2].par_dict['x0'] == [84.5, True, 82, 88]
        assert file.model_active.components[2].par_dict['F'] == [1.0, True, 0.75, 2.5]
        assert file.model_active.components[2].par_dict['m'] == [0.3, True, 0, 1]
        
        # GLP_02
        assert file.model_active.components[3].fct_str == 'GLP'
        assert file.model_active.components[3].comp_name == 'GLP_02'
        assert file.model_active.components[3].N == 2
        assert file.model_active.components[3].par_dict['A'] == ["3/4*GLP_01_A"]
        assert file.model_active.components[3].par_dict['x0'] == ["GLP_01_x0 +3.6"]
        assert file.model_active.components[3].par_dict['F'] == ["GLP_01_F"]
        assert file.model_active.components[3].par_dict['m'] == ["GLP_01_m"]

    #
    def test_energy_expression_fwd_ref_model(self):
        """Test energy parameters with forward reference expressions"""
        file = self.setUp('energy_expression_forward_reference')
        assert file.model_active.name == 'energy_expression_forward_reference'
        assert file.model_active.dim == 1
        assert len(file.model_active.components) == 4
        
        # check components
        # Offset
        assert file.model_active.components[0].fct_str == 'Offset'
        assert file.model_active.components[0].comp_name == 'Offset'
        assert 'y0' in file.model_active.components[0].par_dict
        
        # Shirley
        assert file.model_active.components[1].fct_str == 'Shirley'
        assert file.model_active.components[1].comp_name == 'Shirley'
        assert file.model_active.components[1].par_dict['pShirley'] == [400, True, 1.0E-6, 1.0E+3]    
        
        # GLP_01
        assert file.model_active.components[2].fct_str == 'GLP'
        assert file.model_active.components[2].comp_name == 'GLP_01'
        assert file.model_active.components[2].N == 1
        assert file.model_active.components[2].par_dict['A'] == ["3/4*GLP_02_A"]
        assert file.model_active.components[2].par_dict['x0'] == ["GLP_02_x0 +3.6"]
        assert file.model_active.components[2].par_dict['F'] == ["GLP_02_F"]
        assert file.model_active.components[2].par_dict['m'] == ["GLP_02_m"]
        
        # GLP_02
        assert file.model_active.components[3].fct_str == 'GLP'
        assert file.model_active.components[3].comp_name == 'GLP_02'
        assert file.model_active.components[3].N == 2
        assert file.model_active.components[3].par_dict['A'] == [20, True, 5, 25]
        assert file.model_active.components[3].par_dict['x0'] == [84.5, True, 82, 88]
        assert file.model_active.components[3].par_dict['F'] == [1.0, True, 0.75, 2.5]
        assert file.model_active.components[3].par_dict['m'] == [0.3, True, 0, 1]

#
# test class for 1D time models (mcp.Dynamics)
class TestTimeParsing:
    
    #
    def setUp(self, model_info):
        """Setup function to create project, file, and load model"""
        project = Project(path='tests')
        file = File(parent_project=project)
        file.time = np.linspace(-10, 100, 111) # needed for time-dependent models
        model = file.load_model(model_yaml='test_models_time.yaml',
                        model_info=[model_info,],
                        par_name='parTEST', # this is the name of the time-dependent parameter
                        debug=True)
        return model

    #
    def test_simple_time_model(self):
        """Test simple time model"""
        
        # import the model
        model = self.setUp('MonoExpPos')
        
        # check the model
        assert model.name == 'parTEST'
        assert model.dim == 1
        assert len(model.components) == 1

        # check the component
        assert model.components[0].fct_str == 'expFun'
        assert model.components[0].comp_name == 'expFun_01'
        assert model.components[0].par_dict['A'] == [1, True, 0, 5]
        assert model.components[0].par_dict['tau'] == [2.5, True, 1, 10]
        assert model.components[0].par_dict['t0'] == [0, False, 0, 1]
        assert model.components[0].par_dict['y0'] == [0, False, 0, 1]

    #
    def test_IRF_model(self):
        """Test IRF model"""
        
        # import the model
        model = self.setUp('MonoExpPosIRF')
        
        # check the model
        assert model.name == 'parTEST'
        assert model.dim == 1
        assert len(model.components) == 2

        # check the components
        assert model.components[0].fct_str == 'gaussCONV'
        assert model.components[0].comp_name == 'gaussCONV'
        assert model.components[0].par_dict['SD'] == [5.0E-2, True, 0, 1]
        
        assert model.components[1].fct_str == 'expFun'
        assert model.components[1].comp_name == 'expFun_01'
        assert model.components[1].par_dict['A'] == [1, True, 0, 5]
        assert model.components[1].par_dict['tau'] == [2.5, True, 1, 10]
        assert model.components[1].par_dict['t0'] == [0, False, 0, 1]
        assert model.components[1].par_dict['y0'] == [0, False, 0, 1]

#
# test class for 2D energy- and time-resolved models
class Test2DModelParsing:

    #
    def setUp(self, model_energy, par_name, model_time):
        """Setup function to create project, file, load model, and add dynamics"""
        project = Project(path='tests')
        file = File(parent_project=project)
        file.load_model(model_yaml='test_models_energy.yaml',
                        model_info=[model_energy,],
                        debug=False)
        file.time = np.linspace(-10, 100, 111) # needed for time-dependent models
        file.add_time_dependence(model_yaml='test_models_time.yaml',
                                 model_info=[model_time,],
                                 par_name=par_name)
        return file

    #
    def test_simple_2D_model(self):
        """Add IRF+exp_decay time-dependence to the simple energy model"""
        file = self.setUp(model_energy='simple_energy', par_name='GLP_01_x0', 
                         model_time='MonoExpPosIRF')
        
        # check the model
        assert file.model_active.name == 'simple_energy'
        assert file.model_active.dim == 2
        assert len(file.model_active.components) == 4
        assert file.model_active.components[0].fct_str == 'Offset'
        assert file.model_active.components[0].comp_name == 'Offset'
        assert file.model_active.components[0].par_dict['y0'] == [2, True, 0, 5]
        assert file.model_active.components[1].fct_str == 'Shirley'
        assert file.model_active.components[1].comp_name == 'Shirley'
        # Updated to check [value, vary] format
        assert file.model_active.components[1].par_dict['pShirley'] == [400, False]
        # GLP_01
        assert file.model_active.components[2].fct_str == 'GLP'
        assert file.model_active.components[2].comp_name == 'GLP_01'
        assert file.model_active.components[2].par_dict['A'] == [20, True, 5, 25]
        assert file.model_active.components[2].par_dict['x0'] == [84.5, True, 82, 88]
        # x0 is the time-dependent parameter
        td_par_model = file.model_active.components[2].pars[1].t_model
        assert td_par_model.components[0].comp_name == 'gaussCONV'
        assert td_par_model.components[0].par_dict['SD'] == [5.0E-2, True, 0, 1] 
        assert td_par_model.components[1].fct_str == 'expFun'
        assert td_par_model.components[1].comp_name == 'expFun_01'
        assert td_par_model.components[1].par_dict['A'] == [1, True, 0, 5]
        assert td_par_model.components[1].par_dict['tau'] == [2.5, True, 1, 10]
        assert td_par_model.components[1].par_dict['t0'] == [0, False, 0, 1]
        assert td_par_model.components[1].par_dict['y0'] == [0, False, 0, 1]
        # end of time-dependent parameter model
        assert file.model_active.components[2].par_dict['F'] == [1.0, True, 0.75, 2.5]
        assert file.model_active.components[2].par_dict['m'] == [0.3, True, 0, 1]
        # GLP_02 - x0 now uses [value, vary] format
        assert file.model_active.components[3].fct_str == 'GLP'
        assert file.model_active.components[3].comp_name == 'GLP_02'
        assert file.model_active.components[3].par_dict['A'] == [17, True, 5, 25]
        assert file.model_active.components[3].par_dict['x0'] == [88.1, True]
        assert file.model_active.components[3].par_dict['F'] == [1.0, True, 0.75, 2.5]
        assert file.model_active.components[3].par_dict['m'] == [0.3, True, 0, 1]
        
    #
    def test_time_dependence_on_expression_parameter_raises(self):
        """Adding dynamics to expression-linked parameter should fail."""
        project = Project(path='tests')
        file = File(parent_project=project)
        file.load_model(
            model_yaml='test_models_energy.yaml',
            model_info=['energy_expression'],
            debug=False
        )
        file.time = np.linspace(-10, 100, 111)

        with pytest.raises(ValueError, match='Cannot add time dependence to expression parameter'):
            file.add_time_dependence(
                model_yaml='test_models_time.yaml',
                model_info=['MonoExpPosIRF'],
                par_name='GLP_02_x0'
            )


if __name__ == '__main__':
    pytest.main([__file__])
