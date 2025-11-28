"""
Test parameter sweep functionality for ML dataset generation.
"""

import pytest
import numpy as np
import h5py
import tempfile
from pathlib import Path

from trspecfit.utils.sweep import ParameterSweep
from trspecfit import Simulator, Project, File

#
#
class TestParameterSweep:
    """Test ParameterSweep class"""
    
    #
    def test_creation(self):
        """Test basic sweep creation"""
        sweep = ParameterSweep(strategy='auto', seed=42)
        assert sweep.strategy == 'auto'
        assert sweep.seed == 42
        assert sweep.parameter_specs == {}
    
    #
    def test_invalid_strategy(self):
        """Test that invalid strategy raises error"""
        with pytest.raises(ValueError, match="strategy must be one of"):
            ParameterSweep(strategy='invalid')
    
    #
    def test_add_range(self):
        """Test adding discrete parameter range"""
        sweep = ParameterSweep()
        sweep.add_range('param_A', [1, 2, 3])
        
        assert 'param_A' in sweep.parameter_specs
        assert sweep.parameter_specs['param_A']['type'] == 'range'
        assert len(sweep.parameter_specs['param_A']['values']) == 3
    
    #
    def test_add_uniform(self):
        """Test adding uniform distribution parameter"""
        sweep = ParameterSweep()
        sweep.add_uniform('param_B', min_val=0, max_val=10, n_samples=5)
        
        assert 'param_B' in sweep.parameter_specs
        assert sweep.parameter_specs['param_B']['type'] == 'uniform'
        assert sweep.parameter_specs['param_B']['min'] == 0
        assert sweep.parameter_specs['param_B']['max'] == 10
        assert sweep.parameter_specs['param_B']['n_samples'] == 5
    
    #
    def test_add_normal(self):
        """Test adding normal distribution parameter"""
        sweep = ParameterSweep()
        sweep.add_normal('param_C', mean=5, std=1, n_samples=10)
        
        assert 'param_C' in sweep.parameter_specs
        assert sweep.parameter_specs['param_C']['type'] == 'normal'
        assert sweep.parameter_specs['param_C']['mean'] == 5
        assert sweep.parameter_specs['param_C']['std'] == 1
    
    #
    def test_add_lognormal(self):
        """Test adding lognormal distribution parameter"""
        sweep = ParameterSweep()
        sweep.add_lognormal('param_D', mean=2, std=0.5, n_samples=10)
        
        assert 'param_D' in sweep.parameter_specs
        assert sweep.parameter_specs['param_D']['type'] == 'lognormal'
    
    #
    def test_grid_strategy_detection(self):
        """Test auto strategy detects grid for all discrete parameters"""
        sweep = ParameterSweep(strategy='auto', seed=42)
        sweep.add_range('param_A', [1, 2, 3])
        sweep.add_range('param_B', [10, 20])
        
        assert sweep._determine_strategy() == 'grid'
    
    #
    def test_random_strategy_detection(self):
        """Test auto strategy detects random for mixed parameters"""
        sweep = ParameterSweep(strategy='auto', seed=42)
        sweep.add_range('param_A', [1, 2, 3])
        sweep.add_uniform('param_B', 0, 10, n_samples=5)
        
        assert sweep._determine_strategy() == 'random'
    
    #
    def test_grid_generation_discrete_only(self):
        """Test grid generation with only discrete parameters"""
        sweep = ParameterSweep(strategy='grid', seed=42)
        sweep.add_range('param_A', [1, 2])
        sweep.add_range('param_B', [10, 20, 30])
        
        configs = list(sweep)
        
        assert len(configs) == 6  # 2 × 3 = 6
        assert configs[0] == {'param_A': 1, 'param_B': 10}
        assert configs[-1] == {'param_A': 2, 'param_B': 30}
    
    #
    def test_grid_generation_with_continuous(self):
        """Test grid generation with continuous distributions"""
        sweep = ParameterSweep(strategy='grid', seed=42)
        sweep.add_range('param_A', [1, 2])
        sweep.add_uniform('param_B', 0, 10, n_samples=3)
        
        configs = list(sweep)
        
        assert len(configs) == 6  # 2 × 3 = 6
        # All param_A values should be 1 or 2
        assert all(c['param_A'] in [1, 2] for c in configs)
        # All param_B values should be unique (sampled)
        param_b_values = [c['param_B'] for c in configs]
        assert len(set(param_b_values)) == 3  # 3 unique samples
    
    #
    def test_random_generation(self):
        """Test random sampling generation"""
        sweep = ParameterSweep(strategy='random', seed=42)
        sweep.add_range('param_A', [1, 2, 3])
        sweep.add_uniform('param_B', 0, 10, n_samples=10)
        
        configs = list(sweep)
        
        assert len(configs) == 10
        # All param_A values should be from [1, 2, 3]
        assert all(c['param_A'] in [1, 2, 3] for c in configs)
        # All param_B values should be in range [0, 10]
        assert all(0 <= c['param_B'] <= 10 for c in configs)
    
    #
    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results"""
        sweep1 = ParameterSweep(strategy='random', seed=42)
        sweep1.add_uniform('param_A', 0, 10, n_samples=5)
        
        sweep2 = ParameterSweep(strategy='random', seed=42)
        sweep2.add_uniform('param_A', 0, 10, n_samples=5)
        
        configs1 = list(sweep1)
        configs2 = list(sweep2)
        
        assert configs1 == configs2
    
    #
    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results"""
        sweep1 = ParameterSweep(strategy='random', seed=42)
        sweep1.add_uniform('param_A', 0, 10, n_samples=5)
        
        sweep2 = ParameterSweep(strategy='random', seed=123)
        sweep2.add_uniform('param_A', 0, 10, n_samples=5)
        
        configs1 = list(sweep1)
        configs2 = list(sweep2)
        
        assert configs1 != configs2
    
    #
    def test_get_n_configs_grid(self):
        """Test config count calculation for grid strategy"""
        sweep = ParameterSweep(strategy='grid')
        sweep.add_range('param_A', [1, 2, 3])
        sweep.add_range('param_B', [10, 20])
        
        assert sweep.get_n_configs() == 6  # 3 × 2
    
    #
    def test_get_n_configs_random(self):
        """Test config count calculation for random strategy"""
        sweep = ParameterSweep(strategy='random')
        sweep.add_uniform('param_A', 0, 10, n_samples=15)
        sweep.add_normal('param_B', 5, 1, n_samples=10)
        
        assert sweep.get_n_configs() == 15  # max(15, 10)
    
    #
    def test_iteration_multiple_times(self):
        """Test that sweep can be iterated multiple times"""
        sweep = ParameterSweep(strategy='random', seed=42)
        sweep.add_uniform('param_A', 0, 10, n_samples=5)
        
        configs1 = list(sweep)
        configs2 = list(sweep)
        
        # Should produce same results each time due to seed reset
        assert configs1 == configs2
        assert len(configs1) == 5

#
#
class TestSimulatorParameterSweep:
    """Test Simulator.simulate_parameter_sweep() integration"""
    #
    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing"""
        project = Project(path='tests', name='test')
        file = File(
            parent_project=project,
            energy=np.arange(0, 20, 0.5),  # Coarse for speed
            time=np.arange(-10, 100, 5)  # Coarse for speed
        )
        file.load_model(
            model_yaml='test_models_energy.yaml',
            model_info=['simple_energy']
        )
        file.add_time_dependence(
            model_yaml='test_models_time.yaml',
            model_info=['MonoExpPosIRF'],
            par_name='GLP_01_x0'
        )
        return file.model_active
    
    #
    def test_simulate_parameter_sweep_basic(self, simple_model):
        """Test basic parameter sweep simulation"""
        sweep = ParameterSweep(strategy='grid', seed=42)
        sweep.add_range('GLP_01_A', [15, 20])
        sweep.add_range('GLP_01_x0', [8, 10])
        
        sim = Simulator(
            model=simple_model,
            detection='analog',
            noise_level=0.05,
            seed=42
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test_sweep.h5'
            
            sim.simulate_parameter_sweep(
                parameter_sweep=sweep,
                N_realizations=2,
                filepath=str(filepath),
                show_progress=False
            )
            
            # Check file was created
            assert filepath.exists()
            
            # Check HDF5 structure
            with h5py.File(filepath, 'r') as f:
                assert 'energy' in f
                assert 'time' in f
                assert 'metadata' in f
                assert 'parameter_configs' in f
                assert 'simulated_data' in f
                
                # Check metadata
                assert f['metadata'].attrs['n_configs'] == 4  # 2 × 2
                assert f['metadata'].attrs['n_realizations_per_config'] == 2
                assert f['metadata'].attrs['total_datasets'] == 8  # 4 × 2
    
    #
    def test_hdf5_structure(self, simple_model):
        """Test detailed HDF5 file structure"""
        sweep = ParameterSweep(strategy='grid', seed=42)
        sweep.add_range('GLP_01_A', [15, 20])
        
        sim = Simulator(model=simple_model, detection='analog', 
                       noise_level=0.05, seed=42)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test_structure.h5'
            
            sim.simulate_parameter_sweep(
                parameter_sweep=sweep,
                N_realizations=3,
                filepath=str(filepath),
                show_progress=False
            )
            
            with h5py.File(filepath, 'r') as f:
                # Check axes
                assert f['energy'].shape[0] > 0
                assert f['time'].shape[0] > 0
                
                # Check config structure
                assert 'config_000000' in f['parameter_configs']
                assert 'config_000001' in f['parameter_configs']
                
                # Check config attributes
                config = f['parameter_configs']['config_000000']
                assert 'GLP_01_A' in config.attrs
                assert 'all_parameters' in config.attrs
                
                # Check clean data exists
                assert 'clean' in config
                assert config['clean'].shape == (len(f['time'][:]), len(f['energy'][:]))
                
                # Check noisy data structure
                data_group = f['simulated_data']['config_000000']
                assert '000000' in data_group
                assert '000001' in data_group
                assert '000002' in data_group
                assert len(data_group.keys()) == 3
    
    #
    def test_parameter_values_in_hdf5(self, simple_model):
        """Test that swept parameter values are correctly stored"""
        sweep = ParameterSweep(strategy='grid', seed=42)
        test_values = [15, 20, 25]
        sweep.add_range('GLP_01_A', test_values)
        
        sim = Simulator(model=simple_model, detection='analog',
                       noise_level=0.05, seed=42)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test_params.h5'
            
            sim.simulate_parameter_sweep(
                parameter_sweep=sweep,
                N_realizations=2,
                filepath=str(filepath),
                show_progress=False
            )
            
            with h5py.File(filepath, 'r') as f:
                # Check that all test values appear in configs
                stored_values = []
                for i in range(len(test_values)):
                    config_name = f'config_{i:06d}'
                    stored_values.append(
                        f['parameter_configs'][config_name].attrs['GLP_01_A']
                    )
                
                assert sorted(stored_values) == sorted(test_values)
    
    #
    def test_random_strategy_sweep(self, simple_model):
        """Test parameter sweep with random strategy"""
        sweep = ParameterSweep(strategy='random', seed=42)
        sweep.add_uniform('GLP_01_A', 10, 25, n_samples=5)
        
        sim = Simulator(model=simple_model, detection='analog',
                       noise_level=0.05, seed=42)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test_random.h5'
            
            sim.simulate_parameter_sweep(
                parameter_sweep=sweep,
                N_realizations=2,
                filepath=str(filepath),
                show_progress=False
            )
            
            with h5py.File(filepath, 'r') as f:
                assert f['metadata'].attrs['n_configs'] == 5
                
                # Check that parameter values are in expected range
                for i in range(5):
                    config_name = f'config_{i:06d}'
                    value = f['parameter_configs'][config_name].attrs['GLP_01_A']
                    assert 10 <= value <= 25
    
    #
    def test_detector_settings_in_metadata(self, simple_model):
        """Test that detector settings are stored in metadata"""
        sweep = ParameterSweep(strategy='grid', seed=42)
        sweep.add_range('GLP_01_A', [15, 20])
        
        # Test analog detector
        sim_analog = Simulator(
            model=simple_model,
            detection='analog',
            noise_level=0.08,
            noise_type='gaussian',
            seed=42
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test_analog.h5'
            
            sim_analog.simulate_parameter_sweep(
                parameter_sweep=sweep,
                N_realizations=1,
                filepath=str(filepath),
                show_progress=False
            )
            
            with h5py.File(filepath, 'r') as f:
                assert f['metadata'].attrs['detection'] == 'analog'
                assert f['metadata'].attrs['noise_level'] == 0.08
                assert f['metadata'].attrs['noise_type'] == 'gaussian'
        
        # Test photon counting detector
        sim_photon = Simulator(
            model=simple_model,
            detection='photon_counting',
            counts_per_cycle=5000,
            seed=42
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test_photon.h5'
            
            sim_photon.simulate_parameter_sweep(
                parameter_sweep=sweep,
                N_realizations=1,
                filepath=str(filepath),
                show_progress=False
            )
            
            with h5py.File(filepath, 'r') as f:
                assert f['metadata'].attrs['detection'] == 'photon_counting'
                assert f['metadata'].attrs['counts_per_cycle'] == 5000


if __name__ == '__main__':
    pytest.main([__file__, '-v'])