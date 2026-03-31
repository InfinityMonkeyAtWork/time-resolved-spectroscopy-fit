"""
Test parameter sweep functionality for ML dataset generation.
"""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from trspecfit import File, Project, Simulator
from trspecfit.utils.sweep import ParameterSweep, SweepDataset


#
#
class TestParameterSweep:
    """Test ParameterSweep class"""

    #
    def test_creation(self):
        """Test basic sweep creation"""

        sweep = ParameterSweep(strategy="auto", seed=42)
        assert sweep.strategy == "auto"
        assert sweep.seed == 42
        assert sweep.parameter_specs == {}

    #
    def test_invalid_strategy(self):
        """Test that invalid strategy raises error"""

        with pytest.raises(ValueError, match="strategy must be one of"):
            ParameterSweep(strategy="invalid")

    #
    def test_add_range(self):
        """Test adding discrete parameter range"""

        sweep = ParameterSweep()
        sweep.add_range("param_A", [1, 2, 3])

        assert "param_A" in sweep.parameter_specs
        assert sweep.parameter_specs["param_A"]["type"] == "range"
        assert len(sweep.parameter_specs["param_A"]["values"]) == 3

    #
    def test_add_uniform(self):
        """Test adding uniform distribution parameter"""

        sweep = ParameterSweep()
        sweep.add_uniform("param_B", min_val=0, max_val=10, n_samples=5)

        assert "param_B" in sweep.parameter_specs
        assert sweep.parameter_specs["param_B"]["type"] == "uniform"
        assert sweep.parameter_specs["param_B"]["min"] == 0
        assert sweep.parameter_specs["param_B"]["max"] == 10
        assert sweep.parameter_specs["param_B"]["n_samples"] == 5

    #
    def test_add_normal(self):
        """Test adding normal distribution parameter"""

        sweep = ParameterSweep()
        sweep.add_normal("param_C", mean=5, std=1, n_samples=10)

        assert "param_C" in sweep.parameter_specs
        assert sweep.parameter_specs["param_C"]["type"] == "normal"
        assert sweep.parameter_specs["param_C"]["mean"] == 5
        assert sweep.parameter_specs["param_C"]["std"] == 1

    #
    def test_add_lognormal(self):
        """Test adding lognormal distribution parameter"""

        sweep = ParameterSweep()
        sweep.add_lognormal("param_D", mean=2, std=0.5, n_samples=10)

        assert "param_D" in sweep.parameter_specs
        assert sweep.parameter_specs["param_D"]["type"] == "lognormal"

    #
    def test_auto_strategy_all_discrete_produces_grid(self):
        """Auto strategy with all discrete params should produce cartesian product."""

        sweep = ParameterSweep(strategy="auto", seed=42)
        sweep.add_range("param_A", [1, 2, 3])
        sweep.add_range("param_B", [10, 20])

        configs = list(sweep)
        # Grid: 3 × 2 = 6 combinations
        assert len(configs) == 6
        assert configs[0] == {"param_A": 1, "param_B": 10}

    #
    def test_auto_strategy_mixed_produces_random(self):
        """Auto strategy with continuous params should produce n_samples configs."""

        sweep = ParameterSweep(strategy="auto", seed=42)
        sweep.add_range("param_A", [1, 2, 3])
        sweep.add_uniform("param_B", 0, 10, n_samples=5)

        configs = list(sweep)
        # Random mode: max(n_samples) = 5, not grid's 3 × 5 = 15
        assert len(configs) == 5
        # Continuous draws should spread across [0, 10]
        param_B_values = [c["param_B"] for c in configs]
        assert len(set(param_B_values)) >= 3
        assert all(0 <= v <= 10 for v in param_B_values)

    #
    def test_grid_generation_discrete_only(self):
        """Test grid generation with only discrete parameters"""

        sweep = ParameterSweep(strategy="grid", seed=42)
        sweep.add_range("param_A", [1, 2])
        sweep.add_range("param_B", [10, 20, 30])

        configs = list(sweep)

        assert len(configs) == 6  # 2 × 3 = 6
        assert configs[0] == {"param_A": 1, "param_B": 10}
        assert configs[-1] == {"param_A": 2, "param_B": 30}

    #
    def test_grid_generation_with_continuous(self):
        """Test grid generation with continuous distributions"""

        sweep = ParameterSweep(strategy="grid", seed=42)
        sweep.add_range("param_A", [1, 2])
        sweep.add_uniform("param_B", 0, 10, n_samples=3)

        configs = list(sweep)

        assert len(configs) == 6  # 2 × 3 = 6
        # All param_A values should be 1 or 2
        assert all(c["param_A"] in [1, 2] for c in configs)
        # All param_B values should be unique (sampled)
        param_b_values = [c["param_B"] for c in configs]
        assert len(set(param_b_values)) == 3  # 3 unique samples

    #
    def test_random_generation(self):
        """Test random sampling generation"""

        sweep = ParameterSweep(strategy="random", seed=42)
        sweep.add_range("param_A", [1, 2, 3])
        sweep.add_uniform("param_B", 0, 10, n_samples=10)

        configs = list(sweep)

        assert len(configs) == 10
        # All param_A values should be from [1, 2, 3]
        assert all(c["param_A"] in [1, 2, 3] for c in configs)
        # All param_B values should be in range [0, 10]
        assert all(0 <= c["param_B"] <= 10 for c in configs)

    #
    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results"""

        sweep1 = ParameterSweep(strategy="random", seed=42)
        sweep1.add_uniform("param_A", 0, 10, n_samples=5)

        sweep2 = ParameterSweep(strategy="random", seed=42)
        sweep2.add_uniform("param_A", 0, 10, n_samples=5)

        configs1 = list(sweep1)
        configs2 = list(sweep2)

        assert configs1 == configs2

    #
    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results"""

        sweep1 = ParameterSweep(strategy="random", seed=42)
        sweep1.add_uniform("param_A", 0, 10, n_samples=5)

        sweep2 = ParameterSweep(strategy="random", seed=123)
        sweep2.add_uniform("param_A", 0, 10, n_samples=5)

        configs1 = list(sweep1)
        configs2 = list(sweep2)

        assert configs1 != configs2

    #
    def test_get_n_configs_grid(self):
        """Test config count calculation for grid strategy"""

        sweep = ParameterSweep(strategy="grid")
        sweep.add_range("param_A", [1, 2, 3])
        sweep.add_range("param_B", [10, 20])

        assert sweep.get_n_configs() == 6  # 3 × 2

    #
    def test_get_n_configs_random(self):
        """Test config count calculation for random strategy"""

        sweep = ParameterSweep(strategy="random")
        sweep.add_uniform("param_A", 0, 10, n_samples=15)
        sweep.add_normal("param_B", 5, 1, n_samples=10)

        assert sweep.get_n_configs() == 15  # max(15, 10)

    #
    def test_iteration_multiple_times(self):
        """Test that sweep can be iterated multiple times"""

        sweep = ParameterSweep(strategy="random", seed=42)
        sweep.add_uniform("param_A", 0, 10, n_samples=5)

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
    def _make_1d_model(self):
        """Create a simple 1D model for sweep testing."""

        project = Project(path="tests", name="test")
        file = File(
            parent_project=project,
            energy=np.arange(0, 20, 0.5),  # Coarse for speed
            time=np.arange(-10, 100, 5),  # Coarse for speed
        )
        file.load_model(
            model_yaml="models/file_energy.yaml", model_info=["simple_energy"]
        )
        assert file.model_active is not None  # type guard
        return file.model_active

    #
    def _make_2d_model(self):
        """Create a simple 2D model for sweep testing."""

        project = Project(path="tests", name="test")
        file = File(
            parent_project=project,
            energy=np.arange(0, 20, 0.5),  # Coarse for speed
            time=np.arange(-10, 100, 5),  # Coarse for speed
        )
        file.load_model(
            model_yaml="models/file_energy.yaml", model_info=["simple_energy"]
        )
        file.add_time_dependence(
            target_model="simple_energy",
            target_parameter="GLP_01_x0",
            dynamics_yaml="models/file_time.yaml",
            dynamics_model=["MonoExpPosIRF"],
        )
        assert file.model_active is not None  # type guard
        return file.model_active

    #
    def test_simulate_parameter_sweep_basic(self):
        """Test basic parameter sweep simulation"""

        sweep = ParameterSweep(strategy="grid", seed=42)
        sweep.add_range("GLP_01_A", [15, 20])
        sweep.add_range("GLP_01_x0", [8, 10])

        sim = Simulator(
            model=self._make_2d_model(), detection="analog", noise_level=0.05, seed=42
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_sweep.h5"

            sim.simulate_parameter_sweep(
                parameter_sweep=sweep,
                n_realizations=2,
                filepath=str(filepath),
                show_progress=False,
            )

            # Check file was created
            assert filepath.exists()

            # Check HDF5 structure
            with h5py.File(filepath, "r") as f:
                assert "energy" in f
                assert "time" in f
                assert "metadata" in f
                assert "parameter_configs" in f
                assert "simulated_data" in f

                # Check metadata
                metadata = f["metadata"]
                assert isinstance(metadata, h5py.Group)
                assert metadata.attrs["n_configs"] == 4  # 2 × 2
                assert metadata.attrs["n_realizations_per_config"] == 2
                assert metadata.attrs["total_datasets"] == 8  # 4 × 2

    #
    def test_sweep_dataset_smoke(self):
        """Smoke test end-to-end SweepDataset reading from generated HDF5."""

        sweep = ParameterSweep(strategy="grid", seed=42)
        sweep.add_range("GLP_01_A", [15, 20])
        sweep.add_range("GLP_01_x0", [8, 10])

        sim = Simulator(
            model=self._make_2d_model(), detection="analog", noise_level=0.05, seed=42
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_sweep_dataset.h5"

            sim.simulate_parameter_sweep(
                parameter_sweep=sweep,
                n_realizations=2,
                filepath=str(filepath),
                show_progress=False,
            )

            dataset = SweepDataset(str(filepath))

            # Metadata and JSON attribute parsing
            assert isinstance(dataset.meta, dict)
            assert dataset.meta["n_configs"] == 4
            assert dataset.meta["n_realizations_per_config"] == 2
            assert isinstance(dataset.meta["parameter_space"], dict)

            # Axes
            energy, time = dataset.get_axes()
            assert energy.ndim == 1
            assert time.ndim == 1
            assert energy.size > 0
            assert time.size > 0

            # Parameter table
            summary = dataset.get_parameter_summary()
            assert len(summary) == dataset.meta["n_configs"]
            assert "GLP_01_A" in summary.columns
            assert "GLP_01_x0" in summary.columns

            # Per-config access and JSON parsing for all_parameter_values
            config = dataset.load_config(0)
            assert "parameters" in config
            assert "all_parameter_values" in config
            assert isinstance(config["all_parameter_values"], dict)
            assert "clean" in config
            assert config["clean"].shape == (time.size, energy.size)
            assert "noisy" in config
            assert len(config["noisy"]) == dataset.meta["n_realizations_per_config"]

    #
    def test_hdf5_structure(self):
        """Test detailed HDF5 file structure"""

        sweep = ParameterSweep(strategy="grid", seed=42)
        sweep.add_range("GLP_01_A", [15, 20])

        sim = Simulator(
            model=self._make_2d_model(), detection="analog", noise_level=0.05, seed=42
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_structure.h5"

            sim.simulate_parameter_sweep(
                parameter_sweep=sweep,
                n_realizations=3,
                filepath=str(filepath),
                show_progress=False,
            )

            with h5py.File(filepath, "r") as f:
                # Check axes
                energy_ds = f["energy"]
                assert isinstance(energy_ds, h5py.Dataset)
                assert energy_ds.shape[0] > 0

                time_ds = f["time"]
                assert isinstance(time_ds, h5py.Dataset)
                assert time_ds.shape[0] > 0

                # Check config structure
                configs = f["parameter_configs"]
                assert isinstance(configs, h5py.Group)
                assert "config_000000" in configs
                assert "config_000001" in configs

                # Check config attributes
                config = configs["config_000000"]
                assert isinstance(config, h5py.Group)
                assert "GLP_01_A" in config.attrs
                assert "all_parameter_values" in config.attrs

                # Check clean data exists
                assert "clean" in config
                clean_ds = config["clean"]
                assert isinstance(clean_ds, h5py.Dataset)
                assert clean_ds.shape == (len(time_ds[:]), len(energy_ds[:]))

                # Check noisy data structure
                simulated = f["simulated_data"]
                assert isinstance(simulated, h5py.Group)
                data_group = simulated["config_000000"]
                assert isinstance(data_group, h5py.Group)
                assert "000000" in data_group
                assert "000001" in data_group
                assert "000002" in data_group
                assert len(data_group.keys()) == 3

    #
    def test_parameter_values_in_hdf5(self):
        """Test that swept parameter values are correctly stored"""

        sweep = ParameterSweep(strategy="grid", seed=42)
        test_values = [15, 20, 25]
        sweep.add_range("GLP_01_A", test_values)

        sim = Simulator(
            model=self._make_2d_model(), detection="analog", noise_level=0.05, seed=42
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_params.h5"

            sim.simulate_parameter_sweep(
                parameter_sweep=sweep,
                n_realizations=2,
                filepath=str(filepath),
                show_progress=False,
            )

            with h5py.File(filepath, "r") as f:
                # Check that all test values appear in configs
                configs = f["parameter_configs"]
                assert isinstance(configs, h5py.Group)

                stored_values = []
                for i in range(len(test_values)):
                    config_name = f"config_{i:06d}"
                    config = configs[config_name]
                    assert isinstance(config, h5py.Group)
                    stored_values.append(config.attrs["GLP_01_A"])

                assert sorted(stored_values) == sorted(test_values)

    #
    def test_random_strategy_sweep(self):
        """Test parameter sweep with random strategy"""

        sweep = ParameterSweep(strategy="random", seed=42)
        sweep.add_uniform("GLP_01_A", 10, 25, n_samples=5)

        sim = Simulator(
            model=self._make_2d_model(), detection="analog", noise_level=0.05, seed=42
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_random.h5"

            sim.simulate_parameter_sweep(
                parameter_sweep=sweep,
                n_realizations=2,
                filepath=str(filepath),
                show_progress=False,
            )

            with h5py.File(filepath, "r") as f:
                metadata = f["metadata"]
                assert isinstance(metadata, h5py.Group)
                assert metadata.attrs["n_configs"] == 5

                # Check that parameter values are in expected range
                configs = f["parameter_configs"]
                assert isinstance(configs, h5py.Group)
                for i in range(5):
                    config_name = f"config_{i:06d}"
                    config = configs[config_name]
                    assert isinstance(config, h5py.Group)
                    value = config.attrs["GLP_01_A"]
                    assert 10 <= value <= 25  # type: ignore[operator]

    #
    def test_detector_settings_in_metadata(self):
        """Test that detector settings are stored in metadata"""

        sweep = ParameterSweep(strategy="grid", seed=42)
        sweep.add_range("GLP_01_A", [15, 20])

        # Test analog detector
        sim_analog = Simulator(
            model=self._make_2d_model(),
            detection="analog",
            noise_level=0.08,
            noise_type="gaussian",
            seed=42,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_analog.h5"

            sim_analog.simulate_parameter_sweep(
                parameter_sweep=sweep,
                n_realizations=1,
                filepath=str(filepath),
                show_progress=False,
            )

            with h5py.File(filepath, "r") as f:
                metadata = f["metadata"]
                assert isinstance(metadata, h5py.Group)
                assert metadata.attrs["detection"] == "analog"
                assert metadata.attrs["noise_level"] == 0.08
                assert metadata.attrs["noise_type"] == "gaussian"

        # Test photon counting detector
        sim_photon = Simulator(
            model=self._make_2d_model(),
            detection="photon_counting",
            counts_per_delay=5000,
            seed=42,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_photon.h5"

            sim_photon.simulate_parameter_sweep(
                parameter_sweep=sweep,
                n_realizations=1,
                filepath=str(filepath),
                show_progress=False,
            )

            with h5py.File(filepath, "r") as f:
                metadata = f["metadata"]
                assert isinstance(metadata, h5py.Group)
                assert metadata.attrs["detection"] == "photon_counting"
                assert metadata.attrs["counts_per_delay"] == 5000

    #
    def test_sweep_1d_model(self):
        """Test parameter sweep with a 1D model (dim=1)."""

        sweep = ParameterSweep(strategy="grid", seed=42)
        sweep.add_range("GLP_01_A", [15, 20])

        sim = Simulator(
            model=self._make_1d_model(),
            detection="analog",
            noise_level=0.05,
            seed=42,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_sweep_1d.h5"

            sim.simulate_parameter_sweep(
                parameter_sweep=sweep,
                n_realizations=2,
                dim=1,
                filepath=str(filepath),
                show_progress=False,
            )

            assert filepath.exists()

            with h5py.File(filepath, "r") as f:
                assert "energy" in f
                assert "metadata" in f

                metadata = f["metadata"]
                assert isinstance(metadata, h5py.Group)
                assert metadata.attrs["n_configs"] == 2
                assert metadata.attrs["n_realizations_per_config"] == 2

                # Metadata must reflect actual dimension, not model capability
                assert metadata.attrs["dimension"] == 1

                # Check data is 1D (energy axis only)
                configs = f["parameter_configs"]
                assert isinstance(configs, h5py.Group)
                config = configs["config_000000"]
                assert isinstance(config, h5py.Group)
                clean = config["clean"]
                assert isinstance(clean, h5py.Dataset)
                assert clean.ndim == 1

    #
    def test_simulate_n_rejects_zero(self):
        """Test that simulate_n raises ValueError for N < 1."""

        sim = Simulator(
            model=self._make_2d_model(),
            detection="analog",
            noise_level=0.05,
            seed=42,
        )

        with pytest.raises(ValueError, match="n must be >= 1"):
            sim.simulate_n(n=0, dim=2, show_progress=False)

        with pytest.raises(ValueError, match="n must be >= 1"):
            sim.simulate_n(n=-1, dim=2, show_progress=False)

    #
    def test_sweep_seed_zero_metadata(self):
        """Test that seed=0 is stored correctly, not as 'None'."""

        sweep = ParameterSweep(strategy="grid", seed=0)
        sweep.add_range("GLP_01_A", [15])

        sim = Simulator(
            model=self._make_2d_model(),
            detection="analog",
            noise_level=0.05,
            seed=42,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_seed_zero.h5"

            sim.simulate_parameter_sweep(
                parameter_sweep=sweep,
                n_realizations=1,
                filepath=str(filepath),
                show_progress=False,
            )

            with h5py.File(filepath, "r") as f:
                metadata = f["metadata"]
                assert isinstance(metadata, h5py.Group)
                assert metadata.attrs["sweep_seed"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
