"""
This module provides tools for systematically varying model parameters to
generate and load/inspect training datasets for machine learning applications.

[class "ParameterSweep"]
Parameter space exploration for ML dataset generation.

[class SweepDataset]
Utilities for inspecting parameter sweep datasets.
"""

import itertools
from collections.abc import Generator
from typing import Any, cast

import h5py
import numpy as np
import pandas as pd

from trspecfit.utils.hdf5 import json_loads_attr, require_dataset, require_group


#
#
class ParameterSweep:
    """
    Generator for systematic parameter space exploration.

    Creates parameter combinations for ML training data generation by either
    grid search (combinatorial) or random sampling from distributions.

    Parameters
    ----------
    strategy : {'auto', 'grid', 'random'}, default='auto'
        Sampling strategy:
        - 'auto': Use grid if all parameters are discrete ranges, else random
        - 'grid': Full combinatorial product of all parameter values
        - 'random': Independent random sampling from each parameter
    seed : int, optional
        Random seed for reproducibility

    Examples
    --------
    >>> # Grid over discrete values
    >>> sweep = ParameterSweep(strategy='grid')
    >>> sweep.add_range('GLP_01_A', [10, 15, 20])
    >>> sweep.add_range('GLP_01_x0', [82, 84, 86])
    >>> print(sweep.get_n_configs())
    9

    >>> # Random sampling for ML
    >>> sweep = ParameterSweep(strategy='random', seed=42)
    >>> sweep.add_uniform('GLP_01_A', 5, 30, n_samples=100)
    >>> sweep.add_normal('GLP_01_x0', mean=85, std=2, n_samples=100)
    >>> print(sweep.get_n_configs())
    100

    >>> # Iterate over configurations
    >>> for config in sweep:
    ...     print(config)
    {'GLP_01_A': 12.5, 'GLP_01_x0': 84.8}
    {'GLP_01_A': 18.3, 'GLP_01_x0': 85.2}
    ...

    See Also
    --------
    Simulator.simulate_parameter_sweep : Use sweep to generate datasets
    """

    #
    def __init__(self, strategy: str = "auto", seed: int | None = None) -> None:
        """
        Initialize parameter sweep generator.

        Parameters
        ----------
        strategy : {'auto', 'grid', 'random'}, default='auto'
            Sampling strategy
        seed : int, optional
            Random seed for reproducibility
        """

        valid_strategies = ["auto", "grid", "random"]
        if strategy not in valid_strategies:
            raise ValueError(
                f"strategy must be one of {valid_strategies}, got '{strategy}'"
            )

        self.strategy = strategy
        self.parameter_specs: dict[str, dict[str, Any]] = {}
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

    #
    def add_range(self, par_name: str, values: list[float]) -> None:
        """
        Add discrete parameter values to sweep.

        For grid strategy: each value will be combined with all other parameters.
        For random strategy: values will be randomly sampled from this list.

        Parameters
        ----------
        par_name : str
            Parameter name (e.g., 'GLP_01_A')
        values : array-like
            Discrete values to use

        Examples
        --------
        >>> sweep = ParameterSweep()
        >>> sweep.add_range('GLP_01_A', [10, 15, 20, 25])
        >>> sweep.add_range('GLP_01_x0', [82, 84, 86, 88])
        """

        self.parameter_specs[par_name] = {"type": "range", "values": np.array(values)}

    #
    def add_uniform(
        self, par_name: str, min_val: float, max_val: float, n_samples: int
    ) -> None:
        """
        Add parameter with uniform distribution.

        For grid strategy: n_samples values will be drawn and each combined
        with all other parameters (creates n_samples × other_params configs).
        For random strategy: independent sample drawn for each configuration.

        Parameters
        ----------
        par_name : str
            Parameter name
        min_val : float
            Minimum value (inclusive)
        max_val : float
            Maximum value (inclusive)
        n_samples : int
            Number of samples to draw (for grid) or total configs (for random)

        Examples
        --------
        >>> sweep = ParameterSweep(strategy='random')
        >>> sweep.add_uniform('GLP_01_A', min_val=5, max_val=30, n_samples=100)
        """

        self.parameter_specs[par_name] = {
            "type": "uniform",
            "min": min_val,
            "max": max_val,
            "n_samples": n_samples,
        }

    #
    def add_normal(
        self, par_name: str, mean: float, std: float, n_samples: int
    ) -> None:
        """
        Add parameter with normal (Gaussian) distribution.

        For grid strategy: n_samples values will be drawn and each combined
        with all other parameters.
        For random strategy: independent sample drawn for each configuration.

        Parameters
        ----------
        par_name : str
            Parameter name
        mean : float
            Distribution mean
        std : float
            Distribution standard deviation
        n_samples : int
            Number of samples to draw

        Examples
        --------
        >>> sweep = ParameterSweep(strategy='random')
        >>> sweep.add_normal('GLP_01_F', mean=1.5, std=0.3, n_samples=100)
        """

        self.parameter_specs[par_name] = {
            "type": "normal",
            "mean": mean,
            "std": std,
            "n_samples": n_samples,
        }

    #
    def add_lognormal(
        self, par_name: str, mean: float, std: float, n_samples: int
    ) -> None:
        """
        Add parameter with log-normal distribution.

        Useful for parameters that must be positive and span orders of magnitude
        (e.g., decay times, rate constants).

        Parameters
        ----------
        par_name : str
            Parameter name
        mean : float
            Mean of underlying normal distribution (ln scale)
        std : float
            Std of underlying normal distribution (ln scale)
        n_samples : int
            Number of samples to draw

        Examples
        --------
        >>> # Generate decay times spanning 10-1000 ps
        >>> sweep = ParameterSweep(strategy='random')
        >>> sweep.add_lognormal('tau', mean=np.log(100), std=1, n_samples=100)
        """

        self.parameter_specs[par_name] = {
            "type": "lognormal",
            "mean": mean,
            "std": std,
            "n_samples": n_samples,
        }

    #
    def _determine_strategy(self) -> str:
        """
        Determine which sampling strategy to use.

        For 'auto' mode, uses grid if all parameters are discrete ranges,
        otherwise uses random sampling.

        Returns
        -------
        str : 'grid' or 'random'
        """

        if self.strategy != "auto":
            return self.strategy

        # If all parameters are discrete ranges, use grid
        if all(spec["type"] == "range" for spec in self.parameter_specs.values()):
            return "grid"
        return "random"

    #
    def _sample_distribution(self, spec: dict[str, Any]) -> np.ndarray:
        """
        Generate samples from a distribution specification.

        Parameters
        ----------
        spec : dict
            Distribution specification with 'type' and distribution parameters

        Returns
        -------
        ndarray
            Array of samples from the specified distribution
        """

        n = int(spec.get("n_samples", 10))

        if spec["type"] == "uniform":
            return cast("np.ndarray", np.random.uniform(spec["min"], spec["max"], n))
        if spec["type"] == "normal":
            return cast("np.ndarray", np.random.normal(spec["mean"], spec["std"], n))
        if spec["type"] == "lognormal":
            return cast("np.ndarray", np.random.lognormal(spec["mean"], spec["std"], n))
        raise ValueError(f"Unknown distribution type: {spec['type']}")

    #
    def _generate_grid(self) -> Generator[dict[str, float], None, None]:
        """
        Generate configurations using full combinatorial grid.

        For discrete ranges, uses all values.
        For continuous distributions, pre-generates n_samples and treats
        them as discrete values for the grid.

        Yields
        ------
        dict
            Parameter configuration {par_name: value, ...}
        """

        # Separate and prepare all parameters
        all_params = {}

        for par_name, spec in self.parameter_specs.items():
            if spec["type"] == "range":
                # Use discrete values directly
                all_params[par_name] = spec["values"]
            else:
                # Pre-generate samples from continuous distributions
                all_params[par_name] = self._sample_distribution(spec)

        # Create combinatorial product
        par_names = list(all_params.keys())
        value_lists = [all_params[name] for name in par_names]

        # Yield each combination
        for values in itertools.product(*value_lists):
            yield dict(zip(par_names, values, strict=True))

    #
    def _generate_random(self) -> Generator[dict[str, float], None, None]:
        """
        Generate configurations using independent random sampling.

        Each parameter is sampled independently for each configuration.
        For discrete ranges, randomly selects from available values.

        Yields
        ------
        dict
            Parameter configuration {par_name: value, ...}
        """

        # Determine total number of configurations
        n_configs = max(
            spec.get("n_samples", len(spec.get("values", [10])))
            for spec in self.parameter_specs.values()
        )

        for _ in range(n_configs):
            config = {}
            for par_name, spec in self.parameter_specs.items():
                if spec["type"] == "range":
                    # Random choice from discrete values
                    config[par_name] = np.random.choice(spec["values"])
                elif spec["type"] == "uniform":
                    config[par_name] = np.random.uniform(spec["min"], spec["max"])
                elif spec["type"] == "normal":
                    config[par_name] = np.random.normal(spec["mean"], spec["std"])
                elif spec["type"] == "lognormal":
                    config[par_name] = np.random.lognormal(spec["mean"], spec["std"])
            yield config

    #
    def __iter__(self) -> Generator[dict[str, float], None, None]:
        """
        Iterate over parameter configurations.

        Yields
        ------
        dict
            Parameter configuration {par_name: value, ...}
        """

        # Reset seed at start of iteration for reproducibility
        if self.seed is not None:
            np.random.seed(self.seed)

        strategy = self._determine_strategy()

        if strategy == "grid":
            yield from self._generate_grid()
        else:  # random
            yield from self._generate_random()

    #
    def get_n_configs(self) -> int:
        """
        Calculate total number of configurations that will be generated.

        Returns
        -------
        int
            Total number of parameter configurations

        Examples
        --------
        >>> sweep = ParameterSweep(strategy='grid')
        >>> sweep.add_range('A', [10, 20, 30])
        >>> sweep.add_range('x0', [82, 84, 86, 88])
        >>> sweep.get_n_configs()
        12
        """

        strategy = self._determine_strategy()

        if strategy == "grid":
            # Combinatorial product
            n = 1
            for spec in self.parameter_specs.values():
                if spec["type"] == "range":
                    n *= len(spec["values"])
                else:
                    n *= spec.get("n_samples", 10)
            return n
        # random
        # Maximum n_samples across all parameters
        return int(
            max(
                int(spec.get("n_samples", len(spec.get("values", [10]))))
                for spec in self.parameter_specs.values()
            )
        )


#
#
class SweepDataset:
    """
    Inspector for parameter sweep HDF5 datasets.

    Provides methods to examine dataset structure, metadata, and parameter
    configurations without loading all data into memory.

    Parameters
    ----------
    filepath : str
        Path to HDF5 file generated by simulate_parameter_sweep()

    Attributes
    ----------
    filepath : str
        Path to HDF5 file
    meta : dict
        Dataset metadata (loaded on initialization)

    Examples
    --------
    >>> # Inspect dataset
    >>> dataset = SweepDataset('simulated_data/training_data.h5')
    >>> dataset.print_summary()

    >>> # Get parameter summary
    >>> df = dataset.get_parameter_summary()
    >>> print(df.describe())

    >>> # Load specific configuration for inspection
    >>> config = dataset.load_config(0)
    >>> print(config['parameters'])

    >>> # Access axes
    >>> energy, time = dataset.get_axes()

    Notes
    -----
    This class is designed for inspection and quick data access.
    For ML training, users should implement framework-specific data loaders
    that read from the HDF5 file structure documented in simulate_parameter_sweep.

    See Also
    --------
    Simulator.simulate_parameter_sweep : Generates these datasets
    """

    #
    def __init__(self, filepath: str) -> None:
        """
        Initialize dataset inspector.

        Parameters
        ----------
        filepath : str
            Path to parameter sweep HDF5 file
        """

        self.filepath = filepath
        self.meta = self._load_metadata()

    #
    def _load_metadata(self) -> dict:
        """
        Load metadata from HDF5 file.

        Returns
        -------
        dict
            Metadata dictionary with sweep parameters and settings
        """

        with h5py.File(self.filepath, "r") as f:
            meta = dict(f["metadata"].attrs)

            # Parse JSON strings
            if "parameter_space" in meta:
                meta["parameter_space"] = json_loads_attr(meta["parameter_space"])

            return meta

    #
    def get_axes(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Load energy and time axes.

        Returns
        -------
        energy : ndarray
            Energy axis
        time : ndarray
            Time axis (empty array for 1D data)

        Examples
        --------
        >>> dataset = SweepDataset('data.h5')
        >>> energy, time = dataset.get_axes()
        >>> print(f"Energy: {len(energy)} points")
        """

        with h5py.File(self.filepath, "r") as f:
            energy_ds = require_dataset(f["energy"], "energy")
            time_ds = require_dataset(f["time"], "time")
            energy = np.asarray(energy_ds[:], dtype=float)
            time = np.asarray(time_ds[:], dtype=float)
            if energy.ndim != 1 or time.ndim != 1:
                raise ValueError("HDF5 axes 'energy' and 'time' must be 1D arrays")
            return energy, time

    #
    def get_parameter_summary(self) -> pd.DataFrame:
        """
        Create summary table of all parameter configurations.

        Returns
        -------
        pandas.DataFrame
            DataFrame with one row per configuration, columns for each
            swept parameter

        Examples
        --------
        >>> dataset = SweepDataset('data.h5')
        >>> df = dataset.get_parameter_summary()
        >>> print(df.describe())  # Statistical summary
        >>> df.hist(figsize=(12, 8))  # Visualize distributions
        """

        n_configs = self.meta["n_configs"]

        # Collect all parameter values
        param_data = []
        with h5py.File(self.filepath, "r") as f:
            configs_group = require_group(f["parameter_configs"], "parameter_configs")
            for config_idx in range(n_configs):
                config_name = f"config_{config_idx:06d}"
                config_group = require_group(
                    configs_group[config_name], f"parameter_configs/{config_name}"
                )

                # Get swept parameters (exclude 'all_parameters')
                parameters = {
                    key: value
                    for key, value in config_group.attrs.items()
                    if key != "all_parameters"
                }
                param_data.append(parameters)

        # Convert to DataFrame
        df = pd.DataFrame(param_data)
        df.index.name = "config_idx"

        return df

    #
    def load_config(
        self, config_idx: int, load_clean: bool = True, load_noisy: bool = True
    ) -> dict[str, Any]:
        """
        Load a single parameter configuration and its data.

        Useful for quick inspection. For ML training loops, access HDF5
        directly for better performance.

        Parameters
        ----------
        config_idx : int
            Configuration index (0 to n_configs-1)
        load_clean : bool, default=True
            Load clean (noiseless) data
        load_noisy : bool, default=True
            Load noisy realizations

        Returns
        -------
        dict
            Dictionary with keys:
            - parameters: Dict of swept parameter values
            - all_parameters: Dict of all model parameters (if available)
            - clean: Clean data (if load_clean=True)
            - noisy: List of noisy realizations (if load_noisy=True)

        Examples
        --------
        >>> dataset = SweepDataset('data.h5')
        >>> config = dataset.load_config(0)
        >>> print(f"Parameters: {config['parameters']}")
        >>> print(f"Data shape: {config['clean'].shape}")
        """

        config_name = f"config_{config_idx:06d}"
        result: dict[str, Any] = {}

        with h5py.File(self.filepath, "r") as f:
            # Load parameter values
            configs_group = require_group(f["parameter_configs"], "parameter_configs")
            config_group = require_group(
                configs_group[config_name], f"parameter_configs/{config_name}"
            )

            # Get swept parameters (all attributes except 'all_parameters')
            parameters = {
                key: value
                for key, value in config_group.attrs.items()
                if key != "all_parameters"
            }
            result["parameters"] = parameters

            # Get all parameters (JSON)
            if "all_parameters" in config_group.attrs:
                result["all_parameters"] = json_loads_attr(
                    config_group.attrs["all_parameters"]
                )

            # Load clean data
            if load_clean and "clean" in config_group:
                clean_ds = require_dataset(
                    config_group["clean"], f"parameter_configs/{config_name}/clean"
                )
                result["clean"] = clean_ds[:]

            # Load noisy realizations
            if load_noisy:
                simulated_group = require_group(f["simulated_data"], "simulated_data")
                data_group = require_group(
                    simulated_group[config_name], f"simulated_data/{config_name}"
                )
                noisy_list = []
                for key in sorted(data_group.keys()):
                    noisy_ds = require_dataset(
                        data_group[key], f"simulated_data/{config_name}/{key}"
                    )
                    noisy_list.append(noisy_ds[:])
                result["noisy"] = noisy_list

        return result

    #
    def print_summary(self) -> None:
        """
        Print comprehensive summary of dataset contents.

        Examples
        --------
        >>> dataset = SweepDataset('simulated_data/training_data.h5')
        >>> dataset.print_summary()
        Dataset: simulated_data/training_data.h5
        ════════════════════════════════════════
        Configurations: 100
        Realizations per config: 20
        Total datasets: 2000

        Data dimensions: 2D (time × energy)
        Energy points: 200
        Time points: 110

        Detection: analog
        Noise level: 0.05
        Noise type: poisson

        Parameter space:
        GLP_01_A: discrete [0.25, 1.0, 4.0]
        GLP_01_x0: normal(85.0, 2.0) - 100 samples
        """

        energy, time = self.get_axes()

        print(f"\nDataset: {self.filepath}")
        print("=" * 60)
        print(f"Configurations: {self.meta['n_configs']}")
        print(f"Realizations per config: {self.meta['n_realizations_per_config']}")
        print(f"Total datasets: {self.meta['total_datasets']}")
        print()

        dimension = self.meta.get("dimension", 2)
        if dimension == 2:
            print("Data dimensions: 2D (time × energy)")
            print(f"Energy points: {len(energy)}")
            print(f"Time points: {len(time)}")
        else:
            print("Data dimensions: 1D (energy)")
            print(f"Energy points: {len(energy)}")
        print()

        print(f"Detection: {self.meta['detection']}")
        if self.meta["detection"] == "analog":
            print(f"Noise level: {self.meta.get('noise_level', 'N/A')}")
            print(f"Noise type: {self.meta.get('noise_type', 'N/A')}")
        else:
            print(f"Counts per delay: {self.meta.get('counts_per_delay', 'N/A')}")
        print()

        print("Parameter space:")
        for par_name, spec in self.meta["parameter_space"].items():
            if spec["type"] == "range":
                values = spec["values"]
                # Show all values if few enough, otherwise show count and range
                if len(values) <= 10:
                    values_str = ", ".join(f"{v:.3g}" for v in values)
                    print(f"  {par_name}: discrete [{values_str}]")
                else:
                    print(
                        f"  {par_name}: discrete {len(values)} values "
                        f"[{min(values):.3g} ... {max(values):.3g}]"
                    )
            elif spec["type"] == "uniform":
                print(
                    f"  {par_name}: uniform({spec['min']:.3g}, {spec['max']:.3g}) "
                    f"- {spec['n_samples']} samples"
                )
            elif spec["type"] == "normal":
                print(
                    f"  {par_name}: normal({spec['mean']:.3g}, {spec['std']:.3g}) "
                    f"- {spec['n_samples']} samples"
                )
            elif spec["type"] == "lognormal":
                print(
                    f"  {par_name}: lognormal({spec['mean']:.3g}, {spec['std']:.3g}) "
                    f"- {spec['n_samples']} samples"
                )
        print()

    #
    def __repr__(self) -> str:
        """String representation."""

        return (
            f"SweepDataset('{self.filepath}', "
            f"n_configs={self.meta['n_configs']}, "
            f"n_realizations={self.meta['n_realizations_per_config']})"
        )
