"""
Synthetic data generation for testing, validation, ML training data generation.

This module provides tools for generating realistic simulated spectroscopy
data from models, with support for different detector types and noise models.
Use for:
- Testing fitting algorithms with known ground truth
- Exploring parameter sensitivity and identifiability
- Optimizing experimental design (SNR requirements)
- Generating training data for machine learning
- Validating analysis pipelines

Key Features
------------
- Two detector types: analog and photon counting
- Multiple noise models: Poisson, Gaussian, or none
- 1D and 2D spectrum simulation
- Batch generation for statistical analysis
- Parameter sweeping (grid/random/uniform) for ML training

Detector Types
--------------
**Analog Detectors** (CCD, photodiodes, lock-in amplifiers):
- Continuous signal output
- Additive noise (Gaussian or Poisson)
- Noise level controlled by noise_level parameter

**Photon Counting** (APD, photomultiplier, event mode):
- Discrete photon events
- Shot noise inherent (Poisson statistics)
- Count rate determines signal-to-noise ratio

Workflow
--------
1. Testing and Validation

   - Create model with trspecfit.mcp.Model
   - Initialize Simulator with model and noise parameters
   - Generate data with simulate_1D() or simulate_2D()
     OR generate multiple realizations with simulate_N()
   - Save data and ground truth with save_data()
   - Fit simulated data to validate fitting pipeline

2. Machine Learning Training Data Generation

   - Create model with trspecfit.mcp.Model
   - Define parameter space using trspecfit.utils.sweep.ParameterSweep
   - Initialize Simulator with model and noise parameters
   - Generate multiple realizations (N) for each parameter combination
     (data, ground truth, and relevant metadata get saved automatically)

Examples
--------
See examples/simulator/ directory for complete workflows.
"""

import os
from pathlib import Path
import copy
import numpy as np
import matplotlib.pyplot as plt
import json
import h5py
from typing import Dict, List, Optional, Tuple, cast
from trspecfit.mcp import Model
from trspecfit.utils.hdf5 import require_group
from trspecfit.utils.sweep import ParameterSweep
from trspecfit.utils import plot as uplt

#
#
class Simulator:
    """
    Simulate 2D time- and energy-resolved spectroscopy data with noise.
    
    This class generates synthetic data based on a model, adding realistic noise
    to simulate experimental measurements. Supports both analog detectors (with
    additive noise) and photon counting detectors (with shot noise).
    
    Parameters
    ----------
    model : Model
        Model instance from trspecfit.mcp with defined components and parameters.
        Must have energy and time axes set before simulation.
    detection : {'analog', 'photon_counting'}, default='analog'
        Detection technique to simulate:
        - 'analog': Continuous signal with additive noise
        - 'photon_counting': Discrete photon events with Poisson statistics
    noise_level : float, default=0.05
        Noise amplitude for analog detectors (0.0-1.0 for relative noise).
        Larger values = more noise. Ignored for photon_counting.
    noise_type : {'poisson', 'gaussian', 'none'}, default='poisson'
        Type of noise for analog detectors:
        - 'poisson': Shot noise (realistic for low light)
        - 'gaussian': White noise (simpler, faster)
        - 'none': No noise (testing and debugging)
        Ignored for photon_counting (always Poisson).
    counts_per_delay : int, optional
        Total photon count per time delay (photon_counting only).
        Directly sets signal-to-noise ratio. Mutually exclusive with
        count_rate + integration_time.
    count_rate : float, optional
        Photon count rate in Hz (photon_counting only).
        Combined with integration_time to compute counts_per_delay.
    integration_time : float, optional
        Integration time per delay point in seconds (photon_counting only).
        Combined with count_rate to compute counts_per_delay.
    seed : int, optional
        Random seed for reproducibility. If None, uses random initialization.
    
    Attributes
    ----------
    model : Model
        Model instance used for simulation
    detection : str
        Detection type ('analog' or 'photon_counting')
    seed : int or None
        Random seed value
    noise_level : float
        Analog detector noise level
    noise_type : str
        Analog detector noise type
    counts_per_delay : int
        Photon counting detector count budget
    count_rate : float or None
        Photon counting detector rate
    integration_time : float or None
        Photon counting integration time per delay
    data_clean : ndarray or None
        Most recently generated clean (noiseless) data
    data_noisy : ndarray or None
        Most recently generated noisy data
    noise : ndarray or None
        Most recently generated noise component (noisy - clean)

    Examples
    --------
    See examples/simulator/ directory for complete workflows.

    Notes
    -----
    **Analog vs. Photon Counting:**
    
    Analog detectors (CCD, photodiode, lock-in):
    - Pros: High dynamic range, simple operation
    - Cons: Read noise, dark current
    - Simulation: Continuous signal + additive noise
    
    Photon counting (APD, PMT, event mode):
    - Pros: No read noise, single-photon sensitivity
    - Cons: Dead time, count rate limits, pulse pileup
    - Simulation: Discrete events following Poisson statistics
    
    **Noise Level Selection:**
    
    For analog detectors, noise_level is relative to signal:
    - 0.01 (1%): Very clean, ideal conditions
    - 0.05 (5%): Typical good data
    - 0.10 (10%): Moderate noise, still fittable
    - 0.20 (20%): Challenging, may need averaging
    
    For photon counting, SNR set by counts_per_delay:
    - 100 counts: SNR ~ 10 (marginal)
    - 1000 counts: SNR ~ 32 (good)
    - 10000 counts: SNR ~ 100 (excellent)
    
    **Photon Counting Parameter Resolution:**
    
    The simulator resolves photon counting parameters as:
    1. If counts_per_delay specified directly → use it
    2. Else if count_rate and integration_time specified → compute counts_per_delay
    3. Else → estimate from model scale (prints warning)
    
    The third case assumes model amplitudes represent realistic count rates,
    which may not be true. Always specify counts_per_delay or (count_rate,
    integration_time) explicitly for accurate photon counting simulation.
    
    **Memory Usage:**
    
    Large 2D datasets can use significant memory:
    - Single dataset: ~8 MB per 1000×500 spectrum (float64)
    - simulate_N(N=100): ~800 MB for same size
    - Consider smaller grids or batch processing for large N
    
    See Also
    --------
    trspecfit.mcp.Model : Model class for simulation
    simulate_1D : Generate 1D spectrum
    simulate_2D : Generate 2D spectrum
    simulate_N : Generate multiple realizations
    save_data : Save simulated data to HDF5
    """
    #
    def __init__(
        self,
        model: Model,
        detection: str = 'analog',
        noise_level: float = 0.05,
        noise_type: str = 'poisson',
        counts_per_delay: Optional[int] = None,
        count_rate: Optional[float] = None,
        integration_time: Optional[float] = None,
        seed: Optional[int] = None
    ) -> None:
        """
        Initialize simulator with a model and noise parameters.
        
        Parameters
        ----------
        model : Model
            Model instance with defined components and parameters.
            Must have model.energy and (for 2D) model.time set.
        detection : {'analog', 'photon_counting'}, default='analog'
            Detection technique to simulate
        noise_level : float, default=0.05
            Noise amplitude for analog detectors (0.0-1.0 relative to signal)
        noise_type : {'poisson', 'gaussian', 'none'}, default='poisson'
            Noise type for analog detectors
        counts_per_delay : int, optional
            Total photons per delay (photon_counting only)
        count_rate : float, optional
            Photon rate in Hz (photon_counting only)
        integration_time : float, optional
            Integration time per delay in seconds (photon_counting only)
        seed : int, optional
            Random seed for reproducibility
        
        Raises
        ------
        ValueError
            If detection type is invalid
            If counts_per_delay ≤ 0 (after estimation)
        
        Examples
        --------
        >>> # Analog with default settings
        >>> sim = Simulator(model)
        
        >>> # Analog with custom noise
        >>> sim = Simulator(model, noise_level=0.1, noise_type='gaussian')
        
        >>> # Photon counting with direct count specification
        >>> sim = Simulator(model, detection='photon_counting',
        ...                 counts_per_delay=5000)
        
        >>> # Photon counting with rate specification
        >>> sim = Simulator(model, detection='photon_counting',
        ...                 count_rate=1e6, integration_time=0.01)
        
        >>> # Reproducible simulation
        >>> sim = Simulator(model, seed=42)
        """ 
        self.model = model
        self.detection = detection.lower()
        self.seed = seed
        
        # Analog detector parameters
        self.noise_level = noise_level
        self.noise_type = noise_type.lower()
        
        # Photon counting parameters
        self.counts_per_delay: Optional[int] = counts_per_delay
        self.count_rate: Optional[float] = count_rate
        self.integration_time: Optional[float] = integration_time
        
        # Validate and resolve photon counting parameters
        if self.detection == 'photon_counting':
            self._resolve_photon_counting_params()
        
        # Validate detection type
        if self.detection not in ['analog', 'photon_counting']:
            raise ValueError(f"detection must be 'analog' or 'photon_counting', got '{detection}'")
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Storage for simulated data
        self.data_clean: Optional[np.ndarray] = None  # Without noise
        self.data_noisy: Optional[np.ndarray] = None  # With noise
        self.noise: Optional[np.ndarray] = None       # Just the noise component
    
    #
    def _resolve_photon_counting_params(self) -> None:
        """
        Resolve photon counting parameters. User can specify either:
        - counts_per_delay directly, OR
        - count_rate and integration_time (will compute counts_per_delay), OR
        - nothing (will estimate from model scale with warning)
        """
        if self.counts_per_delay is not None:
            # Direct specification takes precedence
            if self.count_rate is not None or self.integration_time is not None:
                print("Warning: counts_per_delay specified directly. Ignoring count_rate and integration_time.")
        elif self.count_rate is not None and self.integration_time is not None:
            # Calculate from rate and time
            self.counts_per_delay = int(self.count_rate * self.integration_time)
        else:
            # Estimate from model: assume clean_data represents count rates
            # Generate clean data to get the scale
            if self.model.time is not None and len(self.model.time) > 0:
                # 2D data
                self.model.create_value2D()
                clean_data = self.model.value2D
            else:
                # 1D data
                self.model.create_value1D()
                clean_data = self.model.value1D
            if clean_data is None:
                raise RuntimeError("Model evaluation did not produce clean data for photon counting estimation")
            
            # Estimate counts_per_delay as the average total signal per time step
            # This assumes the model amplitudes are in a realistic count rate range
            signal_positive = np.abs(clean_data)
            if clean_data.ndim == 2:
                row_totals = np.sum(signal_positive, axis=1)
                self.counts_per_delay = int(np.mean(row_totals))
            else:
                self.counts_per_delay = int(np.sum(signal_positive))

            print(f"WARNING: No photon count specified for photon_counting detection.")
            print(f"Estimating from model: {self.counts_per_delay:.2e} counts/delay")
            print(f"For accurate simulation, specify counts_per_delay or (count_rate, integration_time).")
            print(f"This estimate assumes your model amplitudes represent realistic count rates.")
        
        # Ensure counts_per_delay is positive
        if self.counts_per_delay <= 0:
            raise ValueError(
                f"counts_per_delay must be positive, got {self.counts_per_delay}.\n"
                f"Model may have negative or zero signal. Check your model or specify counts_per_delay manually."
            )
    
    #
    def generate_clean_data(self, dim: int = 2, t_ind: int = 0) -> np.ndarray:
        """
        Generate clean data from model (no noise)
        
        Parameters:
            dim: Dimension (1 for 1D, 2 for 2D)
            t_ind: Time index for 1D simulations (ignored for 2D)
            
        Returns:
            Clean data array (1D or 2D depending on dim)
        """
        if dim == 1:
            self.model.create_value1D(t_ind=t_ind, return1D=False)
            self.data_clean = copy.deepcopy(self.model.value1D)
        elif dim == 2:
            self.model.create_value2D()
            self.data_clean = copy.deepcopy(self.model.value2D)
        else:
            raise ValueError(f"dim must be 1 or 2, got {dim}")

        if self.data_clean is None:
            raise RuntimeError("Model evaluation did not produce clean data")
        return self.data_clean
    
    #
    def add_noise(self, clean_data: np.ndarray, dim: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add noise to clean data based on detection technique
        
        Parameters:
            clean_data: Clean data array (1D or 2D)
            dim: Dimension (1 for 1D, 2 for 2D)
            
        Returns:
            Tuple of (noisy_data, noise)
        """
        if self.detection == 'analog':
            # Use traditional noise addition
            if dim == 1:
                noise = self._generate_noise_analog_1D(clean_data)
            elif dim == 2:
                noise = self._generate_noise_analog_2D(clean_data)
            else:
                raise ValueError(f"dim must be 1 or 2, got {dim}")
            
            noisy_data = clean_data + noise
            
        elif self.detection == 'photon_counting':
            # Sample photons according to signal distribution
            if dim == 1:
                noisy_data = self._sample_photons_1D(clean_data)
            elif dim == 2:
                noisy_data = self._sample_photons_2D(clean_data)
            else:
                raise ValueError(f"dim must be 1 or 2, got {dim}")
            
            noise = noisy_data - clean_data
        else:
            raise ValueError(f"Unknown detection type: {self.detection}")
        
        return noisy_data, noise
    
    #
    def simulate_1D(self, t_ind: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate 1D spectrum (energy-resolved) at a specific time point.
        
        Generates a single energy-resolved spectrum from the model at the
        specified time index, adds appropriate noise for the detector type,
        and stores results for later access.
        
        Parameters
        ----------
        t_ind : int, default=0
            Time index for which to generate spectrum.
            For models without time-dependence, use default 0.
        
        Returns
        -------
        clean_data : ndarray
            Noiseless spectrum from model (shape: [n_energy])
        noisy_data : ndarray
            Spectrum with added noise (shape: [n_energy])
        noise : ndarray
            Noise component (noisy - clean, shape: [n_energy])
        
        Examples
        --------
        >>> # Simulate baseline spectrum
        >>> sim = Simulator(model, noise_level=0.05)
        >>> clean, noisy, noise = sim.simulate_1D(t_ind=0)
        >>> 
        >>> # Plot comparison
        >>> plt.plot(model.energy, clean, 'k-', label='Clean')
        >>> plt.plot(model.energy, noisy, 'r.', label='Noisy', ms=2)
        >>> plt.legend()
        
        >>> # Calculate SNR
        >>> snr = sim.get_SNR()
        >>> print(f"Signal-to-noise ratio: {snr:.1f}")
        
        >>> # Simulate different time points
        >>> for t_i in [0, 50, 100]:
        ...     clean, noisy, noise = sim.simulate_1D(t_ind=t_i)
        ...     plt.plot(model.energy, noisy, label=f't={model.time[t_i]:.1f}')
        
        Notes
        -----
        Results are stored in simulator attributes for later access:
        - self.data_clean: (Last) clean spectrum
        - self.data_noisy: Last noisy spectrum
        - self.noise: Last noise realization
        
        Can access these without re-simulation:
        >>> sim.simulate_1D()
        >>> snr = sim.get_SNR()  # Uses stored data
        >>> sim.plot_comparison(dim=1)  # Uses stored data
        
        See Also
        --------
        simulate_2D : Simulate full 2D spectrum
        simulate_N : Generate multiple 1D realizations
        plot_comparison : Visualize results
        """
        # Generate clean spectrum from model
        clean_data = self.generate_clean_data(dim=1, t_ind=t_ind)
        
        # Add noise
        noisy_data, noise = self.add_noise(clean_data, dim=1)
        
        # Store for later use (e.g., plotting, SNR calculation)
        self.data_clean = clean_data
        self.data_noisy = noisy_data
        self.noise = noise
        
        return clean_data, noisy_data, noise
    
    #
    def simulate_2D(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate 2D spectrum (time- and energy-resolved).
        
        Generates a complete 2D time- and energy-resolved spectrum from the
        model, adds appropriate noise for each time point, and stores results.
        
        Returns
        -------
        clean_data : ndarray
            Noiseless 2D spectrum from model (shape: [n_time, n_energy])
        noisy_data : ndarray
            2D spectrum with added noise (shape: [n_time, n_energy])
        noise : ndarray
            Noise component (noisy - clean, shape: [n_time, n_energy])
        
        Examples
        --------
        >>> # Basic 2D simulation
        >>> sim = Simulator(model, noise_level=0.05)
        >>> clean, noisy, noise = sim.simulate_2D()
        >>> 
        >>> # Visualize with built-in plotter
        >>> sim.plot_comparison(dim=2)
        
        >>> # Manual visualization
        >>> fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        >>> ax1.pcolormesh(model.energy, model.time, clean)
        >>> ax1.set_title('Clean Model')
        >>> ax2.pcolormesh(model.energy, model.time, noisy)
        >>> ax2.set_title(f'Noisy (SNR={sim.get_SNR():.1f})')
        
        >>> # Test fitting on simulated data
        >>> clean, noisy, noise = sim.simulate_2D()
        >>> # ... set up fitting ...
        >>> file.data = noisy  # Use noisy data for fit
        >>> file.fit_2Dmodel(model_name='test', fit=2)
        >>> # Compare fitted vs. true parameters
        
        >>> # Vary noise level to study impact
        >>> for noise_level in [0.01, 0.05, 0.10]:
        ...     sim.set_noise_level(noise_level)
        ...     clean, noisy, noise = sim.simulate_2D()
        ...     snr = sim.get_SNR()
        ...     print(f"Noise level {noise_level:.2f}: SNR = {snr:.1f}")
        
        Notes
        -----
        **Noise Application:**
        
        For analog detectors, noise is added independently at each pixel.
        For photon counting, photons are distributed across all pixels
        according to the signal probability distribution, then reconverted
        to same scale as input for direct comparison.
        
        **Performance:**
        
        Simulation time scales with:
        - Model evaluation time (dominates for complex models)
        - Array size (n_time × n_energy)
        - Noise generation method
        
        Typical times:
        - Simple model, 200×500 array: ~0.1-1 second
        - Complex model with time-dependence: ~1-10 seconds
        - Photon counting slightly slower than analog
        
        **Memory:**
        
        Three arrays stored (clean, noisy, noise), each:
        - Size: n_time × n_energy × 8 bytes (float64)
        - Example: 200×500 = ~2.4 MB per array, ~7.2 MB total
        
        See Also
        --------
        simulate_1D : Simulate single spectrum
        simulate_N : Generate multiple 2D realizations
        plot_comparison : Visualize results
        save_data : Save to HDF5 file
        """
        # Generate clean spectrum from model
        clean_data = self.generate_clean_data(dim=2)
        
        # Add noise
        noisy_data, noise = self.add_noise(clean_data, dim=2)
        
        # Store for later use (e.g., plotting, SNR calculation)
        self.data_clean = clean_data
        self.data_noisy = noisy_data
        self.noise = noise
        
        return clean_data, noisy_data, noise
    
    #
    def simulate_N(
        self,
        N: int,
        dim: int = 2,
        t_ind: int = 0,
        show_progress: bool = True
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        Generate N simulated datasets with independent noise realizations.
        
        Generates the clean data ONCE from the model, then adds N independent
        noise realizations. Use for statistical analysis of fitting algorithms
        and uncertainty quantification or machine learning model training.
        
        Parameters
        ----------
        N : int
            Number of datasets to generate (number of noise realizations)
        dim : {1, 2}, default=2
            Dimensionality:
            - 1: Generate 1D spectra
            - 2: Generate 2D spectra
        t_ind : int, default=0
            Time index for 1D simulations (ignored for dim=2)
        show_progress : bool, default=True
            Print progress updates during generation
        
        Returns
        -------
        clean_data : ndarray
            Single clean dataset (1D or 2D depending on dim).
            Same for all N realizations (generated once).
        noisy_data_list : list of ndarray
            List of N noisy datasets, each with independent noise.
            Each element has same shape as clean_data.
        noise_list : list of ndarray
            List of N noise realizations (noisy - clean for each dataset).
            Each element has same shape as clean_data.
        
        Examples
        --------
        >>> # Generate 20 independent noisy datasets
        >>> sim = Simulator(model, noise_level=0.05)
        >>> clean, noisy_list, noise_list = sim.simulate_N(N=20, dim=2)
        >>> 
        >>> # Fit each dataset and analyze parameter distribution
        >>> fitted_params = []
        >>> for noisy_data in noisy_list:
        ...     file.data = noisy_data
        ...     file.fit_2Dmodel('test', fit=2)
        ...     fitted_params.append(model.lmfit_pars['amplitude'].value)
        >>> 
        >>> # Check parameter recovery
        >>> true_value = model.lmfit_pars['amplitude'].value
        >>> mean_fitted = np.mean(fitted_params)
        >>> std_fitted = np.std(fitted_params)
        >>> print(f"True: {true_value:.2f}")
        >>> print(f"Mean fitted: {mean_fitted:.2f} ± {std_fitted:.2f}")
        
        >>> # Analyze noise statistics
        >>> noise_mean = np.mean(noise_list, axis=0)
        >>> noise_std = np.std(noise_list, axis=0)
        >>> 
        >>> # Should be close to zero (unbiased)
        >>> print(f"Noise mean: {np.mean(noise_mean):.2e}")
        >>> # Should match noise_level * signal scale
        >>> print(f"Noise std: {np.mean(noise_std):.2e}")
        
        >>> # Save multiple realizations for later use
        >>> clean, noisy_list, noise_list = sim.simulate_N(N=100, dim=2)
        >>> sim.save_data(
        ...     filepath='simulations/batch_001.h5',
        ...     N_data=noisy_list
        ... )
        
        >>> # Test convergence of fitted parameters with N
        >>> for n_datasets in [5, 10, 20, 50]:
        ...     clean, noisy_list, _ = sim.simulate_N(N=n_datasets, dim=2)
        ...     # ... fit each and compute parameter statistics ...
        ...     print(f"N={n_datasets}: parameter std = {param_std:.3f}")
        
        Notes
        -----
        **Efficiency:**
        
        Generating clean data once and adding N noise realizations is much
        faster than generating N complete simulations:
        
        - This method: 1 model evaluation + N noise additions
        - N separate simulate_2D calls: N model evaluations + N noise additions
        
        For complex models where evaluation is slow, this can save
        minutes to hours of computation time.
        
        **Statistical Analysis:**
        
        This function enables:
        - Monte Carlo analysis of fitting uncertainty
        - Algorithm validation (can recover true parameters?)
        - Bias detection (systematic fitting errors)
        - Confidence interval validation (coverage probability)
        - Experimental design optimization (required SNR)
        
        **Memory Considerations:**
        
        All N datasets stored in memory as lists:
        - Memory usage: N × (n_time × n_energy × 8 bytes)
        - Example: 100 datasets of 200×500 = ~800 MB
        
        For very large N or large arrays, consider:
        - Processing in batches
        - Saving to disk incrementally
        - Using generator pattern instead of list
        
        **Progress Display:**
        
        When show_progress=True, prints:
        - "Generating clean data from model... Done"
        - "Adding noise to dataset N/M" (updates in place)
        - "Generated M noisy datasets successfully"
        
        Set show_progress=False for batch processing or when redirecting output.
        
        See Also
        --------
        simulate_1D : Single 1D simulation
        simulate_2D : Single 2D simulation
        save_data : Save multiple datasets to HDF5
        """
        # Generate clean data ONCE
        if show_progress:
            print('Generating clean data from model...', end=' ')
        clean_data = self.generate_clean_data(dim=dim, t_ind=t_ind)
        if show_progress:
            print('Done')
        
        # Generate N independent noise realizations
        noisy_data_list = []
        noise_list = []
        
        for i in range(N):
            if show_progress:
                print(f'Adding noise to dataset {i+1}/{N}', end='\r')
            
            # Just add noise to the same clean data
            noisy_data, noise = self.add_noise(clean_data, dim=dim)
            
            noisy_data_list.append(noisy_data)
            noise_list.append(noise)
        
        if show_progress:
            print(f'Generated {N} noisy datasets successfully')
        
        # Store last realization for later use
        self.data_clean = clean_data
        self.data_noisy = noisy_data_list[-1]
        self.noise = noise_list[-1]
        
        return clean_data, noisy_data_list, noise_list
    
    #
    def _generate_noise_analog_1D(self, signal: np.ndarray) -> np.ndarray:
        """
        Generate 1D noise array for analog detectors
        
        Parameters:
            signal: Clean signal array
            
        Returns:
            Noise array with same shape as signal
        """
        if self.noise_type == 'none':
            return np.zeros_like(signal)
        
        elif self.noise_type == 'gaussian':
            # Gaussian noise with amplitude proportional to noise_level
            noise_amplitude = self.noise_level * np.max(np.abs(signal))
            return cast(np.ndarray, np.random.normal(0, noise_amplitude, signal.shape))
        
        elif self.noise_type == 'poisson':
            # Poisson noise: scale signal to photon counts, add noise, scale back
            # Avoid negative values in Poisson distribution
            signal_positive = np.abs(signal)
            
            # Scale signal to photon counts (higher = less relative noise)
            scale_factor = 1.0 / (self.noise_level + 1e-10)  # Avoid division by zero
            signal_scaled = signal_positive * scale_factor
            
            # Generate Poisson noise
            noisy_scaled = np.random.poisson(signal_scaled)
            
            # Scale back and compute noise component
            signal_noisy = noisy_scaled / scale_factor
            noise = signal_noisy - signal_positive
            
            # Restore original sign
            return cast(np.ndarray, noise * np.sign(signal))
        
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}. "
                           "Use 'poisson', 'gaussian', or 'none'")
    
    #
    def _generate_noise_analog_2D(self, signal: np.ndarray) -> np.ndarray:
        """
        Generate 2D noise array for analog detectors
        
        Parameters:
            signal: Clean 2D signal array
            
        Returns:
            2D noise array with same shape as signal
        """
        if self.noise_type == 'none':
            return np.zeros_like(signal)
        
        elif self.noise_type == 'gaussian':
            # Gaussian noise with amplitude proportional to noise_level
            noise_amplitude = self.noise_level * np.max(np.abs(signal))
            return cast(np.ndarray, np.random.normal(0, noise_amplitude, signal.shape))
        
        elif self.noise_type == 'poisson':
            # Poisson noise: scale signal to photon counts, add noise, scale back
            signal_positive = np.abs(signal)
            
            # Scale signal to photon counts
            scale_factor = 1.0 / (self.noise_level + 1e-10)
            signal_scaled = signal_positive * scale_factor
            
            # Generate Poisson noise
            noisy_scaled = np.random.poisson(signal_scaled)
            
            # Scale back and compute noise component
            signal_noisy = noisy_scaled / scale_factor
            noise = signal_noisy - signal_positive
            
            # Restore original sign
            return cast(np.ndarray, noise * np.sign(signal))
        
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}. "
                           "Use 'poisson', 'gaussian', or 'none'")
    
    #
    def _sample_photons_1D(self, signal: np.ndarray) -> np.ndarray:
        """
        Sample photons for 1D photon counting detection.

        Each energy pixel independently draws from a Poisson distribution
        where the expected count is proportional to the signal intensity.
        The signal is scaled so the total expected counts across all energy
        pixels equals counts_per_delay.

        Parameters:
            signal: Clean signal array (represents expected count rate)

        Returns:
            Noisy data array in same units as input signal
        """
        signal_positive = np.abs(signal)
        total_signal = np.sum(signal_positive)

        if total_signal == 0:
            return np.zeros_like(signal)
        if self.counts_per_delay is None:
            raise ValueError("counts_per_delay must be defined for photon counting simulation")

        # Scale signal to expected photon counts
        # Total expected counts across all pixels = counts_per_delay
        scale_factor = self.counts_per_delay / total_signal
        expected_counts = signal_positive * scale_factor

        # Independent Poisson draw per pixel
        photon_counts = np.random.poisson(expected_counts)

        # Scale back to original signal units and restore sign
        # (negative signal = bleach/emission, positive = absorption)
        noisy_data = photon_counts / scale_factor * np.sign(signal)

        return cast(np.ndarray, noisy_data)

    #
    def _sample_photons_2D(self, signal: np.ndarray) -> np.ndarray:
        """
        Sample photons for 2D photon counting detection.

        Applies independent Poisson noise per pixel across the full 2D array.
        The signal is scaled so the average total expected counts per time step
        equals counts_per_delay. Time steps with stronger signal naturally
        accumulate more photons (better SNR), matching real experiments where
        each time delay is measured for the same integration time.

        Parameters:
            signal: Clean 2D signal array (shape: [n_time, n_energy])

        Returns:
            Noisy 2D data array in same units as input signal
        """
        signal_positive = np.abs(signal)

        # Average total signal per time step
        row_totals = np.sum(signal_positive, axis=1)
        mean_row_total = np.mean(row_totals)

        if mean_row_total == 0:
            return np.zeros_like(signal)
        if self.counts_per_delay is None:
            raise ValueError("counts_per_delay must be defined for photon counting simulation")

        # Scale so the average row has counts_per_delay total expected counts
        scale_factor = self.counts_per_delay / mean_row_total
        expected_counts = signal_positive * scale_factor

        # Independent Poisson draw per pixel
        photon_counts = np.random.poisson(expected_counts)

        # Scale back to original signal units and restore sign
        # (negative signal = bleach/emission, positive = absorption)
        noisy_data = photon_counts / scale_factor * np.sign(signal)

        return cast(np.ndarray, noisy_data)
    
    #
    def set_noise_level(self, noise_level: float) -> None:
        """Update noise level (analog detectors only)"""
        if self.detection != 'analog':
            print("Warning: noise_level only applies to analog detection")
        self.noise_level = noise_level
    
    #
    def set_noise_type(self, noise_type: str) -> None:
        """Update noise type (analog detectors only)"""
        if self.detection != 'analog':
            print("Warning: noise_type only applies to analog detection")
        self.noise_type = noise_type.lower()
    
    #
    def set_counts_per_delay(self, counts_per_delay: int) -> None:
        """Update counts per delay (photon counting only)"""
        if self.detection != 'photon_counting':
            print("Warning: counts_per_delay only applies to photon_counting detection")
        self.counts_per_delay = counts_per_delay
    
    #
    def set_count_rate(self, count_rate: float, integration_time: Optional[float] = None) -> None:
        """
        Update count rate (photon counting only)
        
        Parameters:
            count_rate: Photon rate in Hz
            integration_time: Integration time per delay in seconds (if None, uses existing value)
        """
        if self.detection != 'photon_counting':
            print("Warning: count_rate only applies to photon_counting detection")
            return
        
        self.count_rate = count_rate
        if integration_time is not None:
            self.integration_time = integration_time
        
        if self.integration_time is not None:
            self.counts_per_delay = int(self.count_rate * self.integration_time)
        else:
            raise ValueError("integration_time must be set to calculate counts_per_delay from count_rate")
    
    #
    def set_seed(self, seed: Optional[int]) -> None:
        """Update random seed"""
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    #
    def get_SNR(self, scale: str = 'linear') -> float:
        """
        Calculate Signal-to-Noise Ratio (SNR).
        
        Computes SNR from the most recently simulated data using power-based
        definition: SNR = signal_power / noise_power.
        
        Parameters
        ----------
        scale : {'linear', 'dB'}, default='linear'
            Output scale:
            - 'linear': SNR as ratio (e.g., 25.0)
            - 'dB': SNR in decibels (e.g., 13.98 dB)
        
        Returns
        -------
        float
            SNR value in requested scale.
            Returns np.inf if noise_power is exactly zero.
        
        Raises
        ------
        ValueError
            If no simulated data available (must call simulate_1D/2D/N first),
            or if scale is not 'linear' or 'dB'.
        
        Examples
        --------
        >>> # Calculate SNR after simulation
        >>> sim = Simulator(model, noise_level=0.05)
        >>> clean, noisy, noise = sim.simulate_2D()
        >>> 
        >>> snr_linear = sim.get_SNR(scale='linear')
        >>> print(f"SNR: {snr_linear:.1f}")
        SNR: 25.3
        >>> 
        >>> snr_db = sim.get_SNR(scale='dB')
        >>> print(f"SNR: {snr_db:.1f} dB")
        SNR: 14.0 dB
        
        >>> # Compare SNR across noise levels
        >>> for noise_level in [0.01, 0.05, 0.10, 0.20]:
        ...     sim.set_noise_level(noise_level)
        ...     sim.simulate_2D()
        ...     snr = sim.get_SNR()
        ...     print(f"Noise {noise_level:.2f}: SNR = {snr:.1f}")
        Noise 0.01: SNR = 625.0
        Noise 0.05: SNR = 25.0
        Noise 0.10: SNR = 6.2
        Noise 0.20: SNR = 1.6
        
        >>> # Plot SNR vs photon count
        >>> counts = [100, 500, 1000, 5000, 10000]
        >>> snrs = []
        >>> for count in counts:
        ...     sim = Simulator(model, detection='photon_counting',
        ...                     counts_per_delay=count)
        ...     sim.simulate_2D()
        ...     snrs.append(sim.get_SNR())
        >>> plt.loglog(counts, snrs, 'o-')
        >>> plt.xlabel('Counts per delay')
        >>> plt.ylabel('SNR')
        
        Notes
        -----
        **SNR Definition:**
        
        Uses power-based (energy) definition:
        
        SNR_linear = (mean(signal²)) / (mean(noise²))
        SNR_dB = 10 × log₁₀(SNR_linear)
        
        This differs from amplitude-based definition (20 log₁₀) by factor of 2.
        Power-based is standard in signal processing and communications.
        
        **Interpretation:**
        
        Linear scale:
        - SNR = 1: Signal and noise have equal power (marginal)
        - SNR = 10: Signal 10× stronger than noise (good)
        - SNR = 100: Signal 100× stronger than noise (excellent)
        
        dB scale:
        - 0 dB: Equal signal and noise
        - 10 dB: 10× signal power (good)
        - 20 dB: 100× signal power (excellent)
        - Each 10 dB = 10× power ratio
        
        **Typical Values:**
        
        For spectroscopy data:
        - SNR < 5 (< 7 dB): Difficult to fit reliably
        - SNR 5-20 (7-13 dB): Good quality, typical experimental data
        - SNR 20-100 (13-20 dB): High quality
        - SNR > 100 (> 20 dB): Exceptional, near ideal
        
        **Limitations:**
        
        This is a global SNR averaged over entire spectrum. Local SNR
        may vary significantly, especially for:
        - Weak features vs. strong peaks
        - Time-dependent signals (varying amplitude)
        - Non-uniform noise (detector artifacts)
        
        For accurate local SNR, compute on regions of interest separately.
        
        See Also
        --------
        simulate_1D : Must call before get_SNR
        simulate_2D : Must call before get_SNR
        plot_comparison : Shows SNR in title
        """
        if self.data_clean is None or self.noise is None:
            raise ValueError("No simulated data available. Run simulate_1D or simulate_2D first.")
        #$% enable ROI input here
        signal_power = np.mean(self.data_clean**2)
        noise_power = np.mean(self.noise**2)
        
        if noise_power == 0:
            return float(np.inf)
        
        if scale == 'linear':
            return float(signal_power / noise_power)
        elif scale == 'dB':
            return float(10 * np.log10(signal_power / noise_power))
        else:
            raise ValueError("scale must be either 'linear' or 'dB'")
    
    #
    def plot_comparison(self, t_ind: int = 0, dim: int = 1, SNR_scale: str = 'linear') -> None:
        """
        Plot comparison of clean vs noisy data.
        
        Creates visualization showing clean model data, noisy simulated data,
        and noise component side-by-side. Essential for visually assessing
        simulation quality and noise characteristics.
        
        Parameters
        ----------
        t_ind : int, default=0
            Time index for 1D plots (ignored for dim=2)
        dim : {1, 2}, default=1
            Dimensionality:
            - 1: Create 1D plot with clean, noisy, and noise curves
            - 2: Create three-panel 2D plot (clean, noisy, noise)
        SNR_scale : {'linear', 'dB'}, default='linear'
            Scale for SNR display in title:
            - 'linear': Show as ratio (e.g., "SNR: 25.0 linear")
            - 'dB': Show in decibels (e.g., "SNR: 14.0 dB")
        
        Examples
        --------
        >>> # 1D comparison
        >>> sim = Simulator(model, noise_level=0.05)
        >>> sim.simulate_1D(t_ind=0)
        >>> sim.plot_comparison(dim=1)
        
        >>> # 2D comparison with dB scale
        >>> sim = Simulator(model, noise_level=0.05)
        >>> sim.simulate_2D()
        >>> sim.plot_comparison(dim=2, SNR_scale='dB')
        
        >>> # Compare different noise levels visually
        >>> fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        >>> for i, noise_level in enumerate([0.01, 0.05, 0.10]):
        ...     sim.set_noise_level(noise_level)
        ...     sim.simulate_1D()
        ...     # ... manual plotting on axes[i] ...
        
        >>> # Check photon counting vs analog
        >>> sim_analog = Simulator(model, detection='analog', noise_level=0.05)
        >>> sim_photon = Simulator(model, detection='photon_counting',
        ...                         counts_per_delay=1000)
        >>> sim_analog.simulate_2D()
        >>> sim_photon.simulate_2D()
        >>> # ... compare visually ...
        
        Notes
        -----
        **1D Plot Layout:**
        
        Single plot with three traces:
        - Clean: Black line (ground truth)
        - Noisy: Red scatter points (simulated data)
        - Noise: Gray line (noise component)
        
        Scatter points for noisy data help visualize noise granularity.
        
        **2D Plot Layout:**
        
        Three side-by-side panels:
        - Left: Clean model data
        - Center: Noisy simulated data (with SNR in title)
        - Right: Noise component (difference)
        
        All use same colormap from model.plot_config for consistency.
        
        **Visual Assessment:**
        
        Good simulation should show:
        - Noisy data follows clean data trend
        - Noise is randomly distributed (no patterns)
        - SNR appropriate for intended use case
        - Peak features still distinguishable in noisy data
        
        If noise dominates signal (SNR << 1), features may be
        completely obscured - increase signal or reduce noise.
        
        **Configuration:**
        
        Plot uses model.plot_config for:
        - Axis labels (energy/time labels)
        - Axis direction (e.g., reversed energy)
        - Colormap (for 2D plots)
        - DPI settings
        
        This ensures consistency with other trspecfit plots.
        
        See Also
        --------
        simulate_1D : Generate 1D data to plot
        simulate_2D : Generate 2D data to plot
        get_SNR : SNR calculation shown in title
        """
        detection_str = f' [{self.detection}]'
        plt_title = f'Simulated Data (SNR: {self.get_SNR(scale=SNR_scale):.1f} {SNR_scale}){detection_str}'

        if dim == 1:
            if self.data_clean is None:
                self.simulate_1D(t_ind)
            if self.data_clean is None or self.data_noisy is None or self.noise is None:
                raise RuntimeError("Simulation data not available for plotting")
            
            # Get config from model
            config = self.model.plot_config
            
            # Plot with noisy data as scatter
            uplt.plot_1D(
                data=[self.data_clean, self.data_noisy, self.noise],
                x=self.model.energy,
                config=config,
                title=plt_title,
                legend=['Clean', 'Noisy', 'Noise'],
                linestyles=['-', '', '-'],  # Empty string = no line for noisy data
                markers=[None, 'o', None],  # Scatter points for noisy data
                markersizes=[6, 3, 6]  # Smaller markers for noisy data
            )
        
        elif dim == 2:
            if self.data_clean is None:
                self.simulate_2D()
            if self.data_clean is None or self.data_noisy is None or self.noise is None:
                raise RuntimeError("Simulation data not available for plotting")
            
            # Get config from model
            config = self.model.plot_config
            
            # Create 3-panel plot
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            # Determine axis direction
            x_dir_reversed = (config.x_dir == 'rev')
            
            # Clean data
            im1 = axes[0].pcolormesh(self.model.energy,
                                     self.model.time, 
                                     self.data_clean,
                                     shading='nearest',
                                     cmap=config.z_colormap
                                     )
            axes[0].set_title('Clean Model Data')
            axes[0].set_xlabel(config.x_label)
            axes[0].set_ylabel(config.y_label)
            if x_dir_reversed:
                axes[0].invert_xaxis()
            #$% add x_type and y_type (lin vs log) here 
            plt.colorbar(im1, ax=axes[0])
            
            # Noisy data
            im2 = axes[1].pcolormesh(self.model.energy,
                                     self.model.time, 
                                     self.data_noisy,
                                     shading='nearest',
                                     cmap=config.z_colormap
                                     )
            axes[1].set_title(plt_title)
            axes[1].set_xlabel(config.x_label)
            axes[1].set_ylabel(config.y_label)
            if x_dir_reversed:
                axes[1].invert_xaxis()
            plt.colorbar(im2, ax=axes[1])
            
            # Noise
            im3 = axes[2].pcolormesh(self.model.energy,
                                     self.model.time, 
                                     self.noise,
                                     shading='nearest',
                                     cmap=config.z_colormap
                                     )
            axes[2].set_title('Noise (Simulated - Clean)')
            axes[2].set_xlabel(config.x_label)
            axes[2].set_ylabel(config.y_label)
            if x_dir_reversed:
                axes[2].invert_xaxis()
            plt.colorbar(im3, ax=axes[2])
            
            plt.tight_layout()
            plt.show()
    
    #
    def save_data(
        self,
        filepath: Optional[str] = None,
        save_format: str = 'hdf5',
        N_data: Optional[List[np.ndarray]] = None,
        overwrite: bool = True
    ) -> None:
        """
        Save simulated data to file with metadata.
        
        Exports simulated data in HDF5 format with complete metadata including
        model parameters, noise settings, and experimental axes. Essential for
        sharing simulated datasets and ensuring reproducibility.
        
        Parameters
        ----------
        filepath : str or Path, optional
            Path where to save data. If None, uses default:
            './simulated_data/simulated_data.h5'
            If provided path doesn't include 'simulated_data' directory,
            it will be automatically placed there.
        save_format : str, default='hdf5'
            File format. Currently only 'hdf5' supported.
            Future: could add .mat, .npz, etc.
        N_data : list of ndarray, optional
            Multiple noisy datasets from simulate_N() to save.
            If None, saves single dataset from simulate_1D() or simulate_2D().
        overwrite : bool, default=True
            If True, overwrite existing files.
            If False, raise FileExistsError if file exists.
        
        Raises
        ------
        ValueError
            If no simulated data available (must call simulate first)
        FileExistsError
            If file exists and overwrite=False
        
        Examples
        --------
        >>> # Save single simulation
        >>> sim = Simulator(model, noise_level=0.05, seed=42)
        >>> clean, noisy, noise = sim.simulate_2D()
        >>> sim.save_data('simulation_001.h5')
        Data saved to: ./simulated_data/simulation_001.h5
        
        >>> # Save multiple realizations
        >>> clean, noisy_list, noise_list = sim.simulate_N(N=50, dim=2)
        >>> sim.save_data(
        ...     filepath='batch_simulation.h5',
        ...     N_data=noisy_list
        ... )
        Data saved to: ./simulated_data/batch_simulation.h5
        
        >>> # Prevent accidental overwrites
        >>> sim.save_data('important_data.h5', overwrite=False)
        FileExistsError: File already exists: ./simulated_data/important_data.h5
        Set overwrite=True to overwrite, or provide a different filepath.
        
        >>> # Load saved data later
        >>> import h5py
        >>> with h5py.File('simulated_data/simulation_001.h5', 'r') as f:
        ...     energy = f['energy'][:]
        ...     time = f['time'][:]
        ...     clean = f['clean_data'][:]
        ...     noisy = f['simulated_data/000000'][:]
        ...     
        ...     # Read metadata
        ...     noise_level = f['metadata'].attrs['noise_level']
        ...     model_params = f['metadata'].attrs['model_parameters']
        
        Notes
        -----
        **HDF5 File Structure:**

        ::

            /
            ├── energy              (dataset: 1D array)
            ├── time                (dataset: 1D array, empty for 1D simulations)
            ├── clean_data          (dataset: 1D or 2D array)
            ├── simulated_data/     (group)
            │   ├── 000000          (dataset: first noisy realization)
            │   ├── 000001          (dataset: second noisy realization)
            │   └── ...
            └── metadata/           (group with attributes)
                ├── detection       (attribute: 'analog' or 'photon_counting')
                ├── noise_level     (attribute: analog noise level)
                ├── noise_type      (attribute: analog noise type)
                ├── counts_per_delay (attribute: photon counting counts)
                ├── count_rate      (attribute: photon counting rate, if set)
                ├── integration_time      (attribute: photon counting integration time, if set)
                ├── seed            (attribute: random seed, if set)
                ├── dimension       (attribute: 1 or 2)
                ├── n_datasets      (attribute: number of noisy datasets)
                ├── model_parameters (attribute: JSON string of all parameters)
                └── model_name      (attribute: model name)

        **Why HDF5?**
        
        HDF5 format chosen because:
        - Efficient for large multidimensional arrays
        - Self-describing (metadata embedded)
        - Widely supported (Python, MATLAB, Igor, etc.)
        - Allows partial loading (don't need entire file in memory)
        - Standard in scientific computing
        
        **Model Parameters:**

        All model parameters saved as JSON string in metadata for complete
        reproducibility. Includes:

        - Parameter values
        - vary flags (which parameters were free)
        - Bounds (min/max)
        - Expressions (parameter constraints)
        
        This allows exact recreation of the model used for simulation.
        
        **File Organization:**

        Default directory structure::

            project_directory/
            └── simulated_data/
                ├── simulation_001.h5
                ├── simulation_002.h5
                └── batch_001.h5

        Keeps simulated data organized and separate from experimental data.
        
        **Multiple Datasets:**

        When N_data provided (from simulate_N), all realizations saved in
        simulated_data group with sequential names:

        - 000000, 000001, ..., 000099 for 100 datasets
        - Zero-padded for proper sorting
        
        Clean data saved once (same for all realizations).
        
        **Loading Data:**

        Standard h5py usage::

            import h5py

            with h5py.File('simulated_data/data.h5', 'r') as f:
                # Load axes
                energy = f['energy'][:]
                time = f['time'][:]

                # Load clean data
                clean = f['clean_data'][:]

                # Load all noisy datasets
                noisy_datasets = []
                for key in sorted(f['simulated_data'].keys()):
                    noisy_datasets.append(f['simulated_data'][key][:])

                # Load metadata
                detection = f['metadata'].attrs['detection']
                n_datasets = f['metadata'].attrs['n_datasets']
            
        See Also
        --------
        simulate_N : Generate multiple datasets to save
        simulate_1D : Generate 1D data
        simulate_2D : Generate 2D data
        h5py : Python HDF5 library
        """

        if self.data_noisy is None and N_data is None:
            raise ValueError("No simulated data available. Run simulate_1D, simulate_2D, or simulate_N first.")
        
        # Create simulated_data directory if it doesn't exist
        sim_dir = os.path.join(os.getcwd(), 'simulated_data')
        if not os.path.exists(sim_dir):
            os.makedirs(sim_dir)
        
        # Set default filepath
        if filepath is None:
            if save_format == 'hdf5':
                filepath = os.path.join(sim_dir, 'simulated_data.h5')
        else:
            # User provided filepath - make sure it's in simulated_data directory
            if not filepath.startswith(sim_dir):
                filepath = os.path.join(sim_dir, os.path.basename(filepath))
        
        # Check if file exists and handle overwrite
        if filepath is None:
            raise ValueError("filepath could not be resolved")
        if os.path.exists(filepath) and not overwrite:
            raise FileExistsError(
                f"File already exists: {filepath}\n"
                f"Set overwrite=True to overwrite, or provide a different filepath."
            )
        
        if save_format == 'hdf5':
            self._save_hdf5(filepath, N_data)
        
        # add other formats here
        
        else:
            raise ValueError(f"Unknown save format: {save_format}. Use 'hdf5'.")
        
        print(f"Data saved to: {filepath}")

    #
    def _save_hdf5(self, filepath: str, N_data: Optional[List[np.ndarray]] = None) -> None:
        """
        Save data to HDF5 format with proper structure
        
        HDF5 structure:
            /energy         - energy axis array
            /time           - time axis array (empty if 1D)
            /clean_data     - clean data without noise
            /simulated_data/000000 - first noisy dataset
            /simulated_data/000001 - second noisy dataset
            ...
            /metadata       - group containing simulation parameters
        """      
        with h5py.File(filepath, 'w') as f:
            # Save axes at root level
            f.create_dataset('energy', data=self.model.energy)
            
            # Handle time axis - check if it exists and has valid data
            if self.model.time is not None and len(self.model.time) > 0:
                f.create_dataset('time', data=self.model.time)
            else:
                # For 1D simulations, save empty array
                f.create_dataset('time', data=np.array([]))
            
            # Save clean data at root level
            if self.data_clean is not None:
                f.create_dataset('clean_data', data=self.data_clean)
            
            # Create simulated_data group for noisy datasets
            sim_group = f.create_group('simulated_data')
            
            # Save noisy data - either from N_data list or single simulation
            if N_data is not None:
                # Multiple datasets from simulate_N
                for i, noisy_data in enumerate(N_data):
                    dataset_name = f'{i:06d}'
                    sim_group.create_dataset(dataset_name, data=noisy_data)
            elif self.data_noisy is not None:
                # Single dataset
                sim_group.create_dataset('000000', data=self.data_noisy)
            
            # Save metadata group at root level
            meta = f.create_group('metadata')
            meta.attrs['detection'] = self.detection
            
            # Save detection-specific parameters
            if self.detection == 'analog':
                meta.attrs['noise_level'] = self.noise_level
                meta.attrs['noise_type'] = self.noise_type
            elif self.detection == 'photon_counting':
                meta.attrs['counts_per_delay'] = self.counts_per_delay
                if self.count_rate is not None:
                    meta.attrs['count_rate'] = self.count_rate
                if self.integration_time is not None:
                    meta.attrs['integration_time'] = self.integration_time
            
            if self.seed is not None:
                meta.attrs['seed'] = self.seed
            
            # Determine dimensionality from clean data
            if self.data_clean is not None:
                if self.data_clean.ndim == 1:
                    meta.attrs['dimension'] = 1
                else:
                    meta.attrs['dimension'] = 2
            
            # Save number of datasets
            if N_data is not None:
                meta.attrs['n_datasets'] = len(N_data)
            else:
                meta.attrs['n_datasets'] = 1
            
            # Save model parameters as JSON in metadata
            params_dict = {}
            for par_name in self.model.lmfit_pars:
                par = self.model.lmfit_pars[par_name]
                params_dict[par_name] = {
                    'value': float(par.value),
                    'vary': bool(par.vary),
                    'min': float(par.min) if par.min is not None else None,
                    'max': float(par.max) if par.max is not None else None,
                    'expr': par.expr if par.expr else None
                }
            
            # Save as JSON string in metadata
            meta.attrs['model_parameters'] = json.dumps(params_dict, indent=2)
            meta.attrs['model_name'] = self.model.name

    #
    def simulate_parameter_sweep(self, 
                                parameter_sweep: ParameterSweep,
                                N_realizations: int,
                                filepath: str = 'ml_training_data.h5',
                                show_progress: bool = True) -> None:
        """
        Generate ML training dataset by sweeping parameters.
        
        Processes configurations one at a time, immediately saving to disk.
        Memory usage remains constant regardless of parameter space size.
        
        Parameters
        ----------
        parameter_sweep : ParameterSweep
            Generator yielding parameter configurations
        N_realizations : int
            Number of noisy realizations per parameter configuration
        filepath : str, default='ml_training_data.h5'
            HDF5 file path for output
        show_progress : bool, default=True
            Print progress updates during generation
        
        Examples
        --------
        >>> # Set up parameter space
        >>> sweep = ParameterSweep(strategy='random', seed=42)
        >>> sweep.add_uniform('GLP_01_A', 5, 30, n_samples=100)
        >>> sweep.add_uniform('GLP_01_x0', 5, 15, n_samples=100)
        >>> 
        >>> # Generate dataset
        >>> sim = Simulator(model, noise_level=0.05, seed=42)
        >>> sim.simulate_parameter_sweep(
        ...     parameter_sweep=sweep,
        ...     N_realizations=20,
        ...     filepath='training_data.h5'
        ... )
        Processing config 1/100: {'GLP_01_A': 12.5, 'GLP_01_x0': 8.3}
          Saved config 1 with 20 realizations
        ...
        Parameter sweep complete!
        Generated 100 configs × 20 realizations
        Data saved to: ./simulated_data/training_data.h5
        
        Notes
        -----
        **Memory Efficiency:**
        Only one configuration is in memory at a time. Each is immediately
        written to disk before processing the next. Total memory usage is
        independent of parameter space size.
        
        **Resumability:**
        If interrupted, completed configurations are already saved to disk.
        Currently does not support automatic resume (will overwrite file).
        
        **File Structure:**
        See _initialize_sweep_hdf5 for complete HDF5 structure description.
        
        See Also
        --------
        ParameterSweep : Define parameter space to sweep
        simulate_N : Generate multiple noisy realizations
        _initialize_sweep_hdf5 : HDF5 file structure
        _append_config_to_hdf5 : Incremental saving logic
        """

        # Convert to Path object
        filepath_obj = Path(filepath)
        
        # If filepath is just a filename (no directory component), 
        # put it in simulated_data subdirectory
        if filepath_obj.parent == Path('.'):
            sim_dir = Path.cwd() / 'simulated_data'
            sim_dir.mkdir(parents=True, exist_ok=True)
            filepath = str(sim_dir / filepath_obj.name)
        else:
            # User provided a path with directory, use it as-is but ensure parent exists
            filepath_obj.parent.mkdir(parents=True, exist_ok=True)
            filepath = str(filepath_obj)
        
        # Get total number of configurations
        n_configs = parameter_sweep.get_n_configs()
        
        if show_progress:
            print(f"Starting parameter sweep:")
            print(f"  Total configurations: {n_configs}")
            print(f"  Realizations per config: {N_realizations}")
            print(f"  Total datasets: {n_configs * N_realizations}")
            print(f"  Output file: {filepath}")
            print()
        
        # Initialize HDF5 file with structure
        self._initialize_sweep_hdf5(filepath, parameter_sweep, 
                                    N_realizations, n_configs)
        
        # Process each configuration
        for config_idx, param_config in enumerate(parameter_sweep):
            if show_progress:
                # Format parameters nicely
                param_str = ', '.join(f'{k}={v:.3g}' 
                                     for k, v in param_config.items())
                print(f'Processing config {config_idx+1}/{n_configs}: {{{param_str}}}')
            
            # Update model parameters
            param_names = list(param_config.keys())
            param_values = list(param_config.values())
            self.model.update_value(param_values, par_select=param_names)
            
            # Generate noisy realizations for this config
            clean, noisy_list, noise_list = self.simulate_N(
                N=N_realizations,
                dim=2,
                show_progress=False  # Don't clutter output
            )
            
            # Append to HDF5 immediately (memory-efficient)
            self._append_config_to_hdf5(
                filepath, config_idx, param_config, 
                clean, noisy_list
            )
            
            if show_progress:
                print(f'  ✓ Saved config {config_idx+1} with {N_realizations} realizations')
        
        if show_progress:
            print(f'\n{"="*60}')
            print(f'Parameter sweep complete!')
            print(f'Generated {n_configs} configs × {N_realizations} realizations')
            print(f'Total datasets: {n_configs * N_realizations}')
            print(f'Data saved to: {filepath}')
            print(f'{"="*60}')

    #
    def _initialize_sweep_hdf5(self, filepath: str, 
                              parameter_sweep: ParameterSweep,
                              N_realizations: int, 
                              n_configs: int) -> None:
        """
        Create HDF5 file structure for parameter sweep data.
        
        File Structure
        --------------
        /
        ├── energy (dataset)              # Energy axis
        ├── time (dataset)                # Time axis
        ├── metadata/ (group)             # Sweep metadata
        │   ├── attrs: n_configs, n_realizations_per_config, ...
        │   └── attrs: parameter_space (JSON)
        ├── parameter_configs/ (group)    # Parameter configurations
        │   ├── config_000000/ (group)
        │   │   ├── attrs: GLP_01_A, GLP_01_x0, ...
        │   │   ├── attrs: all_parameters (JSON)
        │   │   └── clean (dataset)       # Clean data for this config
        │   ├── config_000001/ (group)
        │   └── ...
        └── simulated_data/ (group)       # Noisy realizations
            ├── config_000000/ (group)
            │   ├── 000000 (dataset)      # First realization
            │   ├── 000001 (dataset)      # Second realization
            │   └── ...
            ├── config_000001/ (group)
            └── ...
        
        Parameters
        ----------
        filepath : str
            Path to HDF5 file to create
        parameter_sweep : ParameterSweep
            Parameter sweep object (for metadata)
        N_realizations : int
            Number of noisy realizations per config
        n_configs : int
            Total number of parameter configurations
        """
        with h5py.File(filepath, 'w') as f:
            # Save axes (same for all configs)
            f.create_dataset('energy', data=self.model.energy)
            if self.model.time is not None and len(self.model.time) > 0:
                f.create_dataset('time', data=self.model.time)
            else:
                f.create_dataset('time', data=np.array([]))
            
            # Create groups for organization
            f.create_group('parameter_configs')
            f.create_group('simulated_data')
            
            # Save sweep metadata
            meta = f.create_group('metadata')
            meta.attrs['n_configs'] = n_configs
            meta.attrs['n_realizations_per_config'] = N_realizations
            meta.attrs['total_datasets'] = n_configs * N_realizations
            
            # Simulator settings
            meta.attrs['detection'] = self.detection
            if self.detection == 'analog':
                meta.attrs['noise_level'] = self.noise_level
                meta.attrs['noise_type'] = self.noise_type
            elif self.detection == 'photon_counting':
                meta.attrs['counts_per_delay'] = self.counts_per_delay
                if self.count_rate is not None:
                    meta.attrs['count_rate'] = self.count_rate
                if self.integration_time is not None:
                    meta.attrs['integration_time'] = self.integration_time
            
            if self.seed is not None:
                meta.attrs['seed'] = self.seed
            
            # Parameter sweep settings
            meta.attrs['sweep_strategy'] = parameter_sweep.strategy
            meta.attrs['sweep_seed'] = parameter_sweep.seed if parameter_sweep.seed else 'None'
            
            # Save parameter space definition as JSON
            param_space = {}
            for par_name, spec in parameter_sweep.parameter_specs.items():
                # Convert numpy arrays to lists for JSON serialization
                spec_copy = spec.copy()
                if 'values' in spec_copy:
                    spec_copy['values'] = spec_copy['values'].tolist()
                param_space[par_name] = spec_copy
            
            meta.attrs['parameter_space'] = json.dumps(param_space, indent=2)
            
            # Dimension info
            if self.model.time is not None and len(self.model.time) > 0:
                meta.attrs['dimension'] = 2
            else:
                meta.attrs['dimension'] = 1

    #
    def _append_config_to_hdf5(self, filepath: str, 
                               config_idx: int,
                               param_config: Dict[str, float],
                               clean: np.ndarray,
                               noisy_list: List[np.ndarray]) -> None:
        """
        Append single parameter configuration and its realizations to HDF5.
        
        Parameters
        ----------
        filepath : str
            Path to HDF5 file
        config_idx : int
            Configuration index (for naming)
        param_config : dict
            Parameter values for this configuration
        clean : ndarray
            Clean (noiseless) data
        noisy_list : list of ndarray
            List of noisy realizations
        """
        with h5py.File(filepath, 'a') as f:
            # Create group for this configuration
            config_name = f'config_{config_idx:06d}'
            configs_group = require_group(f['parameter_configs'], 'parameter_configs')
            config_group = configs_group.create_group(config_name)
            
            # Save swept parameters as attributes
            for par_name, value in param_config.items():
                config_group.attrs[par_name] = float(value)
            
            # Save ALL model parameters as JSON (complete state)
            params_dict = {}
            for par_name in self.model.lmfit_pars:
                par = self.model.lmfit_pars[par_name]
                params_dict[par_name] = {
                    'value': float(par.value),
                    'vary': bool(par.vary),
                }
            config_group.attrs['all_parameters'] = json.dumps(params_dict)
            
            # Save clean data for this configuration
            config_group.create_dataset('clean', data=clean)
            
            # Save noisy realizations
            simulated_group = require_group(f['simulated_data'], 'simulated_data')
            data_group = simulated_group.create_group(config_name)
            for real_idx, noisy_data in enumerate(noisy_list):
                data_group.create_dataset(f'{real_idx:06d}', data=noisy_data)
                
