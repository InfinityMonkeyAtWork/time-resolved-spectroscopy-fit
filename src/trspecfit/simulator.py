from trspecfit.mcp import Model
import copy
import numpy as np
import matplotlib.pyplot as plt
import json
import h5py
import pandas as pd
from trspecfit.utils import plot as uplt
from trspecfit.config.plot import PlotConfig

#
#
class Simulator:
    """
    Simulate 2D time- and energy-resolved spectroscopy data with noise
    
    This class generates synthetic data based on a model, adding realistic noise
    to simulate experimental measurements. It can be used for testing fitting
    algorithms, exploring parameter sensitivity, and generating training data.
    
    Attributes:
        model: Model instance to use for simulation
        detection: Detection technique ('analog' or 'photon_counting')
        noise_level: Amplitude of noise relative to signal (analog detectors)
        noise_type: Type of noise ('poisson', 'gaussian', 'none') (analog detectors)
        counts_per_cycle: Total photon count per pump-probe cycle (photon counting)
        count_rate: Photon count rate in Hz (photon counting)
        cycle_time: Duration of one pump-probe cycle in seconds (photon counting)
        seed: Random seed for reproducibility
    """
    #
    def __init__(self, model, 
                 detection='analog',
                 noise_level=0.05, 
                 noise_type='poisson',
                 counts_per_cycle=None,
                 count_rate=None,
                 cycle_time=None,
                 seed=None):
        """
        Initialize simulator with a model and noise parameters
        
        Parameters:
            model: Model instance with defined components and parameters
            detection: 'analog' or 'photon_counting'
            noise_level: Noise amplitude (0.0-1.0 for relative noise) - analog only
            noise_type: 'poisson', 'gaussian', or 'none' - analog only
            counts_per_cycle: Total photons per pump-probe cycle - photon counting only
            count_rate: Photon rate in Hz - photon counting only (alternative parameter)
            cycle_time: Pump-probe cycle duration in seconds - photon counting only
            seed: Random seed for reproducibility (None for random)
        """
        self.model = model
        self.detection = detection.lower()
        self.seed = seed
        
        # Analog detector parameters
        self.noise_level = noise_level
        self.noise_type = noise_type.lower()
        
        # Photon counting parameters
        self.counts_per_cycle = counts_per_cycle
        self.count_rate = count_rate
        self.cycle_time = cycle_time
        
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
        self.data_clean = None  # Without noise
        self.data_noisy = None  # With noise
        self.noise = None       # Just the noise component
    
    #
    def _resolve_photon_counting_params(self):
        """
        Resolve photon counting parameters. User can specify either:
        - counts_per_cycle directly, OR
        - count_rate and cycle_time (will compute counts_per_cycle), OR
        - nothing (will estimate from model scale with warning)
        """
        if self.counts_per_cycle is not None:
            # Direct specification takes precedence
            if self.count_rate is not None or self.cycle_time is not None:
                print("Warning: counts_per_cycle specified directly. Ignoring count_rate and cycle_time.")
        elif self.count_rate is not None and self.cycle_time is not None:
            # Calculate from rate and time
            self.counts_per_cycle = int(self.count_rate * self.cycle_time)
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
            
            # Use total integrated signal as estimate
            # This assumes the model amplitudes are in a realistic count rate range
            total_signal = np.sum(np.abs(clean_data))
            avg_per_pixel = total_signal / clean_data.size
            self.counts_per_cycle = int(total_signal)
            
            print(f"WARNING: No photon count specified for photon_counting detection.")
            print(f"Estimating from model: {self.counts_per_cycle:.2e} counts/cycle")
            print(f"  (average ~{avg_per_pixel:.1f} counts/pixel)")
            print(f"For accurate simulation, specify counts_per_cycle or (count_rate, cycle_time).")
            print(f"This estimate assumes your model amplitudes represent realistic count rates.")
        
        # Ensure counts_per_cycle is positive
        if self.counts_per_cycle <= 0:
            raise ValueError(
                f"counts_per_cycle must be positive, got {self.counts_per_cycle}.\n"
                f"Model may have negative or zero signal. Check your model or specify counts_per_cycle manually."
            )
    
    #
    def generate_clean_data(self, dim=2, t_ind=0):
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
        
        return self.data_clean
    
    #
    def add_noise(self, clean_data, dim=2):
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
        
        return noisy_data, noise
    
    #
    def simulate_1D(self, t_ind=0):
        """
        Simulate 1D spectrum (energy-resolved) at a specific time point
        
        Parameters:
            t_ind: Time index for which to generate spectrum
            
        Returns:
            Tuple of (clean_data, noisy_data, noise)
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
    def simulate_2D(self):
        """
        Simulate 2D spectrum (time- and energy-resolved)
            
        Returns:
            Tuple of (clean_data, noisy_data, noise)
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
    def simulate_N(self, N, dim=2, t_ind=0, show_progress=True):
        """
        Generate N simulated datasets with independent noise realizations
        
        Parameters:
            N: Number of datasets to generate
            dim: Dimension (1 for 1D, 2 for 2D)
            t_ind: Time index for 1D simulations (ignored for 2D)
            show_progress: Print progress updates
            
        Returns:
            Tuple of (clean_data, noisy_data_list, noise_list)
            where clean_data is single array and others are lists of N arrays
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
    def _generate_noise_analog_1D(self, signal):
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
            return np.random.normal(0, noise_amplitude, signal.shape)
        
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
            return noise * np.sign(signal)
        
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}. "
                           "Use 'poisson', 'gaussian', or 'none'")
    
    #
    def _generate_noise_analog_2D(self, signal):
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
            return np.random.normal(0, noise_amplitude, signal.shape)
        
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
            return noise * np.sign(signal)
        
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}. "
                           "Use 'poisson', 'gaussian', or 'none'")
    
    #
    def _sample_photons_1D(self, signal):
        """
        Sample photons for 1D photon counting detection
        
        Parameters:
            signal: Clean signal array (represents expected count rate)
            
        Returns:
            Noisy data array with integer photon counts
        """
        # Ensure signal is non-negative
        signal_positive = np.abs(signal)
        
        # Normalize to probability distribution
        total_signal = np.sum(signal_positive)
        if total_signal == 0:
            # No signal, return zeros
            return np.zeros_like(signal)
        
        prob_dist = signal_positive / total_signal
        
        # Sample photons according to multinomial distribution
        photon_counts = np.random.multinomial(self.counts_per_cycle, prob_dist)
        
        # Convert back to same scale as input signal
        # Each photon represents (total_signal / counts_per_cycle) in signal units
        noisy_data = photon_counts * (total_signal / self.counts_per_cycle)
        
        return noisy_data
    
    #
    def _sample_photons_2D(self, signal):
        """
        Sample photons for 2D photon counting detection
        
        Parameters:
            signal: Clean 2D signal array (represents expected count rate)
            
        Returns:
            Noisy 2D data array with integer photon counts
        """
        # Ensure signal is non-negative
        signal_positive = np.abs(signal)
        
        # Flatten for multinomial sampling
        signal_flat = signal_positive.flatten()
        
        # Normalize to probability distribution
        total_signal = np.sum(signal_flat)
        if total_signal == 0:
            # No signal, return zeros
            return np.zeros_like(signal)
        
        prob_dist = signal_flat / total_signal
        
        # Sample photons according to multinomial distribution
        photon_counts_flat = np.random.multinomial(self.counts_per_cycle, prob_dist)
        
        # Reshape back to 2D
        photon_counts = photon_counts_flat.reshape(signal.shape)
        
        # Convert back to same scale as input signal
        # Each photon represents (total_signal / counts_per_cycle) in signal units
        noisy_data = photon_counts * (total_signal / self.counts_per_cycle)
        
        return noisy_data
    
    #
    def set_noise_level(self, noise_level):
        """Update noise level (analog detectors only)"""
        if self.detection != 'analog':
            print("Warning: noise_level only applies to analog detection")
        self.noise_level = noise_level
    
    #
    def set_noise_type(self, noise_type):
        """Update noise type (analog detectors only)"""
        if self.detection != 'analog':
            print("Warning: noise_type only applies to analog detection")
        self.noise_type = noise_type.lower()
    
    #
    def set_counts_per_cycle(self, counts_per_cycle):
        """Update counts per cycle (photon counting only)"""
        if self.detection != 'photon_counting':
            print("Warning: counts_per_cycle only applies to photon_counting detection")
        self.counts_per_cycle = counts_per_cycle
    
    #
    def set_count_rate(self, count_rate, cycle_time=None):
        """
        Update count rate (photon counting only)
        
        Parameters:
            count_rate: Photon rate in Hz
            cycle_time: Cycle duration in seconds (if None, uses existing value)
        """
        if self.detection != 'photon_counting':
            print("Warning: count_rate only applies to photon_counting detection")
            return
        
        self.count_rate = count_rate
        if cycle_time is not None:
            self.cycle_time = cycle_time
        
        if self.cycle_time is not None:
            self.counts_per_cycle = int(self.count_rate * self.cycle_time)
        else:
            raise ValueError("cycle_time must be set to calculate counts_per_cycle from count_rate")
    
    #
    def set_seed(self, seed):
        """Update random seed"""
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    #
    def get_SNR(self, scale='linear'):
        """
        Calculate Signal-to-Noise Ratio (SNR)
        scale: 'linear' or 'dB'

        Returns:
            SNR value (linear or decibels)
        """
        if self.data_clean is None or self.noise is None:
            raise ValueError("No simulated data available. Run simulate_1D or simulate_2D first.")
        
        signal_power = np.mean(self.data_clean**2)
        noise_power = np.mean(self.noise**2)
        
        if noise_power == 0:
            return np.inf
        
        if scale == 'linear':
            return signal_power / noise_power
        elif scale == 'dB':
            return 10 * np.log10(signal_power / noise_power)
    
    #
    def plot_comparison(self, t_ind=0, dim=1, SNR_scale='linear'):
        """
        Plot comparison of clean vs noisy data
        
        Parameters
        ----------
        t_ind : int
            Time index for 1D plots (ignored for 2D)
        dim : int
            Dimension (1 for 1D plot, 2 for 2D plot)
        SNR_scale : str
            'linear' or 'dB' for SNR display in title
        """
        detection_str = f' [{self.detection}]'
        plt_title = f'Simulated Data (SNR: {self.get_SNR(scale=SNR_scale):.1f} {SNR_scale}){detection_str}'

        if dim == 1:
            if self.data_clean is None:
                self.simulate_1D(t_ind)
            
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
    def save_data(self, filepath=None, save_format='hdf5', N_data=None, overwrite=True):
        """
        Save simulated data to file
        
        Parameters:
            filepath: Path where to save data (defaults to './simulated_data/simulated_data.h5')
            save_format: 'hdf5' (default) [others could be added later]
            N_data: For HDF5 - list of multiple noisy datasets (from simulate_N)
                If None, uses self.data_noisy from single simulation
            overwrite: If True (default), overwrite existing files. If False, raise error if file exists.
        """
        if self.data_noisy is None and N_data is None:
            raise ValueError("No simulated data available. Run simulate_1D, simulate_2D, or simulate_N first.")
        
        # Create simulated_data directory if it doesn't exist
        import os
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
    def _save_hdf5(self, filepath, N_data=None):
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
        import h5py
        
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
                meta.attrs['counts_per_cycle'] = self.counts_per_cycle
                if self.count_rate is not None:
                    meta.attrs['count_rate'] = self.count_rate
                if self.cycle_time is not None:
                    meta.attrs['cycle_time'] = self.cycle_time
            
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
            import json
            meta.attrs['model_parameters'] = json.dumps(params_dict, indent=2)
            meta.attrs['model_name'] = self.model.name