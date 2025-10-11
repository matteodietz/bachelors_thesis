import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path

# Load picmus dataset from afe_interface
# load_picmus_data(iq_path, scan_path): ...
from afe_interface import load_picmus_data

class VirtualAFE:
    """
    A class that simulates the entire AFE pipeline.
    ...
    """
    # __init__ method takes the paths as required arguments
    def __init__(self, iq_path, scan_path, adc_sample_rate=125e6):
        """
        Initializes the Virtual AFE by loading and preparing a specific dataset.
        """
        print(f"--- Initializing Virtual AFE for dataset: {Path(iq_path).name} ---")
        self.fs_adc = adc_sample_rate
        self._high_rate_data = None
        self._fs_picmus = None
        
        # Pass the received arguments down to the loading method
        self._load_and_prepare_data(iq_path, scan_path)

    def _load_and_prepare_data(self, iq_path, scan_path):
        """
        Internal method to load PICMUS data and upsample it.
        """
        print("Loading and upsampling PICMUS data...")
        try:
            # The paths are now correctly passed to the loader
            picmus_data, angles, _, _, self._fs_picmus, _, _, _, _ = load_picmus_data(iq_path, scan_path)
            
            # Select data for one angle and transpose
            center_angle_index = np.argmin(np.abs(angles))
            data_for_one_angle = picmus_data[center_angle_index, :, :].T
            
            # Robust fractional upsampling calculation
            upsample_factor_num = int(self.fs_adc)
            upsample_factor_den = int(self._fs_picmus)
            
            # Upsample to create the internal high-rate signal (125MHz)
            self._high_rate_data = signal.resample_poly(data_for_one_angle, up=upsample_factor_num, down=upsample_factor_den, axis=0)
            
            print(f"Virtual AFE Initialized. Internal data shape: {self._high_rate_data.shape}, fs: {self.fs_adc/1e6:.2f} MHz")
        except Exception as e:
            print(f"FATAL ERROR: Could not initialize Virtual AFE. Error: {e}")
            raise

    def get_decimated_data(self, decimation_factor):
        """
        The main public method. Returns data as if it were decimated by the AFE.
        """
        if decimation_factor <= 1:
            return self._high_rate_data.copy()
        
        # Use scipy decimate function
        decimated_iq = signal.decimate(self._high_rate_data, q=decimation_factor, axis=0)
        return decimated_iq

# --- UNIT TEST ---
if __name__ == '__main__':
    print("--- Running unit test for virtual_afe.py ---")

    # Use pathlib to make paths robust
    try:
        SIMULATOR_ROOT = Path(__file__).parent.parent
    except NameError:
        SIMULATOR_ROOT = Path.cwd().parent
    
    iq_path = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_dataset_iq.hdf5"
    scan_path = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_scan.hdf5"
    print("Paths loaded successfully")

    # Create instance of the VirtualAFE, passing the paths
    try:
        # The adc_sample_rate is a keyword argument, no need to pass it
        afe = VirtualAFE(iq_path=iq_path, scan_path=scan_path)
    except Exception as e:
        # Print error for debugging purpose
        print(f"Unit test failed during initialization. The error was: {e}")
        exit()

    # Decimation factor definitions
    baseline_decimation = 4
    test_decimation = 5
    
    # Get baseline (M=4) and decimated data
    baseline_data = afe.get_decimated_data(decimation_factor=baseline_decimation)
    test_data = afe.get_decimated_data(decimation_factor=test_decimation)
    
    fs_baseline = afe.fs_adc / baseline_decimation
    fs_test = afe.fs_adc / test_decimation
    
    print(f"\nSUCCESS: Got baseline data (M={baseline_decimation}) with shape: {baseline_data.shape}")
    print(f"SUCCESS: Got test data (M={test_decimation}) with shape: {test_data.shape}")
    print(f"SUCCESS: Achieved a compression ratio of {(1 - test_data.shape[0] / baseline_data.shape[0]) * 100}%")

    # Visual Verification
    channel_to_plot = 64
    
    freqs_baseline, psd_baseline = signal.welch(baseline_data[:, channel_to_plot], fs=fs_baseline, nperseg=1024)
    freqs_test, psd_test = signal.welch(test_data[:, channel_to_plot], fs=fs_test, nperseg=1024)
    
    plt.figure(figsize=(12, 6))
    plt.semilogy(freqs_baseline / 1e6, psd_baseline, label=f'Baseline Spectrum (M={baseline_decimation}, fs={fs_baseline/1e6:.2f} MHz)')
    plt.semilogy(freqs_test / 1e6, psd_test, label=f'Test Spectrum (M={test_decimation}, fs={fs_test/1e6:.2f} MHz)')
    plt.title(f'Power Spectral Density Comparison (Channel {channel_to_plot})')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plt.show()