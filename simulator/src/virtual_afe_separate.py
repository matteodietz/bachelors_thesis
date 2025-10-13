import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path

from afe_interface_rf import load_picmus_rf_data

def perform_iq_demodulation(rf_data, fs_rf, modulation_frequency):
    """
    Performs I/Q demodulation on a high-rate RF signal.
    """
    print("Performing I/Q Demodulation...")
    num_samples_high_rate = rf_data.shape[0]
    t = np.arange(num_samples_high_rate) / fs_rf
    
    local_oscillator = 2 * np.exp(-1j * 2 * np.pi * modulation_frequency * t)
    analytic_signal_passband = rf_data * local_oscillator[:, np.newaxis]
    
    num_taps = 99
    nyquist_freq = fs_rf / 2
    lpf_cutoff = modulation_frequency / nyquist_freq
    if not (0 < lpf_cutoff < 1): raise ValueError("LPF cutoff is invalid.")
        
    lpf_coeffs = signal.firwin(num_taps, lpf_cutoff)
    high_rate_iq = signal.lfilter(lpf_coeffs, 1.0, analytic_signal_passband, axis=0)
    print("I/Q Demodulation complete.")
    return high_rate_iq

def decimate_iq_data(high_rate_iq, decimation_factor):
    """
    A simple function that only performs decimation on high-rate I/Q data.
    """
    if decimation_factor < 1: raise ValueError("Decimation factor must be >= 1.")
    if decimation_factor == 1:
        return high_rate_iq.copy()
    else:
        return signal.decimate(high_rate_iq, q=decimation_factor, axis=0)

# --- UNIT TEST ---
if __name__ == '__main__':
    print("--- Running unit test for virtual_afe.py (RF input, corrected test) ---")

    # Define paths and parameters
    try:
        SIMULATOR_ROOT = Path(__file__).parent.parent
    except NameError:
        SIMULATOR_ROOT = Path.cwd().parent
    
    rf_path = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_dataset_rf.hdf5"
    iq_path = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_dataset_iq.hdf5"
    scan_path = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_scan.hdf5"
    adc_rate = 125e6

    # --- Step 1: Load the RF data ---
    try:
        rf_data, angles, _, _, fs_picmus, mod_freq, _, _, _ = load_picmus_rf_data(rf_path, iq_path, scan_path)
    except Exception as e:
        print(f"Test failed: Could not load data. Error: {e}")
        exit()

    # --- Step 2: Upsample and I/Q Demodulate ONCE to create the "Ground Truth" I/Q ---
    center_angle_index = np.argmin(np.abs(angles))
    data_for_one_angle_rf = rf_data[center_angle_index, :, :].T
    
    upsample_factor_num = int(adc_rate)
    upsample_factor_den = int(fs_picmus)
    high_rate_rf = signal.resample_poly(data_for_one_angle_rf, up=upsample_factor_num, down=upsample_factor_den, axis=0)
    
    # This is now our master high-rate I/Q signal for the test
    high_rate_iq_master = perform_iq_demodulation(high_rate_rf, adc_rate, mod_freq)

    # --- Step 3: Decimate the master I/Q to create baseline and test cases ---
    baseline_decimation = 4
    test_decimation = 5
    
    baseline_data_iq = decimate_iq_data(high_rate_iq_master, baseline_decimation)
    test_data_iq = decimate_iq_data(high_rate_iq_master, test_decimation)
    
    fs_baseline = adc_rate / baseline_decimation
    fs_test = adc_rate / test_decimation
    
    print(f"\nSUCCESS: Got baseline I/Q (M={baseline_decimation}) with shape: {baseline_data_iq.shape}")
    print(f"SUCCESS: Got test I/Q (M={test_decimation}) with shape: {test_data_iq.shape}")

    # --- Step 4: Visual Verification ---
    channel_to_plot = 64
    
    freqs_baseline, psd_baseline = signal.welch(baseline_data_iq[:, channel_to_plot], fs=fs_baseline, nperseg=1024)
    freqs_test, psd_test = signal.welch(test_data_iq[:, channel_to_plot], fs=fs_test, nperseg=1024)
    
    plt.figure(figsize=(12, 6))
    plt.semilogy(freqs_baseline / 1e6, psd_baseline, label=f'Baseline I/Q Spectrum (M={baseline_decimation}, fs={fs_baseline/1e6:.2f} MHz)')
    plt.semilogy(freqs_test / 1e6, psd_test, label=f'Test I/Q Spectrum (M={test_decimation}, fs={fs_test/1e6:.2f} MHz)')
    plt.title(f'Power Spectral Density of Generated I/Q Data (Channel {channel_to_plot})')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plt.show()