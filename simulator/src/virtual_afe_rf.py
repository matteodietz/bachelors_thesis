import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path

# data loader for RF data
from afe_interface_rf import load_picmus_rf_data

def run_virtual_afe_processing_rf(rf_data, angle_index, fs_picmus, modulation_frequency, decimation_factor, adc_sample_rate=125e6):
    """
    Performs the virtual AFE simulation on pre-loaded PICMUS RF data.

    This function simulates the full pipeline:
    1. Upsamples the RF data for a specific angle to the target ADC rate.
    2. Performs I/Q demodulation on the high-rate RF data.
    3. Decimates the resulting I/Q data by the specified factor.
    
    Args:
        rf_data (np.ndarray): The full, low-rate PICMUS RF data array (angles, channels, samples).
        angle_index (int): The index of the angle from the dataset to process.
        fs_picmus (float): The original sample rate of the PICMUS RF data.
        modulation_frequency (float): The center frequency of the transducer for demodulation.
        decimation_factor (int): The integer decimation factor to apply.
        adc_sample_rate (float): The target sample rate of the virtual ADC in Hz.

    Returns:
        tuple: A tuple containing:
            - decimated_iq (np.ndarray): The final decimated I/Q data.
            - high_rate_iq (np.ndarray): The intermediate high-rate I/Q data (before decimation).
            - fs_new (float): The sample rate of the decimated_iq data.
    """
    print(f"--- Running Virtual AFE Processing for M={decimation_factor} ---")
    
    # Upsample RF Data for the chosen angle
    data_for_one_angle_rf = rf_data[angle_index, :, :].T
    
    upsample_factor_num = int(adc_sample_rate)
    upsample_factor_den = int(fs_picmus)
    
    high_rate_rf = signal.resample_poly(data_for_one_angle_rf, up=upsample_factor_num, down=upsample_factor_den, axis=0)
    print(f"Upsampled RF data to shape: {high_rate_rf.shape}")
    
    # I/Q Demodulation 
    # Time vector for the high-rate RF signal
    num_samples_high_rate = high_rate_rf.shape[0]
    t = np.arange(num_samples_high_rate) / adc_sample_rate
    
    # Complex local oscillator signal
    # Multiply by 2 to get the analytic signal (I + jQ) after filtering
    local_oscillator = 2 * np.exp(-1j * 2 * np.pi * modulation_frequency * t)
    
    # Demodulate by multiplying the RF signal by the complex oscillator
    # Reshape the local_oscillator to multiply it with each channel
    analytic_signal_passband = high_rate_rf * local_oscillator[:, np.newaxis]
    
    # Low-pass filter the result to remove the 2*f_c component and keep the baseband signal
    # Simple low-pass filter for this purpose
    # Cutoff should be less than the modulation frequency
    num_taps = 99
    lpf_cutoff = modulation_frequency / (adc_sample_rate / 2)
    print(f"lpf_cutoff frequency is: {lpf_cutoff}")
    lpf_coeffs = signal.firwin(num_taps, lpf_cutoff)
    
    high_rate_iq = signal.lfilter(lpf_coeffs, 1.0, analytic_signal_passband, axis=0)
    print(f"I/Q Demodulation complete. High-rate IQ shape: {high_rate_iq.shape}")

    # Decimate the High-Rate I/Q Data
    if decimation_factor < 1:
        raise ValueError("Decimation factor must be >= 1.")
    if decimation_factor == 1:
        decimated_iq = high_rate_iq.copy()
    else:
        # The decimate function includes its own anti-aliasing filter
        decimated_iq = signal.decimate(high_rate_iq, q=decimation_factor, axis=0)
    
    fs_new = adc_sample_rate / decimation_factor
    
    return decimated_iq, high_rate_iq, fs_new


# --- UNIT TEST ---
if __name__ == '__main__':
    print("--- Running unit test for virtual_afe.py (RF input) ---")

    # Define paths and parameters
    try:
        SIMULATOR_ROOT = Path(__file__).parent.parent
    except NameError:
        SIMULATOR_ROOT = Path.cwd().parent
    
    # !!! IMPORTANT: Need all three paths since modulation_frequency in RF dataset is 0 !!!
    rf_path = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_dataset_rf.hdf5"
    iq_path = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_dataset_iq.hdf5"
    scan_path = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_scan.hdf5"
    
    baseline_decimation = 4
    test_decimation = 5
    adc_rate = 125e6

    # Load the RF data
    try:
        print("\nLoading PICMUS RF dataset from disk...")
        rf_data, angles, _, _, fs_picmus, mod_freq, _, _, _ = load_picmus_rf_data(rf_path, iq_path, scan_path)
        print("RF Data loaded successfully.")
    except Exception as e:
        print(f"Test failed: Could not load data. Check your afe_interface.py. Error: {e}")
        exit()

    center_angle_index = np.argmin(np.abs(angles))

    # Call the processing function for the baseline case
    baseline_data_iq, _, fs_baseline = run_virtual_afe_processing_rf(
        rf_data=rf_data,
        angle_index=center_angle_index,
        fs_picmus=fs_picmus,
        modulation_frequency=mod_freq,
        decimation_factor=baseline_decimation,
        adc_sample_rate=adc_rate
    )
    print(f"SUCCESS: Got baseline I/Q data (M={baseline_decimation}) with shape: {baseline_data_iq.shape}")
        
    # Call the processing function for the test case
    test_data_iq, _, fs_test = run_virtual_afe_processing_rf(
        rf_data=rf_data,
        angle_index=center_angle_index,
        fs_picmus=fs_picmus,
        modulation_frequency=mod_freq,
        decimation_factor=test_decimation,
        adc_sample_rate=adc_rate
    )
    print(f"SUCCESS: Got test I/Q data (M={test_decimation}) with shape: {test_data_iq.shape}")
    
    # Visual Verification
    channel_to_plot = 64
    
    # Plot the spectra of the generated I/Q data
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