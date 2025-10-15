import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path

# data loader for RF data
from afe_interface_rf import load_picmus_rf_data

def run_virtual_afe_processing_rf(rf_data, angle_index, fs_picmus, decimation_factor, adc_sample_rate=80e6, snr_db=None):
    """
    Performs a virtual AFE simulation on pre-loaded PICMUS RF data,
    outputting decimated RF data (NO I/Q demodulation).

    This function simulates the following pipeline:
    1. Upsamples the RF data for a specific angle to the target ADC rate.
    2. (Optional) Adds Additive White Gaussian Noise (AWGN) to the RF signal.
    3. Decimates the resulting high-rate RF data by the specified factor.
    
    Args:
        rf_data (np.ndarray): The full, low-rate PICMUS RF data array (angles, channels, samples).
        angle_index (int): The index of the angle from the dataset to process.
        fs_picmus (float): The original sample rate of the PICMUS RF data.
        decimation_factor (int): The integer decimation factor to apply.
        adc_sample_rate (float): The target sample rate of the virtual ADC in Hz (e.g., 80e6 for RF).
        snr_db (float, optional): The desired Signal-to-Noise Ratio in dB. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - decimated_rf (np.ndarray): The final decimated RF data (real-valued).
            - high_rate_rf (np.ndarray): The intermediate upsampled RF data (before decimation).
            - fs_new (float): The sample rate of the decimated_rf data.
    """
    print(f"--- Running Virtual AFE RF Processing for M={decimation_factor} ---")
    
    # upsample RF data for the chosen angle
    data_for_one_angle_rf = rf_data[angle_index, :, :].T
    
    upsample_factor_num = int(adc_sample_rate)
    upsample_factor_den = int(fs_picmus)
    
    high_rate_rf = signal.resample_poly(data_for_one_angle_rf, up=upsample_factor_num, down=upsample_factor_den, axis=0)
    print(f"Upsampled RF data to shape: {high_rate_rf.shape}")

    # add AWGN to the high-rate RF signal
    if snr_db is not None:
        print(f"Adding AWGN to achieve an SNR of {snr_db} dB...")
        signal_power = np.var(high_rate_rf)
        snr_linear = 10**(snr_db / 10)
        noise_power = signal_power / snr_linear
        noise_std = np.sqrt(noise_power)
        noise = np.random.normal(loc=0.0, scale=noise_std, size=high_rate_rf.shape)
        high_rate_rf = high_rate_rf + noise
        print("Noise addition complete.")
    
    # decimate the high-rate RF data
    if decimation_factor < 1:
        raise ValueError("Decimation factor must be >= 1.")
    if decimation_factor == 1:
        decimated_rf = high_rate_rf.copy()
    else:
        # The decimate function works on real signals and includes an anti-aliasing filter
        decimated_rf = signal.decimate(high_rate_rf, q=decimation_factor, axis=0)
    
    fs_new = adc_sample_rate / decimation_factor
    
    return decimated_rf, high_rate_rf, fs_new


# --- UNIT TEST ---
if __name__ == '__main__':
    print("--- Running unit test for virtual_afe.py (RF-only output) ---")

    # define paths and parameters
    try:
        SIMULATOR_ROOT = Path(__file__).parent.parent
    except NameError:
        SIMULATOR_ROOT = Path.cwd().parent
    
    rf_path = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_dataset_rf.hdf5"
    iq_path = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_dataset_iq.hdf5" # Still needed for mod_freq
    scan_path = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_scan.hdf5"
    
    baseline_decimation = 2 # RF data has higher bandwidth, so baseline M is lower
    test_decimation = 4
    adc_rate = 80e6 # New ADC rate for RF processing
    snr = 40

    # load the RF data
    try:
        print("\nLoading PICMUS RF dataset from disk...")
        rf_data, angles, _, _, fs_picmus, mod_freq, _, _, _ = load_picmus_rf_data(rf_path, iq_path, scan_path)
        print("RF Data loaded successfully.")
    except Exception as e:
        print(f"Test failed: Could not load data. Check your afe_interface.py. Error: {e}")
        exit()

    center_angle_index = np.argmin(np.abs(angles))

    # call the processing function for the baseline case
    baseline_data_rf, _, fs_baseline = run_virtual_afe_processing_rf(
        rf_data=rf_data,
        angle_index=center_angle_index,
        fs_picmus=fs_picmus,
        decimation_factor=baseline_decimation,
        adc_sample_rate=adc_rate,
        snr_db=snr
    )
    print(f"SUCCESS: Got baseline RF data (M={baseline_decimation}) with shape: {baseline_data_rf.shape}")
        
    # call the processing function for the test case
    test_data_rf, _, fs_test = run_virtual_afe_processing_rf(
        rf_data=rf_data,
        angle_index=center_angle_index,
        fs_picmus=fs_picmus,
        decimation_factor=test_decimation,
        adc_sample_rate=adc_rate,
        snr_db=snr
    )
    print(f"SUCCESS: Got test RF data (M={test_decimation}) with shape: {test_data_rf.shape}")
    
    # visual verification
    channel_to_plot = 64
    
    # plot the spectra of the generated RF data
    # For a real signal, welch returns a one-sided spectrum (positive frequencies only)
    freqs_baseline, psd_baseline = signal.welch(baseline_data_rf[:, channel_to_plot], fs=fs_baseline, nperseg=1024)
    freqs_test, psd_test = signal.welch(test_data_rf[:, channel_to_plot], fs=fs_test, nperseg=1024)

    # convert to a relative dB scale
    peak_power = np.max(psd_baseline)
    psd_baseline_db = 10 * np.log10(psd_baseline / peak_power + 1e-20)
    psd_test_db = 10 * np.log10(psd_test / peak_power + 1e-20)
    
    # create the plot using a linear y-axis
    plt.figure(figsize=(12, 6))
    
    plt.plot(freqs_baseline / 1e6, psd_baseline_db, label=f'Baseline RF Spectrum (M={baseline_decimation}, fs={fs_baseline/1e6:.2f} MHz)')
    plt.plot(freqs_test / 1e6, psd_test_db, label=f'Test RF Spectrum (M={test_decimation}, fs={fs_test/1e6:.2f} MHz)')
    
    plt.title(f'Normalized Power Spectral Density of Processed RF Data (Channel {channel_to_plot})')
    plt.xlabel('Frequency (MHz)') # Note: Not "Frequency Offset" anymore
    plt.ylabel('Power (dB relative to peak)')
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    # plt.ylim(-80, 5)
    
    plt.show()