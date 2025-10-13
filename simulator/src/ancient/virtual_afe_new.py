import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path

# Load picmus dataset from afe_interface
# load_picmus_data(iq_path, scan_path): ...
from afe_interface import load_picmus_data

def run_virtual_afe_processing(picmus_data, angle_index, fs_picmus, decimation_factor, adc_sample_rate=125e6):
    """
    Performs the virtual AFE simulation on pre-loaded PICMUS data for a single angle.

    This function is a pure processing block. It takes the raw data, upsamples a 
    specific angle, and then decimates it.
    
    Args:
        picmus_data (np.ndarray): The full, low-rate PICMUS data array (angles, channels, samples).
        angles (np.ndarray): The array of transmission angles from the dataset.
        fs_picmus (float): The original sample rate of the PICMUS data.
        decimation_factor (int): The decimation factor to apply.
        adc_sample_rate (float): The target sample rate of the virtual ADC in Hz.

    Returns:
        tuple: A tuple containing:
            - decimated_data (np.ndarray): The final decimated I/Q data.
            - high_rate_data (np.ndarray): The intermediate upsampled data (for comparison).
            - fs_new (float): The sample rate of the decimated data.
    """
    print(f"--- Running Virtual AFE Processing for M={decimation_factor} ---")
    
    # Upsample Data for the chosen angle
    # Select data for the center angle and transpose
    data_for_one_angle = picmus_data[angle_index, :, :].T
    
    # Robust fractional upsampling calculation
    upsample_factor_num = int(adc_sample_rate)
    upsample_factor_den = int(fs_picmus)
    
    # Upsample to create the internal high-rate signal (125MHz)
    high_rate_data = signal.resample_poly(data_for_one_angle, up=upsample_factor_num, down=upsample_factor_den, axis=0)
    
    # Decimate the High-Rate Data using scipy decimate function
    if decimation_factor < 1:
        raise ValueError("Decimation factor must be >= 1.")
    if decimation_factor == 1:
        decimated_iq = high_rate_data.copy()
    else:
        decimated_iq = signal.decimate(high_rate_data, q=decimation_factor, axis=0)
    
    fs_new = adc_sample_rate / decimation_factor
    
    return decimated_iq, high_rate_data, fs_new


# --- UNIT TEST ---
if __name__ == '__main__':
    print("--- Running unit test for virtual_afe.py (modular functional version) ---")

    # Define paths and parameters
    # Use pathlib to make paths robust
    try:
        SIMULATOR_ROOT = Path(__file__).parent.parent.parent
    except NameError:
        SIMULATOR_ROOT = Path.cwd().parent.parent
    
    iq_path = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_dataset_iq.hdf5"
    scan_path = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_scan.hdf5"
    
    baseline_decimation = 4
    test_decimation = 5
    adc_rate = 125e6

    # Load the data
    try:
        print("\nLoading PICMUS dataset from disk...")
        picmus_data, angles, _, _, fs_picmus, _, _, _, _ = load_picmus_data(iq_path, scan_path)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Test failed: Could not load data. Error: {e}")
        exit()

    # Call the processing function for the baseline case (M=4)
    center_angle_index = np.argmin(np.abs(angles))

    baseline_data, _, fs_baseline = run_virtual_afe_processing(
        picmus_data=picmus_data,
        fs_picmus=fs_picmus,
        angle_index=center_angle_index,
        decimation_factor=baseline_decimation,
        adc_sample_rate=adc_rate
    )
    print(f"SUCCESS: Got baseline data (M={baseline_decimation}) with shape: {baseline_data.shape}")
        
    # Call the processing function for the test case
    test_data, high_rate_ref, fs_test = run_virtual_afe_processing(
        picmus_data=picmus_data,
        fs_picmus=fs_picmus,
        angle_index=center_angle_index,
        decimation_factor=test_decimation,
        adc_sample_rate=adc_rate
    )
    print(f"SUCCESS: Got test data (M={test_decimation}) with shape: {test_data.shape}")
    print(f"SUCCESS: Achieved a ratio old_data_size / new_data_size = {baseline_data.shape[0]/test_data.shape[0]}")
    print(f"SUCCESS: Achieved a compression of {(1 - test_data.shape[0] / baseline_data.shape[0]) * 100}%")
    
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