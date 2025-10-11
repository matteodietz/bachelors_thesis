import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path

# We only need the loader for the unit test block at the end
from afe_interface import load_picmus_data

def run_virtual_afe_processing(picmus_data, angle_index, fs_picmus, decimation_map, adc_sample_rate=125e6):
    """
    Performs the virtual AFE simulation for a single angle, using a 
    depth-dependent decimation map.
    """
    print(f"--- Running Virtual AFE Processing with map: {decimation_map} ---")
    
    # --- Part 1: Upsample Data for the chosen angle ---
    data_for_one_angle = picmus_data[angle_index, :, :].T
    
    upsample_factor_num = int(adc_sample_rate)
    upsample_factor_den = int(fs_picmus)
    
    high_rate_data = signal.resample_poly(data_for_one_angle, up=upsample_factor_num, down=upsample_factor_den, axis=0)
    
    # --- Part 2: Decimate the High-Rate Data using the map ---
    if not isinstance(decimation_map, dict):
        raise TypeError("decimation_map must be a dictionary.")
        
    num_samples = high_rate_data.shape[0]
    output_chunks = []
    
    sorted_depths = sorted(decimation_map.keys())
    
    start_idx = 0
    for i, depth_start in enumerate(sorted_depths):
        end_idx = sorted_depths[i+1] if i + 1 < len(sorted_depths) else num_samples
        m_factor = decimation_map[depth_start]
        if m_factor < 1: raise ValueError("Decimation factor must be >= 1.")
        
        chunk = high_rate_data[start_idx:end_idx, :]
        decimated_chunk = chunk[::m_factor, :]
        output_chunks.append(decimated_chunk)
        
        start_idx = end_idx
        
    decimated_iq = np.concatenate(output_chunks, axis=0)
    
    return decimated_iq, high_rate_data


# --- UNIT TEST ---
if __name__ == '__main__':
    print("--- Running unit test for virtual_afe.py (map-based functional version) ---")

    # Define paths and parameters
    try:
        SIMULATOR_ROOT = Path(__file__).parent.parent
    except NameError:
        SIMULATOR_ROOT = Path.cwd().parent
    
    iq_path = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_dataset_iq.hdf5"
    scan_path = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_scan.hdf5"
    adc_rate = 125e6

    # --- Step 1: Load the data ONCE ---
    try:
        print("\nLoading PICMUS dataset from disk...")
        picmus_data, angles, _, _, fs_picmus, _, _, _, _ = load_picmus_data(iq_path, scan_path)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Test failed: Could not load data. Error: {e}")
        exit()

    center_angle_index = np.argmin(np.abs(angles))
    
    # --- Step 2: Define and run the adaptive decimation test ---
    adaptive_map_test = {0: 4, 10000: 8, 25000: 12}
    
    adaptive_data, high_rate_ref = run_virtual_afe_processing(
        picmus_data=picmus_data,
        angle_index=center_angle_index,
        fs_picmus=fs_picmus,
        decimation_map=adaptive_map_test,
        adc_sample_rate=adc_rate
    )
    print(f"\nSUCCESS: Got adaptively decimated data with shape: {adaptive_data.shape}")

    # --- Step 3: Reconstruct the adaptive signal for visualization (Simulate the "Harmonizer") ---
    print("\nReconstructing adaptively sampled data back to constant high rate for verification...")
    
    # 3a: Create an empty high-rate array and "scatter" the adaptive samples into it
    reconstructed_signal_with_zeros = np.zeros_like(high_rate_ref)
    current_read_idx = 0
    sorted_depths = sorted(adaptive_map_test.keys())
    
    for i, depth_start in enumerate(sorted_depths):
        end_idx = sorted_depths[i+1] if i + 1 < len(sorted_depths) else high_rate_ref.shape[0]
        m_factor = adaptive_map_test[depth_start]
        
        num_samples_in_segment = len(np.arange(depth_start, end_idx, m_factor))
        
        reconstructed_signal_with_zeros[depth_start:end_idx:m_factor, :] = adaptive_data[current_read_idx : current_read_idx + num_samples_in_segment, :]
        current_read_idx += num_samples_in_segment

    # 3b: Design and apply a low-pass interpolation filter to "fill in the blanks"
    num_taps = 96
    # The cutoff must be based on the highest frequency we expect, which is dictated by the lowest decimation factor (M=4)
    cutoff_freq = 0.5 / min(adaptive_map_test.values()) 
    interp_filter = signal.firwin(num_taps, cutoff_freq)
    
    # Apply the filter. We must also scale the output to compensate for the filter gain.
    reconstructed_signal_filtered = signal.lfilter(interp_filter * min(adaptive_map_test.values()), 1.0, reconstructed_signal_with_zeros, axis=0)
    print("Reconstruction complete.")
    
    # --- Step 4: Visual Verification ---
    channel_to_plot = 64
    
    # compare two signals at the SAME sample rate (adc_rate)
    freqs_ref, psd_ref = signal.welch(high_rate_ref[:, channel_to_plot], fs=adc_rate, nperseg=1024)
    freqs_recon, psd_recon = signal.welch(reconstructed_signal_filtered[:, channel_to_plot], fs=adc_rate, nperseg=1024)
    
    plt.figure(figsize=(12, 6))
    plt.semilogy(freqs_ref / 1e6, psd_ref, label=f'Original High-Rate Spectrum (fs={adc_rate/1e6:.2f} MHz)')
    plt.semilogy(freqs_recon / 1e6, psd_recon, linestyle='--', label=f'Reconstructed Spectrum (from adaptive rate)')
    plt.title(f'Power Spectral Density Comparison (Channel {channel_to_plot})')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plt.show()