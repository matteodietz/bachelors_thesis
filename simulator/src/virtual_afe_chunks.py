import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path

# Import the data loader from afe_interface
from afe_interface import load_picmus_data

def simulate_afe_processing(iq_path, scan_path, angle_index, decimation_map, adc_sample_rate=125e6):
    """
    Simulates the full AFE processing chain for a SINGLE angle.

    This function performs the entire virtual workflow:
    1. Loads the specified low-rate PICMUS dataset.
    2. Upsamples the data for the specified angle to the target ADC rate.
    3. Applies a depth-dependent decimation based on the provided map.
    
    Args:
        iq_path (pathlib.Path or str): Path to the IQ data HDF5 file.
        scan_path (pathlib.Path or str): Path to the scan grid HDF5 file.
        angle_index (int): The index of the angle to process from the dataset.
        decimation_map (dict): A map of {start_sample: decimation_factor}, 
                               e.g., {0: 4, 10000: 8}.
        adc_sample_rate (float): The target sample rate of the virtual ADC in Hz.

    Returns:
        tuple: A tuple containing:
            - decimated_data (np.ndarray): The final variable-rate data stream.
            - baseline_data (np.ndarray): The upsampled, high-fidelity data (for comparison).
            - fs_adc (float): The ADC sample rate.
    """
    # --- 1. Load the Data ---
    try:
        picmus_data, angles, _, _, fs_picmus, _, _, _, _ = load_picmus_data(iq_path, scan_path)
        if not (0 <= angle_index < len(angles)):
            raise IndexError(f"Angle index {angle_index} is out of bounds for {len(angles)} angles.")
    except Exception as e:
        print(f"ERROR: Could not load data. Error: {e}")
        raise

    # --- 2. Upsample to "Virtual ADC" Rate ---
    # Transpose data for the specified angle to (samples, channels)
    data_for_one_angle = picmus_data[angle_index, :, :].T
    
    # Robust fractional upsampling calculation
    upsample_factor_num = int(adc_sample_rate)
    upsample_factor_den = int(fs_picmus)
    
    # Upsample to create the internal high-rate signal. This is our "baseline".
    baseline_data = signal.resample_poly(data_for_one_angle, up=upsample_factor_num, down=upsample_factor_den, axis=0)
    
    # --- 3. Apply Adaptive Decimation ---
    num_samples = baseline_data.shape[0]
    output_chunks = []
    
    # Sort the map by depth (the keys)
    sorted_depths = sorted(decimation_map.keys())
    
    start_idx = 0
    for i, depth_start in enumerate(sorted_depths):
        end_idx = sorted_depths[i+1] if i + 1 < len(sorted_depths) else num_samples
        m_factor = decimation_map[depth_start]
        
        chunk = baseline_data[start_idx:end_idx, :]
        
        # The decimation is simple downsampling, as the signal is already band-limited
        # by the original PICMUS data and our interpolation.
        decimated_chunk = chunk[::m_factor, :]
        output_chunks.append(decimated_chunk)
        
        start_idx = end_idx
        
    decimated_data = np.concatenate(output_chunks, axis=0)
    
    return decimated_data, baseline_data, adc_sample_rate


# --- This block now serves as a simple and clean UNIT TEST ---
if __name__ == '__main__':
    print("--- Running unit test for virtual_afe.py (functional approach) ---")
    
    # Define the paths and parameters for the test
    try:
        SIMULATOR_ROOT = Path(__file__).parent.parent
    except NameError:
        SIMULATOR_ROOT = Path.cwd().parent
    
    iq_path = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_dataset_iq.hdf5"
    scan_path = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_scan.hdf5"
    
    angle_to_test = 37  # Center angle
    # Define a decimation map: M=4 for first 10k samples, M=8 for the rest
    test_map = {0: 4, 10000: 8} 

    # --- Call the main simulation function ---
    try:
        adaptive_data, baseline_data, fs_adc = simulate_afe_processing(
            iq_path=iq_path,
            scan_path=scan_path,
            angle_index=angle_to_test,
            decimation_map=test_map
        )
        
        # ... (Verification prints are the same)
        total_samples_baseline = baseline_data.shape[0]
        # ...

        print(f"\nSUCCESS: Got adaptively decimated data with shape: {adaptive_data.shape}")
        
        # --- NEW: Reconstruct the signal to a constant rate for comparison ---
        # This simulates the job of your FPGA's "Harmonizer" / Resampler block.
        print("\nReconstructing adaptively sampled data back to constant rate for verification...")
        
        # To do this, we first need to "re-insert" the missing samples as zeros.
        # This is the upsampling step.
        reconstructed_signal = np.zeros_like(baseline_data)
        
        current_read_idx = 0
        sorted_depths = sorted(test_map.keys())
        start_write_idx = 0
        
        for i, depth_start in enumerate(sorted_depths):
            end_write_idx = sorted_depths[i+1] if i + 1 < len(sorted_depths) else total_samples_baseline
            m_factor = test_map[depth_start]
            
            # Figure out how many samples we took for this segment
            num_samples_in_segment = len(np.arange(start_write_idx, end_write_idx, m_factor))
            
            # Place the decimated samples back into the high-rate grid
            reconstructed_signal[start_write_idx:end_write_idx:m_factor, :] = adaptive_data[current_read_idx : current_read_idx + num_samples_in_segment, :]
            
            current_read_idx += num_samples_in_segment
            start_write_idx = end_write_idx

        # Now, apply an interpolation filter to fill in the zeros.
        # A simple low-pass FIR filter is a good choice. 'lfilter' is causal like in an FPGA.
        # We need to design a good interpolation filter.
        num_taps = 96
        cutoff_freq = 0.5 / max(test_map.values()) # Cutoff based on the most aggressive decimation
        interp_filter = signal.firwin(num_taps, cutoff_freq)

        # Apply the filter to reconstruct the signal
        reconstructed_signal_filtered = signal.lfilter(interp_filter, 1.0, reconstructed_signal, axis=0)
        
        print("Reconstruction complete.")

        # --- Visual Verification ---
        channel_to_plot = 64
        fs_const = fs_adc # Our constant rate is the full ADC rate
        
        # Calculate spectrum of the original high-rate signal
        freqs_baseline, psd_baseline = signal.welch(baseline_data[:, channel_to_plot], fs=fs_const, nperseg=1024)
        
        # Calculate spectrum of the RECONSTRUCTED signal
        freqs_reconstructed, psd_reconstructed = signal.welch(reconstructed_signal_filtered[:, channel_to_plot], fs=fs_const, nperseg=1024)
        
        # --- Plot both spectra on top of each other ---
        plt.figure(figsize=(12, 6))
        plt.semilogy(freqs_baseline / 1e6, psd_baseline, label=f'Original High-Rate Spectrum (fs={fs_const/1e6:.2f} MHz)')
        plt.semilogy(freqs_reconstructed / 1e6, psd_reconstructed, linestyle='--', label=f'Reconstructed Spectrum (from adaptive rate)')
        
        plt.title(f'Power Spectral Density Comparison (Channel {channel_to_plot})')
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Power/Frequency (dB/Hz)')
        plt.grid(True, which='both', linestyle='--')
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"ERROR: The simulation failed. Error: {e}")