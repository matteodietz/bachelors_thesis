# Save this file as: grand_analysis_test.py

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import all the functional blocks you've built
from afe_interface import load_picmus_data
from virtual_afe_new import run_virtual_afe_processing
from stft_processor import run_stft_analysis

if __name__ == '__main__':
    print("--- Running the 'Grand Analysis' to test the STFT Processor ---")

    # --- 1. Setup and Data Loading ---
    try:
        SIMULATOR_ROOT = Path(__file__).parent.parent
    except NameError:
        SIMULATOR_ROOT = Path.cwd().parent
    
    iq_path = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_dataset_iq.hdf5"
    scan_path = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_scan.hdf5"
    adc_rate = 125e6
    oversampling_factor = 2.5 # Our fixed quality contract
    
    try:
        picmus_data, angles, _, sound_speed, fs_picmus, _, initial_time, _, _ = load_picmus_data(iq_path, scan_path)
    except Exception as e:
        print(f"Test failed: Could not load data. Error: {e}")
        exit()

    # --- 2. Get High-Fidelity Data for Analysis ---
    # For this analysis, we want to analyze the highest quality signal.
    # So we will run the virtual AFE with the baseline decimation M=4.
    baseline_decimation = 4
    center_angle_index = np.argmin(np.abs(angles))

    baseline_data, _, fs_baseline = run_virtual_afe_processing(
        picmus_data=picmus_data,
        angle_index=center_angle_index,
        fs_picmus=fs_picmus,
        decimation_factor=baseline_decimation,
        adc_sample_rate=adc_rate
    )
    
    # --- 3. Run the STFT Processor ---
    # This is the block we are testing.
    freqs, time_bins, spectrogram = run_stft_analysis(
        iq_data=baseline_data,
        fs=fs_baseline
    )

    # --- 4. Analyze the Spectrogram to find the Optimal M vs. Depth ---
    print("\nAnalyzing spectrogram to determine optimal M vs. Depth...")
    
    # First, average the spectrogram across all channels for a robust estimate
    avg_spectrogram = np.mean(spectrogram, axis=0) # Shape is now (freqs, time_bins)
    
    optimal_m_vs_time = []
    bandwidth_vs_time = []
    
    # Iterate through each time window (depth slice) of the spectrogram
    for i in range(avg_spectrogram.shape[1]):
        spectrum_slice = avg_spectrogram[:, i]
        
        # Find the peak power and its index
        peak_power = np.max(spectrum_slice)
        
        # Define the -6dB threshold
        threshold = peak_power * 0.25 # 10*log10(0.25) = -6dB
        
        # Find indices where the spectrum is above the threshold
        above_threshold_indices = np.where(spectrum_slice > threshold)[0]
        
        if len(above_threshold_indices) > 0:
            # Calculate bandwidth in number of bins
            bw_in_bins = above_threshold_indices[-1] - above_threshold_indices[0]
            # Convert bandwidth to Hz
            measured_bw = bw_in_bins * (fs_baseline / (len(freqs) * 2)) # df = fs / nperseg
        else:
            measured_bw = 0
            
        bandwidth_vs_time.append(measured_bw)
        
        # Calculate the maximum safe decimation factor
        if measured_bw > 0:
            max_safe_m = (adc_rate / (measured_bw * oversampling_factor))
            # In a real system, we'd clamp this to a max value and choose an integer/fraction
            optimal_m_vs_time.append(max_safe_m)
        else:
            optimal_m_vs_time.append(baseline_decimation) # Default to baseline if no signal
    
    # --- 5. Generate the "Money Plot" ---
    print("Generating final analysis plot...")

    # Convert time bins to depth in mm
    depth_axis = (initial_time + time_bins) * sound_speed / 2 * 1000

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plot 1: The Spectrogram (Power vs. Depth and Frequency)
    # Use pcolormesh for a 2D color plot. Convert power to dB for better visualization.
    db_spectrogram = 10 * np.log10(avg_spectrogram + 1e-20)
    img = ax1.pcolormesh(depth_axis, freqs / 1e6, db_spectrogram, shading='gouraud', cmap='viridis')
    ax1.set_xlabel('Depth (mm)')
    ax1.set_ylabel('Frequency (MHz)', color='k')
    ax1.tick_params(axis='y', labelcolor='k')
    fig.colorbar(img, ax=ax1, label='Power (dB)')

    # Plot 2: The Optimal Decimation Factor M on a second y-axis
    ax2 = ax1.twinx() # Create a second y-axis that shares the same x-axis
    ax2.plot(depth_axis, optimal_m_vs_time, color='r', linestyle='-', marker='.', label='Optimal M')
    ax2.set_ylabel('Optimal Decimation Factor (M)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylim(bottom=0) # M cannot be negative

    fig.tight_layout()
    plt.title('Spectrogram and Optimal Decimation Factor vs. Depth')
    plt.show()