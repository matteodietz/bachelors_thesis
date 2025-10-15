import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path

# import data loader, virtual afe and stft_processor
from afe_interface_rf import load_picmus_rf_data
from virtual_afe_rf import run_virtual_afe_processing_rf
from stft_processor import run_stft_analysis

# --- MAIN ---
if __name__ == '__main__':
    print("--- Running 'Grand Analysis' to observe spectral changes with depth ---")

    # setup and data loading from hdf5 dataset
    try:
        SIMULATOR_ROOT = Path(__file__).parent.parent
    except NameError:
        SIMULATOR_ROOT = Path.cwd().parent
    
    rf_path = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_dataset_rf.hdf5"
    iq_path = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_dataset_iq.hdf5"
    scan_path = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_scan.hdf5"
    adc_rate = 125e6

    try:
        rf_data, angles, _, sound_speed, fs_picmus, mod_freq, initial_time, _, _ = load_picmus_rf_data(rf_path, iq_path, scan_path)
    except Exception as e:
        print(f"Test failed: Could not load data. Error: {e}")
        exit()

    # get baseline I/Q data (M = 4) as a reference
    baseline_decimation = 4
    center_angle_index = np.argmin(np.abs(angles))

    baseline_iq_data, _, fs_baseline = run_virtual_afe_processing_rf(
        rf_data=rf_data,
        angle_index=center_angle_index,
        fs_picmus=fs_picmus,
        modulation_frequency=mod_freq,
        decimation_factor=baseline_decimation,
        adc_sample_rate=adc_rate
    )
    
    # run the STFT processor to get the full spectrogram
    freqs, time_bins, spectrogram = run_stft_analysis(
        iq_data=baseline_iq_data,
        fs=fs_baseline
    )
    
    # average across all channels for a robust, clean spectrogram
    avg_spectrogram = np.mean(spectrogram, axis=0) # shape is (freqs, time_bins)

    # select and plot spectra at different depths
    num_depths = 5
    print(f"Selecting and plotting spectra at {num_depths} different depths...")
    
    num_time_bins = avg_spectrogram.shape[1]
    print(f"Number of time bins = {num_time_bins}")

    # create equally spaced indices across the time bins
    indices_to_plot = np.linspace(2, num_time_bins - 3, num_depths, dtype=int)
    
    plt.figure(figsize=(14, 7))

    for i, time_idx in enumerate(indices_to_plot):
        # get the spectrum slice for this specific time/depth
        spectrum_slice = avg_spectrogram[:, time_idx]

        # find the peak frequency
        peak_index = np.argmax(spectrum_slice)
        peak_frequency_mhz = freqs[peak_index] / 1e6
        
        # convert to dB and normalize
        spectrum_db = 10 * np.log10(spectrum_slice + 1e-20)
        spectrum_db_normalized = spectrum_db - np.max(spectrum_db)

        # find the -20dB bandwidth edges
        threshold = -20 # dB
        
        # find all indices where the spectrum is above the threshold
        above_threshold_indices = np.where(spectrum_db_normalized > threshold)[0]
        
        if len(above_threshold_indices) > 0:
            lower_edge_index = above_threshold_indices[0]
            upper_edge_index = above_threshold_indices[-1]
            
            lower_edge_freq_mhz = freqs[lower_edge_index] / 1e6
            upper_edge_freq_mhz = freqs[upper_edge_index] / 1e6
            bandwidth_mhz = upper_edge_freq_mhz - lower_edge_freq_mhz

            # calculate maximum safe decimation factor
            max_abs_freq = np.max([np.abs(freqs[lower_edge_index]), np.abs(freqs[upper_edge_index])])
            practical_margin = 4 # oversampling
            fs_min = practical_margin * max_abs_freq
            M_max = np.floor(adc_rate / fs_min)

        else:
            # if no signal is above the threshold, report as invalid
            lower_edge_freq_mhz = float('nan')
            upper_edge_freq_mhz = float('nan')
            bandwidth_mhz = float('nan')

        # calculate depth for reporting
        time_s = initial_time + time_bins[time_idx]
        depth_mm = time_s * sound_speed / 2 * 1000

        # print the results to the terminal
        print(f"Depth: {depth_mm:5.1f} mm -> (Peak @ {peak_frequency_mhz:5.2f} MHz, "
              f"BW = {bandwidth_mhz:4.2f} MHz, "
              f"Edges: [{lower_edge_freq_mhz:5.2f}, {upper_edge_freq_mhz:5.2f}] MHz), "
              f"M_max = {M_max:4.2f}")

        # plot
        plt.plot(freqs / 1e6, spectrum_db_normalized, label=f'Depth = {depth_mm:.1f} mm')

    # configure and show the plot
    plt.title('Normalized Power Spectrum at Different Imaging Depths (Center Angle, Avg. Channels)')
    plt.xlabel('Frequency Offset (MHz)')
    plt.ylabel('Power (dB relative to peak)')
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plt.ylim(-40, 5) # zoom in on the top 40 dB to see the shape clearly
    plt.xlim(-fs_picmus/2 / 1e6, fs_picmus/2 / 1e6) # Set x-axis limits
    plt.show()