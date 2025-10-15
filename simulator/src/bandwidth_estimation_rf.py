import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path

# import the RF-specific data loader, virtual afe, and stft_processor
from afe_interface_rf import load_picmus_rf_data
from virtual_afe_rf import run_virtual_afe_processing_rf
from stft_processor_rf import run_stft_analysis_rf

# --- MAIN ---
if __name__ == '__main__':
    print("--- Running 'Grand Analysis' on RF DATA to observe spectral changes with depth ---")

    # setup and data loading
    try:
        SIMULATOR_ROOT = Path(__file__).resolve().parent.parent
    except NameError:
        SIMULATOR_ROOT = Path.cwd().parent
    
    rf_path = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_dataset_rf.hdf5"
    iq_path = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_dataset_iq.hdf5"
    scan_path = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_scan.hdf5"
    
    adc_rate = 80e6 # use the ADC rate for RF processing

    try:
        rf_data, angles, _, sound_speed, fs_picmus, mod_freq, initial_time, _, _ = load_picmus_rf_data(rf_path, iq_path, scan_path)
    except Exception as e:
        print(f"Test failed: Could not load data. Error: {e}")
        exit()

    # get baseline RF data for analysis
    # use a low decimation factor (e.g., M=2) as the baseline for RF
    baseline_decimation_rf = 2
    center_angle_index = np.argmin(np.abs(angles))

    baseline_rf_data, _, fs_baseline_rf = run_virtual_afe_processing_rf(
        rf_data=rf_data,
        angle_index=center_angle_index,
        fs_picmus=fs_picmus,
        decimation_factor=baseline_decimation_rf,
        adc_sample_rate=adc_rate
    )
    
    # run the RF STFT Processor
    freqs, time_bins, spectrogram = run_stft_analysis_rf(
        rf_data=baseline_rf_data,
        fs=fs_baseline_rf,
        nperseg=256,
        hop=128
    )
    
    avg_spectrogram = np.mean(spectrogram, axis=0)

    # quantitative analysis and plotting
    num_depths = 5
    print(f"\n--- Quantitative Analysis of RF Spectra at {num_depths} different depths ---")
    
    num_time_bins = avg_spectrogram.shape[1]
    
    start_index = int(num_time_bins * 0.1)
    end_index = int(num_time_bins * 0.9)
    indices_to_plot = np.linspace(start_index, end_index, num_depths, dtype=int)
    
    plt.figure(figsize=(14, 7))

    for i, time_idx in enumerate(indices_to_plot):
        spectrum_slice = avg_spectrogram[:, time_idx]

        # find the peak frequency
        peak_index = np.argmax(spectrum_slice)
        peak_frequency_mhz = freqs[peak_index] / 1e6
        
        # convert to dB and normalize
        spectrum_db = 10 * np.log10(spectrum_slice + 1e-20)
        spectrum_db_normalized = spectrum_db - np.max(spectrum_db)

        # find the -20dB bandwidth edges
        threshold = -20 # dB
        above_threshold_indices = np.where(spectrum_db_normalized > threshold)[0]
        
        if len(above_threshold_indices) > 0:
            lower_edge_index = above_threshold_indices[0]
            upper_edge_index = above_threshold_indices[-1]
            
            lower_edge_freq_mhz = freqs[lower_edge_index] / 1e6
            upper_edge_freq_mhz = freqs[upper_edge_index] / 1e6
            bandwidth_mhz = upper_edge_freq_mhz - lower_edge_freq_mhz

            # calculate max safe decimation factor for RF data
            f_max_hz = freqs[upper_edge_index]
            practical_margin = 2.5 # oversampling
            fs_min = practical_margin * f_max_hz
            M_max = adc_rate / fs_min

        else:
            lower_edge_freq_mhz, upper_edge_freq_mhz, bandwidth_mhz, M_max = (float('nan'),)*4

        time_s = initial_time + time_bins[time_idx]
        depth_mm = time_s * sound_speed / 2 * 1000

        print(f"Depth: {depth_mm:5.1f} mm -> (Peak @ {peak_frequency_mhz:5.2f} MHz, "
              f"BW = {bandwidth_mhz:4.2f} MHz, "
              f"Edges: [{lower_edge_freq_mhz:5.2f}, {upper_edge_freq_mhz:5.2f}] MHz), "
              f"M_max = {M_max}")

        plt.plot(freqs / 1e6, spectrum_db_normalized, label=f'Depth = {depth_mm:.1f} mm')

    # configure and save the plot
    plt.axvline(x=mod_freq / 1e6, color='r', linestyle=':', label=f'Modulation Freq ({mod_freq/1e6:.2f} MHz)')
    plt.title(f'Normalized RF Power Spectrum at Different Depths (fs = {fs_baseline_rf/1e6:.2f} MHz)')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power (dB relative to peak)')
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plt.ylim(-60, 5) 
    plt.xlim(0, fs_baseline_rf/2 / 1e6) # RF spectrum is one-sided

    plots_dir = Path(__file__).resolve().parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    file_name = "stft_bandwidth_estimation_rf.png"
    output_path = plots_dir / file_name

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"\nSUCCESS: RF analysis plot saved to {output_path}")
    plt.close()