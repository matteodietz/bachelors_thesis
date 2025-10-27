import numpy as np
from scipy import signal
from pathlib import Path
import matplotlib.pyplot as plt

# Import all the functional blocks
from afe_interface_rf import load_picmus_rf_data
from virtual_afe import run_virtual_afe_processing
from streaming_bw_estimation import streaming_dft_processor, find_bandwidth_edges

# --- MAIN SCRIPT ---
if __name__ == '__main__':
    print("--- Running Full STFT Analysis Across All Windows ---")

    # --- 1. Setup and Data Loading ---
    try:
        SIMULATOR_ROOT = Path(__file__).resolve().parent.parent
    except NameError:
        SIMULATOR_ROOT = Path.cwd().parent
    
    rf_path = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_dataset_rf.hdf5"
    iq_path = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_dataset_iq.hdf5"
    scan_path = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_scan.hdf5"
    
    adc_rate = 125e6
    baseline_decimation = 4

    try:
        rf_data, angles, _, _, fs_picmus, mod_freq, _, _, _ = load_picmus_rf_data(rf_path, iq_path, scan_path)
    except Exception as e:
        print(f"Test failed: Could not load data. Error: {e}")
        exit()

    # --- 2. Get High-Fidelity Baseline I/Q Data ---
    center_angle_index = np.argmin(np.abs(angles))
    baseline_iq_data, _, fs_baseline = run_virtual_afe_processing(
        rf_data=rf_data,
        angle_index=center_angle_index,
        fs_picmus=fs_picmus,
        modulation_frequency=mod_freq,
        decimation_factor=baseline_decimation,
        adc_sample_rate=adc_rate
    )
    
    # --- 3. Setup STFT Parameters ---
    nperseg = 256
    channel_to_test = 64
    hop = nperseg // 2
    threshold_db = -50

    total_samples = baseline_iq_data.shape[0]
    num_windows_total = int(np.floor((total_samples - nperseg) / hop)) + 1
    
    print(f"\n--- STFT Analysis Setup ---")
    print(f"Total number of STFT windows to process: {num_windows_total}")

    # --- 4. Create Output Directory for Plots ---
    # Define the new subdirectory path
    plots_dir = Path(__file__).resolve().parent / "plots" / "bw_est_stress_test"
    # Create the directory and any parent directories if they don't exist
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"Plots will be saved to: {plots_dir}")

    # --- 5. Main Loop: Iterate Through Every Window ---
    for window_num in range(num_windows_total):
        
        start_sample = window_num * hop
        end_sample = start_sample + nperseg
        
        time_window_data = baseline_iq_data[start_sample:end_sample, channel_to_test]
        print(f"\rProcessing window {window_num+1}/{num_windows_total}...", end="")

        # --- 4. Run the Streaming DFT Processor ---
        # Define the fixed bin set for the streaming processor
        s_coarse = np.linspace(-mod_freq, mod_freq, 8)
        # Focus the fine bins on the expected signal region for I/Q data
        print(f"modulation frequency = {mod_freq}")
        # region of interest around +/- fc/2
        # to tune
        delta_f = 0.25e6 
        half_bw_est = mod_freq / 2

        s_fine_left = np.linspace(-half_bw_est - delta_f, -half_bw_est + delta_f, 8) 
        s_fine_right = np.linspace(half_bw_est -delta_f, half_bw_est + delta_f, 8) 
        S_bins = np.unique(np.concatenate([s_coarse, s_fine_left, s_fine_right]))
    
        dft_bins = streaming_dft_processor(time_window_data, fs_baseline, S_bins, window='hann')
        
        # --- Find the Bandwidth Edges ---
        # NOTE: Assumes your find_bandwidth_edges function is imported and has this signature
        f_left, f_right, search_start_freq_left, search_start_freq_right = find_bandwidth_edges(dft_bins, threshold_db=threshold_db)
        
        # --- Ground Truth and Visual Confirmation ---
        freqs_welch, psd_welch = signal.welch(
            time_window_data, fs=fs_baseline, window='hann',
            nperseg=nperseg, return_onesided=False, scaling='density'
        )
        freqs_welch_shifted = np.fft.fftshift(freqs_welch)
        psd_welch_shifted = np.fft.fftshift(psd_welch)
        psd_db_welch_norm = 10 * np.log10(psd_welch_shifted + 1e-20)
        if np.max(psd_db_welch_norm) > -np.inf:
            psd_db_welch_norm -= np.max(psd_db_welch_norm)

        # --- Plotting Section ---
        plt.figure(figsize=(14, 7))
        
        plt.plot(freqs_welch_shifted / 1e6, psd_db_welch_norm, 'k-', label=f'Ground Truth PSD ({nperseg}-pt Welch)', alpha=0.6)

        freqs1 = np.array(list(dft_bins.keys()))
        powers1 = np.abs(np.array(list(dft_bins.values())))**2
        win = signal.windows.get_window('hann', nperseg)
        enbw_scaling = fs_baseline * np.sum(win**2)
        psd1 = powers1 / enbw_scaling
        db1 = 10 * np.log10(psd1 + 1e-20)
        
        if np.max(10 * np.log10(psd_welch_shifted + 1e-20)) > -np.inf:
             plt.plot(freqs1 / 1e6, db1 - np.max(10 * np.log10(psd_welch_shifted + 1e-20)), 'o', markersize=4, label=f'Streaming DFT Bins (PSD, |S|={len(S_bins)})')

        plt.axvline(x=f_left/1e6, color='r', linestyle='--', label=f'Est. Lower Edge ({f_left/1e6:.3f} MHz)')
        plt.axvline(x=f_right/1e6, color='g', linestyle='--', label=f'Est. Upper Edge ({f_right/1e6:.3f} MHz)')
        plt.axhline(y=threshold_db, color='grey', linestyle=':', label=f'{threshold_db} dB Threshold')
        plt.plot(search_start_freq_left / 1e6, 0, 'rX', markersize=7, label=f'Left Search Start')
        plt.plot(search_start_freq_right / 1e6, 0, 'gX', markersize=7, label=f'Right Search Start')
        
        plt.title(f'Streaming DFT Analysis on Real Data (Window #{window_num})')
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Power (dB relative to peak)')
        plt.legend()
        plt.grid(True)
        # plt.ylim(-80, 5)
        
        # Save the figure to the new subdirectory
        output_path = plots_dir / f"window_{window_num:03d}_analysis.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close() # Close the figure to free memory and prevent display

    print(f"\n--- Analysis complete. {num_windows_total} plots saved. ---")