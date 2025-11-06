import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path

from afe_interface_rf import load_picmus_rf_data
from virtual_afe import run_virtual_afe_processing
from golden_model_floating_point import (
    streaming_dft_processor, 
    convert_to_sorted_db_power, 
    find_left_edge_points, 
    find_right_edge_points, 
    linear_interpolate_crossing
)

# --- Main Test Script ---
if __name__ == '__main__':
    print("--- Running Modular Streaming DFT Test on REAL PICMUS Data ---")

    # --- 1. Load PICMUS Data ---
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
    
    # --- 3. Select ONE STFT Window to Analyze ---
    nperseg = 256
    channel_to_test = 64
    window_num_to_test = 29 
    hop = nperseg // 2

    total_samples = baseline_iq_data.shape[0]
    num_windows_total = int(np.floor((total_samples - nperseg) / hop)) + 1
    
    print(f"\n--- STFT Analysis Setup ---")
    print(f"Total samples in A-line: {total_samples}")
    print(f"Window size (nperseg):   {nperseg}")
    print(f"Hop size:                {hop}")
    print(f"Total number of STFT windows available: {num_windows_total}")
    
    start_sample = window_num_to_test * hop
    end_sample = start_sample + nperseg
    
    time_window_data = baseline_iq_data[start_sample:end_sample, channel_to_test]
    print(f"\n--- Analyzing STFT window #{window_num_to_test} from real data ---")

    # --- 4. Define Analysis Parameters ---
    delta_f = 0.25e6 
    half_bw_est = mod_freq / 2

    s_coarse = np.linspace(-mod_freq, mod_freq, 8)
    s_fine_left = np.linspace(-half_bw_est - delta_f, -half_bw_est + delta_f, 8) 
    s_fine_right = np.linspace(half_bw_est -delta_f, half_bw_est + delta_f, 8) 
    S_bins = np.unique(np.concatenate([s_coarse, s_fine_left, s_fine_right]))

    print(f"Bins to calculate: {S_bins}")

    threshold_db = -30

    # --- 5. Run the Modular Processing Pipeline ---
    # Step 1: Core Streaming Processor
    dft_bins = streaming_dft_processor(time_window_data, fs_baseline, S_bins, window='hann')

    # Step 2: Power Conversion and Sorting
    freqs_sorted, power_db_norm_sorted = convert_to_sorted_db_power(dft_bins)
    
    # Step 3: Find Left Edge Points
    f1_left, f2_left, L1_left, L2_left = find_left_edge_points(freqs_sorted, power_db_norm_sorted,  threshold_db=threshold_db)
    
    # Step 4: Find Right Edge Points
    f1_right, f2_right, L1_right, L2_right = find_right_edge_points(freqs_sorted, power_db_norm_sorted,  threshold_db=threshold_db)
    
    # # Step 2: Power Conversion and Sorting
    # freqs_sorted, power_db_norm_sorted, bin_indices_sorted = convert_to_sorted_db_power(dft_bins)
    
    # # Step 3: Find Left Edge Points
    # f1_left, f2_left, L1_left, L2_left, k1_left, k2_left = find_left_edge_points(freqs_sorted, power_db_norm_sorted, bin_indices_sorted, threshold_db=threshold_db)

    # print(f" LEFT EDGE: f1 = {f1_left}, f2 = {f2_left}, k1 = {k1_left}, k2 = {k2_left}")
    
    # # Step 4: Find Right Edge Points
    # f1_right, f2_right, L1_right, L2_right, k1_right, k2_right = find_right_edge_points(freqs_sorted, power_db_norm_sorted, bin_indices_sorted, threshold_db=threshold_db)

    # print(f" RIGHT EDGE: f1 = {f1_right}, f2 = {f2_right}, k1 = {k1_right}, k2 = {k2_right}")

    # Step 5: Interpolate to find final edges
    f_left_final = linear_interpolate_crossing(f1_left, f2_left, L1_left, L2_left, threshold_db=threshold_db)
    f_right_final = linear_interpolate_crossing(f1_right, f2_right, L1_right, L2_right, threshold_db=threshold_db)
    
    print(f"Streaming DFT Estimated Edges: [{f_left_final/1e6:.3f}, {f_right_final/1e6:.3f}] MHz")
    
    # --- 6. Ground Truth and Visual Confirmation ---
    freqs_welch, psd_welch = signal.welch(
        time_window_data, 
        fs=fs_baseline, 
        window='hann', 
        nperseg=nperseg,
        return_onesided=False, 
        scaling='density'
    )
    freqs_welch_shifted = np.fft.fftshift(freqs_welch)
    psd_welch_shifted = np.fft.fftshift(psd_welch)
    psd_db_welch_norm = 10 * np.log10(psd_welch_shifted + 1e-20)
    psd_db_welch_norm -= np.max(psd_db_welch_norm)

    plt.figure(figsize=(14, 7))
    plt.plot(freqs_welch_shifted / 1e6, psd_db_welch_norm, 'k-', label=f'Ground Truth PSD ({nperseg}-pt Welch)', alpha=0.6)

    win = signal.windows.get_window('hann', nperseg)
    enbw_scaling = fs_baseline * np.sum(win**2)
    
    freqs1 = np.array(list(dft_bins.keys()))
    # freqs1 = np.array([res['freq_hz'] for res in dft_bins])
    powers1 = np.abs(np.array(list(dft_bins.values())))**2
    # powers1 = np.array([res['accumulator'] for res in dft_bins])**2
    psd1 = powers1 / enbw_scaling
    db1 = 10 * np.log10(psd1 + 1e-20)
    
    plt.plot(freqs1 / 1e6, db1 - np.max(10 * np.log10(psd_welch_shifted + 1e-20)), 'o', markersize=4, label=f'Streaming DFT Bins (PSD, |S|={len(S_bins)})')
    
    plt.axvline(x=f_left_final/1e6, color='r', linestyle='--', label=f'Est. Lower Edge ({f_left_final/1e6:.3f} MHz)')
    plt.axvline(x=f_right_final/1e6, color='g', linestyle='--', label=f'Est. Upper Edge ({f_right_final/1e6:.3f} MHz)')
    plt.axhline(y=threshold_db, color='grey', linestyle=':', label=f'{threshold_db} dB Threshold')
    
    plt.title(f'Modular Streaming DFT Bandwidth Estimation (Window #{window_num_to_test})')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power (dB relative to peak)')
    plt.legend()
    plt.grid(True)
    plt.show()