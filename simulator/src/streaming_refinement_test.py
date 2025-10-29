import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path

# Import all functional blocks
from afe_interface_rf import load_picmus_rf_data
from virtual_afe import run_virtual_afe_processing
from streaming_bw_estimation import streaming_dft_processor, find_bandwidth_edges

# --- MAIN SCRIPT ---
if __name__ == '__main__':
    print("--- Running Two-Stage Streaming DFT Refinement Test on REAL PICMUS Data ---")

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
    
    # --- STAGE 1: BOOTSTRAP ANALYSIS (N=256) ---
    print("\n" + "="*30 + " STAGE 1: BOOTSTRAP (N=256) " + "="*30)
    nperseg1 = 256
    channel_to_test = 64
    window_num_to_test = 28 
    hop1 = nperseg1 // 2
    
    start_sample1 = window_num_to_test * hop1
    end_sample1 = start_sample1 + nperseg1
    
    time_window_data1 = baseline_iq_data[start_sample1:end_sample1, channel_to_test]
    print(f"--- Analyzing STFT window #{window_num_to_test} (samples {start_sample1}-{end_sample1}) ---")


    s_coarse = np.linspace(-mod_freq, mod_freq, 8)
    # Focus the fine bins on the expected signal region for I/Q data
    print(f"modulation frequency = {mod_freq}")
    # region of interest around +/- fc/2
    # to tune
    delta_f = 0.5e6 
    half_bw_est = mod_freq / 2 + 0.5e6 # to see what happens when my heuristics is worse

    s_fine_left1 = np.linspace(-half_bw_est - delta_f, -half_bw_est + delta_f, 6) 
    s_fine_right1 = np.linspace(half_bw_est -delta_f, half_bw_est + delta_f, 6) 
    S_bins1 = np.unique(np.concatenate([s_coarse, s_fine_left1, s_fine_right1]))
    
    dft_bins1 = streaming_dft_processor(time_window_data1, fs_baseline, S_bins1, window='hann')
    
    # --- Find Bandwidth Edges (Stage 1) ---
    threshold_db = -30
    f_left1, f_right1, _, _ = find_bandwidth_edges(dft_bins1, threshold_db=threshold_db)
    print(f"Stage 1 Estimated Edges: [{f_left1/1e6:.3f}, {f_right1/1e6:.3f}] MHz")
    
    # --- PLOTTING FOR STAGE 1 ---
    freqs_welch1, psd_welch1 = signal.welch(
        time_window_data1, fs=fs_baseline, window='hann',
        nperseg=nperseg1, return_onesided=False, scaling='density'
    )
    freqs_welch_shifted1 = np.fft.fftshift(freqs_welch1)
    psd_welch_shifted1 = np.fft.fftshift(psd_welch1)
    psd_db_welch_norm1 = 10 * np.log10(psd_welch_shifted1 + 1e-20)
    psd_db_welch_norm1 -= np.max(psd_db_welch_norm1)

    plt.figure(figsize=(14, 7))
    plt.plot(freqs_welch_shifted1 / 1e6, psd_db_welch_norm1, 'k-', label=f'Ground Truth PSD ({nperseg1}-pt Welch)', alpha=0.6)

    freqs1_plot = np.array(list(dft_bins1.keys()))
    powers1_plot = np.abs(np.array(list(dft_bins1.values())))**2
    win1 = signal.windows.get_window('hann', nperseg1)
    enbw_scaling1 = fs_baseline * np.sum(win1**2)
    psd1_plot = powers1_plot / enbw_scaling1
    db1_plot = 10 * np.log10(psd1_plot + 1e-20)
    
    plt.plot(freqs1_plot / 1e6, db1_plot - np.max(10 * np.log10(psd_welch_shifted1 + 1e-20)), 'o', markersize=4, label=f'Streaming DFT Bins (|S|={len(S_bins1)})')
    
    plt.axvline(x=f_left1/1e6, color='r', linestyle='--', label=f'Est. Lower Edge ({f_left1/1e6:.3f} MHz)')
    plt.axvline(x=f_right1/1e6, color='g', linestyle='--', label=f'Est. Upper Edge ({f_right1/1e6:.3f} MHz)')
    plt.axhline(y=threshold_db, color='grey', linestyle=':', label=f'{threshold_db} dB Threshold')

    plt.title(f'Bootstrap Streaming DFT Analysis (N={nperseg1})')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power (dB relative to peak)')
    plt.legend()
    plt.grid(True)
    # plt.ylim(-60, 5)
    plt.show()
    # --- END OF PLOTTING FOR STAGE 1 ---

    # --- STAGE 2: REFINEMENT ANALYSIS (N=512) ---
    print("\n" + "="*30 + " STAGE 2: REFINEMENT (N=512) " + "="*30)
    nperseg2 = 512
    window_num_to_test2 = 15 
    hop2 = nperseg2 // 2

    start_sample2 = window_num_to_test2 * hop2
    end_sample2 = start_sample2 + nperseg2
    
    time_window_data2 = baseline_iq_data[start_sample2:end_sample2, channel_to_test]
    print(f"--- Analyzing longer window (samples {start_sample2}-{end_sample2}) ---")

    # --- Run Streaming DFT Processor (Stage 2) ---
    s_coarse2 = np.linspace(-mod_freq, mod_freq, 6)
    
    # Re-center the fine clusters on the results from Stage 1
    delta_f2 = 0.25e6 # Use a narrower zoom window for refinement
    s_fine_left2 = np.linspace(f_left1 - delta_f2, f_left1 + delta_f2, 5)
    s_fine_right2 = np.linspace(f_right1 - delta_f2, f_right1 + delta_f2, 5)
    S_bins2 = np.unique(np.concatenate([s_coarse2, s_fine_left2, s_fine_right2]))

    dft_bins2 = streaming_dft_processor(time_window_data2, fs_baseline, S_bins2, window='hann')

    # --- Find Bandwidth Edges (Stage 2) ---
    f_left2, f_right2, _, _ = find_bandwidth_edges(dft_bins2, threshold_db=threshold_db)
    print(f"Stage 2 Refined Edges: [{f_left2/1e6:.3f}, {f_right2/1e6:.3f}] MHz")

    # --- Ground Truth and Final Visual Confirmation ---
    freqs_welch, psd_welch = signal.welch(
        time_window_data2, fs=fs_baseline, window='hann',
        nperseg=nperseg2, return_onesided=False, scaling='density'
    )
    freqs_welch_shifted = np.fft.fftshift(freqs_welch)
    psd_welch_shifted = np.fft.fftshift(psd_welch)
    psd_db_welch_norm = 10 * np.log10(psd_welch_shifted + 1e-20)
    psd_db_welch_norm -= np.max(psd_db_welch_norm)

    plt.figure(figsize=(14, 7))
    
    plt.plot(freqs_welch_shifted / 1e6, psd_db_welch_norm, 'k-', label=f'Ground Truth PSD ({nperseg2}-pt Welch)', alpha=0.6)

    freqs2_plot = np.array(list(dft_bins2.keys()))
    powers2_plot = np.abs(np.array(list(dft_bins2.values())))**2
    win2 = signal.windows.get_window('hann', nperseg2)
    enbw_scaling2 = fs_baseline * np.sum(win2**2)
    psd2_plot = powers2_plot / enbw_scaling2
    db2_plot = 10 * np.log10(psd2_plot + 1e-20)
    
    plt.plot(freqs2_plot / 1e6, db2_plot - np.max(10 * np.log10(psd_welch_shifted + 1e-20)), 'x', markersize=5, label=f'Refined Streaming DFT Bins (|S|={len(S_bins2)})')
    
    plt.axvline(x=f_left2/1e6, color='r', linestyle='--', label=f'Refined Lower Edge ({f_left2/1e6:.3f} MHz)')
    plt.axvline(x=f_right2/1e6, color='g', linestyle='--', label=f'Refined Upper Edge ({f_right2/1e6:.3f} MHz)')
    plt.axhline(y=threshold_db, color='grey', linestyle=':', label=f'{threshold_db} dB Threshold')

    plt.title(f'Refined Streaming DFT Analysis (N={nperseg2})')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power (dB relative to peak)')
    plt.legend()
    plt.grid(True)
    # plt.ylim(-60, 5)
    plt.show()