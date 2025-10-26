import numpy as np
from scipy import signal
from pathlib import Path
import matplotlib.pyplot as plt

# Import all the functional blocks
from afe_interface_rf import load_picmus_rf_data
# Make sure this is the virtual_afe with your preferred filtering (e.g., ideal FFT-based)
from virtual_afe import run_virtual_afe_processing
# Import your new bandwidth search function
from hcsbs import hcsbs_bandwidth_search

# --- MAIN SCRIPT ---
if __name__ == '__main__':
    print("--- Running HCSBS Bandwidth Search Test on Real PICMUS Data ---")

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
    
    # --- 3. Select ONE STFT Window to Analyze ---
    nperseg = 256 # Window size
    hop = 128
    channel_to_test = 64
    
    window_num_to_test = 30
    start_sample = window_num_to_test * hop
    end_sample = start_sample + nperseg
    
    time_window_data = baseline_iq_data[start_sample:end_sample, channel_to_test]
    
    print(f"\n--- Analyzing STFT window #{window_num_to_test} (samples {start_sample}-{end_sample}) ---")

    # --- 4. Find "Ground Truth" Edges with a High-Resolution FFT ---
    threshold_db = -25 # The target threshold
    
    zero_padding_factor = 1 # 16
    N_fine = nperseg * zero_padding_factor
    
    full_fft = np.fft.fftshift(np.fft.fft(time_window_data * signal.windows.hann(nperseg), n=N_fine))
    freqs_full = np.fft.fftshift(np.fft.fftfreq(N_fine, 1/fs_baseline))
    
    psd_db_full = 10 * np.log10(np.abs(full_fft)**2 + 1e-20)
    psd_db_full_norm = psd_db_full - np.max(psd_db_full)
    
    above_thresh_indices = np.where(psd_db_full_norm > threshold_db)[0]
    if len(above_thresh_indices) > 0:
        true_lower_edge = freqs_full[above_thresh_indices[0]]
        true_upper_edge = freqs_full[above_thresh_indices[-1]]
    else:
        true_lower_edge, true_upper_edge = (float('nan'), float('nan'))
    
    # --- 5. Run your HCSBS Bandwidth Search Algorithm ---
    beta_to_test = 0.5
    
    z_lower_idx, z_upper_idx = hcsbs_bandwidth_search(
        b=time_window_data,
        N=nperseg,
        k=32,
        beta=beta_to_test,
        threshold_db=threshold_db,
        fs=fs_baseline,
        window='hann'
    )
    
    freqs_unshifed = np.fft.fftfreq(nperseg, 1/fs_baseline)
    hcsbs_lower_freq = freqs_unshifed[z_lower_idx]
    hcsbs_upper_freq = freqs_unshifed[z_upper_idx]

    # --- 6. Compare the Results ---
    print("\n--- Bandwidth Edge Estimation Comparison on REAL Data ---")
    print(f"Ground Truth Edges (@ {threshold_db} dB): [{true_lower_edge/1e6:6.3f}, {true_upper_edge/1e6:6.3f}] MHz")
    print(f"HCSBS Estimated Edges (beta={beta_to_test}): [{hcsbs_lower_freq/1e6:6.3f}, {hcsbs_upper_freq/1e6:6.3f}] MHz")

    # --- 7. Visual Verification ---
    standard_fft = np.fft.fftshift(np.fft.fft(time_window_data * signal.windows.hann(nperseg)))
    freqs_standard = np.fft.fftshift(np.fft.fftfreq(nperseg, 1/fs_baseline))
    psd_db_standard_norm = 10 * np.log10(np.abs(standard_fft)**2 + 1e-20)
    psd_db_standard_norm -= np.max(psd_db_standard_norm)
    
    plt.figure(figsize=(12, 6))
    plt.plot(freqs_standard / 1e6, psd_db_standard_norm, 'b-', label=f'Standard {nperseg}-pt Spectrum')
    plt.axhline(y=threshold_db, color='grey', linestyle=':', label=f'{threshold_db} dB Threshold')
    plt.axvline(x=hcsbs_lower_freq / 1e6, color='r', linestyle='--', label=f'HCSBS Lower Edge ({hcsbs_lower_freq/1e6:.3f} MHz)')
    plt.axvline(x=hcsbs_upper_freq / 1e6, color='g', linestyle='--', label=f'HCSBS Upper Edge ({hcsbs_upper_freq/1e6:.3f} MHz)')
    plt.title(f'HCSBS Bandwidth Estimation for Window #{window_num_to_test}')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power (dB relative to peak)')
    plt.legend()
    plt.grid(True)
    # plt.ylim(threshold_db - 20, 5)
    plt.show()