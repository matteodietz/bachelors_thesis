import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from afe_interface_rf import load_picmus_rf_data
from virtual_afe import run_virtual_afe_processing

# --- Function 1: The Core Streaming Processor (Unchanged) ---
def streaming_dft_processor(b, fs, freq_bins_to_calc, window='hann'):
    """
    Simulates a one-pass, streaming, sparse DFT (Goertzel-like).
    """
    N = len(b)
    win = signal.windows.get_window(window, N)
    
    K = len(freq_bins_to_calc)
    A = np.zeros(K, dtype=np.complex128)
    W = np.ones(K, dtype=np.complex128)
    E = np.exp(-1j * 2 * np.pi * freq_bins_to_calc / fs)

    for n in range(N):
        x_n = b[n]
        h_n = win[n]
        A += x_n * h_n * W
        W *= E
        
    final_dft_bins = {freq: accumulator for freq, accumulator in zip(freq_bins_to_calc, A)}
    return final_dft_bins

# --- Function 2: The Corrected Analysis and Edge Finder ---
def find_bandwidth_edges(dft_bins, threshold_db=-20):
    """
    Finds bandwidth edges from a sparse dictionary of DFT bin results,
    correctly handling complex I/Q spectra.
    """
    freqs = np.array(list(dft_bins.keys()))
    accumulators = np.array(list(dft_bins.values()))
    
    powers = np.abs(accumulators)**2
    if np.max(powers) == 0: return float('nan'), float('nan')
    
    power_db = 10 * np.log10(powers + 1e-20)
    power_db_norm = power_db - np.max(power_db)
    
    sort_indices = np.argsort(freqs)
    freqs_sorted = freqs[sort_indices]
    power_db_norm_sorted = power_db_norm[sort_indices]
    
    peak_idx = np.argmax(power_db_norm_sorted)
    
    f_left, f_right = float('nan'), float('nan')
    
    # --- CORRECTED SEARCH LOGIC ---
    # Search for left edge (from peak downwards into NEGATIVE frequencies)
    for i in range(peak_idx, 0, -1):
        if power_db_norm_sorted[i-1] < threshold_db <= power_db_norm_sorted[i]:
            L1, L2 = power_db_norm_sorted[i-1], power_db_norm_sorted[i]
            f1, f2 = freqs_sorted[i-1], freqs_sorted[i]
            f_left = f1 + (f2 - f1) * (threshold_db - L1) / (L2 - L1)
            break
            
    # Search for right edge (from peak upwards into POSITIVE frequencies)
    for i in range(peak_idx, len(freqs_sorted) - 1):
        if power_db_norm_sorted[i+1] < threshold_db <= power_db_norm_sorted[i]:
            L1, L2 = power_db_norm_sorted[i], power_db_norm_sorted[i+1]
            f1, f2 = freqs_sorted[i], freqs_sorted[i+1]
            f_right = f1 + (f2 - f1) * (threshold_db - L1) / (L2 - L1)
            break
    # --- END OF CORRECTION ---
            
    return f_left, f_right

# --- Function 3: The "Master" Orchestrator ---
if __name__ == '__main__':
    print("--- Running Streaming DFT Test on REAL PICMUS Data ---")
    
    # --- 1. Setup and Data Loading ---
    from pathlib import Path
    # Import your data loaders and virtual AFE
    from afe_interface_rf import load_picmus_rf_data
    from virtual_afe_rf import run_virtual_afe_processing_rf

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
    window_num_to_test = 30 # A window from a medium depth
    
    start_sample = window_num_to_test * (nperseg // 2) # Assuming 50% overlap (hop = nperseg/2)
    end_sample = start_sample + nperseg
    
    time_window_data = baseline_iq_data[start_sample:end_sample, channel_to_test]
    print(f"\n--- Analyzing STFT window #{window_num_to_test} from real data ---")

    # --- 4. Run the Streaming DFT Processor ---
    # Define the fixed bin set for the streaming processor
    s_coarse = np.linspace(-mod_freq, mod_freq, 8)
    # Focus the fine bins on the expected signal region for I/Q data
    print(f"modulation frequency = {mod_freq}")
    # region of interest around +/- fc/2
    # to tune
    delta_f = 0.25e6 
    half_bw_est = mod_freq / 2

    s_fine_left = np.linspace(-half_bw_est - delta_f, -half_bw_est + delta_f, 16) 
    s_fine_right = np.linspace(half_bw_est -delta_f, half_bw_est + delta_f, 16) 
    S_bins = np.unique(np.concatenate([s_coarse, s_fine_left, s_fine_right]))
    
    dft_bins = streaming_dft_processor(time_window_data, fs_baseline, S_bins, window='hann')
    
    # --- 5. Find the Bandwidth Edges ---
    threshold_db = -40
    f_left, f_right = find_bandwidth_edges(dft_bins, threshold_db=threshold_db)
    print(f"Streaming DFT Estimated Edges: [{f_left/1e6:.3f}, {f_right/1e6:.3f}] MHz")
    
    # --- 6. Ground Truth and Visual Confirmation ---
    
   # --- Calculate the GROUND TRUTH Power Spectral Density using Welch's method ---
    # `welch` automatically handles windowing, averaging, and PSD scaling.
    freqs_welch, psd_welch = signal.welch(
        time_window_data,
        fs=fs_baseline,
        window='hann',
        nperseg=nperseg,          # Use the full window as a single segment
        return_onesided=False,    # Get the two-sided spectrum for I/Q
        scaling='density'         # This specifies the output is PSD
    )
    
    # Shift for plotting
    freqs_welch_shifted = np.fft.fftshift(freqs_welch)
    psd_welch_shifted = np.fft.fftshift(psd_welch)

    # Normalize to dB relative to the peak
    psd_db_welch_norm = 10 * np.log10(psd_welch_shifted + 1e-20)
    psd_db_welch_norm -= np.max(psd_db_welch_norm)

    plt.figure(figsize=(14, 7))
    
    # Plot the ground truth PSD
    plt.plot(freqs_welch_shifted / 1e6, psd_db_welch_norm, 'k-', label=f'Ground Truth PSD ({nperseg}-pt Welch)', alpha=0.6)

    # --- Convert the sparse DFT bins to PSD and Plot ---
    
    # Get the raw values from your streaming processor
    freqs1 = np.array(list(dft_bins.keys()))
    powers1 = np.abs(np.array(list(dft_bins.values())))**2 # This is Power Spectrum

    # --- THIS IS THE KEY CORRECTION ---
    # To convert Power Spectrum to PSD, we normalize by the Equivalent Noise Bandwidth (ENBW)
    # of the window function and the sample rate.
    # ENBW = fs * sum(win**2) / sum(win)**2
    # For `signal.welch`, the normalization is `fs * sum(win**2)`.
    win = signal.windows.get_window('hann', nperseg)
    enbw_scaling = fs_baseline * np.sum(win**2)
    
    psd1 = powers1 / enbw_scaling
    # --- END OF CORRECTION ---

    db1 = 10 * np.log10(psd1 + 1e-20)
    
    # Normalize by the peak of the GROUND TRUTH PSD for a fair comparison
    plt.plot(freqs1 / 1e6, db1 - np.max(10 * np.log10(psd_welch_shifted + 1e-20)), 'o', markersize=4, label=f'Streaming DFT Bins (PSD, |S|={len(S_bins)})')
    
    # Plot the estimated edges and threshold
    plt.axvline(x=f_left/1e6, color='r', linestyle='--', label=f'Est. Lower Edge ({f_left/1e6:.3f} MHz)')
    plt.axvline(x=f_right/1e6, color='g', linestyle='--', label=f'Est. Upper Edge ({f_right/1e6:.3f} MHz)')
    plt.axhline(y=threshold_db, color='grey', linestyle=':', label=f'{threshold_db} dB Threshold')
    
    plt.title(f'Streaming DFT Bandwidth Estimation on Real Data (Window #{window_num_to_test})')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power (dB relative to peak)')
    plt.legend()
    plt.grid(True)
    # plt.ylim(-60, 5)
    plt.show()

    
    # (The Frame 2 refinement code would follow here if needed)