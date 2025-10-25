import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path

# Import all the functional blocks 
from afe_interface_rf import load_picmus_rf_data
from virtual_afe import run_virtual_afe_processing
from hcsps import hcsps_peak_search 

def hybrid_spectral_analyzer(b, N, k, beta, fs, window='hann'):
    """
    Performs a unified, efficient search for both peak and bandwidth using a
    robust hybrid architecture: HCSPS for the peak, and Trimmed Spectral 
    Moments on the coarse FFT for the bandwidth.
    """
    if len(b) != N: raise ValueError("Input vector `b` must have length `N`.")

    win = signal.windows.get_window(window, N)
    b_windowed = b * win

    # --- Stage 1: Coarse FFT (Used by BOTH paths) ---
    b_coarse = b_windowed[:k]
    X_coarse_wrapped = np.fft.fft(b_coarse)
    
    # --- Path A: Find Precise Peak using HCSPS Refinement ---
    k_coarse_peak_idx = np.argmax(np.abs(X_coarse_wrapped)**2)
    
    resolution_ratio = N / k
    zoom_center_bin = int(k_coarse_peak_idx * resolution_ratio)
    zoom_width = int(resolution_ratio)
    half_width = zoom_width // 2
    bins_to_check = (zoom_center_bin + np.arange(-half_width, half_width + 1)) % N
    
    n = int(beta * N)
    if n < 1: n = 1
    random_indices = np.linspace(0, N - 1, n, dtype=int)
    b_n = b_windowed[random_indices]
    
    powers_in_zoom = []
    for j in bins_to_check:
        exponent = -1j * 2 * np.pi * j * random_indices / N
        inner_product = np.dot(b_n, np.exp(exponent))
        powers_in_zoom.append(np.abs(inner_product)**2)
        
    k_refined_local = np.argmax(powers_in_zoom)
    z_final_peak_index = bins_to_check[k_refined_local]
    
    freqs_unwrapped = np.fft.fftfreq(N, 1/fs)
    f_peak_hz = freqs_unwrapped[z_final_peak_index]

    # --- Path B: Find Bandwidth using TRIMMED Spectral Moments ---
    coarse_powers = np.abs(X_coarse_wrapped)**2
    
    # Find the first valleys on either side of the coarse peak
    # Search left from the peak
    lower_trim_idx = k_coarse_peak_idx
    while lower_trim_idx > 0 and coarse_powers[lower_trim_idx - 1] < coarse_powers[lower_trim_idx]:
        lower_trim_idx -= 1
        
    # Search right from the peak
    upper_trim_idx = k_coarse_peak_idx
    while upper_trim_idx < k - 1 and coarse_powers[upper_trim_idx + 1] < coarse_powers[upper_trim_idx]:
        upper_trim_idx += 1
        
    # Create the "trimmed" spectrum slices for the moment calculation
    trim_indices = np.arange(lower_trim_idx, upper_trim_idx + 1)
    coarse_freqs = np.fft.fftfreq(k, 1/fs)
    trimmed_powers = coarse_powers[trim_indices]
    trimmed_freqs = coarse_freqs[trim_indices]
    
    df = coarse_freqs[1] - coarse_freqs[0] if k > 1 else 1.0
    S0 = np.sum(trimmed_powers) * df
    
    if S0 > 0:
        # Calculate spread around the PRECISE HCSPS peak, using only the trimmed spectrum
        variance = np.sum((trimmed_freqs - f_peak_hz)**2 * trimmed_powers) * df / S0
        sigma_f = np.sqrt(max(0, variance))
        estimated_bw = 6.06 * sigma_f
    else:
        sigma_f, estimated_bw = 0, 0
        
    return f_peak_hz, estimated_bw


# --- UNIT TEST ---
if __name__ == '__main__':
    print("--- Running Hybrid Analyzer Test on Real PICMUS Data ---")

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
        rf_data=rf_data, angle_index=center_angle_index, fs_picmus=fs_picmus,
        modulation_frequency=mod_freq, decimation_factor=baseline_decimation,
        adc_sample_rate=adc_rate
    )
    
    # --- 3. Select ONE STFT Window to Analyze ---
    nperseg = 256
    hop = 128
    channel_to_test = 64
    window_num_to_test = 30
    
    start_sample = window_num_to_test * hop
    end_sample = start_sample + nperseg
    
    time_window_data = baseline_iq_data[start_sample:end_sample, channel_to_test]
    print(f"\n--- Analyzing STFT window #{window_num_to_test} ---")

    # --- 4. Find Ground Truth Bandwidth (High-Res FFT Threshold Method) ---
    threshold_db = -20
    
    full_fft = np.fft.fftshift(np.fft.fft(time_window_data * signal.windows.hann(nperseg), n=nperseg*16))
    freqs_full = np.fft.fftshift(np.fft.fftfreq(nperseg*16, 1/fs_baseline))
    
    psd_db_full = 10 * np.log10(np.abs(full_fft)**2 + 1e-20)
    psd_db_full_norm = psd_db_full - np.max(psd_db_full)
    
    above_thresh_indices = np.where(psd_db_full_norm > threshold_db)[0]
    true_bandwidth = (freqs_full[above_thresh_indices[-1]] - freqs_full[above_thresh_indices[0]]) if len(above_thresh_indices) > 0 else float('nan')

    # --- 5. Run Your NEW Hybrid Algorithm ---
    estimated_peak_freq, estimated_bw = hybrid_spectral_analyzer(
        b=time_window_data,
        N=nperseg,
        k=32,
        beta=0.5,
        fs=fs_baseline,
        window='hann'
    )
    
    # --- 6. Compare the Results ---
    print("\n--- Hybrid Analyzer Comparison on REAL Data ---")
    print(f"Ground Truth Bandwidth (@ {threshold_db} dB): {true_bandwidth / 1e6:6.3f} MHz")
    print(f"Hybrid Estimated Peak:       {estimated_peak_freq / 1e6:6.3f} MHz")
    print(f"Hybrid Estimated Bandwidth:  {estimated_bw / 1e6:6.3f} MHz")
    
    # --- 7. Visual Confirmation Plot (CORRECTED SECTION) ---
    plt.figure(figsize=(12, 6))

    # 7a: Generate the standard N-point spectrum FOR PLOTTING
    standard_fft_wrapped = np.fft.fft(time_window_data * signal.windows.hann(nperseg))
    standard_powers = np.abs(standard_fft_wrapped)**2
    freqs_standard_unwrapped = np.fft.fftfreq(nperseg, 1/fs_baseline)

    # 7b: Shift spectra and frequencies for correct plotting
    freqs_standard_shifted = np.fft.fftshift(freqs_standard_unwrapped)
    psd_db_standard_norm = 10 * np.log10(np.fft.fftshift(standard_powers) + 1e-20)
    psd_db_standard_norm -= np.max(psd_db_standard_norm)
    
    # 7c: Plot the data
    plt.plot(freqs_standard_shifted / 1e6, psd_db_standard_norm, label=f'Standard {nperseg}-pt Spectrum')
    plt.axhline(y=threshold_db, color='grey', linestyle=':', label=f'{threshold_db} dB Threshold')
    
    # 7d: Plot the results from your hybrid analyzer
    plt.axvline(x=estimated_peak_freq / 1e6, color='purple', linestyle='-.', label=f'Hybrid Estimated Peak ({estimated_peak_freq/1e6:.3f} MHz)')

    est_lower = estimated_peak_freq - estimated_bw / 2
    est_upper = estimated_peak_freq + estimated_bw / 2
    plt.axvspan(est_lower/1e6, est_upper/1e6, color='red', alpha=0.2, label=f'Hybrid Estimated BW ({estimated_bw/1e6:.3f} MHz)')

    # 7e: Configure and show
    plt.title(f'Hybrid Spectral Analysis for Window #{window_num_to_test}')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power (dB relative to peak)')
    plt.legend()
    plt.grid(True)
    plt.ylim(-60, 5)
    plt.show()