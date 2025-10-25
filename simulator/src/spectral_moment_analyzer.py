import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path

# Import all the functional blocks 
from afe_interface_rf import load_picmus_rf_data
from virtual_afe import run_virtual_afe_processing
from hcsps import hcsps_peak_search 

def spectral_moment_analysis(spectrum_slice, freqs, f_peak_hz):
    """
    Estimates bandwidth from a power spectrum slice using the spectral moment method,
    referenced to a pre-calculated peak frequency.
    """
    # (Function content is the same as before)
    if len(spectrum_slice) != len(freqs):
        raise ValueError("Spectrum and frequency arrays must have the same length.")
    
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    S0 = np.sum(spectrum_slice) * df
    
    if S0 == 0:
        return 0, 0

    variance = np.sum((freqs - f_peak_hz)**2 * spectrum_slice) * df / S0
    sigma_f = np.sqrt(variance)
    
    bw_est_hz = 6.06 * sigma_f
    
    return bw_est_hz, sigma_f

# --- UNIT TEST ---
if __name__ == '__main__':
    print("--- Running Hybrid HCSPS + Spectral Moment Test on Real PICMUS Data ---")

    # --- 1. Setup and Data Loading ---
    # ... (Setup and data loading code is the same as before)
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
    if len(above_thresh_indices) > 0:
        true_lower_edge = freqs_full[above_thresh_indices[0]]
        true_upper_edge = freqs_full[above_thresh_indices[-1]]
        true_bandwidth = true_upper_edge - true_lower_edge
    else:
        true_bandwidth = float('nan')

    # --- 5. Run Hybrid Algorithm ---
    # 5a: Find the precise peak with HCSPS
    hcsps_peak_index, _ = hcsps_peak_search(
        b=time_window_data, N=nperseg, k=32, beta=0.5, window='hann'
    )
    freqs_unshifed = np.fft.fftfreq(nperseg, 1/fs_baseline)
    hcsps_peak_freq = freqs_unshifed[hcsps_peak_index]
    
    # 5b: Get the standard N-point spectrum for the moment calculation
    standard_fft = np.fft.fft(time_window_data * signal.windows.hann(nperseg))
    standard_powers = np.abs(standard_fft)**2
    freqs_unshifed_for_moments = np.fft.fftfreq(nperseg, 1/fs_baseline)
    
    # 5c: Run the spectral moment analysis, referencing the HCSPS peak
    estimated_bw, sigma_f = spectral_moment_analysis(
        spectrum_slice=standard_powers,
        freqs=freqs_unshifed_for_moments,
        f_peak_hz=hcsps_peak_freq
    )

    # --- 6. Compare the Results ---
    print("\n--- Bandwidth Estimation Comparison on REAL Data ---")
    print(f"HCSPS Estimated Peak:             {hcsps_peak_freq / 1e6:6.3f} MHz")
    print(f"Ground Truth Bandwidth (@ {threshold_db} dB): {true_bandwidth / 1e6:6.3f} MHz")
    print(f"Hybrid HCSPS+Moment BW Est.:      {estimated_bw / 1e6:6.3f} MHz (sigma_f = {sigma_f/1e6:.3f} MHz)")

    # --- 7. Visual Confirmation Plot ---
    plt.figure(figsize=(12, 6))
    # Shift spectra for plotting
    freqs_standard_shifted = np.fft.fftshift(freqs_unshifed_for_moments)
    psd_db_standard_norm = 10 * np.log10(np.fft.fftshift(standard_powers) + 1e-20)
    psd_db_standard_norm -= np.max(psd_db_standard_norm)
    
    plt.plot(freqs_standard_shifted / 1e6, psd_db_standard_norm, label=f'Standard {nperseg}-pt Spectrum')
    plt.axhline(y=threshold_db, color='grey', linestyle=':', label=f'{threshold_db} dB Threshold')
    
    # --- ADDED PEAK VISUALIZATION ---
    plt.axvline(x=hcsps_peak_freq / 1e6, color='purple', linestyle='-.', label=f'HCSPS Peak ({hcsps_peak_freq/1e6:.3f} MHz)')
    # --- END ADDITION ---

    # Plot the estimated bandwidth as a shaded region
    est_lower = hcsps_peak_freq - estimated_bw / 2
    est_upper = hcsps_peak_freq + estimated_bw / 2
    plt.axvspan(est_lower/1e6, est_upper/1e6, color='red', alpha=0.2, label=f'Moment-Estimated BW ({estimated_bw/1e6:.3f} MHz)')

    plt.title(f'Hybrid Bandwidth Estimation for Window #{window_num_to_test}')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power (dB relative to peak)')
    plt.legend()
    plt.grid(True)
    plt.ylim(-60, 5)
    plt.show()