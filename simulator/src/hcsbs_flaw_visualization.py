import numpy as np
from scipy import signal
from pathlib import Path
import matplotlib.pyplot as plt

# Import all the functional blocks
from afe_interface_rf import load_picmus_rf_data
from virtual_afe import run_virtual_afe_processing


# --- MAIN SCRIPT ---
if __name__ == '__main__':
    print("--- Visualizing the Conceptual Flaw in the Coarse FFT Estimator ---")

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
        # Assuming you have a virtual_afe.py with run_virtual_afe_processing
        # that can handle RF input.
        from virtual_afe import run_virtual_afe_processing
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
    nperseg = 256 # Window size for the "fine" FFT
    k = 32      # Window size for the "coarse" FFT
    channel_to_test = 64
    window_num_to_test = 30
    
    # --- THIS IS THE FIX ---
    hop = 128   # Define hop size
    start_sample = window_num_to_test * hop
    # --- END OF FIX ---
    end_sample = start_sample + nperseg
    
    time_window_data = baseline_iq_data[start_sample:end_sample, channel_to_test]
    
    print(f"\n--- Analyzing STFT window #{window_num_to_test} (samples {start_sample}-{end_sample}) ---")

    # --- 4. Calculate BOTH Spectra ---
    
    # a) Fine Spectrum (Ground Truth)
    window_fine = signal.windows.hann(nperseg)
    fine_fft = np.fft.fftshift(np.fft.fft(time_window_data * window_fine))
    freqs_fine = np.fft.fftshift(np.fft.fftfreq(nperseg, 1/fs_baseline))
    psd_db_fine = 10 * np.log10(np.abs(fine_fft)**2 + 1e-20)
    psd_db_fine_norm = psd_db_fine - np.max(psd_db_fine)
    
    # b) Coarse Spectrum (from truncated signal)
    window_coarse = signal.windows.hann(k)
    time_window_coarse = time_window_data[:k] # Truncate the signal
    coarse_fft = np.fft.fftshift(np.fft.fft(time_window_coarse * window_coarse))
    freqs_coarse = np.fft.fftshift(np.fft.fftfreq(k, 1/fs_baseline))
    psd_db_coarse = 10 * np.log10(np.abs(coarse_fft)**2 + 1e-20)
    # Normalize by the peak of the FINE spectrum for a fair comparison
    psd_db_coarse_norm = psd_db_coarse - np.max(psd_db_fine) 

    # --- 5. Visual Verification ---
    threshold_db = -20
    
    plt.figure(figsize=(14, 7))
    plt.plot(freqs_fine / 1e6, psd_db_fine_norm, 'g-', label=f'Fine {nperseg}-pt Spectrum (Ground Truth)')
    plt.plot(freqs_coarse / 1e6, psd_db_coarse_norm, 'b-o', label=f'Coarse {k}-pt Spectrum (from truncated signal)')
    
    plt.axhline(y=threshold_db, color='grey', linestyle=':', label=f'{threshold_db} dB Threshold')
    
    plt.title(f'Power Spectrum Comparison for Window #{window_num_to_test} (Demonstrating Spectral Leakage)')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power (dB relative to peak)')
    plt.legend()
    plt.grid(True)
    plt.ylim(-80, 5)
    plt.show()