import numpy as np
from scipy import signal
from pathlib import Path
import matplotlib.pyplot as plt

# import all the functional blocks
from afe_interface_rf import load_picmus_rf_data
from virtual_afe import run_virtual_afe_processing
from hcsps import hcsps_peak_search

# --- MAIN SCRIPT ---
if __name__ == '__main__':
    print("--- Running HCSPS Peak Search Test on Real PICMUS Data ---")

    # --- setup and data loading ---
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

    # --- get high-fidelity baseline I/Q data ---
    center_angle_index = np.argmin(np.abs(angles))

    baseline_iq_data, _, fs_baseline = run_virtual_afe_processing(
        rf_data=rf_data,
        angle_index=center_angle_index,
        fs_picmus=fs_picmus,
        modulation_frequency=mod_freq,
        decimation_factor=baseline_decimation,
        adc_sample_rate=adc_rate
    )
    
    # --- select ONE STFT window to analyze ---
    nperseg = 256 # window size
    hop = 128
    channel_to_test = 64
    
    window_num_to_test = 30
    start_sample = window_num_to_test * hop
    end_sample = start_sample + nperseg
    
    # extract 1D time-domain signal for this single window and channel
    time_window_data = baseline_iq_data[start_sample:end_sample, channel_to_test]
    
    print(f"\n--- Analyzing STFT window #{window_num_to_test} (samples {start_sample}-{end_sample}) ---")

    
    # find peak with standard windowed fft
    standard_fft = np.fft.fftshift(np.fft.fft(time_window_data * signal.windows.hann(nperseg)))
    freqs_standard = np.fft.fftshift(np.fft.fftfreq(nperseg, 1/fs_baseline))
    
    standard_peak_index_shifted = np.argmax(np.abs(standard_fft)**2)
    standard_peak_freq = freqs_standard[standard_peak_index_shifted]

    # HCSPS peak search algorithm
    beta_to_test = 0.8
    
    hcsps_peak_index, _ = hcsps_peak_search(
        b=time_window_data,
        N=nperseg,
        k=32,            # 32-point coarse FFT
        beta=beta_to_test,
        window='hann'
    )
    # the HCSPS index is a standard, non-shifted index on the nperseg-point grid
    freqs_unshifed = np.fft.fftfreq(nperseg, 1/fs_baseline)
    hcsps_peak_freq = freqs_unshifed[hcsps_peak_index]

    # compare the results
    print("\n--- Peak Frequency Estimation Comparison on REAL Data ---")
    print(f"Standard {nperseg}-point FFT Estimate:    {standard_peak_freq / 1e6:.4f} MHz")
    print(f"HCSPS Estimate (beta={beta_to_test}):    {hcsps_peak_freq / 1e6:.4f} MHz")

    # visual verification
    plt.figure(figsize=(12, 6))
    plt.plot(freqs_standard / 1e6, 10 * np.log10(np.abs(standard_fft)**2), 'o-', label=f'Standard {nperseg}-pt FFT')
    plt.axvline(hcsps_peak_freq / 1e6, color='r', linestyle='--', label=f'HCSPS Estimate ({hcsps_peak_freq/1e6:.4f} MHz)')
    plt.title(f'Spectral Peak Estimation for Window #{window_num_to_test}')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power (dB)')
    plt.legend()
    plt.grid(True)
    # plt.xlim(hcsps_peak_freq / 1e6 - 1, hcsps_peak_freq / 1e6 + 1) # zoom in around the peak
    plt.show()