import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path

# data loader for RF data
from afe_interface_rf import load_picmus_rf_data
from quick_spectrogram_test import load_picmus_iq_data

def run_virtual_afe_processing(rf_data, angle_index, fs_picmus, modulation_frequency, decimation_factor, adc_sample_rate=125e6, snr_db=None, transducer_bw_percent=67):
    """
    Performs the virtual AFE simulation on pre-loaded PICMUS RF data.

    This function simulates the full pipeline:
    1. Upsamples the RF data for a specific angle to the target ADC rate.
    2. Performs I/Q demodulation on the high-rate RF data.
    3. Decimates the resulting I/Q data by the specified factor.
    
    Args:
        rf_data (np.ndarray): The full, low-rate PICMUS RF data array (angles, channels, samples).
        angle_index (int): The index of the angle from the dataset to process.
        fs_picmus (float): The original sample rate of the PICMUS RF data.
        modulation_frequency (float): The center frequency of the transducer for demodulation.
        decimation_factor (int): The integer decimation factor to apply.
        adc_sample_rate (float): The target sample rate of the virtual ADC in Hz.

    Returns:
        tuple: A tuple containing:
            - decimated_iq (np.ndarray): The final decimated I/Q data.
            - high_rate_iq (np.ndarray): The intermediate high-rate I/Q data (before decimation).
            - fs_new (float): The sample rate of the decimated_iq data.
    """
    print(f"--- Running Virtual AFE Processing for M={decimation_factor} ---")
    
    # upsample RF data for the chosen angle
    data_for_one_angle_rf = rf_data[angle_index, :, :].T
    
    upsample_factor_num = int(adc_sample_rate)
    upsample_factor_den = int(fs_picmus)
    
    high_rate_rf = signal.resample_poly(data_for_one_angle_rf, up=upsample_factor_num, down=upsample_factor_den, axis=0)
    print(f"Upsampled RF data to shape: {high_rate_rf.shape}")

    # add AWGN to the high-rate RF signal
    if snr_db is not None:
        print(f"Adding AWGN to achieve an SNR of {snr_db} dB...")
        # calculate the power of the signal
        signal_power = np.var(high_rate_rf)
        
        # calculate the required noise power for the target SNR
        snr_linear = 10**(snr_db / 10)
        noise_power = signal_power / snr_linear
        
        # generate gaussian noise with the required power (std dev = sqrt(power))
        noise_std = np.sqrt(noise_power)
        noise = np.random.normal(loc=0.0, scale=noise_std, size=high_rate_rf.shape)
        
        # add the noise to the signal
        high_rate_rf = high_rate_rf + noise
        print("Noise addition complete.")

    # --- MATHEMATICALLY IDEAL BPF ---
    print("Applying ideal FFT-based band-pass filter to RF signal...")
    
    # define the passband based on transducer specs
    center_freq = modulation_frequency
    bandwidth = center_freq * (87 / 100.0)
    low_cutoff = center_freq - (bandwidth / 2)
    high_cutoff = center_freq + (bandwidth / 2)
    print(f"Ideal BPF Passband: [{low_cutoff/1e6:.2f}, {high_cutoff/1e6:.2f}] MHz")
    
    # go to the frequency domain
    spectrum_rf = np.fft.fft(high_rate_rf, axis=0)
    
    # create the frequency bins vector and the filter mask
    num_samples_high_rate = high_rate_rf.shape[0]
    freqs = np.fft.fftfreq(num_samples_high_rate, 1/adc_sample_rate)
    mask = np.where((np.abs(freqs) >= low_cutoff) & (np.abs(freqs) <= high_cutoff), 1, 0)
    
    # apply the filter mask
    filtered_spectrum_rf = spectrum_rf * mask[:, np.newaxis]
    
    # go back to time domain
    filtered_high_rate_rf = np.fft.ifft(filtered_spectrum_rf, axis=0).real # Use .real to discard tiny imaginary parts due to numerical precision
    print("Band-pass filtering complete.")
    # --- END OF MATHEMATICALLY IDEAL BPF ---

    # --- REALISTIC BPF ---
    # apply band-pass filter based on transducer specs
    # center_freq = modulation_frequency
    # bandwidth = center_freq * (transducer_bw_percent / 100)
    # low_cutoff = center_freq - (bandwidth / 2)
    # high_cutoff = center_freq + (bandwidth / 2)

    # num_taps = 101
    # nyquist = adc_sample_rate / 2

    # # create band-pass filter coefficients
    # bpf_coeffs = signal.firwin(num_taps, [low_cutoff, high_cutoff], fs=adc_sample_rate, pass_zero=False)

    # # apply the BPF to the high-rate RF data
    # filtered_high_rate_rf = signal.lfilter(bpf_coeffs, 1.0, high_rate_rf, axis=0)
    # print(f"Band-pass filtering complete.")
    # --- END OF REALISTIC BPF ---
    
    # I/Q Demodulation 
    # time vector for the high-rate RF signal
    print(f"Performing I/Q Demodulation...")
    num_samples_high_rate = filtered_high_rate_rf.shape[0]
    t = np.arange(num_samples_high_rate) / adc_sample_rate
    
    # Complex local oscillator signal
    # multiply by 2 to get the analytic signal (I + jQ) after filtering
    local_oscillator = 2 * np.exp(-1j * 2 * np.pi * modulation_frequency * t)
    
    # demodulate by multiplying the RF signal by the complex oscillator
    # reshape the local_oscillator to multiply it with each channel
    analytic_signal_passband = filtered_high_rate_rf * local_oscillator[:, np.newaxis]
    
    # low-pass filter the result to remove the 2*f_c component and keep the baseband signal
    # simple low-pass filter for this purpose
    # cutoff should be less than the modulation frequency
    # --- REALISTIC LPF ---
    # lpf_coeffs = signal.firwin(99, bandwidth / 2, fs=adc_sample_rate)
    
    # high_rate_iq = signal.lfilter(lpf_coeffs, 1.0, analytic_signal_passband, axis=0)
    # print(f"I/Q Demodulation complete. High-rate IQ shape: {high_rate_iq.shape}")

    # --- MATHEMATICALLY IDEAL LPF ---
    print("Applying ideal FFT-based low-pass filter...")
    
    # go to the frequency domain
    spectrum = np.fft.fft(analytic_signal_passband, axis=0)
    # absolute cutoff frequency in Hz
    abs_cutoff_hz = (modulation_frequency * (91 / 100.0)) / 2.0
    print(f"Ideal LPF cutoff frequency: {abs_cutoff_hz / 1e6:.2f} MHz")
    # create the frequency bins vector for this FFT
    freqs = np.fft.fftfreq(num_samples_high_rate, 1/adc_sample_rate)
    # filter is 1 inside the passband and 0 outside
    # two-sided spectrum (positive and negative frequencies)
    mask = np.where(np.abs(freqs) <= abs_cutoff_hz, 1, 0)
    # multiply the spectrum by the filter for each channel
    filtered_spectrum = spectrum * mask[:, np.newaxis]
    # go back to the time domain
    high_rate_iq = np.fft.ifft(filtered_spectrum, axis=0)
    print(f"I/Q Demodulation complete. High-rate IQ shape: {high_rate_iq.shape}")
    
    # --- END OF MATHEMATICALLY IDEAL LPF ---

    # add AWGN noise a second time
    if snr_db is not None:
        print(f"Adding AWGN to achieve an SNR of {snr_db} dB...")
        high_rate_iq = high_rate_iq + noise
        print("Noise addition complete.")

    # decimate the high-rate I/Q data
    if decimation_factor < 1:
        raise ValueError("Decimation factor must be >= 1.")
    if decimation_factor == 1:
        decimated_iq = high_rate_iq.copy()
    else:
        # the decimate function includes its own anti-aliasing filter
        decimated_iq = signal.decimate(high_rate_iq, q=decimation_factor, axis=0)
    
    fs_new = adc_sample_rate / decimation_factor
    
    return decimated_iq, high_rate_iq, fs_new


# --- UNIT TEST ---
if __name__ == '__main__':
    print("--- Running unit test for virtual_afe.py (RF input) ---")

    # define paths and parameters
    try:
        SIMULATOR_ROOT = Path(__file__).parent.parent
    except NameError:
        SIMULATOR_ROOT = Path.cwd().parent
    
    # !!! IMPORTANT: need all three paths since modulation_frequency in RF dataset is 0 !!!
    rf_path = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_dataset_rf.hdf5"
    iq_path = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_dataset_iq.hdf5"
    scan_path = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_scan.hdf5"
    
    baseline_decimation = 4
    test_decimation = 5
    adc_rate = 125e6
    snr = None

    # load the RF data
    try:
        print("\nLoading PICMUS RF dataset from disk...")
        rf_data, angles, _, _, fs_picmus, mod_freq, _, _, _ = load_picmus_rf_data(rf_path, iq_path, scan_path)
        print("RF Data loaded successfully.")
    except Exception as e:
        print(f"Test failed: Could not load data. Check your afe_interface.py. Error: {e}")
        exit()

    center_angle_index = np.argmin(np.abs(angles))

    # call the processing function for the baseline case
    baseline_data_iq, _, fs_baseline = run_virtual_afe_processing(
        rf_data=rf_data,
        angle_index=center_angle_index,
        fs_picmus=fs_picmus,
        modulation_frequency=mod_freq,
        decimation_factor=baseline_decimation,
        adc_sample_rate=adc_rate,
        snr_db=snr # None by default
    )
    print(f"SUCCESS: Got baseline I/Q data (M={baseline_decimation}) with shape: {baseline_data_iq.shape}")
        
    # call the processing function for the test case
    test_data_iq, _, fs_test = run_virtual_afe_processing(
        rf_data=rf_data,
        angle_index=center_angle_index,
        fs_picmus=fs_picmus,
        modulation_frequency=mod_freq,
        decimation_factor=test_decimation,
        adc_sample_rate=adc_rate,
        snr_db=snr # None by default
    )
    print(f"SUCCESS: Got test I/Q data (M={test_decimation}) with shape: {test_data_iq.shape}")
    
    # visual verification
    channel_to_plot = 64
    
    # plot the spectra of the generated I/Q data
    freqs_baseline, psd_baseline = signal.welch(baseline_data_iq[:, channel_to_plot], fs=fs_baseline, nperseg=1024)
    freqs_test, psd_test = signal.welch(test_data_iq[:, channel_to_plot], fs=fs_test, nperseg=1024)

    # convert to a relative dB scale
    # find the absolute peak power from the high-quality baseline signal to use as a reference
    peak_power = np.max(psd_baseline)
    print(f"peak_power = {peak_power}")
    
    # convert both PSDs to dB relative to this single peak.
    psd_baseline_db = 10 * np.log10(psd_baseline / peak_power)
    psd_test_db = 10 * np.log10(psd_test / peak_power)


    # --- ADD non upsampled I/Q data as comparison ---
    try:
        picmus_data, angles, fs_picmus = load_picmus_iq_data(iq_path)
    except Exception as e:
        print(f"Test failed: Could not load data. Error: {e}")
        exit()

    signal_iq_non_upsampled = picmus_data[center_angle_index, channel_to_plot, :]
    
    print(f"\nAnalyzing data from center angle #{center_angle_index}, channel #{channel_to_plot}")
    print(f"Data shape: {signal_iq_non_upsampled.shape}, Sample Rate: {fs_picmus/1e6:.2f} MHz")

    # calculate the Power Spectral Density (PSD) using Welch's method
    # return_onesided=False for complex I/Q data.
    freqs, psd = signal.welch(
        signal_iq_non_upsampled,
        fs=fs_picmus,
        nperseg=256,         # Use a segment length for averaging
        return_onesided=False
    )
    
    # shift the frequency axis so 0 Hz is in the center
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)
    
    # convert power to a relative dB scale for better visualization
    psd_db = 10 * np.log10(psd + 1e-20)
    psd_db_normalized = psd_db - np.max(psd_db)
    # --- END OF non upsampled I/Q data

    
    # create the plot using a linear y-axis
    plt.figure(figsize=(12, 6))
    
    # use plt.plot, not plt.semilogy, because the data is now already in a log (dB) scale
    plt.plot(freqs_baseline / 1e6, psd_baseline_db, label=f'Baseline I/Q Spectrum (M={baseline_decimation}, fs={fs_baseline/1e6:.2f} MHz)')
    plt.plot(freqs_test / 1e6, psd_test_db, label=f'Test I/Q Spectrum (M={test_decimation}, fs={fs_test/1e6:.2f} MHz)')
    plt.plot(freqs / 1e6, psd_db_normalized, label=f'Non Upsampled I/Q Data') # non upsampled I/Q data
    
    plt.title(f'Normalized Power Spectral Density of Generated I/Q Data (Channel {channel_to_plot})')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power (dB relative to peak)') # update the y-axis label
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    
    # set the axis limits
    plt.ylim(-100, 5)
    plt.xlim(-2.51, 2.51)

    # define the output path and create the directory if it doesn't exist
    plots_dir = Path(__file__).resolve().parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True) # exist_ok=True prevents an error if the folder already exists
    file_name = "virtual_afe.png"
    output_path = plots_dir / file_name

    # save the figure to the specified path
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"\nSUCCESS: Plot saved to {output_path}")
    plt.close()
