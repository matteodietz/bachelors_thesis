import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path

# data loader for RF data
from afe_interface_rf import load_picmus_rf_data

def run_virtual_afe_processing_rf(rf_data, angle_index, fs_picmus, modulation_frequency, decimation_factor, adc_sample_rate=125e6, snr_db=None):
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
    
    # I/Q Demodulation 
    # time vector for the high-rate RF signal
    num_samples_high_rate = high_rate_rf.shape[0]
    t = np.arange(num_samples_high_rate) / adc_sample_rate
    
    # Complex local oscillator signal
    # multiply by 2 to get the analytic signal (I + jQ) after filtering
    local_oscillator = 2 * np.exp(-1j * 2 * np.pi * modulation_frequency * t)
    
    # demodulate by multiplying the RF signal by the complex oscillator
    # reshape the local_oscillator to multiply it with each channel
    analytic_signal_passband = high_rate_rf * local_oscillator[:, np.newaxis]
    
    # low-pass filter the result to remove the 2*f_c component and keep the baseband signal
    # simple low-pass filter for this purpose
    # cutoff should be less than the modulation frequency
    num_taps = 99
    lpf_cutoff = modulation_frequency / (adc_sample_rate / 2)
    print(f"lpf_cutoff frequency is: {lpf_cutoff}")
    lpf_coeffs = signal.firwin(num_taps, lpf_cutoff)
    
    high_rate_iq = signal.lfilter(lpf_coeffs, 1.0, analytic_signal_passband, axis=0)
    print(f"I/Q Demodulation complete. High-rate IQ shape: {high_rate_iq.shape}")

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
    test_decimation = 14
    adc_rate = 125e6
    snr = 70

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
    baseline_data_iq, _, fs_baseline = run_virtual_afe_processing_rf(
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
    test_data_iq, _, fs_test = run_virtual_afe_processing_rf(
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
    
    # convert both PSDs to dB relative to this single peak.
    psd_baseline_db = 10 * np.log10(psd_baseline / peak_power)
    psd_test_db = 10 * np.log10(psd_test / peak_power)
    
    # create the plot using a linear y-axis
    plt.figure(figsize=(12, 6))
    
    # use plt.plot, not plt.semilogy, because the data is now already in a log (dB) scale
    plt.plot(freqs_baseline / 1e6, psd_baseline_db, label=f'Baseline I/Q Spectrum (M={baseline_decimation}, fs={fs_baseline/1e6:.2f} MHz)')
    plt.plot(freqs_test / 1e6, psd_test_db, label=f'Test I/Q Spectrum (M={test_decimation}, fs={fs_test/1e6:.2f} MHz)')
    
    plt.title(f'Normalized Power Spectral Density of Generated I/Q Data (Channel {channel_to_plot})')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power (dB relative to peak)') # update the y-axis label
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    
    # set the y-axis limits to zoom in on the important part of the spectrum
    # plt.ylim(-80, 5) # Show from +5dB down to -80dB
    
    plt.show()