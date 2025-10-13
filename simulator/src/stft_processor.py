import numpy as np
from scipy import signal

def run_stft_analysis(iq_data, fs, nperseg=256, overlap=128):
    """
    Performs a Short-Time Fourier Transform (STFT) on a multi-channel I/Q data array.

    Args:
        iq_data (np.ndarray): The 2D (samples, channels) I/Q data array.
        fs (float): The sample rate of the iq_data in Hz.
        nperseg (int): The length of each STFT segment (window).
        overlap (int): The number of samples to overlap between segments.

    Returns:
        tuple: A tuple containing:
            - freqs (np.ndarray): 1D array of sample frequencies.
            - time_bins (np.ndarray): 1D array of segment times (depths).
            - spectrogram (np.ndarray): 3D array of STFT power (channels, freqs, time_bins).
    """
    if iq_data.ndim != 2:
        raise ValueError("Input iq_data must be a 2D array of shape (samples, channels).")

    num_channels = iq_data.shape[1]
    print(f"--- Running STFT analysis on {num_channels} channels ---")

    # The STFT function in scipy conveniently works along an axis.
    # We want to perform the STFT on each channel (axis 0 of the transposed data).
    # The input to `stft` should be shape (channels, samples).
    freqs, time_bins, stft_complex = signal.stft(
        iq_data.T,  # Transpose to (channels, samples)
        fs=fs,
        nperseg=nperseg,
        noverlap=overlap,
        window='hann'
    )
    
    # The output `stft_complex` is (channels, freqs, time_bins)
    # We want to return the power, which is the squared magnitude.
    spectrogram_power = np.abs(stft_complex)**2
    
    return freqs, time_bins, spectrogram_power