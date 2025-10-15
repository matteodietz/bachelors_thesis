import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path

# Import the RF-specific data loader and virtual AFE
from afe_interface_rf import load_picmus_rf_data
from virtual_afe_rf import run_virtual_afe_processing_rf

def run_stft_analysis_rf(rf_data, fs, nperseg=256, hop=128):
    """
    Performs a Short-Time Fourier Transform (STFT) on a multi-channel RF data array.

    Args:
        rf_data (np.ndarray): The 2D (samples, channels) real-valued RF data array.
        fs (float): The sample rate of the rf_data in Hz.
        nperseg (int): The length of each STFT segment (window).
        hop (int): The number of samples to advance between segments.

    Returns:
        tuple: A tuple containing:
            - freqs (np.ndarray): 1D array of sample frequencies (one-sided, 0 to fs/2).
            - time_bins (np.ndarray): 1D array of segment times (depths).
            - spectrogram (np.ndarray): 3D array of STFT power (channels, freqs, time_bins).
    """
    if rf_data.ndim != 2:
        raise ValueError("Input rf_data must be a 2D array of shape (samples, channels).")
    if np.iscomplexobj(rf_data):
        raise TypeError("Input rf_data must be real-valued.")

    num_channels = rf_data.shape[1]
    print(f"--- Running STFT analysis on {num_channels} RF channels ---")
    
    noverlap = nperseg - hop

    # Use the standard `stft` function. For real input, it automatically returns a one-sided spectrum.
    # `return_onesided=True` is the default for real data, so we don't need to specify it.
    freqs, time_bins, stft_complex = signal.stft(
        rf_data,
        fs=fs,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        axis=0  # Perform STFT along the first axis (time)
    )
    # The output `stft_complex` has shape (freqs, channels, time_bins)
    
    # We want power, which is the squared magnitude.
    spectrogram_power = np.abs(stft_complex)**2
    
    # Transpose the axes to our desired (channels, freqs, time_bins) format
    spectrogram_power = np.transpose(spectrogram_power, (1, 0, 2))
    
    # No fftshift is needed for a one-sided spectrum.
    return freqs, time_bins, spectrogram_power