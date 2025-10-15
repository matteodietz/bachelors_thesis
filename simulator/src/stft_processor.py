import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path

# data loader and virtual afe
from afe_interface_rf import load_picmus_rf_data
from virtual_afe_rf import run_virtual_afe_processing_rf

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
            - freqs (np.ndarray): 1D array of sample frequencies (shifted to be centered at 0).
            - time_bins (np.ndarray): 1D array of segment times (depths).
            - spectrogram (np.ndarray): 3D array of STFT power (channels, freqs, time_bins).
    """
    if iq_data.ndim != 2:
        raise ValueError("Input iq_data must be a 2D array of shape (samples, channels).")

    num_channels = iq_data.shape[1]
    print(f"--- Running STFT analysis on {num_channels} channels ---")

    # the input to stft is of shape (channels, samples).
    freqs, time_bins, stft_complex = signal.stft(
        iq_data.T,  # transpose to (channels, samples)
        fs=fs,
        nperseg=nperseg,
        noverlap=overlap,
        window='hann',
        return_onesided=False # get both positive and negative frequencies for IQ data
    )
    
    # the output stft_complex is (channels, freqs, time_bins)
    # convert to power
    spectrogram_power = np.abs(stft_complex)**2
    
    # shift the frequency axis so that 0 Hz is in the center
    freqs = np.fft.fftshift(freqs)
    spectrogram_power = np.fft.fftshift(spectrogram_power, axes=1) # Shift along the frequency axis
    
    return freqs, time_bins, spectrogram_power
