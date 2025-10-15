import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path
import h5py

def load_picmus_iq_data(iq_path):
    """
    A simplified loader to get only the I/Q data and essential parameters.
    """
    print(f"--- Loading PICMUS I/Q data from: {Path(iq_path).name} ---")
    with h5py.File(iq_path, 'r') as f:
        base_path = "/US/US_DATASET0000"
        real = f[f"{base_path}/data/real"][:]
        imag = f[f"{base_path}/data/imag"][:]
        iq_data = real + 1j * imag
        angles = f[f"{base_path}/angles"][:]
        fs = f[f"{base_path}/sampling_frequency"][0]
        initial_time = f[f"{base_path}/initial_time"][0]
        sound_speed = f[f"{base_path}/sound_speed"][0]
    print("Data loaded successfully.")
    return iq_data, angles, fs, initial_time, sound_speed

def run_stft_analysis(iq_data, fs, nperseg=256, hop=128):
    """
    Performs STFT on multi-channel I/Q data using scipy.signal.stft function.

    Args:
        iq_data (np.ndarray): The 2D (samples, channels) I/Q data array.
        fs (float): The sample rate of the iq_data in Hz.
        nperseg (int): The length of each STFT segment (window).
        hop (int): The number of samples to advance between segments (hop size).

    Returns:
        tuple: A tuple containing:
            - freqs (np.ndarray): 1D array of sample frequencies (centered at 0).
            - time_bins (np.ndarray): 1D array of segment times (depths).
            - spectrogram (np.ndarray): 3D array of STFT power (channels, freqs, time_bins).
    """
    if iq_data.ndim != 2:
        raise ValueError("Input iq_data must be a 2D array of shape (samples, channels).")

    print(f"--- Running STFT analysis on {iq_data.shape[1]} channels ---")
    
    # Calculate the overlap from the hop size
    noverlap = nperseg - hop

    # Use the standard stft function. It's direct and robust for this task.
    # The key parameters for complex I/Q data are:
    #   - return_onesided=False: Gets the full spectrum (-fs/2 to +fs/2).
    #   - axis=0: Performs the transform along the time axis (the first dimension).
    freqs, time_bins, stft_complex = signal.stft(
        iq_data,
        fs=fs,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False,
        axis=0
    )
    # The output stft_complex has shape (freqs, channels, time_bins)
    
    # We want power, which is the squared magnitude.
    spectrogram_power = np.abs(stft_complex)**2
    
    # Transpose the axes to our desired (channels, freqs, time_bins) format
    spectrogram_power = np.transpose(spectrogram_power, (1, 0, 2))
    
    # The `stft` function returns a "wrapped" frequency array. We need to shift it
    # so that 0 Hz is in the center for correct plotting and analysis.
    freqs = np.fft.fftshift(freqs)
    spectrogram_power = np.fft.fftshift(spectrogram_power, axes=1) # Shift along the frequency axis
    
    return freqs, time_bins, spectrogram_power


# --- Main execution block to run the test ---
if __name__ == '__main__':
    print("--- Running a direct STFT test on PICMUS I/Q data ---")

    # --- 1. Setup and Data Loading ---
    try:
        SIMULATOR_ROOT = Path(__file__).resolve().parent.parent
    except NameError:
        SIMULATOR_ROOT = Path.cwd().parent
    
    iq_path = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_dataset_iq.hdf5"
    
    try:
        picmus_data, angles, fs_picmus, initial_time, sound_speed = load_picmus_iq_data(iq_path)
    except Exception as e:
        print(f"Test failed: Could not load data. Error: {e}")
        exit()

    # --- 2. Select data for a single angle ---
    center_angle_index = np.argmin(np.abs(angles))
    data_for_one_angle = picmus_data[center_angle_index, :, :].T
    print(f"\nAnalyzing data from center angle #{center_angle_index} ({np.rad2deg(angles[center_angle_index]):.2f} degrees)")
    
    # --- 3. Run the STFT Processor to get the full spectrogram ---
    freqs, time_bins, spectrogram = run_stft_analysis(
        iq_data=data_for_one_angle,
        fs=fs_picmus,
        nperseg=128,
        hop=64
    )
    
    # Average across all channels for a robust, clean spectrogram
    avg_spectrogram = np.mean(spectrogram, axis=0)
    
    # --- 4. Select and Plot Spectra at 5 Different Depths ---
    print("\nSelecting and plotting spectra at 5 different depths...")
    
    num_time_bins = avg_spectrogram.shape[1]
    indices_to_plot = np.linspace(1, num_time_bins - 2, 5, dtype=int)
    
    plt.figure(figsize=(14, 7))
    
    for i, time_idx in enumerate(indices_to_plot):
        spectrum_slice = avg_spectrogram[:, time_idx]
        time_s = initial_time + time_bins[time_idx]
        depth_mm = time_s * sound_speed / 2 * 1000

        spectrum_db = 10 * np.log10(spectrum_slice + 1e-20)
        spectrum_db_normalized = spectrum_db - np.max(spectrum_db)
        
        plt.plot(freqs / 1e6, spectrum_db_normalized, label=f'Depth = {depth_mm:.1f} mm')

    # --- 5. Configure and Show the Plot ---
    plt.title(f'Normalized Power Spectrum at Different Depths (fs = {fs_picmus/1e6:.2f} MHz)')
    plt.xlabel('Frequency Offset (MHz)')
    plt.ylabel('Power (dB relative to peak)')
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plt.ylim(-40, 5)
    plt.xlim(-fs_picmus / 2 / 1e6, fs_picmus / 2 / 1e6)
    plt.show()