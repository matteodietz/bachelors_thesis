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
    print("Data loaded successfully.")
    return iq_data, angles, fs

# --- UNIT TEST ---
if __name__ == '__main__':
    print("--- Running a quick PSD test on a single A-line from PICMUS I/Q data ---")

    # setup and data loading
    try:
        # assumes the script is in 'src/' and datasets are in '../datasets/'
        SIMULATOR_ROOT = Path(__file__).resolve().parent.parent
    except NameError:
        SIMULATOR_ROOT = Path.cwd().parent
    
    iq_path = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_dataset_iq.hdf5"
    
    try:
        picmus_data, angles, fs_picmus = load_picmus_iq_data(iq_path)
    except Exception as e:
        print(f"Test failed: Could not load data. Error: {e}")
        exit()

    # select data for a single channel and angle 
    center_angle_index = np.argmin(np.abs(angles))
    channel_to_plot = 64
    
    # get the 1D signal for a single, complete A-line
    # shape is now just (num_samples,)
    signal_1d = picmus_data[center_angle_index, channel_to_plot, :]
    
    print(f"\nAnalyzing data from center angle #{center_angle_index}, channel #{channel_to_plot}")
    print(f"Data shape: {signal_1d.shape}, Sample Rate: {fs_picmus/1e6:.2f} MHz")

    # calculate the Power Spectral Density (PSD) using Welch's method
    # return_onesided=False for complex I/Q data.
    freqs, psd = signal.welch(
        signal_1d,
        fs=fs_picmus,
        nperseg=256,         # Use a segment length for averaging
        return_onesided=False
    )
    
    # shift the frequency axis so 0 Hz is in the center
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)

    # plot the 1D Spectrum
    plt.figure(figsize=(12, 6))
    
    # convert power to a relative dB scale for better visualization
    psd_db = 10 * np.log10(psd + 1e-20)
    psd_db_normalized = psd_db - np.max(psd_db)
    
 
    plt.plot(freqs / 1e6, psd_db_normalized)
    
    # configure and show the plot 
    plt.title(f'Overall Power Spectrum of Raw PICMUS I/Q Data (Channel {channel_to_plot})')
    plt.ylabel('Power (dB relative to peak)')
    plt.xlabel('Frequency Offset (MHz)')
    plt.grid(True, which='both', linestyle='--')
    # plt.ylim(-60, 5) # Zoom in on the top 60 dB
    plt.xlim(-2.51, 2.51) # Show full frequency range
    
    plt.show()