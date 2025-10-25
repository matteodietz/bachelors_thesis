import numpy as np
import scipy.io
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal

def load_pva_data(mat_file_path):
    """
    Loads raw RF data and metadata from a single .mat file from the PVA dataset.
    (Robust version 2.0 with corrected struct access)
    """
    print(f"--- Loading PVA dataset from: {Path(mat_file_path).name} ---")
    
    mat_data = scipy.io.loadmat(mat_file_path)
    print(f"data found")
    
    # --- Extract the raw data ---
    rf_data_raw = mat_data['bscan_imgrf'] # almost certainly needs to be bscan_imgrf
    print(f"shape of data before transpose: {rf_data_raw.shape}")

    plt.figure()
    plt.plot(rf_data_raw[:,1])          # column 0
    plt.title("Column 0 as A-scan")
    plt.show()

    rf_data = rf_data_raw.T   # maybe doesnt need to be transposed
    
    # --- Extract the metadata ---
    config_struct = mat_data['config']
    
    # 1. Get the names of the fields in the struct
    field_names = config_struct.dtype.names
    
    # 2. Extract the 'Fs' field. It's a (1,1) array containing the value.
    #    We access the first element of the struct, then the field by name,
    #    then the first element of the nested array.
    if 'Fs' in field_names:
        fs = config_struct[0, 0]['Fs'][0, 0]
        print(f"fs is given by {fs}")
    else:
        raise KeyError("Could not find 'Fs' field in the 'config' struct.")
        
    # Final check to ensure fs is a valid number
    if fs <= 0:
        raise ValueError(f"Loaded sampling frequency is invalid (fs = {fs}). Check the .mat file structure.")

    x_coords = mat_data['x'].ravel()
    z_coords = mat_data['z'].ravel()
    
    print("PVA data loaded successfully.")
    print(f" -> Original shape: {rf_data_raw.shape} (scan_lines, samples)")
    print(f" -> Transposed shape for processing: {rf_data.shape} (samples, channels)")
    print(f" -> Sampling Frequency: {fs :.2f} MHz") # This should now be correct

    return rf_data, fs*(10**6), x_coords, z_coords

# --- UNIT TEST ---
if __name__ == '__main__':
    print("--- Running unit test for pva_data_loader.py ---")

    try:
        DATASET_ROOT = Path(__file__).resolve().parent.parent / "datasets/pva"
    except NameError:
        DATASET_ROOT = Path.cwd().parent / "datasets/pva"
    
    mat_file_to_test = DATASET_ROOT / "Test_al_1por_mues1_barr2d" / "Test_al_1por_mues1_barr2d_1.mat"

    try:
        if not mat_file_to_test.exists():
             print(f"\nERROR: Could not find the test file at '{mat_file_to_test}'")
             print("Please make sure you have unzipped the dataset and the path is correct.")
             exit()
             
        rf_channels, fs, x, z = load_pva_data(mat_file_to_test)
    except Exception as e:
        print(f"An error occurred during loading: {e}")
        # Add traceback for more detailed debugging
        import traceback
        traceback.print_exc()
        exit()

    center_channel_index = rf_channels.shape[1] // 2 + 1
    signal_1d = rf_channels[:, center_channel_index]
    
    freqs, psd = signal.welch(signal_1d, fs=fs, nperseg=1024)

    plt.figure(figsize=(12, 6))
    psd_db = 10 * np.log10(psd + 1e-20)
    plt.plot(freqs / 1e6, psd_db)
    plt.title(f'Power Spectrum of a Single A-Scan (Channel #{center_channel_index})')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.grid(True, which='both', linestyle='--')
    plt.xlim(0, 100)
    plt.show()