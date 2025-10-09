import h5py
import numpy as np

def load_picmus_data(iq_path="../datasets/experiments/contrast_speckle/contrast_speckle_expe_dataset_iq.hdf5", scan_path="../datasets/experiments/contrast_speckle/contrast_speckle_expe_scan.hdf5"):
    """
    Simulates the analog front-end interface by reading IQ and scan data from HDF5 files.
    """
    # Read HDF5 file containing IQ data
    with h5py.File(iq_path, 'r') as f:
        real = f["/US/US_DATASET0000/data/real"][:]
        imag = f["/US/US_DATASET0000/data/imag"][:]
        iq_data = real + 1j * imag

        angles = f["/US/US_DATASET0000/angles"][:]
        probe_geometry = f["/US/US_DATASET0000/probe_geometry"][:]
        sound_speed = f["/US/US_DATASET0000/sound_speed"][0]
        sampling_frequency = f["/US/US_DATASET0000/sampling_frequency"][0]
        modulation_frequency = f["/US/US_DATASET0000/modulation_frequency"][0]
        initial_time = f["/US/US_DATASET0000/initial_time"][0]

    # Read HDF5 file containing grid to beamform on
    with h5py.File(scan_path, 'r') as f:
        x_axis = f["/US/US_DATASET0000/x_axis"][:]
        z_axis = f["/US/US_DATASET0000/z_axis"][:]

    return iq_data, angles, probe_geometry, sound_speed, sampling_frequency, modulation_frequency, initial_time, x_axis, z_axis


# Making the file runnable for testing resp. finding errors
# if __name__ == '__main__':
#     print("--- Running afe_interface.py as a script for testing ---")
#     try:
#         # Call the function to actually run the code
#         simulate_afe_interface()
#         print("\nSUCCESS: Script completed without errors.")
#     except Exception as e:
#         # This will catch any error and print
#         print(f"\nERROR: The script failed with an exception: {e}")