import h5py
import numpy as np
from pathlib import Path

def load_picmus_rf_data(rf_path="../datasets/experiments/contrast_speckle/contrast_speckle_expe_dataset_rf.hdf5", iq_path="../datasets/experiments/contrast_speckle/contrast_speckle_expe_dataset_iq.hdf5", scan_path="../datasets/experiments/contrast_speckle/contrast_speckle_expe_scan.hdf5"):
    """
    Simulates the analog front-end interface by reading IQ and scan data from HDF5 files.
    """
    # Read HDF5 file containing IQ data
    with h5py.File(rf_path, 'r') as f:
        real = f["/US/US_DATASET0000/data/real"][:]
        imag = f["/US/US_DATASET0000/data/imag"][:]
        rf_data = real

        angles = f["/US/US_DATASET0000/angles"][:]
        probe_geometry = f["/US/US_DATASET0000/probe_geometry"][:]
        sound_speed = f["/US/US_DATASET0000/sound_speed"][0]
        sampling_frequency = f["/US/US_DATASET0000/sampling_frequency"][0] 
        # modulation_frequency = f["/US/US_DATASET0000/modulation_frequency"][0] # this is 0 instead of 5.21MHz
        initial_time = f["/US/US_DATASET0000/initial_time"][0]

    with h5py.File(iq_path, 'r') as f:
        modulation_frequency = f["/US/US_DATASET0000/modulation_frequency"][0] # need to take the mod freq from IQ dataset

    # Read HDF5 file containing grid to beamform on
    with h5py.File(scan_path, 'r') as f:
        x_axis = f["/US/US_DATASET0000/x_axis"][:]
        z_axis = f["/US/US_DATASET0000/z_axis"][:]

    print(f"Picmus dataset loaded successfully")

    return rf_data, angles, probe_geometry, sound_speed, sampling_frequency, modulation_frequency, initial_time, x_axis, z_axis


# --- UNIT TEST --
if __name__ == '__main__':
    print("--- Running afe_interface.py as a script for testing ---")
    
    # Use pathlib to make the paths robust
    try:
        SIMULATOR_ROOT = Path(__file__).parent.parent
    except NameError:
        SIMULATOR_ROOT = Path.cwd().parent
        
    # Define the default paths for the test
    rf_path_default = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_dataset_rf.hdf5"
    iq_path_default = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_dataset_iq.hdf5"
    scan_path_default = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_scan.hdf5"

    try:
        # Call the function and capture all outputs
        (rf_data, angles, probe_geometry, sound_speed, sampling_frequency, 
         modulation_frequency, initial_time, x_axis, z_axis) = load_picmus_rf_data(rf_path_default, iq_path_default, scan_path_default)
        
        print("\nSUCCESS: Script completed without errors.")
        
        # # Print the angles that were used in Acquisition
        # print("\n--- Dataset Information ---")
        # print(f"Number of plane wave angles: {len(angles)}")
        # print("Sequence of angles transmitted (in degrees):")
        
        # # Convert from radians to degrees and print in a nice format
        # angles_degrees = np.rad2deg(angles)
        # print(np.array2string(angles_degrees, precision=2, separator=', '))
        
    except Exception as e:
        # Print error for debugging purpose
        print(f"\nERROR: The script failed with an exception: {e}")