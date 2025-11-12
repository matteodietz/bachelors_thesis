import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from afe_interface_rf import load_picmus_rf_data
from afe_interface import load_picmus_data
from virtual_afe import run_virtual_afe_processing

# --- Main Test Script ---
if __name__ == '__main__':
    print("--- Comparing Generated I/Q Data vs. PICMUS Ground Truth I/Q ---")

    # --- 1. Setup and Data Loading ---
    try:
        SIMULATOR_ROOT = Path(__file__).resolve().parent.parent
    except NameError:
        SIMULATOR_ROOT = Path.cwd().parent
    
    rf_path = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_dataset_rf.hdf5"
    iq_path = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_dataset_iq.hdf5"
    scan_path = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_scan.hdf5"
    
    adc_rate = 125e6

    # --- 2. Generate High-Fidelity I/Q Data ---
    print("\n--- Generating I/Q data from RF source ---")
    try:
        rf_data, angles, _, _, fs_picmus_rf, mod_freq, _, _, _ = load_picmus_rf_data(rf_path, iq_path, scan_path)
        center_angle_index = np.argmin(np.abs(angles))

        _, high_rate_iq_generated, _ = run_virtual_afe_processing(
            rf_data=rf_data,
            angle_index=center_angle_index,
            fs_picmus=fs_picmus_rf,
            modulation_frequency=mod_freq,
            decimation_factor=1,
            adc_sample_rate=adc_rate
        )
    except Exception as e:
        print(f"Failed to generate I/Q data. Error: {e}")
        exit()

    # --- 3. Load the PICMUS Ground Truth I/Q Data ---
    print("\n--- Loading ground truth I/Q data from PICMUS file ---")
    try:
        picmus_iq_data, angles_iq, _, _, fs_picmus_iq, _, _, _, _ = load_picmus_data(iq_path, scan_path)
        center_angle_index_iq = np.argmin(np.abs(angles_iq))
    except Exception as e:
        print(f"Failed to load ground truth I/Q data. Error: {e}")
        exit()

    # --- 4. Prepare Data for Plotting ---
    channel_to_inspect = 64
    
    # generated A-line (at high sample rate)
    your_iq_aline = high_rate_iq_generated[:, channel_to_inspect]
    your_time_axis = np.arange(len(your_iq_aline)) / adc_rate * 1e6 # Time in us
    your_envelope = np.abs(your_iq_aline)

    # Ground truth A-line (at low sample rate)
    gt_iq_aline = picmus_iq_data[center_angle_index_iq, channel_to_inspect, :]
    gt_time_axis = np.arange(len(gt_iq_aline)) / fs_picmus_iq * 1e6 # Time in us
    gt_envelope = np.abs(gt_iq_aline)

    # --- 5. Plot Figure 1: Generated Data ---
    plt.figure(figsize=(14, 7))
    plt.plot(your_time_axis, your_iq_aline.real, 'b-', linewidth=0.5, label='Generated I Component')
    plt.plot(your_time_axis, your_iq_aline.imag, 'r-', linewidth=0.5, label='Generated Q Component')
    plt.plot(your_time_axis, your_envelope, 'k-', linewidth=1.5, label='Generated Envelope')
    
    plt.title(f'YOUR Generated I/Q Data (Channel {channel_to_inspect}, fs={adc_rate/1e6:.2f} MHz)')
    plt.xlabel('Time (µs)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.suptitle("Figure 1: Your Generated Data", fontsize=16)

    # --- 6. Plot Figure 2: Ground Truth Data ---
    plt.figure(figsize=(14, 7))
    plt.plot(gt_time_axis, gt_iq_aline.real, 'b-', linewidth=0.5, label='PICMUS I Component')
    plt.plot(gt_time_axis, gt_iq_aline.imag, 'r-', linewidth=0.5, label='PICMUS Q Component')
    plt.plot(gt_time_axis, gt_envelope, 'k-', linewidth=1.5, label='PICMUS Envelope')
    
    plt.title(f'PICMUS Ground Truth I/Q Data (Channel {channel_to_inspect}, fs={fs_picmus_iq/1e6:.2f} MHz)')
    plt.xlabel('Time (µs)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.suptitle("Figure 2: Ground Truth Data", fontsize=16)
    
    # Show both plots
    plt.show()