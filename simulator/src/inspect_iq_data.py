import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# import data loaders and virtual afe
from afe_interface_rf import load_picmus_rf_data
from afe_interface import load_picmus_data
from virtual_afe import run_virtual_afe_processing

# --- main test script ---
if __name__ == '__main__':
    print("--- Comparing Generated I/Q Data vs. PICMUS Ground Truth I/Q ---")

    # --- 1. setup and data loading ---
    try:
        SIMULATOR_ROOT = Path(__file__).resolve().parent.parent
    except NameError:
        SIMULATOR_ROOT = Path.cwd().parent
    
    # change dataset as needed
    rf_path = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_dataset_rf.hdf5"
    iq_path = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_dataset_iq.hdf5"
    scan_path = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_scan.hdf5"
    
    adc_rate = 125e6
    decimation_for_generation = 1 

    # --- 2. load picmus data ---
    print("\n--- Loading all necessary datasets from disk ---")
    try:
        rf_data, angles_rf, _, _, fs_picmus_rf, mod_freq, _, _, _ = load_picmus_rf_data(rf_path, iq_path, scan_path)
        picmus_iq_data, angles_iq, _, _, fs_picmus_iq, _, _, _, _ = load_picmus_data(iq_path, scan_path)
        print("All data loaded successfully.")
    except Exception as e:
        print(f"Failed to load data. Error: {e}")
        exit()

    # --- 3. define test cases ---
    num_random_channels = 50
    num_angles_per_channel = 10
    num_points_to_trim = 20
    
    channel_for_plot = 64
    angle_for_plot = np.argmin(np.abs(angles_iq))
    
    total_angles, total_channels, _ = picmus_iq_data.shape
    
    possible_random_channels = np.delete(np.arange(total_channels), channel_for_plot)
    channels_to_test = np.random.choice(possible_random_channels, size=num_random_channels, replace=False)
    
    test_configs = []
    for ch in channels_to_test:
        angles_for_ch = np.random.choice(np.arange(total_angles), size=num_angles_per_channel, replace=False)
        for ang in angles_for_ch:
            test_configs.append({'channel': ch, 'angle': ang})
            
    test_configs.append({'channel': channel_for_plot, 'angle': angle_for_plot})
    
    all_max_errors_real, all_avg_errors_real, all_norm_errors_real, all_norm_avg_errors_real = [], [], [], []
    all_max_errors_imag, all_avg_errors_imag, all_norm_errors_imag, all_norm_avg_errors_imag = [], [], [], []
    
    # --- 4. main test loop ---
    print(f"\n--- Running {len(test_configs)} total test cases ---")
    
    for i, config in enumerate(test_configs):
        channel = config['channel']
        angle = config['angle']
        
        print(f"  -> Testing Case {i+1}/{len(test_configs)}: Channel {channel}, Angle {angle}...")
        
        # --- generate high-fidelity I/Q data for this case ---
        _, high_rate_iq_generated, _ = run_virtual_afe_processing(
            rf_data=rf_data, angle_index=angle, fs_picmus=fs_picmus_rf,
            modulation_frequency=mod_freq, decimation_factor=decimation_for_generation,
            adc_sample_rate=adc_rate
        )
        
        # --- prepare data for comparison ---
        my_iq_aline = high_rate_iq_generated[:, channel]
        my_time_axis = np.arange(len(my_iq_aline)) / adc_rate

        gt_iq_aline = picmus_iq_data[angle, channel, :]
        gt_time_axis = np.arange(len(gt_iq_aline)) / fs_picmus_iq
        
        # --- quantitative error analysis (with trimming) ---
        gt_time_axis_trimmed = gt_time_axis[:-num_points_to_trim]
        gt_iq_aline_trimmed = gt_iq_aline[:-num_points_to_trim]

        # need to interpolate otherwise points of the two signals don't align
        my_iq_resampled_real = np.interp(gt_time_axis_trimmed, my_time_axis, my_iq_aline.real)
        my_iq_resampled_imag = np.interp(gt_time_axis_trimmed, my_time_axis, my_iq_aline.imag)
        
        error_real = np.abs(my_iq_resampled_real - gt_iq_aline_trimmed.real)
        error_imag = np.abs(my_iq_resampled_imag - gt_iq_aline_trimmed.imag)
        
        gt_range_real = np.max(gt_iq_aline_trimmed.real) - np.min(gt_iq_aline_trimmed.real)
        gt_range_imag = np.max(gt_iq_aline_trimmed.imag) - np.min(gt_iq_aline_trimmed.imag)

        # store the results
        all_max_errors_real.append(np.max(error_real))
        all_avg_errors_real.append(np.mean(error_real))
        all_norm_errors_real.append(100 * np.max(error_real) / gt_range_real if gt_range_real > 0 else 0)
        all_norm_avg_errors_real.append(100 * np.mean(error_real) / gt_range_real if gt_range_real > 0 else 0)
        
        all_max_errors_imag.append(np.max(error_imag))
        all_avg_errors_imag.append(np.mean(error_imag))
        all_norm_errors_imag.append(100 * np.max(error_imag) / gt_range_imag if gt_range_imag > 0 else 0)
        all_norm_avg_errors_imag.append(100 * np.mean(error_imag) / gt_range_imag if gt_range_imag > 0 else 0)

        # save the data for the last case (fixed) for plotting
        if i == len(test_configs) - 1:
            my_iq_aline_plot = my_iq_aline
            my_time_axis_plot = my_time_axis
            gt_iq_aline_plot = gt_iq_aline
            gt_time_axis_plot = gt_time_axis

    # --- 5. final averaged results ---
    print("\n" + "="*30 + " FINAL AVERAGED RESULTS " + "="*30)
    print(f"Averaged over {len(all_max_errors_real)} total test cases.\n")
    
    print(f"Real (I) Component:")
    print(f"  - Average of Max Errors: {np.mean(all_max_errors_real):.4e}")
    print(f"  - Average of Average Errors: {np.mean(all_avg_errors_real):.4e}")
    print(f"  - Average Max Error as % of Dynamic Range: {np.mean(all_norm_errors_real):.2f}%")
    print(f"  - Average Mean Error as % of Dynamic Range: {np.mean(all_norm_avg_errors_real):.2f}%")
    
    print(f"\nImaginary (Q) Component:")
    print(f"  - Average of Max Errors: {np.mean(all_max_errors_imag):.4e}")
    print(f"  - Average of Average Errors: {np.mean(all_avg_errors_imag):.4e}")
    print(f"  - Average Max Error as % of Dynamic Range: {np.mean(all_norm_errors_imag):.2f}%")
    print(f"  - Average Mean Error as % of Dynamic Range: {np.mean(all_norm_avg_errors_imag):.2f}%")
    
   # --- 6. Plotting (with subplots for direct comparison) ---
    my_envelope_plot = np.abs(my_iq_aline_plot)
    gt_envelope_plot = np.abs(gt_iq_aline_plot)

    # Create a single figure with two subplots stacked vertically
    # `sharex=True` links the x-axes so zooming in one zooms the other.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # --- Top Subplot (ax1): Ground Truth Data ---
    ax1.plot(gt_time_axis_plot * 1e6, gt_iq_aline_plot.real, 'b-', linewidth=0.5, label='PICMUS I Component')
    ax1.plot(gt_time_axis_plot * 1e6, gt_iq_aline_plot.imag, 'r-', linewidth=0.5, label='PICMUS Q Component')
    ax1.plot(gt_time_axis_plot * 1e6, gt_envelope_plot, 'k-', linewidth=1.5, label='PICMUS Envelope')

    # Get the time value where the trimming starts and add the vertical line
    trim_time_us = (gt_time_axis_plot[-num_points_to_trim]) * 1e6
    ax1.axvline(x=trim_time_us, color='m', linestyle='--', linewidth=2, label=f'Error Trim Point')
    
    ax1.set_title(f'PICMUS Ground Truth I/Q Data (Channel {channel_for_plot}, Angle {angle_for_plot}, fs={fs_picmus_iq/1e6:.2f} MHz)')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.grid(True)

    # --- Bottom Subplot (ax2): Your Generated Data ---
    ax2.plot(my_time_axis_plot * 1e6, my_iq_aline_plot.real, 'b-', linewidth=0.5, label='Generated I Component')
    ax2.plot(my_time_axis_plot * 1e6, my_iq_aline_plot.imag, 'r-', linewidth=0.5, label='Generated Q Component')
    ax2.plot(my_time_axis_plot * 1e6, my_envelope_plot, 'k-', linewidth=1.5, label='Generated Envelope')

    # Get the time value where the trimming starts and add the vertical line
    trim_time_us = (gt_time_axis_plot[-num_points_to_trim]) * 1e6
    ax2.axvline(x=trim_time_us, color='m', linestyle='--', linewidth=2, label=f'Error Trim Point')
    
    ax2.set_title(f'Generated I/Q Data by Virtual AFE (Channel {channel_for_plot}, fs={adc_rate/1e6:.2f} MHz)')
    ax2.set_xlabel('Time (µs)')
    ax2.set_ylabel('Amplitude')
    ax2.legend()
    ax2.grid(True)

    # --- Overall figure title and saving ---
    fig.suptitle('Comparison of Generated vs. Ground Truth I/Q Data (Fixed Test Case)', fontsize=16)
    # Adjust layout to prevent titles from overlapping
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Define the output path and create the directory if it doesn't exist
    plots_dir = Path(__file__).resolve().parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    file_name = "virtual_afe_gt_comparison.png"
    output_path = plots_dir / file_name

    # Save the figure to the specified path
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"\nSUCCESS: Comparison plot saved to {output_path}")
    
    # Close the plot to prevent it from showing interactively
    plt.close()