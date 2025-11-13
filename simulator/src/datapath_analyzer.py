import numpy as np
from scipy import signal
from pathlib import Path
import math

# Import your data loaders
from afe_interface_rf import load_picmus_rf_data
from virtual_afe import run_virtual_afe_processing

def analyze_datapath_ranges(iq_data, fs, freq_bins, window='hann'):
    """
    Analyzes the dynamic range of all signals within the streaming DFT processor.
    """
    N = len(iq_data)
    K = len(freq_bins)
    win = signal.windows.get_window(window, N)
    E = np.exp(-1j * 2 * np.pi * freq_bins / fs)

    # --- Storage for Maximum Observed Values ---
    max_vals = {
        'iq_input': 0.0,
        'window': 0.0,
        'oscillator': 0.0,
        'windowed_sample': 0.0,
        'mac_product': 0.0,
        'accumulator': 0.0
    }

    # Initialize states (as floats)
    A = np.zeros(K, dtype=np.complex128)
    W = np.ones(K, dtype=np.complex128)

    # --- Streaming Loop for Analysis ---
    for n in range(N):
        x_n = iq_data[n]
        h_n = win[n]
        
        # --- Track maximums at each stage ---
        max_vals['iq_input'] = max(max_vals['iq_input'], np.max(np.abs([x_n.real, x_n.imag])))
        max_vals['window'] = max(max_vals['window'], np.abs(h_n))
        max_vals['oscillator'] = max(max_vals['oscillator'], np.max(np.abs([W.real, W.imag])))
        
        # 1. Windowing: x_n * h_n
        windowed_sample = x_n * h_n
        max_vals['windowed_sample'] = max(max_vals['windowed_sample'], np.max(np.abs([windowed_sample.real, windowed_sample.imag])))
        
        # 2. Complex Multiplication: (windowed_sample) * W
        mac_product = windowed_sample * W
        max_vals['mac_product'] = max(max_vals['mac_product'], np.max(np.abs([mac_product.real, mac_product.imag])))
        
        # 3. Accumulation: A += mac_product
        A += mac_product
        max_vals['accumulator'] = max(max_vals['accumulator'], np.max(np.abs([A.real, A.imag])))
        
        # 4. Update Oscillator
        W *= E

    return max_vals

def suggest_integer_bits(max_abs_value):
    """Suggests the minimum number of integer bits needed for a signed number."""
    if max_abs_value == 0:
        return 1
    # We need `ceil(log2(val))` bits for the magnitude, plus one for the sign.
    return math.ceil(math.log2(max_abs_value + 1)) + 1


# --- Main Script ---
if __name__ == '__main__':
    # --- 1. Load Real Data ---
    # ... (Your standard PICMUS data loading code goes here)
    # ... This should result in `baseline_iq_data` and `fs_baseline`
    print("--- Running Datapath Dynamic Range Analyzer ---")

    # (Assuming this setup is the same as your previous scripts)
    try:
        SIMULATOR_ROOT = Path(__file__).resolve().parent.parent
    except NameError:
        SIMULATOR_ROOT = Path.cwd().parent
    rf_path = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_dataset_rf.hdf5"
    iq_path = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_dataset_iq.hdf5"
    scan_path = SIMULATOR_ROOT / "datasets/experiments/contrast_speckle/contrast_speckle_expe_scan.hdf5"
    adc_rate = 125e6
    baseline_decimation = 4
    rf_data, angles, _, _, fs_picmus, mod_freq, _, _, _ = load_picmus_rf_data(rf_path, iq_path, scan_path)
    center_angle_index = np.argmin(np.abs(angles))
    baseline_iq_data, _, fs_baseline = run_virtual_afe_processing(
        rf_data=rf_data, angle_index=center_angle_index, fs_picmus=fs_picmus,
        modulation_frequency=mod_freq, decimation_factor=baseline_decimation,
        adc_sample_rate=adc_rate
    )

    # --- 2. Run Analysis over several STFT windows ---
    nperseg = 256
    hop = 128
    channel_to_analyze = 64
    
    total_samples = baseline_iq_data.shape[0]
    num_windows_total = int(np.floor((total_samples - nperseg) / hop)) + 1
    
    # Define a representative set of frequency bins
    s_coarse = np.linspace(-fs_baseline/2, fs_baseline/2, 16)
    s_fine = np.linspace(-4e6, 4e6, 32)
    S_bins = np.unique(np.concatenate([s_coarse, s_fine]))
    
    # Store the overall maximums found across all windows
    overall_max_vals = {key: 0.0 for key in ['iq_input', 'window', 'oscillator', 'windowed_sample', 'mac_product', 'accumulator']}

    print(f"\nAnalyzing {num_windows_total} windows...")
    for window_num in range(num_windows_total):
        start_sample = window_num * hop
        end_sample = start_sample + nperseg
        time_window_data = baseline_iq_data[start_sample:end_sample, channel_to_analyze]
        
        # Run the analyzer on this window
        max_in_window = analyze_datapath_ranges(time_window_data, fs_baseline, S_bins)
        
        # Update the overall maximums
        for key in overall_max_vals:
            overall_max_vals[key] = max(overall_max_vals[key], max_in_window[key])

    # --- 3. Print the Results and Suggestions ---
    print("\n--- Overall Maximum Absolute Values Found Across All Windows ---")
    for key, value in overall_max_vals.items():
        int_bits = suggest_integer_bits(value)
        print(f"{key:>20}: {value:10.6f}  (Suggests >= {int_bits} integer bits)")
        
    print("\n--- Suggested Fixed-Point Parameters ---")
    
    # Your Constraints
    DSP_A_WIDTH = 27
    DSP_B_WIDTH = 18
    DSP_P_WIDTH = 48
    
    # Suggestions based on analysis and constraints
    iq_int_bits = suggest_integer_bits(overall_max_vals['iq_input'])
    IQ_WIDTH = 18 # Your AFE is 16-bit, so 18 is safe.
    IQ_FRAC_BITS = IQ_WIDTH - iq_int_bits
    print(f"IQ_WIDTH:     {IQ_WIDTH} (Q{iq_int_bits}.{IQ_FRAC_BITS})")
    
    win_int_bits = suggest_integer_bits(overall_max_vals['window']) # Should be 1 for window <= 1.0
    WINDOW_WIDTH = 18 # Fits into DSP slice
    WINDOW_FRAC_BITS = WINDOW_WIDTH - win_int_bits
    print(f"WINDOW_WIDTH: {WINDOW_WIDTH} (Q{win_int_bits}.{WINDOW_FRAC_BITS})")

    osc_int_bits = suggest_integer_bits(overall_max_vals['oscillator']) # Should be 1 for |W|<=1
    OSC_WIDTH = 27 # Fits into DSP slice
    OSC_FRAC_BITS = OSC_WIDTH - osc_int_bits
    print(f"OSC_WIDTH:    {OSC_WIDTH} (Q{osc_int_bits}.{OSC_FRAC_BITS})")

    accum_int_bits = suggest_integer_bits(overall_max_vals['accumulator'])
    ACCUM_WIDTH = 48 # Matches DSP slice accumulator
    ACCUM_FRAC_BITS = ACCUM_WIDTH - accum_int_bits
    print(f"ACCUM_WIDTH:  {ACCUM_WIDTH} (Q{accum_int_bits}.{ACCUM_FRAC_BITS})")