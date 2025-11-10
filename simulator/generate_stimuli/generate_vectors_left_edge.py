"""
Generate simulation vectors for find_bw_left_edge.sv module
"""
import numpy as np
from scipy import signal
from pathlib import Path
import sys

# Add parent directory to path to import golden model
SIMULATOR_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SIMULATOR_ROOT / "src"))

from golden_model_floating_point import (
    streaming_dft_processor,
    convert_to_sorted_db_power,
    find_left_edge_points
)

# Import data loading functions
try:
    from afe_interface_rf import load_picmus_rf_data
    from virtual_afe import run_virtual_afe_processing
    PICMUS_AVAILABLE = True
except ImportError:
    print("Warning: PICMUS data loading modules not available. Skipping real data tests.")
    PICMUS_AVAILABLE = False

def float_to_fixed_point(value, int_bits, frac_bits, signed=True):
    """
    Convert floating point to fixed point representation.
    
    Args:
        value: floating point value
        int_bits: number of integer bits
        frac_bits: number of fractional bits
        signed: whether the number is signed
    
    Returns:
        Integer representation of fixed point number
    """
    total_bits = int_bits + frac_bits
    scale = 2 ** frac_bits
    fixed_val = int(round(value * scale))
    
    if signed:
        max_val = 2 ** (total_bits - 1) - 1
        min_val = -(2 ** (total_bits - 1))
    else:
        max_val = 2 ** total_bits - 1
        min_val = 0
    
    # Saturate
    fixed_val = max(min_val, min(max_val, fixed_val))
    
    # Convert to unsigned representation for output
    if signed and fixed_val < 0:
        fixed_val = (1 << total_bits) + fixed_val
    
    return fixed_val

def fixed_point_to_float(fixed_val, total_bits, frac_bits, signed=True):
    """
    Convert a fixed-point integer (potentially in unsigned two's complement format)
    back to a floating point number.
    """
    if not signed:
        return float(fixed_val) / (2**frac_bits)

    # Determine the sign bit
    sign_bit_mask = 1 << (total_bits - 1)
    
    # If the sign bit is set, it's a negative number
    if (fixed_val & sign_bit_mask):
        # Convert from two's complement to negative integer
        signed_int = fixed_val - (1 << total_bits)
    else:
        signed_int = fixed_val
        
    return float(signed_int) / (2**frac_bits)

def generate_test_case(test_name, iq_data, fs, freq_bins, threshold_db, 
                       accum_width, freq_bin_width, num_accums):
    """
    Generate a single test case for the left edge finder.
    
    Returns:
        Dictionary containing inputs and expected outputs
    """
    print(f"\n=== Generating test case: {test_name} ===")
    
    # Run golden model
    dft_bins = streaming_dft_processor(iq_data, fs, freq_bins, window='hann')
    freqs_sorted, power_db_norm_sorted = convert_to_sorted_db_power(dft_bins)
    
    # Find left edge points
    f1_golden, f2_golden, L1_golden, L2_golden = find_left_edge_points(
        freqs_sorted, power_db_norm_sorted, threshold_db=threshold_db
    )
    
    print(f"Golden model results:")
    print(f"  Number of frequency bins: {len(freqs_sorted)}")
    print(f"  Frequency range: [{freqs_sorted[0]/1e6:.3f}, {freqs_sorted[-1]/1e6:.3f}] MHz")
    if f1_golden is not None:
        print(f"  f1 = {f1_golden/1e6:.6f} MHz, f2 = {f2_golden/1e6:.6f} MHz")
        print(f"  L1 = {L1_golden:.3f} dB, L2 = {L2_golden:.3f} dB")
        print(f"  Bandwidth edge at: {f1_golden/1e6:.6f} MHz (between f1 and f2)")
    else:
        print(f"  No crossing found!")
    
    # Convert to fixed point for hardware
    # Frequency bins: represent in MHz as signed fixed-point Q(freq_bin_width-6).6 format
    # This gives us range of +/- 8 MHz (for 9-bit width) with 6 fractional bits (0.015625 MHz resolution)
    freq_frac_bits = 12
    freq_int_bits = freq_bin_width - freq_frac_bits
    
    # For frequencies, we need two's complement representation
    freq_bins_hw = []
    for f in freqs_sorted:
        f_mhz = f / 1e6
        fixed_val = float_to_fixed_point(f_mhz, freq_int_bits, freq_frac_bits, signed=True)
        freq_bins_hw.append(fixed_val)
    
    # Power values: already in dB, use Q(accum_width-8).8 format (8 fractional bits)
    power_frac_bits = 8
    power_int_bits = accum_width - power_frac_bits
    power_db_hw = [float_to_fixed_point(p, power_int_bits, power_frac_bits, signed=True) 
                   for p in power_db_norm_sorted]
    
    # Expected outputs - frequencies in MHz (already in two's complement from float_to_fixed_point)
    f1_hw = float_to_fixed_point(f1_golden / 1e6, freq_int_bits, freq_frac_bits, signed=True) if f1_golden is not None else 0
    f2_hw = float_to_fixed_point(f2_golden / 1e6, freq_int_bits, freq_frac_bits, signed=True) if f2_golden is not None else 0
    L1_hw = float_to_fixed_point(L1_golden, power_int_bits, power_frac_bits, signed=True) if L1_golden is not None else 0
    L2_hw = float_to_fixed_point(L2_golden, power_int_bits, power_frac_bits, signed=True) if L2_golden is not None else 0
    
    valid = 1 if all(v is not None for v in [f1_golden, f2_golden, L1_golden, L2_golden]) else 0
    
    return {
        'test_name': test_name,
        'num_accums': len(freqs_sorted),
        'freq_bins': freq_bins_hw,
        'power_db': power_db_hw,
        'threshold_db': threshold_db,
        'expected_f1': f1_hw,
        'expected_f2': f2_hw,
        'expected_L1': L1_hw,
        'expected_L2': L2_hw,
        'expected_valid': valid,
        # 'freq_resolution': freq_resolution,
        'golden_f1': f1_golden,
        'golden_f2': f2_golden,
        'golden_L1': L1_golden,
        'golden_L2': L2_golden
    }

def generate_synth_test_case(test_name, iq_data, db_data, fs, freq_bins, threshold_db, 
                       accum_width, freq_bin_width, num_accums):
    """
    Generate a single test case for the left edge finder.
    
    Returns:
        Dictionary containing inputs and expected outputs
    """
    print(f"\n=== Generating test case: {test_name} ===")
    
    # Run golden model
    dft_bins = streaming_dft_processor(iq_data, fs, freq_bins, window='hann')
    freqs_sorted, power_db_norm_sorted = convert_to_sorted_db_power(dft_bins)
    
    # Find left edge points
    f1_golden, f2_golden, L1_golden, L2_golden = find_left_edge_points(
        freqs_sorted, db_data, threshold_db=threshold_db
    )
    
    print(f"Golden model results:")
    print(f"  Number of frequency bins: {len(freqs_sorted)}")
    print(f"  Frequency range: [{freqs_sorted[0]/1e6:.3f}, {freqs_sorted[-1]/1e6:.3f}] MHz")
    if f1_golden is not None:
        print(f"  f1 = {f1_golden/1e6:.6f} MHz, f2 = {f2_golden/1e6:.6f} MHz")
        print(f"  L1 = {L1_golden:.3f} dB, L2 = {L2_golden:.3f} dB")
        print(f"  Bandwidth edge at: {f1_golden/1e6:.6f} MHz (between f1 and f2)")
    else:
        print(f"  No crossing found!")
    
    # Convert to fixed point for hardware
    # Frequency bins: represent in MHz as signed fixed-point Q(freq_bin_width-6).6 format
    # This gives us range of +/- 8 MHz (for 9-bit width) with 6 fractional bits (0.015625 MHz resolution)
    freq_frac_bits = 12
    freq_int_bits = freq_bin_width - freq_frac_bits
    
    # For frequencies, we need two's complement representation
    freq_bins_hw = []
    for f in freqs_sorted:
        f_mhz = f / 1e6
        fixed_val = float_to_fixed_point(f_mhz, freq_int_bits, freq_frac_bits, signed=True)
        freq_bins_hw.append(fixed_val)
    
    # Power values: already in dB, use Q(accum_width-8).8 format (8 fractional bits)
    power_frac_bits = 8
    power_int_bits = accum_width - power_frac_bits
    power_db_hw = [float_to_fixed_point(p, power_int_bits, power_frac_bits, signed=True) 
                   for p in db_data]
    
    # Expected outputs - frequencies in MHz (already in two's complement from float_to_fixed_point)
    f1_hw = float_to_fixed_point(f1_golden / 1e6, freq_int_bits, freq_frac_bits, signed=True) if f1_golden is not None else 0
    f2_hw = float_to_fixed_point(f2_golden / 1e6, freq_int_bits, freq_frac_bits, signed=True) if f2_golden is not None else 0
    L1_hw = float_to_fixed_point(L1_golden, power_int_bits, power_frac_bits, signed=True) if L1_golden is not None else 0
    L2_hw = float_to_fixed_point(L2_golden, power_int_bits, power_frac_bits, signed=True) if L2_golden is not None else 0
    
    valid = 1 if all(v is not None for v in [f1_golden, f2_golden, L1_golden, L2_golden]) else 0
    
    return {
        'test_name': test_name,
        'num_accums': len(freqs_sorted),
        'freq_bins': freq_bins_hw,
        'power_db': power_db_hw,
        'threshold_db': threshold_db,
        'expected_f1': f1_hw,
        'expected_f2': f2_hw,
        'expected_L1': L1_hw,
        'expected_L2': L2_hw,
        'expected_valid': valid,
        # 'freq_resolution': freq_resolution,
        'golden_f1': f1_golden,
        'golden_f2': f2_golden,
        'golden_L1': L1_golden,
        'golden_L2': L2_golden
    }

def write_vector_file(test_cases, output_path, accum_width, freq_bin_width):
    """
    Write test vectors to file in a format readable by SystemVerilog testbench.
    """
    with open(output_path, 'w') as f:
        # Write header with metadata
        f.write("# Simulation vectors for find_bw_left_edge.sv\n")
        f.write(f"# ACCUM_WIDTH = {accum_width}\n")
        f.write(f"# FREQ_BIN_WIDTH = {freq_bin_width}\n")
        f.write("#\n")
        f.write("# Frequency representation: Q{}.{} fixed-point in MHz\n".format(
            freq_bin_width - 6, 6))
        f.write("# Power representation: Q{}.{} fixed-point in dB\n".format(
            accum_width - 8, 8))
        f.write("#\n")
        f.write("# Format per test case:\n")
        f.write("# TEST_NAME <n>\n")
        f.write("# NUM_ACCUMS <n>\n")
        f.write("# THRESHOLD_DB <value>\n")
        f.write("# FREQ_BINS <n values in hex> (frequencies in MHz)\n")
        f.write("# POWER_DB <n values in hex> (power in dB normalized)\n")
        f.write("# EXPECTED f1 f2 L1 L2 valid (all in hex)\n")
        f.write("# GOLDEN f1 f2 L1 L2 (floating point in MHz and dB for reference)\n")
        f.write("#\n\n")
        
        for tc in test_cases:
            f.write(f"{tc['test_name']}\n")         # TEST NAME
            f.write(f"{tc['num_accums']}\n")        # NUM ACCUMS
            f.write(f"{tc['threshold_db']}\n")      # THRESHOLD DB
            
            # Write frequency bins (in MHz as fixed-point)
            # f.write("FREQ_BINS")                  # FREQ BINS
            for fb in tc['freq_bins']:
                f.write(f"{fb:03x} ")
            f.write("\n")
            
            # Write power values
            # f.write("POWER_DB")                   # POWER DB
            for p in tc['power_db']:
                f.write(f"{p:04x} ")
            f.write("\n")
            
            # Write expected outputs (frequencies in MHz, powers in dB)
            f.write(f"{tc['expected_f1']:03x} {tc['expected_f2']:03x} {tc['expected_L1']:04x} "
                   f"{tc['expected_L2']:04x} {tc['expected_valid']}\n") # EXPECTED
            
            # Write golden reference (for debugging) - convert Hz to MHz
            if tc['golden_f1'] is not None:
                f.write(f"GOLDEN {tc['golden_f1']/1e6:.6f} {tc['golden_f2']/1e6:.6f} "
                       f"{tc['golden_L1']:.6f} {tc['golden_L2']:.6f}\n") # GOLDEN
            else:
                f.write(f"GOLDEN nan nan nan nan\n")
            
            f.write("\n")  # Blank line between test cases

def main():
    """
    Main function to generate all test vectors.
    """
    print("=== Generating Simulation Vectors for find_bw_left_edge.sv ===\n")
    
    # Hardware parameters
    ACCUM_WIDTH = 18
    FREQ_BIN_WIDTH = 16
    NUM_ACCUMS = 24  # max, actual varies per test
    
    test_cases = []
    
    # ===== Test Cases from PICMUS Data =====
    if PICMUS_AVAILABLE:
        print("\n========== PICMUS Real Data Test Cases ==========")
        
        try:
            # Load PICMUS data
            rf_path = SIMULATOR_ROOT.parent / "simulator/datasets/experiments/contrast_speckle/contrast_speckle_expe_dataset_rf.hdf5"
            iq_path = SIMULATOR_ROOT.parent / "simulator/datasets/experiments/contrast_speckle/contrast_speckle_expe_dataset_iq.hdf5"
            scan_path = SIMULATOR_ROOT.parent / "simulator/datasets/experiments/contrast_speckle/contrast_speckle_expe_scan.hdf5"
            
            adc_rate = 125e6
            baseline_decimation = 4
            
            rf_data, angles, _, _, fs_picmus, mod_freq, _, _, _ = load_picmus_rf_data(rf_path, iq_path, scan_path)
            
            # Get baseline I/Q data
            center_angle_index = np.argmin(np.abs(angles))
            baseline_iq_data, _, fs_baseline = run_virtual_afe_processing(
                rf_data=rf_data,
                angle_index=center_angle_index,
                fs_picmus=fs_picmus,
                modulation_frequency=mod_freq,
                decimation_factor=baseline_decimation,
                adc_sample_rate=adc_rate
            )
            
            # STFT parameters
            nperseg = 256
            hop = nperseg // 2
            
            # Define frequency bins of interest (matching your example)
            delta_f = 0.25e6
            half_bw_est = mod_freq / 2
            s_coarse = np.linspace(-mod_freq, mod_freq, 8)
            s_fine_left = np.linspace(-half_bw_est - delta_f, -half_bw_est + delta_f, 8)
            s_fine_right = np.linspace(half_bw_est - delta_f, half_bw_est + delta_f, 8)
            S_bins = np.unique(np.concatenate([s_coarse, s_fine_left, s_fine_right]))
            
            threshold_db = -30
            
            # Test different channels and windows
            test_configs = [
                ("picmus_ch64_win29", 64, 29),
                ("picmus_ch64_win15", 64, 15),
                ("picmus_ch64_win27", 64, 27),
                ("picmus_ch64_win31", 64, 31),
                ("picmus_ch32_win29", 32, 29),
                ("picmus_ch96_win29", 96, 29),
            ]
            
            for test_name, channel, window_num in test_configs:
                print(f"\n--- Processing {test_name}: Channel {channel}, Window {window_num} ---")
                
                start_sample = window_num * hop
                end_sample = start_sample + nperseg
                
                # Extract window data
                time_window_data = baseline_iq_data[start_sample:end_sample, channel]
                
                tc = generate_test_case(
                    test_name,
                    time_window_data, # change this
                    fs_baseline,
                    S_bins,
                    threshold_db,
                    ACCUM_WIDTH,
                    FREQ_BIN_WIDTH,
                    NUM_ACCUMS
                )
                test_cases.append(tc)

            # ===== Synthethic Test Case to check early exiting properly disabled ===== 
            synth_test_config = [
                ("synth_test_1", 64, 29),
            ]

            for test_name, channel, window_num in synth_test_config:
                print(f"\n=== Generating SYNTHETIC test with multiple threshold crossings ===")

                start_sample = window_num * hop
                end_sample = start_sample + nperseg

                time_window_data = baseline_iq_data[start_sample:end_sample, channel]

                db_data = np.zeros(len(S_bins))
                db_data[0] = -40
                db_data[3] = -40
                db_data[4] = -40
                db_data[5] = -40
                db_data[-1] = -40
                db_data[-4] = -40
                db_data[-5] = -40
                db_data[-6] = -40


                # print(f"\n Synthetic dB data with multiple crossings: {db_data}")

                tc = generate_synth_test_case(
                    test_name,
                    time_window_data,
                    db_data,
                    fs_baseline,
                    S_bins,
                    threshold_db,
                    ACCUM_WIDTH,
                    FREQ_BIN_WIDTH,
                    NUM_ACCUMS
                )
                test_cases.append(tc)

                
        except Exception as e:
            print(f"Error loading PICMUS data: {e}")
            print("Skipping PICMUS test cases.")
    
    # # ===== Synthetic Test Cases (for sanity checking) =====
    # print("\n========== Synthetic Test Cases ==========")  
    # # Test parameters for synthetic signals
    # fs_synth = 31.25e6
    # nperseg_synth = 256
    # mod_freq_synth = 7.8125e6
    # threshold_db_synth = -30
    
    # # Define sparse frequency bins
    # delta_f = 0.25e6
    # half_bw_est = mod_freq_synth / 2
    # s_coarse = np.linspace(-mod_freq_synth, mod_freq_synth, 8)
    # s_fine_left = np.linspace(-half_bw_est - delta_f, -half_bw_est + delta_f, 8)
    # s_fine_right = np.linspace(half_bw_est - delta_f, half_bw_est + delta_f, 8)
    # S_bins_synth = np.unique(np.concatenate([s_coarse, s_fine_left, s_fine_right]))
    
    # t = np.arange(nperseg_synth) / fs_synth
    
    # # Test Case 1: Centered Gaussian pulse
    # print("\n--- Test Case: Centered Gaussian pulse ---")
    # signal_1 = np.exp(-((t - nperseg_synth/(2*fs_synth))**2) / (2*(10e-6)**2)) * np.exp(1j * 2 * np.pi * 0 * t)
    
    # tc1 = generate_test_case(
    #     "synth_centered_gaussian",
    #     signal_1,
    #     fs_synth,
    #     S_bins_synth,
    #     threshold_db_synth,
    #     ACCUM_WIDTH,
    #     FREQ_BIN_WIDTH,
    #     NUM_ACCUMS
    # )
    # test_cases.append(tc1)
    
    # # Test Case 2: Wide bandwidth chirp
    # print("\n--- Test Case: Wide bandwidth chirp ---")
    # chirp_bw = 5e6
    # signal_2 = signal.chirp(t, f0=-chirp_bw/2, f1=chirp_bw/2, t1=t[-1], method='linear')
    # signal_2 = signal_2 + 1j * signal.chirp(t, f0=-chirp_bw/2, f1=chirp_bw/2, t1=t[-1], method='linear', phi=90)
    
    # tc2 = generate_test_case(
    #     "synth_wide_bandwidth",
    #     signal_2,
    #     fs_synth,
    #     S_bins_synth,
    #     threshold_db_synth,
    #     ACCUM_WIDTH,
    #     FREQ_BIN_WIDTH,
    #     NUM_ACCUMS
    # )
    # test_cases.append(tc2)
    
    # # Test Case 3: Narrow bandwidth signal
    # print("\n--- Test Case: Narrow bandwidth signal ---")
    # signal_3 = np.exp(-((t - nperseg_synth/(2*fs_synth))**2) / (2*(50e-6)**2)) * np.exp(1j * 2 * np.pi * 0 * t)
    
    # tc3 = generate_test_case(
    #     "synth_narrow_bandwidth",
    #     signal_3,
    #     fs_synth,
    #     S_bins_synth,
    #     threshold_db_synth,
    #     ACCUM_WIDTH,
    #     FREQ_BIN_WIDTH,
    #     NUM_ACCUMS
    # )
    # test_cases.append(tc3)
    
    # Write to file
    output_dir = SIMULATOR_ROOT.parent / "rtl" / "simvectors"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "find_bw_left_edge_vectors.txt"
    
    write_vector_file(test_cases, output_path, ACCUM_WIDTH, FREQ_BIN_WIDTH)
    
    print(f"\n=== Successfully generated {len(test_cases)} test cases ===")
    print(f"Output file: {output_path}")
    print("\nTest cases generated:")
    for tc in test_cases:
        status = "FOUND" if tc['expected_valid'] else "NOT FOUND"
        print(f"  - {tc['test_name']}: {tc['num_accums']} bins, crossing {status}")

if __name__ == "__main__":
    main()