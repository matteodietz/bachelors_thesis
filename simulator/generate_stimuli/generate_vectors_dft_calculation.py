"""
Generate simulation vectors for dft_accumulation.sv module
"""
import numpy as np
from scipy import signal
from pathlib import Path
import sys

# Add parent directory to path to import golden model
SIMULATOR_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SIMULATOR_ROOT / "src"))

from golden_model_floating_point import streaming_dft_processor
from fixed_float_conversions import float_to_fixed_point, fixed_point_to_float

# Import data loading functions
try:
    from afe_interface_rf import load_picmus_rf_data
    from virtual_afe import run_virtual_afe_processing
    PICMUS_AVAILABLE = True
except ImportError:
    print("Warning: PICMUS data loading modules not available. Skipping real data tests.")
    PICMUS_AVAILABLE = False

def generate_test_case(test_name, iq_data, fs, freq_bins, window_type,
                       iq_width, window_width, accum_width, osc_width, num_bins):
    """
    Generate a single test case for the DFT accumulator.
    
    Returns:
        Dictionary containing all inputs and expected outputs
    """
    print(f"\n=== Generating test case: {test_name} ===")
    
    N = len(iq_data)
    K = len(freq_bins)
    
    print(f"  Sample length: {N}")
    print(f"  Number of bins: {K}")
    print(f"  Frequency bins: {freq_bins/1e6} MHz")
    
    # Run golden model to get expected outputs
    dft_bins = streaming_dft_processor(iq_data, fs, freq_bins, window=window_type)
    
    # Extract results
    freqs = np.array(list(dft_bins.keys()))
    accumulators = np.array(list(dft_bins.values()))
    
    # Sort by frequency for consistent output
    sort_indices = np.argsort(freqs)
    freqs_sorted = freqs[sort_indices]
    accums_sorted = accumulators[sort_indices]
    
    print(f"  Golden accumulator magnitudes: {np.abs(accums_sorted)}")
    
    # Generate window coefficients
    window_coeffs = signal.windows.get_window(window_type, N)
    
    # Generate complex oscillator values W[n,k] = exp(-j*2*pi*k*n/fs)
    # W starts at 1+0j and is multiplied by E each sample
    E = np.exp(-1j * 2 * np.pi * freq_bins / fs)
    
    # Pre-compute all W values for all samples and all bins
    W_values = np.zeros((N, K), dtype=np.complex128)
    W_values[0, :] = 1.0 + 0j  # Initial value
    for n in range(1, N):
        W_values[n, :] = W_values[n-1, :] * E
    
    # Convert to fixed point
    # I/Q samples: Use Q(iq_width-8).8 format (8 fractional bits)
    iq_frac_bits = 16
    iq_int_bits = iq_width - iq_frac_bits
    
    i_samples_hw = [float_to_fixed_point(np.real(s), iq_int_bits, iq_frac_bits, signed=True) 
                    for s in iq_data]
    q_samples_hw = [float_to_fixed_point(np.imag(s), iq_int_bits, iq_frac_bits, signed=True) 
                    for s in iq_data]
    
    # Window coefficients: Use Q(window_width-16).16 format (16 fractional bits)
    # Window values are between 0 and 1
    window_frac_bits = 16
    window_int_bits = window_width - window_frac_bits
    
    window_coeffs_hw = [float_to_fixed_point(w, window_int_bits, window_frac_bits, signed=True) 
                       for w in window_coeffs]
    
    # Oscillator values W: Use Q(osc_width-16).16 format (16 fractional bits)
    # W values are complex with magnitude ~1
    osc_frac_bits = 16
    osc_int_bits = osc_width - osc_frac_bits
    
    W_real_hw = np.zeros((N, K), dtype=int)
    W_imag_hw = np.zeros((N, K), dtype=int)
    
    for n in range(N):
        for k in range(K):
            W_real_hw[n, k] = float_to_fixed_point(np.real(W_values[n, k]), 
                                                    osc_int_bits, osc_frac_bits, signed=True)
            W_imag_hw[n, k] = float_to_fixed_point(np.imag(W_values[n, k]), 
                                                    osc_int_bits, osc_frac_bits, signed=True)
    
    # Expected accumulator outputs: Use Q(accum_width-8).8 format (8 fractional bits)
    # Note: The accumulators grow large, so we need more integer bits
    accum_frac_bits = 36
    accum_int_bits = accum_width - accum_frac_bits
    
    # Scale expected outputs to match hardware scaling
    # The hardware shifts the products: SHIFT_AMOUNT = IQ_WIDTH + WINDOW_WIDTH + OSC_WIDTH + 2 - ACCUM_WIDTH
    shift_amount = iq_width + window_width + osc_width + 2 - accum_width
    
    # Calculate scaling factor for expected values
    # Products are scaled by: 2^(iq_frac + window_frac + osc_frac)
    # Then shifted right by shift_amount
    total_frac_bits = iq_frac_bits + window_frac_bits + osc_frac_bits
    effective_scale = 2 ** (total_frac_bits - shift_amount - accum_frac_bits)
    
    A_real_hw = [float_to_fixed_point(np.real(a) / effective_scale, 
                                       accum_int_bits, accum_frac_bits, signed=True) 
                 for a in accums_sorted]
    A_imag_hw = [float_to_fixed_point(np.imag(a) / effective_scale, 
                                       accum_int_bits, accum_frac_bits, signed=True) 
                 for a in accums_sorted]
    
    return {
        'test_name': test_name,
        'num_samples': N,
        'num_bins': K,
        'fs': fs,
        'freq_bins': freqs_sorted,
        'i_samples': i_samples_hw,
        'q_samples': q_samples_hw,
        'window_coeffs': window_coeffs_hw,
        'W_real': W_real_hw,
        'W_imag': W_imag_hw,
        'expected_A_real': A_real_hw,
        'expected_A_imag': A_imag_hw,
        'golden_A_real': [np.real(a) for a in accums_sorted],
        'golden_A_imag': [np.imag(a) for a in accums_sorted],
        'golden_A_mag': [np.abs(a) for a in accums_sorted]
    }

def write_vector_file(test_cases, output_path, iq_width, window_width, 
                     accum_width, osc_width):
    """
    Write test vectors to file in a format readable by SystemVerilog testbench.
    """
    with open(output_path, 'w') as f:
        # Write header
        f.write("# Simulation vectors for dft_accumulation.sv\n")
        f.write(f"# IQ_WIDTH = {iq_width}\n")
        f.write(f"# WINDOW_WIDTH = {window_width}\n")
        f.write(f"# ACCUM_WIDTH = {accum_width}\n")
        f.write(f"# OSC_WIDTH = {osc_width}\n")
        f.write("#\n")
        f.write("# Fixed-point formats:\n")
        f.write(f"# I/Q samples: Q{iq_width-8}.8\n")
        f.write(f"# Window coeffs: Q{window_width-16}.16\n")
        f.write(f"# Oscillator W: Q{osc_width-16}.16\n")
        f.write(f"# Accumulators: Q{accum_width-8}.8\n")
        f.write("#\n")
        f.write("# Format per test case:\n")
        f.write("# <test_name>\n")
        f.write("# <num_samples> <num_bins> <fs>\n")
        f.write("# FREQ_BINS <freq0_MHz> <freq1_MHz> ... (reference)\n")
        f.write("# SAMPLES (per line: I Q window_coeff W_real[0..K-1] W_imag[0..K-1])\n")
        f.write("# EXPECTED A_real[0..K-1] A_imag[0..K-1] (hex)\n")
        f.write("# GOLDEN A_real[0..K-1] A_imag[0..K-1] |A|[0..K-1] (float reference)\n")
        f.write("#\n\n")
        
        for tc in test_cases:
            f.write(f"{tc['test_name']}\n")
            f.write(f"{tc['num_samples']} {tc['num_bins']} {tc['fs']:.6e}\n")
            
            # Write frequency bins (reference)
            f.write("FREQ_BINS ")
            for freq in tc['freq_bins']:
                f.write(f"{freq/1e6:.6f} ")
            f.write("\n")
            
            # Write sample data (one line per sample)
            f.write("SAMPLES\n")
            for n in range(tc['num_samples']):
                # I, Q, window_coeff
                f.write(f"{tc['i_samples'][n]:05x} ")
                f.write(f"{tc['q_samples'][n]:05x} ")
                f.write(f"{tc['window_coeffs'][n]:05x} ")
                
                # W_real[0..K-1]
                for k in range(tc['num_bins']):
                    f.write(f"{tc['W_real'][n, k]:05x} ")
                
                # W_imag[0..K-1]
                for k in range(tc['num_bins']):
                    f.write(f"{tc['W_imag'][n, k]:05x} ")
                
                f.write("\n")
            
            # Write expected outputs
            f.write("EXPECTED ")
            for k in range(tc['num_bins']):
                f.write(f"{tc['expected_A_real'][k]:05x} ")
            for k in range(tc['num_bins']):
                f.write(f"{tc['expected_A_imag'][k]:05x} ")
            f.write("\n")
            
            # Write golden reference
            f.write("GOLDEN ")
            for k in range(tc['num_bins']):
                f.write(f"{tc['golden_A_real'][k]:.6e} ")
            for k in range(tc['num_bins']):
                f.write(f"{tc['golden_A_imag'][k]:.6e} ")
            for k in range(tc['num_bins']):
                f.write(f"{tc['golden_A_mag'][k]:.6e} ")
            f.write("\n")
            
            f.write("\n")  # Blank line between test cases

def main():
    """
    Main function to generate all test vectors.
    """
    print("=== Generating Simulation Vectors for dft_accumulation.sv ===\n")
    
    # Hardware parameters
    IQ_WIDTH = 24
    WINDOW_WIDTH = 24
    ACCUM_WIDTH = 48  # Needs to be large to avoid overflow
    OSC_WIDTH = 18
    NUM_BINS = 24  # Maximum
    
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
            
            # Define frequency bins of interest
            delta_f = 0.25e6
            half_bw_est = mod_freq / 2
            s_coarse = np.linspace(-mod_freq, mod_freq, 8)
            s_fine_left = np.linspace(-half_bw_est - delta_f, -half_bw_est + delta_f, 8)
            s_fine_right = np.linspace(half_bw_est - delta_f, half_bw_est + delta_f, 8)
            S_bins = np.unique(np.concatenate([s_coarse, s_fine_left, s_fine_right]))
            
            # Test different windows
            test_configs = [
                ("picmus_ch64_win29", 64, 29),
                ("picmus_ch64_win15", 96, 30),
                ("picmus_ch64_win15", 32, 27),
            ]
            
            for test_name, channel, window_num in test_configs:
                print(f"\n--- Processing {test_name}: Channel {channel}, Window {window_num} ---")
                
                start_sample = window_num * hop
                end_sample = start_sample + nperseg
                
                # Extract window data
                time_window_data = baseline_iq_data[start_sample:end_sample, channel]
                
                tc = generate_test_case(
                    test_name,
                    time_window_data,
                    fs_baseline,
                    S_bins,
                    'hann',
                    IQ_WIDTH,
                    WINDOW_WIDTH,
                    ACCUM_WIDTH,
                    OSC_WIDTH,
                    NUM_BINS
                )
                test_cases.append(tc)
                
        except Exception as e:
            print(f"Error loading PICMUS data: {e}")
            import traceback
            traceback.print_exc()
            print("Skipping PICMUS test cases.")
    
    # ===== Synthetic Test Cases =====
    print("\n========== Synthetic Test Cases ==========")
    
    fs_synth = 31.25e6
    nperseg_synth = 256
    mod_freq_synth = 7.8125e6
    
    # Define sparse frequency bins
    delta_f = 0.25e6
    half_bw_est = mod_freq_synth / 2
    s_coarse = np.linspace(-mod_freq_synth, mod_freq_synth, 8)
    s_fine_left = np.linspace(-half_bw_est - delta_f, -half_bw_est + delta_f, 8)
    s_fine_right = np.linspace(half_bw_est - delta_f, half_bw_est + delta_f, 8)
    S_bins_synth = np.unique(np.concatenate([s_coarse, s_fine_left, s_fine_right]))
    
    t = np.arange(nperseg_synth) / fs_synth
    
    # Test Case 1: Simple sine wave
    print("\n--- Test Case: Simple sine wave at 2 MHz ---")
    signal_1 = np.exp(1j * 2 * np.pi * 2e6 * t)
    
    tc1 = generate_test_case(
        "synth_sine_2mhz",
        signal_1,
        fs_synth,
        S_bins_synth,
        'hann',
        IQ_WIDTH,
        WINDOW_WIDTH,
        ACCUM_WIDTH,
        OSC_WIDTH,
        NUM_BINS
    )
    test_cases.append(tc1)
    
    # Test Case 2: DC signal (0 Hz)
    print("\n--- Test Case: DC signal ---")
    signal_2 = np.ones(nperseg_synth, dtype=np.complex128)
    
    tc2 = generate_test_case(
        "synth_dc",
        signal_2,
        fs_synth,
        S_bins_synth,
        'hann',
        IQ_WIDTH,
        WINDOW_WIDTH,
        ACCUM_WIDTH,
        OSC_WIDTH,
        NUM_BINS
    )
    test_cases.append(tc2)
    
    # Write to file
    output_dir = SIMULATOR_ROOT.parent / "rtl" / "simvectors"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dft_accumulation_vectors.txt"
    
    write_vector_file(test_cases, output_path, IQ_WIDTH, WINDOW_WIDTH, 
                     ACCUM_WIDTH, OSC_WIDTH)
    
    print(f"\n=== Successfully generated {len(test_cases)} test cases ===")
    print(f"Output file: {output_path}")
    print("\nTest cases generated:")
    for tc in test_cases:
        print(f"  - {tc['test_name']}: {tc['num_samples']} samples, {tc['num_bins']} bins")

if __name__ == "__main__":
    main()