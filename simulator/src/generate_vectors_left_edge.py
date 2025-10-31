import numpy as np
from scipy import signal
from pathlib import Path
from fixed_point_models import to_fixed_point, to_float

# --- Fixed-Point Parameters ---
# These MUST match the parameters in your SystemVerilog module
ACCUM_WIDTH = 16
FREQ_BIN_WIDTH = 9
NUM_ACCUMS = 16
# Let's define our dB values as Q8.8 format (8 integer, 8 fractional bits)
FRAC_BITS_DB = 8
# The threshold is a positive integer in the DUT, representing the negative dB value
# We calculate the fixed-point equivalent of -30.0 dB
THRESHOLD_DB_FLOAT = -30.0
THRESHOLD_FIXED = to_fixed_point(THRESHOLD_DB_FLOAT, ACCUM_WIDTH, FRAC_BITS_DB)


def find_left_edge_golden_model(accum_vals_db_fixed, freq_bins, threshold_fixed):
    """
    A bit-accurate Python model of the find_bw_left_edge.sv state machine.
    It takes fixed-point integers as input and finds the crossing.
    
    Returns:
        tuple: (f1, f2, L1, L2) in their integer/fixed-point representation.
    """
    # Find the starting index (center of the array, biased left)
    start_search_idx = (len(accum_vals_db_fixed) // 2) - 1
    
    # Initialize outputs to a known "not found" state
    f1_out, f2_out, L1_out, L2_out = (0, 0, 0, 0)
    found = False
    
    # --- This loop mimics the Verilog state machine ---
    for i in range(start_search_idx, -1, -1): # Iterate downwards
        if i == 0: break # Cannot go past the first element
            
        L2 = accum_vals_db_fixed[i]
        L1 = accum_vals_db_fixed[i-1]
        
        # Perform the comparison using signed integer arithmetic, just like in Verilog
        L2_above_thresh = (L2 > threshold_fixed)
        L1_above_thresh = (L1 > threshold_fixed)
        
        # The crossing condition is L1 <= thresh < L2 (or in our code, L1_above=0, L2_above=1)
        # Wait, the Verilog has `power_db_norm_sorted[i-1] < threshold_db <= power_db_norm_sorted[i]`
        # This is `L1 < thresh <= L2`. Let's match that.
        if (L1 <= threshold_fixed) and (L2 > threshold_fixed):
             # This is a potential crossing. Let's adjust to match the Verilog's logic.
             # The Verilog check is `power[i-1] < thresh AND power[i] >= thresh`. Let's use that.
             pass # Let's stick to the Verilog's logic `L1 > thresh, L2 <= thresh`
             
        # The Verilog logic is: `if power_db_norm_sorted[i-1] < threshold_db <= power_db_norm_sorted[i]`
        # Your python `find_bandwidth_edges` has this. Let's use that.
        if (L1 < threshold_fixed) and (L2 >= threshold_fixed):
            f1_out = freq_bins[i-1]
            f2_out = freq_bins[i]
            L1_out = L1
            L2_out = L2
            found = True
            break # Exit the loop once the first crossing from the center is found
    
    return f1_out, f2_out, L1_out, L2_out, found

# --- Main script to generate test vectors ---
if __name__ == '__main__':
    print("--- Generating Test Vectors for find_bw_left_edge ---")

    # 1. Generate a realistic, sorted test spectrum (as floats)
    freqs_hz = np.linspace(-5e6, 5e6, NUM_ACCUMS)
    center_freq = -1.5e6 # A non-trivial center
    bandwidth = 2.5e6
    sigma = bandwidth / 6.06
    
    power_db_norm = -50 * ((freqs_hz - center_freq)**2) / (bandwidth**2)
    power_db_norm -= np.min(power_db_norm)
    power_db_norm = np.clip(power_db_norm, -100, 0)
    
    # 2. Convert to fixed-point integers (DUT inputs)
    accum_vals_db_fixed = [to_fixed_point(val, ACCUM_WIDTH, FRAC_BITS_DB) for val in power_db_norm]
    # For frequency, we can just use the integer indices as the "bins"
    freq_bins_fixed = np.arange(NUM_ACCUMS, dtype=int)
    
    # 3. Run the golden model to get the expected outputs
    f1_golden, f2_golden, L1_golden, L2_golden, found_golden = find_left_edge_golden_model(
        accum_vals_db_fixed, freq_bins_fixed, THRESHOLD_FIXED
    )
    
    if not found_golden:
        print("ERROR: Golden model did not find a crossing. Adjust test signal.")
        exit()

    print("\n--- Golden Model Results ---")
    print(f"Found crossing between bins f1={f1_golden} and f2={f2_golden}")
    print(f"Power levels L1={L1_golden} (float: {to_float(L1_golden, ACCUM_WIDTH, FRAC_BITS_DB):.2f} dB) and "
          f"L2={L2_golden} (float: {to_float(L2_golden, ACCUM_WIDTH, FRAC_BITS_DB):.2f} dB)")

    # 4. Write the simulation vector files
    sim_vectors_dir = Path(__file__).resolve().parent.parent / "sim_vectors"
    sim_vectors_dir.mkdir(parents=True, exist_ok=True)
    
    # Write input files
    with open(sim_vectors_dir / "input_accum_db.txt", "w") as f:
        for val in accum_vals_db_fixed:
            # Write as hexadecimal for $readmemh
            f.write(f"{val & (2**ACCUM_WIDTH - 1):04x}\n")
            
    with open(sim_vectors_dir / "input_freq_bins.txt", "w") as f:
        for val in freq_bins_fixed:
            f.write(f"{val:03x}\n") # Adjust width if needed

    # Write golden output files
    with open(sim_vectors_dir / "golden_outputs.txt", "w") as f:
        f.write(f"f1: {f1_golden:03x}\n")
        f.write(f"f2: {f2_golden:03x}\n")
        f.write(f"L1: {L1_golden & (2**ACCUM_WIDTH - 1):04x}\n")
        f.write(f"L2: {L2_golden & (2**ACCUM_WIDTH - 1):04x}\n")
        
    print(f"\nSUCCESS: Test vectors written to '{sim_vectors_dir}'")
    
    # Optional: Plot for visual confirmation
    plt.figure(figsize=(10, 6))
    plt.plot(freqs_hz / 1e6, power_db_norm, 'b-o', label='Input Spectrum (float)')
    plt.axhline(y=THRESHOLD_DB_FLOAT, color='grey', linestyle=':', label=f'Threshold ({THRESHOLD_DB_FLOAT} dB)')
    plt.axvline(x=freqs_hz[f1_golden]/1e6, color='r', linestyle='--', label=f'Golden f1')
    plt.axvline(x=freqs_hz[f2_golden]/1e6, color='g', linestyle='--', label=f'Golden f2')
    plt.title('Test Vector Generation')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power (dB)')
    plt.legend()
    plt.grid(True)
    plt.show()