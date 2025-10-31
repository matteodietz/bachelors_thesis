import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
import numpy as np
from scipy import signal

# Import our fixed-point conversion helpers
from fixed_point_models import to_fixed_point, to_float

# --- Test Parameters ---
# These MUST match the parameters in your SystemVerilog module
ACCUM_WIDTH = 16
FREQ_BIN_WIDTH = 9
NUM_ACCUMS = 16
# Let's assume a Q-format for the dB values, e.g., Q8.8 (8 integer, 8 fractional bits)
FRAC_BITS_ACCUM = 8
# The threshold is a positive integer in the DUT, representing the negative dB value
THRESHOLD_DB_INT = 30
THRESHOLD_DB_FLOAT = -30.0

@cocotb.test()
async def test_left_edge_finder(dut):
    """Test the find_bw_left_edge module."""
    
    # --- 1. Test Setup and "Golden Reference" Calculation ---
    dut._log.info("--- Starting Test ---")

    # Generate a realistic, sorted test spectrum
    freqs_hz = np.linspace(-5e6, 5e6, NUM_ACCUMS) # Example: -5MHz to +5MHz
    # Create a Gaussian-like power spectrum centered at -1 MHz
    center_freq = -1.0e6
    bandwidth = 2.0e6
    sigma = bandwidth / 6.06 # Approx sigma for -20dB BW
    
    power_db_norm = -40 * ((freqs_hz - center_freq)**2) / (bandwidth**2)
    # Ensure the peak is exactly at 0 dB
    power_db_norm -= np.min(power_db_norm)
    power_db_norm = np.clip(power_db_norm, -100, 0)
    
    # --- Find the "Golden" crossing point with Python ---
    golden_f1, golden_f2 = -1, -1
    golden_L1, golden_L2 = -1, -1
    
    start_search_idx = np.argmin(np.abs(freqs_hz)) # Start from center (0 Hz)
    for i in range(start_search_idx, 0, -1):
        if power_db_norm[i-1] < THRESHOLD_DB_FLOAT <= power_db_norm[i]:
            golden_L1 = power_db_norm[i-1]
            golden_L2 = power_db_norm[i]
            golden_f1 = freqs_hz[i-1]
            golden_f2 = freqs_hz[i]
            break
            
    assert golden_f1 != -1, "Golden reference calculation failed to find a crossing."
    
    dut._log.info("--- Golden Reference Calculated ---")
    dut._log.info(f"Expected f1: {golden_f1/1e6:.3f} MHz, f2: {golden_f2/1e6:.3f} MHz")
    dut._log.info(f"Expected L1: {golden_L1:.2f} dB, L2: {golden_L2:.2f} dB")
    
    # --- 2. Prepare DUT Inputs (Convert to Fixed-Point) ---
    
    # Convert dB values to our Q8.8 fixed-point format
    accum_fixed = [to_fixed_point(val, ACCUM_WIDTH, FRAC_BITS_ACCUM) for val in power_db_norm]
    
    # For frequency bins, we can just use the integer indices
    freq_bins_fixed = np.arange(NUM_ACCUMS)

    # --- 3. Drive the DUT ---
    
    # Start a clock
    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())
    
    # Reset the DUT
    dut.rst_ni.value = 0
    dut.start_i.value = 0
    await ClockCycles(dut.clk_i, 5)
    dut.rst_ni.value = 1
    await RisingEdge(dut.clk_i)

    # Load the inputs
    for i in range(NUM_ACCUMS):
        dut.accumulator_val_i[i].value = accum_fixed[i]
        dut.freq_bin_i[i].value = freq_bins_fixed[i]

    # Start the process
    dut._log.info("Starting DUT processing...")
    dut.start_i.value = 1
    await RisingEdge(dut.clk_i)
    dut.start_i.value = 0
    
    # Wait for the busy signal to go low, which indicates it's done
    # Add a timeout to prevent the test from running forever if there's a bug
    await cocotb.triggers.with_timeout(RisingEdge(dut.valid_o), timeout_time=500, timeout_unit="ns")
    
    dut._log.info("DUT finished processing (valid_o is high).")
    
    # --- 4. Check the Results ---
    
    # Read the outputs from the DUT
    dut_f1_idx = dut.f1_o.value.integer
    dut_f2_idx = dut.f2_o.value.integer
    dut_L1_fixed = dut.L1_o.value.integer
    dut_L2_fixed = dut.L2_o.value.integer
    
    # Convert fixed-point outputs back to float for comparison
    dut_L1_float = to_float(dut_L1_fixed, ACCUM_WIDTH, FRAC_BITS_ACCUM)
    dut_L2_float = to_float(dut_L2_fixed, ACCUM_WIDTH, FRAC_BITS_ACCUM)

    # Get the frequencies corresponding to the DUT's chosen indices
    dut_f1_hz = freqs_hz[dut_f1_idx]
    dut_f2_hz = freqs_hz[dut_f2_idx]

    dut._log.info("--- DUT Results ---")
    dut._log.info(f"DUT f1: {dut_f1_hz/1e6:.3f} MHz, f2: {dut_f2_hz/1e6:.3f} MHz")
    dut._log.info(f"DUT L1: {dut_L1_float:.2f} dB, L2: {dut_L2_float:.2f} dB")
    
    # --- 5. Assert and Verify ---
    
    # Check if the DUT found the same frequency bins
    assert dut_f1_hz == golden_f1, f"f1 mismatch: DUT={dut_f1_hz}, Golden={golden_f1}"
    assert dut_f2_hz == golden_f2, f"f2 mismatch: DUT={dut_f2_hz}, Golden={golden_f2}"
    
    # Check if the power levels are very close (allowing for small fixed-point errors)
    np.testing.assert_allclose(dut_L1_float, golden_L1, atol=1/2**FRAC_BITS_ACCUM, rtol=0)
    np.testing.assert_allclose(dut_L2_float, golden_L2, atol=1/2**FRAC_BITS_ACCUM, rtol=0)
    
    dut._log.info("--- TEST PASSED ---")