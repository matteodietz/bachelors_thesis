import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path

def quantize(value, width, frac_bits):
    """Quantizes a float to a signed fixed-point integer, with saturation."""
    scaling_factor = 2**frac_bits
    min_val = -2**(width - 1)
    max_val = 2**(width - 1) - 1
    fixed_val = int(np.round(value * scaling_factor))
    return max(min_val, min(max_val, fixed_val))

def dequantize(value, width, frac_bits):
    """Converts a signed fixed-point integer back to a float."""
    scaling_factor = 2**frac_bits
    if value >= 2**(width - 1):
        value -= 2**width
    return float(value) / scaling_factor

def streaming_dft_processor_fixed_point(
    b, fs, freq_bins_to_calc, 
    input_width=16, input_frac_bits=15,
    window_width=16, window_frac_bits=15,
    osc_width=18, osc_frac_bits=16,
    accum_width=38, accum_frac_bits=32,
    window='hann'
):
    """
    Simulates a one-pass, streaming, sparse DFT using fully bit-accurate
    fixed-point arithmetic for all state variables.
    """
    N = len(b)
    
    # --- Create Quantized Coefficients and State Variables ---
    win_float = signal.windows.get_window(window, N)
    h_fixed = np.array([quantize(v, window_width, window_frac_bits) for v in win_float])
    
    K = len(freq_bins_to_calc)
    
    # --- Accumulators (A_k) are now large complex integers ---
    A_fixed_real = np.zeros(K, dtype=np.int64)
    A_fixed_imag = np.zeros(K, dtype=np.int64)
    
    # Oscillators (W_k) and Steps (E_k)
    W_fixed_real = np.array([quantize(1.0, osc_width, osc_frac_bits)] * K, dtype=np.int64)
    W_fixed_imag = np.zeros(K, dtype=np.int64)
    
    E_float = np.exp(-1j * 2 * np.pi * freq_bins_to_calc / fs)
    E_fixed_real = np.array([quantize(v.real, osc_width, osc_frac_bits) for v in E_float], dtype=np.int64)
    E_fixed_imag = np.array([quantize(v.imag, osc_width, osc_frac_bits) for v in E_float], dtype=np.int64)

    # --- Streaming DFT Calculation (with integer arithmetic simulation) ---
    for n in range(N):
        # Quantize the incoming sample x[n]
        x_n_real_fixed = quantize(b[n].real, input_width, input_frac_bits)
        x_n_imag_fixed = quantize(b[n].imag, input_width, input_frac_bits)
        
        h_n_fixed = h_fixed[n]
        
        for k in range(K):
            # --- This section now simulates integer MAC operations ---
            
            # 1. Windowing: x_n * h_n
            # Result width = input_width + window_width. Frac bits = input_frac + window_frac
            windowed_real = x_n_real_fixed * h_n_fixed
            windowed_imag = x_n_imag_fixed * h_n_fixed
            win_prod_frac_bits = input_frac_bits + window_frac_bits

            # 2. Complex Multiplication: (windowed_sample) * W_k
            # Result width grows again. Frac bits = win_prod_frac + osc_frac
            mult_real = windowed_real * W_fixed_real[k] - windowed_imag * W_fixed_imag[k]
            mult_imag = windowed_real * W_fixed_imag[k] + windowed_imag * W_fixed_real[k]
            mult_frac_bits = win_prod_frac_bits + osc_frac_bits
            
            # 3. Accumulation: A_k += result
            # To add numbers with different Q-formats, we must align their binary points.
            # We align the product to match the accumulator's fractional bits.
            shift_amount = mult_frac_bits - accum_frac_bits
            
            # Right-shift to align (equivalent to scaling down). This is a truncating shift.
            addend_real = mult_real >> shift_amount
            addend_imag = mult_imag >> shift_amount
            
            A_fixed_real[k] += addend_real
            A_fixed_imag[k] += addend_imag

            # Clamp/saturate the accumulator to model hardware overflow
            min_accum = -2**(accum_width - 1)
            max_accum = 2**(accum_width - 1) - 1
            A_fixed_real[k] = max(min_accum, min(max_accum, A_fixed_real[k]))
            A_fixed_imag[k] = max(min_accum, min(max_accum, A_fixed_imag[k]))

            # 4. Update Oscillator: W_k *= E_k (integer complex multiplication)
            W_next_real = W_fixed_real[k] * E_fixed_real[k] - W_fixed_imag[k] * E_fixed_imag[k]
            W_next_imag = W_fixed_real[k] * E_fixed_imag[k] + W_fixed_imag[k] * E_fixed_real[k]
            
            # The result has 2*osc_frac_bits. We need to shift it back to osc_frac_bits.
            W_fixed_real[k] = W_next_real >> osc_frac_bits
            W_fixed_imag[k] = W_next_imag >> osc_frac_bits

    # --- Final Output ---
    # Dequantize the final integer accumulator values back to floats for Python use
    A_final_float = np.array([
        dequantize(r, accum_width, accum_frac_bits) + 1j * dequantize(i, accum_width, accum_frac_bits)
        for r, i in zip(A_fixed_real, A_fixed_imag)
    ])
    
    final_dft_bins = {freq: accumulator for freq, accumulator in zip(freq_bins_to_calc, A_final_float)}
    return final_dft_bins

# --- function 2: power conversion and sorting ---
def convert_to_sorted_db_power(dft_bins):
    """
    Converts complex accumulator values to sorted, normalized dB power.
    This corresponds to a hardware block that does |A|^2, log, and normalization.
    """
    freqs = np.array(list(dft_bins.keys()))
    accumulators = np.array(list(dft_bins.values()))
    
    powers = np.abs(accumulators)**2
    if np.max(powers) == 0: return float('nan'), float('nan')
    
    power_db = 10 * np.log10(powers + 1e-20)
    power_db_norm = power_db - np.max(power_db)
    
    # sort everything by frequency for the search functions
    sort_indices = np.argsort(freqs)
    freqs_sorted = freqs[sort_indices]
    power_db_norm_sorted = power_db_norm[sort_indices]
    
    return freqs_sorted, power_db_norm_sorted

# --- function 3: left edge finder ---
def find_left_edge_points(freqs_sorted, power_db_norm_sorted, threshold_db=-20):
    """
    Finds the two adjacent points that cross the threshold for the left edge.
    This corresponds to the find_bw_left_edge.sv module.
    """
    # start search from the center (0 Hz)
    start_search_idx = np.argmin(np.abs(freqs_sorted))
    
    f1, f2, L1, L2 = (None,) * 4 # Use None to indicate "not found"
    
    # search from the center downwards into negative frequencies
    for i in range(start_search_idx, 0, -1):
        # crossing condition is: power[i-1] < threshold <= power[i]
        if power_db_norm_sorted[i-1] < threshold_db <= power_db_norm_sorted[i]:
            L1, L2 = power_db_norm_sorted[i-1], power_db_norm_sorted[i]
            f1, f2 = freqs_sorted[i-1], freqs_sorted[i]
            
    return f1, f2, L1, L2

# --- function 4: right edge finder ---
def find_right_edge_points(freqs_sorted, power_db_norm_sorted, threshold_db=-20):
    """
    Finds the two adjacent points that cross the threshold for the right edge.
    This corresponds to a find_bw_right_edge.sv module.
    """
    # start search from the center (0 Hz)
    start_search_idx = np.argmin(np.abs(freqs_sorted)) + 1
    
    f1, f2, L1, L2 = (None,) * 4 # use None to indicate "not found"

    # search from the center upwards into positive frequencies
    for i in range(start_search_idx, len(freqs_sorted) - 1):
        # crossing condition is: power[i+1] < threshold <= power[i]
        if power_db_norm_sorted[i+1] < threshold_db <= power_db_norm_sorted[i]:
            L1, L2 = power_db_norm_sorted[i], power_db_norm_sorted[i+1]
            f1, f2 = freqs_sorted[i], freqs_sorted[i+1]
            
    return f1, f2, L1, L2

# --- function 5: linear interpolation ---
def linear_interpolate_crossing(f1, f2, L1, L2, threshold_db=-20):
    """
    Performs the final linear interpolation to find the precise edge frequency.
    This corresponds to a simple arithmetic module.
    """
    if any(v is None for v in [f1, f2, L1, L2]) or (L2 - L1) == 0:
        return float('nan') # return Not-a-Number if inputs are invalid

    # standard linear interpolation formula
    f_star = f1 + (f2 - f1) * (threshold_db - L1) / (L2 - L1)
    return f_star


