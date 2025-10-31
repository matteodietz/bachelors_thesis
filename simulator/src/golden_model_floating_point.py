import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path

# --- Function 1: The Core Streaming Processor (Unchanged) ---
def streaming_dft_processor(b, fs, freq_bins_to_calc, window='hann'):
    """
    Simulates a one-pass, streaming, sparse DFT (Goertzel-like).
    Returns a dictionary of {frequency: complex_accumulator_value}.
    """
    N = len(b)
    win = signal.windows.get_window(window, N)
    
    K = len(freq_bins_to_calc)
    A = np.zeros(K, dtype=np.complex128)
    W = np.ones(K, dtype=np.complex128)
    E = np.exp(-1j * 2 * np.pi * freq_bins_to_calc / fs)

    for n in range(N):
        x_n = b[n]
        h_n = win[n]
        A += x_n * h_n * W
        W *= E
        
    final_dft_bins = {freq: accumulator for freq, accumulator in zip(freq_bins_to_calc, A)}
    return final_dft_bins

# --- Function 2: Power Conversion and Sorting ---
def convert_to_sorted_db_power(dft_bins):
    """
    Converts complex accumulator values to sorted, normalized dB power.
    This corresponds to a hardware block that does |A|^2, log, and normalization.
    """
    freqs = np.array(list(dft_bins.keys()))
    accumulators = np.array(list(dft_bins.values()))
    
    powers = np.abs(accumulators)**2
    if np.max(powers) == 0:
        # Return empty arrays if there's no signal to avoid errors
        return np.array([]), np.array([])
    
    power_db = 10 * np.log10(powers + 1e-20)
    power_db_norm = power_db - np.max(power_db)
    
    # Sort everything by frequency for the search functions
    sort_indices = np.argsort(freqs)
    freqs_sorted = freqs[sort_indices]
    power_db_norm_sorted = power_db_norm[sort_indices]
    
    return freqs_sorted, power_db_norm_sorted

# --- Function 3: Left Edge Finder ---
def find_left_edge_points(freqs_sorted, power_db_norm_sorted, threshold_db=-20):
    """
    Finds the two adjacent points that cross the threshold for the left edge.
    This corresponds to the find_bw_left_edge.sv module.
    """
    # Start search from the center (0 Hz)
    start_search_idx = np.argmin(np.abs(freqs_sorted))
    
    f1, f2, L1, L2 = (None,) * 4 # Use None to indicate "not found"
    
    # Search from the center downwards into negative frequencies
    for i in range(start_search_idx, 0, -1):
        # The crossing condition is: power[i-1] < threshold <= power[i]
        if power_db_norm_sorted[i-1] < threshold_db <= power_db_norm_sorted[i]:
            L1, L2 = power_db_norm_sorted[i-1], power_db_norm_sorted[i]
            f1, f2 = freqs_sorted[i-1], freqs_sorted[i]
            break # Exit loop once the first crossing is found
            
    return f1, f2, L1, L2

# --- Function 4: Right Edge Finder ---
def find_right_edge_points(freqs_sorted, power_db_norm_sorted, threshold_db=-20):
    """
    Finds the two adjacent points that cross the threshold for the right edge.
    This corresponds to a find_bw_right_edge.sv module.
    """
    # Start search from the center (0 Hz)
    start_search_idx = np.argmin(np.abs(freqs_sorted))
    
    f1, f2, L1, L2 = (None,) * 4 # Use None to indicate "not found"

    # Search from the center upwards into positive frequencies
    for i in range(start_search_idx, len(freqs_sorted) - 1):
        # The crossing condition is: power[i+1] < threshold <= power[i]
        if power_db_norm_sorted[i+1] < threshold_db <= power_db_norm_sorted[i]:
            L1, L2 = power_db_norm_sorted[i], power_db_norm_sorted[i+1]
            f1, f2 = freqs_sorted[i], freqs_sorted[i+1]
            break # Exit loop once the first crossing is found
            
    return f1, f2, L1, L2

# --- Function 5: Linear Interpolation ---
def linear_interpolate_crossing(f1, f2, L1, L2, threshold_db=-20):
    """
    Performs the final linear interpolation to find the precise edge frequency.
    This corresponds to a simple arithmetic module.
    """
    if any(v is None for v in [f1, f2, L1, L2]) or (L2 - L1) == 0:
        return float('nan') # Return Not-a-Number if inputs are invalid

    # Standard linear interpolation formula
    f_star = f1 + (f2 - f1) * (threshold_db - L1) / (L2 - L1)
    return f_star


