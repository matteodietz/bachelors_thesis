import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path

# --- function 1: the core streaming processor ---
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

    # streaming DFT calculation
    for n in range(N):
        x_n = b[n]
        h_n = win[n]
        A += x_n * h_n * W
        W *= E
        
    # dictionary of frequency bins with their corresponding accumulator value (DFT)
    final_dft_bins = {freq: accumulator for freq, accumulator in zip(freq_bins_to_calc, A)}
    return final_dft_bins

# def streaming_dft_processor(b, fs, freq_bins_to_calc, window='hann'):
#     """
#     Simulates a one-pass, streaming, sparse DFT (Goertzel-like).
#     Returns a list of structured results.
#     """
#     N = len(b)
#     win = signal.windows.get_window(window, N)
    
#     K = len(freq_bins_to_calc)
#     A = np.zeros(K, dtype=np.complex128)
#     W = np.ones(K, dtype=np.complex128)
#     E = np.exp(-1j * 2 * np.pi * freq_bins_to_calc / fs)

#     for n in range(N):
#         x_n = b[n]
#         h_n = win[n]
#         A += x_n * h_n * W
#         W *= E
        
#     # --- THIS IS THE NEW, BETTER OUTPUT ---
#     # Create the full N-point frequency axis to find the indices ONCE.
#     full_freq_axis_unwrapped = np.fft.fftfreq(N, 1/fs)
    
#     # Create a structured list of results
#     results = []
#     for i in range(K):
#         freq = freq_bins_to_calc[i]
#         accum = A[i]
#         # Find the closest integer bin index for this frequency
#         bin_index = np.argmin(np.abs(full_freq_axis_unwrapped - freq))
        
#         results.append({
#             'freq_hz': freq,
#             'bin_index': bin_index,
#             'accumulator': accum
#         })
        
#     return results

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

# def convert_to_sorted_db_power(dft_results):
#     """
#     Converts the structured output from the DFT processor to sorted, 
#     normalized dB power.
#     """
#     # Unpack the list of dictionaries
#     freqs = np.array([res['freq_hz'] for res in dft_results])
#     bin_indices = np.array([res['bin_index'] for res in dft_results])
#     accumulators = np.array([res['accumulator'] for res in dft_results])
    
#     powers = np.abs(accumulators)**2
#     if np.max(powers) == 0:
#         return np.array([]), np.array([]), np.array([])
    
#     power_db = 10 * np.log10(powers + 1e-20)
#     power_db_norm = power_db - np.max(power_db)
    
#     # Sort everything by frequency
#     sort_indices = np.argsort(freqs)
    
#     freqs_sorted = freqs[sort_indices]
#     power_db_norm_sorted = power_db_norm[sort_indices]
#     bin_indices_sorted = bin_indices[sort_indices]
    
#     return freqs_sorted, power_db_norm_sorted, bin_indices_sorted


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

# def find_left_edge_points(freqs_sorted, power_db_norm_sorted, bin_indices_sorted, threshold_db=-20):
#     """
#     Finds the two adjacent points that cross the threshold for the left edge.
#     """
#     start_search_idx = np.argmin(np.abs(freqs_sorted))
    
#     f1, f2, L1, L2, k1, k2 = (None,) * 6
    
#     for i in range(start_search_idx, 0, -1):
#         if power_db_norm_sorted[i-1] < threshold_db <= power_db_norm_sorted[i]:
#             L1, L2 = power_db_norm_sorted[i-1], power_db_norm_sorted[i]
#             f1, f2 = freqs_sorted[i-1], freqs_sorted[i]
#             k1, k2 = bin_indices_sorted[i-1], bin_indices_sorted[i]
            
#     return f1, f2, L1, L2, k1, k2


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

# def find_right_edge_points(freqs_sorted, power_db_norm_sorted, bin_indices_sorted, threshold_db=-20):
#     """
#     Finds the two adjacent points that cross the threshold for the left edge.
#     """
#     start_search_idx = np.argmin(np.abs(freqs_sorted)) + 1
    
#     f1, f2, L1, L2, k1, k2 = (None,) * 6
    
#     for i in range(start_search_idx, len(freqs_sorted) - 1):
#         # crossing condition is: power[i+1] < threshold <= power[i]
#         if power_db_norm_sorted[i+1] < threshold_db <= power_db_norm_sorted[i]:
#             L1, L2 = power_db_norm_sorted[i], power_db_norm_sorted[i+1]
#             f1, f2 = freqs_sorted[i], freqs_sorted[i+1]
#             k1, k2 = bin_indices_sorted[i], bin_indices_sorted[i+1]
            
#     return f1, f2, L1, L2, k1, k2


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


