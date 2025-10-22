import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path

def hcsps_peak_search(b, N, k, beta, window='hann'):
    """
    Finds the spectral peak using a corrected Hierarchical CS-based Peak Search.
    Coarse estimation is now done on a contiguous block to prevent aliasing.
    """
    if len(b) != N:
        raise ValueError("Input vector `b` must have length `N`.")

    win = signal.windows.get_window(window, N)
    b_windowed = b * win

    # --- 1. coarse grid estimation ---
    b_coarse = b_windowed[:k]
    X_coarse_wrapped = np.fft.fft(b_coarse)
    
    # find the coarse peak in the standard wrapped FFT output (non-shifted)
    k_coarse_wrapped = np.argmax(np.abs(X_coarse_wrapped)**2)
    
    # --- 2. fine grid refinement ---
    resolution_ratio = N / k
    
    # map coarse bin index to the N-point grid
    zoom_center_bin = int(k_coarse_wrapped * resolution_ratio)

    # define search width and create list of bins to check
    zoom_width = int(resolution_ratio)
    half_width = zoom_width   # size can be adjusted
    
    # use modulo arithmetic to handle wrap-around for a circular array (the spectrum)
    bins_to_check = (zoom_center_bin + np.arange(-half_width, half_width + 1)) % N

    # print to verify that we have valid interval bounds
    print(f"Left Boundary to check: {zoom_center_bin - half_width}")
    print(f"Right Boundary to check: {zoom_center_bin + half_width + 1}")
    
    # handle the edge case where the generated list is empty (which should never occur)
    if len(bins_to_check) == 0:
        print("Warning: Zoom window was empty. Returning coarse estimate.")
        return k_coarse_wrapped * resolution_ratio, 0.0

    # extraction factor
    n = int(beta * N)
    if n < 1: n = 1
    
    # equally spaced bins (can be adjusted to be distributed in a different way)
    random_indices = np.linspace(0, N - 1, n, dtype=int)
    b_n = b_windowed[random_indices]
    
    powers_in_zoom_window = []
    for j in bins_to_check:
        # standard DFT inner product
        exponent = -1j * 2 * np.pi * j * random_indices / N
        inner_product = np.dot(b_n, np.exp(exponent))
        powers_in_zoom_window.append(np.abs(inner_product)**2)
        
    k_refined_local = np.argmax(powers_in_zoom_window)
    z_final = bins_to_check[k_refined_local]
    peak_power = powers_in_zoom_window[k_refined_local]
    
    return z_final, peak_power

# --- UNIT TEST ---
if __name__ == '__main__':
    print("--- Running unit test for hcsps_peak_search.py (Corrected Version) ---")

    fs = 31.25e6
    N = 256
    t = np.arange(N) / fs
    
    true_peak_freq = 3.14159e6
    test_signal = np.exp(1j * 2 * np.pi * true_peak_freq * t) + \
                  0.01 * (np.random.randn(N) + 1j * np.random.randn(N))
    
    # standard N-point FFT
    standard_fft = np.fft.fftshift(np.fft.fft(test_signal * signal.windows.hann(N)))
    freqs_standard = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))
    standard_peak_freq = freqs_standard[np.argmax(np.abs(standard_fft)**2)]

    # HCSPS algorithm
    hcsps_peak_index, _ = hcsps_peak_search(
        b=test_signal, N=N, k=32, beta=0.023, window='hann'
    )
    # the final index is on the N-point grid
    freqs_unshifed = np.fft.fftfreq(N, 1/fs)
    hcsps_peak_freq = freqs_unshifed[hcsps_peak_index]

    print("\n--- Peak Frequency Estimation Comparison ---")
    print(f"True Peak Frequency:              {true_peak_freq / 1e6:.6f} MHz")
    print(f"Standard {N}-point FFT Estimate:    {standard_peak_freq / 1e6:.6f} MHz (Error: {abs(true_peak_freq - standard_peak_freq)/1e3:.2f} kHz)")
    print(f"HCSPS Estimate:                   {hcsps_peak_freq / 1e6:.6f} MHz (Error: {abs(true_peak_freq - hcsps_peak_freq)/1e3:.2f} kHz)")