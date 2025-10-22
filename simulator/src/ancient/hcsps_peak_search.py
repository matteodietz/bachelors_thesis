import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path

# (import statements for loaders would go here)

def hcsps_peak_search(b, N, k, beta, window='hann'):
    """
    Finds the spectral peak of a 1D signal using the Hierarchical
    Compressed Sensing-based Peak Search (HCSPS) algorithm. (Corrected)
    """
    if len(b) != N:
        raise ValueError("Input vector `b` must have length `N`.")

    win = signal.windows.get_window(window, N)
    b_windowed = b * win
    
    # --- Stage 1: Coarse Grid Estimation ---
    step = N // k
    b_coarse = b_windowed[::step]
    
    # Perform a k-point FFT
    X_coarse_wrapped = np.fft.fft(b_coarse)
    
    # Shift the spectrum so 0 Hz is in the center BEFORE finding the peak
    X_coarse_shifted = np.fft.fftshift(X_coarse_wrapped)
    
    # Find the index of the maximum in the SHIFTED spectrum
    k_coarse_shifted = np.argmax(np.abs(X_coarse_shifted)**2)
    
    # Convert this shifted index back to a standard FFT index
    k_coarse = (k_coarse_shifted - k // 2 + k) % k
    print(f"Coarse Peak Estimate:   {k_coarse}") 
    
    
    # --- Stage 2: CS-Based Refinement ---
    zoom_center_bin = k_coarse * step
    zoom_width = step
    
    zoom_start = max(0, zoom_center_bin - zoom_width // 2)
    zoom_end = min(N, zoom_center_bin + zoom_width // 2)
    
    bins_to_check = np.arange(zoom_start, zoom_end)
    
    n = int(beta * N)
    if n < 1: n = 1
    
    random_indices = np.linspace(0, N - 1, n, dtype=int)
    b_n = b_windowed[random_indices]
    
    powers_in_zoom_window = []
    for j in bins_to_check:
        # --- CORRECTED INNER PRODUCT ---
        # The DFT basis vector is e^(-j*...*n). The inner product is sum(x[n] * basis.conj())
        # which is equivalent to sum(x[n] * e^(j*...*n)).
        exponent = 1j * 2 * np.pi * j * random_indices / N
        basis_vector = np.exp(exponent)
        # The inner product is the dot product of the signal with the basis vector
        inner_product = np.dot(b_n, basis_vector)
        # --- END CORRECTION ---
        powers_in_zoom_window.append(np.abs(inner_product)**2)
        
    k_refined_local = np.argmax(powers_in_zoom_window)
    z_final = zoom_start + k_refined_local
    peak_power = powers_in_zoom_window[k_refined_local]
    
    return z_final, peak_power

# --- UNIT TEST (No changes needed here) ---
if __name__ == '__main__':
    print("--- Running unit test for hcsps_peak_search.py ---")

    fs = 31.25e6
    N = 256
    t = np.arange(N) / fs
    
    true_peak_freq = 3.14159e6
    test_signal = np.exp(1j * 2 * np.pi * true_peak_freq * t) + \
                  0.1 * np.random.randn(N) * (1 + 1j) # Noise should be complex for I/Q

    full_fft = np.fft.fftshift(np.fft.fft(test_signal * signal.windows.hann(N), n=N*16))
    freqs_full = np.fft.fftshift(np.fft.fftfreq(N*16, 1/fs))
    true_peak_index_fine = np.argmax(np.abs(full_fft)**2)
    true_peak_freq_fine = freqs_full[true_peak_index_fine]
    
    standard_fft = np.fft.fftshift(np.fft.fft(test_signal * signal.windows.hann(N)))
    freqs_standard = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))
    standard_peak_index_shifted = np.argmax(np.abs(standard_fft)**2)
    standard_peak_freq = freqs_standard[standard_peak_index_shifted]

    hcsps_peak_index, _ = hcsps_peak_search(
        b=test_signal,
        N=N,
        k=32,
        beta=0.5,
        window='hann'
    )
    # The HCSPS index is a standard, non-shifted index. We need to find its frequency.
    # We must use the original, non-shifted freqs array for this.
    freqs_unshifed = np.fft.fftfreq(N, 1/fs)
    hcsps_peak_freq = freqs_unshifed[hcsps_peak_index]

    print("\n--- Peak Frequency Estimation Comparison ---")
    print(f"True Peak Frequency:              {true_peak_freq / 1e6:.6f} MHz")
    print(f"High-Res FFT (Ground Truth):    {true_peak_freq_fine / 1e6:.6f} MHz")
    print(f"Standard {N}-point FFT Estimate:    {standard_peak_freq / 1e6:.6f} MHz (Error: {abs(true_peak_freq - standard_peak_freq)/1e3:.2f} kHz)")
    print(f"HCSPS Estimate:                   {hcsps_peak_freq / 1e6:.6f} MHz (Error: {abs(true_peak_freq - hcsps_peak_freq)/1e3:.2f} kHz)")
    
    # Assert that the HCSPS error is smaller than the standard FFT error
    # assert abs(true_peak_freq - hcsps_peak_freq) < abs(true_peak_freq - standard_peak_freq)
    # print("\nSUCCESS: HCSPS estimate is more accurate than the standard FFT peak.")