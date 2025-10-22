import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def hcsbs_bandwidth_search(b, N, k, beta, threshold_db, fs, window='hann'):
    """
    Finds spectral bandwidth edges using the Hierarchical CS-based search on a
    transformed "meta-spectrum". This is a single, self-contained function.

    Args:
        b (np.ndarray): 1D time-domain I/Q signal vector of length N.
        N (int): Full resolution length of the DFT (must match len(b)).
        k (int): Size of the coarse FFT (power of two).
        beta (float): CS extraction rate for refinement.
        threshold_db (float): Target dB threshold for bandwidth edges (e.g., -20).
        fs (float): Sample rate of the signal `b`.
        window (str): Window function to apply.

    Returns:
        tuple: (z_lower, z_upper) - refined integer indices of the bandwidth edges.
    """
    if len(b) != N:
        raise ValueError("Input vector `b` must have length `N`.")

    win = signal.windows.get_window(window, N)
    b_windowed = b * win

    # --- 1. coarse edge estimation on meta-spectrum ---
    b_coarse = b_windowed[:k]
    X_coarse_wrapped = np.fft.fft(b_coarse)
    
    P_coarse = np.abs(X_coarse_wrapped)**2
    if np.max(P_coarse) == 0: return 0, 0 # handle case of pure zero signal
    
    P_coarse_db = 10 * np.log10(P_coarse + 1e-20)
    P_coarse_db_norm = P_coarse_db - np.max(P_coarse_db)
    
    meta_spectrum = 1.0 - np.abs(threshold_db - P_coarse_db_norm)
    
    peak_indices, _ = signal.find_peaks(meta_spectrum, prominence=0.1)
    if len(peak_indices) < 2:
        return 0, N-1 # Fallback: return full spectrum if edges aren't clear

    peak_heights = meta_spectrum[peak_indices]
    top_two_local_indices = np.argsort(peak_heights)[-2:]
    k_coarse_edges = peak_indices[top_two_local_indices]

    # --- NEW DEBUGGING SECTION ---
    print(f"\n--- Coarse Estimation Debug ---")
    print(f"Coarse edge indices found (out of {k} bins): {k_coarse_edges[0]}, {k_coarse_edges[1]}")
    
    # Create the frequency axis for the k-point coarse FFT
    coarse_freqs_unwrapped = np.fft.fftfreq(k, 1/fs)
    
    # Get the frequencies corresponding to the coarse edge indices
    coarse_edge_freq1_mhz = coarse_freqs_unwrapped[k_coarse_edges[0]] / 1e6
    coarse_edge_freq2_mhz = coarse_freqs_unwrapped[k_coarse_edges[1]] / 1e6
    
    print(f"Coarse edge frequencies estimated at: {coarse_edge_freq1_mhz:.3f} MHz and {coarse_edge_freq2_mhz:.3f} MHz")
    # --- END OF DEBUGGING SECTION ---
    
    z_refined_edges = []
    
    # --- 2. Refine Each Coarse Edge Estimate ---
    for k_coarse_edge in k_coarse_edges:
        resolution_ratio = N / k
        
        zoom_center_bin = int(k_coarse_edge * resolution_ratio)
        zoom_width = int(resolution_ratio)
        half_width = zoom_width // 2
        
        bins_to_check = (zoom_center_bin + np.arange(-half_width, half_width + 1)) % N
        
        n = int(beta * N)
        if n < 1: n = 1
        
        random_indices = np.linspace(0, N - 1, n, dtype=int)
        b_n = b_windowed[random_indices]
        
        # We need the reference peak power for normalization
        coarse_peak_idx = np.argmax(P_coarse)
        peak_ref_bin = int(coarse_peak_idx * resolution_ratio)
        exponent_ref = -1j * 2 * np.pi * peak_ref_bin * random_indices / N
        inner_prod_ref = np.dot(b_n, np.exp(exponent_ref))
        peak_power_ref = np.abs(inner_prod_ref)**2
        
        if peak_power_ref == 0: peak_power_ref = 1e-20

        # Perform CS-based search for the threshold crossing on the meta-spectrum
        meta_values_zoom = []
        for j in bins_to_check:
            exponent = -1j * 2 * np.pi * j * random_indices / N
            inner_product = np.dot(b_n, np.exp(exponent))
            power = np.abs(inner_product)**2
            
            power_db_norm = 10 * np.log10(power / peak_power_ref + 1e-20)
            meta_value = 1.0 - np.abs(threshold_db - power_db_norm)
            meta_values_zoom.append(meta_value)
            
        k_refined_local = np.argmax(meta_values_zoom)
        z_refined_edges.append(bins_to_check[k_refined_local])

    z_lower = min(z_refined_edges)
    z_upper = max(z_refined_edges)
    
    # Final sanity check to ensure lower is not greater than upper
    # This can happen if the spectrum wraps around 0 Hz
    freqs = np.fft.fftfreq(N, 1/fs)
    if freqs[z_lower] > freqs[z_upper]:
        z_lower, z_upper = z_upper, z_lower # Swap them
        
    return z_lower, z_upper

# --- UNIT TEST ---
if __name__ == '__main__':
    print("--- Running unit test for hcsbs_bandwidth_search.py ---")

    # --- 1. Generate a Test Signal with a known bandwidth ---
    fs = 31.25e6
    N = 256
    t = np.arange(N) / fs
    
    center_freq = -2.0e6
    bandwidth_hz = 3.0e6
    low_cut = center_freq - bandwidth_hz / 2 # -3.5 MHz
    high_cut = center_freq + bandwidth_hz / 2 # -0.5 MHz
    
    lpf_truth = signal.firwin(101, bandwidth_hz / 2, fs=fs, pass_zero=True)
    noise = np.random.randn(N*4) + 1j*np.random.randn(N*4)
    baseband_signal = signal.lfilter(lpf_truth, 1.0, noise)[-N:]
    test_signal = baseband_signal * np.exp(1j * 2 * np.pi * center_freq * t)
    
    # --- 2. Find "Ground Truth" Edges with a High-Resolution FFT ---
    threshold_db = -20
    # ... (Ground truth calculation is the same)
    
    # --- 3. Run and Debug the Coarse Estimation Stage ---
    k = 32 # Coarse FFT size
    window = 'hann'
    win = signal.windows.get_window(window, N)
    b_windowed = test_signal * win

    # Perform the coarse estimation steps here for debugging
    b_coarse = b_windowed[:k]
    X_coarse_wrapped = np.fft.fft(b_coarse)
    
    P_coarse = np.abs(X_coarse_wrapped)**2
    P_coarse_db = 10 * np.log10(P_coarse + 1e-20)
    P_coarse_db_norm = P_coarse_db - np.max(P_coarse_db)
    
    meta_spectrum = 1.0 - np.abs(threshold_db - P_coarse_db_norm)

    # --- 4. Plot the Meta-Spectrum for Debugging ---
    coarse_freqs_shifted = np.fft.fftshift(np.fft.fftfreq(k, 1/fs))
    
    plt.figure(figsize=(12, 6))
    plt.title('Coarse Estimation Debug Plot')
    
    # Plot the normalized dB spectrum
    plt.plot(coarse_freqs_shifted / 1e6, np.fft.fftshift(P_coarse_db_norm), 'b-o', label=f'Coarse {k}-pt Spectrum (dB)')
    
    # Plot the meta-spectrum on a secondary y-axis
    ax2 = plt.gca().twinx()
    ax2.plot(coarse_freqs_shifted / 1e6, np.fft.fftshift(meta_spectrum), 'r-x', label='Meta-Spectrum')
    ax2.set_ylabel('Meta-Spectrum Value', color='r')
    
    plt.xlabel('Frequency (MHz)')
    plt.gca().set_ylabel('Power (dB relative to peak)', color='b')
    plt.grid(True)
    # Combine legends
    lines, labels = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    plt.show()
    
    # --- 2. Find "Ground Truth" Edges with a High-Resolution FFT ---
    full_fft = np.fft.fftshift(np.fft.fft(test_signal * signal.windows.hann(N), n=N*16))
    freqs_full = np.fft.fftshift(np.fft.fftfreq(N*16, 1/fs))
    
    psd_db_full = 10 * np.log10(np.abs(full_fft)**2 + 1e-20)
    psd_db_full_norm = psd_db_full - np.max(psd_db_full)
    
    threshold_db = -15
    above_thresh_indices = np.where(psd_db_full_norm > threshold_db)[0]
    true_lower_edge = freqs_full[above_thresh_indices[0]]
    true_upper_edge = freqs_full[above_thresh_indices[-1]]
    
    # --- 3. Run your new HCSBS Bandwidth Search Algorithm ---
    z_lower_idx, z_upper_idx = hcsbs_bandwidth_search(
        b=test_signal, N=N, k=32, beta=0.5, threshold_db=threshold_db, fs=fs, window='hann'
    )
    
    freqs_unshifed = np.fft.fftfreq(N, 1/fs)
    hcsbs_lower_freq = freqs_unshifed[z_lower_idx]
    hcsbs_upper_freq = freqs_unshifed[z_upper_idx]

    # --- 4. Compare the Results ---
    print("\n--- Bandwidth Edge Estimation Comparison ---")
    print(f"Ground Truth Edges:         [{true_lower_edge/1e6:6.3f}, {true_upper_edge/1e6:6.3f}] MHz")
    print(f"HCSBS Estimated Edges:      [{hcsbs_lower_freq/1e6:6.3f}, {hcsbs_upper_freq/1e6:6.3f}] MHz")

    # --- 5. Visual Confirmation Plot ---
    standard_fft = np.fft.fftshift(np.fft.fft(test_signal * signal.windows.hann(N)))
    freqs_standard = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))
    psd_db_standard_norm = 10 * np.log10(np.abs(standard_fft)**2 + 1e-20)
    psd_db_standard_norm -= np.max(psd_db_standard_norm)
    
    plt.figure(figsize=(12, 6))
    plt.plot(freqs_standard / 1e6, psd_db_standard_norm, label=f'Standard {N}-pt Spectrum')
    plt.axhline(y=threshold_db, color='grey', linestyle=':', label=f'{threshold_db} dB Threshold')
    plt.axvline(x=hcsbs_lower_freq / 1e6, color='r', linestyle='--', label=f'HCSBS Lower Edge ({hcsbs_lower_freq/1e6:.3f} MHz)')
    plt.axvline(x=hcsbs_upper_freq / 1e6, color='g', linestyle='--', label=f'HCSBS Upper Edge ({hcsbs_upper_freq/1e6:.3f} MHz)')
    plt.title('HCSBS Bandwidth Edge Estimation')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power (dB relative to peak)')
    plt.legend()
    plt.grid(True)
    plt.ylim(threshold_db - 20, 5)
    plt.show()