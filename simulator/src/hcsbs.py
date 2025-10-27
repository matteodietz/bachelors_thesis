import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def hcsbs_bandwidth_search(b, N, k, beta, threshold_db, fs, window='hann'):
    """
    Finds spectral bandwidth edges using a robust HYBRID approach:
    1. Coarse estimation using Power Spectral Density (PSD).
    2. Fine, directional refinement using CS-style inner products.
    """
    if len(b) != N: raise ValueError("Input vector `b` must have length `N`.")

    win = signal.windows.get_window(window, N)
    b_windowed = b * win

    # --- Stage 1: Coarse Edge Estimation using PSD ---
    b_coarse = b_windowed[:k]
    
    freqs_coarse, P_coarse_psd = signal.welch(b_coarse, fs=fs, nperseg=k, window=window, return_onesided=False, scaling='density')
    
    freqs_coarse_shifted = np.fft.fftshift(freqs_coarse)
    P_coarse_psd_shifted = np.fft.fftshift(P_coarse_psd)
    
    if np.max(P_coarse_psd_shifted) == 0: return 0, 0
    
    P_coarse_db = 10 * np.log10(P_coarse_psd_shifted + 1e-20)
    P_coarse_db_norm = P_coarse_db - np.max(P_coarse_db)
    
    meta_spectrum = 1.0 - np.abs(threshold_db - P_coarse_db_norm)
    
    peak_indices_shifted, _ = signal.find_peaks(meta_spectrum, prominence=0.1)
    if len(peak_indices_shifted) < 2: return 0, N-1

    peak_heights = meta_spectrum[peak_indices_shifted]
    top_two_local_indices = np.argsort(peak_heights)[-2:]
    k_coarse_edges_shifted = peak_indices_shifted[top_two_local_indices]
    
    # Identify which coarse edge is lower vs upper
    k_coarse_lower_shifted = min(k_coarse_edges_shifted)
    k_coarse_upper_shifted = max(k_coarse_edges_shifted)

    # --- TO DO ---
    # --- Stage 2: Directional Refinement for Each Edge ---
    # The normalization factor for converting Power Spectrum to PSD
    win_fine = signal.windows.get_window(window, N)
    psd_scaling_factor_fine = fs * np.sum(win_fine**2)

    # 1. Get a STABLE Reference Peak PSD
    coarse_peak_idx_shifted = np.argmax(P_coarse_db_norm)  # maybe need to take a different reference peak
    coarse_peak_freq = freqs_coarse_shifted[coarse_peak_idx_shifted]
    freqs_fine_unwrapped = np.fft.fftfreq(N, 1/fs)
    peak_ref_bin = np.argmin(np.abs(freqs_fine_unwrapped - coarse_peak_freq))

    exponent_ref = -1j * 2 * np.pi * peak_ref_bin * np.arange(N) / N
    inner_prod_ref = np.dot(b_windowed, np.exp(exponent_ref))
    peak_power_ref = np.abs(inner_prod_ref)**2
    peak_psd_ref = peak_power_ref / psd_scaling_factor_fine # Convert to PSD
    if peak_psd_ref == 0: peak_psd_ref = 1e-20
    # --- TO DO ---
    
    z_refined_edges = []
    
    coarse_edges_map = {'lower': k_coarse_lower_shifted, 'upper': k_coarse_upper_shifted}

    for edge_type, k_coarse_edge_shifted in coarse_edges_map.items():
        # Map the coarse SHIFTED index to a coarse frequency
        coarse_freq = freqs_coarse_shifted[k_coarse_edge_shifted]
        # Find the center of the search in the FINE grid
        zoom_center_bin = np.argmin(np.abs(freqs_fine_unwrapped - coarse_freq))
        
        resolution_ratio = N / k
        zoom_width = int(resolution_ratio)
        half_width = zoom_width // 2
        
        # --- DIRECTIONAL SEARCH OPTIMIZATION ---
        if edge_type == 'lower':
            # For lower edge, search inward (right) from the coarse estimate
            start_offset = 0      # 0
            end_offset = zoom_width         # zoom_width
        else: # edge_type == 'upper'
            # For upper edge, search inward (left) from the coarse estimate
            start_offset = -zoom_width      # -zoom_width
            end_offset = 0       # 0
        
        bins_to_check = (zoom_center_bin + np.arange(start_offset, end_offset + 1)) % N
       

        n = int(beta * N)
        if n < 1: n = 1
        random_indices = np.linspace(0, N - 1, n, dtype=int)
        b_n = b_windowed[random_indices]
        
        meta_values_zoom = []
        for j in bins_to_check:
            exponent = -1j * 2 * np.pi * j * random_indices / N
            inner_product = np.dot(b_n, np.exp(exponent))
            power = np.abs(inner_product)**2
            power_scaled = power * (N/n)**2
            power_db_norm = 10 * np.log10(power_scaled / peak_power_ref + 1e-20)
            meta_value = 1.0 - np.abs(threshold_db - power_db_norm)
            meta_values_zoom.append(meta_value)
            
        k_refined_local = np.argmax(meta_values_zoom)
        z_refined_edges.append(bins_to_check[k_refined_local])

    z_lower = min(z_refined_edges)
    z_upper = max(z_refined_edges)
    
    if freqs_fine_unwrapped[z_lower] > freqs_fine_unwrapped[z_upper]:
        z_lower, z_upper = z_upper, z_lower
        
    return z_lower, z_upper


# --- UNIT TEST ---
if __name__ == '__main__':
    print("--- Running unit test for hcsbs_bandwidth_search.py (PSD-based) ---")

    # --- 1. Generate a Test Signal ---
    fs = 31.25e6
    N = 256
    t = (np.arange(N) - N / 2) / fs
    
    center_freq = -2.0e6
    bandwidth_hz = 3.0e6
    
    # Use the robust Gaussian signal from before
    freqs_shifted = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))
    shifted_mask = np.exp(-((freqs_shifted - center_freq)**2) / (2 * (bandwidth_hz / 6.06)**2))
    lpf_mask = np.fft.ifftshift(shifted_mask)
    test_signal = np.fft.ifft(lpf_mask)

    # --- 2. Find "Ground Truth" Edges ---
    threshold_db = -20
    
    full_fft = np.fft.fftshift(np.fft.fft(test_signal * signal.windows.hann(N), n=N*16))
    freqs_full = np.fft.fftshift(np.fft.fftfreq(N*16, 1/fs))
    psd_db_full_norm = 10 * np.log10(np.abs(full_fft)**2 + 1e-20)
    psd_db_full_norm -= np.max(psd_db_full_norm)
    
    above_thresh_indices = np.where(psd_db_full_norm > threshold_db)[0]
    true_lower_edge = freqs_full[above_thresh_indices[0]]
    true_upper_edge = freqs_full[above_thresh_indices[-1]]

    # --- 3. Run and Debug the Coarse Estimation Stage using PSD ---
    k = 32
    window = 'hann'
    b_windowed = test_signal * signal.windows.get_window(window, N)
    b_coarse = b_windowed[:k]

    # Calculate Coarse PSD
    freqs_coarse, P_coarse_psd = signal.welch(b_coarse, fs=fs, nperseg=k, window=window, return_onesided=False, scaling='density')
    freqs_coarse_shifted = np.fft.fftshift(freqs_coarse)
    P_coarse_psd_shifted = np.fft.fftshift(P_coarse_psd)
    
    P_coarse_db_norm = 10 * np.log10(P_coarse_psd_shifted + 1e-20)
    P_coarse_db_norm -= np.max(P_coarse_db_norm)
    
    meta_spectrum_coarse = 1.0 - np.abs(threshold_db - P_coarse_db_norm)

    # Calculate Fine PSD for comparison
    freqs_fine, P_fine_psd = signal.welch(b_windowed, fs=fs, nperseg=N, window=window, return_onesided=False, scaling='density')
    freqs_fine_shifted = np.fft.fftshift(freqs_fine)
    P_fine_psd_shifted = np.fft.fftshift(P_fine_psd)
    P_fine_db_norm = 10 * np.log10(P_fine_psd_shifted + 1e-20)
    P_fine_db_norm -= np.max(P_fine_db_norm)
    
    meta_spectrum_fine = 1.0 - np.abs(threshold_db - P_fine_db_norm)

    # --- 4. Plot the Meta-Spectrum for Debugging ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    ax1.set_title('Power Spectral Density Comparison (Coarse vs. Full)')
    ax1.plot(freqs_fine_shifted / 1e6, P_fine_db_norm, 'g-', label=f'Full {N}-pt PSD (Ground Truth)')
    ax1.plot(freqs_coarse_shifted / 1e6, P_coarse_db_norm, 'b-o', label=f'Coarse {k}-pt PSD')
    ax1.axhline(y=threshold_db, color='grey', linestyle=':', label=f'{threshold_db} dB Threshold')
    ax1.set_ylabel('Power Spectral Density (dB)')
    ax1.legend()
    ax1.grid(True)
    ax1.set_ylim(-60, 5)

    ax2.set_title('Meta-Spectrum Comparison (Coarse vs. Full)')
    ax2.plot(freqs_fine_shifted / 1e6, meta_spectrum_fine, 'g-', label=f'Full {N}-pt Meta-Spectrum')
    ax2.plot(freqs_coarse_shifted / 1e6, meta_spectrum_coarse, 'r-x', label=f'Coarse {k}-pt Meta-Spectrum')
    ax2.set_xlabel('Frequency (MHz)')
    ax2.set_ylabel('Meta-Spectrum Value')
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    plt.show()

    # --- 5. Run the HCSBS Algorithm and Compare ---
    z_lower_idx, z_upper_idx = hcsbs_bandwidth_search(
        b=test_signal, N=N, k=32, beta=0.8, threshold_db=threshold_db, fs=fs, window='hann'
    )
    
    freqs_unshifed = np.fft.fftfreq(N, 1/fs)
    hcsbs_lower_freq = freqs_unshifed[z_lower_idx]
    hcsbs_upper_freq = freqs_unshifed[z_upper_idx]

    print("\n--- Bandwidth Edge Estimation Comparison (PSD-based) ---")
    print(f"Ground Truth Edges:         [{true_lower_edge/1e6:6.3f}, {true_upper_edge/1e6:6.3f}] MHz")
    print(f"HCSBS Estimated Edges:      [{hcsbs_lower_freq/1e6:6.3f}, {hcsbs_upper_freq/1e6:6.3f}] MHz")