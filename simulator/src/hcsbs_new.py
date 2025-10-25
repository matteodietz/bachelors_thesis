import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def hcsbs_bandwidth_search(b, N, k, beta, threshold_db, fs, window='hann'):
    """
    Finds spectral bandwidth edges using a HYBRID approach with an
    optimized, directional refinement search.
    """
    if len(b) != N: raise ValueError("Input vector `b` must have length `N`.")

    win = signal.windows.get_window(window, N)
    b_windowed = b * win

    # --- Stage 1: Coarse Edge Estimation on Meta-Spectrum ---
    b_coarse = b_windowed[:k]
    X_coarse_wrapped = np.fft.fft(b_coarse)
    
    P_coarse = np.abs(X_coarse_wrapped)**2
    if np.max(P_coarse) == 0: return 0, 0
    
    P_coarse_db = 10 * np.log10(P_coarse + 1e-20)
    P_coarse_db_norm = P_coarse_db - np.max(P_coarse_db)
    
    meta_spectrum = 1.0 - np.abs(threshold_db - P_coarse_db_norm)**2
    
    peak_indices, _ = signal.find_peaks(meta_spectrum, prominence=0.1)
    if len(peak_indices) < 2: return 0, N-1

    peak_heights = meta_spectrum[peak_indices]
    top_two_local_indices = np.argsort(peak_heights)[-2:]
    k_coarse_edges_wrapped = peak_indices[top_two_local_indices]
    
    # --- Stage 2: Robust, Directional Refinement ---
    
    coarse_peak_idx = np.argmax(P_coarse)
    peak_ref_bin = int(coarse_peak_idx * (N/k))
    exponent_ref = -1j * 2 * np.pi * peak_ref_bin * np.arange(N) / N
    inner_prod_ref = np.dot(b_windowed, np.exp(exponent_ref))
    peak_power_ref = np.abs(inner_prod_ref)**2
    if peak_power_ref == 0: peak_power_ref = 1e-20
    
    z_refined_edges = []
    
    # Identify which coarse edge is the lower one vs the upper one
    coarse_freqs = np.fft.fftfreq(k, 1/fs)
    freq1 = coarse_freqs[k_coarse_edges_wrapped[0]]
    freq2 = coarse_freqs[k_coarse_edges_wrapped[1]]

    k_coarse_lower = k_coarse_edges_wrapped[0] if freq1 < freq2 else k_coarse_edges_wrapped[1]
    k_coarse_upper = k_coarse_edges_wrapped[1] if freq1 < freq2 else k_coarse_edges_wrapped[0]
    
    coarse_edges_map = {'lower': k_coarse_lower, 'upper': k_coarse_upper}

    for edge_type, k_coarse_edge in coarse_edges_map.items():
        resolution_ratio = N / k
        
        # Define the initial search interval for this edge
        search_center = int(k_coarse_edge * resolution_ratio)
        search_width = int(resolution_ratio)
        
        # Perform a few iterations of binary-like search
        num_refinement_iterations = 1
        
        for i in range(num_refinement_iterations):
            # The step size halves in each iteration
            step = (search_width // 2) // (2**i)
            if step < 1: step = 1

            # --- THIS IS YOUR OPTIMIZATION ---
            # Based on the edge type, define the three test points directionally
            if edge_type == 'lower':
                # For the lower edge, we search INWARD (to the right)
                k_left = search_center
                k_right = (search_center + step * 2) % N
                k_center = (search_center + step) % N
            else: # edge_type == 'upper'
                # For the upper edge, we search INWARD (to the left)
                k_left = (search_center - step * 2) % N
                k_right = search_center
                k_center = (search_center - step) % N
            # --- END OF OPTIMIZATION ---

            bins_to_check = [k_left, k_center, k_right]
            powers_db_norm = []
            
            # Use the FULL `b_windowed` vector for accurate power calculation
            for j in bins_to_check:
                exponent = -1j * 2 * np.pi * j * np.arange(N) / N
                inner_product = np.dot(b_windowed, np.exp(exponent))
                power = np.abs(inner_product)**2
                powers_db_norm.append(10 * np.log10(power / peak_power_ref + 1e-20))

            meta_values = [1.0 - np.abs(threshold_db - p) for p in powers_db_norm]
            
            # The winner becomes the new edge of our search space
            winner_idx = np.argmax(meta_values)
            search_center = bins_to_check[winner_idx]

        z_refined_edges.append(search_center)

    z_lower = min(z_refined_edges)
    z_upper = max(z_refined_edges)
    
    freqs = np.fft.fftfreq(N, 1/fs)
    if freqs[z_lower] > freqs[z_upper]:
        z_lower, z_upper = z_upper, z_lower
        
    return z_lower, z_upper

# --- UNIT TEST ---
if __name__ == '__main__':
    print("--- Running unit test for hcsbs_bandwidth_search.py (NOISE-FREE) ---")

    # --- 1. Generate a NOISE-FREE Test Signal with a known bandwidth ---
    fs = 31.25e6
    N = 256
    t = np.arange(N) / fs
    
    center_freq = -2.0e6
    bandwidth_hz = 3.0e6
    
    # The ideal baseband signal is a sinc function in the time domain,
    # which corresponds to a perfect rectangle (brick-wall filter) in the frequency domain.
    # We will create this ideal signal.
    
    # Create the frequency axis for the N-point signal
    freqs = np.fft.fftfreq(N, 1/fs)
    
    # Create an ideal low-pass filter mask in the frequency domain
    lpf_mask = np.where(np.abs(freqs) <= bandwidth_hz / 2, 1, 0)
    
    # Create the baseband signal by taking the IFFT of the mask
    # The result is a sinc-like pulse centered in the time domain
    baseband_signal = np.fft.ifft(lpf_mask)
    
    # Shift to the desired center frequency
    test_signal = baseband_signal * np.exp(1j * 2 * np.pi * center_freq * t)
    # The `test_signal` is now a perfect, noise-free, band-limited signal.

    # --- 2. Find "Ground Truth" Edges with a High-Resolution FFT ---
    full_fft = np.fft.fftshift(np.fft.fft(test_signal * signal.windows.hann(N), n=N*16))
    freqs_full = np.fft.fftshift(np.fft.fftfreq(N*16, 1/fs))
    
    psd_db_full = 10 * np.log10(np.abs(full_fft)**2 + 1e-20)
    psd_db_full_norm = psd_db_full - np.max(psd_db_full)
    
    threshold_db = -20
    above_thresh_indices = np.where(psd_db_full_norm > threshold_db)[0]
    # Check if any points were above the threshold
    if len(above_thresh_indices) > 0:
        true_lower_edge = freqs_full[above_thresh_indices[0]]
        true_upper_edge = freqs_full[above_thresh_indices[-1]]
    else:
        true_lower_edge, true_upper_edge = (float('nan'), float('nan'))

    # --- 3. Run and Debug the Coarse Estimation Stage ---
    k = 32
    window = 'hann'
    win = signal.windows.get_window(window, N)
    b_windowed = test_signal * win

    b_coarse = b_windowed[:k]
    X_coarse_wrapped = np.fft.fft(b_coarse)
    
    P_coarse = np.abs(X_coarse_wrapped)**2
    P_coarse_db = 10 * np.log10(P_coarse + 1e-20)
    P_coarse_db_norm = P_coarse_db - np.max(P_coarse_db)
    
    meta_spectrum = 1.0 - np.abs(threshold_db - P_coarse_db_norm)

    # --- 4. Plot the Meta-Spectrum for Debugging ---
    coarse_freqs_shifted = np.fft.fftshift(np.fft.fftfreq(k, 1/fs))
    
    # Create the figure and the first axis explicitly
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.set_title('Coarse Estimation Debug Plot (Noise-Free)')
    
    # Plot the normalized dB spectrum on the first axis
    line1, = ax1.plot(coarse_freqs_shifted / 1e6, np.fft.fftshift(P_coarse_db_norm), 'b-o', label=f'Coarse {k}-pt Spectrum (dB)')
    ax1.set_xlabel('Frequency (MHz)')
    ax1.set_ylabel('Power (dB relative to peak)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True)
    
    # Create the second axis that shares the x-axis
    ax2 = ax1.twinx()
    
    # Plot the meta-spectrum on the second axis
    line2, = ax2.plot(coarse_freqs_shifted / 1e6, np.fft.fftshift(meta_spectrum), 'r-x', label='Meta-Spectrum')
    ax2.set_ylabel('Meta-Spectrum Value', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Manually combine the handles and labels from both plots
    ax1.legend(handles=[line1, line2], loc='upper right')

    # Ensure the layout is clean
    fig.tight_layout()
    plt.show()


    # --- 5. Run the HCSBS Algorithm and Compare ---
    z_lower_idx, z_upper_idx = hcsbs_bandwidth_search(
        b=test_signal, N=N, k=32, beta=0.5, threshold_db=threshold_db, fs=fs, window='hann'
    )
    
    freqs_unshifed = np.fft.fftfreq(N, 1/fs)
    hcsbs_lower_freq = freqs_unshifed[z_lower_idx]
    hcsbs_upper_freq = freqs_unshifed[z_upper_idx]

    print("\n--- Bandwidth Edge Estimation Comparison (Noise-Free) ---")
    print(f"Ground Truth Edges:         [{true_lower_edge/1e6:6.3f}, {true_upper_edge/1e6:6.3f}] MHz")
    print(f"HCSBS Estimated Edges:      [{hcsbs_lower_freq/1e6:6.3f}, {hcsbs_upper_freq/1e6:6.3f}] MHz")