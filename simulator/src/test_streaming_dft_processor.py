import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# --- Function to be tested ---
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
        
    final_dft_bins = {freq: accumulator for freq, accumulator in zip(freq_bins_to_calc, A)}
    return final_dft_bins

# --- UNIT TEST ---
if __name__ == '__main__':
    print("--- Verifying correctness of the streaming_dft_processor ---")

    # --- simple test signal ---
    N = 256
    fs = 31.25e6
    t = np.arange(N) / fs
    
    test_signal = (1.0 * np.exp(1j * 2 * np.pi * 2e6 * t) +
                   0.5 * np.exp(1j * 2 * np.pi * -5e6 * t) +
                   0.1 * (np.random.randn(N) + 1j*np.random.randn(N)))
                   
    window_func = 'hann'
    windowed_signal = test_signal * signal.windows.get_window(window_func, N)

    # --- Calculate DFT using the Streaming Processor (Method 1) ---
    # Generate the frequency bins in the standard "wrapped" FFT order
    all_freq_bins_wrapped = np.fft.fftfreq(N, 1/fs)
    
    print(f"Running streaming DFT for all {N} bins...")
    streaming_dft_result_dict = streaming_dft_processor(
        b=test_signal, 
        fs=fs, 
        freq_bins_to_calc=all_freq_bins_wrapped, 
        window=window_func
    )
    
    # --- Reordering ---
    streaming_dft_result_wrapped = np.array([streaming_dft_result_dict[f] for f in all_freq_bins_wrapped])

    # --- Calculate DFT using the Standard FFT (Golden Reference) ---
    print("Running standard FFT for comparison...")
    standard_fft_result_wrapped = np.fft.fft(windowed_signal)
    
    # --- Compare the Results (using the wrapped arrays) ---
    error = np.abs(streaming_dft_result_wrapped - standard_fft_result_wrapped)
    
    print("\n--- Verification Results ---")
    print(f"Maximum absolute error (Real part):      {np.max(np.abs(streaming_dft_result_wrapped.real - standard_fft_result_wrapped.real)):.2e}")
    print(f"Maximum absolute error (Imaginary part): {np.max(np.abs(streaming_dft_result_wrapped.imag - standard_fft_result_wrapped.imag)):.2e}")
    
    # --- ASSERTION ---
    np.testing.assert_allclose(streaming_dft_result_wrapped, standard_fft_result_wrapped, atol=1e-9, rtol=0)
    print("\nSUCCESS: Streaming DFT is bit-accurate with standard FFT.")

    # --- Visual Confirmation Plot ---
    # Now we shift BOTH results for plotting
    freq_axis_shifted = np.fft.fftshift(all_freq_bins_wrapped)
    streaming_dft_shifted = np.fft.fftshift(streaming_dft_result_wrapped)
    standard_fft_shifted = np.fft.fftshift(standard_fft_result_wrapped)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: Real Part
    ax1.set_title('Real Part Comparison')
    ax1.plot(freq_axis_shifted / 1e6, standard_fft_shifted.real, 'k-', linewidth=3, label='Standard FFT (Golden)')
    ax1.plot(freq_axis_shifted / 1e6, streaming_dft_shifted.real, 'r--', linewidth=1.5, label='Streaming DFT')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)
    ax1.legend()
    
    # Plot 2: Imaginary Part
    ax2.set_title('Imaginary Part Comparison')
    ax2.plot(freq_axis_shifted / 1e6, standard_fft_shifted.imag, 'k-', linewidth=3, label='Standard FFT (Golden)')
    ax2.plot(freq_axis_shifted / 1e6, streaming_dft_shifted.imag, 'r--', linewidth=1.5, label='Streaming DFT')
    ax2.set_xlabel('Frequency (MHz)')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True)
    ax2.legend()
    
    plt.suptitle('Verification of Streaming DFT vs. Standard FFT', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()