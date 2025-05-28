import numpy as np
import torch
import torch.nn.functional as F
import scipy.signal as signal

def torch_spectrogram(waveforms, fs=50, nperseg=256, overlap=0.5):
    """
    Extract spectrograms from waveforms using PyTorch operations,
    but mimicking the behavior of scipy.signal.spectrogram for 
    compatibility with pre-trained QuakeXNet weights.
    
    Parameters:
        waveforms: torch.Tensor of shape (batch_size, n_channels, n_samples)
        fs: sampling rate (Hz)
        nperseg: number of points per FFT segment
        overlap: fractional overlap between segments
    
    Returns:
        spectrograms: torch.Tensor of shape (batch_size, n_channels, freq_bins, time_bins)
    """
    # Make sure we're working with torch tensors
    if not isinstance(waveforms, torch.Tensor):
        waveforms = torch.tensor(waveforms, dtype=torch.float32)
    
    # Calculate overlap parameters
    noverlap = int(nperseg * overlap)
    hop_length = nperseg - noverlap
    
    # Get shapes
    batch_size, n_channels, n_samples = waveforms.shape
    
    # Create window - use Hann window to match scipy default
    window = torch.hann_window(nperseg, device=waveforms.device)
    
    # Process one sample from the batch to determine output dimensions
    # Run scipy.signal.spectrogram for reference
    sample_np = waveforms[0, 0].cpu().detach().numpy()
    
    try:
        f, t, Sxx_ref = signal.spectrogram(
            sample_np, fs=fs, nperseg=nperseg, noverlap=noverlap
        )
        n_freqs, n_times = Sxx_ref.shape
    except Exception as e:
        print(f"Error in reference spectrogram computation: {e}")
        print("Falling back to default sizes")
        # Typical sizes for 5000 samples with nperseg=256, noverlap=128
        n_freqs = nperseg // 2 + 1  # For real signals, this is the number of unique frequencies
        n_times = (n_samples - noverlap) // (nperseg - noverlap)
    
    # Initialize output tensor
    spectrograms = torch.zeros(
        (batch_size, n_channels, n_freqs, n_times), 
        dtype=torch.float32, 
        device=waveforms.device
    )
    
    # Calculate spectrograms using PyTorch STFT
    for b in range(batch_size):
        for c in range(n_channels):
            # Get the waveform segment
            x = waveforms[b, c]
            
            # Handle potential NaN or Inf values
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"Warning: NaN or Inf values detected in waveform at batch {b}, channel {c}")
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Apply STFT
            try:
                stft_result = torch.stft(
                    x, 
                    n_fft=nperseg, 
                    hop_length=hop_length, 
                    win_length=nperseg,
                    window=window,
                    center=False,  # Match scipy default
                    normalized=False,  # Match scipy default
                    onesided=True,  # Match scipy default for real inputs
                    return_complex=True
                )
                
                # Convert to power spectrogram (magnitude squared)
                # Note: scipy.signal.spectrogram uses |STFT|²
                power = torch.abs(stft_result)**2
                
                # Scale by window correction factor to match scipy's output
                # scipy corrects by 1/(sum(window)^2)
                scale = 1.0 / (torch.sum(window)**2)
                power *= scale
                
                # Store in output tensor
                if power.shape[0] <= n_freqs and power.shape[1] <= n_times:
                    spectrograms[b, c, :power.shape[0], :power.shape[1]] = power
                else:
                    # Truncate if larger
                    spectrograms[b, c] = power[:n_freqs, :n_times]
                    
            except Exception as e:
                print(f"Error in STFT computation at batch {b}, channel {c}: {e}")
                # Leave as zeros in case of error
    
    return spectrograms


def validate_pytorch_spectrogram():
    """
    Validate that our PyTorch spectrogram function produces 
    results consistent with scipy.signal.spectrogram
    """
    # Generate test signal
    import numpy as np
    fs = 50
    t = np.linspace(0, 10, 500)
    x = np.sin(2*np.pi*5*t) + np.random.randn(len(t))*0.1
    x = np.stack([x, x, x])  # Create 3-channel test data
    x = np.expand_dims(x, 0)  # Add batch dimension
    
    # Convert to torch tensor
    x_torch = torch.from_numpy(x).float()
    
    # Compute spectrograms
    nperseg = 256
    overlap = 0.5
    noverlap = int(nperseg * overlap)
    
    # SciPy version
    f, t, Sxx_scipy = signal.spectrogram(
        x[0, 0], fs=fs, nperseg=nperseg, noverlap=noverlap
    )
    
    # PyTorch version
    Sxx_torch = torch_spectrogram(
        x_torch, fs=fs, nperseg=nperseg, overlap=overlap
    )
    
    # Compare first channel
    Sxx_torch_np = Sxx_torch[0, 0].numpy()
    
    # Calculate relative error
    abs_diff = np.abs(Sxx_torch_np - Sxx_scipy)
    rel_error = np.mean(abs_diff / (np.abs(Sxx_scipy) + 1e-10))
    max_rel_error = np.max(abs_diff / (np.abs(Sxx_scipy) + 1e-10))
    
    print(f"Spectrogram shape - SciPy: {Sxx_scipy.shape}, PyTorch: {Sxx_torch_np.shape}")
    print(f"Mean relative error: {rel_error:.6f}")
    print(f"Max relative error: {max_rel_error:.6f}")
    
    # Values should be very close if implementation is correct
    return rel_error < 0.05  # 5% tolerance


if __name__ == "__main__":
    result = validate_pytorch_spectrogram()
    print(f"Validation {'PASSED' if result else 'FAILED'}")
