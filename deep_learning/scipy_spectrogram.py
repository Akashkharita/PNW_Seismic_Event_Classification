import numpy as np
import scipy.signal as signal
import torch

def scipy_spectrogram_for_quakexnet(waveforms, fs=50, nperseg=256, overlap=0.5):
    """
    Extract spectrograms from waveforms using scipy.signal.spectrogram for 
    compatibility with pre-trained QuakeXNet weights.
    
    Parameters:
        waveforms: torch.Tensor of shape (batch_size, n_channels, n_samples)
        fs: sampling rate (Hz)
        nperseg: number of points per FFT segment
        overlap: fractional overlap between segments
    
    Returns:
        spectrograms: torch.Tensor of shape (batch_size, n_channels, freq_bins, time_bins)
    """
    noverlap = int(nperseg * overlap)  # Calculate overlap
    
    # Convert torch tensor to numpy array if it's not already
    if isinstance(waveforms, torch.Tensor):
        waveforms_np = waveforms.cpu().numpy()
    else:
        waveforms_np = waveforms
    
    # Get dimensions from the first spectrogram
    f, t, Sxx = signal.spectrogram(
        waveforms_np[0, 0], 
        nperseg=nperseg, 
        noverlap=noverlap, 
        fs=fs
    )
    
    # Initialize spectrogram array
    spectrograms = np.zeros((waveforms_np.shape[0], waveforms_np.shape[1], len(f), len(t)))
    
    # Compute spectrograms for all waveforms
    for i in range(waveforms_np.shape[0]):
        for j in range(waveforms_np.shape[1]):
            _, _, Sxx = signal.spectrogram(
                waveforms_np[i, j], 
                nperseg=nperseg, 
                noverlap=noverlap, 
                fs=fs
            )
            spectrograms[i, j] = Sxx
    
    # Convert back to torch tensor if input was a torch tensor
    if isinstance(waveforms, torch.Tensor):
        spectrograms = torch.tensor(spectrograms, dtype=torch.float32)
    
    return spectrograms
