#!/usr/bin/env python
# Example script to demonstrate using QuakeXNet with scipy spectrogram

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from obspy import read

# Add the deep_learning directory to the path
sys.path.append('/Users/marinedenolle/GitHub/PNW_Seismic_Event_Classification/deep_learning')

# Import required functions and modules
from classifier import QuakeXNet, linear_detrend
from scipy_spectrogram import scipy_spectrogram_for_quakexnet
from scipy import signal

def preprocess_data(waveform, fs_original=100, fs_target=50, lowcut=1, highcut=20, 
                   taper_alpha=0.1, input_length=5000):
    """
    Preprocess raw seismic waveform data
    """
    # Apply Tukey window
    taper = signal.windows.tukey(waveform.shape[-1], alpha=taper_alpha)
    tapered = waveform * taper
    
    # Bandpass filter
    nyquist = 0.5 * fs_original
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered = np.array([signal.filtfilt(b, a, tr) for tr in tapered])
    
    # Resample to target frequency
    num_samples = int(filtered.shape[1] * fs_target / fs_original)
    resampled = np.array([signal.resample(tr, num_samples) for tr in filtered])
    
    # Cut or pad to the desired input length
    if resampled.shape[1] >= input_length:
        resampled = resampled[:, :input_length]
    else:
        # Pad with zeros if too short
        pad_width = input_length - resampled.shape[1]
        resampled = np.pad(resampled, ((0, 0), (0, pad_width)), 'constant')
    
    # Reshape for model input
    resampled = resampled.reshape(1, resampled.shape[0], resampled.shape[1])
    
    # Normalize
    mmax = np.std(np.abs(resampled))
    normalized = resampled / mmax
    
    return normalized

def apply_quakexnet(z_file, n_file, e_file, model_path):
    """
    Apply QuakeXNet to a set of 3-component seismic waveforms.
    
    Parameters:
        z_file: Path to Z-component waveform file
        n_file: Path to N-component waveform file
        e_file: Path to E-component waveform file
        model_path: Path to trained model weights
    
    Returns:
        probabilities: Class probabilities
        predicted_class: The most likely class index
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Read the MSEED files
    z_stream = read(z_file)
    n_stream = read(n_file)
    e_stream = read(e_file)
    
    # Extract waveform data in correct order (Z, N, E)
    waveform = np.array([
        z_stream[0].data,
        n_stream[0].data,
        e_stream[0].data
    ])
    
    # Preprocess data
    preprocessed = preprocess_data(waveform)
    
    # Convert to torch tensor
    tensor_input = torch.tensor(preprocessed, dtype=torch.float32).to(device)
    
    # Apply standard preprocessing steps
    # 1. Linear detrend
    detrended = linear_detrend(tensor_input)
    
    # 2. Apply Tukey window
    tukey_window = signal.windows.tukey(detrended.shape[-1], alpha=0.1)
    windowed = detrended * torch.tensor(tukey_window, dtype=torch.float32)
    
    # 3. Normalize each component
    abs_batch = torch.abs(windowed)
    normalized = windowed / (abs_batch.std(dim=-1, keepdim=True) + 1e-10)
    
    # 4. Calculate spectrograms using our scipy implementation (for compatibility)
    spectrograms = scipy_spectrogram_for_quakexnet(normalized, fs=50)
    
    # Load the model
    model = QuakeXNet(
        num_channels=3,
        num_classes=4,
        dropout_rate=0.4
    ).to(device)
    
    # Load the trained weights
    model_state = torch.load(model_path, map_location=device)
    model.load_state_dict(model_state)
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        outputs = model(spectrograms.to(device))
        probabilities = torch.softmax(outputs, dim=-1)
    
    # Get the predicted class
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    
    # Define class labels
    labels = ["earthquake", "explosion", "noise", "surface event"]
    
    # Print results
    print("\nPrediction Results:")
    print(f"Predicted class: {labels[predicted_class]}")
    print("Class probabilities:")
    for i, label in enumerate(labels):
        print(f"  {label}: {probabilities[0][i].item():.4f}")
    
    # Plot waveforms and results
    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
    
    # Plot waveforms
    components = {'Z': z_stream[0], 'N': n_stream[0], 'E': e_stream[0]}
    for i, comp in enumerate(['Z', 'N', 'E']):
        time = np.arange(len(components[comp].data)) / components[comp].stats.sampling_rate
        axes[i].plot(time, components[comp].data)
        axes[i].set_title(f"{comp} Component")
        axes[i].set_ylabel("Amplitude")
        if i < 2:  # Only for the first two subplots
            axes[i].set_xticks([])
        else:
            axes[i].set_xlabel("Time (s)")
    
    # Plot prediction probabilities
    axes[3].bar(labels, probabilities[0].cpu().numpy())
    axes[3].set_title("Class Probabilities")
    axes[3].set_ylabel("Probability")
    axes[3].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    return probabilities.cpu().numpy(), predicted_class

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Apply QuakeXNet to seismic data files')
    parser.add_argument('--zcomp', type=str, required=True,
                     help='Path to Z-component file')
    parser.add_argument('--ncomp', type=str, required=True,
                     help='Path to N-component file')
    parser.add_argument('--ecomp', type=str, required=True,
                     help='Path to E-component file')
    parser.add_argument('--model', type=str, 
                     default='/Users/marinedenolle/GitHub/PNW_Seismic_Event_Classification/deep_learning/trained_models/best_model_MyCNN_2d.pth',
                     help='Path to trained model weights')
    
    args = parser.parse_args()
    
    apply_quakexnet(args.zcomp, args.ncomp, args.ecomp, args.model)
