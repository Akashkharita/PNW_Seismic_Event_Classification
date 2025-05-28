#!/usr/bin/env python
# Apply a trained QuakeXNet model to seismic data

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import obspy
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from scipy import signal

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add custom module paths
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'scripts'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Import the QuakeXNet model from classifier.py and our scipy spectrogram function
from classifier import QuakeXNet, linear_detrend
from scipy_spectrogram import scipy_spectrogram_for_quakexnet


def preprocess_data(waveform, fs_original=100, fs_target=50, lowcut=1, highcut=20, 
                    taper_alpha=0.1, input_length=5000):
    """
    Preprocess raw seismic waveform data:
    - Apply Tukey window
    - Bandpass filter
    - Resample to target frequency
    - Normalize by standard deviation
    
    Parameters:
        waveform: numpy array of shape (n_channels, n_samples)
        fs_original: original sampling rate of the data
        fs_target: target sampling rate
        lowcut: low frequency cutoff for bandpass filter
        highcut: high frequency cutoff for bandpass filter
        taper_alpha: alpha value for Tukey window
        input_length: number of samples to use (default 5000)
    
    Returns:
        preprocessed: normalized, filtered waveform ready for model input
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


def extract_spectrograms(waveforms, fs=50, nperseg=256, overlap=0.5):
    """
    Extract spectrograms from waveforms.
    
    Parameters:
        waveforms: numpy array of shape (batch_size, n_channels, n_samples)
        fs: sampling rate (Hz)
        nperseg: number of points per FFT segment
        overlap: fractional overlap between segments
    
    Returns:
        spectrograms: numpy array of shape (batch_size, n_channels, freq_bins, time_bins)
    """
    noverlap = int(nperseg * overlap)
    
    # Get dimensions from the first spectrogram
    f, t, Sxx = signal.spectrogram(waveforms[0, 0], nperseg=nperseg, 
                                  noverlap=noverlap, fs=fs)
    
    # Initialize spectrogram array
    spectrograms = np.zeros((waveforms.shape[0], waveforms.shape[1], len(f), len(t)))
    
    # Compute spectrograms for all waveforms
    for i in range(waveforms.shape[0]):
        for j in range(waveforms.shape[1]):
            _, _, Sxx = signal.spectrogram(waveforms[i, j], nperseg=nperseg, 
                                          noverlap=noverlap, fs=fs)
            spectrograms[i, j] = Sxx
    
    return spectrograms


def load_quakexnet_model(model_path, device='cpu'):
    """
    Load the trained QuakeXNet model.
    
    Parameters:
        model_path: path to the trained model weights
        device: 'cpu' or 'cuda'
    
    Returns:
        model: loaded and ready-to-use QuakeXNet model
    """
    # Initialize the model
    model = QuakeXNet(
        num_channels=3,
        num_classes=4,
        dropout_rate=0.4
    ).to(device)
    
    # Load the trained weights
    model_state = torch.load(model_path, map_location=device)
    model.load_state_dict(model_state)
    
    # Set to evaluation mode
    model.eval()
    
    return model


def predict_with_seisbench_quakexnet(waveform, model, device='cpu'):
    """
    Make a prediction using the loaded QuakeXNet model on a seismic waveform.
    
    Parameters:
        waveform: preprocessed waveform data (numpy array)
        model: loaded QuakeXNet model
        device: 'cpu' or 'cuda'
    
    Returns:
        predictions: class probabilities
        predicted_class: the most likely class index
    """
    # Convert to torch tensor
    inputs = torch.tensor(waveform, dtype=torch.float32).to(device)
    
    # Get predictions (no need to call no_grad here as we'll do it at a higher level)
    argdict = {"sampling_rate": 50}
    
    # Apply preprocessing
    batch = inputs.cpu()
    
    # Detrend each component
    batch = linear_detrend(batch)
    
    # Create a Tukey window using scipy
    tukey_window = signal.windows.tukey(batch.shape[-1], alpha=0.1)
    
    # Apply the Tukey window to the batch
    batch *= tukey_window  # Broadcasting over last axis
    
    # Normalize each component by the standard deviation of their absolute values
    batch_abs = torch.abs(batch)
    batch /= batch_abs.std(dim=-1, keepdim=True) + 1e-10  # Avoid division by zero
    
    # Convert the processed waveforms to spectrograms using scipy spectrogram
    spec = scipy_spectrogram_for_quakexnet(batch, fs=argdict["sampling_rate"])
    
    # Forward pass - bypass the model's annotate_batch_pre
    with torch.no_grad():
        outputs = model(spec.to(device))
        probabilities = model.annotate_batch_post(outputs, None, argdict)
    
    # Get the predicted class
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    
    return probabilities.cpu().numpy(), predicted_class


def fetch_seismic_data(network, station, channel, starttime, endtime, client_name='IRIS'):
    """
    Fetch seismic data from FDSN web services.
    
    Parameters:
        network: network code
        station: station code
        channel: channel code (will append wildcard for all components)
        starttime: start time (UTCDateTime)
        endtime: end time (UTCDateTime)
        client_name: FDSN client name
    
    Returns:
        stream: ObsPy Stream object
    """
    client = Client(client_name)
    try:
        stream = client.get_waveforms(
            network=network, 
            station=station, 
            channel=f"{channel}?", 
            location="--", 
            starttime=starttime, 
            endtime=endtime
        )
        return stream
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Apply QuakeXNet to seismic data')
    parser.add_argument('--model', type=str, default='trained_models/best_model_MyCNN_2d.pth',
                        help='Path to trained model weights')
    parser.add_argument('--network', type=str, required=True,
                        help='Network code (e.g., UW)')
    parser.add_argument('--station', type=str, required=True,
                        help='Station code (e.g., RCM)')
    parser.add_argument('--channel', type=str, default='BH',
                        help='Channel code prefix (e.g., BH for BHZ,BHN,BHE)')
    parser.add_argument('--starttime', type=str, required=True,
                        help='Start time in ISO format (e.g., 2020-01-01T00:00:00)')
    parser.add_argument('--duration', type=float, default=100.0,
                        help='Duration in seconds')
    parser.add_argument('--plot', action='store_true',
                        help='Plot waveforms and spectrograms')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu or cuda)')
    args = parser.parse_args()
    
    # Convert start time to UTCDateTime
    starttime = UTCDateTime(args.starttime)
    endtime = starttime + args.duration
    
    print(f"Fetching data for {args.network}.{args.station}.{args.channel}* from {starttime} to {endtime}")
    
    # Fetch seismic data
    stream = fetch_seismic_data(
        network=args.network,
        station=args.station,
        channel=args.channel,
        starttime=starttime,
        endtime=endtime
    )
    
    if stream is None or len(stream) < 3:
        print("Failed to fetch data or insufficient components")
        return
    
    # Find the 3-component channels (Z, N, E)
    components = {}
    for tr in stream:
        if tr.stats.channel[-1] == 'Z':
            components['Z'] = tr
        elif tr.stats.channel[-1] == 'N':
            components['N'] = tr
        elif tr.stats.channel[-1] == 'E':
            components['E'] = tr
    
    if len(components) < 3:
        print("Missing required components (Z, N, E)")
        return
    
    # Extract waveform data in correct order (Z, N, E)
    waveform = np.array([
        components['Z'].data,
        components['N'].data,
        components['E'].data
    ])
    
    # Preprocess data
    preprocessed = preprocess_data(waveform)
    
    # Load the QuakeXNet model
    model = load_quakexnet_model(args.model, device=args.device)
    
    # Make predictions
    probabilities, predicted_class = predict_with_seisbench_quakexnet(
        preprocessed, model, device=args.device
    )
    
    # Define class labels
    labels = ["earthquake", "explosion", "noise", "surface event"]
    
    # Print results
    print("\nPrediction Results:")
    print(f"Predicted class: {labels[predicted_class]}")
    print("Class probabilities:")
    for i, label in enumerate(labels):
        print(f"  {label}: {probabilities[0][i]:.4f}")
    
    # Plot if requested
    if args.plot:
        # Create figure with subplots
        fig, axes = plt.subplots(4, 1, figsize=(10, 12))
        
        # Plot waveforms
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
        axes[3].bar(labels, probabilities[0])
        axes[3].set_title("Class Probabilities")
        axes[3].set_ylabel("Probability")
        axes[3].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.show()
    
if __name__ == "__main__":
    main()
