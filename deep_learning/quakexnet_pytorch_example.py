#!/usr/bin/env python
# Quick example demonstrating the use of the PyTorch spectrogram implementation with QuakeXNet

import os
import sys
import numpy as np
import torch
from obspy import read
import matplotlib.pyplot as plt

# Add the deep_learning directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the QuakeXNet model
from classifier import QuakeXNet


def process_waveform(waveform_z, waveform_n, waveform_e, model_path="trained_models/best_model_MyCNN_2d.pth"):
    """
    Process a single 3-component waveform with QuakeXNet.
    
    Args:
        waveform_z, waveform_n, waveform_e: NumPy arrays for Z, N, E components
        model_path: Path to the trained model weights
        
    Returns:
        prediction: Class prediction (string)
        probabilities: Probability for each class
    """
    # Preprocess waveforms
    # 1. Stack components (Z, N, E)
    waveform = np.stack([waveform_z, waveform_n, waveform_e])
    
    # 2. Apply preprocessing (normalize)
    waveform = waveform / np.std(np.abs(waveform))
    
    # 3. Reshape to batch format (1, 3, samples)
    waveform = waveform.reshape(1, waveform.shape[0], waveform.shape[1])
    
    # Convert to PyTorch tensor
    inputs = torch.tensor(waveform, dtype=torch.float32)
    
    # Load the model
    model = QuakeXNet(
        num_channels=3,
        num_classes=4,
        dropout_rate=0.4
    )
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    # Run prediction
    with torch.no_grad():
        # Preprocess using the model's annotate_batch_pre method
        # (detrending, windowing, and spectrogram calculation)
        processed_inputs = model.annotate_batch_pre(inputs, {"sampling_rate": 50})
        
        # Forward pass
        outputs = model(processed_inputs)
        probabilities = model.annotate_batch_post(outputs, None, {"sampling_rate": 50})
    
    # Get the predicted class
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    
    # Class labels
    labels = ["earthquake", "explosion", "noise", "surface event"]
    
    return labels[predicted_class], probabilities.cpu().numpy()[0]


def process_mseed_files(z_file, n_file, e_file, model_path="trained_models/best_model_MyCNN_2d.pth"):
    """
    Process MSEED files and predict the event class.
    
    Args:
        z_file, n_file, e_file: Paths to MSEED files for Z, N, E components
        model_path: Path to the trained model weights
        
    Returns:
        prediction: Class prediction (string)
        probabilities: Probability for each class
    """
    # Read the MSEED files
    z_stream = read(z_file)[0]
    n_stream = read(n_file)[0]
    e_stream = read(e_file)[0]
    
    # Extract the waveform data
    z_data = z_stream.data
    n_data = n_stream.data
    e_data = e_stream.data
    
    # Process and get prediction
    prediction, probabilities = process_waveform(
        z_data, n_data, e_data, model_path=model_path
    )
    
    # Print results
    labels = ["earthquake", "explosion", "noise", "surface event"]
    print("\nPrediction Results:")
    print(f"Predicted class: {prediction}")
    print("Class probabilities:")
    for i, label in enumerate(labels):
        print(f"  {label}: {probabilities[i]:.4f}")
    
    # Plot results
    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
    
    # Plot waveforms
    for i, (comp, data, stream) in enumerate(zip(['Z', 'N', 'E'], 
                                             [z_data, n_data, e_data],
                                             [z_stream, n_stream, e_stream])):
        time = np.arange(len(data)) / stream.stats.sampling_rate
        axes[i].plot(time, data)
        axes[i].set_title(f"{comp} Component")
        axes[i].set_ylabel("Amplitude")
        if i < 2:
            axes[i].set_xticks([])
        else:
            axes[i].set_xlabel("Time (s)")
    
    # Plot prediction probabilities
    axes[3].bar(labels, probabilities)
    axes[3].set_title("Class Probabilities")
    axes[3].set_ylabel("Probability")
    axes[3].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    return prediction, probabilities


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply QuakeXNet to seismic data with PyTorch spectrograms")
    parser.add_argument("--z_file", type=str, help="Path to Z component MSEED file")
    parser.add_argument("--n_file", type=str, help="Path to N component MSEED file")
    parser.add_argument("--e_file", type=str, help="Path to E component MSEED file")
    parser.add_argument("--model", type=str, default="trained_models/best_model_MyCNN_2d.pth",
                      help="Path to trained model weights")
    
    args = parser.parse_args()
    
    if args.z_file and args.n_file and args.e_file:
        process_mseed_files(args.z_file, args.n_file, args.e_file, model_path=args.model)
    else:
        print("Please provide paths to Z, N, and E component MSEED files using --z_file, --n_file, and --e_file")
        print("Example usage:")
        print("python quakexnet_pytorch_example.py --z_file data/example_Z.mseed --n_file data/example_N.mseed --e_file data/example_E.mseed")
