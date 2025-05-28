#!/usr/bin/env python
# Test and demo script for applying a trained QuakeXNet model to seismic data
# with the updated PyTorch spectrogram implementation

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from obspy import read
from tqdm import tqdm

# Add the deep_learning directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the QuakeXNet model and preprocessing functions
from classifier import QuakeXNet, linear_detrend


def preprocess_data(waveform, fs_original=100, fs_target=50, lowcut=1, highcut=20, 
                   taper_alpha=0.1, input_length=5000):
    """
    Preprocess raw seismic waveform data.
    """
    from scipy import signal
    
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


def load_quakexnet_model(model_path, device='cpu'):
    """
    Load the trained QuakeXNet model.
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


def predict_with_quakexnet(waveform, model, device='cpu'):
    """
    Make a prediction using the loaded QuakeXNet model on a seismic waveform.
    """
    # Convert to torch tensor
    inputs = torch.tensor(waveform, dtype=torch.float32).to(device)
    
    # Predict using our model with the PyTorch spectrogram
    argdict = {"sampling_rate": 50}
    
    # Preprocess using the model's annotate_batch_pre method
    processed_inputs = model.annotate_batch_pre(inputs, argdict)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(processed_inputs)
        probabilities = model.annotate_batch_post(outputs, None, argdict)
    
    # Get the predicted class
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    
    return probabilities.cpu().numpy(), predicted_class


def process_local_mseed(z_file, n_file, e_file, model_path, device='cpu', plot=True):
    """
    Process local MSEED files with QuakeXNet and display results.
    """
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

    # Load the QuakeXNet model
    model = load_quakexnet_model(model_path, device=device)

    # Make predictions
    probabilities, predicted_class = predict_with_quakexnet(
        preprocessed, model, device=device
    )

    # Define class labels
    labels = ["earthquake", "explosion", "noise", "surface event"]

    # Print results
    print("\nPrediction Results:")
    print(f"Predicted class: {labels[predicted_class]}")
    print("Class probabilities:")
    for i, label in enumerate(labels):
        print(f"  {label}: {probabilities[0][i]:.4f}")

    # Plot waveforms and results
    if plot:
        fig, axes = plt.subplots(4, 1, figsize=(10, 12))

        # Plot waveforms
        components = ['Z', 'N', 'E']
        streams = [z_stream[0], n_stream[0], e_stream[0]]

        for i, (comp, stream) in enumerate(zip(components, streams)):
            time = np.arange(len(stream.data)) / stream.stats.sampling_rate
            axes[i].plot(time, stream.data)
            axes[i].set_title(f"{comp} Component")
            axes[i].set_ylabel("Amplitude")
            if i < 2:
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
        
    return labels[predicted_class], probabilities[0]


def batch_process_directory(base_dir, event_ids, model_path, device='cpu', output_file='classification_results.csv'):
    """
    Process multiple event_ids in a batch.
    
    Args:
        base_dir: Base directory containing MSEED files
        event_ids: List of event IDs to process
        model_path: Path to the trained model
        device: 'cpu' or 'cuda'
        output_file: File to save classification results
    """
    import os
    import pandas as pd
    from glob import glob
    
    results = []
    
    for event_id in tqdm(event_ids, desc="Processing events"):
        # Find files for this event ID
        pattern = os.path.join(base_dir, f"{event_id}_*.mseed")
        files = glob(pattern)
        
        # Group files by station
        stations = {}
        for file in files:
            filename = os.path.basename(file)
            parts = filename.split('_')[1].split('.')
            network = parts[0]
            station = parts[1]
            channel = parts[2]
            
            if station not in stations:
                stations[station] = {'Z': None, 'N': None, 'E': None, 'network': network}
            
            # Assign file to the correct component
            if channel.endswith('Z'):
                stations[station]['Z'] = file
            elif channel.endswith('N'):
                stations[station]['N'] = file
            elif channel.endswith('E'):
                stations[station]['E'] = file
        
        # Process each station with complete 3-component data
        for station, components in stations.items():
            # Skip if missing any component
            if not all([components['Z'], components['N'], components['E']]):
                continue
            
            try:
                # Classify the event
                prediction, probs = process_local_mseed(
                    components['Z'],
                    components['N'],
                    components['E'],
                    model_path,
                    device=device,
                    plot=False
                )
                
                # Record the results
                results.append({
                    'event_id': event_id,
                    'network': components['network'],
                    'station': station,
                    'prediction': prediction,
                    'earthquake_prob': probs[0],
                    'explosion_prob': probs[1],
                    'noise_prob': probs[2],
                    'surface_event_prob': probs[3]
                })
            except Exception as e:
                print(f"Error processing {event_id} at station {station}: {e}")
    
    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    else:
        print("No valid results to save")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Apply QuakeXNet to seismic data')
    parser.add_argument('--z_file', type=str, help='Path to Z component MSEED file')
    parser.add_argument('--n_file', type=str, help='Path to N component MSEED file')
    parser.add_argument('--e_file', type=str, help='Path to E component MSEED file')
    parser.add_argument('--model', type=str, default='trained_models/best_model_MyCNN_2d.pth',
                      help='Path to trained model weights')
    parser.add_argument('--batch', action='store_true', help='Process batch of events')
    parser.add_argument('--event_id', type=str, help='Single event ID to process (for batch mode)')
    parser.add_argument('--data_dir', type=str, default='../data/binary_classification_waveforms/',
                      help='Directory containing MSEED files (for batch mode)')
    parser.add_argument('--output', type=str, default='classification_results.csv',
                      help='Output CSV file (for batch mode)')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu or cuda)')
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch processing mode
        data_dir = args.data_dir
        
        if args.event_id:
            # Process a single event ID
            event_ids = [args.event_id]
        else:
            # Process all unique event IDs in the directory
            import os
            import re
            from glob import glob
            
            files = glob(os.path.join(data_dir, "*.mseed"))
            event_ids = set()
            
            for file in files:
                filename = os.path.basename(file)
                match = re.match(r'(uw\d+)_', filename)
                if match:
                    event_ids.add(match.group(1))
            
            event_ids = sorted(list(event_ids))
            print(f"Found {len(event_ids)} unique event IDs")
        
        batch_process_directory(
            data_dir,
            event_ids,
            args.model,
            device=args.device,
            output_file=args.output
        )
    
    elif args.z_file and args.n_file and args.e_file:
        # Single file processing mode
        process_local_mseed(
            args.z_file,
            args.n_file,
            args.e_file,
            args.model,
            device=args.device
        )
    
    else:
        # If no files provided, use an example from the dataset
        data_dir = "../data/binary_classification_waveforms/"
        
        # Find a sample event with 3 components
        from glob import glob
        import os
        
        sample_events = {}
        
        for file in glob(os.path.join(data_dir, "*.mseed")):
            filename = os.path.basename(file)
            parts = filename.split('_')
            
            if len(parts) >= 2:
                event_id = parts[0]
                components = filename.split('.')[2]
                
                if event_id not in sample_events:
                    sample_events[event_id] = {'Z': None, 'N': None, 'E': None}
                
                if components.endswith('Z'):
                    sample_events[event_id]['Z'] = file
                elif components.endswith('N'):
                    sample_events[event_id]['N'] = file
                elif components.endswith('E'):
                    sample_events[event_id]['E'] = file
        
        # Find the first event with all three components
        for event_id, components in sample_events.items():
            if all(components.values()):
                print(f"Using sample event {event_id}")
                process_local_mseed(
                    components['Z'],
                    components['N'],
                    components['E'],
                    args.model,
                    device=args.device
                )
                break
        else:
            print("No suitable sample event found with all three components (Z, N, E).")
            print("Please provide input files using --z_file, --n_file, and --e_file options.")
