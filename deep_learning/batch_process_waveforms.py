#!/usr/bin/env python
# Batch process multiple waveform files using the QuakeXNet model

import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import obspy
import torch

# Add project directory to path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_dir not in sys.path:
    sys.path.append(project_dir)

# Import functions from apply_quakexnet
from deep_learning.apply_quakexnet import (
    load_quakexnet_model,
    preprocess_data,
    predict_with_seisbench_quakexnet
)

def process_waveform_file(filepath, model, device='cpu'):
    """
    Process a single waveform file with the QuakeXNet model.
    
    Parameters:
        filepath: path to waveform file (mseed)
        model: loaded QuakeXNet model
        device: 'cpu' or 'cuda'
    
    Returns:
        result_dict: dictionary with results
    """
    try:
        # Read the file
        stream = obspy.read(filepath)
        
        # Extract event ID from filename
        event_id = os.path.basename(filepath).split('_')[0]
        
        # Ensure we have 3 components
        if len(stream) < 3:
            print(f"Warning: {filepath} does not have enough components")
            return None
        
        # Find Z, N, E components
        comp_z = next((tr for tr in stream if tr.stats.channel.endswith('Z')), None)
        comp_n = next((tr for tr in stream if tr.stats.channel.endswith('N') or tr.stats.channel.endswith('1')), None)
        comp_e = next((tr for tr in stream if tr.stats.channel.endswith('E') or tr.stats.channel.endswith('2')), None)
        
        if not (comp_z and comp_n and comp_e):
            print(f"Warning: {filepath} missing required components")
            return None
        
        # Extract waveform data and ensure equal length
        min_length = min(len(comp_z.data), len(comp_n.data), len(comp_e.data))
        waveform = np.array([
            comp_z.data[:min_length],
            comp_n.data[:min_length],
            comp_e.data[:min_length]
        ])
        
        # Preprocess data
        preprocessed = preprocess_data(waveform, 
                                      fs_original=comp_z.stats.sampling_rate,
                                      fs_target=50)
        
        # Make predictions
        probabilities, predicted_class = predict_with_seisbench_quakexnet(
            preprocessed, model, device=device
        )
        
        # Create result dictionary
        result = {
            'event_id': event_id,
            'filepath': filepath,
            'station': comp_z.stats.station,
            'network': comp_z.stats.network,
            'starttime': comp_z.stats.starttime,
            'predicted_class': predicted_class,
            'prob_earthquake': probabilities[0][0],
            'prob_explosion': probabilities[0][1],
            'prob_noise': probabilities[0][2],
            'prob_surface': probabilities[0][3]
        }
        
        return result
    
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def main():
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Batch process waveform files with QuakeXNet')
    parser.add_argument('--model', type=str, default='deep_learning/trained_models/best_model_QuakeXNet_2d.pth',
                        help='Path to trained model weights')
    parser.add_argument('--input_dir', type=str, default='data/binary_classification_waveforms',
                        help='Directory containing waveform files')
    parser.add_argument('--output', type=str, default='results_quakexnet.csv',
                        help='Output CSV file for results')
    parser.add_argument('--max_files', type=int, default=100,
                        help='Maximum number of files to process')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu or cuda)')
    args = parser.parse_args()
    
    # Set device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        device = 'cpu'
    
    # Find waveform files
    waveform_files = glob.glob(os.path.join(args.input_dir, '*.mseed'))
    if not waveform_files:
        print(f"No waveform files found in {args.input_dir}")
        return
    
    print(f"Found {len(waveform_files)} waveform files")
    
    # Limit the number of files if specified
    if args.max_files > 0 and len(waveform_files) > args.max_files:
        waveform_files = waveform_files[:args.max_files]
        print(f"Processing first {args.max_files} files")
    
    # Load the model
    print(f"Loading model from {args.model}")
    model = load_quakexnet_model(args.model, device=device)
    
    # Process files
    results = []
    for filepath in tqdm(waveform_files):
        result = process_waveform_file(filepath, model, device=device)
        if result:
            results.append(result)
    
    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"Saved results for {len(df)} files to {args.output}")
        
        # Print summary
        class_counts = df['predicted_class'].value_counts()
        print("\nPrediction Summary:")
        class_names = ['earthquake', 'explosion', 'noise', 'surface event']
        for i, name in enumerate(class_names):
            count = class_counts.get(i, 0)
            print(f"  {name}: {count} files ({count/len(df)*100:.1f}%)")
    else:
        print("No results to save")

if __name__ == '__main__':
    main()
