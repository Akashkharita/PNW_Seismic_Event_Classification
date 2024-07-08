# importing the dependencies. 
import pandas as pd
import numpy as np
import h5py
import obspy
from obspy.signal.filter import envelope
from tqdm import tqdm
import tsfel
import warnings
import argparse
import sys
import os

import seis_feature

# Get the absolute path of the directory two levels up
two_levels_up = os.path.abspath(os.path.join(os.getcwd(), "../.."))

# Append the 'src' directory located two levels up to the system path
sys.path.append(os.path.join(two_levels_up, 'src'))

from utils import apply_cosine_taper
from utils import butterworth_filter
from utils import resample_array

# Ignore all warnings
warnings.filterwarnings("ignore")

# displaying all columns from pandas dataframe
pd.set_option('display.max_columns', None)

# Argument parser
parser = argparse.ArgumentParser(description="Seismic Event Classification Script")
parser.add_argument("--low", type=float, default=1, help="Low pass frequency of the bandpass filter")
parser.add_argument("--high", type=float, default=15, help="High pass frequency of the bandpass filter")
parser.add_argument("--new_sr", type=int, default=50, help="Desired sampling rate")
parser.add_argument("--start_time", type=int, default=10, help="Start time (P-X)")
parser.add_argument("--end_time", type=int, default=30, help="End time (P+X)")
parser.add_argument("--taper_amount", type=int, default=10, help="Taper percentage")
parser.add_argument("--original_sr", type=int, default=100, help="Original sampling rate")
parser.add_argument("--num_corners", type=int, default=4, help="Number of corners for the Butterworth filter")
parser.add_argument("--class_type", type=str, required=True, choices=['earthquake', 'surface event', 'thunder', 'sonic boom', 'explosion', 'noise'], help="Type of seismic event to process")
parser.add_argument("--part", type=int, default=1, help="Refers to the part of the earthquake samples from which the features are extracted (only applicable for earthquake)")

args = parser.parse_args()

# Specifying the parameters 
low = args.low
high = args.high
new_sr = args.new_sr
start_time = args.start_time
end_time = args.end_time
taper_amount = args.taper_amount
original_sr = args.original_sr
num_corners = args.num_corners
class_type = args.class_type
part = args.part

# File paths
base_path = "/data/whd01/yiyu_data/PNWML"
output_path = '/home/ak287/PNW_Seismic_Event_Classification/extracted_features'
file_names = {
    'earthquake': ("comcat_waveforms.hdf5", "comcat_metadata.csv"),
    'surface event': ("exotic_waveforms.hdf5", "exotic_metadata.csv"),
    'thunder': ("exotic_waveforms.hdf5", "exotic_metadata.csv"),
    'sonic boom': ("exotic_waveforms.hdf5", "exotic_metadata.csv"),
    'explosion': ("comcat_waveforms.hdf5", "comcat_metadata.csv"),
    'noise': ("noise_waveforms.hdf5", "noise_metadata.csv")
}

# Function to process data
def process_data(class_type):
    file_name, metadata_name = file_names[class_type]
    file_path = os.path.join(base_path, file_name)
    metadata_path = os.path.join(base_path, metadata_name)
    
    csv_file = pd.read_csv(metadata_path)
    f = h5py.File(file_path, 'r')
    
    buckets = [csv_file['trace_name'].values[i].split('$')[0] for i in range(len(csv_file))]
    indices = [int(csv_file['trace_name'].values[i].split('$')[1].split(',')[0]) for i in range(len(csv_file))]
    source = csv_file['source_type'].values
    
    data = []
    
    if class_type == 'earthquake':
        buck_eq = np.array(buckets)[np.where(source == 'earthquake')[0]]
        ind_eq = np.array(indices)[np.where(source == 'earthquake')[0]]
        
        if part == 4:
            for i in tqdm(range(int(50000 * (part - 1)), len(ind_eq))):
                data.append(f['/data/' + buck_eq[i]][ind_eq[i], 2, int(5000 - start_time * original_sr): int(5000 + end_time * original_sr)])
        elif 0 <= part < 4:
            for i in tqdm(range(int(50000 * (part - 1)), int(50000 * part))):
                data.append(f['/data/' + buck_eq[i]][ind_eq[i], 2, int(5000 - start_time * original_sr): int(5000 + end_time * original_sr)])
        else:
            print("part cannot be less than 0 or more than 4")
            return
    else:
        buck = np.array(buckets)[np.where(source == class_type)[0]]
        ind = np.array(indices)[np.where(source == class_type)[0]]
        
        for i in tqdm(range(len(buck))):
            if class_type in ['surface event', 'thunder', 'sonic boom']:
                start_idx = 7000 - start_time * original_sr
                end_idx = 7000 + end_time * original_sr
            else:
                start_idx = 5000 - start_time * original_sr
                end_idx = 5000 + end_time * original_sr
                
            data.append(f['/data/' + buck[i]][ind[i], 2, start_idx: end_idx])
    
    data = np.array(data)
    return data

# Process and extract features
data = process_data(class_type)

if data is None:
    sys.exit("No data to process.")

tapered_data = apply_cosine_taper(data, taper_percent=taper_amount)
filtered_data = np.array(butterworth_filter(tapered_data, low, high, original_sr, num_corners, 'bandpass'))
normalized_data = filtered_data / np.max(abs(filtered_data), axis=1)[:, np.newaxis]
resampled_data = np.array([resample_array(arr, original_sr, new_sr) for arr in normalized_data])

# Feature extraction
features_df = pd.DataFrame([])
for i in tqdm(range(len(resampled_data))):
    try:
        df = seis_feature.FeatureCalculator(resampled_data[i], fs=new_sr).compute_features()
        df['serial_no'] = i + (int(50000 * (part - 1)) if class_type == 'earthquake' else 0)
        features_df = pd.concat([features_df, df])
    except:
        pass

# Save features to CSV
features_df['source'] = class_type
output_file = os.path.join(output_path, f'physical_features_{class_type}_P_{start_time}_{end_time}_F_{int(low)}_{int(high)}_{new_sr}_part_{part}.csv')
features_df.to_csv(output_file, index=False)

print(f"Features for {class_type} saved to {output_file}")
