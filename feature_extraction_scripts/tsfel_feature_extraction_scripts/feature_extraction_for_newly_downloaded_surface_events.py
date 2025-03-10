# importing the dependencies. 
import pandas as pd
import numpy as np
import h5py
import obspy
from obspy.signal.filter import envelope
from tqdm import tqdm
import tsfel
from tsfel import time_series_features_extractor

import warnings
import argparse
import sys
import os



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


from glob import glob


# Load dataset
new_su_cat = pd.read_csv(glob('../../data/additional_surface_events_good_snr.csv')[0], index_col=0)



# Constants
TAPER_AMOUNT = 10
LOW_FREQ, HIGH_FREQ = 1, 10
ORIGINAL_SR, NEW_SR = 100, 50
NUM_CORNERS = 4
START, END = 50, 100
ARRIVAL = 7000

cfg_file = tsfel.get_features_by_domain()
# Initialize lists for storing results
data, evids = [], []

# Process each event
for _, row in tqdm(new_su_cat.iterrows(), total=len(new_su_cat)):
    try:
        new_file_path = row['file_path'].replace('HN', 'HZ')
        evid = f"{row['event_id']}_{row['station_network_code']}.{row['station_code']}"

        # Read and preprocess waveform
        stream = obspy.read('../' + new_file_path)
        stream.detrend('linear')
        stream.resample(ORIGINAL_SR)

        tr = stream[0].data[int(ARRIVAL - START * ORIGINAL_SR) : int(ARRIVAL + END * ORIGINAL_SR)]
        data.append(tr)
        evids.append(evid)
    except Exception as e:
        print(f"Skipping event {row['event_id']} due to error: {e}")

# Convert to NumPy array
data = np.array(data)
print(f"Data shape: {data.shape}")

# Signal processing
tapered_data = apply_cosine_taper(data, taper_percent=TAPER_AMOUNT)
filtered_data = np.array(butterworth_filter(tapered_data, LOW_FREQ, HIGH_FREQ, ORIGINAL_SR, NUM_CORNERS, 'bandpass'))
normalized_data = filtered_data / np.max(abs(filtered_data), axis=1, keepdims=True)
resampled_data = np.array([resample_array(arr, ORIGINAL_SR, NEW_SR) for arr in normalized_data])

# Feature extraction
feature_list = []
for i in tqdm(range(len(resampled_data))):
    try:
        features = time_series_features_extractor(cfg_file, resampled_data[i], fs= NEW_SR, verbose = 0)
        features['new_event_id'] = evids[i]
        feature_list.append(features)
    except Exception as e:
        print(f"Feature extraction failed for {evids[i]}: {e}")

# Combine features into a DataFrame
features_df = pd.concat(feature_list, ignore_index=True)
print(f"Feature DataFrame shape: {features_df.shape}")

low_freq = str(LOW_FREQ).replace('.','')
features_df.to_csv(f'../../extracted_features/tsfel_features_new_surface event_P_{START}_{END}_F_'+low_freq+f'_{HIGH_FREQ}_{NEW_SR}_part_1.csv')



# Constants
TAPER_AMOUNT = 10
LOW_FREQ, HIGH_FREQ = 0.5, 15
ORIGINAL_SR, NEW_SR = 100, 50
NUM_CORNERS = 4
START, END = 50, 100
ARRIVAL = 7000

cfg_file = tsfel.get_features_by_domain()
# Initialize lists for storing results
data, evids = [], []

# Process each event
for _, row in tqdm(new_su_cat.iterrows(), total=len(new_su_cat)):
    try:
        new_file_path = row['file_path'].replace('HN', 'HZ')
        evid = f"{row['event_id']}_{row['station_network_code']}.{row['station_code']}"

        # Read and preprocess waveform
        stream = obspy.read('../' + new_file_path)
        stream.detrend('linear')
        stream.resample(ORIGINAL_SR)

        tr = stream[0].data[int(ARRIVAL - START * ORIGINAL_SR) : int(ARRIVAL + END * ORIGINAL_SR)]
        data.append(tr)
        evids.append(evid)
    except Exception as e:
        print(f"Skipping event {row['event_id']} due to error: {e}")

# Convert to NumPy array
data = np.array(data)
print(f"Data shape: {data.shape}")

# Signal processing
tapered_data = apply_cosine_taper(data, taper_percent=TAPER_AMOUNT)
filtered_data = np.array(butterworth_filter(tapered_data, LOW_FREQ, HIGH_FREQ, ORIGINAL_SR, NUM_CORNERS, 'bandpass'))
normalized_data = filtered_data / np.max(abs(filtered_data), axis=1, keepdims=True)
resampled_data = np.array([resample_array(arr, ORIGINAL_SR, NEW_SR) for arr in normalized_data])

# Feature extraction
feature_list = []
for i in tqdm(range(len(resampled_data))):
    try:
        features = time_series_features_extractor(cfg_file, resampled_data[i], fs= NEW_SR, verbose = 0)
        features['new_event_id'] = evids[i]
        feature_list.append(features)
    except Exception as e:
        print(f"Feature extraction failed for {evids[i]}: {e}")

# Combine features into a DataFrame
features_df = pd.concat(feature_list, ignore_index=True)
print(f"Feature DataFrame shape: {features_df.shape}")

low_freq = str(LOW_FREQ).replace('.','')
features_df.to_csv(f'../../extracted_features/tsfel_features_new_surface event_P_{START}_{END}_F_'+low_freq+f'_{HIGH_FREQ}_{NEW_SR}_part_1.csv')











# Constants
TAPER_AMOUNT = 10
LOW_FREQ, HIGH_FREQ = 1, 10
ORIGINAL_SR, NEW_SR = 100, 50
NUM_CORNERS = 4
START, END = 10, 100
ARRIVAL = 7000

cfg_file = tsfel.get_features_by_domain()
# Initialize lists for storing results
data, evids = [], []

# Process each event
for _, row in tqdm(new_su_cat.iterrows(), total=len(new_su_cat)):
    try:
        new_file_path = row['file_path'].replace('HN', 'HZ')
        evid = f"{row['event_id']}_{row['station_network_code']}.{row['station_code']}"

        # Read and preprocess waveform
        stream = obspy.read('../' + new_file_path)
        stream.detrend('linear')
        stream.resample(ORIGINAL_SR)

        tr = stream[0].data[int(ARRIVAL - START * ORIGINAL_SR) : int(ARRIVAL + END * ORIGINAL_SR)]
        data.append(tr)
        evids.append(evid)
    except Exception as e:
        print(f"Skipping event {row['event_id']} due to error: {e}")

# Convert to NumPy array
data = np.array(data)
print(f"Data shape: {data.shape}")

# Signal processing
tapered_data = apply_cosine_taper(data, taper_percent=TAPER_AMOUNT)
filtered_data = np.array(butterworth_filter(tapered_data, LOW_FREQ, HIGH_FREQ, ORIGINAL_SR, NUM_CORNERS, 'bandpass'))
normalized_data = filtered_data / np.max(abs(filtered_data), axis=1, keepdims=True)
resampled_data = np.array([resample_array(arr, ORIGINAL_SR, NEW_SR) for arr in normalized_data])

# Feature extraction
feature_list = []
for i in tqdm(range(len(resampled_data))):
    try:
        features = time_series_features_extractor(cfg_file, resampled_data[i], fs= NEW_SR, verbose = 0)
        features['new_event_id'] = evids[i]
        feature_list.append(features)
    except Exception as e:
        print(f"Feature extraction failed for {evids[i]}: {e}")

# Combine features into a DataFrame
features_df = pd.concat(feature_list, ignore_index=True)
print(f"Feature DataFrame shape: {features_df.shape}")

low_freq = str(LOW_FREQ).replace('.','')
features_df.to_csv(f'../../extracted_features/tsfel_features_new_surface event_P_{START}_{END}_F_'+low_freq+f'_{HIGH_FREQ}_{NEW_SR}_part_1.csv')



# Constants
TAPER_AMOUNT = 10
LOW_FREQ, HIGH_FREQ = 0.5, 15
ORIGINAL_SR, NEW_SR = 100, 50
NUM_CORNERS = 4
START, END = 10, 100
ARRIVAL = 7000

cfg_file = tsfel.get_features_by_domain()
# Initialize lists for storing results
data, evids = [], []

# Process each event
for _, row in tqdm(new_su_cat.iterrows(), total=len(new_su_cat)):
    try:
        new_file_path = row['file_path'].replace('HN', 'HZ')
        evid = f"{row['event_id']}_{row['station_network_code']}.{row['station_code']}"

        # Read and preprocess waveform
        stream = obspy.read('../' + new_file_path)
        stream.detrend('linear')
        stream.resample(ORIGINAL_SR)

        tr = stream[0].data[int(ARRIVAL - START * ORIGINAL_SR) : int(ARRIVAL + END * ORIGINAL_SR)]
        data.append(tr)
        evids.append(evid)
    except Exception as e:
        print(f"Skipping event {row['event_id']} due to error: {e}")

# Convert to NumPy array
data = np.array(data)
print(f"Data shape: {data.shape}")

# Signal processing
tapered_data = apply_cosine_taper(data, taper_percent=TAPER_AMOUNT)
filtered_data = np.array(butterworth_filter(tapered_data, LOW_FREQ, HIGH_FREQ, ORIGINAL_SR, NUM_CORNERS, 'bandpass'))
normalized_data = filtered_data / np.max(abs(filtered_data), axis=1, keepdims=True)
resampled_data = np.array([resample_array(arr, ORIGINAL_SR, NEW_SR) for arr in normalized_data])

# Feature extraction
feature_list = []
for i in tqdm(range(len(resampled_data))):
    try:
        features = time_series_features_extractor(cfg_file, resampled_data[i], fs= NEW_SR, verbose = 0)
        features['new_event_id'] = evids[i]
        feature_list.append(features)
    except Exception as e:
        print(f"Feature extraction failed for {evids[i]}: {e}")

# Combine features into a DataFrame
features_df = pd.concat(feature_list, ignore_index=True)
print(f"Feature DataFrame shape: {features_df.shape}")

low_freq = str(LOW_FREQ).replace('.','')
features_df.to_csv(f'../../extracted_features/tsfel_features_new_surface event_P_{START}_{END}_F_'+low_freq+f'_{HIGH_FREQ}_{NEW_SR}_part_1.csv')



# Constants
TAPER_AMOUNT = 10
LOW_FREQ, HIGH_FREQ = 0.5, 15
ORIGINAL_SR, NEW_SR = 100, 50
NUM_CORNERS = 4
START, END = 10, 30
ARRIVAL = 7000

cfg_file = tsfel.get_features_by_domain()
# Initialize lists for storing results
data, evids = [], []

# Process each event
for _, row in tqdm(new_su_cat.iterrows(), total=len(new_su_cat)):
    try:
        new_file_path = row['file_path'].replace('HN', 'HZ')
        evid = f"{row['event_id']}_{row['station_network_code']}.{row['station_code']}"

        # Read and preprocess waveform
        stream = obspy.read('../' + new_file_path)
        stream.detrend('linear')
        stream.resample(ORIGINAL_SR)

        tr = stream[0].data[int(ARRIVAL - START * ORIGINAL_SR) : int(ARRIVAL + END * ORIGINAL_SR)]
        data.append(tr)
        evids.append(evid)
    except Exception as e:
        print(f"Skipping event {row['event_id']} due to error: {e}")

# Convert to NumPy array
data = np.array(data)
print(f"Data shape: {data.shape}")

# Signal processing
tapered_data = apply_cosine_taper(data, taper_percent=TAPER_AMOUNT)
filtered_data = np.array(butterworth_filter(tapered_data, LOW_FREQ, HIGH_FREQ, ORIGINAL_SR, NUM_CORNERS, 'bandpass'))
normalized_data = filtered_data / np.max(abs(filtered_data), axis=1, keepdims=True)
resampled_data = np.array([resample_array(arr, ORIGINAL_SR, NEW_SR) for arr in normalized_data])

# Feature extraction
feature_list = []
for i in tqdm(range(len(resampled_data))):
    try:
        features = time_series_features_extractor(cfg_file, resampled_data[i], fs= NEW_SR, verbose = 0)
        features['new_event_id'] = evids[i]
        feature_list.append(features)
    except Exception as e:
        print(f"Feature extraction failed for {evids[i]}: {e}")

# Combine features into a DataFrame
features_df = pd.concat(feature_list, ignore_index=True)
print(f"Feature DataFrame shape: {features_df.shape}")

low_freq = str(LOW_FREQ).replace('.','')
features_df.to_csv(f'../../extracted_features/tsfel_features_new_surface event_P_{START}_{END}_F_'+low_freq+f'_{HIGH_FREQ}_{NEW_SR}_part_1.csv')




# Constants
TAPER_AMOUNT = 10
LOW_FREQ, HIGH_FREQ = 1, 10
ORIGINAL_SR, NEW_SR = 100, 50
NUM_CORNERS = 4
START, END = 10, 30
ARRIVAL = 7000

cfg_file = tsfel.get_features_by_domain()
# Initialize lists for storing results
data, evids = [], []

# Process each event
for _, row in tqdm(new_su_cat.iterrows(), total=len(new_su_cat)):
    try:
        new_file_path = row['file_path'].replace('HN', 'HZ')
        evid = f"{row['event_id']}_{row['station_network_code']}.{row['station_code']}"

        # Read and preprocess waveform
        stream = obspy.read('../' + new_file_path)
        stream.detrend('linear')
        stream.resample(ORIGINAL_SR)

        tr = stream[0].data[int(ARRIVAL - START * ORIGINAL_SR) : int(ARRIVAL + END * ORIGINAL_SR)]
        data.append(tr)
        evids.append(evid)
    except Exception as e:
        print(f"Skipping event {row['event_id']} due to error: {e}")

# Convert to NumPy array
data = np.array(data)
print(f"Data shape: {data.shape}")

# Signal processing
tapered_data = apply_cosine_taper(data, taper_percent=TAPER_AMOUNT)
filtered_data = np.array(butterworth_filter(tapered_data, LOW_FREQ, HIGH_FREQ, ORIGINAL_SR, NUM_CORNERS, 'bandpass'))
normalized_data = filtered_data / np.max(abs(filtered_data), axis=1, keepdims=True)
resampled_data = np.array([resample_array(arr, ORIGINAL_SR, NEW_SR) for arr in normalized_data])

# Feature extraction
feature_list = []
for i in tqdm(range(len(resampled_data))):
    try:
        features = time_series_features_extractor(cfg_file, resampled_data[i], fs= NEW_SR, verbose = 0)
        features['new_event_id'] = evids[i]
        feature_list.append(features)
    except Exception as e:
        print(f"Feature extraction failed for {evids[i]}: {e}")

# Combine features into a DataFrame
features_df = pd.concat(feature_list, ignore_index=True)
print(f"Feature DataFrame shape: {features_df.shape}")

low_freq = str(LOW_FREQ).replace('.','')
features_df.to_csv(f'../../extracted_features/tsfel_features_new_surface event_P_{START}_{END}_F_'+low_freq+f'_{HIGH_FREQ}_{NEW_SR}_part_1.csv')


