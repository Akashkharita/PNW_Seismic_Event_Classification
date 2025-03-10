# Import dependencies
import os
import json
import numpy as np
import pandas as pd
import obspy
import h5py
import pickle
import warnings
import argparse
import sys
import matplotlib.dates as mdates
from tqdm import tqdm
from scipy.signal import butter, filtfilt, resample_poly
from scatseisnet import ScatteringNetwork
from glob import glob
from obspy.signal.filter import envelope




# Constants
START_TIME = 10
END_TIME = 30
LOWCUT = 0.5
HIGHCUT = 15
TAPER_AMOUNT = 10
ORIGINAL_SR = 100
NEW_SR = 50
SEGMENT_DURATION = START_TIME + END_TIME
SAMPLES_PER_SEGMENT = int(SEGMENT_DURATION * NEW_SR)
OUTPUT_PATH = '/home/ak287/PNW_Seismic_Event_Classification/extracted_features'


# Bandpass filter with cosine taper
def cosine_taper_bandpass_filter(data, sr, lowcut=LOWCUT, highcut=HIGHCUT, taper_amount=TAPER_AMOUNT):
    taper_len = int(len(data) * (taper_amount / 100))
    taper = 0.5 * (1 - np.cos(2 * np.pi * np.arange(taper_len) / (taper_len - 1)))

    # Apply taper
    data[:taper_len] *= taper
    data[-taper_len:] *= taper[::-1]

    # Design and apply Butterworth filter
    nyq = 0.5 * sr
    b, a = butter(4, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)

# Normalize arrays
def normalize_arrays(arr):
    max_vals = np.max(np.abs(arr), axis=-1, keepdims=True)
    return np.where(max_vals != 0, arr / max_vals, arr)

# Resample arrays
def resample_arrays(arr, orig_sr, new_sr):
    gcd = np.gcd(orig_sr, new_sr)
    up, down = new_sr // gcd, orig_sr // gcd
    return np.array([resample_poly(sub_arr, up, down) for sub_arr in arr])










# Initialize Scattering Network
BANK_KWARGS = ({"octaves": 5, "resolution": 2, "quality": 1}, {"octaves": 5, "resolution": 2, "quality": 3})
network = ScatteringNetwork(*BANK_KWARGS, bins=SAMPLES_PER_SEGMENT, sampling_rate=NEW_SR)

# Load data and IDs
data_path = '../../data/new_curated_surface_event_data.npy'
ids_path = '../../data/new_curated_surface_event_ids.json'
surface_data = np.load(data_path, allow_pickle=True)
with open(ids_path, "r") as file:
    surface_ids = json.load(file)

# Process data
data, new_ids = [], []
for i in tqdm(range(len(surface_data)), desc="Extracting Data"):
    if surface_data[i].shape == (3, 18000):
        data.append(surface_data[i][:, int((70 - START_TIME) * ORIGINAL_SR): int((70 + END_TIME) * ORIGINAL_SR)])
        new_ids.append(surface_ids[i])

# Convert to numpy array
data = np.array(data)

# Apply filtering and resampling
filtered_data = np.array([[cosine_taper_bandpass_filter(tr, ORIGINAL_SR, lowcut=LOWCUT, highcut=HIGHCUT, taper_amount=TAPER_AMOUNT) for tr in sample] for sample in tqdm(data, desc="Filtering Data")])
normalized_data = normalize_arrays(filtered_data)
resampled_data = np.array([resample_arrays(sample, ORIGINAL_SR, NEW_SR) for sample in tqdm(normalized_data, desc="Resampling Data")])

# Compute Scattering Coefficients
scattering_coeffs, serial_ids = [], []
for i in tqdm(range(len(resampled_data)), desc="Computing Scattering Coefficients"):
    scattering_coeffs.append(network.transform(resampled_data[i], reduce_type=np.max))
    serial_ids.append(new_ids[i])

# Extract orders
order_1 = np.array([coeff[0][2] for coeff in scattering_coeffs])
order_2 = np.array([coeff[1][2] for coeff in scattering_coeffs])

# Save results
np.savez(
    os.path.join(OUTPUT_PATH, f'scattering_coefficients_new_surface_events_P_{START_TIME}_{END_TIME}_F_{int(LOWCUT)}_{int(HIGHCUT)}_{NEW_SR}_part_1.npz'),
    order_1=order_1,
    order_2=order_2,
    serial_ids=serial_ids
)






# Constants
START_TIME = 10
END_TIME = 30
LOWCUT = 1
HIGHCUT = 10
TAPER_AMOUNT = 10
ORIGINAL_SR = 100
NEW_SR = 50
SEGMENT_DURATION = START_TIME + END_TIME
SAMPLES_PER_SEGMENT = int(SEGMENT_DURATION * NEW_SR)
OUTPUT_PATH = '/home/ak287/PNW_Seismic_Event_Classification/extracted_features'




# Initialize Scattering Network
BANK_KWARGS = ({"octaves": 5, "resolution": 2, "quality": 1}, {"octaves": 5, "resolution": 2, "quality": 3})
network = ScatteringNetwork(*BANK_KWARGS, bins=SAMPLES_PER_SEGMENT, sampling_rate=NEW_SR)

# Load data and IDs
data_path = '../../data/new_curated_surface_event_data.npy'
ids_path = '../../data/new_curated_surface_event_ids.json'
surface_data = np.load(data_path, allow_pickle=True)
with open(ids_path, "r") as file:
    surface_ids = json.load(file)

# Process data
data, new_ids = [], []
for i in tqdm(range(len(surface_data)), desc="Extracting Data"):
    if surface_data[i].shape == (3, 18000):
        data.append(surface_data[i][:, int((70 - START_TIME) * ORIGINAL_SR): int((70 + END_TIME) * ORIGINAL_SR)])
        new_ids.append(surface_ids[i])

# Convert to numpy array
data = np.array(data)

# Apply filtering and resampling
filtered_data = np.array([[cosine_taper_bandpass_filter(tr, ORIGINAL_SR, lowcut=LOWCUT, highcut=HIGHCUT, taper_amount=TAPER_AMOUNT) for tr in sample] for sample in tqdm(data, desc="Filtering Data")])
normalized_data = normalize_arrays(filtered_data)
resampled_data = np.array([resample_arrays(sample, ORIGINAL_SR, NEW_SR) for sample in tqdm(normalized_data, desc="Resampling Data")])

# Compute Scattering Coefficients
scattering_coeffs, serial_ids = [], []
for i in tqdm(range(len(resampled_data)), desc="Computing Scattering Coefficients"):
    scattering_coeffs.append(network.transform(resampled_data[i], reduce_type=np.max))
    serial_ids.append(new_ids[i])

# Extract orders
order_1 = np.array([coeff[0][2] for coeff in scattering_coeffs])
order_2 = np.array([coeff[1][2] for coeff in scattering_coeffs])

# Save results
np.savez(
    os.path.join(OUTPUT_PATH, f'scattering_coefficients_new_surface_events_P_{START_TIME}_{END_TIME}_F_{int(LOWCUT)}_{int(HIGHCUT)}_{NEW_SR}_part_1.npz'),
    order_1=order_1,
    order_2=order_2,
    serial_ids=serial_ids
)







# Constants
START_TIME = 10
END_TIME = 100
LOWCUT = 0.5
HIGHCUT = 15
TAPER_AMOUNT = 10
ORIGINAL_SR = 100
NEW_SR = 50
SEGMENT_DURATION = START_TIME + END_TIME
SAMPLES_PER_SEGMENT = int(SEGMENT_DURATION * NEW_SR)
OUTPUT_PATH = '/home/ak287/PNW_Seismic_Event_Classification/extracted_features'




# Initialize Scattering Network
BANK_KWARGS = ({"octaves": 5, "resolution": 2, "quality": 1}, {"octaves": 5, "resolution": 2, "quality": 3})
network = ScatteringNetwork(*BANK_KWARGS, bins=SAMPLES_PER_SEGMENT, sampling_rate=NEW_SR)

# Load data and IDs
data_path = '../../data/new_curated_surface_event_data.npy'
ids_path = '../../data/new_curated_surface_event_ids.json'
surface_data = np.load(data_path, allow_pickle=True)
with open(ids_path, "r") as file:
    surface_ids = json.load(file)

# Process data
data, new_ids = [], []
for i in tqdm(range(len(surface_data)), desc="Extracting Data"):
    if surface_data[i].shape == (3, 18000):
        data.append(surface_data[i][:, int((70 - START_TIME) * ORIGINAL_SR): int((70 + END_TIME) * ORIGINAL_SR)])
        new_ids.append(surface_ids[i])

# Convert to numpy array
data = np.array(data)

# Apply filtering and resampling
filtered_data = np.array([[cosine_taper_bandpass_filter(tr, ORIGINAL_SR, lowcut=LOWCUT, highcut=HIGHCUT, taper_amount=TAPER_AMOUNT) for tr in sample] for sample in tqdm(data, desc="Filtering Data")])
normalized_data = normalize_arrays(filtered_data)
resampled_data = np.array([resample_arrays(sample, ORIGINAL_SR, NEW_SR) for sample in tqdm(normalized_data, desc="Resampling Data")])

# Compute Scattering Coefficients
scattering_coeffs, serial_ids = [], []
for i in tqdm(range(len(resampled_data)), desc="Computing Scattering Coefficients"):
    scattering_coeffs.append(network.transform(resampled_data[i], reduce_type=np.max))
    serial_ids.append(new_ids[i])

# Extract orders
order_1 = np.array([coeff[0][2] for coeff in scattering_coeffs])
order_2 = np.array([coeff[1][2] for coeff in scattering_coeffs])

# Save results
np.savez(
    os.path.join(OUTPUT_PATH, f'scattering_coefficients_new_surface_events_P_{START_TIME}_{END_TIME}_F_{int(LOWCUT)}_{int(HIGHCUT)}_{NEW_SR}_part_1.npz'),
    order_1=order_1,
    order_2=order_2,
    serial_ids=serial_ids
)










# Constants
START_TIME = 10
END_TIME = 100
LOWCUT = 1
HIGHCUT = 10
TAPER_AMOUNT = 10
ORIGINAL_SR = 100
NEW_SR = 50
SEGMENT_DURATION = START_TIME + END_TIME
SAMPLES_PER_SEGMENT = int(SEGMENT_DURATION * NEW_SR)
OUTPUT_PATH = '/home/ak287/PNW_Seismic_Event_Classification/extracted_features'




# Initialize Scattering Network
BANK_KWARGS = ({"octaves": 5, "resolution": 2, "quality": 1}, {"octaves": 5, "resolution": 2, "quality": 3})
network = ScatteringNetwork(*BANK_KWARGS, bins=SAMPLES_PER_SEGMENT, sampling_rate=NEW_SR)

# Load data and IDs
data_path = '../../data/new_curated_surface_event_data.npy'
ids_path = '../../data/new_curated_surface_event_ids.json'
surface_data = np.load(data_path, allow_pickle=True)
with open(ids_path, "r") as file:
    surface_ids = json.load(file)

# Process data
data, new_ids = [], []
for i in tqdm(range(len(surface_data)), desc="Extracting Data"):
    if surface_data[i].shape == (3, 18000):
        data.append(surface_data[i][:, int((70 - START_TIME) * ORIGINAL_SR): int((70 + END_TIME) * ORIGINAL_SR)])
        new_ids.append(surface_ids[i])

# Convert to numpy array
data = np.array(data)

# Apply filtering and resampling
filtered_data = np.array([[cosine_taper_bandpass_filter(tr, ORIGINAL_SR, lowcut=LOWCUT, highcut=HIGHCUT, taper_amount=TAPER_AMOUNT) for tr in sample] for sample in tqdm(data, desc="Filtering Data")])
normalized_data = normalize_arrays(filtered_data)
resampled_data = np.array([resample_arrays(sample, ORIGINAL_SR, NEW_SR) for sample in tqdm(normalized_data, desc="Resampling Data")])

# Compute Scattering Coefficients
scattering_coeffs, serial_ids = [], []
for i in tqdm(range(len(resampled_data)), desc="Computing Scattering Coefficients"):
    scattering_coeffs.append(network.transform(resampled_data[i], reduce_type=np.max))
    serial_ids.append(new_ids[i])

# Extract orders
order_1 = np.array([coeff[0][2] for coeff in scattering_coeffs])
order_2 = np.array([coeff[1][2] for coeff in scattering_coeffs])

# Save results
np.savez(
    os.path.join(OUTPUT_PATH, f'scattering_coefficients_new_surface_events_P_{START_TIME}_{END_TIME}_F_{int(LOWCUT)}_{int(HIGHCUT)}_{NEW_SR}_part_1.npz'),
    order_1=order_1,
    order_2=order_2,
    serial_ids=serial_ids
)












# Constants
START_TIME = 50
END_TIME = 100
LOWCUT = 0.5
HIGHCUT = 15
TAPER_AMOUNT = 10
ORIGINAL_SR = 100
NEW_SR = 50
SEGMENT_DURATION = START_TIME + END_TIME
SAMPLES_PER_SEGMENT = int(SEGMENT_DURATION * NEW_SR)
OUTPUT_PATH = '/home/ak287/PNW_Seismic_Event_Classification/extracted_features'




# Initialize Scattering Network
BANK_KWARGS = ({"octaves": 5, "resolution": 2, "quality": 1}, {"octaves": 5, "resolution": 2, "quality": 3})
network = ScatteringNetwork(*BANK_KWARGS, bins=SAMPLES_PER_SEGMENT, sampling_rate=NEW_SR)

# Load data and IDs
data_path = '../../data/new_curated_surface_event_data.npy'
ids_path = '../../data/new_curated_surface_event_ids.json'
surface_data = np.load(data_path, allow_pickle=True)
with open(ids_path, "r") as file:
    surface_ids = json.load(file)

# Process data
data, new_ids = [], []
for i in tqdm(range(len(surface_data)), desc="Extracting Data"):
    if surface_data[i].shape == (3, 18000):
        data.append(surface_data[i][:, int((70 - START_TIME) * ORIGINAL_SR): int((70 + END_TIME) * ORIGINAL_SR)])
        new_ids.append(surface_ids[i])

# Convert to numpy array
data = np.array(data)

# Apply filtering and resampling
filtered_data = np.array([[cosine_taper_bandpass_filter(tr, ORIGINAL_SR, lowcut=LOWCUT, highcut=HIGHCUT, taper_amount=TAPER_AMOUNT) for tr in sample] for sample in tqdm(data, desc="Filtering Data")])
normalized_data = normalize_arrays(filtered_data)
resampled_data = np.array([resample_arrays(sample, ORIGINAL_SR, NEW_SR) for sample in tqdm(normalized_data, desc="Resampling Data")])

# Compute Scattering Coefficients
scattering_coeffs, serial_ids = [], []
for i in tqdm(range(len(resampled_data)), desc="Computing Scattering Coefficients"):
    scattering_coeffs.append(network.transform(resampled_data[i], reduce_type=np.max))
    serial_ids.append(new_ids[i])

# Extract orders
order_1 = np.array([coeff[0][2] for coeff in scattering_coeffs])
order_2 = np.array([coeff[1][2] for coeff in scattering_coeffs])

# Save results
np.savez(
    os.path.join(OUTPUT_PATH, f'scattering_coefficients_new_surface_events_P_{START_TIME}_{END_TIME}_F_{int(LOWCUT)}_{int(HIGHCUT)}_{NEW_SR}_part_1.npz'),
    order_1=order_1,
    order_2=order_2,
    serial_ids=serial_ids
)





# Constants
START_TIME = 50
END_TIME = 100
LOWCUT = 1
HIGHCUT = 10
TAPER_AMOUNT = 10
ORIGINAL_SR = 100
NEW_SR = 50
SEGMENT_DURATION = START_TIME + END_TIME
SAMPLES_PER_SEGMENT = int(SEGMENT_DURATION * NEW_SR)
OUTPUT_PATH = '/home/ak287/PNW_Seismic_Event_Classification/extracted_features'




# Initialize Scattering Network
BANK_KWARGS = ({"octaves": 5, "resolution": 2, "quality": 1}, {"octaves": 5, "resolution": 2, "quality": 3})
network = ScatteringNetwork(*BANK_KWARGS, bins=SAMPLES_PER_SEGMENT, sampling_rate=NEW_SR)

# Load data and IDs
data_path = '../../data/new_curated_surface_event_data.npy'
ids_path = '../../data/new_curated_surface_event_ids.json'
surface_data = np.load(data_path, allow_pickle=True)
with open(ids_path, "r") as file:
    surface_ids = json.load(file)

# Process data
data, new_ids = [], []
for i in tqdm(range(len(surface_data)), desc="Extracting Data"):
    if surface_data[i].shape == (3, 18000):
        data.append(surface_data[i][:, int((70 - START_TIME) * ORIGINAL_SR): int((70 + END_TIME) * ORIGINAL_SR)])
        new_ids.append(surface_ids[i])

# Convert to numpy array
data = np.array(data)

# Apply filtering and resampling
filtered_data = np.array([[cosine_taper_bandpass_filter(tr, ORIGINAL_SR, lowcut=LOWCUT, highcut=HIGHCUT, taper_amount=TAPER_AMOUNT) for tr in sample] for sample in tqdm(data, desc="Filtering Data")])
normalized_data = normalize_arrays(filtered_data)
resampled_data = np.array([resample_arrays(sample, ORIGINAL_SR, NEW_SR) for sample in tqdm(normalized_data, desc="Resampling Data")])

# Compute Scattering Coefficients
scattering_coeffs, serial_ids = [], []
for i in tqdm(range(len(resampled_data)), desc="Computing Scattering Coefficients"):
    scattering_coeffs.append(network.transform(resampled_data[i], reduce_type=np.max))
    serial_ids.append(new_ids[i])

# Extract orders
order_1 = np.array([coeff[0][2] for coeff in scattering_coeffs])
order_2 = np.array([coeff[1][2] for coeff in scattering_coeffs])

# Save results
np.savez(
    os.path.join(OUTPUT_PATH, f'scattering_coefficients_new_surface_events_P_{START_TIME}_{END_TIME}_F_{int(LOWCUT)}_{int(HIGHCUT)}_{NEW_SR}_part_1.npz'),
    order_1=order_1,
    order_2=order_2,
    serial_ids=serial_ids
)

