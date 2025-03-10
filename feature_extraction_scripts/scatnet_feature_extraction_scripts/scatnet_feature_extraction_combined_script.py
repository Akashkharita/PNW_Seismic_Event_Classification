# importing the dependencies. 
import pandas as pd
import numpy as np
import h5py
import obspy
from obspy.signal.filter import envelope
from tqdm import tqdm
import warnings
import argparse
import sys
import os
from scipy.signal import resample_poly
from scatseisnet import ScatteringNetwork
import matplotlib.dates as mdates
import pickle
from scipy.signal import butter, filtfilt
from scipy.signal import butter, lfilter

output_path = '/home/ak287/PNW_Seismic_Event_Classification/extracted_features'


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
parser.add_argument("--class_type", type=str, default = 'explosion', required=True, choices=['earthquake', 'surface event', 'thunder', 'sonic boom', 'explosion', 'noise'], help="Type of seismic event to process")
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
class_type =  args.class_type
part = args.part



segment_duration_seconds = start_time+end_time
sampling_rate_hertz = new_sr
samples_per_segment = int(segment_duration_seconds * sampling_rate_hertz)
bank_keyword_arguments = (
    {"octaves": 5, "resolution": 2, "quality": 1},
    {"octaves": 5, "resolution": 2, "quality": 3},
)


# Total wavelets = no. of octaves x resolution
# quality factors controls the appearance of wavelet
# So basically a wavelet at different frequency at a given time is multiplied to the waveform at a given time and
# coefficients are computed. So we will have a total of 16x 18000 samples in the deep scattering spectrogram. 
# Now in the second layer, treating the each frequency as a separate series, the second order scattering coefficients
# are multiplied. So we will get 10x180000x16

network = ScatteringNetwork(
    *bank_keyword_arguments,
    bins=samples_per_segment,
    sampling_rate=sampling_rate_hertz,
)


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

# Function to extract data
def extract_data(class_type):
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
                data.append(f['/data/' + buck_eq[i]][ind_eq[i], :, int(5000 - start_time * original_sr): int(5000 + end_time * original_sr)])
        elif 0 < part < 4:
            for i in tqdm(range(int(50000 * (part - 1)), int(50000 * part))):
                data.append(f['/data/' + buck_eq[i]][ind_eq[i], :, int(5000 - start_time * original_sr): int(5000 + end_time * original_sr)])
        else:
            print("part cannot be less than 1 or more than 4")
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
                
            data.append(f['/data/' + buck[i]][ind[i], :, start_idx: end_idx])
    
    data = np.array(data)
    return data

    
    
def cosine_taper_bandpass_filter(data, sampling_rate, lowcut = low, highcut = high, taper_amount = taper_amount):
    # Define the percentage of tapering (10%)
    taper_percentage = taper_amount/100

    # Create a cosine taper window
    taper_length = int(len(data) * taper_percentage)
    taper = 0.5 * (1 - np.cos(2 * np.pi * np.arange(taper_length) / (taper_length - 1)))

    # Apply the taper to the beginning and end of the data
    data[:taper_length] *= taper
    data[-taper_length:] *= taper[::-1]


    # Normalize the cutoff frequencies
    lowcut /= (0.5 * sampling_rate)
    highcut /= (0.5 * sampling_rate)

    # Design the Butterworth bandpass filter
    b, a = butter(4, [lowcut, highcut], btype='band')

    # Apply the bandpass filter to the data using filtfilt
    filtered_data = filtfilt(b, a, data)

    return filtered_data



def normalize_arrays(array_of_arrays):
    normalized_arrays = []
    
    for sub_arrays in array_of_arrays:
        normalized_sub_arrays = []
        for arr in sub_arrays:
            max_val = np.max(np.abs(arr))
            normalized_arr = arr / max_val if max_val != 0 else arr  # Avoid division by zero
            normalized_sub_arrays.append(normalized_arr)
        normalized_arrays.append(normalized_sub_arrays)
    
    return np.array(normalized_arrays)



def resample_arrays_to_rate(array_of_arrays, original_rate, desired_rate):
    resampled_arrays = []
    gcd = np.gcd(original_rate, desired_rate)  # Greatest common divisor
    up = desired_rate // gcd  # Upsampling factor
    down = original_rate // gcd  # Downsampling factor
    
    for sub_arrays in array_of_arrays:
        resampled_sub_arrays = []
        for arr in sub_arrays:
            resampled_arr = resample_poly(arr, up, down)
            resampled_sub_arrays.append(resampled_arr)
        resampled_arrays.append(resampled_sub_arrays)
    
    return np.array(resampled_arrays)


## defining filter banks




# Process and extract features
data = extract_data(class_type)

if data is None:
    sys.exit("No data to process.")
    

if len(data.shape) <=2:
    data = np.reshape(data,[1, 3, int((start_time+end_time)*100)])
    
    

filtered_data = np.zeros_like(data)
for i in range(len(data)):
    for j in range(len(data[i])):
        filtered_data[i][j] = cosine_taper_bandpass_filter(data[i][j], original_sr)
        
norm_array = normalize_arrays(filtered_data)
resampled_array = resample_arrays_to_rate(norm_array, original_sr, new_sr)


scattering_coefficients = []
serial_no = []

for i in tqdm(range(len(resampled_array))):
    segments = resampled_array[i]
    scattering_coefficients.append(network.transform(segments, reduce_type=np.max))
    serial_no.append(i+ (int(50000 * (part - 1)) if class_type == 'earthquake' else 0))

order_1 = []
order_2 = []
for i in range(len(scattering_coefficients)):
    order_1.append(scattering_coefficients[i][0][2])
    order_2.append(scattering_coefficients[i][1][2])
    
order_1 = np.array(order_1)
order_2 = np.array(order_2)
    
np.savez(
    os.path.join(output_path, f'scattering_coefficients_{class_type}_P_{start_time}_{end_time}_F_{int(low)}_{int(high)}_{new_sr}_part_{part}.npz'),
    order_1 = order_1,
    order_2 = order_2,
    serial_ids = serial_no
)

    
    