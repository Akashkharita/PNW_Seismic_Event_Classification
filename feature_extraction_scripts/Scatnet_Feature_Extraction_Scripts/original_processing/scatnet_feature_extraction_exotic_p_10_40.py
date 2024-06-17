import pandas as pd
import numpy as np
import scipy as sc
from scipy import signal

import matplotlib.pyplot as plt
import h5py
import obspy
from obspy.signal.filter import envelope
from obspy.clients.fdsn import Client
from tqdm import tqdm
from glob import glob
import tsfel
import random


client = Client('IRIS')

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from scatseisnet import ScatteringNetwork
import matplotlib.dates as mdates


import time

import numpy as npls


from scipy.signal import butter, filtfilt
from scipy.signal import butter, lfilter

#%config InlineBackend.figure_format = "svg"

#from Feature_Extraction import compute_hibert




# collecting waveform data and corresponding catalog.
exotic_file_name = "/data/whd01/yiyu_data/PNWML/exotic_waveforms.hdf5"
exotic_csv_file = "/data/whd01/yiyu_data/PNWML/exotic_metadata.csv"



cat_exotic = pd.read_csv(exotic_csv_file)



## According to paper the waveforms are detrended but not filtered. (https://seismica.library.mcgill.ca/article/view/368/868)
## According to tutorial  - we need to detrend and bandpass filter, Since we highpass filtered the data in case of other features we are going to do the same in this case as well. 


def extract_waveforms(cat, file_name):
    
    
    st = []
    cat_trace = cat['trace_name'].values
    serial_no = []
    exotic_id = []
    for i in range(len(cat_trace)):

            
 
        f = h5py.File(file_name, 'r')
            
            
        bucket = cat['trace_name'].values[i].split('$')[0]
        ind = int(cat['trace_name'].values[i].split('$')[1].split(',')[0])
        # ENZ
        
        if np.sum(f['/data/'+bucket][ind, 0, :]) != 0:
            exotic_id.append(cat['event_id'].values[i])
            st.append(f['/data/'+bucket][ind, :3, :])
            serial_no.append(i)
        
    return st, exotic_id, serial_no







def cosine_taper_bandpass_filter(data, sampling_rate):
    # Define the percentage of tapering (10%)
    taper_percentage = 0.10

    # Create a cosine taper window
    taper_length = int(len(data) * taper_percentage)
    taper = 0.5 * (1 - np.cos(2 * np.pi * np.arange(taper_length) / (taper_length - 1)))

    # Apply the taper to the beginning and end of the data
    data[:taper_length] *= taper
    data[-taper_length:] *= taper[::-1]

    # Define the bandpass filter parameters
    lowcut = 1.0  # Lower cutoff frequency in Hz
    highcut = 10.0  # Upper cutoff frequency in Hz

    # Normalize the cutoff frequencies
    lowcut /= (0.5 * sampling_rate)
    highcut /= (0.5 * sampling_rate)

    # Design the Butterworth bandpass filter
    b, a = butter(4, [lowcut, highcut], btype='band')

    # Apply the bandpass filter to the data
    filtered_data = lfilter(b, a, data)

    return filtered_data



segment_duration_seconds = 50
sampling_rate_hertz = 100
samples_per_segment = int(segment_duration_seconds * sampling_rate_hertz)
bank_keyword_arguments = (
    {"octaves": 7, "resolution": 4, "quality": 3},
    {"octaves": 8, "resolution": 3, "quality": 4},
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










st_exotic, exotic_id, serial_no = extract_waveforms(cat_exotic, exotic_file_name)

fs = 100  # Sampling frequency in Hz
cutoff = 1  # Cutoff frequency in Hz

data = np.array(st_exotic)[:,:, 6000:11000]
filtered_data = np.zeros_like(data)
for i in tqdm(range(len(data))):
    for j in range(len(data[i])):
        filtered_data[i][j] = cosine_taper_bandpass_filter(data[i][j], fs)
        
        
print(filtered_data.shape)        
segments = filtered_data
starttime = time.time()
scattering_coefficients = network.transform(segments, reduce_type=np.max)
endtime = time.time()
dur_scatnet_exotic = endtime - starttime
        
        
       
np.savez(
    "/home/ak287/PNW_ML_Classification/extracted_features/scatnet_features/p_10_40/exotic_scattering_coefficients.npz",
    order_1=scattering_coefficients[0],
    order_2=scattering_coefficients[1],
    event_ids = exotic_id,
    serial_ids = serial_no,
    duration = dur_scatnet_exotic
)



















