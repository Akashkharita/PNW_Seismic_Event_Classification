# importing the dependencies. 
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
from datetime import timedelta
import calendar
from tsfel import time_series_features_extractor











#%config InlineBackend.figure_format = "png"

#from Feature_Extraction import compute_hibert

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# displaying all columns from pandas dataframe
# Set display options to show all columns
pd.set_option('display.max_columns', None)

import time



              
noise_file_name = "/data/whd01/yiyu_data/PNWML/noise_waveforms.hdf5"          
noise_csv_file = pd.read_csv("/data/whd01/yiyu_data/PNWML/noise_metadata.csv")
f = h5py.File(noise_file_name, 'r')          
buckets = [noise_csv_file['trace_name'].values[i].split('$')[0] for i in range(len(noise_csv_file))]
indices = [int(noise_csv_file['trace_name'].values[i].split('$')[1].split(',')[0]) for i in range(len(noise_csv_file))]
buck_no = np.array(buckets)
ind_no = np.array(indices)

data_no = []

for i in tqdm(range(len(buck_no))):
     data_no.append(f['/data/'+buck_no[i]][ind_no[i], 2, 4000:9000])
              
data_no = np.array(data_no) 
        
   
              
              
              
              
def apply_cosine_taper(arrays, taper_percent=10):
    tapered_arrays = []
    
    #print(arrays.shape)
    num_samples = arrays.shape[1]  # Assuming each sub-array has the same length
    
    for array in arrays:
        

        taper_length = int(num_samples * taper_percent / 100)
        taper_window = np.hanning(2 * taper_length)
        
     
        tapered_array = array.copy()
        tapered_array[:taper_length] = tapered_array[:taper_length] * taper_window[:taper_length]
        tapered_array[-taper_length:] = tapered_array[-taper_length:] * taper_window[taper_length:]
        
        tapered_arrays.append(tapered_array)
    
    return np.array(tapered_arrays)
                
              
              
              
              

import numpy as np
import scipy.signal as signal

def butterworth_filter(arrays, lowcut, highcut, fs, num_corners, filter_type='bandpass'):
    """
    Apply a Butterworth filter (bandpass, highpass, or lowpass) to each array in an array of arrays.

    Parameters:
        arrays (list of numpy arrays): List of arrays to be filtered.
        lowcut (float): Lower cutoff frequency in Hz.
        highcut (float): Upper cutoff frequency in Hz.
        fs (float): Sampling frequency in Hz.
        num_corners (int): Number of corners (filter order).
        filter_type (str, optional): Type of filter ('bandpass', 'highpass', or 'lowpass'). Default is 'bandpass'.

    Returns:
        list of numpy arrays: List of filtered arrays.
    """
    filtered_arrays = []
    for data in arrays:
        # Normalize the frequency values to Nyquist frequency (0.5*fs)
        lowcut_norm = lowcut / (0.5 * fs)
        highcut_norm = highcut / (0.5 * fs)

        # Design the Butterworth filter based on the filter type
        if filter_type == 'bandpass':
            b, a = signal.butter(num_corners, [lowcut_norm, highcut_norm], btype='band')
        elif filter_type == 'highpass':
            b, a = signal.butter(num_corners, lowcut_norm, btype='high')
        elif filter_type == 'lowpass':
            b, a = signal.butter(num_corners, highcut_norm, btype='low')
        else:
            raise ValueError("Invalid filter_type. Use 'bandpass', 'highpass', or 'lowpass'.")

        # Apply the filter to the data using lfilter
        filtered_data = signal.lfilter(b, a, data)

        filtered_arrays.append(filtered_data)

    return filtered_arrays

              
tapered_no = apply_cosine_taper(data_no, taper_percent = 10)          
           
filtered_no = np.array(butterworth_filter(tapered_no, 1, 10, 100, 4, 'bandpass'))       

no_Z = filtered_no

              
# Normalizing the data.              
no_Z = no_Z/np.max(abs(no_Z), axis = 1)[:, np.newaxis]


cfg_file = tsfel.get_features_by_domain()


    
# Extract features for sonic boom
features_no = pd.DataFrame([])
for i in range(len(no_Z)):
    try:
        
        df = time_series_features_extractor(cfg_file, no_Z[i], fs=100,)
        df['serial_no'] = i
        features_no = pd.concat([features_no,df]) 
        
    except:
        pass
    
    


              
X = features_no
y = ['noise']*len(features_no)
X['source'] = y


X.to_csv('/home/ak287/Data_Mining_in_the_PNW/Extracted_Features/Tsfel_Features/tsfel_features_noise_10_40.csv')


              
              
              
              
              
              
              
              