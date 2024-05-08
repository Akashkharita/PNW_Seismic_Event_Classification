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

import seis_feature









#%config InlineBackend.figure_format = "png"

#from Feature_Extraction import compute_hibert

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# displaying all columns from pandas dataframe
# Set display options to show all columns
pd.set_option('display.max_columns', None)

import time












import numpy as np
import scipy.signal as signal

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

              
            
        
        
comcat_file_name = "/data/whd01/yiyu_data/PNWML/comcat_waveforms.hdf5"
comcat_csv_file = pd.read_csv("/data/whd01/yiyu_data/PNWML/comcat_metadata.csv")


f = h5py.File(comcat_file_name, 'r')

buckets = [comcat_csv_file['trace_name'].values[i].split('$')[0] for i in range(len(comcat_csv_file))]
indices = [int(comcat_csv_file['trace_name'].values[i].split('$')[1].split(',')[0]) for i in range(len(comcat_csv_file))]

source = comcat_csv_file['source_type'].values




buck_eq = np.array(buckets)[np.where(source == 'earthquake')[0]]
ind_eq = np.array(indices)[np.where(source == 'earthquake')[0]]

data_eq1 = []



for i in tqdm(range(125000, len(buck_eq))):
     data_eq1.append(f['/data/'+buck_eq[i]][ind_eq[i], 2, 4000:9000])
        
data_eq1 = np.array(data_eq1)







tapered_eq1 = apply_cosine_taper(data_eq1, taper_percent = 10)
filtered_eq1 = np.array(butterworth_filter(tapered_eq1, 1, 10, 100, 4, 'bandpass'))



              
        
        
filtered_eq1 = np.array(filtered_eq1)

              
              


              

eq_Z1 = filtered_eq1



              
# Normalizing the data.              
eq_Z1 = eq_Z1/np.max(abs(eq_Z1), axis = 1)[:, np.newaxis]


 
eq_Z = np.concatenate([eq_Z1])
              
eq_df_Z = pd.DataFrame([])
for i in tqdm(range(len(eq_Z))):
    try:
        tr = obspy.Trace(eq_Z[i])
        tr.stats.sampling_rate = 100
        df = seis_feature.compute_physical_features(tr = tr, envfilter = False)
        df['serial_no'] = i+125000
        eq_df_Z = pd.concat([eq_df_Z, df])
    except:
        
        pass
    

    
 



              
X = pd.concat([eq_df_Z])
y = ['earthquake']*len(eq_df_Z)
X['source'] = y


X.to_csv('/home/ak287/Data_Mining_in_the_PNW/Extracted_Features/Physical_Features/physical_features_comcat_z_earthquake_10_40_part4.csv')


              
              
              
              
              
              
              
              