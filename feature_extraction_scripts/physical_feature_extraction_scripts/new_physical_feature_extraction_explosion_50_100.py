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









comcat_file_name = "/data/whd01/yiyu_data/PNWML/comcat_waveforms.hdf5"
comcat_csv_file = pd.read_csv("/data/whd01/yiyu_data/PNWML/comcat_metadata.csv")


f = h5py.File(comcat_file_name, 'r')

buckets = [comcat_csv_file['trace_name'].values[i].split('$')[0] for i in range(len(comcat_csv_file))]
indices = [int(comcat_csv_file['trace_name'].values[i].split('$')[1].split(',')[0]) for i in range(len(comcat_csv_file))]

source = comcat_csv_file['source_type'].values


buck_exp = np.array(buckets)[np.where(source == 'explosion')[0]]
ind_exp = np.array(indices)[np.where(source == 'explosion')[0]]

data_exp = []
for i in tqdm(range(len(buck_exp))):
     data_exp.append(f['/data/'+buck_exp[i]][ind_exp[i], 2, :])
        

        

data_exp = np.array(data_exp)





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
              
    
    
def apply_cosine_taper(arrays, taper_percent=10):
    num_samples = arrays.shape[1]
    taper_length = int(num_samples * taper_percent / 100)
    taper_window = np.hanning(2 * taper_length)
    taper = np.tile(taper_window, (len(arrays), 1))
    tapered_arrays = arrays.copy()
    tapered_arrays[:, :taper_length] *= taper[:, :taper_length]
    tapered_arrays[:, -taper_length:] *= taper[:, taper_length:]
    return tapered_arrays

    

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

              
 

tapered_exp = apply_cosine_taper(data_exp, taper_percent = 10)
filtered_exp = np.array(butterworth_filter(tapered_exp, 1, 10, 100, 4, 'bandpass')) 



              
        
     

filtered_exp = np.array(filtered_exp)


exp_Z = filtered_exp



exp_Z /= np.max(abs(exp_Z), axis = 1)[:, np.newaxis]

              

    
exp_df_Z_list = []
for i in tqdm(range(len(exp_Z))):
    try:
        df =   seis_feature.FeatureCalculator(exp_Z[i]).compute_features()  #seis_feature.compute_physical_features(tr=tr, envfilter=False)
        df['serial_no'] = i
        exp_df_Z_list.append(df)
    except:
        pass
exp_df_Z = pd.concat(exp_df_Z_list)


 


              
X = pd.concat([exp_df_Z])
y = ['explosion']*len(exp_df_Z) 
X['source'] = y


X.to_csv('/home/ak287/PNW_Seismic_Event_Classification/extracted_features/new_physical_features_comcat_z_explosion_50_100.csv')


              
              
              
              
              
              
              
              