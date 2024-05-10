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


        
              
              

              
exotic_file_name = "/data/whd01/yiyu_data/PNWML/exotic_waveforms.hdf5"
exotic_csv_file = pd.read_csv("/data/whd01/yiyu_data/PNWML/exotic_metadata.csv")
f = h5py.File(exotic_file_name, 'r')

buckets = [exotic_csv_file['trace_name'].values[i].split('$')[0] for i in range(len(exotic_csv_file))]
indices = [int(exotic_csv_file['trace_name'].values[i].split('$')[1].split(',')[0]) for i in range(len(exotic_csv_file))]

source = exotic_csv_file['source_type'].values


buck_su = np.array(buckets)[np.where(source == 'surface event')[0]]
ind_su = np.array(indices)[np.where(source == 'surface event')[0]]

data_su = []
for i in tqdm(range(len(buck_su))):
     data_su.append(f['/data/'+buck_su[i]][ind_su[i], 2, :])
        
        
data_su = np.array(data_su)



buck_th = np.array(buckets)[np.where(source == 'thunder')[0]]
ind_th = np.array(indices)[np.where(source == 'thunder')[0]]

data_th = []
for i in tqdm(range(len(buck_th))):
     data_th.append(f['/data/'+buck_th[i]][ind_th[i], 2, :])
        
        
data_th = np.array(data_th)


buck_sb = np.array(buckets)[np.where(source == 'sonic boom')[0]]
ind_sb = np.array(indices)[np.where(source == 'sonic boom')[0]]

data_sb = []
for i in tqdm(range(len(buck_sb))):
     data_sb.append(f['/data/'+buck_sb[i]][ind_sb[i], 2, :])
        
        
data_sb = np.array(data_sb) 
              
  
              
              
       
            
        
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

              
              
tapered_su = apply_cosine_taper(data_su, taper_percent = 10) 
tapered_th = apply_cosine_taper(data_th, taper_percent = 10)  
tapered_sb = apply_cosine_taper(data_sb, taper_percent = 10)               
              

filtered_su = np.array(butterworth_filter(tapered_su, 1, 10, 100, 4, 'bandpass'))
filtered_th = np.array(butterworth_filter(tapered_th, 1, 10, 100, 4, 'bandpass'))
filtered_sb = np.array(butterworth_filter(tapered_sb, 1, 10, 100, 4, 'bandpass'))

              
              
              


su_Z = filtered_su
th_Z = filtered_th
sb_Z = filtered_sb


              
# Normalizing the data.              

su_Z = su_Z/np.max(abs(su_Z), axis = 1)[:, np.newaxis]
th_Z = th_Z/np.max(abs(th_Z), axis = 1)[:, np.newaxis]
sb_Z = sb_Z/np.max(abs(sb_Z), axis = 1)[:, np.newaxis]

              
    
    



su_df_Z = pd.DataFrame([])
for i in tqdm(range(len(su_Z))):
    df = seis_feature.FeatureCalculator(su_Z[i]).compute_features()
    df['serial_no'] = i
    su_df_Z = pd.concat([su_df_Z, df])
    

th_df_Z = pd.DataFrame([])
for i in tqdm(range(len(th_Z))):
    df = seis_feature.FeatureCalculator(th_Z[i]).compute_features()
    df['serial_no'] = i
    th_df_Z = pd.concat([th_df_Z, df])
    

sb_df_Z = pd.DataFrame([])
for i in tqdm(range(len(sb_Z))):
    df = seis_feature.FeatureCalculator(sb_Z[i]).compute_features()
    df['serial_no'] = i
    sb_df_Z = pd.concat([sb_df_Z, df])
    


              
X = pd.concat([ su_df_Z, th_df_Z, sb_df_Z])
y =  ['surface']*len(su_df_Z)+['thunder']*len(th_df_Z)+['sonic']*len(sb_df_Z)
X['source'] = y

s


X.to_csv('/home/ak287/PNW_Seismic_Event_Classification/extracted_features/new_physical_features_exotic_z_50_100.csv')






              
              
              
              
              
              
              
              