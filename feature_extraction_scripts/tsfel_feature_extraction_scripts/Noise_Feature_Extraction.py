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
import warnings
import time
import argparse

import sys
import os

# Get the absolute path of the directory two levels up
two_levels_up = os.path.abspath(os.path.join(os.getcwd(), "../.."))

# Append the 'src' directory located two levels up to the system path
sys.path.append(os.path.join(two_levels_up, 'src'))

from common_processing_functions import apply_cosine_taper
from common_processing_functions import butterworth_filter
from common_processing_functions import resample_array

# Ignore all warnings
warnings.filterwarnings("ignore")

# displaying all columns from pandas dataframe
pd.set_option('display.max_columns', None)




## Example usage 
## python script_name.py --low 0.5 --high 20 --new_sr 100 --start_time 5 --end_time 25 --taper_amount 5 --original_sr 200 --num_corners 3 



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






noise_file_name = "/data/whd01/yiyu_data/PNWML/noise_waveforms.hdf5"          
noise_csv_file = pd.read_csv("/data/whd01/yiyu_data/PNWML/noise_metadata.csv")
f = h5py.File(noise_file_name, 'r')          
buckets = [noise_csv_file['trace_name'].values[i].split('$')[0] for i in range(len(noise_csv_file))]
indices = [int(noise_csv_file['trace_name'].values[i].split('$')[1].split(',')[0]) for i in range(len(noise_csv_file))]
buck_no = np.array(buckets)
ind_no = np.array(indices)

data_no = []

for i in tqdm(range(len(buck_no))):
     data_no.append(f['/data/'+buck_no[i]][ind_no[i], 2,  int(5000 - start_time*100): int(5000+end_time*100)])
              
data_no = np.array(data_no) 



              
tapered_no = apply_cosine_taper(data_no, taper_percent = taper_amount)                   
filtered_no = np.array(butterworth_filter(tapered_no,low, high, original_sr, num_corners, 'bandpass'))       
no_Z = filtered_no


# Resampling the data
no_Z = np.array([resample_array(arr, original_sr, new_sr) for arr in no_Z])
              
# Normalizing the data.              
no_Z = no_Z/np.max(abs(no_Z), axis = 1)[:, np.newaxis]


cfg_file = tsfel.get_features_by_domain()


    
# Extract features for sonic boom
features_no = pd.DataFrame([])
for i in tqdm(range(len(no_Z))):
    try:
        
        df = time_series_features_extractor(cfg_file, no_Z[i], fs = new_sr, verbose = 0)
        df['serial_no'] = i
        features_no = pd.concat([features_no,df]) 
        
    except:
        pass
    
    


              
X = features_no
y = ['noise']*len(features_no)
X['source'] = y




X.to_csv('/home/ak287/PNW_Seismic_Event_Classification/extracted_features/tsfel_features_noise_P_'+str(start_time)+'_'+str(end_time)+'_F_'+str(int(low))+'_'+str(int(high))+'_'+str(new_sr)+'.csv')











