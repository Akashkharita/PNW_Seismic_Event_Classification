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


import seis_feature

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
     data_su.append(f['/data/'+buck_su[i]][ind_su[i], 2, int(7000 - start_time*original_sr): int(7000+end_time*original_sr)])
        
        
data_su = np.array(data_su)



buck_th = np.array(buckets)[np.where(source == 'thunder')[0]]
ind_th = np.array(indices)[np.where(source == 'thunder')[0]]

data_th = []
for i in tqdm(range(len(buck_th))):
     data_th.append(f['/data/'+buck_th[i]][ind_th[i], 2, int(7000 - start_time*original_sr): int(7000+end_time*original_sr)])
        
        
data_th = np.array(data_th)


buck_sb = np.array(buckets)[np.where(source == 'sonic boom')[0]]
ind_sb = np.array(indices)[np.where(source == 'sonic boom')[0]]

data_sb = []
for i in tqdm(range(len(buck_sb))):
     data_sb.append(f['/data/'+buck_sb[i]][ind_sb[i], 2, int(7000 - start_time*original_sr): int(7000+end_time*original_sr)])
        
        
data_sb = np.array(data_sb) 











              
tapered_su = apply_cosine_taper(data_su, taper_percent = taper_amount) 
tapered_th = apply_cosine_taper(data_th, taper_percent = taper_amount)  
tapered_sb = apply_cosine_taper(data_sb, taper_percent = taper_amount)  

filtered_su = np.array(butterworth_filter(tapered_su,  low, high, original_sr, num_corners, 'bandpass'))
filtered_th = np.array(butterworth_filter(tapered_th,  low, high, original_sr, num_corners, 'bandpass'))
filtered_sb = np.array(butterworth_filter(tapered_sb,  low, high, original_sr, num_corners, 'bandpass'))

              


su_Z = filtered_su
th_Z = filtered_th
sb_Z = filtered_sb


# Resampling the data
su_Z = np.array([resample_array(arr, original_sr, new_sr) for arr in su_Z])
th_Z = np.array([resample_array(arr, original_sr, new_sr) for arr in th_Z])
sb_Z = np.array([resample_array(arr, original_sr, new_sr) for arr in sb_Z])
       
    
# Normalizing the data.              

su_Z = su_Z/np.max(abs(su_Z), axis = 1)[:, np.newaxis]
th_Z = th_Z/np.max(abs(th_Z), axis = 1)[:, np.newaxis]
sb_Z = sb_Z/np.max(abs(sb_Z), axis = 1)[:, np.newaxis]

              
    
    



cfg_file = tsfel.get_features_by_domain()

    
    
# Extract features for surface event
features_su = pd.DataFrame([])
for i in tqdm(range(len(su_Z))):
    try:
        df = seis_feature.FeatureCalculator(su_Z[i]).compute_features()
        df['serial_no'] = i
        features_su = pd.concat([features_su,df])
    except:
        pass
    
    
# Extract features for thunder
features_th = pd.DataFrame([])
for i in tqdm(range(len(th_Z))):
    
    try:
        df = seis_feature.FeatureCalculator(th_Z[i]).compute_features()
        df['serial_no'] = i
        features_th = pd.concat([features_th,df])
    except:
        pass
    
    
# Extract features for sonic boom
features_sb = pd.DataFrame([])
for i in tqdm(range(len(sb_Z))):
    try:
        df = seis_feature.FeatureCalculator(sb_Z[i]).compute_features()
        df['serial_no'] = i
        features_sb = pd.concat([features_sb,df])
    except:
        pass
    

    
    


              
X = pd.concat([ features_su, features_th, features_sb])
y =  ['surface']*len(features_su)+['thunder']*len(features_th)+['sonic']*len(features_sb)
X['source'] = y


X.to_csv('/home/ak287/PNW_Seismic_Event_Classification/extracted_features/physical_features_exotic_P_'+str(start_time)+'_'+str(end_time)+'_F_'+str(int(low))+'_'+str(int(high))+'_'+str(new_sr)+'.csv')

