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

## This is just for earthquakes, since earthquake samples are large in number, it is often recommended to divide the feature extraction in few parts to avoid memory crashes. 
parser.add_argument("--part", type = int, default=1, help = "Refers to the part of the earthquake samples from which the features are extracted")

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
part = args.part






comcat_file_name = "/data/whd01/yiyu_data/PNWML/comcat_waveforms.hdf5"
comcat_csv_file = pd.read_csv("/data/whd01/yiyu_data/PNWML/comcat_metadata.csv")


f = h5py.File(comcat_file_name, 'r')

buckets = [comcat_csv_file['trace_name'].values[i].split('$')[0] for i in range(len(comcat_csv_file))]
indices = [int(comcat_csv_file['trace_name'].values[i].split('$')[1].split(',')[0]) for i in range(len(comcat_csv_file))]

source = comcat_csv_file['source_type'].values



buck_eq = np.array(buckets)[np.where(source == 'earthquake')[0]]
ind_eq = np.array(indices)[np.where(source == 'earthquake')[0]]

data_eq = []




if part == 4:
    for i in tqdm(range(int(50000 * (part - 1)), len(ind_eq))):
        data_eq.append(f['/data/' + buck_eq[i]][ind_eq[i], 2, int(5000 - start_time * original_sr): int(5000 + end_time * original_sr)])
    
elif 0 <= part < 4:
    for i in tqdm(range(int(50000 * (part - 1)), int(50000 * part))):
        data_eq.append(f['/data/' + buck_eq[i]][ind_eq[i], 2, int(5000 - start_time * original_sr): int(5000 + end_time * original_sr)])
                  
else:
    print("part cannot be less than 0 or more than 4")

        

                  
data_eq = np.array(data_eq)


tapered_eq = apply_cosine_taper(data_eq, taper_percent = taper_amount) 
            
filtered_eq = np.array(butterworth_filter(data_eq, low, high, original_sr,  num_corners, 'bandpass'))

eq_Z = filtered_eq


# Resampling the data
eq_Z = np.array([resample_array(arr, original_sr, new_sr) for arr in eq_Z])

              
# Normalizing the data.              
eq_Z = eq_Z/np.max(abs(eq_Z), axis = 1)[:, np.newaxis]




cfg_file = tsfel.get_features_by_domain()

# Extract features for earthquakes
features_eqz = pd.DataFrame([])
for i in tqdm(range(len(eq_Z))):
    try:
    
        df = time_series_features_extractor(cfg_file, eq_Z[i], fs= new_sr, verbose = 0)
        df['serial_no'] = i+int(50000*(part-1))
        features_eqz = pd.concat([features_eqz,df])
        
    except:
        pass
    
    




              
X = pd.concat([features_eqz])
y = ['earthquake']*len(features_eqz) 
X['source'] = y


X.to_csv('/home/ak287/PNW_Seismic_Event_Classification/extracted_features/tsfel_features_earthquake_part_'+str(int(part))+'_P_'+str(start_time)+'_'+str(end_time)+'_F_'+str(int(low))+'_'+str(int(high))+'_'+str(new_sr)+'.csv')










