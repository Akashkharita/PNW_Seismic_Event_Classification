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
sys.path.append('/home/ak287/PNW_Seismic_Event_Classification/common_scripts')

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
     data_exp.append(f['/data/'+buck_exp[i]][ind_exp[i], 2, int(5000 - start_time*100): int(5000+end_time*100)])
               
data_exp = np.array(data_exp)




tapered_exp = apply_cosine_taper(data_exp, taper_percent = taper_amount)  
## Mention the bandpass filter frequencies. 
filtered_exp = np.array(butterworth_filter(data_exp, low, high, original_sr, num_corners, 'bandpass'))

exp_Z = filtered_exp         
# Normalizing the data.              
exp_Z = exp_Z/np.max(abs(exp_Z), axis = 1)[:, np.newaxis]

# Resampling the data
exp_Z = np.array([resample_array(arr, original_sr, new_sr) for arr in exp_Z])
              
    
    
    
    


cfg_file = tsfel.get_features_by_domain()


# Extract features for explosion
# Make sure to change the fs parameter to the modified sampling rate. 

features_expz = pd.DataFrame([])
for i in tqdm(range(len(exp_Z))):
    try:
        df = time_series_features_extractor(cfg_file, exp_Z[i], fs= new_sr, verbose = 0)
        df['serial_no'] = i
        features_expz = pd.concat([features_expz,df])
        
    except:
        pass
    
    


              
X = pd.concat([features_expz])
y = ['explosion']*len(features_expz) 
X['source'] = y


X.to_csv('/home/ak287/PNW_Seismic_Event_Classification/extracted_features/tsfel_features_explosion_P_'+str(start_time)+'_'+str(end_time)+'_F_'+str(int(low))+'_'+str(int(high))+'_'+str(new_sr)+'.csv')
