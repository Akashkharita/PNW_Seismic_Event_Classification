import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import seaborn as sns
import argparse
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, auc, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from datetime import datetime
import h5py
from scipy import stats, signal
from sklearn.preprocessing import StandardScaler
import obspy
from obspy.geodetics.base import gps2dist_azimuth
from obspy.clients.fdsn import Client
pd.set_option('display.max_columns', None)
from joblib import dump, load
from obspy.signal.filter import envelope
import tsfel

import sys
sys.path.append('../feature_extraction_scripts/physical_feature_extraction_scripts')
import seis_feature
from tsfel import time_series_features_extractor, get_features_by_domain
from datetime import timedelta
import os
sys.path.append('../src')

from utils import apply_cosine_taper, butterworth_filter

import pickle
from zenodo_get import zenodo_get

from multiprocessing import Pool, cpu_count
from scipy.signal import resample
from obspy import UTCDateTime


# Argument parser
parser = argparse.ArgumentParser(description="Seismic Event Classification Script")
parser.add_argument("--new_sr", type=int, default=50, help="Desired sampling rate")
args = parser.parse_args()
new_sr = args.new_sr


## Example usage 
## python3 hyperparameter_tests.py --new_sr 50


def resample_array(arr, original_rate, desired_rate):
    num_samples = len(arr)
    duration = num_samples / original_rate  # Duration of the array in seconds
    new_num_samples = int(duration * desired_rate)
    
    
    ## The scipy signal resample works in the following way - (i) it converts the signal into fourier domain, then depending on
    ## whether we are upsampling or downsampling, it will padd the fourier domain with more zeros (upsampling), or remove some higher frequencies (downsampling) which when converted back into original domain will be downsampled or upsampled without changing its flavor. It likes increasing or decreasing the size of a cake without changing its flavour. 
    return resample(arr, new_num_samples)



def process_file(h5_file, indices, buckets, times, win_before, win_after, pick_time, nos):
    
    ## This function is to retreive the data given the h5 file, and starting time. Since exotic events have different pick times compared to comcat events. 
    
    data = []
    t = []
    with h5py.File(h5_file, 'r') as f:
        for i in tqdm(range(nos)):
            data.append(f['/data/' + buckets[i]][indices[i], 2, pick_time - win_before:pick_time + win_after])
            t.append(UTCDateTime(times[i]))
    return np.array(data), np.array(t)




def extract_features(data, t, cfg_file, fs):
    df = pd.DataFrame([])
    for i in tqdm(range(len(data))):
        try:
            tsfel_features = time_series_features_extractor(cfg_file, data[i], fs=fs, verbose = 0)
            physical_features = seis_feature.FeatureCalculator(data[i], fs=fs).compute_features()
            final_features = pd.concat([tsfel_features, physical_features], axis=1)
            final_features['hod'] = t[i].hour - 8
            final_features['dow'] = t[i].weekday
            final_features['moy'] = t[i].month
            df = pd.concat([df, final_features])
        except:
            pass
  
    return df

def compute_features(win_before=2000, win_after=3000, nos=300, fmin=1, fmax=10, fs=100):
    
    
    ## retreiving the data and metadata information. 
    comcat_file_name = "/data/whd01/yiyu_data/PNWML/comcat_waveforms.hdf5"
    comcat_csv_file = pd.read_csv("/data/whd01/yiyu_data/PNWML/comcat_metadata.csv")

    
    ## comcat_csv_file['trace_name'] contains values like this 'bucket4$0,:3,:15001', in the part after $, first is index, second is channel "ENZ" and third is number of samples. 
    buckets = [name.split('$')[0] for name in comcat_csv_file['trace_name'].values]
    indices = [int(name.split('$')[1].split(',')[0]) for name in comcat_csv_file['trace_name'].values]
    source = comcat_csv_file['source_type'].values

    ## extracting the indices corresponding to explosions and earthquakes respectively. 
    exp_indices = np.where(source == 'explosion')[0]
    eq_indices = np.where(source == 'earthquake')[0]

    data_exp, t_exp = process_file(comcat_file_name, np.array(indices)[exp_indices], np.array(buckets)[exp_indices], 
                                   comcat_csv_file['trace_start_time'].values[exp_indices], win_before, win_after, 5000, nos)
    
    data_eq, t_eq = process_file(comcat_file_name, np.array(indices)[eq_indices], np.array(buckets)[eq_indices], 
                                 comcat_csv_file['trace_start_time'].values[eq_indices], win_before, win_after, 5000, nos)

    exotic_file_name = "/data/whd01/yiyu_data/PNWML/exotic_waveforms.hdf5"
    exotic_csv_file = pd.read_csv("/data/whd01/yiyu_data/PNWML/exotic_metadata.csv")
    
    buckets = [name.split('$')[0] for name in exotic_csv_file['trace_name'].values]
    indices = [int(name.split('$')[1].split(',')[0]) for name in exotic_csv_file['trace_name'].values]
    source = exotic_csv_file['source_type'].values
    
    su_indices = np.where(source == 'surface event')[0]

    data_su, t_su = process_file(exotic_file_name, np.array(indices)[su_indices], np.array(buckets)[su_indices], 
                                 exotic_csv_file['trace_start_time'].values[su_indices], win_before, win_after, 7000, nos)

    noise_file_name = "/data/whd01/yiyu_data/PNWML/noise_waveforms.hdf5"
    noise_csv_file = pd.read_csv("/data/whd01/yiyu_data/PNWML/noise_metadata.csv")

    buckets = [name.split('$')[0] for name in noise_csv_file['trace_name'].values]
    indices = [int(name.split('$')[1].split(',')[0]) for name in noise_csv_file['trace_name'].values]

    data_no, t_no = process_file(noise_file_name, np.array(indices), np.array(buckets), 
                                 noise_csv_file['trace_start_time'].values, win_before, win_after, 5000, nos)

    tp = 10
    nc = 4

    def process_data(data):
        tapered = apply_cosine_taper(data, taper_percent=tp)
        filtered = np.array(butterworth_filter(tapered, fmin, fmax, fs, nc, 'bandpass'))
        return filtered / np.max(np.abs(filtered), axis=1)[:, np.newaxis]

    norm_eq = process_data(data_eq)
    norm_exp = process_data(data_exp)
    norm_su = process_data(data_su)
    norm_no = process_data(data_no)

    norm_eq = np.array([resample_array(arr, 100, fs) for arr in norm_eq])
    norm_exp = np.array([resample_array(arr, 100, fs) for arr in norm_exp])
    norm_su = np.array([resample_array(arr, 100, fs) for arr in norm_su])
    norm_no = np.array([resample_array(arr, 100, fs) for arr in norm_no])
    
    print(len(norm_eq[0]))
    
    cfg_file = tsfel.get_features_by_domain()

    with Pool(cpu_count()) as pool:
        df_eq = pool.apply_async(extract_features, (norm_eq, t_eq, cfg_file, fs)).get()
        df_exp = pool.apply_async(extract_features, (norm_exp, t_exp, cfg_file, fs)).get()
        df_su = pool.apply_async(extract_features, (norm_su, t_su, cfg_file, fs)).get()
        df_no = pool.apply_async(extract_features, (norm_no, t_no, cfg_file, fs)).get()

    return df_eq, df_exp, df_su, df_no






import time
import pickle

def run_compute_features(params):
    results = []
    for param in params:
        # Extract prefix and remove it from the parameter dictionary
        prefix = param.pop("prefix")
        results.append((prefix, compute_features(**param)))
        # Re-add the prefix for any future use of the parameter set
        param["prefix"] = prefix
    return results

# Parameters for different runs
param_sets = [
    {"win_before": 1000, "win_after": 3000, "nos": 3000, "fmin": 1, "fmax": 10, "fs": new_sr, "prefix": "10_30_1_10_"+str(new_sr)},
    {"win_before": 1000, "win_after": 4000, "nos": 3000, "fmin": 1, "fmax": 10, "fs": new_sr, "prefix": "10_40_1_10_"+str(new_sr)},
    {"win_before": 2000, "win_after": 5000, "nos": 3000, "fmin": 1, "fmax": 10, "fs": new_sr, "prefix": "20_50_1_10_"+str(new_sr)},
    {"win_before": 500, "win_after": 2000, "nos": 3000, "fmin": 1, "fmax": 10, "fs": new_sr, "prefix": "05_20_1_10_"+str(new_sr)},
    {"win_before": 5000, "win_after": 10000, "nos": 3000, "fmin": 1, "fmax": 10, "fs": new_sr, "prefix": "50_100_1_10_"+str(new_sr)},
    {"win_before": 1000, "win_after": 10000, "nos": 3000, "fmin": 1, "fmax": 10, "fs": new_sr, "prefix": "10_100_1_10_"+str(new_sr)},
    
    
    
    
    {"win_before": 1000, "win_after": 3000, "nos": 3000, "fmin": 1, "fmax": 15, "fs": new_sr, "prefix": "10_30_1_15_"+str(new_sr)},
    {"win_before": 1000, "win_after": 4000, "nos": 3000, "fmin": 1, "fmax": 15, "fs": new_sr, "prefix": "10_40_1_15_"+str(new_sr)},
    {"win_before": 2000, "win_after": 5000, "nos": 3000, "fmin": 1, "fmax": 15, "fs": new_sr, "prefix": "20_50_1_15_"+str(new_sr)},
    {"win_before": 500, "win_after": 2000, "nos": 3000, "fmin": 1, "fmax": 15, "fs": new_sr, "prefix": "05_20_1_15_"+str(new_sr)},
    {"win_before": 5000, "win_after": 10000, "nos": 3000, "fmin": 1, "fmax": 15, "fs": new_sr, "prefix": "50_100_1_15_"+str(new_sr)},
    {"win_before": 1000, "win_after": 10000, "nos": 3000, "fmin": 1, "fmax": 15, "fs": new_sr, "prefix": "10_100_1_15_"+str(new_sr)},
    
    
    
    {"win_before": 1000, "win_after": 3000, "nos": 3000, "fmin": 0.5, "fmax": 10, "fs": new_sr,  "prefix": "10_30_05_10_"+str(new_sr)},
    {"win_before": 1000, "win_after": 4000, "nos": 3000, "fmin": 0.5, "fmax": 10, "fs": new_sr, "prefix": "10_40_05_10_"+str(new_sr)},
    {"win_before": 2000, "win_after": 5000, "nos": 3000, "fmin": 0.5, "fmax": 10, "fs": new_sr, "prefix": "20_50_05_10_"+str(new_sr)},
    {"win_before": 500, "win_after": 2000, "nos": 3000, "fmin": 0.5, "fmax": 10, "fs": new_sr, "prefix": "05_20_05_10_"+str(new_sr)},
    {"win_before": 5000, "win_after": 10000, "nos": 3000, "fmin": 0.5, "fmax": 10, "fs": new_sr, "prefix": "50_100_05_10_"+str(new_sr)},
    {"win_before": 1000, "win_after": 10000, "nos": 3000, "fmin": 0.5, "fmax": 10, "fs": new_sr, "prefix": "10_100_05_10_"+str(new_sr)},
    
    
    
    
    {"win_before": 1000, "win_after": 3000, "nos": 3000, "fmin": 0.5, "fmax": 15, "fs": new_sr, "prefix": "10_30_05_15_"+str(new_sr)},
    {"win_before": 1000, "win_after": 4000, "nos": 3000, "fmin": 0.5, "fmax": 15, "fs": new_sr, "prefix": "10_40_05_15_"+str(new_sr)},
    {"win_before": 2000, "win_after": 5000, "nos": 3000, "fmin": 0.5, "fmax": 15, "fs": new_sr, "prefix": "20_50_05_15_"+str(new_sr)},
    {"win_before": 500, "win_after": 2000, "nos": 3000, "fmin": 0.5, "fmax": 15, "fs": new_sr, "prefix": "05_20_05_15_"+str(new_sr)},
    {"win_before": 5000, "win_after": 10000, "nos": 3000, "fmin": 0.5, "fmax": 15, "fs": new_sr, "prefix": "50_100_05_15_"+str(new_sr)},
    {"win_before": 1000, "win_after": 10000, "nos": 3000, "fmin": 0.5, "fmax": 15, "fs": new_sr, "prefix": "10_100_05_15_"+str(new_sr)}
]

# Run the computations and measure time
all_results = []
for param_set in param_sets:
    init_time = time.time()
    results = run_compute_features([param_set])
    for prefix, result in results:
        a, b, c, d = result
        print(f"{prefix}: {time.time() - init_time}")
        all_results.append((prefix, result))


        
        
# Save the results to disk
with open('../results/results_'+str(new_sr)+'.pkl', 'wb') as f:
    pickle.dump(all_results, f)

