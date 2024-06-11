import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import obspy
from obspy.signal.filter import envelope
from obspy.clients.fdsn import Client
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, auc, classification_report, confusion_matrix, f1_score
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
#from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler
from scipy import stats, signal
from sklearn.datasets import load_iris
from sklearn.metrics import precision_score, recall_score
from sklearn.feature_selection import RFECV
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
pd.set_option('display.max_columns', None)
from obspy.geodetics.base import gps2dist_azimuth
from datetime import datetime, timedelta
import time
#import lightgbm as lgb
import re
import tsfel
import random
import calendar
import concurrent.futures
import seaborn as sns
from scipy.signal import resample




# We need to change the path to import the seis feature library. 



def resample_array(arr, original_rate, desired_rate):
    num_samples = len(arr)
    duration = num_samples / original_rate  # Duration of the array in seconds
    new_num_samples = int(duration * desired_rate)
    return resample(arr, new_num_samples)




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
              
    
    

def butterworth_filter(arrays, lowcut = 1, highcut = 10, fs = 100, num_corners = 4, filter_type='bandpass'):
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

              
    

trace_cm_phy_tsf_man = []
annot_kws = {"fontsize": 15}


def plot_confusion_matrix(cf = trace_cm_phy_tsf_man, class_labels = ['Earthquake', 'Explosion','Noise','Surface']):


    labels = ['Precision', 'Recall', 'F1-Score']
   
    plt.figure(figsize = [8,6])

    # Set annotation font size within each block
    
    ax = sns.heatmap(cf, annot=True, cmap='Blues', fmt='d', xticklabels = class_labels, yticklabels = class_labels, annot_kws=annot_kws)


    # Set tick label font size
    ax.set_xticklabels(class_labels, fontsize=15)
    ax.set_yticklabels(class_labels, fontsize=15)


    plt.xlabel('Predicted', fontsize = 15)
    plt.ylabel('Actual', fontsize = 15)
    #plt.title('Total samples: '+str(len(y_pred)), fontsize = 20)
    plt.tight_layout()


    
    
trace_report_phy_tsf_man = []

def plot_classification_report(cr = trace_report_phy_tsf_man, class_labels = ['Earthquake', 'Explosion','Noise','Surface']): 


    labels = ['Precision', 'Recall', 'F1-Score']
    #class_labels = ['Earthquake', 'Explosion','Noise','Surface']
    # Set a pleasing style
    sns.set_style("whitegrid")

    # Create a figure and axes for the heatmap
    plt.figure(figsize = [8,6])
    ax = sns.heatmap(pd.DataFrame(cr).iloc[:3, :len(class_labels)], annot=True, cmap='Blues', yticklabels = labels, xticklabels=class_labels, vmin=0.8, vmax=1, annot_kws=annot_kws)

    # Set labels and title
    # Set tick label font size
    ax.set_xticklabels(class_labels, fontsize=15)
    ax.set_yticklabels(labels, fontsize=15)

    ax.set_xlabel('Metrics', fontsize=15)
    ax.set_ylabel('Classes', fontsize=15)
    ax.set_title('Classification Report', fontsize=18)

    # Create a colorbar
    #cbar = ax.collections[0].colorbar
    #cbar.set_ticks([0.5, 1])  # Set custom tick locations
    #cbar.set_ticklabels(['0', '0.5', '1'])  # Set custom tick labels

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()
    
    
    
from math import radians, sin, cos, sqrt, atan2

def calculate_distance(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    
    # Radius of the Earth in kilometers
    R = 6371.0
    
    # Calculate the difference in latitude and longitude
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Calculate the distance using the Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    # Distance in kilometers
    distance = R * c
    
    return distance
    

    
## Some helpful functions in plotting 
def interquartile(df):

    # Set the lower and upper quantile thresholds (25% and 75%)
    lower_quantile = 0.10
    upper_quantile = 0.90

    # Filter the DataFrame based on the quantile range for all columns
    filtered_df = df[
        (df >= df.quantile(lower_quantile)) &
        (df <= df.quantile(upper_quantile))
    ]

    # Drop rows with any NaN values (if needed)
    #filtered_df = filtered_df.dropna(axis = 1)

    return filtered_df

