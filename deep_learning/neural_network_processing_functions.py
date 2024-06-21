import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy import signal
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
from scipy.signal import resample
from torch.utils.data import random_split


class PNWDataSet(Dataset): # create custom dataset
    def __init__(self, data,labels): # initialize
        self.data = data 
        self.labels = labels 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample_data = self.data[index]
        sample_labels = self.labels[index]
        return torch.Tensor(sample_data),(sample_labels) # return data as a tensor

def extract_waveforms(cat, file_name, start=7000, num_features=5000, before=5000, after=10000, number_samples=1000, num_channels=1, all_samples=False, shifting=True):
    
    """
    This is a function defined to extract the waveforms stored in the disk. 
    Inputs:
    cat -  Catalog containing metadata of the events, so we can extract the data using the bucket information
    file_name - path of the h5py file containing the data
    start - origin or first arrival time
    num_features - window length to extract
    before - number of samples to take before the arrival time
    after - number of samples to take after the arrival time.
    num_samples - no. of events per class to extract
    num_channels - no. of channels per event to extract, if set 1, will extract Z component, if set any other number, will extract - ZNE component. 
    all_samples - if true, will extract all the samples corresponding of a given class
    shifting - if true, will extract windows randomly starting between P-5, P-20. The random numbers follow a gaussian distribution. 
    Outputs:
    
    """   
    
    
    # This line initializes empty lists to store waveform data (not traces as in obspy definition)
    st = []    
    # This line initializes empty list to store corresponding event ids. 
    event_ids = []    
    
    # This line opens an HDF5 file in read only mode, the with statement ensures that the file is properly
    # closed after the block of code is executed. 
    with h5py.File(file_name, 'r') as f:
        cat_trace = cat['trace_name'].values        
        
        # If all_samples flag is true, it assigns the values equal to the length of cat_trace to number_samples
        if all_samples:
            number_samples = len(cat_trace)
            
        # Generates a list of random integers between 500 and 2000 (inclusive) if shifting flag is true, otherwise
        # it will generate a list of before equal to number_samples in the length. 
        
        # Note that the np.full function is defined to create a numpy array of specific shape and fill it with a constant value. 
        random_integer_list = np.random.randint(500, 2001, size=number_samples) if shifting else np.full(number_samples, before)
        
        # Note - so since we are taking the first number_samples from the dataset mainly for training,
        # it may include some temporal bias, in future. a to-do will be to randomize this extraction. 
        for i in tqdm(range(number_samples)):
            
            # taking the before samples
            before = random_integer_list[i]
            
            # taking the after samples
            after = num_features - before
            
            # so this code is taking the trace information and splitting it using the $ delimiter
            # because the trace bucket and index are split. 
            
            
            ## here is really a random sampling.
            ii = np.random.randint(len(cat_trace))
            
            trace_info = cat_trace[ii].split('$')
            
            # storing the bucket information
            bucket = trace_info[0]
            
            # storing the index information, 
            ind = int(trace_info[1].split(',')[0])

            if num_channels == 1:
                z_component = f['/data/'+bucket][ind, 2, start - before: start + after]
                
                # This is a kind of quality check applied on the data. 
                # we can also apply some other kind of quality check at this stage. 
                if np.sum(z_component) != 0:
                    event_ids.append(cat['event_id'].values[ii])
                    st.append(z_component)
            else:
                trace_data = f['/data/'+bucket][ind, :, start - before: start + after]
                if np.sum(trace_data) != 0:
                    event_ids.append(cat['event_id'].values[ii])
                    st.append(trace_data)

    return st, event_ids


              
def apply_cosine_taper(arrays, taper_percent=10):
    """
    Function to apply cosine tapering to arrays.
    This is an important operation to do before filtering.

    arrays - list of arrays, each array representing a waveform.
             All the arrays must have the same shape.

    taper_percent - Amount of tapering we want.
    """

    tapered_arrays = []

    num_samples = arrays.shape[-1]  # Assuming each sub-array has the same length

    for array_set in arrays:
        tapered_array_set = []

        for array in array_set:
            taper_length = int(num_samples * taper_percent / 100)
            taper_window = np.hanning(2 * taper_length)

            tapered_array = array.copy()
            tapered_array[:taper_length] = tapered_array[:taper_length] * taper_window[:taper_length]
            tapered_array[-taper_length:] = tapered_array[-taper_length:] * taper_window[taper_length:]

            tapered_array_set.append(tapered_array)

        tapered_arrays.append(np.array(tapered_array_set))

    return np.array(tapered_arrays)


def butterworth_filter(arrays, lowcut, highcut, fs, num_corners, filter_type='bandpass'):
    """
    Apply a Butterworth filter (bandpass, highpass, or lowpass) to each array in an array of arrays.

    Parameters:
        arrays (numpy array): Array of shape (N, C, M) where N is the number of samples, C is the number of channels,
                              and M is the number of data points.
        lowcut (float): Lower cutoff frequency in Hz.
        highcut (float): Upper cutoff frequency in Hz.
        fs (float): Sampling frequency in Hz.
        num_corners (int): Number of corners (filter order).
        filter_type (str, optional): Type of filter ('bandpass', 'highpass', or 'lowpass'). Default is 'bandpass'.

    Returns:
        numpy array: Filtered array with the same shape as the input.
    """
    
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


    filtered_arrays = []
    for data in arrays:
        filtered_channels = []
        # Iterate over channels
        for channel in data:
            # Apply the filter to the channel using lfilter
            filtered_channel = signal.filtfilt(b, a, channel)
            filtered_channels.append(filtered_channel)
        # Stack the filtered channels along the second dimension (axis 1)
        filtered_data = np.stack(filtered_channels)
        filtered_arrays.append(filtered_data)

    return np.array(filtered_arrays)



def normalize_arrays_by_max(arrays):
    """
    Normalize each individual array in an array of shape (N, C, M) by its maximum value.

    Parameters:
        arrays (numpy array): Array of shape (N, C, M) where N is the number of samples, C is the number of channels,
                              and M is the number of data points.

    Returns:
        numpy array: Normalized array with the same shape as the input.
    """
    normalized_arrays = []

    for data in arrays:
        normalized_channels = []

        # Iterate over channels
        for channel in data:
            # Normalize the channel by its maximum value
            max_value = np.max(np.abs(channel))
            normalized_channel = channel / max_value
            normalized_channels.append(normalized_channel)

        # Stack the normalized channels along the second dimension (axis 1)
        normalized_data = np.stack(normalized_channels)
        normalized_arrays.append(normalized_data)

    return np.array(normalized_arrays)


def retain_nonzero_arrays(arrays):
    """
    Normalize each individual array in an array of shape (N, C, M) by its maximum value.

    Parameters:
        arrays (numpy array): Array of shape (N, C, M) where N is the number of samples, C is the number of channels,
                              and M is the number of data points.

    Returns:
        numpy array: Normalized array with the same shape as the input.
    """
    nonzero_arrays = []
    for data in arrays:
        if np.sum(data[0] != 0):
           nonzero_arrays.append(data)     
    return np.array(nonzero_arrays)



def extract_datasets(data_noise="/data/whd01/yiyu_data/PNWML/noise_waveforms.hdf5", medata_noise="/data/whd01/yiyu_data/PNWML/noise_metadata.csv", data_comcat=  "/data/whd01/yiyu_data/PNWML/comcat_waveforms.hdf5", metadata_comcat="/data/whd01/yiyu_data/PNWML/comcat_metadata.csv", data_exotic="/data/whd01/yiyu_data/PNWML/exotic_waveforms.hdf5",metadata_exotic="/data/whd01/yiyu_data/PNWML/exotic_metadata.csv",before = 5000, after = 10000, num_samples = 1000, batch_size = 32, num_channels = 1, train_size = 2400, test_size = 600, num_features = 5000, shifting = True, all_samples = True):

    
    """
    This is a function to extract train, test and validation dataset in tensor format as required by Pytorch
    Here is a description of the parameters: - 
    
    Parameters
    -----------
    
    before: if shifting is not true, samples will be extracted (P-50), 
    where P refers to the starttime of the P/pick time of the event.
    The shifting helps in generalizability of the model. The samples will be picked randomly from
    (P-20, P-5)    
    after: if shifting is not true, samples will be extracted (P+100)    
    num_samples: number of samples per classs to extract    
    batch size: batch size of the samples that would be loaded in one iteration from the dataloader    
    num_channels: 1, Currently just using the Z component, but we can use multiple channels. 
    train_size: its the number of elements (per class) in the training dataset on the first split.  (splitting the dataset into train and temp)    
    test_size: its the number of elements (per class) in the testing dataset on the second split. (splitting the temp further into test and val)
    num_features: The number of features or window length.     
    shifting: If true, the samples will be extracted randomly from P-5, P-20s    
    all_samples: if true, all the samples will be loaded in each class
    
    
    
    Returns
    -------
    train_dataset: the dataset containing the examples on which the model was trained. 
    It contains the features and corresponding samples, the size of the training dataset would be  4*train_size
    
    train_dataloader: The dataloader is required when training the model, taking a batch of the samples at a given time. 
    
    y_train: the training labels,  
    
    test_dataset, test_dataloader, y_test: self explanatory, the size of test will be determined by the 4*test_size parameter. 
    
    val_dataset, val_dataloader, y_val: self explanatory, the size of validation set would be determined as 
      (total_samples - 4*train_size - 4*test_size)
     
    """
    
        
    noise_metadata = pd.read_csv(metadata_noise)
    noise_metadata['event_id'] = [noise_metadata['trace_start_time'][i]+'_noise' for i in range(len(noise_metadata))]

    # accessing the data files
    comcat_metadata = pd.read_csv(metadata_comcat)

    # accessing the data files
    exotic_metadata = pd.read_csv(metadata_exotic)
    
    cat_exp = comcat_metadata[comcat_metadata['source_type'] == 'explosion']
    cat_eq = comcat_metadata[comcat_metadata['source_type'] == 'earthquake']
    cat_su = exotic_metadata[exotic_metadata['source_type'] == 'surface event']
    
    
    #extract wavefpr,s
    ## So in the below I am taking a 50s window which starts anywhere randomly from (P-20, P-5) - 
    ## a is a list of obspy traces, b is a list of eventid
    
    a_noise, b_noise = extract_waveforms(noise_metadata, data_noise, num_features = num_features, start = 5000, before = before, after = after, number_samples = num_samples, num_channels = num_channels, shifting = shifting, all_samples = all_samples)
    
    a_exp, b_exp = extract_waveforms(cat_exp, data_comcat, num_features = num_features, start = 5000, before = before, after = after, number_samples = num_samples, num_channels = num_channels, shifting = shifting, all_samples = all_samples)
    
    a_eq, b_eq = extract_waveforms(cat_eq, data_comcat, num_features = num_features,  start = 5000, before = before, after = after, number_samples = num_samples, num_channels = num_channels, shifting = shifting, all_samples = all_samples)
    
    a_su, b_su = extract_waveforms(cat_su, data_exotic, num_features = num_features, start = 7000, before = before, after = after, number_samples = num_samples, num_channels = num_channels, shifting = shifting, all_samples = all_samples)
    
    
    
    
    # stacking the data
    d_noise = np.stack(a_noise)
    d_exp = np.stack(a_exp)
    d_eq = np.stack(a_eq)
    d_su = np.stack(a_su)

    
    if num_channels == 1:
        d_noise = d_noise[:, np.newaxis, :]
        d_exp = d_exp[:, np.newaxis, :]
        d_eq = d_eq[:, np.newaxis, :]
        d_su = d_su[:, np.newaxis, :]
    
    # remove zero data, which is only necessary if we just use single-comp sensors, and I am not even sure it would be necessary actually...
    d_noise = retain_nonzero_arrays(d_noise)
    d_exp = retain_nonzero_arrays(d_exp)
    d_eq = retain_nonzero_arrays(d_eq)
    d_su = retain_nonzero_arrays(d_su)
    
    X = np.vstack([d_noise, d_exp, d_eq, d_su])
    
    tapered = apply_cosine_taper(X)
    filtered = butterworth_filter(tapered, lowcut = 1, highcut = 10, fs = 100, num_corners = 4, filter_type='bandpass')
    data = normalize_arrays_by_max(filtered)
  
    
    # labels to encode   
    y = ['noise']*len(d_noise)+['explosion']*len(d_exp)+['earthquake']*len(d_eq)+['surface']*len(d_su)
    event_ids = np.hstack([b_noise, b_exp, b_eq, b_su])
    y_encoded = label_encoder.fit_transform(y)
       
    
    # Make the data a PNWDataSet
    custom_dataset = PNWDataSet(data,y)
    train_dataset_torch, val_dataset = random_split(custom_dataset, [train_size, test_size])
    
    
    # Split data into training and testing sets
    train_dataset, test_dataset = random_split(custom_dataset, [train_size, test_size])

    train_data, temp_data, y_train, y_temp = train_test_split(data, y,  test_size= 4*num_samples - 4*train_size, random_state=42)
    val_data, test_data, y_val, y_test = train_test_split(temp_data, y_temp, test_size = 4*test_size, random_state = 42)

    
    label_encoder = LabelEncoder()

    # Create TensorDataset and DataLoader for training data
    train_labels = y_train # Define your training labels
    train_labels_encoded = label_encoder.fit_transform(train_labels)
    train_labels = torch.Tensor(train_labels_encoded) # Suitable for use in pytorch. 

    train_data = torch.Tensor(train_data)

    train_dataset = TensorDataset(train_data, train_labels) # Combines the training data and the numerical labels into 
    # single dataset. 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # it allows you to efficiently iterate 
    # through the data in mini-batches during training. 

    
    # Similarly, create a DataLoader for validation data
    val_labels = y_val
    val_labels_encoded = label_encoder.fit_transform(val_labels)
    val_labels = torch.Tensor(val_labels_encoded)
    
    val_data = torch.Tensor(val_data)
    val_dataset = TensorDataset(val_data, val_labels)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)


    # Similarly, create a DataLoader for testing data
    test_labels = y_test  # Define your testing labels
    test_labels_encoded = label_encoder.fit_transform(test_labels)
    test_labels = torch.Tensor(test_labels_encoded)

    test_data = torch.Tensor(test_data)

    test_dataset = TensorDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    
    
    return train_dataset, train_loader, y_train, test_dataset, test_loader, y_test, val_dataset, val_loader, y_val, event_ids
    
    


def train_model(model, train_loader, val_dataset, val_loader, optimizer, n_epochs=100, batch_size=32, num_features=15000, num_channels=3,criterion=nn.CrossEntropyLoss()):
    """
    Function to train and evaluate the defined model.

    Parameters:
        model (torch.nn.Module): The neural network model.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_dataset (torch.utils.data.Dataset): Validation dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer for training the model.
        n_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        number_features (int): Number of features in the input data.
        num_channels (int): Number of channels in the input data.

    Returns:
        accuracy_list (list): List of accuracies computed from each epoch.
        train_loss_list (list): List of training losses from each epoch.
        val_loss_list (list): List of validation losses from each epoch.
        y_pred (list): List of predicted values.
        y_true (list): List of true values.
    """
    # Check if a GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    N_test = len(val_dataset)

    # to store the accuracies computed from each epoch.
    accuracy_list = []

    # to store the losses from each epoch.
    train_loss_list = []
    val_loss_list = []

    # to store the predicted values
    y_pred = []
    y_true = []

    for epoch in tqdm(range(n_epochs)):
        train_loss_data = 0
        for x, y in train_loader:
            # setting the variable to run on GPU.
            x, y = x.to(device), y.to(device)

            # setting the model in training mode.
            model.train()

            # setting the gradients to zero.
            optimizer.zero_grad()

            # computing the output
            z = model(x.view(x.shape[0], num_channels, num_features))

            # converting the labels to standard type.
            y = torch.tensor(y, dtype=torch.long)

            # computing the loss
            loss = criterion(z, y)

            # computing the gradients
            loss.backward()

            # updating the parameters
            optimizer.step()

            train_loss_data += loss.data.cpu().numpy()

        # updating the training loss list
        train_loss_list.append(train_loss_data / len(train_loader))

        val_loss_data = 0
        correct = 0
        for x_test, y_test in val_loader:
            # setting the model in evaluation mode.
            model.eval()

            # pass the data to GPU
            x_test, y_test = x_test.to(device), y_test.to(device)

            # computing the output.
            z = model(x_test.view(x_test.shape[0], num_channels, num_features))
            
             # Convert z to 'Float' data type and y_test to 'Long' data type
            z = z.to(torch.float)
            y_test = y_test.to(torch.long)

            # computing the loss
            val_loss = criterion(z, y_test)
            val_loss_data += val_loss.data.cpu().numpy()

            # computing the number of correct predictions.
            _, yhat = torch.max(z.data, 1)
            correct += (yhat == y_test).sum().item()
            y_pred.append(yhat.cpu().numpy())
            y_true.append(y_test.cpu().numpy())

        # updating the validation loss list
        val_loss_list.append(val_loss_data / len(val_loader))

        accuracy = correct / N_test
        accuracy_list.append(accuracy)

    return accuracy_list, train_loss_list, val_loss_list, y_pred, y_true


import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Rest of the file code...








def plot_train_val_loss(train_loss, val_loss, title = 'Archtime'):
    
    plt.figure(figsize = [8,5])

    # Plotting the data with labels and colors
    plt.plot(train_loss, label='Training Loss '+title, color='blue', linestyle='-', linewidth=2)
    plt.plot(val_loss, label='Validation Loss '+title, color='orange', linestyle='-', linewidth=2)

    # Adding labels and title
    plt.xlabel('Epoch #', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss Over Epochs', fontsize=14)

    # Adding legend
    plt.legend(loc='upper right', fontsize=10)

    # Adding grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)

    # Show plot
    plt.show()

    
    
def plot_accuracy(acc1, acc2, label1, label2):
    
    plt.figure(figsize = [10,5])

    # Plotting the data with labels and colors
    plt.plot(acc1, label=label1, color='orange', linestyle='-', linewidth=2)
    plt.plot(acc2, label=label2, color='blue', linestyle='-', linewidth=2)

    # Adding vertical lines at the peak accuracy points
    max_accuracy_archtime = np.argmax(acc1)
    max_accuracy_archtime_do = np.argmax(acc2)
    plt.axvline(max_accuracy_archtime, 0, max(acc1), color='orange', linestyle='--', linewidth=1.5)
    plt.axvline(max_accuracy_archtime_do, 0, max(acc2), color='blue', linestyle='--', linewidth=1.5)

    # Adding labels and title
    plt.ylabel('Validation Accuracy', fontsize=12)
    plt.xlabel('Epoch #', fontsize=12)
    plt.title('Validation Accuracy Over Epochs', fontsize=14)

    # Adding legend
    plt.legend(loc='lower right', fontsize=10)

    # Adding grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)

    # Adding annotations for maximum accuracies
    plt.text(max_accuracy_archtime, max(acc1), f'Max Acc: {max(acc1):.2f}', 
             verticalalignment='bottom', horizontalalignment='center', color='orange', fontsize=10)
    plt.text(max_accuracy_archtime_do, max(acc2), f'Max Acc: {max(acc2):.2f}', 
             verticalalignment='bottom', horizontalalignment='center', color='blue', fontsize=10)

    # Show plot
    plt.show()

    
    
    
def extract_datasets_for_test(noise_csv_file, cat_exp, cat_eq, cat_su,  before = 5000, after = 10000, num_samples = 1000, batch_size = 32, num_channels = 1, num_features = 5000, shifting = True, all_samples = True):

    
    
    """
    This is a function to extract train, test and validation dataset in tensor format as required by Pytorch
    Here is a description of the parameters: - 
    
    
    
    
    Parameters
    -----------
    
    before: if shifting is not true, samples will be extracted (P-50), 
    where P refers to the starttime of the P/pick time of the event.
    The shifting helps in generalizability of the model. The samples will be picked randomly from
    (P-20, P-5)
    
    after: if shifting is not true, samples will be extracted (P+100)
    
    num_samples: number of samples per classs to extract
    
    batch size: batch size of the samples that would be loaded in one iteration from the dataloader
    
    num_channels: 1, Currently just using the Z component, but we can use multiple channels. 
    
    train_size: its the number of elements (per class) in the training dataset on the first split.  (splitting the dataset into train and temp)
    
    test_size: its the number of elements (per class) in the testing dataset on the second split. (splitting the temp further into test and val)
    
    num_features: The number of features or window length. 
    
    shifting: If true, the samples will be extracted randomly from P-5, P-20s
    
    all_samples: if true, all the samples will be loaded in each class
    
    
    
    Returns
    -------
    train_dataset: the dataset containing the examples on which the model was trained. 
    It contains the features and corresponding samples, the size of the training dataset would be  4*train_size
    
    train_dataloader: The dataloader is required when training the model, taking a batch of the samples at a given time. 
    
    y_train: the training labels, 
    
    
    
    test_dataset, test_dataloader, y_test: self explanatory, the size of test will be determined by the 4*test_size parameter. 
    
    
    
    val_dataset, val_dataloader, y_val: self explanatory, the size of validation set would be determined as 
    
    (total_samples - 4*train_size - 4*test_size)
     
    """
    
    
    
    
    
    
    
    noise_file_name = "/data/whd01/yiyu_data/PNWML/noise_waveforms.hdf5"
    # accessing the data files
    comcat_file_name = "/data/whd01/yiyu_data/PNWML/comcat_waveforms.hdf5"
    # accessing the data files
    exotic_file_name = "/data/whd01/yiyu_data/PNWML/exotic_waveforms.hdf5"

    
    
    
    noise_csv_file = noise_csv_file.reset_index(drop = True)
    cat_exp = cat_exp.reset_index(drop = True)
    cat_eq  = cat_eq.reset_index(drop = True)
    cat_su = cat_su.reset_index(drop = True)
    
    
    noise_csv_file['event_id'] = [noise_csv_file['trace_start_time'][i]+'_noise' for i in range(len(noise_csv_file))]


    
   
    
    
    
    
    
    ## So in the below I am taking a 50s window which starts anywhere randomly from (P-20, P-5) - 
    ## so the first output of the below is the array containing trace data and second output is the corresponding event id
    
    a_noise, b_noise = extract_waveforms(noise_csv_file, noise_file_name, num_features = num_features, start = 5000, before = before, after = after, number_samples = num_samples, num_channels = num_channels, shifting = shifting, all_samples = all_samples)
    a_exp, b_exp = extract_waveforms(cat_exp, comcat_file_name, num_features = num_features, start = 5000, before = before, after = after, number_samples = num_samples, num_channels = num_channels, shifting = shifting, all_samples = all_samples)
    a_eq, b_eq = extract_waveforms(cat_eq, comcat_file_name, num_features = num_features,  start = 5000, before = before, after = after, number_samples = num_samples, num_channels = num_channels, shifting = shifting, all_samples = all_samples)
    a_su, b_su = extract_waveforms(cat_su, exotic_file_name, num_features = num_features, start = 7000, before = before, after = after, number_samples = num_samples, num_channels = num_channels, shifting = shifting, all_samples = all_samples)
    
    
    

    
    # stacking the data
    d_noise = np.stack(a_noise)
    d_exp = np.stack(a_exp)
    d_eq = np.stack(a_eq)
    d_su = np.stack(a_su)

    
    if num_channels == 1:
        d_noise = d_noise[:, np.newaxis, :]
        d_exp = d_exp[:, np.newaxis, :]
        d_eq = d_eq[:, np.newaxis, :]
        d_su = d_su[:, np.newaxis, :]
    
    
    d_noise = retain_nonzero_arrays(d_noise)
    d_exp = retain_nonzero_arrays(d_exp)
    d_eq = retain_nonzero_arrays(d_eq)
    d_su = retain_nonzero_arrays(d_su)
    
    
    
    X = np.vstack([d_noise, d_exp, d_eq, d_su])
    y = ['noise']*len(d_noise)+['explosion']*len(d_exp)+['earthquake']*len(d_eq)+['surface']*len(d_su)
    event_ids_test = np.hstack([b_noise, b_exp, b_eq, b_su])

    tapered = apply_cosine_taper(X)
    filtered = butterworth_filter(tapered, lowcut = 1, highcut = 10, fs = 100, num_corners = 4, filter_type='bandpass')

    
    # This step was causing the problem. it was making the data appear similar. 
    #scaler = MinMaxScaler()
    #data = scaler.fit_transform(filtered)
    
    print(filtered.shape)
    data = normalize_arrays_by_max(filtered)
    #print(y)
    
    data = resample(data, num_features, axis = 2)
    
    
    

    
    
    
    
    label_encoder = LabelEncoder()
    
    
    
    class CustomDataset(Dataset):
        def __init__(self, x_data, y_data, event_ids):
            self.x_data = x_data
            self.y_data = y_data
            self.event_ids = event_ids

        def __len__(self):
            return len(self.x_data)

        def __getitem__(self, idx):
            return self.x_data[idx], self.y_data[idx], self.event_ids[idx]

    
    
    
    
    # Create TensorDataset and DataLoader for training data
    train_labels = y # Define your training labels
    train_labels_encoded = label_encoder.fit_transform(train_labels)
    train_labels = torch.Tensor(train_labels_encoded) # Suitable for use in pytorch. 
    
    
    # Example usage
    x_train = torch.Tensor(data)
    y_train = train_labels

    event_ids_train = event_ids_test

    train_dataset = CustomDataset(x_train, y_train, event_ids_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

 
    
    
    
    
    return train_dataset, train_loader, y_train, event_ids_test
    
    
    



    
def train_model_for_test(model, train_loader, optimizer, n_epochs=10, batch_size=32, num_features=15000, num_channels=3):
    """
    Function to train and evaluate the defined model.

    Parameters:
        model (torch.nn.Module): The neural network model.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_dataset (torch.utils.data.Dataset): Validation dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer for training the model.
        n_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        number_features (int): Number of features in the input data.
        num_channels (int): Number of channels in the input data.

    Returns:
        accuracy_list (list): List of accuracies computed from each epoch.
        train_loss_list (list): List of training losses from each epoch.
        val_loss_list (list): List of validation losses from each epoch.
        y_pred (list): List of predicted values.
        y_true (list): List of true values.
    """
    # Check if a GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
 

    # to store the accuracies computed from each epoch.
    accuracy_list = []

    # to store the losses from each epoch.
    train_loss_list = []
    val_loss_list = []

    # to store the predicted values
    y_pred = []
    y_true = []
    event_ids = []
    for epoch in tqdm(range(n_epochs)):
        train_loss_data = 0
        for x, y, evid in train_loader:
            # setting the variable to run on GPU.
            x, y = x.to(device), y.to(device)

            # setting the model in training mode.
            model.train()

            # setting the gradients to zero.
            optimizer.zero_grad()

            # computing the output
            z = model(x.view(x.shape[0], num_channels, num_features))

            # converting the labels to standard type.
            y = torch.tensor(y, dtype=torch.long)

            # computing the loss
            loss = criterion(z, y)

            # computing the gradients
            loss.backward()

            # updating the parameters
            optimizer.step()

            train_loss_data += loss.data.cpu().numpy()

            
            event_ids.append(evid)
        # updating the training loss list
        train_loss_list.append(train_loss_data / len(train_loader))



    return train_loss_list, event_ids




def test_model(model, test_loader, optimizer, batch_size = 32, number_features = 15000, num_channels = 1):
    
    "function to train and evaluate the defined model"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    #N_test = len(test_dataset)
    

    # to store the predicted values
    y_pred = []
    y_true = []
    
    event_ids = []
    logits = []
    for x_test, y_test, evid in test_loader:

            # setting the model in evaluation mode.
            model.eval()

            # pass the data to gpu device
            x_test, y_test = x_test.to(device), y_test.to(device)

            # computing the output. 
            z = model(x_test.view(x_test.shape[0], num_channels, number_features))

            logits.append(z.detach().cpu().numpy())
            # Notice this step. Computing the max and argmax for z along dimension 1
            _, yhat = torch.max(z.data, 1)

            
            y_pred.append(yhat.cpu().numpy())
            y_true.append(y_test.cpu().numpy())
            
            event_ids.append(evid)

        
    return y_pred, y_true, logits, event_ids #.detach()

