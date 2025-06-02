# === Standard Libraries ===
import os
import sys
import random
import json
from typing import Any

# === Scientific Libraries ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# === Signal Processing ===
from scipy import signal
from scipy.signal import butter, filtfilt, correlate

# === Seismology Libraries ===
import obspy
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

# === Machine Learning Libraries ===
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

# === PyTorch ===
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# === File Handling ===
import h5py
from sklearn.model_selection import train_test_split

# === Seismology Client ===
client = Client('IRIS')




import json

FS = 50
X = []
y = []
train_split = 0
val_split = 0
test_split = 0
batch_size = 0
criterion=nn.CrossEntropyLoss()
batch_size = 128



class WaveformPreprocessor:
    def __init__(self, input_fs=100, target_fs=50, lowcut=1, highcut=20, order=4, taper_alpha=0.1):
        self.input_fs = input_fs
        self.target_fs = target_fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
        self.taper_alpha = taper_alpha

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Preprocess a single waveform: detrend, taper, filter, resample, normalize.
        Input:  (C, T) or (1, C, T)
        Output: (C, T_new)
        """
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)  # → (1, C, T)
        
        x = waveform.clone()

        x = self._linear_detrend(x)
        x = self._taper_tukey(x, alpha=self.taper_alpha)
        x = self._bandpass_filter(x, fs=self.input_fs, lowcut=self.lowcut, highcut=self.highcut, order=self.order)
        x = self._resample(x, self.input_fs, self.target_fs)
        x = self._normalize_per_trace(x)

        return x.squeeze(0)  # → (C, T_new)

    def _linear_detrend(self, batch: torch.Tensor) -> torch.Tensor:
        time = torch.arange(batch.shape[-1], dtype=batch.dtype, device=batch.device)
        time_mean = time.mean()
        time_var = ((time - time_mean) ** 2).sum()
        slope = ((batch * (time - time_mean)).sum(dim=-1, keepdim=True)) / time_var
        intercept = batch.mean(dim=-1, keepdim=True) - slope * time_mean
        trend = slope * time + intercept
        return batch - trend

    def _taper_tukey(self, batch: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
        tukey_window = scipy.signal.windows.tukey(batch.shape[-1], alpha=alpha)
        window = torch.tensor(tukey_window, dtype=batch.dtype, device=batch.device)
        return batch * window

    def _bandpass_filter(self, batch: torch.Tensor, fs: float, lowcut: float, highcut: float, order: int) -> torch.Tensor:
        numpy_batch = batch.cpu().numpy()
        nyquist = 0.5 * fs
        b, a = scipy.signal.butter(order, [lowcut / nyquist, highcut / nyquist], btype='band')

        filtered = np.zeros_like(numpy_batch)
        for i in range(numpy_batch.shape[0]):
            for j in range(numpy_batch.shape[1]):
                filtered[i, j] = scipy.signal.filtfilt(b, a, numpy_batch[i, j])

        return torch.tensor(filtered, dtype=batch.dtype, device=batch.device)

    def _resample(self, batch: torch.Tensor, fs_in: float, fs_out: float) -> torch.Tensor:
        orig_len = batch.shape[-1]
        new_len = int(orig_len * fs_out / fs_in)
        return F.interpolate(batch, size=new_len, mode='linear', align_corners=False)

    def _normalize_per_trace(self, batch: torch.Tensor) -> torch.Tensor:
        stds = torch.std(torch.abs(batch.reshape(batch.shape[0], -1)), dim=1, keepdim=True)
        stds = stds.view(-1, 1, 1)
        return batch / (stds + 1e-10)

    
    
def extract_waveforms(cat, file_name, start=-20, input_window_length=100, fs=50, number_data=1000,
                      num_channels=3, all_data=False, shifting=True, lowcut=1, highcut=10):
    """
    Extract and preprocess waveform windows using the WaveformPreprocessor.
    """

    random.seed(1234)
    cat = cat.sample(frac=1).reset_index(drop=True)
    cat = cat.reset_index(drop = True)
    

    if all_data:
        number_data = len(cat)

    f = h5py.File(file_name, 'r')

    x = np.zeros(shape=(number_data, 3, int(fs * input_window_length)))
    event_ids = cat['event_id'].values + '_' + cat['station_network_code'].values + '.' + cat['station_code'].values

    if not all_data:
        event_ids = event_ids[:number_data]

    for index in tqdm(range(number_data)):
        bucket, narray = cat.loc[index]['trace_name'].split('$')
        xx, _, _ = [int(i) for i in narray.split(',:')]
        data = f[f'/data/{bucket}'][xx, :, :]  # (3, T)

        original_fs = cat.loc[index, 'trace_sampling_rate_hz']
        trace_len = data.shape[-1]
        window_len = int(original_fs * input_window_length)

        # Determine istart, iend
        if "noise" in event_ids[index].split("_")[1]:
            istart = 0
        else:
            shift_samples = int(np.random.randint(start, -4) * original_fs)

            if np.isnan(cat.loc[index, 'trace_P_arrival_sample']):
                assumed_p_sample = int(7000)
                istart = assumed_p_sample + shift_samples
            else:
                p_sample = int(cat.loc[index, 'trace_P_arrival_sample'])
                istart = p_sample + shift_samples

        istart = max(0, istart)
        iend = istart + window_len
        
        if iend > trace_len:
            istart = max(0, istart - (iend - trace_len))
            iend = window_len

        # Safety check
        if iend - istart != window_len:
            continue

            

        sliced = data[:, istart:iend]
        sliced_tensor = torch.tensor(sliced, dtype=torch.float32)

        processor = WaveformPreprocessor(
            input_fs=original_fs,
            target_fs=fs,
            lowcut=lowcut,
            highcut=highcut
        )

        processed = processor(sliced_tensor)  # (C, T)
        
        if processed.shape[-1] != int(window_len*fs/original_fs):
            print('error')
            continue
            
        x[index, :, :] = processed.numpy()
        
    

    # Drop zero-filled entries
    if num_channels == 1:
        x2 = x[:, 2, :]
        x = x2.reshape(x2.shape[0], 1, x2.shape[1])
        idx = np.where(np.mean(np.abs(x[:, 0, 0:10]), axis=-1) > 0)[0]
    else:
        idx = np.where(np.mean(np.abs(x[:, 0, 0:10]), axis=-1) > 0)[0]

    f.close()
    return x[idx, :, :], event_ids[idx]


def compute_spectrogram(batch: torch.Tensor, fs: int = FS, nperseg: int = 256, overlap: float = 0.5):
    """
    Compute PSD spectrogram (B, C, T) → (B, C, F, T_spec)
    """
    B, C, N = batch.shape
    noverlap = int(nperseg * overlap)
    hop = nperseg - noverlap
    win = torch.hann_window(nperseg, periodic=True, dtype=batch.dtype, device=batch.device)
    
    segs = batch.unfold(-1, nperseg, hop)
    segs = segs - segs.mean(dim=-1, keepdim=True)
    segs = segs * win

    Z = torch.fft.rfft(segs, n=nperseg, dim=-1).permute(0, 1, 3, 2)
    W = win.pow(2).sum()
    psd = (Z.abs() ** 2) / (W * fs)
    if nperseg % 2 == 0:
        psd[..., 1:-1, :] *= 2.0
    else:
        psd[..., 1:, :] *= 2.0

    freqs = torch.fft.rfftfreq(nperseg, 1 / fs).to(batch.device)
    times = (torch.arange(psd.shape[-1], dtype=batch.dtype, device=batch.device) * hop + nperseg // 2) / fs

    return psd, freqs, times



def normalize_spectrogram_minmax(spectrogram: torch.Tensor) -> torch.Tensor:
    """
    Normalize each trace spectrogram (B, C, F, T) independently to [0, 1]
    """
    B, C, F, T = spectrogram.shape
    spec_flat = spectrogram.reshape(B, C, -1)
    min_vals = spec_flat.min(dim=-1, keepdim=True)[0].view(B, C, 1, 1)
    max_vals = spec_flat.max(dim=-1, keepdim=True)[0].view(B, C, 1, 1)
    return (spectrogram - min_vals) / (max_vals - min_vals + 1e-10)




def plot_waveforms(waveform, title):
    
    components = ['Z', 'N', 'E']
    fs = 50  # sampling rate in Hz
    time = np.arange(waveform.shape[1]) / fs  # convert sample index to seconds

    plt.figure(figsize=(12, 8))

    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(time, waveform[i], linewidth=1.2)
        plt.title(f"{components[i]} Component", fontsize=14)
        plt.ylabel("Amplitude", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        if i == 2:
            plt.xlabel("Time (s)", fontsize=12)
        else:
            plt.xticks([])  # hide x-ticks for top plots to reduce clutter

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # adjust layout to fit suptitle
    plt.show()
    
    
  
def return_train_val_loaders(X,y,  num_classes = 4, train_split = train_split, 
                                  val_split = val_split, test_split = test_split, batch_size = batch_size):   
        
        # Make the data a PNWDataSet
        custom_dataset = PNWDataSet(X,y,num_classes)

        # first split train+val
        # Determine the size of the training set
        train_size = int(train_split/100 * len(custom_dataset)) # 70% of the data set
        val_size = int(val_split/100 * len(custom_dataset)) # 20% of the data set
        
        
        train_dataset, val_dataset = random_split(custom_dataset, [train_size, val_size])
        # then split val into val+test

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,drop_last=True)

        print(len(train_loader),len(val_loader))
        
        return train_loader, val_loader
    
    
    
def plot_confusion_matrix_and_cr(model, test_loader, show_plot = True, criterion = criterion, batch_size = batch_size):
    
    """
    inputs
    
    model: A trained neural network model in PyTorch. This model will be evaluated on the test data.
    test_loader: A PyTorch DataLoader containing batches of input data (features) and corresponding labels (ground truth) 
    from the test dataset. It iterates through the test set, providing data to the model for inference.
    inputs: 2D or 3D tensors (depending on the model type, usually spectrograms or seismic waveform windows in your case).
    labels: One-hot encoded labels corresponding to the classification categories (e.g., earthquake, explosion, noise, surface event).
    
    outputs
    Confusion Matrix (cm): A NumPy array representing the confusion matrix. It contains the counts of 
    actual vs. predicted labels for all classes (e.g., earthquakes predicted as explosions, etc.). The confusion matrix helps in identifying misclassification patterns.

    classification Report (report): A dictionary (or DataFrame) output from 
    sklearn.metrics.classification_report, containing precision, recall, F1-score, and support for each class. This provides a more comprehensive evaluation by detailing the performance of the model for each individual class.
    """
    
    
    with torch.no_grad(): # Context-manager that disabled gradient calculation.
        # Loop on samples in test set
        total = 0
        correct = 0
        running_test_loss = 0
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs = inputs.float()
            labels = labels.float()

            outputs = model(inputs)



            running_test_loss += criterion(outputs, labels).item()

            correct += (outputs.argmax(1) == labels.argmax(1)).sum().item()
            total += labels.size(0)


        test_loss = running_test_loss/len(test_loader)
        test_accuracy = 100 * correct / total  
        print('test loss: %.3f and accuracy: %.3f' % ( test_loss,test_accuracy))
        
        
        
        # performance evaluation

        classes = ['earthquake', 'explosion','noise', 'surface']



        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        #plt.style.use('seaborn')

        with torch.no_grad(): # Context-manager that disabled gradient calculation.
            # Loop on samples in test set
            total = 0
            correct = 0
            running_test_loss = 0
            y_pred=np.zeros(len(test_loader)*batch_size)
            y_test=np.zeros(len(test_loader)*batch_size)
            for i,data in enumerate(test_loader):
                inputs, labels = data[0].to(device), data[1].to(device)
                inputs = inputs.float()
                labels = labels.float()

                outputs = model(inputs)
                y_pred[i*batch_size:(i+1)*batch_size]=outputs.argmax(1).cpu().numpy()
                y_test[i*batch_size:(i+1)*batch_size]=labels.argmax(1).cpu().numpy()
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels = classes, yticklabels = classes)
            plt.xlabel('Predicted', fontsize = 15)
            plt.ylabel('Actual', fontsize = 15)
            plt.title('Total samples: '+str(len(y_pred)), fontsize = 20)
            plt.show()
            
            
            
            
            # Calculate the classification report
            from sklearn.metrics import classification_report
            report = classification_report(y_test, y_pred, output_dict=True)

            # Set a pleasing style
            sns.set_style("whitegrid")

            # Create a figure and axes for the heatmap
            plt.figure()
            ax = sns.heatmap(pd.DataFrame(report).iloc[:3, :4], annot=True, cmap='Blues', xticklabels=classes, vmin=0.8, vmax=1)

            # Set labels and title
            ax.set_xlabel('Metrics', fontsize=15)
            ax.set_ylabel('Classes', fontsize=15)
            ax.set_title('Classification Report', fontsize=18)

            # Create a colorbar
            #cbar = ax.collections[0].colorbar
            #cbar.set_ticks([0.5, 1])  # Set custom tick locations
            #cbar.set_ticklabels(['0', '0.5', '1'])  # Set custom tick labels

            # Adjust layout
            plt.tight_layout()

            if show_plot:
            # Show the plot
                plt.show()
            
            
            return cm, report
            
            
            
def train_model(model, train_loader, val_loader,  n_epochs=100,
                 learning_rate=0.001,criterion=nn.CrossEntropyLoss(),
                 augmentation=False,patience=10, model_path = 'trained_models/best_model_'):
    """
    Function to train and evaluate the defined model.

    Parameters:
        model (torch.nn.Module): The neural network model.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.Dataset): Validation dataset.
        test_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer for training the model.
        n_epochs (int): Number of training epochs.
        number_input (int): Number of points in the input data.
        num_channels (int): Number of channels in the input data.

    Returns:
        accuracy_list (list): List of accuracies computed from each epoch.
        train_loss_list (list): List of training losses from each epoch.
        val_loss_list (list): List of validation losses from each epoch.
        y_pred (list): List of predicted values.
        y_true (list): List of true values.
    """

    model_name = str(model).split('(')[0]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # # Save loss and error for plotting
    loss_time = np.zeros(n_epochs)
    val_loss_time = np.zeros(n_epochs)
    val_accuracy_time = np.zeros(n_epochs)

    best_val_loss = float('inf')
    total = 0   # to store the total number of samples
    correct = 0 # to store the number of correct predictions
    
    model_training_time = 0
    for epoch in tqdm(range(n_epochs)):
        running_loss = 0
        
        
        # putting the model in training mode
        model.train()
        
        initial_time = time.time()
        for data in train_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs = inputs.float()

    
            # Data augmentation.
            if augmentation:
                # Find indices of noise labels in the entire data
                inoise = torch.where(labels.argmax(1) == 2)[0]

                # Determine the number of batches
                num_batches = inputs.shape[0]

                # Generate random numbers for augmentation decision and noise scaling
                random_decisions = torch.rand(num_batches, device=device) > 0.5
                noise_scales = torch.rand(num_batches, device=device) / 2

                # Generate a list of unique indices for noise samples
                unique_indices = torch.randperm(len(inoise), device=device)

                # Prepare noise for augmentation
                noises = torch.empty(num_batches, *inputs.shape[1:], device=device)
                for i, idx in enumerate(unique_indices):
                    noise = shuffle_phase_tensor(inputs[inoise[idx], :, :]).to(device)
                    noises[i % num_batches] = noise

                # Apply noise augmentation
                mask = random_decisions.unsqueeze(1).unsqueeze(2)  # Shape: (num_batches, 1, 1)
                scaled_noises = mask * noise_scales.unsqueeze(1).unsqueeze(2) * noises
                inputs += scaled_noises



            # Set the parameter gradients to zero
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # computing the gradients
            loss.backward()

            # updating the parameters
            optimizer.step()

            running_loss += loss.item()
        final_time = time.time() - initial_time
        model_training_time += final_time
        

        # updating the training loss list
        loss_time[epoch] = running_loss/len(train_loader)

        # putting the model in evaluation mode. when you switch to model.eval(),
        #you’re preparing the model to test its knowledge without any further learning or adjustments.
        model.eval()
        
        with torch.no_grad(): # Context-manager that disabled gradient calculation.
            # Loop on samples in test set
            total = 0
            correct = 0
            running_test_loss = 0
            for data in val_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                inputs = inputs.float()
                labels = labels.float()

                outputs = model(inputs)
                running_test_loss += criterion(outputs, labels).item()

                correct += (outputs.argmax(1) == labels.argmax(1)).sum().item()
                total += labels.size(0)

    # Check for improvement
            if running_test_loss/len(val_loader) < best_val_loss:
                best_val_loss = running_test_loss/len(val_loader)
                epochs_no_improve = 0
                # Save the model if you want to keep the best one
                torch.save(model.state_dict(), model_path+model_name+'.pth')
            else:
                epochs_no_improve += 1
                # print(f'No improvement in validation loss for {epochs_no_improve} epochs.')

            if epochs_no_improve == patience:
                # print('Early stopping triggered.')
                
                break

        
        val_loss_time[epoch] = running_test_loss/len(val_loader)

        val_accuracy_time[epoch]=100 * correct / total
        # Print intermediate results on screen
        if (epoch+1) % 10 == 0:
            if val_loader is not None:
                print('[Epoch %d] loss: %.3f - accuracy: %.3f' %
                (epoch + 1, running_loss/len(train_loader), 100 * correct / total))
            else:
                print('[Epoch %d] loss: %.3f' %
                (epoch + 1, running_loss/len(train_loader)))


    # Optionally, load the best model saved
    model.load_state_dict(torch.load(model_path+model_name+'.pth'))
    # testing
        # We evaluate the model, so we do not need the gradient
        
    """
    model.eval() 
    with torch.no_grad(): # Context-manager that disabled gradient calculation.
        # Loop on samples in test set
        total = 0
        correct = 0
        running_test_loss = 0
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs = inputs.float()
            labels = labels.float()

            outputs = model(inputs)
            running_test_loss += criterion(outputs, labels).item()

            correct += (outputs.argmax(1) == labels.argmax(1)).sum().item()
            total += labels.size(0)


        test_loss = running_test_loss/len(test_loader)
        test_accuracy = 100 * correct / total  
        print('test loss: %.3f and accuracy: %.3f' % ( test_loss,test_accuracy))
      
     """

   
    return loss_time, val_loss_time, val_accuracy_time, model_training_time

    
    