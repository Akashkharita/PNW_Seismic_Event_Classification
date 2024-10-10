import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
import obspy
# from tqdm import tqdm
from glob import glob
# import time
import random
import sys
from datetime import datetime
from tqdm import tqdm

from scipy import stats,signal


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
# from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


import numpy as np
import scipy.signal as signal



# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


criterion=nn.CrossEntropyLoss()
batch_size = 128
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

        plt.style.use('seaborn')

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
            
            

X = []
y = []
train_split = 0
val_split = 0
test_split = 0
batch_size = 0


def return_train_test_val_loaders(X = X,y = y, num_classes = 4, train_split = train_split, 
                                  val_split = val_split, test_split = test_split, batch_size = batch_size):   
        
        # Make the data a PNWDataSet
        custom_dataset = PNWDataSet(X,y,num_classes)

        # first split train+val
        # Determine the size of the training set
        train_size = int(train_split/100 * len(custom_dataset)) # 70% of the data set
        val_size = int(val_split/100 * len(custom_dataset)) # 20% of the data set
        test_size = len(custom_dataset) - train_size - val_size # the rest is test
        
        train_dataset, val_dataset = random_split(custom_dataset, [train_size, test_size+val_size])
        # then split val into val+test
        test_dataset, val_dataset = random_split(val_dataset, [test_size,val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
        print(len(train_loader),len(val_loader),len(test_loader))
        
        
        
        return train_loader, val_loader, test_loader
    
    
    
    
def extract_waveforms(cat, file_name, start=-20, input_window_length=100, fs=50, number_data=1000, num_channels=3, all_data=False, shifting=True,
                     lowcut = 1, highcut = 10):
    
    """
    This is a function defined to extract the waveforms from file of waveforms and a dataframe of metadata. 
    The functions will also filter and resample the data if the data sampling rate is different from the target sampling rate
    The data is shuffled in order it is called from the file it was stored in.
    The data is shuffled in time by allowing a shift in selecting the waveform window with some of the pre-P data.
    The data us normalized to its max(abs) on either component.
    
    Inputs:
    cat -  Catalog containing metadata of the events, so we can extract the data using the bucket information
    file_name - path of the h5py file containing the data
    start - origin or first arrival time
    num_features - window length to extract
    before - number of samples to take before the arrival time
    after - number of samples to take after the arrival time.
    num_samples - no. of events per class to extract
    
    input_window_length: desired window length in seconds
    fs: desired sampling rate.
    num_channels - no. of channels per event to extract, if set 1, will extract Z component, if set any other number, will extract - ZNE component. 
    all_samples - if true, will extract all the samples corresponding of a given class
    shifting - if true, will extract windows randomly starting between P-5, P-20. The random numbers follow a gaussian distribution. 
    Outputs:
    
    """   
    random.seed(1234) # set seed for reproducibility
    cat = cat.sample(frac=1).reset_index(drop=True)
    if all_data:number_data = len(cat) # how many data to include
    # open the file
    f = h5py.File(file_name, 'r')
    x=np.zeros(shape=(number_data, 3, int(fs*input_window_length)))
    event_ids = cat['event_id'].values
    if not all_data:event_ids=event_ids[:number_data]
        
    for index in tqdm(range(number_data)):
        # read data
        # a sample cat['trace_name'] looks like bucket1$0,:3,:15001
        # and for a surface event, it will look like bucket1$0,:3,:18001
        
        bucket, narray = cat.loc[index]['trace_name'].split('$')
        
        # bucket = bucket1,  narray = 1, 0,:3,:15001
        
        xx, _, _ = iter([int(i) for i in narray.split(',:')])
        
        ## xx = 1, _ = 0, _ = 15001
        
        
        data = f['/data/%s' % bucket][xx, :, : ] # get all of the three component data,
        
        
        
  
        nyquist = 0.5 * cat.loc[index,'trace_sampling_rate_hz']
        low = lowcut / nyquist;  high = highcut / nyquist
        b, a = signal.butter(4, [low, high], btype='band')

        # Apply the taper+filter to the signal
        taper = signal.windows.tukey(data.shape[-1],alpha=0.1)
        data = np.array([np.multiply(taper,row) for row in data])
        filtered_signal = np.array([signal.filtfilt(b, a, row) for row in data])

        # resample
        number_of_samples = int(filtered_signal.shape[1] * fs / cat.loc[index,'trace_sampling_rate_hz'])
        data = np.array([signal.resample(row, number_of_samples) for row in filtered_signal])

            
        if event_ids[index].split("_")[-1]!="noise":
            #random start between P-20 and P-5 (upper bound is exclusive in numpy.random.randint)        
            ii = int(np.random.randint(start,-4)*fs)
            
            
            ## in the following code, if the condition is true, it will skip to next index and wont execute any code.          
            if np.isnan(cat.loc[index, 'trace_P_arrival_sample']):continue
            
            
            istart = int(cat.loc[index, 'trace_P_arrival_sample']*fs/cat.loc[index,'trace_sampling_rate_hz']) + ii # start around the P
            iend  = istart + int(fs*input_window_length)
            if istart<0:
                istart = 0
                iend = int(fs*input_window_length)

            if iend>data.shape[-1]:
                istart = istart - (iend-data.shape[-1])
                iend = data.shape[-1]
        else:
            istart=0
            iend=istart+int(fs*input_window_length)

        
        # normalize the data
        # mmax = np.max(np.abs(data[:,istart:iend]))
        mmax = np.std(np.abs(data[:,istart:iend]))
        # store data in big index
        x[index,:,:iend-istart] = data[:,istart:iend]/mmax
        
    if num_channels==1:
        x2 = x[:,2,:]
        del x
        
        x = x2.reshape(x2.shape[0],1, x2.shape[1])
        # remove rows with zeros if there are any
        idx=np.where(np.mean(np.abs(x[:,0,0:10]),axis=-1)>0)[0]
                     
    else:
        # remove rows with zeros if there are any
        idx=np.where(np.mean(np.abs(x[:,0,0:10]),axis=-1)>0)[0]
                     
        
    
    f.close()
    return x[idx,:,:], event_ids[idx]


a = []
fs = 50

def extract_spectrograms(waveforms = a, fs = fs, nperseg=256, overlap=0.5):
    noverlap = int(nperseg * overlap)  # Calculate overlap

    # Example of how to get the shape of one spectrogram
    f, t, Sxx = signal.spectrogram(waveforms[0, 0], nperseg=nperseg, noverlap=noverlap, fs=fs)

    # Initialize an array of zeros with the shape: (number of waveforms, channels, frequencies, time_segments)
    spectrograms = np.zeros((waveforms.shape[0], waveforms.shape[1], len(f), len(t)))

    for i in tqdm(range(waveforms.shape[0])):  # For each waveform
        for j in range(waveforms.shape[1]):  # For each channel
            _, _, Sxx = signal.spectrogram(waveforms[i, j], nperseg=nperseg, noverlap=noverlap, fs=fs)
            spectrograms[i, j] = Sxx  # Fill the pre-initialized array

    print(spectrograms.shape)
    return spectrograms




def train_model(model, train_loader, val_loader, test_loader, n_epochs=100,
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

    for epoch in tqdm(range(n_epochs)):
        running_loss = 0
        
        
        # putting the model in training mode
        model.train()
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

   
    return loss_time, val_loss_time, val_accuracy_time, test_loss,test_accuracy





# adding some more comments here
from torch.utils.data import Dataset
class PNWDataSet(Dataset): # create custom dataset
    def __init__(self, data,labels,num_classes): # initialize
        self.data = data 
        self.labels = labels
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample_data = self.data[index]
        sample_labels = self.labels[index]
        
        # Convert labels to one-hot encoded vectors
        sample_labels = torch.nn.functional.one_hot(torch.tensor(sample_labels), num_classes=self.num_classes)
        
        return torch.Tensor(sample_data), sample_labels.float()  # return data as a tensor
    
    
    
    

import numpy as np
import matplotlib.pyplot as plt

def shuffle_phase_tensor(time_series):
    # Compute the Fourier transform
    new_time_series = torch.zeros(time_series.shape)
    for ichan in range(time_series.shape[0]):
        fourier_tensor = torch.fft.fft(torch.tensor(time_series[ichan,:]).float())
        # Get amplitude and phase
        amp_tensor = torch.abs(fourier_tensor)
        phase_tensor = torch.angle(fourier_tensor)
        
        # Shuffle the phase
        indices = torch.randperm(phase_tensor.size(-1)) 
        # in torch
        phase_tensor[1:len(phase_tensor)//2] = phase_tensor[indices[1:len(phase_tensor)//2]]
        phase_tensor[len(phase_tensor)//2+1:] = -torch.flip(phase_tensor[len(phase_tensor)//2+1:],dims=[0])  # Ensure conjugate symmetry
        
        # Reconstruct the Fourier transform with original amplitude and shuffled phase
        shuffled_fourier_tensor = amp_tensor * torch.exp(1j * phase_tensor)
        
        # Perform the inverse Fourier transform
        new_time_series[ichan,:] = torch.fft.ifft(shuffled_fourier_tensor).real


        window_length = new_time_series[ichan,:].size(-1)  # Taper along the last dimension
        hann_window = torch.hann_window(window_length).to(new_time_series.device)  # Ensure window is on the same device as tensor
        new_time_series[ichan,:] *= hann_window
    
    return new_time_series  # Return the real part



def shuffle_phase(time_series):
    # Compute the Fourier transform
    fourier_transform = np.fft.fft(time_series)
    
    # Get amplitude and phase
    amplitude = np.abs(fourier_transform)
    phase = np.angle(fourier_transform)
  
    # in numpy
    np.random.shuffle(phase[1:len(phase)//2])
    phase[len(phase)//2+1:] = -phase[len(phase)//2-1:0:-1]  # Ensure conjugate symmetry
    # in torch
    # Reconstruct the Fourier transform with original amplitude and shuffled phase
    shuffled_fourier = amplitude * np.exp(1j * phase)
    
    # Perform the inverse Fourier transform
    new_time_series = np.fft.ifft(shuffled_fourier)
    
    return new_time_series.real  # Return the real part
