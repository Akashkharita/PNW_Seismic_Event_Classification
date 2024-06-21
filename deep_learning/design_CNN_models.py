import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# defining a very simple CNN

class SeismicCNN(nn.Module):
    def __init__(self, num_classes=4, num_channels = 3):
        super(SeismicCNN, self).__init__()
        # Define the layers of the CNN architecture
        self.conv1 = nn.Conv1d(in_channels= num_channels, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(79808, 128)  # Adjust input size based on your data
        self.fc2 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    # Lets define a function to visualize the activation as well. 
    
    
# defining a very simple CNN

class SeismicCNN_batch(nn.Module):
    def __init__(self, num_classes=4, num_channels = 3):
        super(SeismicCNN_batch, self).__init__()
        # Define the layers of the CNN architecture
        self.conv1 = nn.Conv1d(in_channels= num_channels, out_channels=32, kernel_size=5)
        self.conv1_bn = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5)
        self.conv2_bn = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(79808, 128)  # Adjust input size based on your data
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.fc2_bn = nn.BatchNorm1d(num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)
        x = self.pool(x)
        
        
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.fc2_bn(self.fc2(x))
        return x

    # Lets define a function to visualize the activation as well. 
    
    
# defining a very simple CNN

class SeismicCNN_batch_do(nn.Module):
    def __init__(self, num_classes=4, num_channels = 3):
        super(SeismicCNN_batch_do, self).__init__()
        # Define the layers of the CNN architecture
        self.conv1 = nn.Conv1d(in_channels= num_channels, out_channels=32, kernel_size=5)
        self.conv1_bn = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5)
        self.conv2_bn = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(79808, 128)  # Adjust input size based on your data
        self.dropout = nn.Dropout(0.25)
        self.dropoutfc = nn.Dropout(0.75)
        self.fc1_bn = nn.BatchNorm1d(128)
        
        self.fc2 = nn.Linear(128, num_classes)
        
        self.fc2_bn = nn.BatchNorm1d(num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)
        x = self.pool(x)
        
        
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.dropoutfc(x)
        x = self.fc2_bn(self.fc2(x))
        return x

    # Lets define a function to visualize the activation as well. 
    
    
    
class SeismicCNN_more(nn.Module):
    def __init__(self, num_classes = 4, num_channels = 1, num_features = 5000, num_additional_conv_layers = 0):
        super(SeismicCNN_more, self).__init__()
        
        
        self.num_additional_conv_layers = num_additional_conv_layers
        
        # Define the initial convolutional layers
        self.conv1 = nn.Conv1d(in_channels = num_channels, out_channels = 32, kernel_size = 5)
        self.conv2 = nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 5)
        
        # Add additional convolutional layers dynamically
        in_channels = 64
        for i in range(num_additional_conv_layers):
            out_channels = 128
            conv_layer = nn.Conv1d(in_channels = in_channels, out_channels = out_channels, kernel_size = 5)
            setattr(self, f'conv{i+3}', conv_layer) #Set attribute dynamically
            in_channels = out_channels
            
            
            
        # Define the pooling layer
        self.pool = nn.MaxPool1d(kernel_size = 2)
        
        self.num_features_after_conv = self.calculate_num_features_after_conv()
        
        self.fc1 = nn.Linear(self.num_features_after_conv, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim = 1)
        
    def forward(self, x):
        x = self.forward_conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        
        for i in range(self.num_additional_conv_layers):
            conv_layer = getattr(self, f'conv{i+3}')
            x = F.relu(conv_layer(x))
            x = self.pool(x)
            
        return x
    
    def calculate_num_features_after_conv(self):
        # Dummy input to calculate the number of features after convolutional layers
        dummy_input = torch.randn(num_channels, 1, num_features)  # Adjust the size based on your input size
        with torch.no_grad():
            conv_output = self.forward_conv(dummy_input)
        num_features_after_conv = conv_output.view(1, -1).size(1)
        return num_features_after_conv   
    
    
    # defining a very simple CNN

class SeismicNet(nn.Module):
    def __init__(self, num_classes=4, num_channels = 3, num_features = 15000):
        super(SeismicNet, self).__init__()
        
        
        self.num_features = num_features
        self.num_channels = num_channels
        # Define the layers of the CNN architecture
        self.conv1 = nn.Conv1d(in_channels= num_channels, out_channels=32, kernel_size= 64, stride = 1, padding = 0)       
        self.pool1 = nn.MaxPool1d(kernel_size = 8, stride = 8)
        self.dropout1 = nn.Dropout(0.15)
        
        
        self.conv2 = nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 32, stride = 1, padding = 0)
        self.pool2 = nn.MaxPool1d(kernel_size = 8, stride = 8)
        self.dropout2 = nn.Dropout(0.35)
        
        
        self.conv3 = nn.Conv1d(in_channels = 64, out_channels = 128, kernel_size = 16, stride = 1, padding = 0)
        self.pool3 = nn.MaxPool1d(kernel_size = 8, stride = 8)
        
        self.conv4 = nn.Conv1d(in_channels = 128, out_channels = 256, kernel_size = 8, stride = 1, padding = 0)
        self.conv5 = nn.Conv1d(in_channels = 256, out_channels = 1401, kernel_size = 16, stride = 1, padding = 0)
        self.num_features_after_conv = self.calculate_num_features_after_conv()
        
        
        self.fc1 = nn.Linear(self.num_features_after_conv, 1500)
        self.dropoutfc = nn.Dropout(0.75)
        
        self.fc2 = nn.Linear(1500,4)
        self.softmax = nn.Softmax(dim = 1)
        
        

    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        #print(x.shape)
        x = self.pool1(x)
        #print(x.shape)
        x = self.dropout1(x)
        
        x = F.relu(self.conv2(x))
        #print(x.shape)
        x = self.pool2(x)
        #print(x.shape)
        x = self.dropout2(x)
        
        x = F.relu(self.conv3(x))
        #print(x.shape)
        x = self.pool3(x)
        #print(x.shape)
        
        x = F.relu(self.conv4(x))
        #print(x.shape)
        x = F.relu(self.conv5(x))
        #print(x.shape)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropoutfc(x)
        x = self.fc2(x)
        #x = self.softmax(x)

        return x

    # Outputs in each steps
    
    
    # Lets define a function to visualize the activation as well.
    def activations(self, x):
        #outputs activation this is not necessary
        z1 = self.pool1(F.relu(self.conv1(x)))
        z2 = self.pool2(F.relu(self.conv2(z1)))
        z3 = self.pool3(F.relu(self.conv3(z2)))
        z4 = F.relu(self.conv4(z3))
        z5 = F.relu(self.conv5(z4))
        z5_flat = z5.view(z5.size(0), -1)  # Flatten z5
        z6 = self.fc1(z5_flat)
        z7 = self.fc2(z6)
        
        return z1, z2, z3, z4, z5, z6, z7
        
        
    def forward_conv(self, x):
        x = F.relu(self.conv1(x))
        #print(x.shape)
        x = self.pool1(x)
        #print(x.shape)
        x = self.dropout1(x)
        
        x = F.relu(self.conv2(x))
        #print(x.shape)
        x = self.pool2(x)
        #print(x.shape)
        x = self.dropout2(x)
        
        x = F.relu(self.conv3(x))
        #print(x.shape)
        x = self.pool3(x)
        #print(x.shape)
        
        x = F.relu(self.conv4(x))
        #print(x.shape)
        x = F.relu(self.conv5(x))
        #print(x.shape)
            
        return x
    
    def calculate_num_features_after_conv(self):
        # Dummy input to calculate the number of features after convolutional layers
        dummy_input = torch.randn(1, self.num_channels, self.num_features)  # Adjust the size based on your input size
        with torch.no_grad():
            conv_output = self.forward_conv(dummy_input)
        num_features_after_conv = conv_output.view(1, -1).size(1)
        return num_features_after_conv   

    



# defining a very simple CNN

class SeismicNet_do(nn.Module):
    def __init__(self, num_classes=4, num_channels = 3, num_features = 15000):
        super(SeismicNet_do, self).__init__()
        
        
        self.num_features = num_features
        self.num_channels = num_channels
        # Define the layers of the CNN architecture
        self.conv1 = nn.Conv1d(in_channels= num_channels, out_channels=32, kernel_size= 64, stride = 1, padding = 0)       
        self.pool1 = nn.MaxPool1d(kernel_size = 8, stride = 8)
        self.dropout1 = nn.Dropout(0.15)
        
        
        self.conv2 = nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 32, stride = 1, padding = 0)
        self.pool2 = nn.MaxPool1d(kernel_size = 8, stride = 8)
        self.dropout2 = nn.Dropout(0.35)
        
        
        self.conv3 = nn.Conv1d(in_channels = 64, out_channels = 128, kernel_size = 16, stride = 1, padding = 0)
        self.pool3 = nn.MaxPool1d(kernel_size = 8, stride = 8)
        self.dropout3  = nn.Dropout(0.35)
        
        self.conv4 = nn.Conv1d(in_channels = 128, out_channels = 256, kernel_size = 8, stride = 1, padding = 0)
        self.dropout4 = nn.Dropout(0.45)
        
        
        self.conv5 = nn.Conv1d(in_channels = 256, out_channels = 1401, kernel_size = 16, stride = 1, padding = 0)
        #self.dropout5 = nn.Dropout(0.35)
        
        self.num_features_after_conv = self.calculate_num_features_after_conv()
        
        
        self.fc1 = nn.Linear(self.num_features_after_conv, 1500)
        self.dropoutfc = nn.Dropout(0.75)
        
        self.fc2 = nn.Linear(1500,4)
        self.softmax = nn.Softmax(dim = 1)
        
        

    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        #print(x.shape)
        x = self.pool1(x)
        #print(x.shape)
        x = self.dropout1(x)
        
        x = F.relu(self.conv2(x))
        #print(x.shape)
        x = self.pool2(x)
        #print(x.shape)
        x = self.dropout2(x)
        
        x = F.relu(self.conv3(x))
        #print(x.shape)
        x = self.pool3(x)
        x = self.dropout3(x)
        #print(x.shape)
        
        x = F.relu(self.conv4(x))
        x = self.dropout4(x)
        
        #print(x.shape)
        x = F.relu(self.conv5(x))
        #x = self.dropout5(x)
        #print(x.shape)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropoutfc(x)
        x = self.fc2(x)
        #x = self.softmax(x)

        return x

    # Outputs in each steps
    
    
    # Lets define a function to visualize the activation as well.
    def activations(self, x):
        #outputs activation this is not necessary
        z1 = self.pool1(F.relu(self.conv1(x)))
        z2 = self.pool2(F.relu(self.conv2(z1)))
        z3 = self.pool3(F.relu(self.conv3(z2)))
        z4 = F.relu(self.conv4(z3))
        z5 = F.relu(self.conv5(z4))
        z5_flat = z5.view(z5.size(0), -1)  # Flatten z5
        z6 = self.fc1(z5_flat)
        z7 = self.fc2(z6)
        
        return z1, z2, z3, z4, z5, z6, z7
        
        
    def forward_conv(self, x):
        x = F.relu(self.conv1(x))
        #print(x.shape)
        x = self.pool1(x)
        #print(x.shape)
        x = self.dropout1(x)
        
        x = F.relu(self.conv2(x))
        #print(x.shape)
        x = self.pool2(x)
        #print(x.shape)
        x = self.dropout2(x)
        
        x = F.relu(self.conv3(x))
        #print(x.shape)
        x = self.pool3(x)
        #print(x.shape)
        
        x = F.relu(self.conv4(x))
        #print(x.shape)
        x = F.relu(self.conv5(x))
        #print(x.shape)
            
        return x
    
    def calculate_num_features_after_conv(self):
        # Dummy input to calculate the number of features after convolutional layers
        dummy_input = torch.randn(1, self.num_channels, self.num_features)  # Adjust the size based on your input size
        with torch.no_grad():
            conv_output = self.forward_conv(dummy_input)
        num_features_after_conv = conv_output.view(1, -1).size(1)
        return num_features_after_conv   

    
 
    
    
    
    
# Arch-time from Deepquake paper. 
# it originally takes 2000 samples as input. 

# defining a very simple CNN

class Archtime(nn.Module):
    def __init__(self, num_classes=4, num_channels = 1, num_features = 5000):
        super(Archtime, self).__init__()
        
        self.num_features = num_features
        self.num_channels = num_channels
        # Define the layers of the CNN architecture
        self.conv1 = nn.Conv1d(in_channels= num_channels, out_channels=64, kernel_size= 10, stride = 4, padding = 0)       
        self.conv2 = nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 10, stride = 2)
        self.conv3 = nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 10, stride = 2)
        self.conv4 = nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 10, stride = 2)
        self.conv5 = nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 10, stride = 2)
        self.conv6 = nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 10, stride = 2)
        self.flatten = nn.Flatten() 
        self.num_features_after_conv = self.calculate_num_features_after_conv()
        
    
        self.fc1 = nn.Linear(self.num_features_after_conv, 64)
        #self.dropoutfc = nn.Dropout(0.75)
        
        self.fc2 = nn.Linear(64,4)
        self.softmax = nn.Softmax(dim = 1)
        
        

    def forward(self, x):
        
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = torch.relu(self.conv6(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        #x = self.softmax(self.fc2(x))

        return x

    # Lets define a function to visualize the activation as well. 
    
    
    
    def forward_conv(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = torch.relu(self.conv6(x))
        x = self.flatten(x)
            
        return x
    
    def calculate_num_features_after_conv(self):
        # Dummy input to calculate the number of features after convolutional layers
        dummy_input = torch.randn(1, self.num_channels, self. num_features)  # Adjust the size based on your input size
        with torch.no_grad():
            conv_output = self.forward_conv(dummy_input)
        num_features_after_conv = conv_output.view(1, -1).size(1)
        return num_features_after_conv   
    
    
    
    
# defining a very simple CNN

class Archtime_do(nn.Module):
    def __init__(self, num_classes=4, num_channels = 1, num_features = 5000):
        super(Archtime_do, self).__init__()
        
        self.num_features = num_features
        self.num_channels = num_channels
        # Define the layers of the CNN architecture
        self.conv1 = nn.Conv1d(in_channels= num_channels, out_channels=64, kernel_size= 10, stride = 4, padding = 0)       
        self.dropout = nn.Dropout(0.25)
        
        self.conv2 = nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 10, stride = 2)
        
        
        self.conv3 = nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 10, stride = 2)
        
        self.conv4 = nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 10, stride = 2)
        self.conv5 = nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 10, stride = 2)
        self.conv6 = nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 10, stride = 2)
        self.flatten = nn.Flatten() 
        self.num_features_after_conv = self.calculate_num_features_after_conv()
        
    
        self.fc1 = nn.Linear(self.num_features_after_conv, 64)
        self.dropoutfc = nn.Dropout(0.75)
        
        self.fc2 = nn.Linear(64,4)
        self.softmax = nn.Softmax(dim = 1)
        
        

    def forward(self, x):
        
        x = torch.relu(self.conv1(x))
        x = self.dropout(x)
        
        x = torch.relu(self.conv2(x))
        x = self.dropout(x)
        
        x = torch.relu(self.conv3(x))
        x = self.dropout(x)
        
        x = torch.relu(self.conv4(x))
        x = self.dropout(x)
        
        x = torch.relu(self.conv5(x))
        x = self.dropout(x)
        
        x = torch.relu(self.conv6(x))
        x = self.dropout(x)
        
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropoutfc(x)
        x = self.fc2(x)
        #x = self.softmax(self.fc2(x))

        return x

    # Lets define a function to visualize the activation as well. 
    
    
    
    def forward_conv(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = torch.relu(self.conv6(x))
        x = self.flatten(x)
            
        return x
    
    def calculate_num_features_after_conv(self):
        # Dummy input to calculate the number of features after convolutional layers
        dummy_input = torch.randn(1, self.num_channels, self. num_features)  # Adjust the size based on your input size
        with torch.no_grad():
            conv_output = self.forward_conv(dummy_input)
        num_features_after_conv = conv_output.view(1, -1).size(1)
        return num_features_after_conv   
    

    
    
 


# WaveDecomPNet from Jiuxin's paper. 
# it originally takes 000 samples as input. 
# defining a very simple CNN

class WaveDecompNet(nn.Module):
    def __init__(self, num_classes=4, num_channels = 1, num_features = 5000):
        super(WaveDecompNet, self).__init__()
        self.num_channels = num_channels
        self.num_features = num_features
        # Define the layers of the CNN architecture
        self.conv1 = nn.Conv1d(in_channels= num_channels, out_channels= 8, kernel_size= 9, stride = 1, padding = 0)       
        self.conv1_bn = nn.BatchNorm1d(8)
        
        self.conv2 = nn.Conv1d(in_channels = 8, out_channels = 8, kernel_size = 9, stride = 2)
        self.conv2_bn = nn.BatchNorm1d(8)
        
        
        self.conv3 = nn.Conv1d(in_channels = 8, out_channels = 16, kernel_size = 7, stride = 1)
        self.conv3_bn = nn.BatchNorm1d(16)
        
        self.conv4 = nn.Conv1d(in_channels = 16, out_channels = 16, kernel_size = 7, stride = 2)
        self.conv4_bn = nn.BatchNorm1d(16)
        
        self.conv5 = nn.Conv1d(in_channels = 16, out_channels = 32, kernel_size = 5, stride = 1)
        self.conv5_bn = nn.BatchNorm1d(32)
        
        self.conv6 = nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 5, stride = 2)
        self.conv6_bn = nn.BatchNorm1d(32)
        
        self.conv7 = nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1)
        self.conv7_bn = nn.BatchNorm1d(64)
        
        
        self.flatten = nn.Flatten() 
        self.num_features_after_conv = self.calculate_num_features_after_conv()
        
    
        self.fc1 = nn.Linear(self.num_features_after_conv, 64)
        self.dropoutfc = nn.Dropout(0.75)
        
        self.fc2 = nn.Linear(64,4)
        self.softmax = nn.Softmax(dim = 1)
        
        

    def forward(self, x):
        
        x = torch.relu(self.conv1_bn(self.conv1(x)))
        x = torch.relu(self.conv2_bn(self.conv2(x)))
        x = torch.relu(self.conv3_bn(self.conv3(x)))
        x = torch.relu(self.conv4_bn(self.conv4(x)))
        x = torch.relu(self.conv5_bn(self.conv5(x)))
        x = torch.relu(self.conv6_bn(self.conv6(x)))
        x = torch.relu(self.conv7_bn(self.conv7(x)))
        x = self.flatten(x)
        #print(x.shape)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        #x = self.softmax(self.fc2(x))

        return x

    # Lets define a function to visualize the activation as well. 
    
    
    
    def forward_conv(self, x):
        x = torch.relu(self.conv1_bn(self.conv1(x)))
        x = torch.relu(self.conv2_bn(self.conv2(x)))
        x = torch.relu(self.conv3_bn(self.conv3(x)))
        x = torch.relu(self.conv4_bn(self.conv4(x)))
        x = torch.relu(self.conv5_bn(self.conv5(x)))
        x = torch.relu(self.conv6_bn(self.conv6(x)))
        x = torch.relu(self.conv7_bn(self.conv7(x)))
        x = self.flatten(x)
            
        return x
    
    def calculate_num_features_after_conv(self):
        # Dummy input to calculate the number of features after convolutional layers
        dummy_input = torch.randn(1, self.num_channels, self.num_features)  # Adjust the size based on your input size
        with torch.no_grad():
            conv_output = self.forward_conv(dummy_input)
        num_features_after_conv = conv_output.view(1, -1).size(1)
        return num_features_after_conv   
    
    
    
    
   


# defining a very simple CNN

class WaveDecompNet_do(nn.Module):
    def __init__(self, num_classes=4, num_channels = 1, num_features = 5000):
        super(WaveDecompNet_do, self).__init__()
        self.num_channels = num_channels
        self.num_features = num_features
        # Define the layers of the CNN architecture
        
        self.dropout = nn.Dropout(0.5)
        self.conv1 = nn.Conv1d(in_channels= num_channels, out_channels= 8, kernel_size= 9, stride = 1, padding = 0)       
        self.conv1_bn = nn.BatchNorm1d(8)
        
        self.conv2 = nn.Conv1d(in_channels = 8, out_channels = 8, kernel_size = 9, stride = 2)
        self.conv2_bn = nn.BatchNorm1d(8)
        
        
        self.conv3 = nn.Conv1d(in_channels = 8, out_channels = 16, kernel_size = 7, stride = 1)
        self.conv3_bn = nn.BatchNorm1d(16)
        
        self.conv4 = nn.Conv1d(in_channels = 16, out_channels = 16, kernel_size = 7, stride = 2)
        self.conv4_bn = nn.BatchNorm1d(16)
        
        self.conv5 = nn.Conv1d(in_channels = 16, out_channels = 32, kernel_size = 5, stride = 1)
        self.conv5_bn = nn.BatchNorm1d(32)
        
        self.conv6 = nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 5, stride = 2)
        self.conv6_bn = nn.BatchNorm1d(32)
        
        self.conv7 = nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1)
        self.conv7_bn = nn.BatchNorm1d(64)
        
        
        self.flatten = nn.Flatten() 
        self.num_features_after_conv = self.calculate_num_features_after_conv()
        
    
        self.fc1 = nn.Linear(self.num_features_after_conv, 64)
        self.dropoutfc = nn.Dropout(0.75)
        
        self.fc2 = nn.Linear(64,4)
        self.softmax = nn.Softmax(dim = 1)
        
        

    def forward(self, x):
        
        x = torch.relu(self.conv1_bn(self.conv1(x)))
        x = self.dropout(x)
        
        x = torch.relu(self.conv2_bn(self.conv2(x)))
        x = self.dropout(x)
        
        x = torch.relu(self.conv3_bn(self.conv3(x)))
        x = self.dropout(x)
        
        x = torch.relu(self.conv4_bn(self.conv4(x)))
        x = self.dropout(x)
        
        x = torch.relu(self.conv5_bn(self.conv5(x)))
        x = self.dropout(x)
        
        x = torch.relu(self.conv6_bn(self.conv6(x)))
        x = self.dropout(x)
        
        x = torch.relu(self.conv7_bn(self.conv7(x)))
        x = self.dropout(x)
        x = self.flatten(x)
        #print(x.shape)
        x = torch.relu(self.fc1(x))
        x = self.dropoutfc(x)
        x = self.fc2(x)
        #x = self.softmax(self.fc2(x))

        return x

    # Lets define a function to visualize the activation as well. 
    
    
    
    def forward_conv(self, x):
        x = torch.relu(self.conv1_bn(self.conv1(x)))
        x = torch.relu(self.conv2_bn(self.conv2(x)))
        x = torch.relu(self.conv3_bn(self.conv3(x)))
        x = torch.relu(self.conv4_bn(self.conv4(x)))
        x = torch.relu(self.conv5_bn(self.conv5(x)))
        x = torch.relu(self.conv6_bn(self.conv6(x)))
        x = torch.relu(self.conv7_bn(self.conv7(x)))
        x = self.flatten(x)
            
        return x
    
    def calculate_num_features_after_conv(self):
        # Dummy input to calculate the number of features after convolutional layers
        dummy_input = torch.randn(1, self.num_channels, self.num_features)  # Adjust the size based on your input size
        with torch.no_grad():
            conv_output = self.forward_conv(dummy_input)
        num_features_after_conv = conv_output.view(1, -1).size(1)
        return num_features_after_conv   
    
    
    
    
    
  