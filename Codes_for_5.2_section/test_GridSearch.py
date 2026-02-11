
import time
import numpy as np
import copy
import matplotlib.pyplot as plt
import numpy
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from concrete.ml.quantization.quantized_module import QuantizedModule
from concrete.ml.torch.compile import compile_brevitas_qat_model

#import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV


IN_FEAT = 20531
OUT_FEAT = 37
# Define hyperparameters
input_channels = 1
#sequence_length = IN_FEAT
num_classes = OUT_FEAT
BATCH_SIZE = 4
n_epochs = 5
############################## input real dataset #########################################
from configparser import ConfigParser
import sys
import os
config = ConfigParser()
config_file_path = './config/config.ini'
config.read(config_file_path)
types = config.get('Data', 'types')
types_lst = [item.strip() for item in types.split(',')]
datafile = config.get('Data', 'data')
current_directory = os.getcwd()
realdata = np.load(os.path.join(current_directory,"data/{}.npy".format(datafile)),allow_pickle=True).item()
values_list=list(realdata.values())
#print(values_list)
values_len = len(values_list)
features=[]
label_idx=[]
for x in range(values_len):
    feature = values_list[x]['features']
    label = values_list[x]['type']
    features.append(feature)
    
    label_idx.append(types_lst.index(label))
#features = data_dic['features']

#print(label_idx)
#print(features)
features_np = np.array(features)
label_idx_np = np.array(label_idx)

# And, finally, split it into train/test sets
# train 2089 samples, test 523 samples
X_train, X_test, y_train, y_test = train_test_split(
    features_np, label_idx_np, test_size=0.1, random_state=40
)
print(len(X_train))
print(len(X_test))
# print(y_test)

# ####### Optional step: create small train and test samples to short running time ###############
# ####### train 470 samples, test 53 samples. ####################################################

# X_train, X_test, y_train, y_test = train_test_split(
    # X_test, y_test, test_size=0.1, random_state=42
# )
# print(f"small train samples: {len(X_train)}")
# print(f"small test samples: {len(X_test)}")
#################################################################################################

#convert data shape from [num_sample, sequence_length] to [num_sample, num_channel, sequence_length]
X_train = torch.tensor(X_train).float()
print(X_train.shape)
print("@@@@@@@@@@@@@@@@")
X_train = X_train.unsqueeze(1)
print(X_train.shape)
print("###############")
X_test = torch.tensor(X_test).float()
print(X_test.shape)
print("@@@@@@@@@@@@@@@@")
X_test = X_test.unsqueeze(1)
print(X_test.shape)
y_train = torch.tensor(y_train)

# QAT model contains a 1D max pooling layer, which is not supported by Concrete-ML's compiler. 
# To resolve this, we need to adjust our model's architecture by replacing the 1D max pooling layer with a 2D equivalent and ensuring our input data has the correct dimensions
# convert data shape from 3D [num_sample, num_channel, sequence_length] to 4D [num_sample, num_channel(1), input_length(1), input_width]
X_train = X_train.unsqueeze(2)
print("###### X_train 4D shape #########")
print(X_train.shape)
X_test = X_test.unsqueeze(2)
print("######X_test 4D shape #########")
print(X_test.shape)

################################ Quantization (QAT) ##################################
import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat
import math
import torch.nn.functional as F

# calculate the output shape after pool2 and before flaten
# by the formula:
#  L_out = match.floor((L_in+2*padding - dilation*(kernel_size - 1)-1)/stride)+1
# by default, padding=0, dilation=1, stride=kernel_size
shape_conv1_w = math.floor((IN_FEAT+2*25-(50-1)-1)/1)+1         #  in torch.nn.conv1d()and nn.cov2d, stride default is 1, dilation default is 1
shape_conv1_h = math.floor((1+2*25-(50-1)-1)/1)+1

# shape_conv2_w = math.floor((shape_conv1_w+2*35-(70-1)-1)/1)+1
# shape_conv2_h = math.floor((shape_conv1_h+2*35-(70-1)-1)/1)+1

# shape_maxpool2d_w = math.floor((shape_conv1_w+2*1-(5-1)-1)/5)+1    # in torch.nn.MaxPool2d(), stride default is kernel size, dilation default is 1, padding default is 0
# shape_maxpool2d_h = math.floor((shape_conv1_h+2*1-(5-1)-1)/5)+1
# Define the quantized CNN (cov1d) model by subclassing nn.Module
class QuantizedConv1DModel(nn.Module):
    def __init__(self, dropout_rate=0.5, bit_width=8):
        super().__init__()
        
        # Define quantization parameters for weights and activations
        # weight_quant = Int8WeightPerTensorFloat(bit_width=bit_width)
        # act_quant = Int8ActPerTensorFloat(bit_width=bit_width)
        # weight_quant = Int8WeightPerTensorFloat
        # act_quant = Int8ActPerTensorFloat

        # Replace standard PyTorch layers with quantized Brevitas layers
        ### Note: you can combine PyTorch and Brevitas layers, as long as a QuantIdentity layer follows the PyTorch layer
        # The quantization settings for weights and activations are passed during initialization
        self.quant_input = qnn.QuantIdentity(bit_width=8)
        self.conv1 = qnn.QuantConv2d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=50, 
            padding=25, 
            weight_bit_width=8, 
            bias=True
        )
        self.relu1 = qnn.QuantReLU(bit_width=8)
        
        #add dropout layer for overfitting
        self.quant_in1 = qnn.QuantIdentity(bit_width=8)
        #self.dropout1 = nn.Dropout(p=0.3)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.quant_in3 = qnn.QuantIdentity(bit_width=8)     # Some PyTorch operators (torch.transpose,torch.add,torch.reshape,torch.flatten), require a brevitas.quant.QuantIdentity to be applied on their inputs
        # Flatten the output for the fully connected layer
        self.flatten = nn.Flatten()
        
        self.fc = qnn.QuantLinear(
            #in_features=64 * shape_pool2, # Adjust based on input sequence length and pooling
            #in_features=1*shape_maxpool2d_w*shape_maxpool2d_h,
            in_features=1*shape_conv1_w*shape_conv1_h,
            #in_features=1*shape_conv2_w*shape_conv2_h,
            #in_features=61590,
            out_features=37,
            weight_bit_width=8,
            bias=True
        )

    def forward(self, x):
        x = self.quant_input(x)
        #print(f"Shape of x after input: {x.shape}")
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.quant_in1(x)
        x = self.dropout1(x)
        x = self.quant_in3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


################################### Train the QAT network ##################################################

              
device = "cuda" if torch.cuda.is_available() else "cpu"

################################ Grid search for hyparameter tuning #######################
# create model with skorch
model = NeuralNetClassifier(
    QuantizedConv1DModel,
    criterion=nn.CrossEntropyLoss,
    optimizer=optim.Adam,
    batch_size=80,
    max_epochs=50,
    verbose=False
)
 
# define the grid search parameters
param_grid = {
    'optimizer__lr': [0.001, 0.01, 0.1, 0.2, 0.3],
    'module__dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
}
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X_train, y_train)
 
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))