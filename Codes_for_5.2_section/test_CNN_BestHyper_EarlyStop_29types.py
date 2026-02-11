
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

import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt 

IN_FEAT = 20531
OUT_FEAT = 37
# Define hyperparameters
input_channels = 1
#sequence_length = IN_FEAT
num_classes = OUT_FEAT
BATCH_SIZE = 80
# n_epochs = 5
n_epochs = 50
############################## input real dataset #########################################
from configparser import ConfigParser
import sys
import os
config = ConfigParser()
config_file_path = './config/config.ini'
config.read(config_file_path)
#types = config.get('Data', 'types')
#types = config.get('Data', 'types30')
types = config.get('Data', 'types29')

types_lst = [item.strip() for item in types.split(',')]

for index, item in enumerate(types_lst):
    print(f"{index}: {item}")

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
    
    if label != "KICH" and label != "KIRC" and label != "KIRP" and label != "READ" and label != "STES" and label != "LGG" and label != "COADREAD" and label != "GBM":
       features.append(feature)    
       label_idx.append(types_lst.index(label))

#print(label_idx)
#print(features)
features_np = np.array(features)
label_idx_np = np.array(label_idx)

# And, finally, split it into train/test sets
# train 2089 samples, test 523 samples
X_train, X_test, y_train, y_test = train_test_split(
    features_np, label_idx_np, test_size=0.3, random_state=35
)
print(len(X_train))
print(len(X_test))
# print(y_test)

# ####### Optional step: create small train and test samples to short running time ###############
# ####### train 470 samples, test 53 samples. ####################################################

#### add validation group ###
X_val, X_test, y_val, y_test = train_test_split(
    X_test, y_test, test_size=0.5, random_state=25
)
print(f"val samples: {len(X_val)}")
print(f"test samples: {len(X_test)}")
#################################################################################################

#convert data shape from [num_sample, sequence_length] to [num_sample, num_channel, sequence_length]
X_train = torch.tensor(X_train).float()
# print(X_train.shape)
# print("@@@@@@@@@@@@@@@@")
X_train = X_train.unsqueeze(1)
# print(X_train.shape)
# print("###############")
X_test = torch.tensor(X_test).float()
# print(X_test.shape)
# print("@@@@@@@@@@@@@@@@")
X_test = X_test.unsqueeze(1)
# print(X_test.shape)
y_train = torch.tensor(y_train)

X_val = torch.tensor(X_val).float()
X_val = X_val.unsqueeze(1)


# QAT model contains a 1D max pooling layer, which is not supported by Concrete-ML's compiler. 
# To resolve this, we need to adjust our model's architecture by replacing the 1D max pooling layer with a 2D equivalent and ensuring our input data has the correct dimensions
# convert data shape from 3D [num_sample, num_channel, sequence_length] to 4D [num_sample, num_channel(1), input_length(1), input_width]
X_train = X_train.unsqueeze(2)
# print("###### X_train 4D shape #########")
# print(X_train.shape)
X_test = X_test.unsqueeze(2)
# print("######X_test 4D shape #########")
# print(X_test.shape)
X_val = X_val.unsqueeze(2)

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
    def __init__(self, input_size, num_classes, bit_width=8):
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
            in_channels=input_size, 
            out_channels=1, 
            kernel_size=50, 
            padding=25, 
            weight_bit_width=8, 
            bias=True
        )
        self.relu1 = qnn.QuantReLU(bit_width=8)
        
        #add dropout layer for overfitting
        self.quant_in1 = qnn.QuantIdentity(bit_width=8)
        self.dropout1 = nn.Dropout(p=0.4)
        
        # # add one more CNN2d layer
        # self.conv2 = qnn.QuantConv2d(
            # in_channels=input_size, 
            # out_channels=1, 
            # kernel_size=70, 
            # padding=35, 
            # weight_bit_width=4, 
            # bias=True
        # )
        # self.relu2 = qnn.QuantReLU(bit_width=4)
        
        # #add dropout layer for overfitting
        # self.quant_in2 = qnn.QuantIdentity(bit_width=4)
        # self.dropout2 = nn.Dropout(p=0.3)
        
        #self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # self.pool1 = qnn.QuantMaxPool2d(kernel_size=(5,5),padding=1)
        
        # #add dropout layer for overfitting
        # self.quant_in2 = qnn.QuantIdentity(bit_width=4)
        # self.dropout2 = nn.Dropout(p=0.3)
        
        self.quant_in3 = qnn.QuantIdentity(bit_width=8)     # Some PyTorch operators (torch.transpose,torch.add,torch.reshape,torch.flatten), require a brevitas.quant.QuantIdentity to be applied on their inputs
        # Flatten the output for the fully connected layer
        self.flatten = nn.Flatten()
        
        self.fc = qnn.QuantLinear(
            #in_features=64 * shape_pool2, # Adjust based on input sequence length and pooling
            #in_features=1*shape_maxpool2d_w*shape_maxpool2d_h,
            in_features=1*shape_conv1_w*shape_conv1_h,
            #in_features=1*shape_conv2_w*shape_conv2_h,
            #in_features=61590,
            out_features=num_classes,
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
        # x = self.conv2(x)
        # x = self.relu2(x)
        #print(f"Shape of x after cov1d: {x.shape}")        
        # x = self.pool1(x)
        # #print(f"Shape of x after  maxpool2d: {x.shape}")
        # x = self.quant_in2(x)
        # x = self.dropout2(x)
        #print(f"Shape of x after reshape: {x.shape}")
        x = self.quant_in3(x)
        x = self.flatten(x)
        #print(f"Shape of x after flatten and before linear layer: {x.shape}")
        x = self.fc(x)
        return x


################################ EearlyStopping class ###########################################################
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)

########### add EarlyStopping instance
early_stopping = EarlyStopping(patience=10, delta=0.01)


##############################  define train ##########################################
def train(
    torch_model,
    X_train,
    X_test,
    X_val,
    y_train,
    y_test,
    y_val,
    criterion,
    optimizer,
    epochs=200,
    batch_size=1,
    shuffle=True,
    device="cpu",
):
    # X_train = torch.tensor(X_train).float()
    # print(X_train.shape)
    # print("@@@@@@@@@@@@@@@@")
    # X_train = X_train.unsqueeze(1)
    # print(X_train.shape)
    # print("###############")
    # X_test = torch.tensor(X_test).float()
    # print(X_test.shape)
    # print("@@@@@@@@@@@@@@@@")
    # X_test = X_test.unsqueeze(1)
    # print(X_test.shape)
    # y_train = torch.tensor(y_train)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=shuffle
    )
    torch_model.train()
    
    epochs_loss_train = []
    epochs_loss_val = []
    epochs_accu_train = []
    epochs_accu_val = []
    run_epochs = 0
    
    for epoch in range(epochs):
        total_loss = []
        y_pred_all = []
        y_true_all = []
        torch_model.train()
        for batch_index, (X_batch, y_batch) in enumerate(train_loader):
            # Forward pass
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = torch_model(X_batch)
            y_pred_all.append(y_pred.argmax(1).detach().cpu().numpy())
            y_true_all.append(y_batch.detach().cpu().numpy())

            # Compute loss
            loss = criterion(y_pred, y_batch)
            if torch.isnan(loss):
                print("y_pred", y_pred)
                print("y_batch", y_batch)
                raise ValueError(f"Loss diverged at step: {batch_index}")

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Update weights
            optimizer.step()

            total_loss.append(loss.cpu().item())

        # Print epoch number, loss and accuracy
        y_pred_all = numpy.concatenate(y_pred_all)
        y_true_all = numpy.concatenate(y_true_all)
        accuracy = numpy.mean(y_pred_all == y_true_all)
        # print(
            # f"Epoch: {epoch:02} | Loss: {numpy.mean(total_loss):.4f} |"
            # f" Train Accuracy: {100*accuracy:.2f}%"
        # )
        # Validation phase for the current epoch
        torch_model.eval()
        val_fp32_pred = torch_model(X_val.to(device)).cpu().argmax(1).float().detach().numpy()
        val_accuracy = numpy.mean(val_fp32_pred == y_val)
        
        ######### add validation average loss ###################################################
        val_output = torch_model(X_val.to(device)).cpu()
        y_val_tensor = torch.tensor(y_val)
        val_loss = criterion(val_output, y_val_tensor)
        val_loss_value = val_loss.item()
        
        ###### store each epoch's train average-loss and accuracy, validation average-loss and accuracy
        epochs_loss_train.append(numpy.mean(total_loss))
        epochs_loss_val.append(val_loss_value)
        epochs_accu_train.append(accuracy)
        epochs_accu_val.append(val_accuracy)
        run_epochs += 1
        ##### print summary of each epoch
        print(
                    f"Epoch: {epoch:02} | Train Loss: {numpy.mean(total_loss):.4f} |"
                    f" Train Accuracy: {100*accuracy:.2f}% |"
                    f"Validation Loss: {val_loss_value:.4f} |"
                    f"Validation Accuracy Fp32: {val_accuracy*100:.2f}%"
        )
        
        ####### add Early Stopping ######################################################
        early_stopping(val_loss_value, torch_model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    ##### load best-model parameters to the model 
    early_stopping.load_best_model(torch_model)
    
    # Compute test accuracy once training is done
    torch_model.eval()
    fp32_pred = torch_model(X_test.to(device)).cpu().argmax(1).float().detach().numpy()
    accuracy = numpy.mean(fp32_pred == y_test)
    print(f"\nTest Accuracy Fp32: {accuracy*100:.2f}%")
    
    print("prediction labels:")
    print(f"{list(fp32_pred)}")
    print("real labels:")
    print(f"{list(y_test)}")
    
    ##### plot both lines on the same axis
    plt.figure(1)
    plt.plot(range(run_epochs), epochs_loss_train, label='Training Loss')
    plt.plot(range(run_epochs), epochs_loss_val, label='Validation Loss')
    # Add labels, a title, and a legend for better readability
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # Display the plot
    #plt.show()
    # Save the plot as a PNG image
    plt.savefig("Training and Validation Loss.png")
    
    ##### plot both lines on the same axis
    plt.figure(2)
    plt.plot(range(run_epochs), epochs_accu_train, label='Training Accuracy')
    plt.plot(range(run_epochs), epochs_accu_val, label='Validation Accuracy')
    # Add labels, a title, and a legend for better readability
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    # Display the plot
    #plt.show()
    # Save the plot as a PNG image
    plt.savefig("Training and Validation Accuracy.png")
    
    return accuracy

##################################  Conversion to FHE and compilation   ############################################
def test_in_fhe(quantized_numpy_module, X_test, y_test, simulate=True):
    if not simulate:
        print("Generating key")
        start_key = time.time()
        quantized_numpy_module.fhe_circuit.keygen()
        end_key = time.time()
        print(f"Key generation finished in {end_key - start_key:.2f} seconds")

    fhe_mode = "simulate" if simulate else "execute"

    start_infer = time.time()
    predictions = quantized_numpy_module.forward(X_test, fhe=fhe_mode).argmax(1)
    end_infer = time.time()

    if not simulate:
        print(
            f"Inferences finished in {end_infer - start_infer:.2f} seconds "
            f"({(end_infer - start_infer)/len(X_test):.2f} seconds/sample)"
        )

    # Compute accuracy
    accuracy = numpy.mean(predictions == y_test) * 100
    print(
        "FHE " + ("(simulation) " * simulate) + f"accuracy: {accuracy:.2f}% on "
        f"{len(X_test)} examples."
    )
    return predictions
 

################################### Train the QAT network ##################################################
             
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define loss function
criterion = nn.CrossEntropyLoss()
torch_model = QuantizedConv1DModel(input_channels, num_classes)
torch_model = torch_model.to(device)
optimizer = torch.optim.AdamW(torch_model.parameters(), lr=0.001)

accuracy = train(
    torch_model,
    X_train,
    X_test,
    X_val,
    y_train,
    y_test,
    y_val,
    criterion,
    optimizer,
    epochs=n_epochs,
    batch_size=BATCH_SIZE,
    device=device,
)

#Test the quantization aware trained model, running in torch in floating point
torch_model.eval()

fp32_pred = (
    torch_model(torch.tensor(X_test).float().to(device)).cpu().argmax(1).float().detach().numpy()
)
