
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
from concrete.ml.deployment import FHEModelDev
from configparser import ConfigParser
import sys
import os


IN_FEAT = 20531
#OUT_FEAT = 37
#OUT_FEAT = 29
# Define hyperparameters
input_channels = 1
#sequence_length = IN_FEAT
#num_classes = OUT_FEAT

num_classes = 29
BATCH_SIZE = 8
n_epochs = 40

############################## input real dataset #########################################
config = ConfigParser()
config_file_path = './config/config.ini'
config.read(config_file_path)
#types = config.get('Data', 'types')
types = config.get('Data', 'types29')
types_lst = [item.strip() for item in types.split(',')]

#datafile = config.get('Data', 'data')          ### use a small sample data
datafile = config.get('Data', 'train_data')     ### use a large sample data

current_directory = os.getcwd()
realdata = np.load(os.path.join(current_directory,"data/{}.npy".format(datafile)),allow_pickle=True).item()
values_list=list(realdata.values())
#print(values_list)
values_len = len(values_list)
features=[]
label_idx=[]
# for x in range(values_len):
    # feature = values_list[x]['features']
    # label = values_list[x]['type']
    # features.append(feature)
    
    # label_idx.append(types_lst.index(label))

for x in range(values_len):
    feature = values_list[x]['features']
    label = values_list[x]['type']
    
    if label != "KICH" and label != "KIRC" and label != "KIRP" and label != "READ" and label != "STES" and label != "LGG" and label != "COADREAD" and label != "GBM":
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
    features_np, label_idx_np, test_size=0.2, random_state=42
)
print(len(X_train))
print(len(X_test))
# print(y_test)

####### Optional step: create small train and test samples to short running time ###############
####### train 470 samples, test 53 samples. ####################################################

X_train, X_test, y_train, y_test = train_test_split(
    X_test, y_test, test_size=0.1, random_state=42
)
print(f"small train samples: {len(X_train)}")
print(f"small test samples: {len(X_test)}")
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
# import brevitas.nn as qnn
# from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat
import math
import torch.nn.functional as F

# calculate the output shape after pool2 and before flaten
# by the formula:
#  L_out = match.floor((L_in+2*padding - dilation*(kernel_size - 1)-1)/stride)+1
# by default, padding=0, dilation=1, stride=kernel_size
shape_conv1_w = math.floor((IN_FEAT+2-(3-1)-1)/1)+1         #  in torch.nn.conv1d()and nn.cov2d, stride default is 1, dilation default is 1
shape_conv1_h = math.floor((1+2-(3-1)-1)/1)+1
shape_maxpool2d_w = math.floor((shape_conv1_w+0-(2-1)-1)/2)+1    # in torch.nn.MaxPool2d(), stride default is kernel size, dilation default is 1, padding default is 0
shape_maxpool2d_h = math.floor((shape_conv1_h+0-(1-1)-1)/1)+1
# Define the quantized CNN (cov1d) model by subclassing nn.Module
# class QuantizedConv1DModel(nn.Module):
    # def __init__(self, input_size, num_classes, bit_width=8):
        # super().__init__()
        
        # # Define quantization parameters for weights and activations
        # # weight_quant = Int8WeightPerTensorFloat(bit_width=bit_width)
        # # act_quant = Int8ActPerTensorFloat(bit_width=bit_width)
        # # weight_quant = Int8WeightPerTensorFloat
        # # act_quant = Int8ActPerTensorFloat

        # # Replace standard PyTorch layers with quantized Brevitas layers
        # ### Note: you can combine PyTorch and Brevitas layers, as long as a QuantIdentity layer follows the PyTorch layer
        # # The quantization settings for weights and activations are passed during initialization
        # self.quant_input = qnn.QuantIdentity(bit_width=4)
        # self.conv1 = qnn.QuantConv2d(
            # in_channels=input_size, 
            # out_channels=6, 
            # kernel_size=3, 
            # padding=1, 
            # weight_bit_width=4, 
            # bias=True
        # )
        # self.relu1 = qnn.QuantReLU(bit_width=4)
        
        # #add dropout layer for overfitting
        # self.quant_in1 = qnn.QuantIdentity(bit_width=4)
        # self.dropout1 = nn.Dropout(p=0.2)
        
        # #self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # self.pool1 = qnn.QuantMaxPool2d(kernel_size=(1,2))
        
        # #add dropout layer for overfitting
        # self.quant_in2 = qnn.QuantIdentity(bit_width=4)
        # self.dropout2 = nn.Dropout(p=0.2)
        
        # self.quant_in3 = qnn.QuantIdentity(bit_width=4)     # Some PyTorch operators (torch.transpose,torch.add,torch.reshape,torch.flatten), require a brevitas.quant.QuantIdentity to be applied on their inputs
        # # Flatten the output for the fully connected layer
        # self.flatten = nn.Flatten()
        
        # self.fc = qnn.QuantLinear(
            # #in_features=64 * shape_pool2, # Adjust based on input sequence length and pooling
            # in_features=6*shape_maxpool2d_w*shape_maxpool2d_h,
            # #in_features=61590,
            # out_features=num_classes,
            # weight_bit_width=4,
            # bias=True
        # )

    # def forward(self, x):
        # x = self.quant_input(x)
        # #print(f"Shape of x after input: {x.shape}")
        # x = self.conv1(x)
        # x = self.relu1(x)
        # x = self.quant_in1(x)
        # x = self.dropout1(x)
        # #print(f"Shape of x after cov1d: {x.shape}")        
        # x = self.pool1(x)
        # #print(f"Shape of x after  maxpool2d: {x.shape}")
        # x = self.quant_in2(x)
        # x = self.dropout2(x)
        # #print(f"Shape of x after reshape: {x.shape}")
        # x = self.quant_in3(x)
        # x = self.flatten(x)
        # #print(f"Shape of x after flatten and before linear layer: {x.shape}")
        # x = self.fc(x)
        # return x

################# Non-quantization model ###########################################################################
import torch.nn as nn

class Conv2DModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=input_size, 
            out_channels=6, 
            kernel_size=3, 
            padding=1,  
            bias=True
        )
        self.relu1 = nn.ReLU()
        
        #add dropout layer for overfitting
        #self.quant_in1 = qnn.QuantIdentity(bit_width=4)
        self.dropout1 = nn.Dropout(p=0.2)
        
        #self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.pool1 = nn.MaxPool2d(kernel_size=(1,2))
        
        #add dropout layer for overfitting
        #self.quant_in2 = qnn.QuantIdentity(bit_width=4)
        self.dropout2 = nn.Dropout(p=0.2)
        
        #self.quant_in3 = qnn.QuantIdentity(bit_width=4)     # Some PyTorch operators (torch.transpose,torch.add,torch.reshape,torch.flatten), require a brevitas.quant.QuantIdentity to be applied on their inputs
        # Flatten the output for the fully connected layer
        self.flatten = nn.Flatten()
        
        self.fc = nn.Linear(
            #in_features=64 * shape_pool2, # Adjust based on input sequence length and pooling
            in_features=6*shape_maxpool2d_w*shape_maxpool2d_h,
            #in_features=61590,
            out_features=num_classes,
            bias=True
        )

    def forward(self, x):
        #x = self.quant_input(x)
        #print(f"Shape of x after input: {x.shape}")
        x = self.conv1(x)
        x = self.relu1(x)
        #x = self.quant_in1(x)
        x = self.dropout1(x)
        #print(f"Shape of x after cov1d: {x.shape}")        
        x = self.pool1(x)
        #print(f"Shape of x after  maxpool2d: {x.shape}")
        #x = self.quant_in2(x)
        x = self.dropout2(x)
        #print(f"Shape of x after reshape: {x.shape}")
        #x = self.quant_in3(x)
        x = self.flatten(x)
        #print(f"Shape of x after flatten and before linear layer: {x.shape}")
        x = self.fc(x)
        return x



################# define a function to output the model checkpoint (best model) 
def save_checkpoint(model, is_best):

    expname = "CNN2d_no_Quantization_bigdata"
    checkpoint = "output_model"
    #model_name = os.path.join(checkpoint, 'mode_{}_'.format(mode)+'epoch_' + str(epoch) + '_model.pth.tar')
    model_name=os.path.join(checkpoint, '{}'.format(expname) + '_modelbest.tar')

    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)

    if is_best:
        print("Saving checkpoint... ")
        print("Saved to {}".format(model_name))
        print("Checkpoint is best!")
        with open(model_name, 'wb') as f:
            torch.save(model.state_dict(), f)
    else:
        print("Model not good, skip!")



##############################  define train ##########################################
def train(
    torch_model,
    X_train,
    X_test,
    y_train,
    y_test,
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
    
    best_loss = 10 ** 15
    best_acc = 0
    is_best = False
        
    for epoch in range(epochs):
        total_loss = []
        y_pred_all = []
        y_true_all = []
        
        #### prepare for DP
        old_model = copy.deepcopy(torch_model)
        
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
        
        ####################### add Differential Privacy (DP) to all layers #######################
        # lr = 0.001
        # #### prepare for DP
        # new_model = copy.deepcopy(torch_model)        
        # grads = [(new_param.data - old_param.data)/(-lr) for old_param, new_param in zip(old_model.parameters(), new_model.parameters())]
        
        # ### Laplace mechanism
        # GS=5
        # #epsilon=200
        # epsilon=10
        # #delta=1e-3
        # sigma = GS/epsilon
        
        # ### add noise to gradients
        # aggregated=[]
        # for i in range(len(grads)):
            # aggregated.append(torch.tensor(np.random.laplace(0, sigma, grads[i].shape), device=device) + grads[i])
        
        # ### add updated gradients to the model
        # weight = -1*lr
        # for param_model, param_update in zip(torch_model.parameters(), aggregated):
            # param_model.data += weight * param_update.data        
        
        ############################################################################
        # ############## only add DP noise to one layer #############################################
        # lr = 0.001
        # #### prepare for DP
        # new_model = copy.deepcopy(torch_model)        
        # grads = [(new_param.data - old_param.data)/(-lr) for old_param, new_param in zip(old_model.parameters(), new_model.parameters())]
        # layer_name = 'conv1.weight'
        # layer_weight_grad = None
        # idx_specific_layer = 0
        # for i, (name, old_param) in enumerate(old_model.named_parameters()):
            # if name == layer_name:
               # layer_weight_grad = grads[i]
               # idx_specific_layer = i
               # break
        
        # ### Laplace mechanism
        # GS=5
        # epsilon=200
        # #delta=1e-3
        # sigma = GS/epsilon           
        
        # ### only add noise to specific layer's gradients
        # aggregated=[]
        # for i in range(len(grads)):
            # if i == idx_specific_layer:
               # aggregated.append(torch.tensor(np.random.laplace(0, sigma, grads[i].shape), device=device) + grads[i])
            # else:
               # aggregated.append(grads[i])
        
        # ### add updated gradients to the model
        # weight = -1.0*lr
        # for param_model, param_update in zip(torch_model.parameters(), aggregated):
            # param_model.data += weight * param_update.data   
        ###########################################################################################################
               
        # Print epoch number, loss and accuracy
        y_pred_all = numpy.concatenate(y_pred_all)
        y_true_all = numpy.concatenate(y_true_all)
        accuracy = numpy.mean(y_pred_all == y_true_all)
        print(
            f"Epoch: {epoch:02} | Loss: {numpy.mean(total_loss):.4f} |"
            f" Train Accuracy: {100*accuracy:.2f}%"
        )

        ########### output best model to file
        if  accuracy > best_acc:
            is_best = True
            best_loss = numpy.mean(total_loss)
            best_acc = accuracy
        else:
            is_best = False

        save_checkpoint(torch_model, is_best)


    # Compute test accuracy once training is done
    torch_model.eval()
    fp32_pred = torch_model(X_test.to(device)).cpu().argmax(1).float().detach().numpy()
    accuracy = numpy.mean(fp32_pred == y_test)
    print(f"\nTest Accuracy Fp32: {accuracy*100:.2f}%")

    return accuracy

##################################  Conversion to FHE and compilation   ############################################
# def test_in_fhe(quantized_numpy_module, X_test, y_test, simulate=True):
    # if not simulate:
        # print("Generating key")
        # start_key = time.time()
        # quantized_numpy_module.fhe_circuit.keygen()
        # end_key = time.time()
        # print(f"Key generation finished in {end_key - start_key:.2f} seconds")

    # fhe_mode = "simulate" if simulate else "execute"

    # start_infer = time.time()
    # predictions = quantized_numpy_module.forward(X_test, fhe=fhe_mode).argmax(1)
    # end_infer = time.time()

    # if not simulate:
        # print(
            # f"Inferences finished in {end_infer - start_infer:.2f} seconds "
            # f"({(end_infer - start_infer)/len(X_test):.2f} seconds/sample)"
        # )

    # # Compute accuracy
    # accuracy = numpy.mean(predictions == y_test) * 100
    # print(
        # "FHE " + ("(simulation) " * simulate) + f"accuracy: {accuracy:.2f}% on "
        # f"{len(X_test)} examples."
    # )
    # return predictions
 
################################### Train the non-QAT network ##################################################

              
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define loss function
criterion = nn.CrossEntropyLoss()

#torch_model = QuantizedConv1DModel(input_channels, num_classes)
torch_model = Conv2DModel(input_channels, num_classes)
torch_model = torch_model.to(device)
optimizer = torch.optim.AdamW(torch_model.parameters(), lr=0.001)

accuracy = train(
    torch_model,
    X_train,
    X_test,
    y_train,
    y_test,
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

# #Import and test the quantization aware trained model in Concrete ML
# # Move torch_model to CPU
# torch_model = torch_model.cpu()

# # Compile the model using a representative input-set (only 1 sample data should be inputted)
# X_train_one = X_train[0:1,:,:,:]
# quantized_numpy_module = compile_brevitas_qat_model(torch_model, X_train_one)

# X_test = X_test.cpu().numpy()
# prediction_simulated = test_in_fhe(quantized_numpy_module, X_test, y_test, simulate=True)
# # Reduce the test set for faster running time
# FHE_SAMPLE = 10
# prediction_fhe = test_in_fhe(
    # quantized_numpy_module, X_test[:FHE_SAMPLE], y_test[:FHE_SAMPLE], simulate=False
# )

# # prediction_fhe = test_in_fhe(
    # # quantized_numpy_module, X_test, y_test, simulate=False
# # )

# ######## save the FHE model for client/server deployment #######################################
# ### The server.zip file is deployed to the server and loaded using FHEModelServer
# ### The client.zip file is used on the client machine to generate keys, encrypt data, and later decrypt the results, using FHEModelClient
# ### Detailed deployment steps : https://docs.zama.ai/concrete-ml/guides/client_server

# # Define the directory for FHE client/server files
# fhe_directory = "output_FHE_model"
# if not os.path.exists(fhe_directory):
    # os.mkdir(fhe_directory)

# # Setup the development environment
# dev = FHEModelDev(path_dir=fhe_directory, model=quantized_numpy_module)
# dev.save()
