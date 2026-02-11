'''
To use the saved model to predict the sample's class(such as cancer type), the input sample data should be a 2-dimensional tensor ([batch_size, feature_dim]). For a sample data, batch_size is 1, feature_dim is a array of all feature values.
The useer may first generates a 1-dim numpy array including all feature values for a sample. Then change the numpy array to a pytorch tensor. Then add a new dimension at position 0 (batch dimension).
A sample's data to ready for input should look like:

tensor([[0.0000, 0.4649, 0.4987,  ..., 0.7420, 0.6316, 0.3820]])
torch.Size([1, 20531])

A sample's prediction result should look like:

tensor([[-23.0980, -21.7357,   0.0000, -28.3535, -23.4941, -37.2695, -31.2366,
         -23.9042, -19.3992, -19.7708, -20.4782, -24.3024, -26.9595, -18.9245,
         -21.8257, -24.1601, -34.9532, -31.0570, -37.3571, -22.7628, -18.0850,
         -26.7884, -21.5650, -18.8348, -40.2854, -24.7833, -31.2071, -17.9241,
         -27.1042, -23.1422, -16.7693, -30.5169, -28.2224, -24.1577, -24.5664,
         -23.0578, -37.2448]], grad_fn=<LogSoftmaxBackward0>)

Finally, The index of the largest value is the prediction class of the sample. The prediction class number corresponds to a class name in the type_list.

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import argparse
from configparser import ConfigParser
import sys
import importlib

### Import user's model from the folder DL_modules 
config = ConfigParser()
config_file_path = './config/config.ini'
config.read(config_file_path)
module_file_name = config['DL_model_class']['class_file']
#print(module_file_name)
DLmodel_para = config.get('DL_model_class', 'class_parameters')
DLmodel_para_lst = [item.strip() for item in DLmodel_para.split(',')] # Split the string by comma
DLmodel_para_values = config.get('DL_model_class', 'class_parameter_values') # get the string values
DLmodel_para_val_lst = [float(x.strip()) if '.' in x else int(x.strip()) for x in DLmodel_para_values.split(',')]  # Split the string by comma and convert to actual int and float types
DLmodel_para_dict = dict(zip(DLmodel_para_lst, DLmodel_para_val_lst))
# print(DLmodel_para_lst)
# print(DLmodel_para_val_lst)
#print(DLmodel_para_dict)
current_directory = os.getcwd()
module_path = os.path.join(current_directory, 'DL_modules')
#print(module_path)
sys.path.insert(0, module_path)
DL_module = importlib.import_module(module_file_name)

### Upload user's data from the folder data
data_path = os.path.join(current_directory, 'data')
sys.path.insert(0, data_path)
#train_data_file = config.get('Data', 'train_data')
test_data_file = config.get('Data', 'test_data')
# numFeatures = int(config.get('Data', 'num_features').strip())
types = config.get('Data', 'types')
types_lst = [item.strip() for item in types.split(',')]

### Import other settings
#device_name = config.get('Settings', 'device')
#EPOCHS = int(config.get('Settings', 'epochs').strip())
#BATCH_SIZE = int(config.get('Settings', 'batch_size').strip())
#lr = float(config.get('Settings', 'lr').strip())
#epsilon = float(config.get('Settings', 'epsilon').strip())
#delta = float(config.get('Settings', 'delta').strip())
#mode = config.get('Settings', 'mode')
#numberClients = int(config.get('Settings', 'client').strip())
#l2_norm_clip = int(config.get('Settings', 'l2_norm_clip').strip())
expname = config.get('Settings', 'expname')

#root = '.'
#######################
## load saved model
model = DL_module.Model(**DLmodel_para_dict)

experiment_name = config['DEFAULT']['expname']
checkpoint_path = os.path.join(current_directory, 'model')
model_path = checkpoint_path+"/"+experiment_name+"_modelbest.tar"
state_dict = torch.load(model_path, map_location=torch.device('cpu')) # Use 'cuda' if loading to GPU

model.load_state_dict(state_dict)
model.eval()
###################################
### load data
s_samples = np.load(os.path.join(current_directory,"data/{}.npy".format(test_data_file)),allow_pickle=True).item()
values_list=list(s_samples.values())
num_x = len(values_list)
# features=[]
# for i in range(num_x):
   # features.append(values_list[i]['features']) 
# np_features = np.array(features)
# tensor_features = torch.from_numpy(np_features)
# #print(features)
# print("@@@@@@@@@@@@@@@@@@@@")
# print(np_features)
# print("&&&&&&&&&&&&&&&&&&&&&&&&")
# print(tensor_features)
preds = []
for i in range(num_x):
   asample_features=values_list[i]['features']
   np_asample_features = np.array(asample_features)
   tensor_asample_features = torch.from_numpy(np_asample_features)
   tensor_asample_features = tensor_asample_features.to(torch.float32)
   
   #### Add a new dimension at position 0 (batch dimension). Because the saved model expects a 2-dimensional tensor (namely,[batch_size, feature_dim]), unsqueeze() can add the missing dimension
   tensor_asample_features_2d = tensor_asample_features.unsqueeze(0) 
   
   # print("@@@@@@@@@@@@@@@@@@@@")
   # print(np_asample_features)
   # print("&&&&&&&&&&&&&&&&&&&&&&&&")
   # print(tensor_asample_features_2d)
   # print(tensor_asample_features_2d.shape)
   
   #### predict 
   pred = model(tensor_asample_features_2d)
   
   # print("******************")
   # print(pred)
   
   preds += list(pred.detach().cpu().numpy())

pred_idx=np.argmax(preds,axis=1).flatten().tolist()
print("predicted_class_index: \n"+str(pred_idx))
pred_class=[types_lst[i] for i in pred_idx]
print("predicted_class_name: \n"+str(pred_class))


