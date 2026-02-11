import datetime
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
#from easydl import *
from PIL import Image
import logging
import argparse
from torch.optim import AdamW

def log_creater(output_dir,expname):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    log_name='{}.log'.format(expname)
    #log_name = mode+'_{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    final_log_file = os.path.join(output_dir,log_name)

    # creat a log
    log = logging.getLogger('train_log')
    log.setLevel(logging.DEBUG)

    # FileHandler
    file = logging.FileHandler(final_log_file,'w')
    file.setLevel(logging.DEBUG)

    # StreamHandler
    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s][line: %(lineno)d] ==> [INFO] %(message)s')

    # setFormatter
    file.setFormatter(formatter)
    stream.setFormatter(formatter)

    # addHandler
    log.addHandler(file)
    log.addHandler(stream)

    log.info('creating {}'.format(final_log_file))
    return log

def Process(im_flatten):
    maxValue = torch.max(im_flatten)
    minValue = torch.min(im_flatten)
    im_flatten = im_flatten-minValue
    im_flatten = im_flatten/(maxValue-minValue)
    return im_flatten


def Attack(mynet, target_label,device):
    #torch.set_printoptions(profile="full")
    
    #target_label=torch.tensor(target_label)
    target_output = torch.zeros(1, 29)
    target_output[0, target_label] = 1.0
    
    #aim_flatten = torch.zeros(1, 1, 1, 20531, requires_grad=True).to(device)
    aim_flatten_unnorm = torch.randn(1, 1, 1, 20531, requires_grad=True).to(device)
    
    ### Min-Max Normalization at [0,1]
    ### Note: Normalization creates a non-leaf tensor, so we need detach and create a new leaf tensor from the normalized values    
    aim_flatten = Process(aim_flatten_unnorm).detach().requires_grad_(True).to(device)
    
    ### Initialize the AdamW optimizer with the input tensor    
    optimizer = AdamW([aim_flatten], lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    costn_1 = 10
    b = 0
    g = 0
    
    ### Optimization loop
    for i in range(alpha):
        ### Zero the gradients
        optimizer.zero_grad()
        
        print(f"Loop {i}:")
        print(f"un-updated input tensor: {aim_flatten}")
        out = mynet.forward(aim_flatten)
        #print(f"mynet.forward:\n {str(out)}")
        out = out.reshape(1, classes)
        #print(f"out.reshape:\n {str(out)}")
        
        #target_class = torch.tensor([target_label]).to(device)
        target_class = target_output.to(device)
        # print("target_class: ")
        # print(str(target_class))
        
        ### Define the loss
        cost = criterion(out, target_class)
        #print(f"loss: {cost}")
        ### Call .backward() on the loss to compute the gradients of the loss with respect to all tensors that requires_grad=True, including the input_tensor
        cost.backward()
        
        ### Update the input tensor
        optimizer.step()
        print(f"after-updated input tensor: {aim_flatten}")
        
        aim_grad = aim_flatten.grad
        print(f"aim_flatten gradients: {aim_flatten.grad}")
        
        logger.info('{}/{}: {}'.format(i,alpha,cost))
        if cost >= costn_1:
            b = b+1
            if b > beta:
                break
        else:
            b = 0
        costn_1 = cost
        if cost < gama:
            break
    out = mynet.forward(aim_flatten.detach())
    # print("final out:")
    # print(out)
    
    predict = torch.argmax(out)
    print(f"Final predict and loss of target class {target_label}: {predict},{cost}")
    print("###################################################################################################################")
    oim_flatten = aim_flatten.detach().cpu().numpy()
    with open(f'{log_dir}/{target_label}.npy', 'wb') as f:
        np.save(f, oim_flatten)


###################### CNN model class ###############################################
################################ Quantization (QAT) ##################################
import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat
import math
import torch.nn.functional as F

IN_FEAT = 20531

# calculate the output shape after pool2 and before flaten
# by the formula:
#  L_out = match.floor((L_in+2*padding - dilation*(kernel_size - 1)-1)/stride)+1
# by default, padding=0, dilation=1, stride=kernel_size
shape_conv1_w = math.floor((IN_FEAT+2-(3-1)-1)/1)+1         #  in torch.nn.conv1d()and nn.cov2d, stride default is 1, dilation default is 1
shape_conv1_h = math.floor((1+2-(3-1)-1)/1)+1
shape_maxpool2d_w = math.floor((shape_conv1_w+0-(2-1)-1)/2)+1    # in torch.nn.MaxPool2d(), stride default is kernel size, dilation default is 1, padding default is 0
shape_maxpool2d_h = math.floor((shape_conv1_h+0-(1-1)-1)/1)+1
# Define the quantized CNN (cov1d) model by subclassing nn.Module
class Model(nn.Module):
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
        self.quant_input = qnn.QuantIdentity(bit_width=4)
        self.conv1 = qnn.QuantConv2d(
            in_channels=input_size, 
            out_channels=6, 
            kernel_size=3, 
            padding=1, 
            weight_bit_width=4, 
            bias=True
        )
        self.relu1 = qnn.QuantReLU(bit_width=4)
        
        #add dropout layer for overfitting
        self.quant_in1 = qnn.QuantIdentity(bit_width=4)
        self.dropout1 = nn.Dropout(p=0.2)
        
        #self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.pool1 = qnn.QuantMaxPool2d(kernel_size=(1,2))
        
        #add dropout layer for overfitting
        self.quant_in2 = qnn.QuantIdentity(bit_width=4)
        self.dropout2 = nn.Dropout(p=0.2)
        
        self.quant_in3 = qnn.QuantIdentity(bit_width=4)     # Some PyTorch operators (torch.transpose,torch.add,torch.reshape,torch.flatten), require a brevitas.quant.QuantIdentity to be applied on their inputs
        # Flatten the output for the fully connected layer
        self.flatten = nn.Flatten()
        
        self.fc = qnn.QuantLinear(
            #in_features=64 * shape_pool2, # Adjust based on input sequence length and pooling
            in_features=6*shape_maxpool2d_w*shape_maxpool2d_h,
            #in_features=61590,
            out_features=num_classes,
            weight_bit_width=4,
            bias=True
        )

    def forward(self, x):
        x = self.quant_input(x)
        #print(f"Shape of x after input: {x.shape}")
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.quant_in1(x)
        x = self.dropout1(x)
        #print(f"Shape of x after cov1d: {x.shape}")        
        x = self.pool1(x)
        #print(f"Shape of x after  maxpool2d: {x.shape}")
        x = self.quant_in2(x)
        x = self.dropout2(x)
        #print(f"Shape of x after reshape: {x.shape}")
        x = self.quant_in3(x)
        x = self.flatten(x)
        #print(f"Shape of x after flatten and before linear layer: {x.shape}")
        x = self.fc(x)
        return x
##################################################################################################3




### Eight potentially problematic cancer types (KICH, KIRC, KIRP, COADREAD, READ, LGG, STES, GBM) are removed, retaining the other 29 cancer types
cancerTypes = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBMLGG', 'HNSC',
                'KIPAN', 'LAML', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG',
                   'PRAD', 'SARC', 'SKCM', 'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM']

parser = argparse.ArgumentParser(description='dfsa')
parser.add_argument('--model')
#parser.add_argument('--lr',type=float,default=0.1)
args = parser.parse_args()
model_id=args.model
model_id=model_id.split('/')[1]

gpus = 0
data_workers = 0
# batch_size = 64
classes = len(cancerTypes)
root_dir = "MIA_CC/"
#model_id='SGD_data0_rep0_modelbest.tar'
model_weight = "model/{}".format(model_id)    ### model file must be in the folder "model"
print("model_weight:")
print(model_weight)

alpha = 500
beta = 10
gama = 0.001
#learning_rate = args.lr
# momentum = 0.9

cudnn.benchmark = True
cudnn.deterministic = True
seed = 9970
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

###### change to 'cpu' ###################################################################
#device='cuda:0'
device='cpu'

log_dir = f'{root_dir}/{model_id}'
logger = log_creater(output_dir=log_dir, expname=model_id)

logger.info("model: {}".format(model_id))

print(len(cancerTypes))
net = Model(input_size=1,num_classes=len(cancerTypes))

############ Only if GPU, use  nn.DataParallel(). Note: nn.DataParallel() is also comment in 02.simulationApp_debug.cpu.py. #############################################################
############ As the saved model/checkpoint did not use "nn.DataParallel()" in the file 02.simulationApp_debug.cpu.py, here we cannot use nn.ataParallel() to load the saved model. ######
# mynet = nn.DataParallel(net, output_device=device).train(False)
# mynet=mynet.to(device)

mynet = net.train(False)
#mynet = net.eval()

assert os.path.exists(model_weight)
mynet.load_state_dict(torch.load(open(model_weight, 'rb')))

for i in range(classes):
    logger.info(f'---class{i}---')
    Attack(mynet=mynet, target_label=i,device=device)
