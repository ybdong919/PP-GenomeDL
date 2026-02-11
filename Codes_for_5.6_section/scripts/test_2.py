'''
The script uses the data with the simplified dictionary structure 

The structure of the original data (such as, train_data_norm_log10_x.npy):
    {'samples': {
            'TCGA-A2-A0ST-01A-12R-A084-07(sample1_name)': {'day2birth': '-22922', 'day2death': nan, 'dayrecord': '22922', 'cancertype': 'BRCA', 'genes': array([0.        , 0.46489265, 0.49866569, ..., 0.74204937, 0.63159535,0.38199453])},
            'TCGA-BR-6706-01A-11R-1884-13_y(sample2_name)': {'day2birth': '-23053', 'day2death': nan, 'dayrecord': '23053', 'cancertype': 'STES', 'genes': array([0.42482528, 0.51957173, 0.52230517, ..., 0.68832371, 0.57021237,0.     ])},
            ......
       }    
            
     'feature_names': 'AGFG1|3267', 'AGFG2|3268', ......
    }

The structure of the simplified .npy file (such as,train_data_norm_log10_simplified.npy ):
    {'sample1_name': {
              'type': 'BRCA', 'features': array([0.        , 0.46489265, 0.49866569, ..., 0.74204937, 0.63159535,0.38199453])
             },
     'sample2_name': {
              'type': 'STES', 'features': array([0.42482528, 0.51957173, 0.52230517, ..., 0.68832371, 0.57021237,0.     ])
             },
      .......
        
    }
    
In addition, the codes below are modified for the new data structure:
1) For the data with "features" and "type", not "genes" and "cancertype"
2) In the codes below, all "genes" are replaced with "features" and all "cancertype" are replaced with "type"
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from scipy import stats
from tqdm import tqdm
import copy
from math import sqrt, exp
from scipy.special import erf
import logging
import time
import argparse
from multiprocessing import Pool
import random
from sklearn.metrics import f1_score
# for levy_stable distribution
from scipy.stats import levy_stable


from configparser import ConfigParser
import sys
import os
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
#DLmodel_para_val_lst = [int(x.strip()) for x in DLmodel_para_values.split(',')]  # Split the string by comma and convert to integers
DLmodel_para_val_lst = [float(x.strip()) if '.' in x else int(x.strip()) for x in DLmodel_para_values.split(',')]  # Split the string by comma and convert to actual int and float types
DLmodel_para_dict = dict(zip(DLmodel_para_lst, DLmodel_para_val_lst))
# print(DLmodel_para_lst)
# print(DLmodel_para_val_lst)
print(DLmodel_para_dict)

current_directory = os.getcwd()
#parent_directory = os.path.dirname(current_directory)
#print(f"Parent Directory: {parent_directory}")
module_path = os.path.join(current_directory, 'DL_modules')
#print(module_path)
sys.path.insert(0, module_path)

DL_module = importlib.import_module(module_file_name)
print(DL_module.Model)

###
# clientModel=DL_module.Model(**DLmodel_para_dict)

# #check para and values
# for name, param in clientModel.named_parameters():
        # #print(f"Parameter Name: {name}, Value: {param.data}")
        # print(f"Parameter Name: {name}, Value: {param.shape}")

# def initialClientModel(serverModel,device):
    # #global cancerTypes
    # clientModel=DL_module.Model(**DLmodel_para_dict)
    
    # ############ If device is 'cpu', no need move model to GPU. ##########################
    # ############ If device is GPU, we need move model from cpu to GPU  ###################
    # #clientModel=clientModel.to(device=device)
    # #clientModel = torch.nn.DataParallel(clientModel).cuda()
    # clientModel.load_state_dict(serverModel.state_dict())
    # return clientModel
    
    
### Upload user's data from the folder data
data_path = os.path.join(current_directory, 'data')
sys.path.insert(0, data_path)
train_data_file = config.get('Data', 'train_data')
test_data_file = config.get('Data', 'test_data')
print(train_data_file)

numFeatures = int(config.get('Data', 'num_features').strip())

types = config.get('Data', 'types')
types_lst = [item.strip() for item in types.split(',')]

### Import other settings
device_name = config.get('Settings', 'device')
EPOCHS = int(config.get('Settings', 'epochs').strip())
BATCH_SIZE = int(config.get('Settings', 'batch_size').strip())
lr = float(config.get('Settings', 'lr').strip())
epsilon = float(config.get('Settings', 'epsilon').strip())
delta = float(config.get('Settings', 'delta').strip())
mode = config.get('Settings', 'mode')
numberClients = int(config.get('Settings', 'client').strip())
l2_norm_clip = int(config.get('Settings', 'l2_norm_clip').strip())
expname = config.get('Settings', 'expname')

model_path = 'model'
root = '.'

####################################### define functions ################################################
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
    
logger = log_creater(output_dir='log', expname=expname)

class Clinical_Data(Dataset):
    def __init__(self, index):
        print("[INFO] Loading {}".format(os.path.join(root,"data/{}.npy".format(index))))
        self.data = np.load(os.path.join(current_directory,"data/{}.npy".format(index)),allow_pickle=True).item()
        
        # self.samples dic
        # 'TCGA-OR-A5J1-01A-11R-A29S-07':
        
        #self.samples = self.data['samples']  
        ### In the simplified dictionary, the Top-level key 'samples' is deleted.
        self.samples = self.data
        
        print("[INFO] {} Data has {} samples".format(index, len(self.samples)))

    def __getitem__(self, idx):
        #### get the values of the self.samples. Note: self.samples is a dictionary {sample name: {day2birth, day2death, day2record, cancertype, genes}}#############################
        values_list=list(self.samples.values())
        
        #data_dic = copy.deepcopy(self.samples[idx])
        data_dic = copy.deepcopy(values_list[idx])
        
        #genes = data_dic['genes']
        features = data_dic['features']  #### change 'genes' to 'features'
        #label=types_lst.index(data_dic['cancertype'])
        #### change 'cancertype' to 'type'
        label=types_lst.index(data_dic['type'])
        
        #del data_dic['genes']
        del data_dic['features']
        info = data_dic

        # Genes: 20531 len Tensor
        # lable: float value
        # info:  dic info
        
        #return torch.Tensor(genes), label, info
        #return torch.Tensor(genes), label
        return torch.Tensor(features), label

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return len(self.samples)

    def __alltypes__(self):
        out=[]
        #print("__alltypes_ get:")
        #print(self.samples)
        
        #### get the values of the self.samples. Note: self.samples is a dictionary {sample name: {day2birth, day2death, day2record, cancertype, genes}}#############################
        values_list=list(self.samples.values())
        
        for idx in range(len(self.samples)):
            #print("idx is:")
            #print(idx)
            #print(values_list[idx])
            #out.append(values_list[idx]['cancertype'])
            ### change 'cancertype' to 'type'
            out.append(values_list[idx]['type'])
            
        print("cacertype labels from inputted data:")
        print(out)
        return out

def getSurvivalDataset(numClients, train_index='train', test_index='test'):
    global logger
    
    trainDataset = Clinical_Data(index=train_index)
    testDataset = Clinical_Data(index=test_index)

    train_counts={}
    test_counts={}
    for t in types_lst:
        train_counts[t]=0
        test_counts[t]=0

    for t in trainDataset.__alltypes__():
        train_counts[t]+=1
    for t in testDataset.__alltypes__():
        test_counts[t]+=1

    logger.info('Train Counts: {}'.format(train_counts))
    logger.info('Test Counts: {}'.format(test_counts))
    
    ##### np.int replaced by int ##############################
    numItems = int(np.floor(len(trainDataset) / numClients))
    print("numItems: "+ str(numItems))
    print("len(trainDataset) in line 193:")
    print(len(trainDataset))
    trainDatasets=[]
    
    if numClients!=1:
        #print("random_split(), the second parameter:")
        #print(str([numItems for i in range(numClients)]+[len(trainDataset)-numItems*numClients]))
        #print(str([numItems for i in range(numClients)]))
        #print(str([len(trainDataset)-numItems*numClients]))
        ############## debug
        list_x = [numItems for i in range(numClients)]
        if len(trainDataset)-numItems*numClients !=0:
            list_x[-1]=list_x[-1]+(len(trainDataset)-numItems*numClients)
         
        #for dataset in torch.utils.data.random_split(trainDataset, [numItems for i in range(numClients)]+[len(trainDataset)-numItems*numClients]):
        for dataset in torch.utils.data.random_split(trainDataset, list_x):
            print("length of dataset in line 209: "+ str(len(dataset)))
            trainDatasets.append(dataset)
    else:
        trainDatasets.append(trainDataset)

    return trainDatasets, testDataset

class RunningAverage():
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)

def initialClientModel(serverModel,device):
    clientModel=DL_module.Model(**DLmodel_para_dict)
    
    if device_name != 'cpu':
          clientModel=clientModel.to(device=device)
          clientModel = torch.nn.DataParallel(clientModel).cuda()
    clientModel.load_state_dict(serverModel.state_dict())
    return clientModel

def compute_grad_update(old_model, new_model, lr, device):
    if device:
        old_model, new_model = old_model.to(device), new_model.to(device)
    return [(new_param.data - old_param.data)/(-lr) for old_param, new_param in zip(old_model.parameters(), new_model.parameters())]

def calibrateAnalyticGaussianMechanism(epsilon, delta, GS=1, tol=1.e-12):

    """ Calibrate a Gaussian perturbation for differential privacy using the analytic Gaussian mechanism of [Balle and Wang, ICML'18]
        Arguments:
        epsilon : target epsilon (epsilon > 0)
        delta : target delta (0 < delta < 1)
        GS : upper bound on L2 global sensitivity (GS >= 0). A higher sensitivity means the function is more susceptible to changes in individual data points, requiring more noise to mask those changes.
        tol : error tolerance for binary search (tol > 0)
        Output:
        sigma : standard deviation of Gaussian noise needed to achieve (epsilon,delta)-DP under global sensitivity GS
    """

    def Phi(t):
        return 0.5 * (1.0 + erf(float(t) / sqrt(2.0)))

    def caseA(epsilon, s):
        return Phi(sqrt(epsilon * s)) - exp(epsilon) * Phi(-sqrt(epsilon * (s + 2.0)))

    def caseB(epsilon, s):
        return Phi(-sqrt(epsilon * s)) - exp(epsilon) * Phi(-sqrt(epsilon * (s + 2.0)))

    def doubling_trick(predicate_stop, s_inf, s_sup):
        while (not predicate_stop(s_sup)):
            s_inf = s_sup
            s_sup = 2.0 * s_inf
        return s_inf, s_sup

    def binary_search(predicate_stop, predicate_left, s_inf, s_sup):
        s_mid = s_inf + (s_sup - s_inf) / 2.0
        while (not predicate_stop(s_mid)):
            if (predicate_left(s_mid)):
                s_sup = s_mid
            else:
                s_inf = s_mid
            s_mid = s_inf + (s_sup - s_inf) / 2.0
        return s_mid

    delta_thr = caseA(epsilon, 0.0)

    if (delta == delta_thr):
        alpha = 1.0

    else:
        if (delta > delta_thr):
            predicate_stop_DT = lambda s: caseA(epsilon, s) >= delta
            function_s_to_delta = lambda s: caseA(epsilon, s)
            predicate_left_BS = lambda s: function_s_to_delta(s) > delta
            function_s_to_alpha = lambda s: sqrt(1.0 + s / 2.0) - sqrt(s / 2.0)

        else:
            predicate_stop_DT = lambda s: caseB(epsilon, s) <= delta
            function_s_to_delta = lambda s: caseB(epsilon, s)
            predicate_left_BS = lambda s: function_s_to_delta(s) < delta
            function_s_to_alpha = lambda s: sqrt(1.0 + s / 2.0) + sqrt(s / 2.0)

        predicate_stop_BS = lambda s: abs(function_s_to_delta(s) - delta) <= tol

        s_inf, s_sup = doubling_trick(predicate_stop_DT, 0.0, 1.0)
        s_final = binary_search(predicate_stop_BS, predicate_left_BS, s_inf, s_sup)
        alpha = function_s_to_alpha(s_final)

    sigma = alpha * GS / sqrt(2.0 * epsilon)
    return sigma

def dp(grad, device,epsilon=8, delta=1e-3):
    global l2_norm_clip
    sigma = calibrateAnalyticGaussianMechanism(epsilon=epsilon, delta=delta, GS=l2_norm_clip)
    result = torch.tensor(np.random.normal(0, sigma, grad.shape), device=device) + grad
    return result

def updateServerModel(server_model, grad_updates, lr, device=None, mode='SGD',epsilon=8, delta=1e-3):
    aggregated=[]
    if mode=='DP':
        for i in range(len(grad_updates)):
            aggregated.append(dp(grad_updates[i], device=device ,epsilon=epsilon, delta=delta))
    if mode=='SGD':
        aggregated=grad_updates
    
    if aggregated:
        if device:
           server_model = server_model.to(device)
           aggregated = [param.to(device) for param in aggregated]
        for param_model, param_update in zip(server_model.parameters(), aggregated):
           param_model.data += (-1.0 * lr) * param_update.data
 
    return server_model

def accuracy(preds, labels):
    acc=(torch.tensor(preds).argmax(dim=1) == torch.tensor(labels).squeeze()).sum()/len(labels)
    return acc

def f1(preds, labels):
    y_pred=torch.tensor(preds).argmax(dim=1)
    y_true=torch.tensor(labels).squeeze()
    macro_f1=f1_score(y_true, y_pred, average='macro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    return macro_f1,micro_f1

def Test(logger,
         test_loader,
         model,
         criterions,
         device
         ):
    loss_avg = RunningAverage()

    logger.info("Testing...")

    labels = []
    preds = []

    with tqdm(total=len(test_loader)) as t:
        #for genes, label, info in tqdm(test_loader):
        #for genes, label in tqdm(test_loader):
        for features, label in tqdm(test_loader):
            #genes, label = genes.to(device), label.to(device)
            features, label = features.to(device), label.to(device)
            #pred = model(genes)
            pred = model(features)
            loss = criterions(pred, label)
            loss_avg.update(loss.item())
            labels += list(label.detach().cpu().numpy())
            preds += list(pred.detach().cpu().numpy())

            t.set_postfix(loss_avg='{:05.3f}'.format(loss_avg()),
                              )
            t.update()
    acc = accuracy(preds, labels)
    macro_f1, micro_f1=f1(preds, labels)
    logger.info('Test loss: {} Test, acc: {}, macro_f1: {}, micro_f1: {}'.format(loss_avg(),acc,macro_f1,micro_f1))
    return loss_avg(), acc   

def Train(logger,
          trainLoaders,
          testLoader,
          serverModel,
          criterions,
          device,
          num_epochs=25,
          model_path="model",
          mode='SGD'):

    global lr
    global epsilon
    global delta
    global numberClients
    global l2_norm_clip
    global shuffle_model

    loss_avg = RunningAverage()

    best_loss = 10 ** 15
    best_acc = 0
    is_best = False

    #optimizer = torch.optim.Adam(serverModel.parameters(), lr=lr)

    for epoch in range(num_epochs):

        logger.info('Epoch {}/{}'.format(epoch+1, num_epochs))
        labels = []
        preds = []
        #serverModel.train()               ########## ????????????????????????????

        if True:

            client_Model_list = []

            for client_idx in range(numberClients):
                logger.info('client id {}'.format(client_idx))
                trainLoader = trainLoaders[client_idx]
                #print("trainLoader:")
                #print(trainLoader)
                
                
                # initial clientModel
                clientModel = initialClientModel(serverModel, device=device)
                optimizer = torch.optim.SGD(clientModel.parameters(), lr=lr)
                clientModel.train()
                                
                with tqdm(total=len(trainLoader)) as t:
                    
                    #### Note: trainLoader will automatically utilizes the __getitem__ method in class Clinical_Data(Dataset). Because trainLoader is a DataLoader. The DataLoader in Python, particularly in libraries like PyTorch, automatically utilizes the __getitem__ method to fetch data because it's fundamental to how Python handles iteration and indexing.
                    #for genes, label, info in tqdm(trainLoader):
                    #for genes, label in tqdm(trainLoader):
                    for features, label in tqdm(trainLoader):
                        #print("genes, label:")
                        #print(genes, label)
                        #genes, label = genes.to(device), label.to(device, dtype=torch.long)
                        features, label = features.to(device), label.to(device, dtype=torch.long)
                        
                        # serverModel.zero_grad()
                        # pred = serverModel(genes)
                        clientModel.zero_grad()
                        #pred = clientModel(genes)
                        pred = clientModel(features)

                        loss = criterions(pred, label)
                        loss.backward()
                        optimizer.step()
                        loss_avg.update(loss.item())
                        # statistics

                        labels += list(label.detach().cpu().numpy())
                        preds += list(pred.detach().cpu().numpy())

                        t.set_postfix(
                            loss_avg='{:05.3f}'.format(loss_avg()),
                            total_loss='{:05.3f}'.format(loss.item()),
                        )
                        t.update()
                        
                client_Model_list.append(clientModel)
                logger.info("Length of model list: {}".format(len(client_Model_list)))

            aggregated_grads = []
            for clientModel in client_Model_list:
                grads = compute_grad_update(serverModel, clientModel, lr, device)
                # update to serverModel
                try:
                    aggregated_grads=[aggregated_grads[i] + grads[i] for i in range(len(grads))]
                except:
                    print('** can not sum up, if this alarm continuously shows more than once, check!')
                    aggregated_grads = grads
            aggregated_grads = [x/numberClients for x in aggregated_grads]

            serverModel = updateServerModel(serverModel, aggregated_grads, mode=mode, lr=lr, device=device, epsilon=epsilon_l,delta=delta_l)

        acc=accuracy(preds, labels)
        macro_f1, micro_f1 = f1(preds, labels)
        logger.info('Epoch Training loss: {}, acc: {}, macro_f1: {}, micro_f1: {}'.format(loss_avg(), acc,macro_f1,micro_f1))

        serverModel.eval()
        test_loss, test_acc = Test(logger,testLoader,serverModel,criterions,device)

        if  test_acc > best_acc:
            is_best = True
            best_loss = test_loss
            best_acc = test_acc
        else:
            is_best = False

        #save_checkpoint(serverModel, is_best, model_path, logger, epoch)
        if not os.path.exists(model_path):
            print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
            os.mkdir(model_path)
        model_name=os.path.join(model_path, '{}'.format(expname)+'_epoch_' + str(epoch) + '_modelbest.tar')
        if is_best:
           logger.info("Saving checkpoint... ")
           logger.info("Saved to {}".format(model_name))
           logger.info("Checkpoint is best!")
           with open(model_name, 'wb') as f:
               torch.save(serverModel.state_dict(), f)
        else:
           logger.info("Model not good, skip!")
        
    return serverModel

####################################### main #######################################################################################################################
start = time.time()
if mode == 'DP' or mode == 'SGD':
     epsilon_l = epsilon/(2*np.sqrt(2*EPOCHS*np.log(2/delta)))
     delta_l = delta/(2*EPOCHS)
     
if 'cuda' in device_name:
     os.environ['CUDA_VISIBLE_DEVICES'] = device_name.split(':')[1]
     device_name = 'cuda:0'

device = torch.device(device_name)
trainDatasets, testDataset = getSurvivalDataset(numberClients,train_index=train_data_file, test_index=test_data_file)

trainLoaders=[DataLoader(dataset, BATCH_SIZE, True) for dataset in trainDatasets]
testLoader = DataLoader(testDataset, BATCH_SIZE, False)

# Loss Function
criterion = nn.CrossEntropyLoss()

# Model
serverModel = DL_module.Model(**DLmodel_para_dict)

if device_name != 'cpu':
    serverModel = serverModel.to(device)
    serverModel = torch.nn.DataParallel(serverModel).cuda()


    
# start train
trained_model = Train(logger,
                    trainLoaders,
                    testLoader,
                    serverModel,
                    criterions=criterion,
                    device=device,
                    num_epochs=EPOCHS,
                    model_path=model_path,
                    mode=mode)

Test(logger,
    testLoader,
    model=trained_model,
    criterions=criterion,
    device=device)  
stop = time.time()
logger.info(f"The time of the run: {stop - start}")