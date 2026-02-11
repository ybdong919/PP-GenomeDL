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
    target_label=torch.tensor(target_label)
    aim_flatten = torch.zeros(1, 20531).to(device)
    v = torch.zeros(1, 20531).to(device)
    aim_flatten.requires_grad = True
    costn_1 = 10
    b = 0
    g = 0
    out = mynet.forward(aim_flatten.detach())
    after_softmax = F.softmax(out, dim=-1)
    predict = torch.argmax(after_softmax)
    #print(predict)
    for i in range(alpha):
        out = mynet.forward(aim_flatten)
        if aim_flatten.grad is not None:
            aim_flatten.grad.zero_()
        out = out.reshape(1, classes)
        target_class = torch.tensor([target_label]).to(device)
        
        print("target_class: ")
        print(str(target_class))
        
        cost = nn.CrossEntropyLoss()(out, target_class)
        cost.backward()
        aim_grad = aim_flatten.grad
        # see https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
        aim_flatten = aim_flatten-learning_rate*(momentum*v+aim_grad)
        aim_flatten = Process(aim_flatten)
        aim_flatten = torch.clamp(aim_flatten.detach(), 0, 1)
        aim_flatten.requires_grad = True
        #print(i,cost)
        logger.info('{}/{}: {}'.format(i,alpha,cost.detach().cpu().numpy()))
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
    after_softmax = F.softmax(out, dim=-1)
    print("after_softmax:")
    print(after_softmax)
    
    predict = torch.argmax(after_softmax)
    print(predict,cost)
    oim_flatten = aim_flatten.detach().cpu().numpy()
    with open(f'{log_dir}/{target_label}.npy', 'wb') as f:
        np.save(f, oim_flatten)

# class Model(nn.Module):
    # def __init__(self, input_size, output_size):
        # super(Model,self).__init__()

        # self.hidden1 = nn.Linear(input_size,2048) # 2048
        # self.hidden2 = nn.Linear(2048, 1024)  # hidden layer
        # self.hidden3 = nn.Linear(1024, 256)
        # self.hidden4 = nn.Linear(256, output_size)  # output layer

    # def forward(self, x):
        # x = F.relu(self.hidden1(x))  # activation function for hidden layer
        # x = F.relu(self.hidden2(x))
        # x = F.relu(self.hidden3(x))
        # x = self.hidden4(x)  # linear output
        # x = F.log_softmax(x,dim=1)
        # return x


##################################################################################################################################################################
######################################################### instert Attention Models ################################################################################
save_memory = False
#act_fun=args.act_fun
act_fun='relu'
class mulitiattention(torch.nn.Module):
    #def __init__(self, batch_size,n_head,n_gene,n_feature,query_gene,mode):
    def __init__(self, batch_size,n_head,n_gene,n_feature):
        super(mulitiattention, self).__init__()
        self.n_head=n_head
        self.n_gene = n_gene
        self.batch_size=batch_size
        self.n_feature=n_feature
        #self.mode=mode
        #self.query_gene=query_gene

        self.WQ = nn.Parameter(torch.Tensor(self.n_head, n_feature, 1), requires_grad=True)
        self.WK = nn.Parameter(torch.Tensor(self.n_head,n_feature,1),requires_grad=True)
        self.WV = nn.Parameter(torch.Tensor(self.n_head,n_feature,1),requires_grad=True)
        torch.nn.init.xavier_normal_(self.WQ,gain=1)
        torch.nn.init.xavier_normal_(self.WK,gain=1)

        torch.nn.init.xavier_normal_(self.WV)
        self.W_0=nn.Parameter(torch.Tensor(self.n_head*[0.001]),requires_grad=True)
        print('init')
        # gpu_tracker.track()

    def QK_diff(self,Q_seq, K_seq):
        QK_dif = -1 * torch.pow((Q_seq - K_seq),2)
        return torch.nn.Softmax(dim=2)(QK_dif)

    def mask_softmax_self(self,x):
        d=x.shape[1]
        x = x *((1 - torch.eye(d, d)).to(device))
        return x

    def attention(self,x,Q_seq,WK,WV):
        #if self.mode == 0:
        K_seq = x * WK
        K_seq = K_seq.expand(K_seq.shape[0], K_seq.shape[1], self.n_gene)
        K_seq = K_seq.permute(0, 2, 1)
        V_seq = x * WV
        QK_product = Q_seq * K_seq
        z=torch.nn.Softmax(dim=2)(QK_product)

        z=self.mask_softmax_self(z)
        out_seq=torch.matmul(z, V_seq)

        ############this part is not working well
        # if self.mode == 1:
            # zz_list = []
            # for q in range(self.n_gene // self.query_gene):
                # # gpu_tracker.track()
                # K_seq = x * WK
                # V_seq = x * WV
                # Q_seq_x = x[:, (q * self.query_gene):((q + 1) * self.query_gene), :]
                # Q_seq = Q_seq_x.expand(Q_seq_x.shape[0], Q_seq_x.shape[1], self.n_gene)
                # K_seq = K_seq.expand(K_seq.shape[0], K_seq.shape[1], self.query_gene)
                # K_seq = K_seq.permute(0, 2, 1)

                # QK_diff = self.QK_diff(Q_seq, K_seq)
                # z = torch.nn.Softmax(dim=2)(QK_diff)
                # z = torch.matmul(z, V_seq)
                # zz_list.append(z)
            # out_seq = torch.cat(zz_list, dim=1)
            ####################################
        return out_seq

    def forward(self, x):

        x = torch.reshape(x, (x.shape[0], x.shape[1], 1))
        out_h = []
        for h in range(self.n_head):
            Q_seq = x * self.WQ[h,:,:]
            Q_seq = Q_seq.expand(Q_seq.shape[0], Q_seq.shape[1], self.n_gene)
            if save_memory:
                attention_out=cp(self.attention,x, Q_seq, self.WK[h,:,:], self.WV[h,:,:])
            else:
                attention_out=self.attention(x, Q_seq, self.WK[h,:,:], self.WV[h,:,:])

            out_h.append(attention_out)
        out_seq=torch.cat(out_h,dim=2)
        out_seq=torch.matmul(out_seq,self.W_0)
        return out_seq

class layernorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(layernorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class res_connect(nn.Module):
##########    A residual connection followed by a layer norm.
    def __init__(self, size, dropout):
        super(res_connect, self).__init__()
        self.norm = layernorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, out):
        ###Apply residual connection to any sublayer with the same size
        return x + self.norm(self.dropout(out))

#class MyNet(torch.nn.Module):
class Model(torch.nn.Module):
    #def __init__(self, batch_size,n_head,n_gene,n_feature,n_class,query_gene,d_ff,dropout_rate,mode):
    def __init__(self, batch_size,n_head,n_gene,n_feature,n_class,d_ff,dropout_rate):
        #super(MyNet, self).__init__()
        super(Model, self).__init__()
        self.n_head=n_head
        self.n_gene = n_gene
        self.batch_size=batch_size
        self.n_feature=n_feature
        self.n_class=n_class
        self.d_ff=d_ff
        self.mulitiattention1=mulitiattention(self.batch_size,self.n_head,self.n_gene,self.n_feature)
        #self.mulitiattention2 = mulitiattention(self.batch_size, self.n_head, self.n_gene, self.n_feature)
        #self.mulitiattention3 = mulitiattention(self.batch_size, self.n_head, self.n_gene, self.n_feature)
        self.fc = nn.Linear(self.n_gene, self.n_class)
        torch.nn.init.xavier_uniform_(self.fc.weight,gain=1)
        self.ffn1=nn.Linear(self.n_gene, self.d_ff)
        self.ffn2 = nn.Linear(self.d_ff,self.n_gene)
        self.dropout=nn.Dropout(dropout_rate)
        self.sublayer=res_connect(n_gene,dropout_rate)

    def feedforward(self,x):
        out=F.relu(self.ffn1(x))
        out=self.ffn2(self.dropout(out))
        return out

    def forward(self, x):

        out_attn= self.mulitiattention1(x)
        out_attn_1=self.sublayer(x,out_attn)
        # out_attn_2 = self.mulitiattention2(out_attn_1)
        # out_attn_2=self.sublayer(out_attn_1,out_attn_2)
        # out_attn_3 = self.mulitiattention3(out_attn_2)
        # out_attn_3=self.sublayer(out_attn_2,out_attn_3)
        if act_fun=='relu':
            #out_attn_3=F.relu(out_attn_3)
            out_attn_1=F.relu(out_attn_1)
        if act_fun=='leakyrelu':
            m=torch.nn.LeakyReLU(0.1)
            #out_attn_3=m(out_attn_3)
            out_attn_1=m(out_attn_1)
        if act_fun=='gelu':
            m = torch.nn.GELU()
            #out_attn_3=m(out_attn_3)
            out_attn_1=m(out_attn_1)
        #y_pred = self.fc(out_attn_3)
        y_pred = self.fc(out_attn_1)
        y_pred=F.log_softmax(y_pred, dim=1)

        return y_pred

########### how to use the Model above ####################
#model=Model(batch_size,n_head,n_gene,n_feature,n_class,query_gene,d_ff,dropout_rate,mode=0).to(device) # mode 1 is not working right now.
#optimizer =torch.optim.Adam(model.parameters(), lr=lr_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

##################################################################################################################################################################
##################################################################################################################################################################

cancerTypes = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'COADREAD', 'DLBC', 'ESCA', 'GBM', 'GBMLGG', 'HNSC',
                   'KICH', 'KIPAN', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG',
                   'PRAD', 'READ', 'SARC', 'SKCM', 'STAD', 'STES', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM']

parser = argparse.ArgumentParser(description='dfsa')
parser.add_argument('--model', default='SGD_data0_rep0_modelbest.tar')
parser.add_argument('--lr',type=float,default=0.1)
args = parser.parse_args()
model_id=args.model
model_id=model_id.split('/')[1]

gpus = 0
data_workers = 0
# batch_size = 64
classes = len(cancerTypes)
root_dir = "MIA_CC/"
#model_id='SGD_data0_rep0_modelbest.tar'
model_weight = "model/{}".format(model_id)
print("model_weight:")
print(model_weight)

alpha = 10000
beta = 100
gama = 0.001
learning_rate = args.lr
momentum = 0.9

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

#net = Model(input_size=20531,output_size=len(cancerTypes))
net = Model(batch_size=8,n_head=1,n_gene=20531,n_feature=20531,n_class=len(cancerTypes),d_ff=1024,dropout_rate=0.3)


############ Only if GPU, use  nn.DataParallel(). Note: nn.DataParallel() is also comment in 02.simulationApp_debug.cpu.py. #############################################################
############ As the saved model/checkpoint did not use "nn.DataParallel()" in the file 02.simulationApp_debug.cpu.py, here we cannot use nn.ataParallel() to load the saved model. ######
# mynet = nn.DataParallel(net, output_device=device).train(False)
# mynet=mynet.to(device)
mynet = net.train(False)


assert os.path.exists(model_weight)
mynet.load_state_dict(torch.load(open(model_weight, 'rb')))

for i in range(classes):
    logger.info(f'---class{i}---')
    Attack(mynet=mynet, target_label=i,device=device)
