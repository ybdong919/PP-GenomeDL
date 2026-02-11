import torch
import torch.nn as nn
import torch.nn.functional as F

save_memory = False
#act_fun=args.act_fun
act_fun='relu'

########### the device name should be consistent with device setting value in the file config.ini.
#device = torch.device('cpu')

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
        #x = x *((1 - torch.eye(d, d)).to(device))
        x = x *(1 - torch.eye(d, d))
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