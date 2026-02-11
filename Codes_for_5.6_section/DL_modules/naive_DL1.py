import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model,self).__init__()

        self.hidden1 = nn.Linear(input_size,2048) # 2048
        self.hidden2 = nn.Linear(2048, 1024)  # hidden layer
        self.hidden3 = nn.Linear(1024, 256)
        self.hidden4 = nn.Linear(256, output_size)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden1(x))  # activation function for hidden layer
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.hidden4(x)  # linear output
        x = F.log_softmax(x,dim=1)
        return x