# build the neural network to approximate the selection net
import numpy
import torch
from torch import tanh, squeeze, sin, sigmoid, autograd
from torch.nn.functional import relu


# 4 layer FC ResNet with 1 residual block
class network_time_depedent(torch.nn.Module):
    def __init__(self, d, m):
        super(network_time_depedent, self).__init__()
        self.layer1 = torch.nn.Linear(d+1,m)
        self.layer2 = torch.nn.Linear(m,m)
        self.layer3 = torch.nn.Linear(m, m)
        self.layer4 = torch.nn.Linear(m, m)
        self.layer5 = torch.nn.Linear(m,1)
        self.activation = lambda x: relu(x**3)

    def forward(self, tensor_x_batch):  
        y = self.layer1(tensor_x_batch)
        y = self.layer2(self.activation(y))
        y_copy = y
        y = self.layer3(self.activation(y))
        y = self.layer4(self.activation(y))
        y = y + y_copy
        y = self.layer5(self.activation(y))
        y = y.squeeze(1)
        return y


    # to evaluate the solution with numpy array input and output
    def predict(self, x_batch):
        tensor_x_batch = torch.Tensor(x_batch)
        tensor_x_batch.requires_grad=False
        y = self.forward(tensor_x_batch)
        return y.cpu().detach().numpy()
    

