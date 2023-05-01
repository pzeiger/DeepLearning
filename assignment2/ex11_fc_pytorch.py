#!/usr/bin/env python3

import torch
from torch import nn
#import data_handling as dh
import my_nn_pytorch as mnn
#import sys
#import pickle

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 98),
            nn.ReLU(),
            nn.Linear(98, 98),
            nn.ReLU(),
            nn.Linear(98, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def params2fname(nepochs, lr, batch_size_train, suffix=None):
    if suffix == None:
        out = 'ex11_fc_%iepochs_lr%.4f_bs%i.pkl' % (nepochs, lr, batch_size_train)
    else:
        out = 'ex11_fc_%iepochs_lr%.4f_bs%i_%s.pkl' % (nepochs, lr, batch_size_train, suffix)
    return out


#def train(nepochs=30, lr=None, batch_size_train=None, batch_size_test=None):
#    
#    if batch_size_train is None:
#        batch_size_train = 64
#    if batch_size_test is None:
#        batch_size_test = 10000
#    if lr is None:
#        lr = 1e-2
#
#    # Get cpu or gpu device for training.
#    device = "cuda" if torch.cuda.is_available() else "cpu"
#    print("Using {} device".format(device))
#    
#    dataloader_train, dataloader_test = dh.init_MNIST(batch_size_train, batch_size_test)
#    
#    model = Ex11_FullyConnectedNeuralNetwork()
#    loss_fn = nn.CrossEntropyLoss()
#    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
#    mynn = mnn.MyNeuralNetwork(model, loss_fn, optimizer, dataloader_train, dataloader_test, device=device)
#    mynn.train(nepochs=nepochs)
#    mynn.to_disk(params2fname(nepochs, lr, batch_size_train))
#    return
#
#
#
#if __name__ == '__main__':
#    if len(sys.argv) == 1:
#        train()
#    elif len(sys.argv) == 2:
#        train(nepochs=int(sys.argv[1]))
#    elif len(sys.argv) == 3:
#        train(nepochs=int(sys.argv[1]), lr=float(sys.argv[2]))
#    elif len(sys.argv) == 4:
#        train(nepochs=int(sys.argv[1]), lr=float(sys.argv[2]),
#              batch_size_train=int(sys.argv[3]))
#    elif len(sys.argv) == 5:
#        train(nepochs=int(sys.argv[1]), lr=float(sys.argv[2]), 
#              batch_size_train=int(sys.argv[3]), batch_size_test=int(sys.argv[4]))
#

