#!/usr/bin/env python3

import torch
from torch import nn
import my_nn_pytorch as mnn

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


