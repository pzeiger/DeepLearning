#!/usr/bin/env python3

import torch
from torch import nn
import my_nn_pytorch as mnn

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv_relu_stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2,2),
            nn.Tanh(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2,2),
            nn.Tanh(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(32*7*7, 10),
        )
    def forward(self, x):
        logits = self.conv_relu_stack(x)
        return logits


def params2fname(nepochs, lr, batch_size_train, suffix=None):
    if suffix == None:
        out = 'ex133_convnn_swaporder_tanh_%iepochs_lr%.4f_bs%i.pkl' % (nepochs, lr, batch_size_train)
    else:
        out = 'ex133_convnn_swaporder_tanh_%iepochs_lr%.4f_bs%i_%s.pkl' % (nepochs, lr, batch_size_train, suffix)
    return out


