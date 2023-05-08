
import torch
from torch import nn
import numpy as np


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.nn_stack = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(in_channels=8, out_channels=2, kernel_size=1, stride=1, padding=0),
        )
    
    def forward(self, x):
        logits = self.nn_stack(x)
        return logits


def params2fname(nepochs, lr, batch_size_train, suffix=None):
    if suffix == None:
        out = 'ex21_convnn_warwick_%iepochs_lr%.4f_bs%i.pkl' % (nepochs, lr, batch_size_train)
    else:
        out = 'ex21_convnn_warwick_%iepochs_lr%.4f_bs%i_%s.pkl' % (nepochs, lr, batch_size_train, suffix)
    return out


