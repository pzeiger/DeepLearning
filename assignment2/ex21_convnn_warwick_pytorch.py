
import torch
from torch import nn
import torch.nn.functional as torchfunc
import numpy as np


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        
        self.relu = nn.ReLU()
        self.maxpool_22 = nn.MaxPool2d(2,2)
        self.flatten = nn.Flatten()
        
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3,
                               stride=1, padding=1)
        
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3,
                               stride=1, padding=1)
        
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,
                               stride=1, padding=1)
        
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1,
                               stride=1, padding=0)
        
        self.upconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4,
                                          stride=2, padding=1)
        
        self.upconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4,
                                          stride=2, padding=1)
        
        self.upconv3 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=4,
                                          stride=2, padding=1)
        
        self.upconv4 = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=4,
                                          stride=2, padding=1)

        self.convout = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=1,
                               stride=1, padding=0)
        
        self.softmax_layer = nn.Softmax2d()



    def forward(self, x):
        print('model.forward():')
#        x = self.flatten(x)
#        logits = self.nn_stack(x)
        
        print('input     ', x.shape)
       
        x = self.conv1(x)
        print('conv1     ', x.shape)
        x = self.maxpool_22(x)
        print('maxpool   ', x.shape)
        x = self.relu(x)
       
        x = self.conv2(x)
        print('conv2     ', x.shape)
        x = self.maxpool_22(x)
        print('maxpool   ', x.shape)
        x = self.relu(x)
       
        x = self.conv3(x)
        print('conv3     ', x.shape)
        x = self.maxpool_22(x)
        print('maxpool   ', x.shape)
        x = self.relu(x)
       
        x = self.conv4(x)
        print('conv4     ', x.shape)
        x = self.maxpool_22(x)
        print('maxpool   ', x.shape)
        x = self.relu(x)
       
        x = self.upconv1(x)
        print('upconv1    ', x.shape)
        x = self.relu(x)
       
        x = self.upconv2(x)
        print('upconv2    ', x.shape)
        x = self.relu(x)
       
        x = self.upconv3(x)
        print('upconv3    ', x.shape)
        x = self.relu(x)
       
        x = self.upconv4(x)
        print('upconv4    ', x.shape)
       
        x = self.convout(x)
        print('convout    ', x.shape)
        
        output = self.softmax_layer(x)
        return output




def params2fname(nepochs, lr, batch_size_train, suffix=None):
    if suffix == None:
        out = 'ex21_convnn_warwick_%iepochs_lr%.4f_bs%i.pkl' % (nepochs, lr, batch_size_train)
    else:
        out = 'ex21_convnn_warwick_%iepochs_lr%.4f_bs%i_%s.pkl' % (nepochs, lr, batch_size_train, suffix)
    return out


