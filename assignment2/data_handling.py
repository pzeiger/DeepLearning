#!/usr/bin/env python3

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

import pickle
import numpy as np
#from load_warwick import load_warwick

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def init_MNIST(batch_size_train, batch_size_test, shuffle=False):
    
    # Download training data from open datasets.
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    print(training_data)
    
    # Download test data from open datasets.
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    print(test_data)
    
    # Create data loaders
    dataloader_train = DataLoader(training_data, batch_size=batch_size_train, shuffle=shuffle)
    dataloader_test = DataLoader(test_data, batch_size=batch_size_test, shuffle=shuffle)

    for X, y in dataloader_train:
        print("Shape of X for train data [N, C, H, W]: ", X.shape)
        print("Shape of y for train data: ", y.shape, y.dtype)
        break
    
    for X, y in dataloader_test:
        print("Shape of X for test data [N, C, H, W]: ", X.shape)
        print("Shape of y for test data: ", y.shape, y.dtype)
        break
    
    return dataloader_train, dataloader_test



    



