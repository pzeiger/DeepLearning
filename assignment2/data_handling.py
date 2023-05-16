#!/usr/bin/env python3

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from PIL import Image
import glob

import pickle
import numpy as np


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']



class MNISTDataset(datasets.MNIST):
    
    def __getitem__(self, index):
        
        img, target = super(MNISTDataset, self).__getitem__(index)
        
        return img, target, index




def init_MNIST(batch_size_train, batch_size_test, shuffle=False):
    
    # Download training data from open datasets.
    training_data = MNISTDataset(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    print(training_data)
    
    # Download test data from open datasets.
    test_data = MNISTDataset(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    print(test_data)
    
    # Create data loaders
    dataloader_train = DataLoader(training_data, batch_size=batch_size_train, shuffle=shuffle)
    dataloader_test = DataLoader(test_data, batch_size=batch_size_test, shuffle=shuffle)

    for X, y, index in dataloader_train:
        print("Shape of X for train data [N, C, H, W]: ", X.shape)
        print("Shape of y for train data: ", y.shape, y.dtype)
        break
    
    for X, y, index in dataloader_test:
        print("Shape of X for test data [N, C, H, W]: ", X.shape)
        print("Shape of y for test data: ", y.shape, y.dtype)
        break
    

    return dataloader_train, dataloader_test

    

class WarwickDataset(VisionDataset):
    """WARWICK dataset."""

    def __init__(self, root_dir, transform=ToTensor(), target_transform=ToTensor()):
        """
        Args:
            root_dir (string): Directory with all the images and labelmasks
        """
        self.transform = transform
        self.target_transform = target_transform
        self.data, self.targets = self.from_dir(root_dir)
   

    def get_stats_data(self):
        tmp = np.array([np.array(x) for x in self.data], dtype=np.single)
        mean = tmp.mean(axis=(0,1,2))
        std = tmp.std(axis=(0,1,2))
        return mean, std
        
    
    def from_dir(self, root_dir):
        # Loads the WARWICK dataset from png images
         
        # create list of image objects
        images = []
        labelmasks = []    
        
        for ipath, lpath in zip(sorted(glob.glob(root_dir + "/image*.png")),\
                                sorted(glob.glob(root_dir + "/label*.png"))):
            
            image = Image.open(ipath)
            images.append(image)
            
            labelmask = Image.open(lpath)
            labelmasks.append(labelmask)
            
        return images, labelmasks


    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, index):
        
        img, target = self.data[index], self.targets[index]
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target, index
    
    
    def dataset2tensors(self):
        X = []
        y = []
        for td, tt in zip(self.data, self.targets):
            X.append(self.transform(td))
            y.append(self.target_transform(tt))
        return torch.stack(X), torch.stack(y)




#class ImgToTensor2Channels():
#    def __init__(self, transform=ToTensor()):
#        self.transform = transform
#        
#    def __call__(self, img):
#        img = self.transform(img)
#        return img[:2,...]
#
#
#class TargetTransformCrossEntropy():
#    def __call__(self, target):
#        tmparr = np.array(target)  # Needs to have shape HxW
#        return torch.as_tensor(tmparr.astype(np.double)/255., dtype=torch.int64)
#
#
#def init_warwick(batch_size_train, batch_size_test, shuffle=False):
#
#    training_data = WarwickDataset('WARWICK/Train', transform=ImgToTensor2Channels(), 
#                                   target_transform=TargetTransformCrossEntropy())
#
#    test_data = WarwickDataset('WARWICK/Test', transform=ImgToTensor2Channels(),
#                               target_transform=TargetTransformCrossEntropy())
#
#    train_dataloader = DataLoader(training_data, batch_size=batch_size_train, shuffle=shuffle)
#    test_dataloader = DataLoader(test_data, batch_size=batch_size_test, shuffle=shuffle)
#    
#    return train_dataloader, test_dataloader
#

