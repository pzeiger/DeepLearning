#!/usr/bin/env python3

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

import pickle
import numpy as np
from load_warwick import load_warwick

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




    



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __call__(self, sample):
        image, labelmask = sample['image'], sample['labelmask']
        
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        # labelmasks have only one color dimension
#        labelmask = labelmask.transpose((2, 0, 1))
#        print(image.shape)
        
        # Blue color is always zero
        assert np.sum(image[2,...]) == 0.0
        image = image[:2,...]
#        print(image.shape)
        
        return {'image': torch.from_numpy(image),
                'labelmask': torch.tensor(labelmask, dtype=torch.float)}


class WarwickDataset(Dataset):
    """WARWICK dataset."""

    def __init__(self, root_dir, transform=ToTensor()):
        """
        Args:
            root_dir (string): Directory with all the images and labelmasks
        """
        self.transform = transform
        try:
            with open(str(root_dir) + '/data.npy','wb') as fh:
                self.images = np.load(fh)
                self.labelmasks = np.load(fh)
        except:
            print('loading')
            self.images, self.labelmasks = load_warwick(root_dir)
            with open(str(root_dir) + '/data.npy','wb') as fh:
                np.save(fh, self.images)
                np.save(fh, self.labelmasks)
#        print(self.images.shape)
#        print(self.images.dtype)
#        print(self.images.min())
#        print(self.images.max())
        
#        print(self.labelmasks.shape)
#        print(self.labelmasks.min())
#        print(self.labelmasks.max())
#        print(np.unique(self.labelmasks))
        
    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = {'image': self.images[idx,:,:,:], 'labelmask': self.labelmasks[idx,:,:]}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


def init(batch_size, shuffle):

    training_data = WarwickDataset('WARWICK/Train')
    test_data = WarwickDataset('WARWICK/Test')
    
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)
    
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    
    return train_dataloader, test_dataloader, device
