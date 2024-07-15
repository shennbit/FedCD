# encoding: utf-8
"""
Read images and corresponding labels.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import os
import itertools
from torch.utils.data.sampler import Sampler

import random
from glob import glob
import h5py
from scipy.ndimage.interpolation import zoom
from scipy import ndimage


class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = os.listdir(self._base_dir + '/data')

        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]

        train_data_name = self._base_dir + '/data/' + case
        I = Image.open(train_data_name)
        if len(I.split()) != 1:
            I = I.convert("L")

        if self.transform is not None:
                I = self.transform(I)
            
        train_label_name = self._base_dir + '/label/' + case
        if os.path.exists(train_label_name):
            L = Image.open(train_label_name)
            if len(L.split()) != 1:
                L = L.convert("L")
            L = np.array(L)
            L = L.astype('uint8')
            #L = torch.FloatTensor(L)

            sample = {'image': I, 'label': L, 'idx':idx}
        else:
            sample = {'image': I, 'idx':idx}

        return sample


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2