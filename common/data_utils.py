import os
import yaml
import natsort
from datetime import datetime
from PIL import Image

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torchvision.datasets import *
from torch.utils.data import Dataset, DataLoader, Subset

from torchvision.utils import save_image, make_grid

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_config(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

class nir_to_3ch:
    def __call__(self, img):
        if np.shape(img)[0] == 1:  # Check if the image has a single channel
            img = img.repeat(3, 1, 1)  # Repeat the channel to make it 3 channels
        return img

def build_transforms(config):
    transform = transforms.Compose([
        #nir_to_3ch(),
        transforms.Resize(tuple(config['data']['resize']), antialias=True),
        transforms.ToTensor(),
        #transforms.RandomHorizontalFlip(),
        transforms.Normalize(tuple(config['data']['normalize_mean']), tuple(config['data']['normalize_std']))
    ])
    return transform

def build_subset(config, dataset, data_type):
    subset_len = config['loader'][data_type]['subset']
    
    if subset_len > len(dataset):
        subset_len = len(dataset)
        print(f"Warning: Subset length for {data_type} is larger than the dataset length. Using the full dataset.")
    return Subset(dataset, range(subset_len))

def build_loader(config):
    loaders = {}

    data_types = config['loader']['target']
    data_root = config['data']['root']
    transform = build_transforms(config)

    for data_type in data_types:
        _subset = config['loader'][data_type]['subset']

        dataset = PairedDataset(data_root, data_type, transform)
        print(f'{data_type} loader generated:', len(dataset))

        if _subset != False:
            dataset = build_subset(config, dataset, data_type)
            print(f'subset applied for {data_type}:', len(dataset))

        loader = DataLoader(
            dataset, 
            batch_size=config['loader'][data_type]['batch_size'],
            shuffle=config['loader'][data_type]['shuffle']
            )
        
        loaders[data_type] = loader

    return loaders

class PairedDataset(Dataset):
    def __init__(self, data_root, data_type, transform):
        assert data_type == 'train' or data_type == 'vali' or data_type == 'test', "d_type must be train or vali or test"

        self.base_path = data_root

        if data_type == 'train':
            self.rgb_dir, self.nir_dir = self._set_path(data_type)

        elif data_type == 'vali':
            self.rgb_dir, self.nir_dir = self._set_path(data_type)

        elif data_type == 'test':
            self.rgb_dir, self.nir_dir = self._set_path(data_type)

        self.transform = transform

        self.rgb_images = natsort.natsorted(os.listdir(self.rgb_dir))
        self.nir_images = natsort.natsorted(os.listdir(self.nir_dir))
        assert len(self.rgb_images) == len(self.nir_images), "RGB and NIR folders must contain the same number of images"

    def _set_path(self, d_type):
        rgb_dir = self.base_path + d_type + '_A'
        nir_dir = self.base_path + d_type + '_B'
        return rgb_dir, nir_dir
    
    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):
        rgb_path = os.path.join(self.rgb_dir, self.rgb_images[idx])
        nir_path = os.path.join(self.nir_dir, self.nir_images[idx])

        rgb_image = Image.open(rgb_path)
        rgb_image = rgb_image.convert("RGB")

        nir_image = Image.open(nir_path)
        nir_image = nir_image.convert("RGB")

        if self.transform:
            rgb_image = self.transform(rgb_image)
            nir_image = self.transform(nir_image)

        return rgb_image, nir_image