import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from snntorch.spikevision import spikedata

def load_data(config):
        data_dir = config['data_dir']

        trainset = spikedata.DVSGesture(data_dir, train=True, num_steps=100, dt=5000, ds=4)
        testset = spikedata.DVSGesture(data_dir, train=False, num_steps=360, dt=5000, ds=4)

        return trainset, testset