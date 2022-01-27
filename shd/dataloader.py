import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from snntorch.spikevision import spikedata

def load_data(config):

        data_dir = config['data_dir']
        dt_scalar = 3  # set to 2 for float in our experiments
        
        
        dt = int(1000*dt_scalar)
        num_steps = int(1000/dt_scalar)

        trainset = spikedata.SHD(data_dir, train=True, num_steps=num_steps, dt=dt)
        testset = spikedata.SHD(data_dir, train=False, num_steps=num_steps, dt=dt)

        return trainset, testset