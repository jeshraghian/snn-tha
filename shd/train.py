# snntorch
import snntorch as snn
from snntorch import spikegen
from snntorch import surrogate
from snntorch import functional as SF

# torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

# misc
import os
import numpy as np
import math
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import time

# raytune
# from functools import partial
# from ray import tune
# from ray.tune import CLIReporter
# # from ray.tune import JupyterNotebookReporter
# from ray.tune.schedulers import ASHAScheduler

from dataloader import *
from test import *
from test_acc import *
from tha import *

def train(config, net, epoch, trainloader, testloader, criterion, optimizer, scheduler, device):
    
    net.train()
    loss_accum = []
    lr_accum = []

    # TRAIN
    for data, labels in trainloader:
        data, labels = data.to(device), labels.to(device) 
        spk_rec2, _ = net(data.permute(1, 0, 2))
        loss = criterion(spk_rec2, labels.long()) 
        optimizer.zero_grad()
        loss.backward()

        if config['grad_clip']:
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        if config['weight_clip']:
            with torch.no_grad():
                for param in net.parameters():
                    param.clamp_(-1, 1)

        optimizer.step()
        scheduler.step()
        thr_annealing(config, net)


        loss_accum.append(loss.item()/config['num_steps'])
        lr_accum.append(optimizer.param_groups[0]["lr"])

    return loss_accum, lr_accum
