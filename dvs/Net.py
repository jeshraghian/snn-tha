# snntorch
import snntorch as snn
from snntorch import surrogate

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# local
from bnn import *


class Net(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.thr1 = config['threshold1']
        self.thr2 = config['threshold2']
        self.thr3 = config['threshold3']
        slope = config['slope']
        beta = config['beta']
        self.num_steps = config['num_steps']
        self.batch_norm = config['batch_norm']
        p1 = config['dropout1']
        self.binarize = config['binarize']


        spike_grad = surrogate.fast_sigmoid(slope)
        # Initialize layers with spike operator
        self.bconv1 = BinaryConv2d(2, 16, 5, bias=False)
        self.conv1 = nn.Conv2d(2, 16, 5, bias=False)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.lif1 = snn.Leaky(beta, threshold=self.thr1, spike_grad=spike_grad)
        self.bconv2 = BinaryConv2d(16, 32, 5, bias=False)
        self.conv2 = nn.Conv2d(16, 32, 5, bias=False)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.lif2 = snn.Leaky(beta, threshold=self.thr2, spike_grad=spike_grad)
        self.bfc1 = BinaryLinear(32 * 5 * 5, 11)
        self.fc1 = nn.Linear(32 * 5 * 5, 11)
        self.lif3 = snn.Leaky(beta, threshold=self.thr3, spike_grad=spike_grad)
        self.dropout = nn.Dropout(p1)


    def forward(self, x):

        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky() 
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        
        # Record the final layer
        spk3_rec = []
        mem3_rec = []

        # Binarization

        if self.binarize:

            for step in range(x.size(0)):

                # fc1weight = self.fc1.weight.data
                cur1 = F.avg_pool2d(self.bconv1(x[step]), 2)
                if self.batch_norm:
                    cur1 = self.conv1_bn(cur1)
                spk1, mem1 = self.lif1(cur1, mem1)
                cur2 = F.avg_pool2d(self.bconv2(spk1), 2)
                if self.batch_norm:
                    cur2 = self.conv2_bn(cur2)
                spk2, mem2 = self.lif2(cur2, mem2) 
             
                cur3 = self.dropout(self.bfc1(spk2.flatten(1)))
                spk3, mem3 = self.lif3(cur3, mem3)

                spk3_rec.append(spk3)
                mem3_rec.append(mem3)

            return torch.stack(spk3_rec, dim=0), torch.stack(mem3_rec, dim=0)
        
        # Full Precision
        
        else:  

            for step in range(x.size(0)):
                # fc1weight = self.fc1.weight.data
                cur1 = F.avg_pool2d(self.conv1(x[step]), 2)
                if self.batch_norm:
                    cur1 = self.conv1_bn(cur1)
                spk1, mem1 = self.lif1(cur1, mem1)
                cur2 = F.avg_pool2d(self.conv2(spk1), 2)
                if self.batch_norm:
                    cur2 = self.conv2_bn(cur2)
                spk2, mem2 = self.lif2(cur2, mem2)
             
                cur3 = self.dropout(self.fc1(spk2.flatten(1)))
                spk3, mem3 = self.lif3(cur3, mem3)


                spk3_rec.append(spk3)
                mem3_rec.append(mem3)


            return torch.stack(spk3_rec, dim=0), torch.stack(mem3_rec, dim=0)
        