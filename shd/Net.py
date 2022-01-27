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
        slope1 = config['slope1']
        beta1 = config['beta1']
        self.num_steps = config['num_steps']
        p1 = config['dropout1']
        p2 = config['dropout2']
        self.binarize = config['binarize']
        num_hidden = 3000

        spike_grad1 = surrogate.fast_sigmoid(slope1)
        spike_grad2 = surrogate.fast_sigmoid(slope2)
        # Initialize layers with spike operator
        
        
        self.bfc1 = BinaryLinear(700, num_hidden)
        self.fc1 = nn.Linear(700, num_hidden)
        self.lif1 = snn.Leaky(beta1, threshold=self.thr1, spike_grad=spike_grad1)
        self.dropout1 = nn.Dropout(p1)
        
        self.bfc2 = BinaryLinear(num_hidden, 20)
        self.fc2 = nn.Linear(num_hidden, 20)
        self.lif2 = snn.Leaky(beta2, threshold=self.thr2, spike_grad=spike_grad2)
        self.dropout2 = nn.Dropout(p2)


    def forward(self, x):

        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky() 
        mem2 = self.lif2.init_leaky()
        
        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        # Binarization

        if self.binarize:

            for step in range(x.size(0)):
            
                cur1 = self.dropout1(self.bfc1(x[step].flatten(1)))
                spk1, mem1 = self.lif1(cur1, mem1)
                cur2 = self.dropout2(self.bfc2(spk1))
                spk2, mem2 = self.lif2(cur2, mem2)


                spk2_rec.append(spk2)
                mem2_rec.append(mem2)

            return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)
        
        # Full Precision
        
        else:

            for step in range(x.size(0)):
            
                cur1 = self.dropout1(self.fc1(x[step].flatten(1)))
                spk1, mem1 = self.lif1(cur1, mem1)
                cur2 = self.dropout2(self.fc2(spk1))
                spk2, mem2 = self.lif2(cur2, mem2)
                spk2_rec.append(spk2)
                mem2_rec.append(mem2)

            return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

        