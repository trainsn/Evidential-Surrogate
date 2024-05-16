# Generator architecture

import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

from resblock import BasicBlockGenerator

class Generator(nn.Module):
    def __init__(self, dsp=3, dspe=512, ch=1, out_features=4, dropout=False):
        """
        Generator Network Constructor
        :param dsp: dimensions of the simulation parameters
        :param dspe: dimensions of the simulation parameters' encode
        :param ch: channel multiplier
        """
        super(Generator, self).__init__()

        self.dsp, self.dspe = dsp, dspe
        self.ch = ch
        self.out_features = out_features

        # simulation parameters subnet
        self.sparams_subnet = nn.Sequential(
            nn.Linear(dsp, dspe), nn.ReLU(),
            nn.Linear(dspe, dspe), nn.ReLU(),
            nn.Linear(dspe, dspe), nn.ReLU(),
            nn.Linear(dspe, ch * 4 * 100)
        )

        # Image generation subnet
        data_layers = [
            BasicBlockGenerator(ch * 4, ch * 2, kernel_size=3, stride=1, padding=1),
            BasicBlockGenerator(ch * 2, ch, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm1d(ch),
            nn.ReLU(),
            nn.Conv1d(ch, out_features, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        ]
        if dropout:
            data_layers.append(nn.Dropout1d(p=0.02))
        
        self.data_subnet = nn.Sequential(*data_layers)

        self.tanh = nn.Tanh()

    def evidence(self, x):
        # Using softplus as the activation function for evidence
        return F.softplus(x)
    
    def DenseNormalGamma(self, x):
        mu, logv, logalpha, logbeta = x.chunk(4, dim=1)
        mu = torch.tanh(mu)
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        # Concatenating the tensors along the last dimension
        return torch.cat([mu, v, alpha, beta], dim=1)

    def forward(self, sp):
        sp = self.sparams_subnet(sp)

        x = sp.view(sp.size(0), self.ch * 4, 100)
        x = self.data_subnet(x)

        if self.out_features == 4:
            x = self.DenseNormalGamma(x)
        else:
            x = self.tanh(x)

        return x