# Generator architecture

import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

class Generator(nn.Module):
    def __init__(self, dsp, out_features):
        """
        Generator Network Constructor
        :param dsp: dimensions of the simulation parameters
        """
        super(Generator, self).__init__()

        self.out_features = out_features

        # simulation parameters subnet
        self.sparams_subnet = nn.Sequential(
            nn.Linear(dsp, 1024), nn.ReLU(),
            nn.Linear(1024, 800), nn.ReLU(),
            nn.Linear(800, 500), nn.ReLU(),
            nn.Linear(500, 400 * out_features)
        )
        self.tanh = nn.Tanh()

    def evidence(self, x):
        # Using softplus as the activation function for evidence
        return F.softplus(x)
    
    def DenseNormalGamma(self, x):
        x = x.reshape(-1, 400, self.out_features)
        mu, logv, logalpha, logbeta = x.chunk(4, dim=-1)
        mu = F.tanh(mu)
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        # Concatenating the tensors along the last dimension
        return torch.cat([mu, v, alpha, beta], dim=-1)

    def forward(self, sp):
        x = self.sparams_subnet(sp)

        if self.out_features == 4:
            x = self.DenseNormalGamma(x)
        else:
            x = self.tanh(x)
        return x
