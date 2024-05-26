import torch
from torch import nn as nn
import torch.nn.functional as F

from NF import thops
from NF.Basic import ResNet1D

import pdb

class AffineCoupling(nn.Module):
    def __init__(self, in_channels, cond_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = 32 

        self.n_hidden_layers = 6
        self.cond_channels = cond_channels
        f_in_channels = self.in_channels//2 if cond_channels is None else self.in_channels//2 + cond_channels
        f_out_channels = (self.in_channels - self.in_channels//2) * 2
        # print(f_in_channels, f_out_channels)

        nn_module = 'Resnet'
        if nn_module == 'Resnet':
            self.f = ResNet1D(in_channels=f_in_channels, out_channels=f_out_channels, hidden_channels=32, num_blocks=[1,1])


    def forward(self, z, u=None, y=None, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(z, u, y, logdet)
        else:
            return self.reverse_flow(z, u, y, logdet)


    def normal_flow(self, z, u=None, y=None, logdet=None):
        z1, z2 = thops.split_feature(z, "split")
        h = self.f(z1) if self.cond_channels is None else self.f(thops.cat_feature(z1, u))
        shift, scale = thops.split_feature(h, "cross")
        # adding 1e-4 is crucial for torch.slogdet(), as used in Glow (leads to black rect in experiments).
        # see https://github.com/didriknielsen/survae_flows/issues/5 for discussion.
        # or use `torch.exp(2. * torch.tanh(s / 2.)) as in SurVAE (more unstable in practice).

        # version 1, srflow (use FCN)
        scale = torch.sigmoid(scale + 2.) + 1e-4
        z2 = (z2 + shift) * scale
        logdet += thops.sum(torch.log(scale), dim=[1, 2])
        
        z = thops.cat_feature(z1, z2)

        return z, logdet

    def reverse_flow(self, z, u=None, y=None, logdet=None):
        z1, z2 = thops.split_feature(z, "split")

        h = self.f(z1) if self.cond_channels is None else self.f(thops.cat_feature(z1, u))
        shift, scale = thops.split_feature(h, "cross")

        # version1, srflow
        scale = torch.sigmoid(scale + 2.) + 1e-4
        z2 = (z2 / scale) -shift
        
        logdet -= thops.sum(torch.log(scale), dim=[1, 2])

        z = thops.cat_feature(z1, z2)

        return z, logdet

 