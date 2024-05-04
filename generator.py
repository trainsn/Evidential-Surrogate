# Generator architecture

import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

class Generator(nn.Module):
    def __init__(self, dsp=3, dspe=512, ch=1):
        """
        Generator Network Constructor
        :param dsp: dimensions of the simulation parameters
        :param dspe: dimensions of the simulation parameters' encode
        :param ch: channel multiplier
        """
        super(Generator, self).__init__()

        self.dsp, self.dspe = dsp, dspe
        self.ch = ch

        # simulation parameters subnet
        self.sparams_subnet = nn.Sequential(
            nn.Linear(dsp, 1024), nn.ReLU(),
            nn.Linear(1024, 800), nn.ReLU(),
            nn.Linear(800, 500), nn.ReLU(),
            nn.Linear(500, 400),
            nn.Tanh()
        )

    def forward(self, sp):
        x = self.sparams_subnet(sp)
        return x
