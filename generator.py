# Generator architecture

import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

from resblock import BasicBlockGenerator

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
            nn.Linear(dsp, dspe), nn.ReLU(),
            nn.Linear(dspe, dspe), nn.ReLU(),
            nn.Linear(dspe, ch * 4 * 400)
        )

        # image generation subnet
        self.data_subnet = nn.Sequential(
            BasicBlockGenerator(ch * 4, ch * 2, kernel_size=3, stride=1, padding=1),
            BasicBlockGenerator(ch * 2, ch, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm1d(ch),
            nn.ReLU(),
            nn.Conv1d(ch, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, sp):
        sp = self.sparams_subnet(sp)

        x = sp.view(sp.size(0), self.ch * 4, 400)
        x = self.data_subnet(x)

        return x
