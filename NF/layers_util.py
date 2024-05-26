import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# used in ResNet
class BasicBlockUp_leaky(nn.Module):
    def __init__(self, inchannel, outchannel, sf=2, upsample=False):
        super(BasicBlockUp_leaky, self).__init__()
        self.upsample = upsample
        self.conv_res = None
        if self.upsample or inchannel != outchannel:
            self.conv_res = nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=1, padding=0, bias=False)

        self.bn1 = nn.BatchNorm1d(inchannel)
        self.conv1 = nn.Conv1d(inchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm1d(outchannel)
        self.conv2 = nn.Conv1d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)

        self.upsample_fun = nn.Upsample(scale_factor=sf, mode='linear', align_corners=True)   

    def forward(self, x):
        residual = x

        if self.upsample:
            residual = self.upsample_fun(residual)
            # residual = F.interpolate(residual, scale_factor=2)
        if self.conv_res is not None:
            residual = self.conv_res(residual)

        out = self.bn1(x)
        out = self.leakyrelu(out)
        if self.upsample:
            # out = F.interpolate(out, scale_factor=2)
            out = self.upsample_fun(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.leakyrelu(out)
        out = self.conv2(out)
        return out + residual


# used in ResNet
class BasicBlockDown_leaky(nn.Module):
    def __init__(self, inchannel, outchannel, downsample=False):
        super(BasicBlockDown_leaky, self).__init__()
        self.downsample = downsample
        self.conv_res = None
        if self.downsample or inchannel != outchannel:
            self.conv_res = nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=1, padding=0, bias=False)

        if self.downsample:
            self.downconv1 = nn.Conv1d(inchannel, inchannel, 4, stride=2, padding=1)
            self.downconv2 = nn.Conv1d(inchannel, inchannel, 4, stride=2, padding=1)
        
        self.bn1 = nn.BatchNorm1d(inchannel)
        self.conv1 = nn.Conv1d(inchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm1d(outchannel)
        self.conv2 = nn.Conv1d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)
    
    def forward(self, x):
        residual = x

        if self.downsample:
            # residual = F.interpolate(residual, scale_factor=2)
            residual = self.downconv1(residual)
        if self.conv_res is not None:
            residual = self.conv_res(residual)

        out = self.bn1(x)
        out = self.leakyrelu(out)
        if self.downsample:
            # out = F.interpolate(out, scale_factor=2)
            out = self.downconv2(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.leakyrelu(out)
        out = self.conv2(out)
        return out + residual


class ResidualBlock(torch.nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size=3, stride=1):

        super(ResidualBlock, self).__init__()

        self.residual_block = torch.nn.Sequential(
            torch.nn.Conv1d(inchannel, out_channels=outchannel, kernel_size=kernel_size, stride=stride, padding=1),
            torch.nn.BatchNorm1d(outchannel),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            torch.nn.Conv1d(outchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=1),
            torch.nn.BatchNorm1d(outchannel),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        return x + self.residual_block(x)
    