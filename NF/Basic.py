import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
import math

from NF.ActNorms import ActNorm1d
from NF import thops

from NF.layers_util import BasicBlockUp_leaky, BasicBlockDown_leaky


class Conv1d(nn.Conv1d):
    pad_dict = {
        "same": lambda kernel, stride: [((k - 1) * s + 1) // 2 for k, s in zip(kernel, stride)],
        "valid": lambda kernel, stride: [0 for _ in kernel]
    }

    @staticmethod
    def get_padding(padding, kernel_size, stride):
        # make padding
        if isinstance(padding, str):
            if isinstance(kernel_size, int):
                kernel_size = [kernel_size]
            if isinstance(stride, int):
                stride = [stride]
            padding = padding.lower()
            try:
                padding = Conv1d.pad_dict[padding](kernel_size, stride)
            except KeyError:
                raise ValueError("{} is not supported".format(padding))
        return padding

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding="same", groups=1,
                 do_actnorm=True, weight_std=0.05):
        padding = Conv1d.get_padding(padding, kernel_size, stride)
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, groups=groups, bias=(not do_actnorm))
        # init weight with std
        self.weight.data.normal_(mean=0.0, std=weight_std)
        if not do_actnorm:
            self.bias.data.zero_()
        else:
            self.actnorm = ActNorm1d(out_channels)
        self.do_actnorm = do_actnorm

    def forward(self, input):
        x = super().forward(input)
        if self.do_actnorm:
            x, _ = self.actnorm(x)
        return x


# added
class Conv1dZeros(nn.Conv1d):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1,
                 padding="same", logscale_factor=3):
        padding = Conv1d.get_padding(padding, kernel_size, stride)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)
        self.logscale_factor = logscale_factor
        self.register_parameter("logs_", nn.Parameter(torch.zeros(out_channels, 1)))
        self.register_parameter("bias_", nn.Parameter(torch.zeros(out_channels, 1)))
        # init
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        output = super().forward(input)
        return (output + self.bias_) * torch.exp(self.logs_ * self.logscale_factor)


class GaussianDiag:
    Log2PI = float(np.log(2 * np.pi))

    @staticmethod
    def likelihood(mean, logs, x): # logs: log(sigma)
        """
        lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
              k = 1 (Independent)
              Var = logs ** 2
              ln|Var| = logs * 2
        """
        if mean is None and logs is None:
            return -0.5 * (x ** 2 + GaussianDiag.Log2PI)
        else:
            # x = x - mean
            # return (-0.5 * (logs * 2. + torch.matmul(x[:, :, None], x[:, None, :]) / torch.exp(logs * 2.) + GaussianDiag.Log2PI))
            return -0.5 * (logs * 2. + ((x - mean) ** 2) / torch.exp(logs * 2.) + GaussianDiag.Log2PI)

    @staticmethod
    def logp(mean, logs, x):
        likelihood = GaussianDiag.likelihood(mean, logs, x)
        if len(x.shape) == 2:
            return thops.sum(likelihood, dim=[1])
        elif len(x.shape) == 3:
            return thops.sum(likelihood, dim=[1, 2])
        elif len(x.shape) == 4:
            return thops.sum(likelihood, dim=[1, 2, 3])
        elif len(x.shape) == 5:
            return thops.sum(likelihood, dim=[1, 2, 3, 4])

    @staticmethod
    def sample(mean, logs, eps_std=1, eps=None):
        # eps_std = eps_std or 1 # may cause problem when eps_std is 0
        if eps is None:
            eps = torch.normal(mean=torch.zeros_like(mean),
                                std=torch.ones_like(logs) * eps_std)
        return mean + torch.exp(logs) * eps

    @staticmethod
    def sample_eps(shape, eps_std, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        eps = torch.normal(mean=torch.zeros(shape),
                           std=torch.ones(shape) * eps_std)
        return eps


def squeeze2d(input, factor=2):
    assert factor >= 1 and isinstance(factor, int)
    if factor == 1:
        return input
    size = input.size()
    B = size[0]
    C = size[1]
    H = size[2]
    W = size[3]
    # print(H % factor, W % factor, factor, size) # 11217
    assert H % factor == 0 and W % factor == 0, "{}".format((H, W, factor))
    x = input.view(B, C, H // factor, factor, W // factor, factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(B, C * factor * factor, H // factor, W // factor)
    return x


def unsqueeze2d(input, factor=2):
    assert factor >= 1 and isinstance(factor, int)
    factor2 = factor ** 2
    if factor == 1:
        return input
    size = input.size()
    B = size[0]
    C = size[1]
    H = size[2]
    W = size[3]
    assert C % (factor2) == 0, "{}".format(C)
    x = input.view(B, C // factor2, factor, factor, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(B, C // (factor2), H * factor, W * factor)
    return x


def squeeze1d(input, factor=4):
    assert factor >= 1 and isinstance(factor, int)
    if factor == 1:
        return input
    size = input.size()
    B = size[0]
    C = size[1]
    H = size[2]
    assert H % factor == 0, "{}".format((H, factor))
    x = input.view(B, C, H // factor, factor)
    x = x.permute(0, 1, 3, 2).contiguous()
    x = x.view(B, C * factor, H // factor)
    return x


def unsqueeze1d(input, factor=4):
    assert factor >= 1 and isinstance(factor, int)
    if factor == 1:
        return input
    size = input.size()
    B = size[0]
    C = size[1]
    H = size[2]
    assert C % (factor) == 0, "{}".format(C)
    x = input.view(B, C // factor, factor, H)
    x = x.permute(0, 1, 3, 2).contiguous()
    x = x.view(B, C // (factor), H * factor)
    return x


def squeeze3d(input, factor=2):
    assert factor >= 1 and isinstance(factor, int)
    if factor == 1:
        return input
    size = input.size()
    B = size[0]
    C = size[1]
    D = size[2]
    H = size[3]
    W = size[4]
    # print(H % factor, W % factor, factor, size) # 11217
    assert D % factor == 0 and H % factor == 0 and W % factor == 0, "{}".format((D, H, W, factor))
    x = input.view(B, C, D // factor, factor, H // factor, factor, W // factor, factor)
    x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
    x = x.view(B, C * factor * factor * factor, D // factor, H // factor, W // factor)
    return x


def unsqueeze3d(input, factor=2):
    assert factor >= 1 and isinstance(factor, int)
    factor3 = factor ** 3
    if factor == 1:
        return input
    size = input.size()
    B = size[0]
    C = size[1]
    D = size[2]
    H = size[3]
    W = size[4]
    assert C % (factor3) == 0, "{}".format(C)
    x = input.view(B, C // factor3, factor, factor, factor, D, H, W)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
    x = x.view(B, C // (factor3), D * factor, H * factor, W * factor)
    return x



class SqueezeLayer(nn.Module):
    def __init__(self, factor, dim='3D'):
        super().__init__()
        self.factor = factor
        self.dim = dim

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            if self.dim == '2D':
                output = squeeze2d(input, self.factor)
            if self.dim == '1D':
                output = squeeze1d(input, self.factor)
            if self.dim == '3D':
                output = squeeze3d(input, self.factor)
            return output, logdet
        else:
            if self.dim == '2D':
                output = unsqueeze2d(input, self.factor)
            if self.dim == '1D':
                output = unsqueeze1d(input, self.factor)
            if self.dim == '3D':
                output = unsqueeze3d(input, self.factor)
            return output, logdet


class UnSqueezeLayer(nn.Module):
    def __init__(self, factor, dim='2D'):
        super().__init__()
        self.factor = factor
        self.dim = dim

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            if self.dim == '2D':
                output = unsqueeze2d(input, self.factor)
            if self.dim == '1D':
                output = unsqueeze1d(input, self.factor)
            if self.dim == '3D':
                output = unsqueeze3d(input, self.factor)
            return output, logdet
        else:
            if self.dim == '2D':
                output = squeeze2d(input, self.factor)
            if self.dim == '1D':
                output = squeeze1d(input, self.factor)
            if self.dim == '3D':
                output = squeeze3d(input, self.factor)
            return output, logdet


class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            output = torch.sigmoid(input)
            logdet += -thops.sum(F.softplus(input)+F.softplus(-input), dim=[1, 2, 3, 4])
            return output, logdet
        else:
            output = -torch.log(torch.reciprocal(input) - 1.)
            logdet += -thops.sum(torch.log(input) + torch.log(1.-input), dim=[1, 2, 3, 4])
            return output, logdet


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet1D(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_blocks):
        super(ResNet1D, self).__init__()
        self.in_channels = in_channels

        # Initial convolution
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # Residual blocks
        self.layer1 = self._make_layer(hidden_channels, hidden_channels*2, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(hidden_channels*2,hidden_channels*4 , num_blocks[1], stride=1)

        # Classifier
        self.conv2 = nn.Conv1d(hidden_channels*4, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock1D(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x


class Split(nn.Module):
    def __init__(self, num_channels_split, level=0):
        super().__init__()
        self.num_channels_split = num_channels_split
        self.level = level

    def forward(self, z, z2=None, reverse=False):
        if not reverse:
            return z[:, :self.num_channels_split, ...], z[:, self.num_channels_split:, ...]
        else:
            return torch.cat((z, z2), dim=1)

# -----------------------------------------------------------------------------------------
# upscale 1D feature map
class UpscaleNetX4(nn.Module):
    def __init__(self, inch=8, outch=8):
        super(UpscaleNetX4, self).__init__() # inchannel, outchannel, upsample=False
        self.Block1 = BasicBlockUp_leaky(inch, inch*4, upsample=False)   # (8)  4  --> (32) 4
        self.Block2 = BasicBlockUp_leaky(inch*4, inch*2, upsample=True)  # (32) 4  --> (16) 8
        self.Block3 = BasicBlockUp_leaky(inch*2, inch,  upsample=True)   # (16) 8  --> (8)  16
        self.bn = nn.BatchNorm1d(inch)
        self.conv = nn.Conv1d(inch, outch, kernel_size=3, stride=1, padding=1)
        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = self.Block1(x)
        out = self.Block2(out)
        out = self.Block3(out)
        out = self.bn(out)
        out = self.leakyrelu(self.conv(out)) # in: 8, 4, 4, 4 ; out: 8, 16, 16, 16
        return out


# upscale1D feature map
class UpscaleNetX1(nn.Module):
    def __init__(self, inch=8, outch=8):
        super(UpscaleNetX1, self).__init__() # inchannel, outchannel, upsample=False
        self.Block1 = BasicBlockUp_leaky(inch, inch*4, upsample=False)    # (8)  16  --> (32) 16
        self.Block2 = BasicBlockUp_leaky(inch*4, inch*2, upsample=False)  # (32) 16  --> (16) 16
        self.Block3 = BasicBlockUp_leaky(inch*2, inch,  upsample=False)   # (16) 16  --> (8)  16
        self.bn = nn.BatchNorm1d(inch)
        self.conv = nn.Conv1d(inch, outch, kernel_size=3, stride=1, padding=1)
        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = self.Block1(x)
        out = self.Block2(out)
        out = self.Block3(out)
        out = self.bn(out)
        out = self.leakyrelu(self.conv(out)) # in: 8, 4, 4, 4 ; out: 8, 16, 16, 16
        return out
        

class UpscaleNetX2(nn.Module):
    def __init__(self, inch=8, outch=8):
        super(UpscaleNetX2, self).__init__() # inchannel, outchannel, upsample=False
        self.Block1 = BasicBlockUp_leaky(inch, inch*4, upsample=False)    # (8)  16  --> (32) 16
        self.Block2 = BasicBlockUp_leaky(inch*4, inch*2, upsample=True)  # (32) 16  --> (16) 16
        self.Block3 = BasicBlockUp_leaky(inch*2, inch,  upsample=False)   # (16) 16  --> (8)  16
        self.bn = nn.BatchNorm1d(inch)
        self.conv = nn.Conv1d(inch, outch, kernel_size=3, stride=1, padding=1)
        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = self.Block1(x)
        out = self.Block2(out)
        out = self.Block3(out)
        out = self.bn(out)
        out = self.leakyrelu(self.conv(out)) # in: 8, 4, 4, 4 ; out: 8, 16, 16, 16
        return out


class DownscaleNetX2(nn.Module):
    def __init__(self, inch=8, outch=8):
        super(DownscaleNetX2, self).__init__() # inchannel, outchannel, upsample=False
        self.Block1 = BasicBlockDown_leaky(inch, inch*4, downsample=False)    # (8)  16  --> (32) 16
        self.Block2 = BasicBlockDown_leaky(inch*4, inch*2, downsample=True)  # (32) 16  --> (16) 16*3
        self.Block3 = BasicBlockDown_leaky(inch*2, inch,  downsample=False)   # (16) 16*3  --> (8)  16*3 
        self.bn = nn.BatchNorm1d(inch)
        self.conv = nn.Conv1d(inch, outch, kernel_size=3, stride=1, padding=1)
        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        # import pdb
        # pdb.set_trace()
        # print(x.shape)
        out = self.Block1(x)
        out = self.Block2(out)
        out = self.Block3(out)
        out = self.bn(out)
        out = self.leakyrelu(self.conv(out)) # in: 8, 4, 4, 4 ; out: 8, 16, 16, 16
        return out
