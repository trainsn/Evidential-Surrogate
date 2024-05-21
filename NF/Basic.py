import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
import math

from NF.ActNorms import ActNorm2d, ActNorm3d
from NF import thops

from NF import module_util as mutil
from NF.layers_util import BasicBlockEncoder, BasicBlockUp_leaky, BasicBlockDown_leaky

# from models.modules.utils.probs import GaussianDistribution

class Conv2d(nn.Conv2d):
    pad_dict = {
        "same": lambda kernel, stride: [((k - 1) * s + 1) // 2 for k, s in zip(kernel, stride)],
        "valid": lambda kernel, stride: [0 for _ in kernel]
    }

    @staticmethod
    def get_padding(padding, kernel_size, stride):
        # make padding
        if isinstance(padding, str):
            if isinstance(kernel_size, int):
                kernel_size = [kernel_size, kernel_size]
            if isinstance(stride, int):
                stride = [stride, stride]
            padding = padding.lower()
            try:
                padding = Conv2d.pad_dict[padding](kernel_size, stride)
            except KeyError:
                raise ValueError("{} is not supported".format(padding))
        return padding

    def __init__(self, in_channels, out_channels,
                 kernel_size=[3, 3], stride=[1, 1], padding="same", groups=1,
                 do_actnorm=True, weight_std=0.05):
        padding = Conv2d.get_padding(padding, kernel_size, stride)
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, groups=groups, bias=(not do_actnorm))
        # init weight with std
        self.weight.data.normal_(mean=0.0, std=weight_std)
        if not do_actnorm:
            self.bias.data.zero_()
        else:
            self.actnorm = ActNorm2d(out_channels)
        self.do_actnorm = do_actnorm

    def forward(self, input):
        x = super().forward(input)
        if self.do_actnorm:
            x, _ = self.actnorm(x)
        return x

# Zero initialization. We initialize the last convolution of each NN() with zeros, such that each affine
# coupling layer initially performs an identity function; we found that this helps training very deep networks.
class Conv2dZeros(nn.Conv2d):
    def __init__(self, in_channels, out_channels,
                 kernel_size=[3, 3], stride=[1, 1],
                 padding="same", logscale_factor=3):
        padding = Conv2d.get_padding(padding, kernel_size, stride)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)
        # logscale_factor
        self.logscale_factor = logscale_factor
        self.register_parameter("logs_", nn.Parameter(torch.zeros(out_channels, 1, 1)))
        self.register_parameter("bias_", nn.Parameter(torch.zeros(out_channels, 1, 1)))
        # init
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        output = super().forward(input)
        return (output + self.bias_) * torch.exp(self.logs_ * self.logscale_factor)


class Conv3d(nn.Conv3d):
    pad_dict = {
        "same": lambda kernel, stride: [((k - 1) * s + 1) // 2 for k, s in zip(kernel, stride)],
        "valid": lambda kernel, stride: [0 for _ in kernel]
    }

    @staticmethod
    def get_padding(padding, kernel_size, stride):
        # make padding
        if isinstance(padding, str):
            if isinstance(kernel_size, int):
                kernel_size = [kernel_size, kernel_size, kernel_size]
            if isinstance(stride, int):
                stride = [stride, stride, stride]
            padding = padding.lower()
            try:
                padding = Conv2d.pad_dict[padding](kernel_size, stride)
            except KeyError:
                raise ValueError("{} is not supported".format(padding))
        return padding

    def __init__(self, in_channels, out_channels,
                 kernel_size=[3, 3, 3], stride=[1, 1, 1], padding="same", groups=1,
                 do_actnorm=True, weight_std=0.05):
        padding = Conv3d.get_padding(padding, kernel_size, stride)
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, groups=groups, bias=(not do_actnorm))
        # init weight with std
        self.weight.data.normal_(mean=0.0, std=weight_std)
        if not do_actnorm:
            self.bias.data.zero_()
        else:
            self.actnorm = ActNorm3d(out_channels)
        self.do_actnorm = do_actnorm

    def forward(self, input):
        x = super().forward(input)
        if self.do_actnorm:
            x, _ = self.actnorm(x)
        return x


# added
class Conv3dZeros(nn.Conv3d):
    def __init__(self, in_channels, out_channels,
                 kernel_size=[3, 3, 3], stride=[1, 1, 1],
                 padding="same", logscale_factor=3):
        padding = Conv3d.get_padding(padding, kernel_size, stride)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)
        self.logscale_factor = logscale_factor
        self.register_parameter("logs_", nn.Parameter(torch.zeros(out_channels, 1, 1, 1)))
        self.register_parameter("bias_", nn.Parameter(torch.zeros(out_channels, 1, 1, 1)))
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


class LaplaceDiag:
    Log2= float(np.log(2))

    @staticmethod
    def likelihood(mean, logs, x): # logs: log(sigma)
        if mean is None and logs is None:
            return  - (torch.abs(x) +  LaplaceDiag.Log2)
        else:
            return - (logs + (torch.abs(x - mean)) / torch.exp(logs) + LaplaceDiag.Log2)

    @staticmethod
    def logp(mean, logs, x):
        likelihood = LaplaceDiag.likelihood(mean, logs, x)
        return thops.sum(likelihood, dim=[1, 2, 3])


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


# used in SRFlow
class Split2d_conditional(nn.Module):
    def __init__(self, num_channels, logs_eps=0, cond_channels=0, position=None, consume_ratio=0.5):
        super().__init__()

        self.num_channels_consume = int(round(num_channels * consume_ratio))
        self.num_channels_pass = num_channels - self.num_channels_consume

        self.conv = Conv2dZeros(in_channels=self.num_channels_pass + cond_channels,
                                out_channels=self.num_channels_consume * 2)
        self.logs_eps = logs_eps
        self.position = position

    def split2d_prior(self, z, ft):
        if ft is not None:
            z = torch.cat([z, ft], dim=1)
        h = self.conv(z)
        return thops.split_feature(h, "cross")

    def exp_eps(self, logs):
        return torch.exp(logs) + self.logs_eps

    def forward(self, input, logdet=0., reverse=False, eps_std=None, eps=None, ft=None, y_onehot=None):
        if not reverse:
            # self.input = input
            z1, z2 = self.split_ratio(input)
            mean, logs = self.split2d_prior(z1, ft)

            eps = (z2 - mean) / self.exp_eps(logs)

            logdet = logdet + self.get_logdet(logs, mean, z2)

            # print(logs.shape, mean.shape, z2.shape)
            # self.eps = eps
            # print('split, enc eps:', eps)
            return z1, logdet, eps
        else:
            z1 = input
            mean, logs = self.split2d_prior(z1, ft)

            if eps is None:
                #print("WARNING: eps is None, generating eps untested functionality!")
                eps = GaussianDiag.sample_eps(mean.shape, eps_std)

            eps = eps.to(mean.device)
            z2 = mean + self.exp_eps(logs) * eps

            z = thops.cat_feature(z1, z2)
            logdet = logdet - self.get_logdet(logs, mean, z2)

            return z, logdet
            # return z, logdet, eps

    def get_logdet(self, logs, mean, z2):
        logdet_diff = GaussianDiag.logp(mean, logs, z2)
        # print("Split2D: logdet diff", logdet_diff.item())
        return logdet_diff

    def split_ratio(self, input):
        z1, z2 = input[:, :self.num_channels_pass, ...], input[:, self.num_channels_pass:, ...]
        return z1, z2


# DenseBlock for affine coupling (flow)
class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, gc=16, bias=True, init='xavier', for_flow=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv3d(in_channels + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv3d(in_channels + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv3d(in_channels + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv3d(in_channels + 4 * gc, out_channels, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # init as 'xavier', following the practice in https://github.com/VLL-HD/FrEIA/blob/c5fe1af0de8ce9122b5b61924ad75a19b9dc2473/README.rst#useful-tips--engineering-heuristics
        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

        # initialiize input to all zeros to have zero mean and unit variance
        if for_flow:
            mutil.initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5


# ResidualDenseBlock for multi-layer feature extraction
class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=32, gc=16, bias=True, init='xavier'):
        super(ResidualDenseBlock, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv3d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv3d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv3d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv3d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv3d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # init as 'xavier', following the practice in https://github.com/VLL-HD/FrEIA/blob/c5fe1af0de8ce9122b5b61924ad75a19b9dc2473/README.rst#useful-tips--engineering-heuristics
        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x # residual scaling are helpful


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''
    def __init__(self, nf, gc=16):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock(nf, gc)
        self.RDB2 = ResidualDenseBlock(nf, gc)
        self.RDB3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x # residual scaling are helpful


class RRDB_v2(nn.Module):
    '''Residual in Residual Dense Block'''
    def __init__(self, nf, gc=16):
        super(RRDB_v2, self).__init__()
        self.RDB1 = ResidualDenseBlock(nf, gc)
        self.RDB2 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        return out * 0.2 + x # residual scaling are helpful


class RDN(nn.Module):
    '''composed of rrdb blocks'''

    def __init__(self, in_channels, out_channels, nb=3, nf=32, gc=16, init='xavier', for_flow=True):
        super(RDN, self).__init__()

        RRDB_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.conv_first = nn.Conv3d(in_channels, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = mutil.make_layer(RRDB_f, nb)
        self.trunk_conv = nn.Conv3d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv3d(nf, out_channels, 3, 1, 1, bias=True)

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv_first, self.trunk_conv, self.conv_last], 0.1)
        else:
            mutil.initialize_weights([self.conv_first, self.trunk_conv, self.conv_last], 0.1)

        if for_flow:
            mutil.initialize_weights(self.conv_last, 0)

    def forward(self, x):
        x = self.conv_first(x)
        x = self.trunk_conv(self.RRDB_trunk(x)) + x
        return self.conv_last(x)


class FCN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, n_hidden_layers=1, init='xavier', for_flow=True):
        super(FCN, self).__init__()
        
        self.conv1 = Conv3d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.n_hidden_layers = n_hidden_layers
        hidden_layers = []
        for _ in range(n_hidden_layers):
            hidden_layers += [Conv3d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)]
        self.conv2 = nn.Sequential(*hidden_layers)
        # self.conv2 = Conv3d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = Conv3dZeros(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=False)

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3], 0.1)

        if for_flow:
            mutil.initialize_weights(self.conv3, 0)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        for i in range(self.n_hidden_layers):
            x = self.relu(self.conv2[i](x))
        # x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
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


class ResNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_blocks):
        super(ResNet3D, self).__init__()
        self.in_channels = in_channels

        # Initial convolution
        self.conv1 = nn.Conv3d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(hidden_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # Residual blocks
        self.layer1 = self._make_layer(hidden_channels, hidden_channels*2, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(hidden_channels*2,hidden_channels*4 , num_blocks[1], stride=1)

        # Classifier
        self.conv2 = nn.Conv3d(hidden_channels*4, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock3D(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock3D(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x


class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, logdet=None, reverse=False):
        if not reverse:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(1/16.)

            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out, logdet
        else:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(16.)

            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.channel_in), logdet


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
class IdentityLayer(nn.Module):
    def forward(self, x, **kwargs):
        return x

class SoftClampling(nn.Module):
    """
    From https://github.com/VLL-HD/FrEIA/blob/a5069018382d3bef25a6f7fa5a51c810b9f66dc5/FrEIA/modules/coupling_layers.py#L88
    """

    def __init__(self, is_enable=True, clamp=1.9):
        super(SoftClampling, self).__init__()

        self.is_enable = is_enable
        if is_enable:
            self.clamp = 2.0 * clamp / math.pi
        else:
            self.clamp = None

    def forward(self, scale):
        if self.is_enable:
            return self.clamp * torch.atan(scale)
        else:
            return scale

# -----------------------------------------------------------------------------------------
# upscale 3D feature map
class UpscaleNetX4(nn.Module):
    def __init__(self, inch=8, outch=8):
        super(UpscaleNetX4, self).__init__() # inchannel, outchannel, upsample=False
        self.Block1 = BasicBlockUp_leaky(inch, inch*4, upsample=False)   # (8)  4  --> (32) 4
        self.Block2 = BasicBlockUp_leaky(inch*4, inch*2, upsample=True)  # (32) 4  --> (16) 8
        self.Block3 = BasicBlockUp_leaky(inch*2, inch,  upsample=True)   # (16) 8  --> (8)  16
        self.bn = nn.BatchNorm3d(inch)
        self.conv = nn.Conv3d(inch, outch, kernel_size=3, stride=1, padding=1)
        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = self.Block1(x)
        out = self.Block2(out)
        out = self.Block3(out)
        out = self.bn(out)
        out = self.leakyrelu(self.conv(out)) # in: 8, 4, 4, 4 ; out: 8, 16, 16, 16
        return out


# upscale 3D feature map
class UpscaleNetX1(nn.Module):
    def __init__(self, inch=8, outch=8):
        super(UpscaleNetX1, self).__init__() # inchannel, outchannel, upsample=False
        self.Block1 = BasicBlockUp_leaky(inch, inch*4, upsample=False)    # (8)  16  --> (32) 16
        self.Block2 = BasicBlockUp_leaky(inch*4, inch*2, upsample=False)  # (32) 16  --> (16) 16
        self.Block3 = BasicBlockUp_leaky(inch*2, inch,  upsample=False)   # (16) 16  --> (8)  16
        self.bn = nn.BatchNorm3d(inch)
        self.conv = nn.Conv3d(inch, outch, kernel_size=3, stride=1, padding=1)
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
        self.bn = nn.BatchNorm3d(inch)
        self.conv = nn.Conv3d(inch, outch, kernel_size=3, stride=1, padding=1)
        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = self.Block1(x)
        out = self.Block2(out)
        out = self.Block3(out)
        out = self.bn(out)
        out = self.leakyrelu(self.conv(out)) # in: 8, 4, 4, 4 ; out: 8, 16, 16, 16
        return out


class UpscaleNetX3(nn.Module):
    def __init__(self, inch=8, outch=8):
        super(UpscaleNetX3, self).__init__() # inchannel, outchannel, upsample=False
        self.Block1 = BasicBlockUp_leaky(inch, inch*4, upsample=False)    # (8)  16  --> (32) 16
        self.Block2 = BasicBlockUp_leaky(inch*4, inch*2, sf=3, upsample=True)  # (32) 16  --> (16) 16*3
        self.Block3 = BasicBlockUp_leaky(inch*2, inch,  upsample=False)   # (16) 16*3  --> (8)  16*3 
        self.bn = nn.BatchNorm3d(inch)
        self.conv = nn.Conv3d(inch, outch, kernel_size=3, stride=1, padding=1)
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
        self.bn = nn.BatchNorm3d(inch)
        self.conv = nn.Conv3d(inch, outch, kernel_size=3, stride=1, padding=1)
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
