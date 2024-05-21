import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LowerBoundFunction(torch.autograd.Function):
    """Autograd function for the `LowerBound` operator."""

    @staticmethod
    def forward(ctx, input_, bound):
        ctx.save_for_backward(input_, bound)
        return torch.max(input_, bound)

    @staticmethod
    def backward(ctx, grad_output):
        input_, bound = ctx.saved_tensors
        pass_through_if = (input_ >= bound) | (grad_output < 0)
        return pass_through_if.type(grad_output.dtype) * grad_output, None


class LowerBound(nn.Module):
    """Lower bound operator, computes `torch.max(x, bound)` with a custom
    gradient.

    The derivative is replaced by the identity function when `x` is moved
    towards the `bound`, otherwise the gradient is kept to zero.
    """

    def __init__(self, bound):
        super().__init__()
        self.register_buffer("bound", torch.Tensor([float(bound)]))

    @torch.jit.unused
    def lower_bound(self, x):
        return LowerBoundFunction.apply(x, self.bound)

    def forward(self, x):
        if torch.jit.is_scripting():
            return torch.max(x, self.bound)
        return self.lower_bound(x)


class NonNegativeParametrizer(nn.Module):
    """
    Non negative reparametrization.

    Used for stability during training.
    """

    def __init__(self, minimum=0, reparam_offset=2 ** -18):
        super().__init__()

        self.minimum = float(minimum)
        self.reparam_offset = float(reparam_offset)

        pedestal = self.reparam_offset ** 2
        self.register_buffer("pedestal", torch.Tensor([pedestal]))
        bound = (self.minimum + self.reparam_offset ** 2) ** 0.5
        self.lower_bound = LowerBound(bound)

    def init(self, x):
        return torch.sqrt(torch.max(x + self.pedestal, self.pedestal))

    def forward(self, x):
        out = self.lower_bound(x)
        out = out ** 2 - self.pedestal
        return out


class GDN(nn.Module):
    """Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
    """
    def __init__(self, channels, inverse=False, beta_min=1e-6, gamma_init=0.1):  # beta_min=1e-6
        super().__init__()

        beta_min = float(beta_min)
        gamma_init = float(gamma_init)
        self.inverse = bool(inverse)

        self.beta_reparam = NonNegativeParametrizer(minimum=beta_min)
        beta = torch.ones(channels)
        beta = self.beta_reparam.init(beta)
        self.beta = nn.Parameter(beta)

        self.gamma_reparam = NonNegativeParametrizer()
        gamma = gamma_init * torch.eye(channels)
        gamma = self.gamma_reparam.init(gamma)
        self.gamma = nn.Parameter(gamma)

    def forward(self, x):
        _, C, _, _, _ = x.size()

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1, 1, 1)
        norm = F.conv3d(x ** 2, gamma, beta)

        if self.inverse:
            norm = torch.sqrt(norm)  # may cause nan.
            # norm = torch.sqrt(torch.relu(norm))
        else:
            norm = torch.rsqrt(norm)

        out = x * norm

        return out


class GDN3d(GDN):
    def forward(self, x):
        _, C, _, _, _ = x.size()

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1, 1, 1)
        norm = F.conv3d(torch.abs(x), gamma, beta)

        if not self.inverse:
            norm = 1.0 / (norm + 1e-4)
        out = x * norm
        return out


# used in ResNet
class BasicBlockEncoder(nn.Module):
    def __init__(self, inchannel, outchannel, upsample=False):
        super(BasicBlockEncoder, self).__init__()
        self.upsample = upsample
        self.conv_res = None
        if self.upsample or inchannel != outchannel:
            self.conv_res = nn.Conv3d(inchannel, outchannel, kernel_size=1, stride=1, padding=0, bias=False)

        self.bn1 = nn.BatchNorm3d(inchannel)
        self.conv1 = nn.Conv3d(inchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm3d(outchannel)
        self.conv2 = nn.Conv3d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x

        if self.upsample:
            residual = F.interpolate(residual, scale_factor=2)
        if self.conv_res is not None:
            residual = self.conv_res(residual)

        out = self.bn1(x)
        out = self.relu(out)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + residual


# used in ResNet
class BasicBlockUp_leaky(nn.Module):
    def __init__(self, inchannel, outchannel, sf=2, upsample=False):
        super(BasicBlockUp_leaky, self).__init__()
        self.upsample = upsample
        self.conv_res = None
        if self.upsample or inchannel != outchannel:
            self.conv_res = nn.Conv3d(inchannel, outchannel, kernel_size=1, stride=1, padding=0, bias=False)

        self.bn1 = nn.BatchNorm3d(inchannel)
        self.conv1 = nn.Conv3d(inchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm3d(outchannel)
        self.conv2 = nn.Conv3d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)

        self.upsample_fun = nn.Upsample(scale_factor=sf, mode='trilinear', align_corners=True)   

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
            self.conv_res = nn.Conv3d(inchannel, outchannel, kernel_size=1, stride=1, padding=0, bias=False)

        if self.downsample:
            self.downconv1 = nn.Conv3d(inchannel, inchannel, 4, stride=2, padding=1)
            self.downconv2 = nn.Conv3d(inchannel, inchannel, 4, stride=2, padding=1)
        
        self.bn1 = nn.BatchNorm3d(inchannel)
        self.conv1 = nn.Conv3d(inchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm3d(outchannel)
        self.conv2 = nn.Conv3d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False)
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
            torch.nn.Conv3d(inchannel, out_channels=outchannel, kernel_size=kernel_size, stride=stride, padding=1),
            torch.nn.BatchNorm3d(outchannel),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            torch.nn.Conv3d(outchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=1),
            torch.nn.BatchNorm3d(outchannel),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        return x + self.residual_block(x)


def weights_init_kaiming(m):
	classname = m.__class__.__name__
	if classname.find("Conv")!=-1:
		init.kaiming_uniform_(m.weight.data)
	elif classname.find("Linear")!=-1:
		init.kaiming_uniform_(m.weight.data)
	elif classname.find("BatchNorm")!=-1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)

def weights_init_normal(m):
	classname = m.__class__.__name__
	if classname.find("Conv")!=-1:
		init.normal_(m.weight.data, 1.0, 0.02)
	elif classname.find("Linear")!=-1:
		init.normal_(m.weight.data, 1.0, 0.02)
	elif classname.find("BatchNorm")!=-1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)

