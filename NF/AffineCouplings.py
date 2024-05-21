import torch
from torch import nn as nn
import torch.nn.functional as F

from NF import thops
from NF.Basic import DenseBlock, FCN, RDN, IdentityLayer, ResNet3D

class AffineCoupling(nn.Module):
    def __init__(self, in_channels, cond_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = 32 # 32 # 64 # opt_get(opt, ['hidden_channels'], 64)
        # self.n_hidden_layers = 6 #6 #4, 6
        self.n_hidden_layers = 6
        self.cond_channels = cond_channels
        f_in_channels = self.in_channels//2 if cond_channels is None else self.in_channels//2 + cond_channels
        f_out_channels = (self.in_channels - self.in_channels//2) * 2
        # print(f_in_channels, f_out_channels)

        # input: [32,4,16,16,16], output: [32,8,16,16,16]
        # nn_module = 'FCN' # opt_get(opt, ['nn_module'], 'FCN')
        nn_module = 'Resnet'
        # nn_module = 'DenseBlock'
        if nn_module == 'DenseBlock':
            self.f = DenseBlock(in_channels=f_in_channels, out_channels=f_out_channels, gc=self.hidden_channels)
        elif nn_module == 'FCN':
            # TODO: add the option to use the 3d version
            # f_in_channels = 4 f_out_channels = (8-4)*2 = 8, hidden_channels = 32, n_hidden_layers = 6
            self.f = FCN(in_channels=f_in_channels, out_channels=f_out_channels, hidden_channels=self.hidden_channels,
                         n_hidden_layers=self.n_hidden_layers)
        elif nn_module == 'Resnet':
            self.f = ResNet3D(in_channels=f_in_channels, out_channels=f_out_channels, hidden_channels=32, num_blocks=[1,1])


    def forward(self, z, u=None, y=None, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(z, u, y, logdet)
        else:
            return self.reverse_flow(z, u, y, logdet)


    def normal_flow(self, z, u=None, y=None, logdet=None):
        # import pdb
        # pdb.set_trace()
        z1, z2 = thops.split_feature(z, "split")
        # if u is not None:
        #     print('z1.shape, z2.shape', z1.shape, z2.shape)
        #     print('u.shape', u.shape)
        h = self.f(z1) if self.cond_channels is None else self.f(thops.cat_feature(z1, u))
        shift, scale = thops.split_feature(h, "cross")
        # adding 1e-4 is crucial for torch.slogdet(), as used in Glow (leads to black rect in experiments).
        # see https://github.com/didriknielsen/survae_flows/issues/5 for discussion.
        # or use `torch.exp(2. * torch.tanh(s / 2.)) as in SurVAE (more unstable in practice).

        # version 1, srflow (use FCN)
        scale = torch.sigmoid(scale + 2.) + 1e-4
        z2 = (z2 + shift) * scale
        logdet += thops.sum(torch.log(scale), dim=[1, 2, 3, 4])

        # version2, survae
        # logscale = 2. * torch.tanh(scale / 2.)
        # z2 = (z2+shift) * torch.exp(logscale) # as in glow, it's shift+scale!
        # logdet += thops.sum(logscale, dim=[1, 2, 3])

        # version3, FrEIA, now have problem with FCN, but densenet is ok. (use FCN2/Denseblock)
        # logscale = 0.5 * 0.636 * torch.atan(scale / 0.5) # clamp it to be between [-0.5,0.5]
        # logscale = 0.318 * torch.atan(2 * scale)
        # z2 = (z2 + shift) * torch.exp(logscale)
        
        z = thops.cat_feature(z1, z2)

        return z, logdet

    def reverse_flow(self, z, u=None, y=None, logdet=None):
        z1, z2 = thops.split_feature(z, "split")

        h = self.f(z1) if self.cond_channels is None else self.f(thops.cat_feature(z1, u))
        shift, scale = thops.split_feature(h, "cross")

        # version1, srflow
        scale = torch.sigmoid(scale + 2.) + 1e-4
        z2 = (z2 / scale) -shift

        # version2, survae
        # logscale = 2. * torch.tanh(scale / 2.)
        # z2 = z2 * torch.exp(-logscale) - shift

        # version3, FrEIA
        # logscale = 0.5 * 0.636 * torch.atan(scale / 0.5)
        # logscale = 0.318 * torch.atan(2 * scale)
        # z2 = z2 * torch.exp(-logscale) - shift
        
        logdet -= thops.sum(torch.log(scale), dim=[1, 2, 3, 4])

        z = thops.cat_feature(z1, z2)

        return z, logdet


'''3 channel conditional on the rest channels, or vice versa. only shift LR.
   used in image rescaling to divide the low-frequencies and the high-frequencies apart from early flow layers.'''
class AffineCoupling3shift(nn.Module):
    def __init__(self, in_channels, cond_channels=None, LRvsothers=True):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = 32 # opt_get(opt, ['hidden_channels'], 64)
        self.n_hidden_layers = 1
        self.kernel_hidden = 1
        self.cond_channels = cond_channels
        self.LRvsothers = LRvsothers
        if LRvsothers:
            f_in_channels = 3 if cond_channels is None else 3 + cond_channels
            f_out_channels = (self.in_channels - 3) * 2
        else:
            f_in_channels = self.in_channels - 3 if cond_channels is None else self.in_channels - 3 + cond_channels
            f_out_channels = 3
        nn_module = 'FCN' # opt_get(opt, ['nn_module'], 'FCN')

        if nn_module == 'DenseBlock':
            self.f = DenseBlock(in_channels=f_in_channels, out_channels=f_out_channels, gc=self.hidden_channels)
        elif nn_module == 'FCN':
            self.f = FCN(in_channels=f_in_channels, out_channels=f_out_channels, hidden_channels=self.hidden_channels,
                          kernel_hidden=self.kernel_hidden, n_hidden_layers=self.n_hidden_layers)


    def forward(self, z, u=None, y=None, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(z, u, y, logdet)
        else:
            return self.reverse_flow(z, u, y, logdet)

    def normal_flow(self, z, u=None, y=None, logdet=None):
        if self.LRvsothers:
            z1, z2 = z[:, :3, ...], z[:, 3:, ...]
            h = self.f(z1) if self.cond_channels is None else self.f(thops.cat_feature(z1, u))
            shift, scale = thops.split_feature(h, "cross")
            # logscale = 0.318 * torch.atan(2 * scale)
            # z2 = (z2 + shift) * torch.exp(logscale)
            scale = torch.sigmoid(scale + 2.) + 1e-4
            z2 = (z2 + shift) * scale
            if logdet is not None:
                # logdet += thops.sum(logscale, dim=[1, 2, 3])
                logdet += thops.sum(torch.log(scale), dim=[1, 2, 3])
        else:
            z2, z1 = z[:, :3, ...], z[:, 3:, ...]
            shift = self.f(z1) if self.cond_channels is None else self.f(thops.cat_feature(z1, u))
            z2 = z2 + shift

        if self.LRvsothers:
            z = thops.cat_feature(z1, z2)
        else:
            z = thops.cat_feature(z2, z1)

        return z, logdet

    def reverse_flow(self, z, u=None, y=None, logdet=None):
        if self.LRvsothers:
            z1, z2 = z[:, :3, ...], z[:, 3:, ...]
            h = self.f(z1) if self.cond_channels is None else self.f(thops.cat_feature(z1, u))
            shift, scale = thops.split_feature(h, "cross")
            # logscale = 0.318 * torch.atan(2 * scale)
            # z2 = z2 * torch.exp(-logscale) - shift
            scale = torch.sigmoid(scale + 2.) + 1e-4
            z2 = z2 / scale - shift
            logdet -= thops.sum(torch.log(scale), dim=[1, 2, 3])
        else:
            z2, z1 = z[:, :3, ...], z[:, 3:, ...]
            shift = self.f(z1)
            z2 = z2 - shift

        if self.LRvsothers:
            z = thops.cat_feature(z1, z2)
        else:
            z = thops.cat_feature(z2, z1)

        return z, logdet


''' srflow's affine injector + original affine coupling, not used in this project'''
class AffineCouplingInjector(nn.Module):
    def __init__(self, in_channels, cond_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = 32 # opt_get(opt, ['hidden_channels'], 64)
        self.n_hidden_layers = 1
        self.kernel_hidden = 1
        self.cond_channels = cond_channels
        f_in_channels = self.in_channels//2 if cond_channels is None else self.in_channels//2 + cond_channels
        f_out_channels = (self.in_channels - self.in_channels//2) * 2
        nn_module = 'FCN' # opt_get(opt, ['nn_module'], 'FCN')
        if nn_module == 'DenseBlock':
            self.f = DenseBlock(in_channels=f_in_channels, out_channels=f_out_channels, gc=self.hidden_channels)
            self.f_injector = DenseBlock(in_channels=cond_channels, out_channels=self.in_channels*2, gc=self.hidden_channels)
        elif nn_module == 'FCN':
            self.f = FCN(in_channels=f_in_channels, out_channels=f_out_channels, hidden_channels=self.hidden_channels,
                          kernel_hidden=self.kernel_hidden, n_hidden_layers=self.n_hidden_layers)
            self.f_injector = FCN(in_channels=cond_channels, out_channels=self.in_channels*2, hidden_channels=self.hidden_channels,
                          kernel_hidden=self.kernel_hidden, n_hidden_layers=self.n_hidden_layers)

    def forward(self, z, u=None, y=None, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(z, u, y, logdet)
        else:
            return self.reverse_flow(z, u, y, logdet)

    def normal_flow(self, z, u=None, y=None, logdet=None):
        # overall-conditional
        h = self.f_injector(u)
        shift, scale = thops.split_feature(h, "cross")
        # logscale = 0.318 * torch.atan(2 * scale) # clamp it to be between [-5,5]
        # z = (z + shift) * torch.exp(logscale)
        # logdet += thops.sum(logscale, dim=[1, 2, 3])
        scale = torch.sigmoid(scale + 2.) + 1e-4
        z = (z + shift) * scale
        logdet += thops.sum(torch.log(scale), dim=[1, 2, 3])

        # self-conditional
        z1, z2 = thops.split_feature(z, "split")
        h = self.f(z1) if self.cond_channels is None else self.f(thops.cat_feature(z1, u))
        shift, scale = thops.split_feature(h, "cross")
        # logscale = 0.318 * torch.atan(2 * scale) # clamp it to be between [-5,5]
        # z2 = (z2 + shift) * torch.exp(logscale)
        # logdet += thops.sum(logscale, dim=[1, 2, 3])
        scale = torch.sigmoid(scale + 2.) + 1e-4
        z2 = (z2 + shift) * scale
        logdet += thops.sum(torch.log(scale), dim=[1, 2, 3])
        
        z = thops.cat_feature(z1, z2)
        return z, logdet

    def reverse_flow(self, z, u=None, y=None, logdet=None):
        # self-conditional
        z1, z2 = thops.split_feature(z, "split")
        h = self.f(z1) if self.cond_channels is None else self.f(thops.cat_feature(z1, u))
        shift, scale = thops.split_feature(h, "cross")
        # logscale = 0.318 * torch.atan(2 * scale)
        # z2 = z2 * torch.exp(-logscale) - shift
        scale = torch.sigmoid(scale + 2.) + 1e-4
        z2 = z2 / scale - shift
        logdet -= thops.sum(torch.log(scale), dim=[1, 2, 3])
        z = thops.cat_feature(z1, z2)

        # overall-conditional
        h = self.f_injector(u)
        shift, scale = thops.split_feature(h, "cross")
        # logscale = 0.318 * torch.atan(2 * scale)
        # z = z * torch.exp(-logscale) - shift
        scale = torch.sigmoid(scale + 2.) + 1e-4
        z = z / scale - shift
        logdet -= thops.sum(torch.log(scale), dim=[1, 2, 3])
        return z, logdet
       
 