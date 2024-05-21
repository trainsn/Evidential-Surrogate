import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F

from NF import Basic
from NF.FlowStep import FlowStep
from NF.ConditionalFlow import ConditionalFlow
from NF.layers_util import BasicBlockUp_leaky, BasicBlockDown_leaky, ResidualBlock
from NF import thops

class FlowNet(nn.Module):
    def __init__(self, C=1, K=8):
        super().__init__()
        self.C = C
        self.K = K
        
        flow_actNorm='actNorm3d'
        flow_permutation = 'invconv' 
        flow_coupling = 'Affine' 
        print(flow_actNorm, flow_permutation, flow_coupling)
        
        cond_channels = None 
        # flow_permutation: none # bettter than invconv
        # flow_coupling: Affine3shift # better than affine
        
        # construct flow
        self.layers = nn.ModuleList()
        # self.output_shapes = []
        # print('after_splitOff_flowStep', after_splitOff_flowStep)
        
        # coupling layers
        # 1. Squeeze
        self.layers.append(Basic.SqueezeLayer(2)) # may need a better way for squeezing
        self.C = self.C * (2**3) #, D // 2, H // 2, W // 2
        
        # 2. main FlowSteps (unconditional flow)
        for k in range(self.K):
            self.layers.append(FlowStep(in_channels=self.C, cond_channels=cond_channels,
                                            flow_actNorm=flow_actNorm,
                                            flow_permutation=flow_permutation,
                                            flow_coupling=flow_coupling))
        # self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)

    def forward(self, d_data, shape=None, eps_std=None, reverse=False, training=True):
        B = d_data.shape[0]
        device = d_data.device
        if shape is None:
            shape = d_data.shape
        logdet = torch.zeros((B,), device=device)
        if not reverse:
            return self.normal_flow(d_data, logdet=logdet, training=training)
        else:
            return self.reverse_flow(z=None, shape=shape, device=device, logdet=logdet, eps_std=eps_std, training=training)

    def normal_flow(self, z, logdet=None, training=True):
        for layer in self.layers:
            if isinstance(layer, FlowStep):
                z, logdet = layer(z, u=None, logdet=logdet, reverse=False)
            elif isinstance(layer, Basic.SqueezeLayer):
                z, logdet = layer(z, logdet=logdet, reverse=False)
        
        logdet += Basic.GaussianDiag.logp(mean=torch.zeros_like(z), logs=torch.ones_like(z), x=z)
        return z, logdet

    def reverse_flow(self, z, shape, device='cpu', logdet=None, eps_std=1, training=True):
        if z is None:
            z = Basic.GaussianDiag.sample(mean=torch.zeros(shape).to(device), logs=torch.ones(shape).to(device), eps_std=eps_std)
            # z = self.prior.sample(sample_shape=shape).to(device)
        logdet -= Basic.GaussianDiag.logp(torch.zeros(shape).to(device), torch.ones(shape).to(device), z)
        for layer in reversed(self.layers): 
            if isinstance(layer, FlowStep):
                z, logdet = layer(z, u=None, logdet=logdet, reverse=True)
            elif isinstance(layer, Basic.SqueezeLayer):
                z, logdet = layer(z, logdet=logdet, reverse=True)
        return z, logdet
        
    def sample(self, z, shape, device='cpu', logdet=0, eps_std=0.9, training=True):
        return self.reverse_flow(z, shape, device=device, logdet=logdet, eps_std=eps_std)



class ParamFlowNet(nn.Module):
    def __init__(self, C=1, K=8, dataname='nyx'):
        super().__init__()
        self.C = C
        self.K = K
        self.dataname = dataname
        
        flow_actNorm='actNorm3d'
        flow_permutation = 'invconv' 
        flow_coupling = 'Affine' 
        print(flow_actNorm, flow_permutation, flow_coupling)
        
        cond_channels = None 
        # flow_permutation: none # bettter than invconv
        # flow_coupling: Affine3shift # better than affine
        
        # construct flow
        self.layers = nn.ModuleList()
        # self.output_shapes = []
        # print('after_splitOff_flowStep', after_splitOff_flowStep)
        
        # coupling layers
        # 1. Squeeze
        self.layers.append(Basic.SqueezeLayer(2)) 
        self.C = self.C * (2**3) # 8C, D // 2, H // 2, W // 2
        
        # 2. main FlowSteps (unconditional flow)
        for k in range(self.K):
            self.layers.append(FlowStep(in_channels=self.C, cond_channels=cond_channels,
                                    flow_actNorm=flow_actNorm,
                                    flow_permutation=flow_permutation,
                                    flow_coupling=flow_coupling))
        # self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)

        # conditional Gaussian
        if self.dataname == 'nyx':
            self.param_dim = 3
            self.mlp_first = nn.Sequential(nn.Linear(self.param_dim, self.C * 4 * 4 * 4, bias=False)) # C = 8c
            self.resnet_feat1 = Basic.UpscaleNetX4(inch=self.C, outch=self.C) # C,4,4,4 --> C,16,16,16
        
        elif self.dataname == 'mpas':
            self.param_dim = 4
            # v4
            self.mlp_first = nn.Sequential(nn.Linear(self.param_dim, self.C * 2 * 4 * 8, bias=False)) 
            self.resnet_feat1 = nn.Sequential(
                Basic.UpscaleNetX1(inch=self.C, outch=self.C),
                Basic.UpscaleNetX3(inch=self.C, outch=self.C)  # C,2,4,8 --> C,6,12,24
            )
        
        self.resnet_feat2 = Basic.UpscaleNetX1(inch=self.C, outch=self.C)
        # self.resnet_feat2 = nn.Conv3d(self.C, self.C*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.f = Basic.Conv3dZeros(self.C, self.C*2)
        
    def get_conditional_feature(self, u):
        # import pdb
        # pdb.set_trace()
        u_feature_first = self.mlp_first(u)
        if self.dataname == 'nyx':
            u_feature_first = u_feature_first.view(u_feature_first.size(0), self.C, 4, 4, 4)
        elif self.dataname == 'mpas':
            u_feature_first = u_feature_first.view(u_feature_first.size(0), self.C, 2, 4, 8)
        u_feature = self.resnet_feat2(self.resnet_feat1(u_feature_first))
        return u_feature
    
    def forward(self, d_data, d_param=None, shape=None, eps_std=None, reverse=False, training=True):
        if d_param is not None:
            B = d_param.shape[0]
            device = d_param.device
        else:
            B = d_data.shape[0]
            device = d_data.device

        if shape is None:
            shape = d_data.reshape(d_data.shape[0], -1).shape 
        
        logdet = torch.zeros((B,), device=device)
        if not reverse:
            return self.normal_flow(d_data, d_param=None, logdet=logdet, training=training)
        else:
            return self.reverse_flow(dummy_z=None, param_z=d_param, shape=shape, device=device, logdet=logdet, eps_std=eps_std, training=training)

    def normal_flow(self, z, d_param=None, logdet=None, training=True):
        for layer in self.layers:
            if isinstance(layer, FlowStep):
                z, logdet = layer(z, u=None, logdet=logdet, reverse=False)
            elif isinstance(layer, Basic.SqueezeLayer):
                z, logdet = layer(z, logdet=logdet, reverse=False)
        
        # nyx: batch, 8, 16, 16, 16;    
        # mpas: batch, 8, 12, 12, 24;  batch, 8, 6, 12, 24
        z = z.reshape(z.shape[0], -1) 
        param_z = z[:, :self.param_dim]
        dummy_z = z[:, self.param_dim:]
        # logdet += Basic.GaussianDiag.logp(mean=torch.zeros_like(dummy_z), logs=torch.ones_like(dummy_z), x=dummy_z)
        
        # import pdb
        # pdb.set_trace()
        conditional_feature = self.get_conditional_feature(param_z)
        h = self.f(conditional_feature)
        mean, logs = thops.split_feature(h, "cross")
        mean = mean.reshape(mean.shape[0], -1)[:, self.param_dim:]
        logs = logs.reshape(logs.shape[0], -1)[:, self.param_dim:]
        
        logdet += Basic.GaussianDiag.logp(mean, logs, dummy_z)
        # conditional_feature = self.get_conditional_feature(d_param)
        return dummy_z, param_z, logdet

    def reverse_flow(self, dummy_z, param_z, shape, device='cpu', logdet=None, eps_std=1, recoverZ=False):
        # shape: [batch, dim]
        conditional_feature = self.get_conditional_feature(param_z)
        h = self.f(conditional_feature)
        mean, logs = thops.split_feature(h, "cross")
        mean = mean.reshape(mean.shape[0], -1)[:, self.param_dim:]
        logs = logs.reshape(logs.shape[0], -1)[:, self.param_dim:]
        
        if dummy_z is None:
            dummy_z = Basic.GaussianDiag.sample(mean=mean, logs=logs, eps_std=eps_std)
        logdet -= Basic.GaussianDiag.logp(mean, logs, dummy_z)
        
        z = torch.cat(([param_z, dummy_z]), dim=1)
        z = z.reshape(shape)

        if recoverZ:
            for layer in reversed(self.layers): 
                if isinstance(layer, FlowStep):
                    z, logdet = layer(z, u=None, logdet=logdet, reverse=True)
                elif isinstance(layer, Basic.SqueezeLayer):
                    z, logdet = layer(z, logdet=logdet, reverse=True)
        return z, dummy_z, logdet
    
    def sample(self, dummy_z, param_z, shape=[1,1,1], device='cpu', logdet=0, eps_std=0.9, recoverZ=False):
        return self.reverse_flow(dummy_z, param_z, shape, device=device, logdet=logdet, eps_std=eps_std, recoverZ=recoverZ)

    def sampleZ(self, dummy_z, param_z, eps_std=0.9):
        conditional_feature = self.get_conditional_feature(param_z)
        h = self.f(conditional_feature)
        mean, logs = thops.split_feature(h, "cross")
        mean = mean.reshape(mean.shape[0], -1)[:, self.param_dim:]
        logs = logs.reshape(logs.shape[0], -1)[:, self.param_dim:]
        
        if dummy_z is None:
            dummy_z = Basic.GaussianDiag.sample(mean=mean, logs=logs, eps_std=eps_std)
        return dummy_z



class ParamFlowNetCond(nn.Module):
    def __init__(self, C=1, K=8, K_cond=5, dataname='nyx'):
        super().__init__()
        self.C = C
        self.K = K
        self.K_cond = K_cond
        self.dataname = dataname
        
        flow_actNorm='actNorm3d'
        flow_permutation = 'invconv' 
        flow_coupling = 'Affine' 
        print(flow_actNorm, flow_permutation, flow_coupling)
        
        cond_channels = None 
        
        self.layers = nn.ModuleList()
        
        # coupling layers
        # 1. Squeeze
        self.layers.append(Basic.SqueezeLayer(2)) 
        self.C = self.C * (2**3) # 8C, D // 2, H // 2, W // 2
        
        # 2. main FlowSteps (unconditional flow)
        for k in range(self.K):
            self.layers.append(FlowStep(in_channels=self.C, cond_channels=cond_channels,
                                        flow_actNorm=flow_actNorm,
                                        flow_permutation=flow_permutation,
                                        flow_coupling=flow_coupling))
        self.layers.append(Basic.Split(num_channels_split=self.C//2)) # WE ACTUALLY DO NOT USE THIS SPLIT LAYER
        
        self.condFlow = ConditionalFlow(dataname,
                                        num_channels=self.C,
                                        n_flow_step=self.K_cond,
                                        flow_actNorm = flow_actNorm,
                                        flow_permutation=flow_permutation,
                                        flow_coupling=flow_coupling)
        self.layers.append(self.condFlow)
        # self.C = self.C // 2 # if level < self.L-1 else 3

        # conditional Gaussian
        if self.dataname == 'nyx':
            self.param_dim = 3
        
        elif self.dataname == 'mpas':
            self.param_dim = 4
    
    def forward(self, d_data, d_param=None, shape=None, eps_std=None, reverse=False, end_p=False):
        if d_param is not None:
            B = d_param.shape[0]
            device = d_param.device
        else:
            B = d_data.shape[0]
            device = d_data.device

        if shape is None:
            shape = d_data.reshape(d_data.shape[0], -1).shape 
        
        logdet = torch.zeros((B,), device=device)
        if not reverse:
            return self.normal_flow(d_data, d_param=d_param, logdet=logdet, end_p=end_p)
        else:
            return self.reverse_flow(z=None, d_param=d_param, shape=shape, device=device, logdet=logdet, eps_std=eps_std)

    def normal_flow(self, z, d_param=None, logdet=None, end_p=False):
        for layer in self.layers:
            if isinstance(layer, FlowStep):
                z, logdet = layer(z, u=None, logdet=logdet, reverse=False)
            elif isinstance(layer, Basic.SqueezeLayer):
                z, logdet = layer(z, logdet=logdet, reverse=False)
            elif isinstance(layer, Basic.Split):
                shape = z.shape
                z = z.reshape(z.shape[0], -1) 
                # print(z[0][:20])
                param_z = z[:, :self.param_dim]
                z = z.reshape(shape)
                if end_p: # for parameter prediction
                    return z, param_z, logdet
            elif isinstance(layer, ConditionalFlow):
                z, logdet, var_reduction_loss, mean_loss = layer(z, d_param, logdet=logdet, reverse=False) 
        
        return z, param_z, logdet, var_reduction_loss, mean_loss

    def noisy_z0(self, z0, d_param):
        z = Basic.GaussianDiag.sample(z0, torch.ones_like(z0), eps_std=0.01)
        logdet = torch.zeros((len(z0),), device=z0.device)
        
        for layer in reversed(self.layers): 
            if isinstance(layer, FlowStep):
                z, logdet = layer(z, u=None, logdet=logdet, reverse=True)
            elif isinstance(layer, Basic.SqueezeLayer):
                z, logdet = layer(z, logdet=logdet, reverse=True)
            elif isinstance(layer, Basic.Split):
                shape = z.shape
                z = z.reshape(z.shape[0], -1) 
                param_z = z[:, :self.param_dim]
                z = z.reshape(shape)
            elif isinstance(layer, ConditionalFlow):
                # important!!!! from noisy z0 to m, from m to zk
                z = layer.z0_to_m(z, d_param)
        return z, param_z

    def reverse_flow(self, z, d_param, shape, device='cpu', logdet=None, eps_std=1, recoverZ=True):
        for layer in reversed(self.layers): 
            if isinstance(layer, FlowStep):
                z, logdet = layer(z, u=None, logdet=logdet, reverse=True)
            elif isinstance(layer, Basic.SqueezeLayer):
                z, logdet = layer(z, logdet=logdet, reverse=True)
            elif isinstance(layer, Basic.Split):
                shape = z.shape
                z = z.reshape(z.shape[0], -1) 
                param_z = z[:, :self.param_dim]
                # z[:, :self.param_dim] = torch.zeros(([param_z.shape])).to('cuda')
                # z[:, :1000] = torch.zeros(([1, 1000])).to('cuda')
                # print(z[0][:20])
                z = z.reshape(shape)
                if not recoverZ: # for rev zc loss
                    return param_z
            elif isinstance(layer, ConditionalFlow):
                # important!!!! add recoverZ=False here to avoid recover z from Gaussian to VAE latent space
                z, logdet = layer(z, d_param, eps_std=eps_std, logdet=logdet, reverse=True, recoverZ=recoverZ)
                # if not recoverZ:
                #     return z, None, logdet
                    # important!!!! return early
        return z, param_z, logdet
    
    def sample(self, dummy_z, d_param, shape=[1,1,1], device='cpu', logdet=0, eps_std=0.9, recoverZ=True):
        return self.reverse_flow(dummy_z, d_param, shape, device=device, logdet=logdet, eps_std=eps_std, recoverZ=recoverZ)



######################################### Related to AE ###############################################################################
class ResNetEncoder(nn.Module):
    def __init__(self, C, n_ResBlock=8, n_levels=3, z_dim=1, bUseMultiResSkips=False):
        super(ResNetEncoder, self).__init__()
        
        self.max_filters = 2 ** (n_levels+2) # 2 ** (n_levels+1)
        self.n_levels = n_levels
        self.bUseMultiResSkips = bUseMultiResSkips

        self.conv_list = nn.ModuleList()
        self.res_blk_list = nn.ModuleList()
        if self.bUseMultiResSkips:
            self.multi_res_skip_list = nn.ModuleList()

        
        self.input_conv = nn.Sequential(
            nn.Conv3d(C, 4, kernel_size=3, stride=1, padding=1), # 4 ; 2
            nn.BatchNorm3d(4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        for i in range(n_levels):
            n_filters_1 = 2 ** (i + 2) #3 #1 #2
            n_filters_2 = 2 ** (i + 3) #4 #2 #3
            ks = 2 ** (n_levels - i)
            # print('en', i, n_filters_1, n_filters_2, ks)

            self.res_blk_list.append(
                nn.Sequential(*[ResidualBlock(n_filters_1, n_filters_1)
                                      for _ in range(n_ResBlock)])
            )
            self.conv_list.append(
                nn.Sequential(
                    nn.Conv3d(n_filters_1, n_filters_2, kernel_size=2, stride=2, padding=0),
                    nn.BatchNorm3d(n_filters_2),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                )
            )
            if bUseMultiResSkips:
                self.multi_res_skip_list.append(
                    nn.Sequential(
                        nn.Conv3d(n_filters_1, self.max_filters, kernel_size=ks, stride=ks, padding=0),
                        nn.BatchNorm3d(self.max_filters),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )
        
        self.output_conv = nn.Conv3d(self.max_filters, z_dim, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        x = self.input_conv(x)
        skips = []
        for i in range(self.n_levels):
            x = self.res_blk_list[i](x)
            if self.bUseMultiResSkips:
                skips.append(self.multi_res_skip_list[i](x))
            x = self.conv_list[i](x)
        
        if self.bUseMultiResSkips:
            x = sum([x] + skips)
        
        x = self.output_conv(x)
        return x


class ResNetDecoder(nn.Module):
    def __init__(self, C, n_ResBlock=8, n_levels=3, z_dim=1, bUseMultiResSkips=False):
        super(ResNetDecoder, self).__init__()
        
        self.max_filters = 2 ** (n_levels+1)
        self.n_levels = n_levels
        self.bUseMultiResSkips = bUseMultiResSkips

        self.conv_list = nn.ModuleList()
        self.res_blk_list = nn.ModuleList()
        if self.bUseMultiResSkips:
            self.multi_res_skip_list = nn.ModuleList()

        self.input_conv = nn.Sequential(
            nn.Conv3d(z_dim, self.max_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(self.max_filters),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        for i in range(n_levels):
            n_filters_0 = 2 ** (self.n_levels - i + 1) #3
            n_filters_1 = 2 ** (self.n_levels - i + 0) #2
            ks = 2 ** (i + 1)
            # print('de', i, n_filters_0, n_filters_1, ks)

            self.res_blk_list.append(
                nn.Sequential(*[ResidualBlock(n_filters_1, n_filters_1)
                                      for _ in range(n_ResBlock)])
            )
            self.conv_list.append(
                nn.Sequential(
                    nn.ConvTranspose3d(n_filters_0, n_filters_1, kernel_size=2, stride=2, padding=0),
                    nn.BatchNorm3d(n_filters_1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                )
            )
            if bUseMultiResSkips:
                self.multi_res_skip_list.append(
                    nn.Sequential(
                        nn.ConvTranspose3d(self.max_filters, n_filters_1, kernel_size=ks, stride=ks, padding=0),
                        nn.BatchNorm3d(n_filters_1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )
        
        self.output_conv = nn.Conv3d(n_filters_1, C, kernel_size=3, stride=1, padding=1)
    
    def forward(self, z):
        z = z_top = self.input_conv(z)

        for i in range(self.n_levels):
            z = self.conv_list[i](z)
            z = self.res_blk_list[i](z)
            if self.bUseMultiResSkips:
                z += self.multi_res_skip_list[i](z_top)
        z = self.output_conv(z)
        return z


class ResNetDecoderUpsample(nn.Module):
    def __init__(self, C, n_ResBlock=8, n_levels=3, z_dim=1, bUseMultiResSkips=False):
        super(ResNetDecoderUpsample, self).__init__()
        
        self.max_filters = 2 ** (n_levels+2) # 2 ** (n_levels+2)
        self.n_levels = n_levels
        self.bUseMultiResSkips = bUseMultiResSkips

        self.conv_list = nn.ModuleList()
        self.res_blk_list = nn.ModuleList()
        if self.bUseMultiResSkips:
            self.multi_res_skip_list = nn.ModuleList()

        self.input_conv = nn.Sequential(
            nn.Conv3d(z_dim, self.max_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(self.max_filters),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        for i in range(n_levels):
            n_filters_0 = 2 ** (self.n_levels - i + 2) #3 #1 #2
            n_filters_1 = 2 ** (self.n_levels - i + 1) #2 #0 #1
            ks = 2 ** (i + 1)
            
            self.res_blk_list.append(
                nn.Sequential(*[ResidualBlock(n_filters_1, n_filters_1)
                                      for _ in range(n_ResBlock)])
            )
            self.conv_list.append(
                nn.Sequential(
                    # nn.ConvTranspose3d(n_filters_0, n_filters_1, kernel_size=2, stride=2, padding=0),
                    nn.Upsample(scale_factor=(2, 2, 2), mode="trilinear"),
                    nn.Conv3d(n_filters_0, n_filters_1, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm3d(n_filters_1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                )
            )
            if bUseMultiResSkips:
                self.multi_res_skip_list.append(                    
                    nn.Sequential(                    
                        # nn.ConvTranspose3d(self.max_filters, n_filters_1, kernel_size=ks, stride=ks, padding=0),
                        nn.Upsample(scale_factor=(ks, ks, ks), mode="trilinear"),
                        nn.Conv3d(self.max_filters, n_filters_1, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm3d(n_filters_1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )

        self.output_conv = nn.Conv3d(n_filters_1, C, kernel_size=3, stride=1, padding=1)
    
    def forward(self, z):
        z = z_top = self.input_conv(z)

        for i in range(self.n_levels):
            z = self.conv_list[i](z)
            z = self.res_blk_list[i](z)
            if self.bUseMultiResSkips:
                z += self.multi_res_skip_list[i](z_top)
        z = self.output_conv(z)
        return z

# change "deconvolution" to nearest-neighbor upsampling followed by regular convolution
class ResNetAE(nn.Module):
    def __init__(self, C, n_ResBlock=8, n_levels=3, z_dim=1, bUseMultiResSkips=False, upsample=False):
        super(ResNetAE, self).__init__()
        self.z_dim = z_dim # channel size
        # self.d_latent_dim = input_shape[0] // (2 ** n_levels)
        self.encoder = ResNetEncoder(C, n_ResBlock, n_levels, z_dim, bUseMultiResSkips=bUseMultiResSkips)
        
        if upsample:
            self.decoder = ResNetDecoderUpsample(C, n_ResBlock, n_levels, z_dim, bUseMultiResSkips=bUseMultiResSkips)
        else:
            self.decoder = ResNetDecoder(C, n_ResBlock, n_levels, z_dim, bUseMultiResSkips=bUseMultiResSkips)
        
        # self.fc1 = torch.nn.Linear(self.z_dim * self.img_latent_dim * self.img_latent_dim, bottleneck_dim)
        # self.fc2 = torch.nn.Linear(bottleneck_dim, self.z_dim * self.img_latent_dim * self.img_latent_dim)
    
    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

        

class ResAE_NF(nn.Module):
    def __init__(self, C, K, hc=4, containNF=True):
        super(ResAE_NF, self).__init__()
        self.containNF = containNF
        
        self.encoder = nn.Sequential(
            BasicBlockDown_leaky(C, hc, downsample=True),
            BasicBlockDown_leaky(hc, hc*2, downsample=True),
            BasicBlockDown_leaky(hc*2, hc*4, downsample=True),
            BasicBlockDown_leaky(hc*4, hc*2, downsample=False),
            BasicBlockDown_leaky(hc*2, C, downsample=False)
        )
        
        # upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)    
        
        # self.decoder = nn.Sequential(
        #     BasicBlockUp_leaky(C, hc*4, upsample=True),
        #     BasicBlockUp_leaky(hc*4, hc*2, upsample=True),
        #     BasicBlockUp_leaky(hc*2, hc, upsample=True),
        #     BasicBlockUp_leaky(hc, C, upsample=False),
        # )

        self.decoder = nn.Sequential(
            BasicBlockUp_leaky(C, hc*2, upsample=False),
            BasicBlockUp_leaky(hc*2, hc*4, upsample=True),
            BasicBlockUp_leaky(hc*4, hc*2, upsample=True),
            BasicBlockUp_leaky(hc*2, hc, upsample=True),
            BasicBlockUp_leaky(hc, C, upsample=False),
        )
        if self.containNF:
            self.flow = ParamFlowNet(C=C, K=K)
        
        self.leakyrelu = nn.LeakyReLU(0.1)
    
    def encode(self, x):
        z = self.encoder(x)
        return z
    
    def decode(self, z):
        x = self.decoder(z)
        return x

    def forward(self, data, param=None):
        z = self.encode(data)
        decoded = self.decode(z)
        if self.containNF:
            dummy_z, param_z, logdet = self.flow.forward(z, param)
            return decoded, z, dummy_z, param_z, logdet
        return decoded, z



class Discriminator(nn.Module):
    def __init__(self, C):
        super(Discriminator, self).__init__()
        self.C = C
        self.dis_1 = nn.Sequential(
            Basic.DownscaleNetX2(inch=self.C, outch=self.C*2),  # 1, 12, 24, 48 --> 4,6,12,24
            Basic.DownscaleNetX2(inch=self.C*2, outch=self.C)
            # Basic.UpscaleNetX1(inch=self.C*2, outch=self.C)  # 2, 3, 6, 12   --> 1,3,6,12
        )
        self.classifier = nn.Sequential(
            nn.Linear(1*3*6*12, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        d = self.dis_1(x)
        d = self.classifier(d.view(d.size(0), -1))
        return d


