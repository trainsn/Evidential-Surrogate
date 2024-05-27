import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F

from NF import Basic
from NF.FlowStep import FlowStep
from NF.ConditionalFlow import ConditionalFlow

import pdb

class ParamFlowNetCond(nn.Module):
    def __init__(self, C=1, K=8, K_cond=5):
        super().__init__()
        self.C = C
        self.K = K
        self.K_cond = K_cond
        
        flow_actNorm='actNorm1d'
        flow_permutation = 'invconv' 
        flow_coupling = 'Affine' 
        print(flow_actNorm, flow_permutation, flow_coupling)
        
        cond_channels = None 
        
        self.layers = nn.ModuleList()
        
        # coupling layers
        # 1. Squeeze
        self.layers.append(Basic.SqueezeLayer(2, dim='1D')) 
        self.C = self.C * 2
        
        # 2. main FlowSteps (unconditional flow)
        for k in range(self.K):
            self.layers.append(FlowStep(in_channels=self.C, cond_channels=cond_channels,
                                        flow_actNorm=flow_actNorm,
                                        flow_permutation=flow_permutation,
                                        flow_coupling=flow_coupling))
        self.layers.append(Basic.Split(num_channels_split=self.C // 2)) # WE ACTUALLY DO NOT USE THIS SPLIT LAYER
        
        self.condFlow = ConditionalFlow(num_channels=self.C,
                                        n_flow_step=self.K_cond,
                                        flow_actNorm = flow_actNorm,
                                        flow_permutation=flow_permutation,
                                        flow_coupling=flow_coupling)
        self.layers.append(self.condFlow)

        # conditional Gaussian
        self.param_dim = 28
    
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

    def reverse_flow(self, z, d_param, logdet=None, eps_std=1, recoverZ=True):
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
                if not recoverZ: # for rev zc loss
                    return param_z
            elif isinstance(layer, ConditionalFlow):
                # important!!!! add recoverZ=False here to avoid recover z from Gaussian to VAE latent space
                z, logdet = layer(z, d_param, eps_std=eps_std, logdet=logdet, reverse=True, recoverZ=recoverZ)
        return z, param_z, logdet
    
    def sample(self, dummy_z, d_param, logdet=0, eps_std=0.9, recoverZ=True):
        return self.reverse_flow(dummy_z, d_param, logdet=logdet, eps_std=eps_std, recoverZ=recoverZ)
