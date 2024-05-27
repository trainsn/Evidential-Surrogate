import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from NF import thops, Basic
from NF.FlowStep import FlowStep

import pdb


class ConditionalFlow(nn.Module):
    def __init__(self, num_channels, n_flow_step=0,  
                    flow_actNorm='actNorm3d', flow_permutation='invconv', flow_coupling='Affine'):
        super().__init__()
        self.C = num_channels

        self.param_dim = 28
        self.mlp_first = nn.Sequential(nn.Linear(self.param_dim, self.C * 50, bias=False))

        self.resnet_feat2_1 = Basic.UpscaleNetX2(inch=self.C, outch=self.C)
        self.resnet_feat2_2 = Basic.UpscaleNetX2(inch=self.C, outch=self.C)
        self.resnet_feat1_1 = Basic.UpscaleNetX1(inch=self.C, outch=self.C)
        self.resnet_feat1_2 = Basic.UpscaleNetX1(inch=self.C, outch=self.C//2)

        self.f = Basic.Conv1dZeros(self.C//2, self.C*2)
        
        # conditional flow
        self.additional_flow_steps = nn.ModuleList()
        for k in range(n_flow_step):
            self.additional_flow_steps.append(FlowStep(in_channels=self.C,
                                                        cond_channels=self.C//2, 
                                                        flow_actNorm=flow_actNorm,
                                                        flow_permutation=flow_permutation,  #opt['flow_permutation'],
                                                        flow_coupling=flow_coupling)) #opt['flow_coupling']))

    def forward(self, z, d_param, eps_std=None, logdet=0, reverse=False, recoverZ=True):
        if not reverse:
            conditional_feature = self.get_conditional_feature(d_param)
            
            for layer in self.additional_flow_steps:
                z, logdet = layer(z, u=conditional_feature, logdet=logdet, reverse=False)
            # print('lat z0', z[0][0][0][0][:20])
            # np.save('tmp_z.npy', z.cpu().numpy())
            h = self.f(conditional_feature)
            mean, logs = thops.split_feature(h, "cross")
            logdet += Basic.GaussianDiag.logp(mean, logs, z)
            var_reduction_loss = torch.exp(logs).mean()
            mean_loss = torch.abs(z - mean).mean()
            return z, logdet, var_reduction_loss, mean_loss

        else:
            conditional_feature = self.get_conditional_feature(d_param)

            h = self.f(conditional_feature)
            mean, logs = thops.split_feature(h, "cross")
            if z is None:
                z = Basic.GaussianDiag.sample(mean, logs, eps_std)
            logdet -= Basic.GaussianDiag.logp(mean, logs, z)

            if recoverZ:
                for layer in reversed(self.additional_flow_steps):
                    z, logdet = layer(z, u=conditional_feature, logdet=logdet, reverse=True)
            
            return z, logdet
    
    # for loss computation
    def z0_to_m(self, z0, d_param):
        conditional_feature = self.get_conditional_feature(d_param)
        for layer in reversed(self.additional_flow_steps):
            z, _ = layer(z0, u=conditional_feature, logdet=0, reverse=True)
        return z

    def get_conditional_feature(self, u):
        u_feature_first = self.mlp_first(u)

        u_feature_first = u_feature_first.view(u_feature_first.size(0), self.C, 50)
        u_feature = self.resnet_feat2_2(self.resnet_feat2_1(u_feature_first))
        u_feature = self.resnet_feat1_1(u_feature)
        u_feature = self.resnet_feat1_2(u_feature)

        return u_feature
        