import os 
import numpy as np
import torch
from torch.distributions.normal import Normal

import pdb

def norm_ppf(p, mean=0.0, std=1.0):
    """
    Compute the quantile function (percent point function or inverse CDF) of the normal distribution.

    Args:
    p (Tensor): Probabilities.
    mean (float or Tensor): Mean of the normal distribution.
    std (float or Tensor): Standard deviation of the normal distribution.

    Returns:
    Tensor: the corresponding quantile values of the probabilities p for a normal distribution.
    """
    normal_dist = Normal(mean, std)
    return normal_dist.icdf(torch.tensor(p))

def gen_cutoff(all_mse, var, method):
    percentiles = np.linspace(0, 1, 100, endpoint=False)
    cutoff_inds = (percentiles * var.numel()).astype(int)
    _, sorted_varidx = torch.sort(var.flatten(), descending=True)

    cutoff_psnrs = []
    for cutoff in cutoff_inds:
        cutoff_mse = all_mse.flatten()[sorted_varidx[cutoff:]].mean().item()
        cutoff_psnrs.append(20. * np.log10(2.) - 10. * np.log10(cutoff_mse))
    np.save(os.path.join("figs", method + "_cutoff_psnrs"), np.array(cutoff_psnrs))

def gen_calibration(mu, var, gt):
    expected_p = np.linspace(0, 1, 40, endpoint=True)

    observed_p = []
    for p in expected_p:
        ppf = norm_ppf(p, mu, var)
        obs_p = (gt < ppf).sum().item() / var.numel()
        observed_p.append(obs_p)
    observed_p = np.array(observed_p)

    calibration_err = np.abs(expected_p - observed_p).mean()
    return calibration_err, observed_p
