import os 
import numpy as np
import torch

def gen_cutoff(all_mse, var, method):
    percentiles = np.linspace(0, 1, 100, endpoint=False)
    cutoff_inds = (percentiles * var.numel()).astype(int)
    _, sorted_varidx = torch.sort(var.flatten(), descending=True)

    cutoff_psnrs = []
    for cutoff in cutoff_inds:
        cutoff_mse = all_mse.flatten()[sorted_varidx[cutoff:]].mean().item()
        cutoff_psnrs.append(20. * np.log10(2.) - 10. * np.log10(cutoff_mse))
    np.save(os.path.join("figs", method + "_cutoff_psnrs"), np.array(cutoff_psnrs))