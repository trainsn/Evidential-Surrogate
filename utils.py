import os 
import numpy as np
import matplotlib.pyplot as plt
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

def gen_cutoff_uncertainty(all_mse, var, method):
    percentiles = np.linspace(0, 1, 100, endpoint=False)
    cutoff_inds = (percentiles * var.numel()).astype(int)
    _, sorted_varidx = torch.sort(var.flatten(), descending=True)

    cutoff_psnrs = []
    for cutoff in cutoff_inds:
        cutoff_mse = all_mse.flatten()[sorted_varidx[cutoff:]].mean().item()
        cutoff_psnrs.append(20. * np.log10(2.) - 10. * np.log10(cutoff_mse))
    np.save(os.path.join("figs", method + "_cutoff_uncertainty_psnrs"), np.array(cutoff_psnrs))

def gen_ret_value(all_mse, data, method):
    percentiles = np.linspace(0, 1, 51, endpoint=True)
    ret_inds = (percentiles * data.numel()).astype(int)
    _, sorted_dataidx = torch.sort(data.flatten(), descending=True)

    ret_psnrs = []
    for ret in ret_inds[1:]:
        ret_mse = all_mse.flatten()[sorted_dataidx[:ret]].mean().item()
        ret_psnrs.append(20. * np.log10(2.) - 10. * np.log10(ret_mse))
    np.save(os.path.join("figs", method + "_ret_value_psnrs"), np.array(ret_psnrs))

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

def render_one_circle(approach, uncertainty_type, input_id, example_test, example_mu, example_var, active=False):
    # Create angles for the points on the circle
    angles = np.linspace(0, 2 * np.pi, 400, endpoint=False)

    n_stds = 1

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 5))

    # Plot the circle
    ax.set_theta_zero_location('S')  # 'S' is for South
    ax.plot(angles, example_test, color='#000000', linewidth=1, zorder=0)
    ax.plot(angles, example_mu, color='#0000ff', linewidth=1, zorder=0)
    for k in np.linspace(0, n_stds, 2):
        ax.fill_between(
            angles, (example_mu - k * example_var), (example_mu + k * example_var),
            alpha=0.3,
            edgecolor=None,
            facecolor='#00aeef',
            linewidth=0,
            zorder=1,
            label="Unc." if k == 0 else None)
    ax.set_ylim(0, None)  
    # axs[1].set_yticks(np.arange(0, 1.1 * max_mu, 1))  # Set y-axis ticks
    # axs[1].set_yticklabels(np.arange(0, 1.1 * max_mu, 1))  # Set y-axis tick labels
    ax.set_title(uncertainty_type + " Uncertainty", fontsize=20)

    ax.tick_params(labelsize=14)  # Adjust tick label size

    plt.savefig(os.path.join("figs", "uncertainty_" + approach + "_id" + str(input_id) + ("_active" if active else "") + ".png"))
    plt.close()

def render_two_circles(approach, input_id, example_test, example_mu, example_sigma, example_var):    
    # Create angles for the points on the circle
    angles = np.linspace(0, 2 * np.pi, 400, endpoint=False)

    n_stds = 1
            
    # Create subplots for two circles
    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(12, 5)) 

    # Adjust margins
    plt.subplots_adjust(left=0.0, right=1.0)

    # Plot the circle
    axs[0].set_theta_zero_location('S')  # 'S' is for South
    axs[0].plot(angles, example_test, color='#000000', linewidth=1, zorder=0)
    axs[0].plot(angles, example_mu, color='#0000ff', linewidth=1, zorder=0)
    for k in np.linspace(0, n_stds, 2):
        axs[0].fill_between(
            angles, (example_mu - k * example_sigma), (example_mu + k * example_sigma),
            alpha=0.3,
            edgecolor=None,
            facecolor='#00aeef',
            linewidth=0,
            zorder=1,
            label="Unc." if k == 0 else None)
    axs[0].set_ylim(0, None)  
    # axs[0].set_yticks(np.arange(0, 1.1 * max_mu, 1))  # Set y-axis ticks
    # axs[0].set_yticklabels(np.arange(0, 1.1 * max_mu, 1))  # Set y-axis tick labels
    axs[0].set_title("Aleatoric Uncertainty", fontsize=20)


    # Plot the circle
    axs[1].set_theta_zero_location('S')  # 'S' is for South
    axs[1].plot(angles, example_test, color='#000000', linewidth=1, zorder=0)
    axs[1].plot(angles, example_mu, color='#0000ff', linewidth=1, zorder=0)
    for k in np.linspace(0, n_stds, 2):
        axs[1].fill_between(
            angles, (example_mu - k * example_var), (example_mu + k * example_var),
            alpha=0.3,
            edgecolor=None,
            facecolor='#00aeef',
            linewidth=0,
            zorder=1,
            label="Unc." if k == 0 else None)
    axs[1].set_ylim(0, None)  
    # axs[1].set_yticks(np.arange(0, 1.1 * max_mu, 1))  # Set y-axis ticks
    # axs[1].set_yticklabels(np.arange(0, 1.1 * max_mu, 1))  # Set y-axis tick labels
    axs[1].set_title("Epistemic Uncertainty", fontsize=20)

    for ax in axs:
        ax.tick_params(labelsize=14)  # Adjust tick label size

    plt.savefig(os.path.join("figs", "uncertainty_" + approach + "_id" + str(input_id) + ".png"))
    plt.close()
