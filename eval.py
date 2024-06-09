# main file

from __future__ import absolute_import, division, print_function

import os
import argparse
import math

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from yeast import *
from generator import Generator
import loss_helper
import utils

import pdb

# parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Deep Learning Model")

    parser.add_argument("--no-cuda", action="store_true", default=False,
                        help="disables CUDA training")
    parser.add_argument("--data-parallel", action="store_true", default=False,
                        help="enable data parallelism")

    parser.add_argument("--resume", type=str, default="",
                        help="path to the latest checkpoint (default: none)")

    parser.add_argument("--dsp", type=int, default=35,
                        help="dimensions of the simulation parameters (default: 35)")
    parser.add_argument("--dspe", type=int, default=512,
                        help="dimensions of the simulation parameters' encode (default: 512)")
    parser.add_argument("--ch", type=int, default=4,
                        help="channel multiplier (default: 4)")

    parser.add_argument("--sn", action="store_true", default=False,
                        help="enable spectral normalization")
    parser.add_argument("--active", action="store_true", default=False,
                        help="active learning version")

    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate (default: 1e-3)")
    parser.add_argument("--loss", type=str, default='MSE',
                        help="loss function for training (default: MSE)")
    parser.add_argument("--dropout", action="store_true", default=False,
                        help="using dropout layer after convolution")
    parser.add_argument("--beta1", type=float, default=0.9,
                        help="beta1 of Adam (default: 0.9)")
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="beta2 of Adam (default: 0.999)")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="start epoch number (default: 0)")
    parser.add_argument("--n-samples", type=int, default=10,
                        help="number of samples run for the dropout or ensemble model")

    parser.add_argument("--log-every", type=int, default=40,
                        help="log training status every given number of batches")
    parser.add_argument("--check-every", type=int, default=200,
                        help="save checkpoint every given number of epochs")
    
    parser.add_argument("--id", type=int, default=51,
                        help="instance id in the testing set")

    return parser.parse_args()

# the main function
def main(args):
    # log hyperparameters
    print(args)

    # select device
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")

    if args.loss == 'Evidential':
        out_features = 4
    elif args.loss == 'Gaussian':
        out_features = 2
    else:
        out_features = 1

    # model
    def weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def add_sn(m):
        for name, c in m.named_children():
            m.add_module(name, add_sn(c))
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            return nn.utils.spectral_norm(m, eps=1e-4)
        else:
            return m

    g_model = Generator(args.dsp, args.dspe, args.ch, out_features, dropout=args.dropout)
    # if args.sn:
    #     g_model = add_sn(g_model)

    g_model.to(device)

    mse_criterion = nn.MSELoss(reduction='none')

    # optimizer
    g_optimizer = optim.Adam(g_model.parameters(), lr=args.lr,
                             betas=(args.beta1, args.beta2))

    # load checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint {}".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            g_model.load_state_dict(checkpoint["g_model_state_dict"])
            g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
            print("=> loaded checkpoint {} (epoch {})"
                    .format(args.resume, checkpoint["epoch"]))
            
    params, C42a_data, sample_weight, dmin, dmax = ReadYeastDataset(args.active)
    params, C42a_data, sample_weight = torch.from_numpy(params).float().cuda(), torch.from_numpy(C42a_data).float().cuda(), torch.from_numpy(sample_weight).float().cuda()
    train_split = torch.from_numpy(np.load('train_split.npy'))
    if args.active:
        train_split = torch.cat((train_split, torch.ones(2400, dtype=torch.bool)), dim=0)
    test_params, test_C42a_data = params[~train_split], C42a_data[~train_split]

    # testing...
    g_model.train()
    with torch.no_grad():
        if args.dropout:
            assert args.loss == 'MSE'
            fake_data = []
            table_size = 1009  
            for i in range(args.n_samples):
                torch.cuda.manual_seed(np.mod(np.power(7, i), table_size))
                fake_data.append(g_model(test_params))
            fake_data = torch.stack(fake_data, dim=0)
            mu = torch.mean(fake_data, dim=0)[:, 0]
            var = torch.std(fake_data, dim=0)[:, 0]
            all_mse = mse_criterion(test_C42a_data, mu)
            all_mse /= (696.052 / dmax) ** 2
            mse = all_mse.mean().item()
            nll = loss_helper.Gaussian_NLL(test_C42a_data, mu, var, reduce=False)
            print(f"NLL: {nll.median().item():.2f}")

            utils.gen_cutoff_uncertainty(all_mse, var, "dropout")
            calibration_err, observed_p = utils.gen_calibration(mu, var, test_C42a_data)
            np.save(os.path.join("figs", "dropout_observed_conf"), observed_p)
            print(f"Calibration Error: {calibration_err:.4f}")

            mu = ((mu + 1) * (dmax - dmin) / 2) + dmin
            var = var * (dmax - dmin) / 2
        elif args.loss == 'Evidential':
            fake_data = g_model(test_params)
            gamma, v, alpha, beta = torch.chunk(fake_data, 4, dim=1) 
            nll = loss_helper.NIG_NLL(test_C42a_data.unsqueeze(1), gamma, v, alpha, beta, reduce=False)
            print(f"NLL: {nll.median().item():.2f}")
            mu = gamma[:, 0]
            all_mse = mse_criterion(test_C42a_data, mu)
            all_mse /= (696.052 / dmax) ** 2
            if args.active:
                utils.gen_ret_value(all_mse, test_C42a_data, "activebase")
            mse = all_mse.mean().item()
            sigma = torch.sqrt(beta / (alpha - 1 + 1e-6))[:, 0]    
            var = torch.sqrt(beta / (v * (alpha - 1 + 1e-6)))[:, 0]

            title = "evidential"
            if args.active:
                title += "_active"
            utils.gen_cutoff_uncertainty(all_mse, var, title)
            calibration_err, observed_p = utils.gen_calibration(mu, var, test_C42a_data)
            title = "evidential_observed_conf"
            if args.active:
                title = "evidential_active_observed_conf"
            np.save(os.path.join("figs", title), observed_p)
            print(f"Calibration Error: {calibration_err:.4f}")

            sigma = sigma * (dmax - dmin) / 2
            mu = ((mu + 1) * (dmax - dmin) / 2) + dmin
            var = var * (dmax - dmin) / 2
        else:
            fake_data = g_model(test_params)
            fake_data = fake_data[:, 0]
            mse = mse_criterion(test_C42a_data, fake_data).item()
            fake_data = ((fake_data + 1) * (dmax - dmin) / 2) + dmin
        psnr = 20. * np.log10(2.) - 10. * np.log10(mse)
        print(f"PSNR: {psnr:.2f} dB")

        # Rescale data back to original range
        test_C42a_data = ((test_C42a_data + 1) * (dmax - dmin) / 2) + dmin


    if args.id >= 0:
        # Create angles for the points on the circle
        angles = np.linspace(0, 2*np.pi, 400, endpoint=False) 

        if args.dropout:
            assert args.loss == 'MSE'

            example_test = test_C42a_data[args.id].cpu().numpy()
            example_mu = mu[args.id].cpu().numpy()
            example_var = var[args.id].cpu().numpy()
            # example_var = np.minimum(example_var, np.percentile(example_var, 90))
            print("max var: ",  np.max(example_var))

            n_stds = 1

            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 5))

            # Plot the circle
            ax.plot(angles, example_test, color='#000000', linewidth=1, zorder=0, label="Train")
            ax.plot(angles, example_mu, color='#0000ff', linewidth=1, zorder=0, label="Train")
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
            ax.set_title("Epistemic Uncertainty")

            plt.show()

        elif args.loss == "Evidential":
            example_test = test_C42a_data[args.id].cpu().numpy()
            example_mu = mu[args.id].cpu().numpy()
            example_sigma = sigma[args.id].cpu().numpy()
            # example_sigma = np.minimum(example_sigma, np.percentile(example_sigma, 90))
            example_var = var[args.id].cpu().numpy()
            # example_var = np.minimum(example_var, np.percentile(example_var, 90))
            print("max sigma: ", np.max(example_sigma), "max var: ",  np.max(example_var))

            n_stds = 1
            
            # Create subplots for two circles
            fig, axs = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(12, 5)) 

            # Plot the circle
            axs[0].plot(angles, example_test, color='#000000', linewidth=1, zorder=0, label="Train")
            axs[0].plot(angles, example_mu, color='#0000ff', linewidth=1, zorder=0, label="Train")
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
            axs[0].set_title("Aleatoric Uncertainty")

            # Plot the circle
            axs[1].plot(angles, example_test, color='#000000', linewidth=1, zorder=0, label="Train")
            axs[1].plot(angles, example_mu, color='#0000ff', linewidth=1, zorder=0, label="Train")
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
            axs[1].set_title("Epistemic Uncertainty")

            plt.show()


if __name__ == "__main__":
    main(parse_args())
