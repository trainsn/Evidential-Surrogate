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

import pdb

# parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Deep Learning Model")

    parser.add_argument("--no-cuda", action="store_true", default=False,
                        help="disables CUDA training")
    parser.add_argument("--data-parallel", action="store_true", default=False,
                        help="enable data parallelism")

    parser.add_argument("--model-paths", type=str, nargs='+', default=[],
                        help="paths to the checkpoints of the models for ensemble")

    parser.add_argument("--dsp", type=int, default=35,
                        help="dimensions of the simulation parameters (default: 35)")
    parser.add_argument("--dspe", type=int, default=512,
                        help="dimensions of the simulation parameters' encode (default: 512)")
    parser.add_argument("--ch", type=int, default=4,
                        help="channel multiplier (default: 4)")

    parser.add_argument("--sn", action="store_true", default=False,
                        help="enable spectral normalization")

    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate (default: 1e-3)")
    parser.add_argument("--loss", type=str, default='MSE',
                        help="loss function for training (default: MSE)")
    parser.add_argument("--beta1", type=float, default=0.9,
                        help="beta1 of Adam (default: 0.9)")
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="beta2 of Adam (default: 0.999)")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="start epoch number (default: 0)")

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

    out_features = 4 if args.loss == 'Evidential' else 1

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

    # Load models for ensemble
    models = []
    for model_path in args.model_paths:
        model = Generator(args.dsp, args.dspe, args.ch, out_features, dropout=False)
        # if args.sn:
            # model = add_sn(model)
        model.to(device)
        if os.path.isfile(model_path):
            print(f"=> loading checkpoint {model_path}")
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint["g_model_state_dict"])
            print(f"=> loaded checkpoint {model_path}")
        else:
            print(f"=> no checkpoint found at {model_path}")
        models.append(model)

    mse_criterion = nn.MSELoss()
            
    params, C42a_data, sample_weight = ReadYeastDataset()
    params, C42a_data, sample_weight = torch.from_numpy(params).float().cuda(), torch.from_numpy(C42a_data).float().cuda(), torch.from_numpy(sample_weight).float().cuda()
    train_split = torch.from_numpy(np.load('train_split.npy'))
    test_params, test_C42a_data = params[~train_split], C42a_data[~train_split]
    dmin, dmax = 0.0, 580.783

    # testing...
    assert args.loss == 'MSE'
    with torch.no_grad():
        fake_data = []
        for model in models:
            model.train()
            fake_data.append(model(test_params))
        fake_data = torch.stack(fake_data, dim=0)
        mu = torch.mean(fake_data, dim=0)[:, 0]
        var = torch.std(fake_data, dim=0)[:, 0]
        mse = mse_criterion(test_C42a_data, mu).item()
        nll, trimmed_nll = loss_helper.Gaussian_NLL(test_C42a_data, mu, var)
        print(f"NLL: {nll:.2f}\tTrimmed NLL: {trimmed_nll:.2f}")
        mu = ((mu + 1) * (dmax - dmin) / 2) + dmin
        var = var * (dmax - dmin) / 2
        
        psnr = 20. * np.log10(2.) - 10. * np.log10(mse)
        print(f"PSNR: {psnr:.2f} dB")

        # Rescale data back to original range
        test_C42a_data = ((test_C42a_data + 1) * (dmax - dmin) / 2) + dmin


    # Create angles for the points on the circle
    angles = np.linspace(0, 2*np.pi, 400, endpoint=False) 

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


if __name__ == "__main__":
    main(parse_args())
