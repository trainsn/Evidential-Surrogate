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

import pdb

# parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Deep Learning Model")

    parser.add_argument("--no-cuda", action="store_true", default=False,
                                            help="disables CUDA training")
    parser.add_argument("--data-parallel", action="store_true", default=False,
                                            help="enable data parallelism")
    parser.add_argument("--seed", type=int, default=1,
                                            help="random seed (default: 1)")

    parser.add_argument("--resume", type=str, default="",
                                            help="path to the latest checkpoint (default: none)")

    parser.add_argument("--dsp", type=int, default=35,
                                            help="dimensions of the simulation parameters (default: 35)")

    parser.add_argument("--sn", action="store_true", default=False,
                                            help="enable spectral normalization")

    parser.add_argument("--lr", type=float, default=1e-3,
                                            help="learning rate (default: 1e-3)")
    parser.add_argument("--beta1", type=float, default=0.9,
                                            help="beta1 of Adam (default: 0.9)")
    parser.add_argument("--beta2", type=float, default=0.999,
                                            help="beta2 of Adam (default: 0.999)")
    parser.add_argument("--batch-size", type=int, default=32,
                                            help="batch size for training (default: 32)")
    parser.add_argument("--start-epoch", type=int, default=0,
                                            help="start epoch number (default: 0)")
    parser.add_argument("--epochs", type=int, default=50000,
                                            help="number of epochs to train")

    parser.add_argument("--log-every", type=int, default=40,
                                            help="log training status every given number of batches")
    parser.add_argument("--check-every", type=int, default=200,
                                            help="save checkpoint every given number of epochs")

    return parser.parse_args()

# the main function
def main(args):
    # log hyperparameters
    print(args)

    # select device
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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

    g_model = Generator(args.dsp)
    # if args.sn:
    #     g_model = add_sn(g_model)

    g_model.to(device)

    mse_criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()

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
            
    params, C42a_data, sample_weight = ReadYeastDataset()
    params, C42a_data, sample_weight = torch.from_numpy(params).float().cuda(), torch.from_numpy(C42a_data).float().cuda(), torch.from_numpy(sample_weight).float().cuda()
    train_split = torch.from_numpy(np.load('train_split.npy'))
    test_params, test_C42a_data = params[~train_split], C42a_data[~train_split]
    dmin, dmax = 0.0, 580.783

    # testing...
    # g_model.eval()
    with torch.no_grad():
        fake_data = g_model(test_params)
        mse = mse_criterion(test_C42a_data, fake_data).item()
        psnr = 20. * np.log10(2.) - 10. * np.log10(mse)
        print(f"PSNR: {psnr:.2f} dB")

        # Rescale data back to original range
        fake_data = ((fake_data + 1) * (dmax - dmin) / 2) + dmin
        test_C42a_data = ((test_C42a_data + 1) * (dmax - dmin) / 2) + dmin
    pdb.set_trace()

    example_data = fake_data[51].cpu().numpy()
    
    # Create angles for the points on the circle
    angles = np.linspace(0, 2*np.pi, 400, endpoint=False) 

    # Plot the circle
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    sc = ax.scatter(angles, np.ones_like(angles), c=example_data, cmap='viridis', s=10)
    ax.set_yticklabels([])  # Hide radial ticks

    # Add colorbar
    cbar = plt.colorbar(sc, orientation='vertical')
    cbar.set_label('C42a')

    plt.show()


if __name__ == "__main__":
    main(parse_args())
