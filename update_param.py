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

from generator import Generator

import pdb

# parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Deep Learning Model")

    parser.add_argument("--no-cuda", action="store_true", default=False,
                        help="disables CUDA training")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")

    parser.add_argument("--resume", type=str, default="",
                        help="path to the latest checkpoint (default: none)")

    parser.add_argument("--dsp", type=int, default=35,
                        help="dimensions of the simulation parameters (default: 35)")
    parser.add_argument("--dspe", type=int, default=512,
                        help="dimensions of the simulation parameters' encode (default: 512)")
    parser.add_argument("--ch", type=int, default=4,
                        help="channel multiplier (default: 4)")

    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate (default: 1e-3)")
    parser.add_argument("--beta1", type=float, default=0.9,
                        help="beta1 of Adam (default: 0.9)")
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="beta2 of Adam (default: 0.999)")
    parser.add_argument("--epochs", type=int, default=50000,
                        help="number of epochs to update")
    
    parser.add_argument("--log-every", type=int, default=40,
                        help="log training status every given number of batches")
    
    parser.add_argument("--lam", type=float, default=1e-2,
                        help="l2-norm regularizer to constrain the input search space within a known confinement")

    return parser.parse_args()

# the main function
def main(args):
    # log hyperparameters
    print(args)

    # select device
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")

    out_features = 4

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    g_model = Generator(args.dsp, args.dspe, args.ch, out_features, dropout=False)

    g_model.to(device)
    criterion = nn.MSELoss()

    # Random initial values in range [-1, 1]
    tmp_params = (2 * torch.rand((1, args.dsp)) - 1).to(device)
    tmp_params.requires_grad_(True)
    initial_params = tmp_params.detach().clone()
    # optimizer
    # p_optimizer = optim.Adam([tmp_params], lr=args.lr, betas=(args.beta1, args.beta2))
    p_optimizer = optim.SGD([tmp_params], lr=args.lr)

    # load checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint {}".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            g_model.load_state_dict(checkpoint["g_model_state_dict"])
            print("=> loaded checkpoint {} (epoch {})"
                    .format(args.resume, checkpoint["epoch"]))

    # udpating params...
    g_model.train()
    for step in range(args.epochs):
        fake_data = g_model(tmp_params)
        gamma, v, alpha, beta = torch.chunk(fake_data, 4, dim=1) 
        sigma = torch.sqrt(beta / (alpha - 1 + 1e-6))[:, 0]    
        var = torch.sqrt(beta / (v * (alpha - 1 + 1e-6)))[:, 0]

        dist = criterion(tmp_params, initial_params)
        obj = -var.mean()
        obj.backward()
        p_optimizer.step()
        if step % args.log_every == 0:
            print(f'Step {step}: obj = {obj}, Uncertainty = {var.mean().item()}, Dist = {dist.item()}')

if __name__ == "__main__":
    main(parse_args())
