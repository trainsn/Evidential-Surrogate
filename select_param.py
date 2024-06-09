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
    
    parser.add_argument("--n-candidates", type=int, default=10000,
                        help="number of candidates run for selection")
    parser.add_argument("--k", type=int, default=2400,
                        help="number of selected samples")
    
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

    # load checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint {}".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            g_model.load_state_dict(checkpoint["g_model_state_dict"])
            print("=> loaded checkpoint {} (epoch {})"
                    .format(args.resume, checkpoint["epoch"]))
            
    params, C42a_data, sample_weight = ReadYeastDataset(active=False)
    params, C42a_data, sample_weight = torch.from_numpy(params).float().cuda(), torch.from_numpy(C42a_data).float().cuda(), torch.from_numpy(sample_weight).float().cuda()
    train_split = torch.from_numpy(np.load('train_split.npy'))
    train_params = params[train_split]

    # Function to randomly initialize input parameters
    def initialize_inputs(num_samples=10):
        return torch.rand(num_samples, args.dsp) * 2. - 1.
    
    # Randomly initialize input parameters
    inputs = initialize_inputs(args.n_candidates).to(device)

    # udpating params...
    g_model.train()
    fake_data = g_model(inputs)
    gamma, v, alpha, beta = torch.chunk(fake_data, 4, dim=1) 
    sigma = torch.sqrt(beta / (alpha - 1 + 1e-6))[:, 0]    
    var = torch.sqrt(beta / (v * (alpha - 1 + 1e-6)))[:, 0]

    selected_indices = torch.zeros(args.k, dtype=torch.long)
    distances = torch.full((args.n_candidates,), float('inf'), dtype=torch.float).to(device)

    for i in range(train_params.shape[0]):
        exist_param = train_params[i]
        new_distances = torch.norm(inputs - exist_param, dim=1)
        distances = torch.min(distances, new_distances)

    for i in range(args.k):
        scores = args.lam * var.mean(1) + distances 
        scores[selected_indices[:i]] = float('-inf')
        selected_indices[i] = torch.argmax(scores).item()
        
        new_point_distances = torch.norm(inputs - inputs[selected_indices[i]], dim=1)
        distances = torch.min(distances, new_point_distances)
    
    selected_inputs_slice = inputs[selected_indices]
    selected_inputs_slice = selected_inputs_slice.cpu().numpy()

    selected_inputs = np.zeros((args.k, 35))
    selected_inputs[: ,:25] = selected_inputs_slice[:, :25]
    selected_inputs[: ,32:] = selected_inputs_slice[:, 25:]
    selected_inputs[:, 25:32] = np.random.rand(args.k, 7) * 2 - 1

     # Open file and save the points
    with open("/fs/ess/PAS0027/yeast_polarization_Neng/run_base1/list_of_parameters", 'w') as file:
        for input in selected_inputs:
            # Create a string for each point, joining coordinates with a space
            input_str = '\t'.join(f"{coord:.6f}" for coord in input)
            file.write(input_str + '\n')
    print(f"Selected inputs have been saved to 'list_of_parameters'")

if __name__ == "__main__":
    main(parse_args())
