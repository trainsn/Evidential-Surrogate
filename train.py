# main file

from __future__ import absolute_import, division, print_function

import os
import argparse
import math

import numpy as np
from tqdm import tqdm

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

    if args.loss == 'Evidential':
        out_features = 4
    elif args.loss == 'Gaussian':
        out_features = 2
    else:
        out_features = 1

    network_str = "model_" + args.loss + "_seed" + str(args.seed) 
    if args.dropout: 
        network_str += "_dp" 
    if args.active: 
        network_str += "_active" 

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # model
    def weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.orthogonal_(m.weight)
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
    print(g_model)
    # g_model.apply(weights_init)
    # if args.sn:
    #     g_model = add_sn(g_model)

    g_model.to(device)
    if args.loss == 'Evidential':
        print('Use Evidential Loss')
    elif args.loss == 'Gaussian':
        print('Use Gaussian Loss')
    elif args.loss == 'MSE':
        print('Use MSE Loss')
        criterion = nn.MSELoss()
    elif args.loss == 'L1':
        print('Use L1 Loss')
        criterion = nn.L1Loss()
    train_losses, test_losses = [], []

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
            train_losses = checkpoint["train_losses"]
            test_losses = checkpoint["test_losses"]
            print("=> loaded checkpoint {} (epoch {})"
                .format(args.resume, checkpoint["epoch"]))
            
    params, C42a_data, sample_weight = ReadYeastDataset(args.active)
    params, C42a_data, sample_weight = torch.from_numpy(params).float().cuda(), torch.from_numpy(C42a_data).float().cuda(), torch.from_numpy(sample_weight).float().cuda()
    train_split = torch.from_numpy(np.load('train_split.npy'))
    if args.active:
        train_split = torch.cat((train_split, torch.ones(2400, dtype=torch.bool)), dim=0)
    train_params, train_C42a_data, train_sample_weight = params[train_split], C42a_data[train_split], sample_weight[train_split]
    test_params, test_C42a_data = params[~train_split], C42a_data[~train_split]
    len_train = train_params.shape[0]
    num_batches = (len_train - 1) // args.batch_size + 1

    # main loop
    for epoch in range(args.start_epoch, args.epochs):
        # training...
        g_model.train()
        train_loss = 0.
        train_mse = 0.

        for _ in range(num_batches): 
            e_rndidx = torch.multinomial(train_sample_weight.flatten(), args.batch_size, replacement=True)
            sub_params = train_params[e_rndidx]
            sub_data = train_C42a_data[e_rndidx]

            g_optimizer.zero_grad()
            fake_data = g_model(sub_params)

            if args.loss == 'Evidential':
                sub_data = sub_data.unsqueeze(1)
                loss = loss_helper.EvidentialRegression(sub_data, fake_data, coeff=1e-2)
                gamma, _, _, _ = torch.chunk(fake_data, out_features, dim=1) 
                mse = torch.mean((gamma - sub_data) ** 2)
            elif args.loss == 'Gaussian':
                sub_data = sub_data.unsqueeze(1)
                mu, sigma = fake_data.chunk(2, dim=1)
                loss = loss_helper.Gaussian_NLL(sub_data, mu, sigma)
                mse = torch.mean((mu - sub_data) ** 2)
            else:
                fake_data = fake_data[:, 0]
                loss = criterion(sub_data, fake_data)
                mse = torch.mean((fake_data - sub_data) ** 2)

            loss.backward()
            g_optimizer.step()
            train_loss += loss.item()
            train_mse += mse.item()


        if (epoch + 1) % args.log_every == 0:
            print("====> Epoch: {} Average loss: {:.6f}, Average MSE: {:.6f}".format(
                        epoch + 1, train_loss / num_batches, train_mse / num_batches))

        # testing...
        # g_model.eval()
        test_loss = 0.
        with torch.no_grad():
            fake_data = g_model(test_params)
            if args.loss == 'Evidential':
                test_loss = loss_helper.EvidentialRegression(test_C42a_data.unsqueeze(1), fake_data, coeff=1e-2)
                gamma, _, _, _ = torch.chunk(fake_data, out_features, dim=1) 
                test_mse = torch.mean((gamma - test_C42a_data.unsqueeze(1)) ** 2)
            elif args.loss == 'Gaussian':
                mu, sigma = fake_data.chunk(2, dim=1)
                test_loss = loss_helper.Gaussian_NLL(test_C42a_data.unsqueeze(1), mu, sigma)
                test_mse = torch.mean((mu - test_C42a_data.unsqueeze(1)) ** 2)
            else:
                fake_data = fake_data[:, 0]
                test_loss = criterion(test_C42a_data, fake_data).item()
                test_mse = torch.mean((fake_data - test_C42a_data) ** 2)

        test_losses.append(test_loss)
        if (epoch + 1) % args.log_every == 0:
            print("====> Epoch: {} Test set loss: {:.6f}, Test set MSE {:.6f}".format(
                        epoch + 1, test_losses[-1], test_mse))

        # saving...
        if (epoch + 1) % args.check_every == 0:
            print("=> saving checkpoint at epoch {}".format(epoch))
            torch.save({"epoch": epoch + 1,
                        "g_model_state_dict": g_model.state_dict(),
                        "g_optimizer_state_dict": g_optimizer.state_dict(),
                        "train_losses": train_losses,
                        "test_losses": test_losses},
                        os.path.join("models", network_str + "_" + str(epoch + 1) + ".pth.tar"))

            torch.save(g_model.state_dict(), 
                       os.path.join("models", network_str + "_" + str(epoch + 1) + ".pth"))

if __name__ == "__main__":
    main(parse_args())
