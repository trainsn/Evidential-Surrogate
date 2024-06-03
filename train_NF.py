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
from NF.FlowNet_surrogate import ParamFlowNetCond

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

    parser.add_argument("--K", type=int, default=3,
                        help="the number of unconditional transformations")
    parser.add_argument("--K_cond", type=int, default=3,
                        help="the number of conditional transformations")

    parser.add_argument("--sn", action="store_true", default=False,
                        help="enable spectral normalization")

    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate (default: 1e-3)")
    parser.add_argument("--logx_loss", type=float, default=1, 
                        help="logx loss (default: 1)")
    parser.add_argument("--param_l1_loss", type=float, default=0, 
                        help="param l1 loss (default: 0)")
    parser.add_argument("--var_loss", type=float, default=0, 
                        help="variance reduction loss (default: 0)")
    parser.add_argument("--mean_loss", type=float, default=0, 
                        help="mean loss (default: 0)")
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

    network_str = "model_NF"

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

    g_model = ParamFlowNetCond(C=1, K=args.K, K_cond=args.K_cond)
    print(g_model)
    # g_model.apply(weights_init)
    # if args.sn:
    #     g_model = add_sn(g_model)

    g_model.to(device)

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
    train_params, train_C42a_data, train_sample_weight = params[train_split], C42a_data[train_split], sample_weight[train_split]
    test_params, test_C42a_data = params[~train_split], C42a_data[~train_split]
    len_train = train_params.shape[0]
    num_batches = (len_train - 1) // args.batch_size + 1

    MAE = nn.L1Loss()

    # main loop
    for epoch in range(args.start_epoch, args.epochs):
        # training...
        g_model.train()

        epoch_logpx_loss = 0
        epoch_param_l1_loss = 0
        epoch_var_loss = 0
        epoch_mean_loss = 0

        for _ in range(num_batches): 
            e_rndidx = torch.multinomial(train_sample_weight.flatten(), args.batch_size, replacement=True)
            sub_params = train_params[e_rndidx]
            sub_data = train_C42a_data[e_rndidx]

            g_optimizer.zero_grad()
            dummy_z, param_z, logdet, var_loss, mean_loss = g_model(sub_data.unsqueeze(1), sub_params)
            loss_x = (-logdet / (math.log(2) * sub_data.shape[1])).mean()
            loss = args.logx_loss * loss_x
            epoch_logpx_loss += args.logx_loss * loss_x.item()

            if args.mean_loss > 0:
                mean_loss = args.mean_loss * mean_loss
                loss += args.mean_loss * mean_loss 
                epoch_mean_loss += args.mean_loss * mean_loss.item()
            
            if args.var_loss > 0:
                var_loss = args.var_loss * var_loss
                loss += args.var_loss * var_loss 
                epoch_var_loss += args.var_loss * var_loss.item()
            
            if args.param_l1_loss > 0:
                param_l1_loss = MAE(param_z, sub_params)
                loss += args.param_l1_loss * param_l1_loss 
                epoch_param_l1_loss += args.param_l1_loss * param_l1_loss.item()
               
            loss.backward()
            g_optimizer.step()

        if (epoch + 1) % args.log_every == 0:
            print("====> Epoch: {} Average logp_x_loss: {:.6f}, Average mean_loss: {:.6f}, Average var_loss: {:.6f}， Average param_l1_loss: {:.6f}".format(
                        epoch + 1, epoch_logpx_loss / num_batches, epoch_mean_loss / num_batches, var_loss / num_batches, epoch_param_l1_loss / num_batches))

        # testing...
        g_model.eval()
        with torch.no_grad():
            fake_data, _, _ = g_model.sample(dummy_z=None, d_param=test_params, eps_std=0.8)
            fake_data = fake_data[:, 0]
            test_mse = torch.mean((fake_data - test_C42a_data) ** 2)

        if (epoch + 1) % args.log_every == 0:
            print("====> Epoch: {} Test set MSE {:.6f}".format(
                        epoch + 1, test_mse))

        # saving...
        if (epoch + 1) % args.check_every == 0:
            print("=> saving checkpoint at epoch {}".format(epoch))
            torch.save({"epoch": epoch + 1,
                        "g_model_state_dict": g_model.state_dict(),
                        "g_optimizer_state_dict": g_optimizer.state_dict()},
                        os.path.join("models", network_str + "_" + str(epoch + 1) + ".pth.tar"))

            torch.save(g_model.state_dict(), 
                       os.path.join("models", network_str + "_" + str(epoch + 1) + ".pth"))

if __name__ == "__main__":
    main(parse_args())
