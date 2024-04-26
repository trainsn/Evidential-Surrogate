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
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

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

    parser.add_argument("--root", required=True, type=str,
                                            help="root of the dataset")
    parser.add_argument("--resume", type=str, default="",
                                            help="path to the latest checkpoint (default: none)")

    parser.add_argument("--dsp", type=int, default=35,
                                            help="dimensions of the simulation parameters (default: 35)")
    parser.add_argument("--dspe", type=int, default=512,
                                            help="dimensions of the simulation parameters' encode (default: 512)")
    parser.add_argument("--ch", type=int, default=1,
                                            help="channel multiplier (default: 1)")

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
    parser.add_argument("--epochs", type=int, default=10,
                                            help="number of epochs to train (default: 10)")

    parser.add_argument("--log-every", type=int, default=10,
                                            help="log training status every given number of batches (default: 10)")
    parser.add_argument("--check-every", type=int, default=20,
                                            help="save checkpoint every given number of epochs (default: 20)")

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

    # data loader
    train_dataset = YeastDataset(
            root=args.root,
            train=True,
            transform=transforms.Compose([Normalize(), ToTensor()]))

    test_dataset = YeastDataset(
            root=args.root,
            train=False,
            transform=transforms.Compose([Normalize(), ToTensor()]))

    kwargs = {"num_workers": 4, "pin_memory": True} if args.cuda else {}
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                                        shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                                     shuffle=True, **kwargs)

    # model
    def weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def add_sn(m):
        for name, c in m.named_children():
            m.add_module(name, add_sn(c))
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            return nn.utils.spectral_norm(m, eps=1e-4)
        else:
            return m

    g_model = Generator(args.dsp, args.dspe, args.ch)
    g_model.apply(weights_init)
    # if args.sn:
    #     g_model = add_sn(g_model)

    g_model.to(device)

    mse_criterion = nn.MSELoss(reduction='none')
    train_losses, test_losses = [], []
    d_losses, g_losses = [], []

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

    # main loop
    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        # training...
        g_model.train()
        train_loss = 0.

        C42_data, PF_C42a, sample_weight = ReadYeastDataset(self.root, train)

        # for 
        #     g_optimizer.zero_grad()
        #     fake_data = g_model(sparams)

        #     loss = (mse_criterion(image, fake_data) * sample_weight).mean()


        #     # mse loss
        #     if args.mse_loss:
        #         mse_loss = 
        #         loss += mse_loss

        #     loss.backward()
        #     g_optimizer.step()
        #     train_loss += loss.item() * len(sparams)

        #     # log training status
        #     if i % args.log_every == 0:
        #         print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
        #             epoch, i * len(sparams), len(train_loader.dataset),
        #             100. * i / len(train_loader),
        #             loss.item()))
        #         train_losses.append(loss.item())

        # print("====> Epoch: {} Average loss: {:.4f}".format(
        #     epoch, train_loss / len(train_loader.dataset)))

        # # testing...
        # g_model.eval()
        # test_loss = 0.
        # with torch.no_grad():
        #     for i, sample in enumerate(test_loader):
        #         image = sample["image"].to(device)
        #         sparams = sample["sparams"].to(device)
        #         vparams = sample["vparams"].to(device)
        #         fake_image = g_model(sparams, vparams)
        #         test_loss += mse_criterion(image, fake_image).item() * len(sparams)

        # test_losses.append(test_loss / len(test_loader.dataset))
        # print("====> Epoch: {} Test set loss: {:.4f}".format(
        #     epoch, test_losses[-1]))

        # # saving...
        # if epoch % args.check_every == 0:
        #     print("=> saving checkpoint at epoch {}".format(epoch))
        #     if args.gan_loss != "none":
        #         torch.save({"epoch": epoch + 1,
        #                                 "g_model_state_dict": g_model.state_dict(),
        #                                 "g_optimizer_state_dict": g_optimizer.state_dict(),
        #                                 "d_model_state_dict": d_model.state_dict(),
        #                                 "d_optimizer_state_dict": d_optimizer.state_dict(),
        #                                 "d_losses": d_losses,
        #                                 "g_losses": g_losses,
        #                                 "train_losses": train_losses,
        #                                 "test_losses": test_losses},
        #                              os.path.join(args.root, "model_" + str(args.mse_loss) + "_" + \
        #                                                         args.perc_loss + "_" + str(args.gan_loss) + "_" + \
        #                                                         str(epoch) + ".pth.tar"))
        #     else:
        #         torch.save({"epoch": epoch + 1,
        #                                 "g_model_state_dict": g_model.state_dict(),
        #                                 "g_optimizer_state_dict": g_optimizer.state_dict(),
        #                                 "train_losses": train_losses,
        #                                 "test_losses": test_losses},
        #                              os.path.join(args.root, "model_" + str(args.mse_loss) + "_" + \
        #                                                         args.perc_loss + "_" + str(args.gan_loss) + "_" + \
        #                                                         str(epoch) + ".pth.tar"))

        #     torch.save(g_model.state_dict(),
        #                          os.path.join(args.root, "model_" + str(args.mse_loss) + "_" + \
        #                                                     args.perc_loss + "_" + str(args.gan_loss) + "_" + \
        #                                                     str(epoch) + ".pth"))

if __name__ == "__main__":
    main(parse_args())
