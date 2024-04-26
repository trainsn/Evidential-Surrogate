# yeast simulation dataset

from __future__ import absolute_import, division, print_function

import os

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import pdb

def read_data_from_file(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            row = line.strip().split('\t')
            data.append([float(x) for x in row if x])
    return np.array(data)


def ReadYeastDataset(root, train=True):
    train_sets = range(1, 31)  # set1 to set30
    test_sets = range(31, 41)  # set31 to set40
    set_range = train_sets if train else test_sets

    params = []
    C42a_data = []
    PF_C42a = []

    # Load the data from files
    for i in set_range:
        set_dir = os.path.join(root, f'set{i}')
        params.append(read_data_from_file(os.path.join(set_dir, 'list_of_parameters')))
        C42_data.append(read_data_from_file(os.path.join(set_dir, 'C42_dat')))
        PF_C42a.append(read_data_from_file(os.path.join(set_dir, 'PF_C42a_set_of_50')))

    # Concatenate all data from each file type into single arrays
    params = np.concatenate(params, axis=0) if params else np.array([], dtype=float)
    C42_data = np.concatenate(C42_data, axis=0) if C42_data else np.array([], dtype=float)
    PF_C42a = np.concatenate(PF_C42a, axis=0) if PF_C42a else np.array([], dtype=float)

    samp_weight1 = np.ones_like(PF_C42a)
    samp_weight1[PF_C42a >= 0.35] = 3

    return torch.from_numpy(params).cuda(), torch.from_numpy(C42_data).cuda(), torch.from_numpy(samp_weight1).cuda()
