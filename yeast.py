# yeast simulation dataset

from __future__ import absolute_import, division, print_function

import os

import numpy as np

import pdb

def read_data_from_file(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            row = line.strip().split('\t')
            data.append([float(x) for x in row if x])
    return np.array(data)


def ReadYeastDataset():
    params = []
    C42a_dat = []
    PF_C42a = []

    set_range = range(1, 41)
    # Load the data from files
    for i in set_range:
        set_dir = os.path.join('/fs/ess/PAS0027/yeast_polarization_data/yeast_polarization/all_rerun_data/rerun', f'set{i}')
        params.append(read_data_from_file(os.path.join(set_dir, 'list_of_parameters')))
        C42a_dat.append(read_data_from_file(os.path.join(set_dir, 'C42a_dat')))
        PF_C42a.append(read_data_from_file(os.path.join(set_dir, 'PF_C42a_set_of_50')))

    set_range = range(1, 11)
    # Load the data from files
    for i in set_range:
        set_dir = os.path.join('/fs/ess/PAS0027/yeast_polarization_data/yeast_polarization/all_rerun_data/rerun_imp_sample', f'set{i}')
        params.append(read_data_from_file(os.path.join(set_dir, 'list_of_parameters')))
        C42a_dat.append(read_data_from_file(os.path.join(set_dir, 'C42a_dat')))
        PF_C42a.append(read_data_from_file(os.path.join(set_dir, 'PF_C42a_set_of_100')))

    # Concatenate all data from each file type into single arrays
    params = np.concatenate(params, axis=0) if params else np.array([], dtype=float)
    # params_slice = params[:, np.r_[0:1, 4:35]]
    params_slice = params[:, np.r_[0:25, 32:35]]
    C42a_dat = np.concatenate(C42a_dat, axis=0) if C42a_dat else np.array([], dtype=float)
    PF_C42a = np.concatenate(PF_C42a, axis=0) if PF_C42a else np.array([], dtype=float)

    dmin, dmax = C42a_dat.min(), C42a_dat.max()
    C42a_dat_scaled = 2 * (C42a_dat - dmin) / (dmax - dmin) - 1

    samp_weight1 = np.where(PF_C42a >= 0.35, 3, 1)

    return params_slice, C42a_dat_scaled, samp_weight1
