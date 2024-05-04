#perform multiple regression using a FC neural network
import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
from keras.models import Model
from keras.layers import Input, Reshape
from tensorflow.keras.layers import Dense, Activation, Dropout
from keras.layers import Dropout

from keras.models import load_model

from yeast import *

import pdb

# rate_dropout = 0.3

def get_yeast_model(param_in, out_dim=400):
    x = Dense(1024)(param_in)
    x = Activation('relu')(x)
    # x = Dropout(rate_dropout)(x)
    x = Dense(800)(x)
    x = Activation('relu')(x)
    # x = Dropout(rate_dropout)(x)
    x = Dense(500)(x)
    x = Activation('relu')(x)
    model_out = Dense(out_dim)(x)
    M = Model(param_in, model_out)
    M.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
    return M, model_out

params, C42a_data, sample_weight = ReadYeastDataset()

#create the model
param_in = Input(shape=[35])
M, model_out = get_yeast_model(param_in)
M.summary()

hstry = M.fit(params.astype(np.float32), C42a_data.astype(np.float32), sample_weight=sample_weight.astype(np.float32).flatten(), epochs=5000, batch_size=32, validation_split=0.25)
M.save('new_MLP_yeast_35_1024_800_500_400_epochs5000_dropout_datasize_3000.h5')

train_loss_array = np.array(hstry.history['loss'])
train_acc_array = np.array(hstry.history['accuracy'])
val_loss_array = np.array(hstry.history['val_loss'])
val_acc_array = np.array(hstry.history['val_accuracy'])

train_loss_array.tofile("train_loss_array_5000_dp.raw")
train_acc_array.tofile("train_acc_array_5000_dp.raw")
val_loss_array.tofile("val_loss_array_5000_dp.raw")
val_acc_array.tofile("val_acc_array_5000_dp.raw")