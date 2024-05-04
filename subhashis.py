import numpy as np
import os
from sklearn.metrics import mean_squared_error

import keras
from keras.models import load_model
import keras.backend as K

from yeast import *

import pdb

#class to enable uncertainty quantification with NN
class KerasDropoutPrediction:
    def __init__(self, model):
        self.model = model

    def predict(self, x, n_iter=10):
        result = []
        for _ in range(n_iter):
            # Run the model with dropout by setting training=True
            prediction = self.model(x, training=True)
            result.append(prediction)
        result = np.array(result).reshape(n_iter, x.shape[0], -1)  # Adjust the reshape parameters as needed
        return result

#the predictor function
#input : KerasDropoutPrediction object, and (35,) size input numpy array 
#output: Mean predicted protein values(400,), Std deviation of the predicted values (400,) 
def dropout_predictor(m_object, param_in):
    y_pred = m_object.predict(param_in, n_iter=100)
    y_pred_mean = y_pred.mean(axis=0)
    y_pred_std = y_pred.std(axis=0)
    return y_pred_mean, y_pred_std

#load the trained NN surrogate model
M3 = load_model(os.path.join("models", "./new_MLP_yeast_35_1024_800_500_400_epochs5000_dropout_datasize_3000_split_0_1.h5"))

#printout the model configuration
M3.summary()

#create the objects for dropout predictor
M3_dp_predictor = KerasDropoutPrediction(M3)

def main():
    #dummy test case
    # input_parameter = np.array([ 0.799611,  0.191463,  0.308464,  0.207551,  0.074747,  0.949241,  0.580422,  
    # 	0.791716, -0.473602, -0.025531, -0.314472, 0.862159, -0.239092, -0.712813,
    #  -0.225801, -0.334405, -0.609486, -0.518853, -0.503734, -0.897176,  0.979674,
    #   0.017872,  0.943028, -0.356691,  0.656346, -0.217127,  0.782005,  0.143829,
    #  -0.535368, -0.205101,  0.626929,  0.357102,  0.813448, -0.859384,  0.635532])

    params, C42a_data, _ = ReadYeastDataset()
    train_split = np.load('train_split.npy')
    test_params, test_C42a_data = params[~train_split], C42a_data[~train_split]
    
    #actual call to the predictor
    mean_outcome, std_outcome = dropout_predictor(M3_dp_predictor, test_params)

    #sanity check only
    print("MEAN:")
    print(mean_outcome)
    print("STD:")
    print(std_outcome)

    mse = mean_squared_error(test_C42a_data, mean_outcome)
    print("Mean Squared Error:", mse)

    pdb.set_trace()

if __name__ == "__main__":
    main()