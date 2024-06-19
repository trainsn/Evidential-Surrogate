import numpy as np
import matplotlib.pyplot as plt

# Load the data from the uploaded files
ensemble_data = np.load('ensemble_observed_conf.npy')
evidential_data = np.load('evidential_observed_conf.npy')
dropout_data = np.load('dropout_observed_conf.npy')

# Define the x-axis
x_axis = np.linspace(0, 1, 40, endpoint=True)

calibration_err_ensemble = np.abs(x_axis - ensemble_data).mean()
calibration_err_evidential = np.abs(x_axis - evidential_data).mean()
calibration_err_dropout = np.abs(x_axis - dropout_data).mean()

# Check the shapes of the loaded arrays and the x-axis
ensemble_data.shape, evidential_data.shape, x_axis.shape

# Create the line plot
plt.figure(figsize=(10, 6))
plt.plot(x_axis, dropout_data, label=f'Dropout, Error = {calibration_err_dropout:.4f}')
plt.plot(x_axis, ensemble_data, label=f'Ensembles, Error = {calibration_err_ensemble:.4f}')
plt.plot(x_axis, evidential_data, label=f'Evidential, Error = {calibration_err_evidential:.4f}')
plt.plot(x_axis, x_axis, label='Ideal calibration', linestyle='--', color='gray')
plt.xlabel('Expected Confidence Level', fontsize=20)
plt.ylabel('Observed Confidence Level', fontsize=20)
plt.legend(fontsize=20)
plt.grid(True)

# Set the font sizes of the axis tick labels
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.savefig('observed_conf.png')