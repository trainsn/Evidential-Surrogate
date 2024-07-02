import numpy as np
import matplotlib.pyplot as plt

import pdb

# Load the data from the uploaded files
activebase_data = np.load('evidential_active0_observed_conf.npy')
active25_data = np.load('evidential_active25_observed_conf.npy')
active50_data = np.load('evidential_active50_observed_conf.npy')
active100_data = np.load('evidential_active100_observed_conf.npy')
singleloop_data = np.load('evidential_observed_conf.npy')

# Define the x-axis
x_axis = np.linspace(0, 1, 40, endpoint=True)

calibration_err_activebase = np.abs(x_axis - activebase_data).mean()
calibration_err_active25 = np.abs(x_axis - active25_data).mean()
calibration_err_active50 = np.abs(x_axis - active50_data).mean()
calibration_err_active100 = np.abs(x_axis - active100_data).mean()
calibration_err_singleloop = np.abs(x_axis - singleloop_data).mean()


# Create the line plot
plt.figure(figsize=(10, 6))
plt.plot(x_axis, activebase_data, label=f'Active: Proximity, Error = {calibration_err_activebase:.4f}')
plt.plot(x_axis, active25_data, label=f'Active: U+P ($\Phi=25$), Error = {calibration_err_active25:.4f}')
plt.plot(x_axis, active50_data, label=f'Active: U+P ($\Phi=50$), Error = {calibration_err_active50:.4f}')
plt.plot(x_axis, active100_data, label=f'Active: U+P ($\Phi=100$), Error = {calibration_err_active100:.4f}')
plt.plot(x_axis, singleloop_data, label=f'Single Loop, Error = {calibration_err_singleloop:.4f}')
plt.plot(x_axis, x_axis, label='Ideal calibration', linestyle='--', color='gray')
plt.xlabel('Expected Confidence Level', fontsize=20)
plt.ylabel('Observed Confidence Level', fontsize=20)
plt.legend(fontsize=20)
plt.grid(True)

# Set the font sizes of the axis tick labels
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.savefig('observed_conf_active.png')