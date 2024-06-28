import numpy as np
import matplotlib.pyplot as plt

# Load the data files
base_psnrs = np.load('activebase_ret_value_psnrs.npy')[::-1]
active_psnrs = np.load('active_ret_value_psnrs.npy')[::-1]
singleloop_psnrs = np.load('singleloop_ret_value_psnrs.npy')[::-1]

# Generate x values
x_values = np.linspace(0, 100, 51, endpoint=True)[1:]

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(x_values, base_psnrs, label='Active: Proximity')
plt.plot(x_values, active_psnrs, label='Active: Uncertainty + Proximity')
plt.plot(x_values, singleloop_psnrs, label='Single Loop')

plt.xlabel('Data Value Percentile', fontsize=20)
plt.ylabel('PSNR Values', fontsize=20)
plt.legend(fontsize=20)
plt.grid(True)

# Set the font sizes of the axis tick labels
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Reverse the x-axis
plt.gca().invert_xaxis()

plt.savefig("rentetion_datavalue_psnrs.png")