import numpy as np
import matplotlib.pyplot as plt

# Load the data files
activebase_psnrs = np.load('active0_ret_uncertainty_psnrs.npy')[::-1]
active25_psnrs = np.load('active25_ret_uncertainty_psnrs.npy')[::-1]
active50_psnrs = np.load('active50_ret_uncertainty_psnrs.npy')[::-1]
active100_psnrs = np.load('active100_ret_uncertainty_psnrs.npy')[::-1]
singleloop_psnrs = np.load('singleloop_ret_uncertainty_psnrs.npy')[::-1]

# Generate x values
x_values = np.linspace(0, 100, 31, endpoint=True)[1:]

# Plotting the data
plt.figure(figsize=(12, 8))
plt.plot(x_values, activebase_psnrs, label='Active: Proximity')
plt.plot(x_values, active25_psnrs, label='Active: U+P ($\Phi=25$)')
plt.plot(x_values, active50_psnrs, label='Active: U+P ($\Phi=50$)')
plt.plot(x_values, active100_psnrs, label='Active: U+P ($\Phi=100$)')
plt.plot(x_values, singleloop_psnrs, label='Single Loop')

plt.xlabel('Uncertainty Value Percentile', fontsize=20)
plt.ylabel('PSNR Values', fontsize=20)
plt.legend(fontsize=20)
plt.grid(True)

# Set the font sizes of the axis tick labels
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Reverse the x-axis
plt.gca().invert_xaxis()

plt.savefig("retention_uncertainty_psnrs.png")