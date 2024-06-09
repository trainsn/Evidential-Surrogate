import numpy as np
import matplotlib.pyplot as plt

# Load the data files
base_psnrs = np.load('activebase_ret_value_psnrs.npy')
active_psnrs = np.load('active_ret_value_psnrs.npy')

# Generate x values
x_values = np.linspace(0, 1, 51, endpoint=True)[1:]

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(x_values, base_psnrs, label='Baseline')
plt.plot(x_values, active_psnrs, label='Active')

plt.title('Comparison of PSNRs')
plt.xlabel('Data Value Level')
plt.ylabel('PSNR Values')
plt.legend()
plt.grid(True)
plt.show()