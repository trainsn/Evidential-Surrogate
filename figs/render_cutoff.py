import numpy as np
import matplotlib.pyplot as plt

# Load the data files
dropout_psnrs = np.load('dropout_cutoff_uncertainty_psnrs.npy')
ensemble_psnrs = np.load('ensemble_cutoff_uncertainty_psnrs.npy')
evidential_psnrs = np.load('evidential_cutoff_uncertainty_psnrs.npy')

# Generate x values
x_values = np.linspace(0, 1, 100, endpoint=False)

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(x_values, dropout_psnrs, label='Dropout')
plt.plot(x_values, ensemble_psnrs, label='Ensembles')
plt.plot(x_values, evidential_psnrs, label='Evidential')

plt.xlabel('Confidence Level', fontsize=20)
plt.ylabel('PSNR Values', fontsize=20)
plt.legend(fontsize=20)  
plt.grid(True)

# Set the font sizes of the axis tick labels
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.savefig('cutoff_uncertainty_psnrs.png')