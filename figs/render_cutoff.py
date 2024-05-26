import numpy as np
import matplotlib.pyplot as plt

# Load the data files
dropout_psnrs = np.load('dropout_cutoff_psnrs.npy')
ensemble_psnrs = np.load('ensemble_cutoff_psnrs.npy')
evidential_psnrs = np.load('evidential_cutoff_psnrs.npy')

# Generate x values
x_values = np.linspace(0, 1, 100, endpoint=False)

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(x_values, dropout_psnrs, label='Dropout')
plt.plot(x_values, ensemble_psnrs, label='Ensemble')
plt.plot(x_values, evidential_psnrs, label='Evidential')

plt.title('Comparison of PSNRs')
plt.xlabel('Confidence Level')
plt.ylabel('PSNR Values')
plt.legend()
plt.grid(True)
plt.show()