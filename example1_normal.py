from probpipe import NormalDistribution
import numpy as np

import matplotlib.pyplot as plt

# Create the NormalDistribution instance
dist = NormalDistribution(mean=0, std_dev=1)

# Define the statistic function: median (or you can use np.mean, np.var etc.)
stat_func = np.mean

# Get bootstrap empirical distribution of the statistic
boot_dist = dist.expectation(stat_func, n_samples=1000, n_boot=500)

# Draw samples from the bootstrap distribution to visualize
bootstrap_samples = boot_dist.sample(1000)

# Plot histogram of the bootstrap distribution
plt.hist(bootstrap_samples, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel(f'Bootstrap distribution of {stat_func.__name__}(X)')
plt.ylabel('Density')
plt.title(f'Bootstrap empirical distribution of {stat_func.__name__} estimates')
plt.grid(True)

# Mark the bootstrap mean on the plot
plt.axvline(np.mean(bootstrap_samples), color='red', linestyle='--', label='Bootstrap mean')
plt.legend()

plt.show()




