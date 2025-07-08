import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
from probpipe import NormalDistribution

n_samples_raw = 100000
n_samples_boot = 100
n_boot = 100


# Distribution instance
dist = NormalDistribution(2, 1)

# Bootstrap distribution of mean
boot_mean = dist.expectation(np.mean, n_samples=n_samples_boot, n_boot=n_boot)
mean_samples = np.array(boot_mean.sample(1000))

# Bootstrap distribution of mean of squares
boot_m2 = dist.expectation(lambda x: x**2, n_samples=n_samples_boot, n_boot=n_boot)
m2_samples = np.array(boot_m2.sample(1000))

fig, axs = plt.subplots(1, 2, figsize=(14,5))

# Mean bootstrap distribution
axs[0].hist(mean_samples, bins=30, density=True, alpha=0.6, color='orange')

axs[0].set_title('Bootstrap Distribution of Mean')
axs[0].set_xlabel('Mean Estimate')
axs[0].set_ylabel('Density')
axs[0].legend()
axs[0].grid(True)

# Mean of squares bootstrap distribution
axs[1].hist(m2_samples, bins=30, density=True, alpha=0.6, color='skyblue')
axs[1].set_title('Bootstrap Distribution of Mean of Squares')
axs[1].set_xlabel('Mean of Squares Estimate')
axs[1].set_ylabel('Density')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()