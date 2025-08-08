import numpy as np
from probpipe import NormalDistribution, BootstrapDistribution, EmpiricalDistribution
import matplotlib.pyplot as plt

# Config
np.random.seed(42)
d = 3  # Number of predictors
true_beta0 = 1.0
true_beta1 = np.array([2.0, 3.0, 1.5])
obs_noise = 1.0

# Observation model
obs_dist = NormalDistribution(mean=0.0, std_dev=obs_noise)




# Define a statistic function: for example, calculate the mean of a sample
def sample_mean(samples: np.ndarray) -> float:
    return np.mean(samples)


normal_dist = NormalDistribution(mean=10, std_dev=2)

bootstrap_dist = normal_dist.expectation(func=sample_mean, n_samples=1000, n_boot=500)


import matplotlib.pyplot as plt

plt.hist(bootstrap_dist.data, bins=30, alpha=0.7)
plt.title("Bootstrap distribution of sample mean")
plt.xlabel("Sample mean")
plt.ylabel("Frequency")
plt.show()


bootstrap_dist.data.std()


normal_dist = EmpiricalDistribution.from_samples(obs_dist.sample(1000))
normal_dist

