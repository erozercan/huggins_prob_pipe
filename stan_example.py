import numpy as np
from cmdstanpy import CmdStanModel
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(123)
N = 50
x = np.linspace(0, 10, N)
alpha_true = 1.0
beta_true = 2.5
sigma_true = 0.5
y = alpha_true + beta_true * x + np.random.normal(0, sigma_true, size=N)

# Compile Stan model
stan_model = CmdStanModel(stan_file='linear_regression.stan')

# Prepare data dictionary for Stan
data = {'N': N, 'x': x, 'y': y}

# Fit model with NUTS sampler
fit = stan_model.sample(data=data, chains=4, parallel_chains=4, iter_sampling=1000, iter_warmup=1000)

# Print summary
print(fit.summary())

# Extract posterior samples
posterior = fit.draws_pd()

# Plot posterior distributions (e.g., for alpha, beta, sigma)
posterior[['alpha', 'beta', 'sigma']].plot(kind='density')
plt.show()