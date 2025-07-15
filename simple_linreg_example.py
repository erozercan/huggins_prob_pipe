import numpy as np
from probpipe import simple_linreg, NormalDistribution

from probpipe import Distribution, NormalDistribution, HalfNorm, EmpiricalDistribution, CompositePrior
from probpipe import nuts_sample, run_chain_nuts



# Simulated data
np.random.seed(42)
N, d = 100, 3
X = np.random.randn(N, d)
true_beta0 = 1.0
true_beta1 = np.array([2.0, 3.0, 1.5])
y = true_beta0 + X @ true_beta1 + np.random.normal(0, 1.0, size=N)
data = np.column_stack((X, y))

# priors
prior0_std = 1.0
prior1_std = 2.0

prior_dist = CompositePrior([
    (slice(0, 1), HalfNorm(prior0_std)),
    (slice(1, None), HalfNorm(prior1_std)),
])


obs_dist = NormalDistribution(mean=0.0, std_dev=1.0)



first_posterior = simple_linreg(
    data,
    prior_dist,
    obs_dist,
    num_samples=1000,
    step_size=0.01,
    sampler=run_chain_nuts,
)

first_posterior.summary()

# Updates 1 
post_mean = first_posterior.mean()
post_std = first_posterior.std()


jitter = 1e-6 # Add small jitter to std to avoid zero std_dev

updated_prior_dist = NormalDistribution(mean=post_mean, std_dev=post_std + jitter)

# Run inference again with the updated prior
second_posterior = simple_linreg(
    data,
    updated_prior_dist,
    obs_dist,
    num_samples=1000,
    step_size=0.01,
    sampler=run_chain_nuts,
)



second_posterior.summary()