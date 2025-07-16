import numpy as np
from probpipe import simple_linreg, NormalDistribution, HalfNorm, CompositePrior, run_chain_nuts
import matplotlib.pyplot as plt

# Config
np.random.seed(42)
d = 3  # Number of predictors
true_beta0 = 1.0
true_beta1 = np.array([2.0, 3.0, 1.5])
obs_noise = 1.0

# Initial prior
prior0_std = 1.0
prior1_std = 2.0
prior_dist = CompositePrior([
    (slice(0, 1), HalfNorm(prior0_std)),
    (slice(1, None), HalfNorm(prior1_std)),
])

# Observation model
obs_dist = NormalDistribution(mean=0.0, std_dev=obs_noise)

# Iterative updating over new data batches
num_iterations = 5
batch_size = 100
jitter = 1e-6

all_posteriors = []
current_prior = prior_dist

for i in range(num_iterations):
    print(f"\n--- Iteration {i+1} ---")

    # Generate new data batch
    X_batch = np.random.randn(batch_size, d)
    y_batch = true_beta0 + X_batch @ true_beta1 + np.random.normal(0, obs_noise, size=batch_size)
    data_batch = np.column_stack((X_batch, y_batch))

    # Run inference using current prior
    posterior = simple_linreg(
        data=data_batch,
        prior_dist=current_prior,
        obs_dist=obs_dist,
        num_samples=1000,
        step_size=0.01,
        sampler=run_chain_nuts,
    )

    posterior.summary()
    all_posteriors.append(posterior)

    # Update prior for next batch
    post_mean = posterior.mean()
    post_std = posterior.std()
    current_prior = NormalDistribution(mean=post_mean, std_dev=post_std + jitter)


posterior_means = [p.mean() for p in all_posteriors]
posterior_means = np.stack(posterior_means)  # shape: (iterations, d+1)


for j in range(posterior_means.shape[1]):
    plt.plot(range(1, num_iterations + 1), posterior_means[:, j], label=f"param {j}")
plt.xlabel("Iteration")
plt.ylabel("Posterior Mean")
plt.title("Posterior Parameter Estimates Over Time")
plt.legend()
plt.show()
