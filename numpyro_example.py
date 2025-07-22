import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.random as random
import numpy as np

# Generate synthetic data
np.random.seed(123)
N = 50
x_data = np.linspace(0, 10, N)
alpha_true = 1.0
beta_true = 2.5
sigma_true = 0.5
y_data = alpha_true + beta_true * x_data + np.random.normal(0, sigma_true, size=N)

# Convert data to jax arrays
x_jax = jnp.array(x_data)
y_jax = jnp.array(y_data)

# Define model function (no `with` because NumPyro does not use context managers)
def model(x, y=None):
    alpha = numpyro.sample("alpha", dist.Normal(0, 10))
    beta = numpyro.sample("beta", dist.Normal(0, 10))
    sigma = numpyro.sample("sigma", dist.HalfNormal(5))
    mu = alpha + beta * x
    numpyro.sample("y_obs", dist.Normal(mu, sigma), obs=y)

# Run inference
rng_key = random.PRNGKey(0)
kernel = NUTS(model)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000)
mcmc.run(rng_key, x=x_jax, y=y_jax)
samples = mcmc.get_samples()

# Print summary
print(mcmc.print_summary())