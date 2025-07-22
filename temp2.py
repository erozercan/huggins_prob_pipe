import jax.numpy as jnp
import pymc as pm
import pytensor.tensor as pt
import numpy as np
import numpyro.distributions as dist

# Suppose mean and std are JAX arrays
mu_jax = jnp.array(1.0)
sigma_jax = jnp.array(2.0)

# Convert JAX arrays to NumPy arrays
mu_np = np.array(mu_jax)
sigma_np = np.array(sigma_jax)

# Convert NumPy arrays to pytensor symbolic variables
mu_pt = pt.as_tensor_variable(mu_np)
sigma_pt = pt.as_tensor_variable(sigma_np)

with pm.Model() as model:
    # Now create a PyMC Normal distribution with symbolic parameters
    x = pm.Normal("x", mu=mu_pt, sigma=sigma_pt)




d1 = dist.Normal(0, 10)