import jax
import jax.numpy as jnp
from jax import random, lax
import blackjax
import numpy as np
from tqdm import trange
from typing import Optional, Callable
from .distributions import EmpiricalDistribution


def simple_linreg(
    data: np.ndarray,
    prior_dist,
    obs_dist,
    num_samples: int = 1000,
    step_size: float = 0.01,
    seed: int = 0,
    sampler: Optional[Callable] = None,
    sampler_kwargs: Optional[dict] = None,
):
    """
    Bayesian linear regression model:

        y_i = beta0 + x_i^T * beta1 + epsilon_i,   for i = 1, ..., n

    where:
        - y_i: observed response (scalar)
        - x_i: predictor vector of length d
        - beta0: scalar intercept parameter
        - beta1: vector of regression coefficients of length d
        - epsilon_i: noise term ~ Normal(0, sigma^2), independent across i

    Model assumptions:
    1. Likelihood:
        y_i | x_i, beta0, beta1, sigma ~ Normal(mu_i, sigma^2)
        with mu_i = beta0 + dot(x_i, beta1)

    2. Prior distributions:
        beta0 ~ prior_dist on intercept (e.g., Normal, Half-Normal)
        beta1 ~ prior_dist on slopes (can be joint or factorized)
        sigma assumed fixed and known or specified separately

    3. Posterior distribution:
        p(beta0, beta1 | data) ∝
            [product over i=1 to n of p(y_i | x_i, beta0, beta1, sigma)] x p(beta0) x p(beta1)

    Inputs:
        - data: numpy array with predictors and response concatenated (shape: n x (d+1))
        - prior_dist: prior distribution object with log_prob method over parameter vector (beta0 and beta1)
        - obs_dist: observation noise distribution object (e.g., NormalDistribution)
        - num_samples: number of MCMC samples
        - step_size: step size for sampler
        - seed: PRNG seed
        - sampler: sampling function or None to use default

    Output:
        - EmpiricalDistribution object containing posterior samples over parameters
    """   


    key = random.PRNGKey(seed)

    X = jnp.array(data[:, :-1])
    y = jnp.array(data[:, -1])
    n, d = X.shape

    def log_posterior(params):
        betas = params  # treat entire vector including intercept as one vector
        y_pred = betas[0] + X @ betas[1:]
        log_prior = prior_dist.log_prob(betas)
        residuals = y - y_pred
        log_lik = obs_dist.log_prob(residuals)
        return log_prior + log_lik

    init_params = jnp.zeros(d + 1)

    if sampler is None:
        samples, accept_rates = run_chain_nuts(key, init_params, log_posterior, num_samples, step_size)
    elif callable(sampler):
        if sampler_kwargs is None:
            sampler_kwargs = {}
        samples, accept_rates = sampler(
            key, init_params, log_posterior, num_samples, step_size, **sampler_kwargs
        )
    else:
        raise ValueError("sampler must be None or a callable")

    mean_accept_rate = jnp.mean(accept_rates)
    print(f"Mean acceptance rate: {mean_accept_rate:.3f}")

    return EmpiricalDistribution(samples)



