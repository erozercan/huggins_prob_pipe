from abc import ABC, abstractmethod
from typing import Callable
import jax.numpy as jnp
from jax import jit, random, vmap, grad


class Distribution(ABC):
    @abstractmethod
    def sample(self, key: random.PRNGKey, n_samples: int) -> jnp.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def expectation(self, func: Callable[[jnp.ndarray], jnp.ndarray], n_samples: int, key: random.PRNGKey) -> 'Distribution':
        raise NotImplementedError()



    

class NormalDistribution(Distribution):
    def __init__(self, mean, std_dev):
        self.mean = jnp.array(mean)
        self.std_dev = jnp.array(std_dev)

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        var = self.std_dev ** 2
        log_scale = jnp.log(self.std_dev)
        logp = -0.5 * jnp.log(2 * jnp.pi) - log_scale - 0.5 * ((x - self.mean) ** 2) / var
        # Sum over last axis if x is multidimensional (e.g., batch)
        return jnp.sum(logp, axis=-1)

    def sample(self, key: random.PRNGKey, n_samples: int = 1) -> jnp.ndarray:
        shape = self.mean.shape
        samples = self.mean + self.std_dev * random.normal(key, shape=(n_samples,) + shape)
        if n_samples == 1:
            return samples[0]
        return samples

    def expectation(self, func, n_samples, key):
        key_sample, _ = random.split(key)
        samples = self.sample(key_sample, n_samples)
        evaluated = func(samples)
        mean_val = jnp.mean(evaluated)
        std_val = jnp.std(evaluated)
        return NormalDistribution(mean_val, std_val)

    def __repr__(self):
        return f"NormalDistribution(mean={self.mean}, std_dev={self.std_dev})"
    


class HalfNorm(Distribution):
    """
    Half-Normal distribution with scale parameter `std_dev`.

    PDF:
        f(x) = (sqrt(2) / (sigma sqrt(pi))) * exp(-x^2 / (2sigma²))     for x geq 0
        f(x) = 0                                   for x < 0

    Log PDF:
        log f(x) = 0.5 * log(2/pi) - log(sigma) - x^2 / (2sigma^2)    for x geq 0
        log f(x) = -infty                                      for x < 0
    """

    def __init__(self, std_dev: float):
        self.std_dev = jnp.array(std_dev)

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        log_pdf = jnp.where(
            x >= 0,
            0.5 * jnp.log(2 / jnp.pi) - jnp.log(self.std_dev) - (x ** 2) / (2 * self.std_dev ** 2),
            -jnp.inf,
        )
        return jnp.sum(log_pdf, axis=-1)

    def sample(self, key: random.PRNGKey, n_samples: int = 1) -> jnp.ndarray:
        # Sample from Normal(0, std_dev), then take absolute value for Half-Normal
        samples = jnp.abs(self.std_dev * random.normal(key, shape=(n_samples,)))
        if n_samples == 1:
            return samples[0]
        return samples

    def expectation(self, func, n_samples: int, key: random.PRNGKey):
        # Estimate expectation and std by sampling
        key_sample, _ = random.split(key)
        samples = self.sample(key_sample, n_samples)
        evaluated = func(samples)
        mean_val = jnp.mean(evaluated)
        std_val = jnp.std(evaluated)
        # Return as a NormalDistribution 
        return NormalDistribution(mean_val, std_val)

    def __repr__(self):
        return f"HalfNorm(std_dev={self.std_dev})"


class MultiNorm(Distribution):
    """
    Multivariate Gaussian Distribution (x ~ N(m, C)) for d-dimensional x.

    Parameters:
        m: Mean vector, shape (d,)
        C: Covariance matrix, shape (d, d), positive semidefinite
    """

    def __init__(self, m, C):
        self.m = jnp.array(m)
        self.C = jnp.array(C)
        self.dim = self.m.shape[0]

        self.L = jnp.linalg.cholesky(self.C)
        self.log_det_cov = 2.0 * jnp.sum(jnp.log(jnp.diag(self.L)))
        self._inv_cov = jax.scipy.linalg.cho_solve((self.L, True), jnp.eye(self.dim))
        self._norm_const = -0.5 * self.dim * jnp.log(2 * jnp.pi) - 0.5 * self.log_det_cov

    def sample(self, key: random.PRNGKey, n_samples: int = 1) -> jnp.ndarray:
        z = random.normal(key, shape=(n_samples, self.dim))
        samples = self.m + (self.L @ z.T).T
        if n_samples == 1:
            return samples[0]
        return samples

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.atleast_2d(x)
        diff = x - self.m
        maha = jnp.sum(diff @ self._inv_cov * diff, axis=1)
        return self._norm_const - 0.5 * maha

    def prob(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(self.log_prob(x))

    def expectation(self, func: Callable[[jnp.ndarray], jnp.ndarray], n_samples: int, key: random.PRNGKey) -> 'Distribution':
        raise NotImplementedError()

    def __repr__(self):
        return f"MultiNorm(mean={self.m}, cov=\n{self.C})"





class EmpiricalDistribution(Distribution):
    def __init__(self, samples: jnp.ndarray):
        """
        samples: array of shape (num_samples, *param_shape)
        """
        self.samples = samples
        self.num_samples = samples.shape[0]

    def mean(self):
        return jnp.mean(self.samples, axis=0)

    def std(self):
        return jnp.std(self.samples, axis=0)

    def quantile(self, q):
        # q can be scalar or array of quantiles between 0 and 1
        return jnp.quantile(self.samples, q, axis=0)

    def expectation(self, func):
        vals = func(self.samples)
        return jnp.mean(vals, axis=0)

    def summary(self):
        mean = self.mean()
        std = self.std()
        print(f"Mean:\n{mean}")
        print(f"Std deviation:\n{std}")

    def log_prob(self, params): 
        raise NotImplementedError()

    def sample(self, key, n_samples):  
        raise NotImplementedError()
    
    def expectation(self, func: Callable[[jnp.ndarray], jnp.ndarray], n_samples: int, key: random.PRNGKey) -> 'Distribution':
        raise NotImplementedError()




class BootstrapDistribution(Distribution):  ## Todo
    def __init__(self, data: jnp.ndarray, sample_size: int = None, axis: int = 0):
        self.data = data
        self.axis = axis
        self.sample_size = sample_size or data.shape[axis]

    def sample(self, key: random.PRNGKey, n_samples: int) -> jnp.ndarray:
        data_len = self.data.shape[self.axis]
        keys = random.split(key, n_samples)

        def single_bootstrap(k):
            idxs = random.randint(k, (self.sample_size,), 0, data_len)
            sample = jnp.take(self.data, idxs, axis=self.axis)
            return jnp.mean(sample)

        return vmap(single_bootstrap)(keys)

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError("BootstrapDistribution does not implement log_prob")

    def expectation(self, func: Callable[[jnp.ndarray], jnp.ndarray], n_samples: int, key: random.PRNGKey) -> 'BootstrapDistribution':
        samples = self.sample(key, n_samples)
        evaluated = func(samples)
        return BootstrapDistribution(evaluated)

    def __repr__(self):
        return f"BootstrapDistribution(data_shape={self.data.shape}, sample_size={self.sample_size}, axis={self.axis})"
    


class CompositePrior(Distribution):
    def __init__(self, components):
        """
        components: list of (slice, Distribution) tuples
        Example: [(slice(0,1), NormalDistribution), (slice(1,None), LaplaceDistribution)]
        """
        self.components = components

    def log_prob(self, params):
        log_prob_sum = 0.0
        for slc, dist in self.components:
            log_prob_sum += dist.log_prob(params[slc])
        return log_prob_sum

    def sample(self, key, n_samples):
        samples_list = []
        keys = random.split(key, len(self.components))
        for (slc, dist), k in zip(self.components, keys):
            samples_list.append(dist.sample(k, n_samples))
        # Concatenate along last axis to form shape (n_samples, total_dim)
        return jnp.concatenate(samples_list, axis=-1)
    
    def expectation(self, func: Callable[[jnp.ndarray], jnp.ndarray], n_samples: int, key: random.PRNGKey) -> 'Distribution':
        raise NotImplementedError()
    


