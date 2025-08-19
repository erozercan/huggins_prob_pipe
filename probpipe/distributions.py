from typing import Generic, TypeVar, Callable, Optional, Any
from abc import ABC, abstractmethod
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from numpy.typing import NDArray
from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist
from jax import random
import jax.numpy as jnp

T = TypeVar("T")

class Distribution(Generic[T], ABC):
    @abstractmethod
    def sample(self, n_samples: int) -> pt.TensorVariable:
        """
        Sample n_samples items from the distribution.
        Returns a pytensor tensor of sampled raw data.
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    @abstractmethod
    def expectation(self, func: Callable[[pt.TensorVariable], pt.TensorVariable]) -> 'Distribution[T]':
        """
        Return a distribution representing the expectation of func under this distribution.
        """
        raise NotImplementedError("This method should be implemented by subclasses")
    

class Distribution_Empirical(Distribution[T]):
    """
    Abstract base class for empirical (sample-based) distributions.
    Empirical distributions are defined by observed or simulated samples
    rather than a closed-form density.
    """
    @abstractmethod
    def approx_log_prob(self, data: pt.TensorVariable) -> pt.TensorVariable:
        """
        Approximate log probability of `data` under the empirical distribution.
        This might be done via kernel density estimation- KDELogPDF() class.
        """
        raise NotImplementedError("This method should be implemented by subclasses")
    

    
class BootstrapDistribution(Distribution_Empirical[float]):
    def __init__(
        self,
        data: np.ndarray,
        sample_size: Optional[int | float] = None,
        axis: int = 0
    ):
        # Store data as numpy array (bootstrap is empirical, numpy based)
        self.data = np.array(data)
        self.axis = axis

        if sample_size is None:
            self.sample_size = self.data.shape[axis] if self.data.ndim > 0 else len(self.data)
        elif isinstance(sample_size, float) and 0 < sample_size < 1:
            self.sample_size = int(sample_size * (self.data.shape[axis] if self.data.ndim > 0 else len(self.data)))
        elif isinstance(sample_size, int):
            self.sample_size = sample_size
        else:
            raise ValueError("sample_size must be None, float between 0 and 1, or int >= 1")

    def sample(self, n_samples: int) -> np.ndarray:
        bootstrap_samples = []
        data_len = self.data.shape[self.axis] if self.data.ndim > 0 else len(self.data)

        for _ in range(n_samples):
            indices = np.random.randint(0, data_len, size=self.sample_size)
            if self.data.ndim > 0:
                resampled = np.take(self.data, indices, axis=self.axis)
            else:
                resampled = self.data[indices]
            bootstrap_stat = np.mean(resampled)
            bootstrap_samples.append(bootstrap_stat)
        return np.array(bootstrap_samples)

    def approx_log_prob(self, data: float) -> float:
        data_min = np.min(self.data)
        data_max = np.max(self.data)
        if data_min <= data <= data_max:
            return -np.log(len(self.data))  # approximate uniform log prob over empirical data
        else:
            return float('-inf')

    def expectation(self, func: Callable[[float], float]) -> 'Distribution':
        raise NotImplementedError("bootstrap distribution expectation not implemented")
    
    
class EmpiricalDistribution(Distribution_Empirical[T]):
    def __init__(self, samples: NDArray):
        self.samples = np.asarray(samples)
        self.num_samples = self.samples.shape[0]
        self.samples_var = pt.constant(self.samples)
        #self.kde = KDE(samples) #or KDE(samples)

        #self.logpdf_op = KDE(self.kde)

    def sample(self, n_samples: int) -> pt.TensorVariable:
        indices = np.random.choice(self.num_samples, size=n_samples, replace=True)
        sampled = self.samples[indices]
        return pt.constant(sampled)

    def approx_log_prob(self, data):
        # data can be a PyTensor variable
        raise NotImplementedError
    
    def expectation(
        self,
        func: Callable[[pt.TensorVariable], pt.TensorVariable],
        n_boot: int = 1000,
        sample_size: int | None = None  # optional size for bootstrap resample, defaults to empirical size
    ) -> BootstrapDistribution:
        """
        Estimate the distribution of func(X) where X ~ EmpiricalDistribution by bootstrap.

        Parameters
        ----------
        func : Callable[[pt.TensorVariable], pt.TensorVariable]
            Function to apply on bootstrap resampled samples.
        n_boot : int
            Number of bootstrap iterations.
        sample_size : int or None
            Size of each bootstrap sample (defaults to original num_samples).

        Returns
        -------
        BootstrapDistribution
            Distribution of bootstrap estimates of func(X).
        """
        if sample_size is None:
            sample_size = self.num_samples

        estimates = []
        for _ in range(n_boot):
            indices = np.random.choice(self.num_samples, size=sample_size, replace=True)
            resampled = self.samples[indices]

            # Apply func on pytensor constant wrapping bootstrap resample,
            # then evaluate the computation to get a numpy scalar
            stat = func(pt.constant(resampled)).eval()

            # Reduce to scalar if numpy array is returned
            if isinstance(stat, np.ndarray):
                stat = np.mean(stat)
            estimates.append(stat)

        return BootstrapDistribution(np.array(estimates))
    
class JointEmpiricalDistribution(Distribution_Empirical[T]):
    """
    joint_post = JointEmpiricalDistribution(samples)

    # Sample from joint
    samples = joint_post.sample(100)

    # Get marginal of alpha
    alpha_marginal = joint_post.marginal("alpha")

    # Estimate E[alpha + beta[0]]
    def func(s): return s["alpha"] + s["beta"][:, 0]
    expectation = joint_post.expectation(func)
    """
    def __init__(self, samples: dict[str, NDArray]):
        """
        samples: dict mapping variable names to arrays of shape (n_samples, ...) 
        Each array must have the same number of samples (first dim).
        """
        self.samples = {k: np.asarray(v) for k, v in samples.items()}
        self.num_samples = next(iter(self.samples.values())).shape[0]

        # Ensure all samples are aligned
        for v in self.samples.values():
            #print(f"v.shape is {v.shape}")
            #print(f"v.shape[0] is {v.shape[0]}")
            #print(f"self.num_samples is {self.num_samples}")
            assert v.shape[0] == self.num_samples, "Mismatched sample sizes"

    def sample(self, n_samples: int) -> dict[str, NDArray]:
        indices = np.random.choice(self.num_samples, size=n_samples, replace=True)
        return {k: v[indices] for k, v in self.samples.items()}
    
    def approx_log_prob(self, data):
        # data can be a PyTensor variable
        raise NotImplementedError

    def marginal(self, var_name: str) -> EmpiricalDistribution:
        return EmpiricalDistribution(self.samples[var_name])
    
    def summary(self):
        print("Posterior summary:")
        for name, values in self.samples.items():
            values = np.asarray(values)
            mean = values.mean(axis=0)
            std = values.std(axis=0)
            print(f"  {name}: mean={mean}, std={std}")

    def conditional(self, given: dict[str, float], bandwidth: float = 1.0) -> dict[str, gaussian_kde]:
        """
        Returns conditional densities (via KDE) of remaining variables given fixed values for some.

        Note: Assumes all variables are numeric. This is an approximation using KDE.
        """
        
        # Build data matrix of shape (n_samples, n_dims)
        data_matrix = np.column_stack([self.samples[k] for k in self.samples])
        var_names = list(self.samples.keys())
        idx_given = [var_names.index(k) for k in given]
        idx_rest = [i for i in range(len(var_names)) if var_names[i] not in given]

        given_vals = np.array([given[k] for k in given]).reshape(1, -1)
        given_samples = data_matrix[:, idx_given]

        # Find weights using RBF kernel centered at given_vals
        distances = cdist(given_vals, given_samples).flatten()
        weights = np.exp(- (distances / bandwidth) ** 2)
        weights /= np.sum(weights)

        cond_kdes = {}
        for i in idx_rest:
            var_name = var_names[i]
            var_samples = data_matrix[:, i]
            cond_kdes[var_name] = gaussian_kde(var_samples, weights=weights)

        return cond_kdes  # Use `.evaluate(x)` to get pdfs

    #should this return an Empirical Distribution?
    def expectation(self, func: Callable[[dict[str, NDArray]], NDArray]) -> float:
        return np.mean(func(self.samples))


class Distribution_Density(Distribution[T]):
    """
    Abstract base class for distributions that admit a (log-)density.

    Subclasses represent *parametric* distributions with densities; e.g.,
    Normal, MultivariateNormal, Gamma, Dirichlet, etc.
    """

    @abstractmethod
    def log_prob(self, data: pt.TensorVariable) -> pt.TensorVariable:
        """
        Compute log p(data) under this distribution.
        `data` should broadcast to (*, *event_shape).
        Returns a tensor of log-prob values reduced over event dims
        (i.e., shape matching batch shape).
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    @classmethod
    @abstractmethod
    def from_distribution(
        cls,
        empirical: JointEmpiricalDistribution,
        **fit_kwargs: Any,
    ) -> 'Distribution_Density[T]':
        """
        Fit/convert from an empirical distribution to this parametric family.
        Typical implementations perform MLE/MoM/VI/MAP on `empirical.samples`
        (and optional weights) and return an instance of `cls`.
        """
        raise NotImplementedError("This method should be implemented by subclasses")


class NormalDistribution(Distribution_Density[T]):
    def __init__(self, mean: float, std_dev: float):
        self.mean = mean
        self.std_dev = std_dev
        self.rv = pm.Normal.dist(mu=self.mean, sigma=self.std_dev)

    def sample(self, n_samples: int) -> pt.TensorVariable:
        samples = np.random.normal(self.mean, self.std_dev, size=n_samples)
        return pt.constant(samples)

    def log_prob(self, data: pt.TensorVariable) -> pt.TensorVariable:
        return pm.logp(self.rv, data)
    
    @classmethod
    def from_distribution(cls, dist: EmpiricalDistribution)->'NormalDistribution':
        samples = dist.samples  # already a NumPy array
        mean = np.mean(samples)
        std_dev = np.std(samples, ddof=1)  # unbiased estimator
    
        return cls(mean=mean, std_dev=std_dev)

       

    def expectation(self, func: Callable[[pt.TensorVariable], pt.TensorVariable]) -> EmpiricalDistribution:
        # Monte Carlo approximation: sample numerically, evaluate func, average result
        n_samples = 10_000
        samples = self.rv.random(size=n_samples)  # numpy array
        values = func(pt.constant(samples)).eval()  # evaluate symbolic function on samples

        if isinstance(values, np.ndarray):
            stat = np.mean(values)
        else:
            stat = float(values)  # scalar
        
        # Return new empirical distribution with single value result
        # (wrap as singleton np array)
        return EmpiricalDistribution(np.array([stat]))




class HalfNorm(Distribution_Density[T]):
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
        self.std_dev = np.array(std_dev)

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        log_pdf = jnp.where(
            x >= 0,
            0.5 * jnp.log(2 / jnp.pi) - jnp.log(self.std_dev) - (x ** 2) / (2 * self.std_dev ** 2),
            -jnp.inf,
        )
        return jnp.sum(log_pdf, axis=-1)

    # CHANGE THIS: DON'T USE JNP!
    #key: random.PRNGKey 
    def sample(self, n_samples: int = 1) -> jnp.ndarray:
        # Sample from Normal(0, std_dev), then take absolute value for Half-Normal
        samples = jnp.abs(self.std_dev * random.normal(..., shape=(n_samples,)))
        if n_samples == 1:
            return samples[0]
        return samples
    
    @classmethod
    def from_distribution(cls, dist: EmpiricalDistribution)->'NormalDistribution':
        
        return NotImplementedError

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




class MultiVarNormalDistribution(Distribution_Density[T]):  # inherits your Distribution[T] if you have one
    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        """
        mean: (d,)
        cov:  (d, d) symmetric positive definite
        """
        self.mean = np.asarray(mean, dtype=float)
        self.cov = np.asarray(cov, dtype=float)
        if self.mean.ndim != 1:
            raise ValueError("mean must be 1D (shape (d,))")
        if self.cov.ndim != 2 or self.cov.shape[0] != self.cov.shape[1] or self.cov.shape[0] != self.mean.shape[0]:
            raise ValueError("cov must be (d,d) and match mean dimension")

        # PyMC random variable (distribution object)
        self.rv = pm.MvNormal.dist(mu=self.mean, cov=self.cov)

    def sample(self, n_samples: int) -> pt.TensorVariable:
        """
        Draw n_samples from the MVN. Returns a (n_samples, d) tensor constant.
        """
        draws = self.rv.random(size=n_samples)  # numpy array, shape (n, d)
        return pt.constant(draws)

    def log_prob(self, data: pt.TensorVariable) -> pt.TensorVariable:
        """
        Log-density at data. 'data' can be shape (d,) or (n, d).
        """
        return pm.logp(self.rv, data)

    @classmethod
    def from_distribution(
        cls,
        dist,                        # EmpiricalDistribution or anything with .samples ndarray
        jitter: float = 1e-6,
        max_tries: int = 5
    ) -> "MultiVarNormalDistribution":
        """
        Fit a MVN to empirical vector samples: mean and covariance (unbiased).
        Adds small jitter to ensure positive-definite covariance.
        Expects dist.samples shape (n, d) with n >= d+1.
        """
        samples = np.asarray(dist.samples)
        if samples.ndim != 2:
            raise ValueError("Empirical samples must be 2D of shape (n, d)")
        n, d = samples.shape
        if n <= d:
            raise ValueError(f"Not enough samples (n={n}) to estimate a {d}x{d} covariance.")

        mean = samples.mean(axis=0)
        cov = np.cov(samples, rowvar=False, ddof=1)

        # Ensure symmetry
        cov = 0.5 * (cov + cov.T)

        # Add jitter until Cholesky succeeds (or give up)
        tries, jitter_now = 0, jitter
        while tries < max_tries:
            try:
                np.linalg.cholesky(cov + jitter_now * np.eye(d))
                cov_pd = cov + jitter_now * np.eye(d)
                break
            except np.linalg.LinAlgError:
                jitter_now *= 10.0
                tries += 1
        else:
            # last resort: eigenvalue clipping
            w, V = np.linalg.eigh(cov)
            w_clipped = np.clip(w, a_min=1e-12, a_max=None)
            cov_pd = (V * w_clipped) @ V.T

        return cls(mean=mean, cov=cov_pd)

    def expectation(self, func: Callable[[pt.TensorVariable], pt.TensorVariable], n_samples: int = 10_000):
        """
        Monte Carlo estimate of E[f(X)] with X ~ MVN.
        Returns an EmpiricalDistribution over the statistic, if you have that class;
        otherwise, returns the numeric estimate.
        """
        draws = self.rv.random(size=n_samples)  # (n, d) numpy
        vals = func(pt.constant(draws)).eval()

        # Reduce to scalar if function returns vector/array
        stat = float(np.mean(vals)) if isinstance(vals, np.ndarray) else float(vals)

        # If you have EmpiricalDistribution available, wrap it; else return float.
        try:
            return JointEmpiricalDistribution(np.array([stat]))
        except NameError:
            return stat
        
    def mean_np(self) -> np.ndarray:
        return self.mean
    
    def cov_np(self) -> np.ndarray | float:
        return self.cov

        


class KDE(Distribution_Density[np.ndarray]):
    """
    Multivariate Gaussian Kernel Density Estimator as a density-based Distribution.

    - Fits from samples (e.g., MCMC draws) using Scott's rule by default:
        H = h^2 * Cov(samples),  h = n^(-1/(d+4))
    - Evaluates log density via log-sum-exp of Gaussian kernels.
    - Samples by mixture sampling: pick a data point uniformly and add N(0, H).
    """

    def __init__(
        self,
        samples: np.ndarray,
        *,
        bandwidth: str | float | np.ndarray = "scott",
        diag: bool = False,
        eps: float = 1e-8,
        weights: Optional[np.ndarray] = None,
    ) -> None:
        xs = np.asarray(samples, dtype=float)
        if xs.ndim == 1:
            xs = xs[:, None]  # (n, 1)
        if xs.ndim != 2:
            raise ValueError("samples must be (n,) or (n,d)")

        n, d = xs.shape
        if n < 2:
            raise ValueError("KDE requires at least 2 samples")

        self._xs = xs  # (n, d)
        self._n = n
        self._d = d

        # Normalize weights if provided
        if weights is not None:
            w = np.asarray(weights, dtype=float).reshape(-1)
            if w.shape[0] != n or np.any(w < 0):
                raise ValueError("weights must be length n and nonnegative")
            w_sum = w.sum()
            if w_sum <= 0:
                raise ValueError("sum of weights must be > 0")
            self._w = w / w_sum
        else:
            self._w = None  # uniform by default

        # Empirical covariance (optionally diagonal)
        xm = xs - xs.mean(axis=0, keepdims=True)
        if self._w is None:
            cov = (xm.T @ xm) / (n - 1)
        else:
            # Weighted covariance
            xm_w = xm * self._w[:, None]
            cov = (xm_w.T @ xm) / (1 - np.sum(self._w**2) + 1e-16)

        if diag:
            cov = np.diag(np.diag(cov))

        # Bandwidth selection → H (kernel covariance)
        H = self._build_covariance_matrix(cov, n, d, bandwidth)
        H = 0.5 * (H + H.T)  # symmetrize
        H.flat[:: d + 1] += eps  # jitter on the diagonal

        # Precompute constants for log_prob and sampling
        self._H = H
        try:
            self._L = np.linalg.cholesky(H)  # lower-triangular
        except np.linalg.LinAlgError:
            # Fallback: add more jitter
            H2 = H.copy()
            H2.flat[:: d + 1] += max(1e-6, eps * 10)
            self._L = np.linalg.cholesky(H2)
            self._H = H2

        self._H_inv = np.linalg.inv(self._H)
        sign, logdet = np.linalg.slogm(self._H) if hasattr(np.linalg, "slogm") else (None, None)  # rarely available
        # Use slogdet for SPD:
        sgn, logdet = np.linalg.slogdet(self._H)
        if sgn <= 0:
            raise ValueError("Kernel covariance must be SPD; got nonpositive determinant")
        self._logdetH = float(logdet)

        # Numeric constants
        self._log_norm_const = -0.5 * (self._d * np.log(2.0 * np.pi) + self._logdetH)

        # Cache PyTensor constants
        self._XS_pt = pt.as_tensor_variable(self._xs)              # (n, d)
        self._Hinv_pt = pt.as_tensor_variable(self._H_inv)         # (d, d)
        self._log_norm_const_pt = pt.as_tensor_variable(self._log_norm_const)
        if self._w is not None:
            self._W_pt = pt.as_tensor_variable(self._w)            # (n,)
        else:
            self._W_pt = None

    # ---------- required abstract API ----------

    @classmethod
    def from_distribution(
        cls,
        empirical: EmpiricalDistribution[np.ndarray],
        **fit_kwargs: Any,
    ) -> 'KDE':
        """
        Convert an empirical distribution (samples ± weights) into a KDE.

        Expected `empirical` interface:
        - .samples -> np.ndarray of shape (n,) or (n,d)
        - .weights -> Optional[np.ndarray] of shape (n,)
        Additional kwargs:
        - bandwidth: "scott" | "silverman" | float | np.ndarray
        - diag: bool
        - eps: float
        """
        samples = np.asarray(empirical.samples)
        weights = getattr(empirical, "weights", None)
        return cls(samples, weights=weights, **fit_kwargs)

    def sample(self, n_samples: int, rng: Optional[np.random.Generator] = None) -> pt.TensorVariable:
        """
        Draw from the KDE mixture: pick centers uniformly (or by weights) and add N(0, H) noise.
        Returns shape (n_samples,) for d=1, else (n_samples, d).
        """
        if rng is None:
            rng = np.random.default_rng()

        if self._w is None:
            idx = rng.integers(0, self._n, size=n_samples)
        else:
            idx = rng.choice(self._n, size=n_samples, p=self._w)

        z = rng.normal(size=(n_samples, self._d)) @ self._L.T   # N(0, H)
        draws = self._xs[idx, :] + z
        if self._d == 1:
            draws = draws[:, 0]
        return pt.as_tensor_variable(draws)

    def log_prob(self, data: pt.TensorVariable) -> pt.TensorVariable:
        """
        log p(x) = logsum_i [ log N(x | x_i, H) + log w_i ]  (or uniform weights)
                   - log (sum_i w_i)   (weights are normalized beforehand)
        Implemented with log-sum-exp for stability.
        Accepts data shape (..., d) or (...,) for d=1; returns shape (...,).
        """
        x = pt.as_tensor_variable(data)
        if self._d == 1:
            x = pt.atleast_1d(x)[:, None]        # (..., 1) with a trailing dim
        else:
            x = x.reshape((-1, self._d)) if x.ndim == 1 else x
            x = x.reshape((-1, self._d)) if x.ndim > 2 else x   # flatten leading dims if any

        # Ensure shape (m, d)
        if x.ndim == 1:
            x = x[None, :]

        # diff: (m, n, d)
        diff = x[:, None, :] - self._XS_pt[None, :, :]

        # Quadratic form for each pair (x_m, x_i): q = (x - xi)^T H^{-1} (x - xi)
        tmp = pt.dot(diff, self._Hinv_pt)        # (m, n, d)
        q = pt.sum(tmp * diff, axis=2)           # (m, n)

        log_kernel = self._log_norm_const_pt - 0.5 * q  # (m, n)

        if self._W_pt is None:
            # uniform weights = 1/n
            log_mix = pt.logsumexp(log_kernel, axis=1) - np.log(self._n)
        else:
            # weighted sum: log( sum_i w_i * exp(log_kernel) )
            log_mix = pt.logsumexp(log_kernel + pt.log(self._W_pt)[None, :], axis=1)

        # Shape (m,)
        return log_mix

    def expectation(
        self,
        func: Callable[[pt.TensorVariable], pt.TensorVariable],
        *,
        num_mc: int = 2048,
        rng: Optional[np.random.Generator] = None,
    ) -> pt.TensorVariable:
        """
        Monte Carlo estimate: E[f(X)] ≈ (1/M) Σ f(X_m), X_m ~ KDE.
        `func` should accept a tensor of draws with shape (M,) if d=1, else (M,d).
        """
        draws = self.sample(num_mc, rng=rng)           # (M,) or (M,d)
        vals = func(draws)                             # shape (M, ...) or (...)
        # If func returns per-draw values, average over axis 0; otherwise just return as-is.
        return pt.mean(vals, axis=0) if vals.ndim >= 1 else vals

    @property
    def event_shape(self) -> tuple[int, ...]:
        return () if self._d == 1 else (self._d,)

    # ---------- helpers ----------

    @staticmethod
    def _build_covariance_matrix(
        emp_cov: np.ndarray,
        n: int,
        d: int,
        bandwidth: str | float | np.ndarray,
    ) -> np.ndarray:
        if isinstance(bandwidth, str):
            b = bandwidth.lower()
            if b == "scott":
                h = n ** (-1.0 / (d + 4.0))
            elif b == "silverman":
                h = (n * (d + 2.0) / 4.0) ** (-1.0 / (d + 4.0))
            else:
                raise ValueError("bandwidth must be 'scott', 'silverman', float, or (d,d) array")
            return (h ** 2) * emp_cov
        elif np.isscalar(bandwidth):
            h = float(bandwidth)
            if h <= 0:
                raise ValueError("scalar bandwidth must be > 0")
            return (h ** 2) * emp_cov
        else:
            H = np.asarray(bandwidth, dtype=float)
            if H.shape != (d, d):
                raise ValueError(f"bandwidth array must have shape ({d},{d})")
            return H

    def mean_np(self) -> np.ndarray:
        """Mixture mean (d,) or scalar for 1D."""
        if self._w is None:
            m = self._xs.mean(axis=0)
        else:
            m = (self._w[:, None] * self._xs).sum(axis=0)
        return m if self._d > 1 else float(m[0])

    def cov_np(self) -> np.ndarray | float:
        """
        Mixture covariance: Cov(samples) + H.
        Returns (d,d) or scalar for 1D.
        """
        xs = self._xs
        m = self.mean_np()
        xm = xs - (m if self._d > 1 else np.array([m]))
        if self._w is None:
            emp_cov = (xm.T @ xm) / (self._n - 1)
        else:
            w = self._w[:, None]
            emp_cov = ( (w * xm).T @ xm ) / (1 - (self._w**2).sum() + 1e-16)

        C = emp_cov + self._H
        if self._d == 1:
            return float(C[0, 0])
        return C

    def var_np(self) -> float:
        """Scalar variance for 1D KDE."""
        if self._d != 1:
            raise ValueError("var_np is only defined for 1D KDE")
        return float(self.cov_np())
    

#Parametric extraction helpers (moment-matching)

def kde_as_mvnormal_params(kde: KDE):
    """
    Return (mu, cov) suitable for pm.MvNormal(.., mu=mu, cov=cov).
    mu: (d,), cov: (d,d)  (or scalars for 1D if you prefer)
    """
    mu = kde.mean_np()
    cov = kde.cov_np()
    # Ensure proper shapes for PyMC MvNormal when d=1
    if np.isscalar(mu):
        mu = np.array([mu], dtype=float)
        cov = np.array([[cov]], dtype=float)
    return mu, cov


def kde_as_exponential_rate(kde: KDE, *, eps: float = 1e-12) -> float:
    """
    Exponential(λ) by moment-matching mean m: λ = 1/m.
    Assumes positive support; raise if mean <= 0 or KDE has substantial mass ≤ 0.
    """
    if kde.event_shape != ():
        raise ValueError("Exponential fit requires a 1D KDE")
    m = float(kde.mean_np())
    if not np.isfinite(m) or m <= eps:
        raise ValueError(f"Invalid mean {m} for Exponential moment-matching")
    return 1.0 / m


def kde_as_gamma_params(kde: KDE, *, eps: float = 1e-12) -> tuple[float, float]:
    """
    Gamma(alpha, β_rate) by matching mean m and variance v:
        alpha= m^2 / v,   β = m / v
    Assumes positive support.
    """
    if kde.event_shape != ():
        raise ValueError("Gamma fit requires a 1D KDE")
    m = float(kde.mean_np())
    v = float(kde.var_np())
    if not (np.isfinite(m) and np.isfinite(v)) or m <= eps or v <= eps:
        raise ValueError(f"Invalid moments m={m}, v={v} for Gamma moment-matching")
    alpha = (m * m) / v
    beta_rate = m / v
    return alpha, beta_rate


