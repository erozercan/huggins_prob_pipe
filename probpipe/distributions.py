from typing import Generic, TypeVar, Callable, Optional
from abc import ABC, abstractmethod
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from numpy.typing import NDArray
from scipy.stats import gaussian_kde
from pytensor.graph.op import Op

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
    def log_prob(self, data: pt.TensorVariable) -> pt.TensorVariable:
        """
        Compute the log probability of the given raw data.
        Returns pytensor scalar tensor.
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    @abstractmethod
    def expectation(self, func: Callable[[pt.TensorVariable], pt.TensorVariable]) -> 'Distribution[T]':
        """
        Return a distribution representing the expectation of func under this distribution.
        """
        raise NotImplementedError("This method should be implemented by subclasses")


class BootstrapDistribution(Distribution[float]):
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

    def log_prob(self, data: float) -> float:
        data_min = np.min(self.data)
        data_max = np.max(self.data)
        if data_min <= data <= data_max:
            return -np.log(len(self.data))  # approximate uniform log prob over empirical data
        else:
            return float('-inf')

    def expectation(self, func: Callable[[float], float]) -> 'Distribution':
        raise NotImplementedError("bootstrap distribution expectation not implemented")



class NormalDistribution:
    def __init__(self, mean: float, std_dev: float):
        self.mean = mean
        self.std_dev = std_dev
        self.rv = pm.Normal.dist(mu=self.mean, sigma=self.std_dev)

    def sample(self, n_samples: int) -> pt.TensorVariable:
        samples = np.random.normal(self.mean, self.std_dev, size=n_samples)
        return pt.constant(samples)

    def log_prob(self, data: pt.TensorVariable) -> pt.TensorVariable:
        return pm.logp(self.rv, data)
       

    def expectation(self, func: Callable[[pt.TensorVariable], pt.TensorVariable]) -> 'EmpiricalDistribution':
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


class KDELogPDF(Op):
    itypes = [pt.dscalar]
    otypes = [pt.dscalar]
    def __init__(self, kde):
        self.kde = kde
    def perform(self, node, inputs, outputs):
        (x,) = inputs    # x is a float scalar
        outputs[0][0] = np.array(self.kde.logpdf([x]))  
        # returns array([value]), so take the first element or just use [x]!



class EmpiricalDistribution(Distribution[T]):
    def __init__(self, samples: NDArray):
        self.samples = np.asarray(samples)
        self.num_samples = self.samples.shape[0]
        self.samples_var = pt.constant(self.samples)
        self.kde = gaussian_kde(samples)
        self.logpdf_op = KDELogPDF(self.kde)

    def sample(self, n_samples: int) -> pt.TensorVariable:
        indices = np.random.choice(self.num_samples, size=n_samples, replace=True)
        sampled = self.samples[indices]
        return pt.constant(sampled)

    def log_prob(self, data):
        # data can be a PyTensor variable
        return self.logpdf_op(data)
    
    
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