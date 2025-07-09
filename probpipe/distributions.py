from typing import Generic, TypeVar, Optional, Dict, Any, List, Callable, Mapping, Union
from abc import ABC, abstractmethod
import numpy as np
import pymc as pm
import scipy.stats as sp
from scipy.special import logsumexp
from scipy.stats import bootstrap
from numpy.typing import NDArray


#T = TypeVar('T')

#class Information[T]:
#    def __init__(self, data: T | None = None, metadata: T | None = None):
#        self.data = data
#        self.metadata = metadata if metadata is not None else {}
#
#    def __repr__(self):
#        return f"Information(data={self.data}, metadata={self.metadata})"


T = TypeVar("T")


class Distribution[T](ABC):
    @abstractmethod
    def sample(self, n_samples: int) -> NDArray[T]:
        """
        Sample n_samples items from the distribution.
        Returns a list of sampled raw data.
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    @abstractmethod
    def log_prob(self, data: NDArray) -> float:
        """
        Compute the log probability of the given raw data.
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    @abstractmethod
    def expectation(self, func: Callable[[T], float]) -> 'Distribution[T]':
        """
        Return a distribution representing the expectation of func under this distribution.
        """
        raise NotImplementedError("This method should be implemented by subclasses")

class BootstrapDistribution[T](Distribution[T]):
    def __init__(
        self,
        data: NDArray,
        sample_size: float | int | None = None,
        axis: int = 0
    ):
        # Store data as numpy array
        self.data = np.array(data)
        self.axis = axis

        if sample_size is None:
            # Default to same size as data along axis
            self.sample_size = self.data.shape[axis] if self.data.ndim > 0 else len(self.data)
        elif isinstance(sample_size, float) and 0 < sample_size < 1:
            # If fraction, take fraction of data length
            self.sample_size = int(sample_size * (self.data.shape[axis] if self.data.ndim > 0 else len(self.data)))
        elif isinstance(sample_size, int):
            self.sample_size = sample_size
        else:
            raise ValueError("sample_size must be None, float between 0 and 1, or int >= 1")

    def sample(self, n_samples: int) -> NDArray[np.float64]:
        """
        Draw n_samples bootstrap samples (with replacement) from the stored data.
        Returns a list of scalar bootstrap samples (averages).
        """
        bootstrap_samples = []
        data_len = self.data.shape[self.axis] if self.data.ndim > 0 else len(self.data)

        for _ in range(n_samples):
            # Sample indices with replacement
            indices = np.random.randint(0, data_len, size=self.sample_size)
            if self.data.ndim > 0:
                resampled = np.take(self.data, indices, axis=self.axis)
            else:
                resampled = self.data[indices]
            # Average the resampled data to form bootstrap sample
            bootstrap_stat = np.mean(resampled)
            bootstrap_samples.append(bootstrap_stat)
        return np.array(bootstrap_samples)

    def log_prob(self, data: float) -> float:
        """
        Raise NotImplementedError for now.
        """
        raise NotImplementedError("boostrap distribution log_prob not implemented")

    def expectation(self, func: Callable[[float], float]) -> 'Distribution':
        """
        Raise NotImplementedError for now.
        """
        raise NotImplementedError("bootstrap distribution expectation not implemented")


class NormalDistribution(Distribution[float]):
    def __init__(self, mean: float, std_dev: float):
        self.mean = mean
        self.std_dev = std_dev
        self._rv = sp.norm(loc=mean, scale=std_dev)

    def sample(self, n_samples: int) -> NDArray[np.float64]:
        # Return a list of floats (np.float64)
        samples = self._rv.rvs(size=n_samples)
        return np.array(samples)

    def log_prob(self, data: np.ndarray) -> float:
        # scalar log_pdf for one data point
        return self._rv.logpdf(data)

    def expectation(
        self,
        func: Callable[[np.ndarray], float],
        n_samples: int = 10000,
        n_boot: int = 1000
    ) -> BootstrapDistribution:
        """
        Monte Carlo bootstrap to estimate the empirical distribution of func(X).

        Parameters
        ----------
        func : Callable[[np.ndarray], float]
            Statistic function to apply to bootstrap sample arrays (e.g. np.mean, np.median).
        n_samples : int
            Number of samples per bootstrap iteration.
        n_boot : int
            Number of bootstrap iterations.

        Returns
        -------
        BootstrapDistribution
            Empirical distribution of bootstrap estimates of func(X).
        """
        estimates = []
        for _ in range(n_boot):
            samples = self._rv.rvs(size=n_samples)
            stat = func(samples)  # func operates on vector not element-wise

            if isinstance(stat, np.ndarray):
                stat = np.mean(stat) # if func returns an array, get mean func(samples)

            estimates.append(stat)
            
        return BootstrapDistribution[float](estimates)


    #def __repr__(self):
    #    return f"BootstrapDistribution(data_len={len(self.data)}, sample_size={self.sample_size})"