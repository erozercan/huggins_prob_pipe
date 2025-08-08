from typing import Generic, TypeVar, Optional, Dict, Any, List, Callable, Mapping, Union, Optional
from abc import ABC, abstractmethod
import numpy as np
import pymc as pm
import scipy.stats as sp
from scipy.special import logsumexp
from scipy.stats import bootstrap
from numpy.typing import NDArray
from pytensor.graph.op import Op
import pytensor.tensor as pt
from scipy.stats import gaussian_kde

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
        raise NotImplementedError("log prob not implemented")

    def expectation(self, func: Callable[[float], float]) -> 'Distribution':
        """
        Raise NotImplementedError for now.
        """
        raise NotImplementedError("bootstrap distribution expectation not implemented")






class NormalDistribution(Distribution):
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




    @classmethod
    def from_distribution(cls, dist: Distribution) -> 'NormalDistribution':
        mean = dist.mean()
        variance = dist.covariance()

        std_dev = np.sqrt(variance)

        return cls(mean=mean, std_dev=std_dev)


    #def __repr__(self):
    #    return f"BootstrapDistribution(data_len={len(self.data)}, sample_size={self.sample_size})"




# class KDELogPDF(Op):
#     itypes = [pt.dscalar]
#     otypes = [pt.dscalar]
#     def __init__(self, kde):
#         self.kde = kde
#     def perform(self, node, inputs, outputs):
#         (x,) = inputs    # x is a float scalar
#         outputs[0][0] = np.array(self.kde.logpdf([x]))  
#         # returns array([value]), so take the first element or just use [x]!




# class EmpiricalDistribution(Distribution[T]):
#     def __init__(self, samples: NDArray):
#         self.samples = np.asarray(samples)
#         self.num_samples = self.samples.shape[0]
#         self.samples_var = pt.constant(self.samples)
#         self.kde = gaussian_kde(samples)
#         self.logpdf_op = KDELogPDF(self.kde)

#     def sample(self, n_samples: int) -> pt.TensorVariable:
#         indices = np.random.choice(self.num_samples, size=n_samples, replace=True)
#         sampled = self.samples[indices]
#         return pt.constant(sampled)

#     def log_prob(self, data):
#         # data can be a PyTensor variable
#         return self.logpdf_op(data)
    
    
#     def expectation(
#         self,
#         func: Callable[[pt.TensorVariable], pt.TensorVariable],
#         n_boot: int = 1000,
#         sample_size: int | None = None  # optional size for bootstrap resample, defaults to empirical size
#     ) -> BootstrapDistribution:
#         """
#         Estimate the distribution of func(X) where X ~ EmpiricalDistribution by bootstrap.

#         Parameters
#         ----------
#         func : Callable[[pt.TensorVariable], pt.TensorVariable]
#             Function to apply on bootstrap resampled samples.
#         n_boot : int
#             Number of bootstrap iterations.
#         sample_size : int or None
#             Size of each bootstrap sample (defaults to original num_samples).

#         Returns
#         -------
#         BootstrapDistribution
#             Distribution of bootstrap estimates of func(X).
#         """
#         if sample_size is None:
#             sample_size = self.num_samples

#         estimates = []
#         for _ in range(n_boot):
#             indices = np.random.choice(self.num_samples, size=sample_size, replace=True)
#             resampled = self.samples[indices]

#             # Apply func on pytensor constant wrapping bootstrap resample,
#             # then evaluate the computation to get a numpy scalar
#             stat = func(pt.constant(resampled)).eval()

#             # Reduce to scalar if numpy array is returned
#             if isinstance(stat, np.ndarray):
#                 stat = np.mean(stat)
#             estimates.append(stat)

#         return BootstrapDistribution(np.array(estimates))



class EmpiricalDistribution(Generic[T]):
    def __init__(self, samples: np.ndarray):
        self.samples = np.asarray(samples)
        self.num_samples = self.samples.shape[0]

        # KDE constructed with the original samples 
        self.kde = gaussian_kde(self.samples)

    def sample(self, n_samples: int) -> np.ndarray:
        """
        Bootstrap-resample from empirical samples with replacement.
        """
        indices = np.random.choice(self.num_samples, size=n_samples, replace=True)
        return self.samples[indices]
    
    def mean(self) -> np.ndarray:
        return np.mean(self.samples)

    def covariance(self) -> np.ndarray:
        return np.cov(self.samples)


    def log_prob(self, data: np.ndarray) -> np.ndarray:
        """
        Compute log-density for input data using KDE.
        
        Parameters
        ----------
        data : np.ndarray
            Points at which to evaluate the log-density. Shape (n_points, d) 
            for multi-dimensional or (n_points,) for 1D.
            
        Returns
        -------
        np.ndarray
            Log probability density values.
        """
        data = np.asarray(data)
        # KDE expects shape=(d, n), so transpose if needed
        data_for_kde = data
        prob_density = self.kde(data_for_kde)
        return np.log(prob_density + 1e-300)  # Add small epsilon to avoid log(0)

    def expectation(
        self,
        func: Callable[[np.ndarray], float],
        n_boot: int = 1000,
        sample_size: int | None = None
    ) -> BootstrapDistribution:
        """
        Estimate the distribution of func(X) where X ~ EmpiricalDistribution by bootstrap.

        Parameters
        ----------
        func : Callable[[np.ndarray], float]
            Function applied to bootstrap resampled samples.
        n_boot : int
            Number of bootstrap iterations.
        sample_size : int or None
            Size of each bootstrap sample (defaults to original num_samples).

        Returns
        -------
        BootstrapDistribution
            Bootstrap distribution of estimates.
        """
        if sample_size is None:
            sample_size = self.num_samples

        estimates = []
        for _ in range(n_boot):
            indices = np.random.choice(self.num_samples, size=sample_size, replace=True)
            resampled = self.samples[indices]

            stat = func(resampled)

            # If func returns array, reduce to scalar by mean as fallback
            if isinstance(stat, np.ndarray):
                stat = np.mean(stat)

            estimates.append(stat)

        return BootstrapDistribution(np.array(estimates))