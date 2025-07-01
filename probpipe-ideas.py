from typing import Callable
from prefect import flow, task, unmapped
from abc import ABC, abstractmethod


class Information:
    #TODO not yet clear to met what this class should contain, and in particular
    # how we want to represent the data and metadata.
    def __init__(self, data=None, metadata=None):
        self.data = data
        self.metadata = metadata if metadata is not None else {}


class Distribution(ABC):
    @abstractmethod
    def sample(self, n_samples: int) -> list[Information]:
        raise NotImplementedError("This method should be implemented by subclasses")

    @abstractmethod 
    def log_prob(self, data: Information) -> float:
        raise NotImplementedError("This method should be implemented by subclasses")
    
    @abstractmethod 
    def expectation(self, func: Callable[[Information], float]) -> 'Distribution':
        # Implement expectation logic here
        raise NotImplementedError("This method should be implemented by subclasses")


@task
def bootstrap_distribution(data: Information, sample_size: float | int | None = None, axis: int = 0) -> Distribution:
	return BootstrapDistribution(data, sample_size, axis)

@task
def simple_linreg(data: Information, sigma: float) -> Distribution:
    return NormalDistribution(0, 1)  # Placeholder for a simple linear regression model

@flow
def bayesbag_linreg(data: Information, sigma: float):
	bd = bootstrap_distribution(data)
	samples = bd.sample(100) # need a default and way for the user to change it
	dists = simple_linreg.map(samples, unmapped(sigma)).result()
	#TODO: need to combine these dists to create a joint distribution
	return None # Placeholder for the joint distribution
	
#TODO how would we make bayesbag_linreg into an ``abstract flow'' that could take, say, 
# a generic conditional distribution, and use it approximate the bootstrapped posterior? 
# For example, replace simple_linreg with a robust regression model that's fit using MCMC:
@task
def robust_regression(data: Information, sigma: float, dof: float) -> Distribution:
    #TODO implement robust regression model using PyMC3 or similar
    return NormalDistribution(0, 1)  # Placeholder for robust regression posterior 


class BootstrapDistribution(Distribution):
    def __init__(self, data: Information, sample_size: float | int | None = None, axis: int = 0):
        self.data = data
        self.sample_size = sample_size
        self.axis = axis

    def sample(self, n_samples: int) -> list[Information]:
        # Implement sampling logic here
        return []

    def log_prob(self, data: Information) -> float:
        # Implement log probability calculation here
        return 0.0

    def expectation(self, func: Callable[[Information], float]):
        raise NotImplementedError


class NormalDistribution(Distribution):
    def __init__(self, mean: float, std_dev: float):
        self.mean = mean
        self.std_dev = std_dev

    def sample(self, n_samples: int) -> list[Information]:
        # Implement sampling logic here
        return [] 
    
    def log_prob(self, data: Information) -> float:
        # Implement log probability calculation here
        return 0.0
    
    def expectation(self, func: Callable[[Information], float]) -> Distribution:
        raise NotImplementedError




def bayesbag_linreg(data: Information, sigma: float, n_bootstrap=100, n_posterior_samples=100) -> List[Information]:
    # Create bootstrap distribution object
    bd = bootstrap_distribution(data)
    
    # Generate bootstrap samples (Information objects)
    bootstrap_samples = bd.sample(n_bootstrap)
    
    # For each bootstrap sample, fit Bayesian linear regression posterior
    posteriors = [simple_linreg(sample, sigma) for sample in bootstrap_samples]
    
    # Draw samples from each posterior and combine
    combined_posterior_samples = []
    for posterior in posteriors:
        samples = posterior.sample(n_posterior_samples)
        combined_posterior_samples.extend(samples)
    
    return combined_posterior_samples


