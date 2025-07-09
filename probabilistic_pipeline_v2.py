from typing import Callable, List, TypeVar, Mapping
from abc import ABC, abstractmethod
import numpy as np
import pymc as pm
import scipy.stats as sp
from scipy.special import logsumexp
from prefect import flow, task, unmapped

#This is an attempt to push the code 

T=TypeVar("T")

#class Information[T]:
#    def __init__(self, data: T | None = None, metadata: T | None = None):
#        self.data = data
#        self.metadata = metadata if metadata is not None else {}
#
#    def __repr__(self):
#        return f"Information(data={self.data}, metadata={self.metadata})"


class Distribution[T](ABC):
    @abstractmethod
    def sample(self, n_samples: int) -> List[Information[T]]:
        raise NotImplementedError("This method should be implemented by subclasses")

    @abstractmethod
    def log_prob(self, data: Information[T]) -> float:
        raise NotImplementedError("This method should be implemented by subclasses")

    @abstractmethod
    def expectation(self, func: Callable[[Information[T]], float]) -> 'Distribution':
        raise NotImplementedError("This method should be implemented by subclasses")


class NormalDistribution(Distribution[float]):
    def __init__(self, mean: float, std_dev: float):
        self.mean = mean
        self.std_dev = std_dev
        self._rv = sp.norm(loc=mean, scale=std_dev)

    def sample(self, n_samples: int) -> List[Information[float]]:
        samples = self._rv.rvs(size=n_samples)
        # Wrap each sample individually
        return [Information(data=float(sample)) for sample in samples]

    def log_prob(self, data: Information[float]) -> float:
        return self._rv.logpdf(data.data)

    def expectation(self, func: Callable[[Information[float]], float]) -> 'Distribution':   #Question here
        # To implement depending on use case - placeholder raises
        raise NotImplementedError("expectation not implemented")
    
    @property
    def std(self):
        return self.std_dev

    def __repr__(self):
        return f"NormalDistribution(mean={self.mean}, std_dev={self.std_dev})"


class BootstrapDistribution[T](Distribution[T]):
    def __init__(self, data: Information[T], sample_size: int | None = None):
        self.data = data
        self.sample_size = sample_size

    def sample(self, n_samples: int) -> List[Information[T]]:
        data_dict = self.data.data
        sample_size = self.sample_size 

        samples = []
        for _ in range(n_samples):
            indices = np.random.choice(n, size=sample_size, replace=True)
            boot_sample = {}
            for key, value in data_dict.items():
                arr = np.array(value)
                if arr.ndim > 0 and len(arr) == n:
                    # Array-like data of correct length, bootstrap with indices
                    boot_sample[key] = arr[indices]
                else:
                    # Scalar or metadata, copy as is
                    boot_sample[key] = value
            samples.append(Information(data=boot_sample))
        return samples

    def log_prob(self, data: Information[T]) -> float:
        raise NotImplementedError("log_prob not implemented for BootstrapDistribution")

    def expectation(self, func: Callable[[Information[T]], float]) -> 'Distribution':
        raise NotImplementedError("expectation not implemented for BootstrapDistribution")

    def __repr__(self):
        return f"BootstrapDistribution(data={self.data}, sample_size={self.sample_size})"

@task
def bootstrap_distribution(data: Information[T], sample_size: int | None = None) -> Distribution:
    return BootstrapDistribution(data, sample_size)

@task
def simple_linreg(data: Information[T], sigma: float) -> Distribution:
    """
    y = alpha * X + epsilon, where epsilon ~ Normal(0, sigma^2) 

    alpha ~ Normal(mean_alpha, variance_alpha)
   
    Parameters:
    - data: Information object containing 'X', 'y', 'mean_alpha' (prior mean), and 'variance_alpha' (prior variance).
    - sigma: known standard deviation of Gaussian noise in the linear model.

    Returns:
    - Posterior distribution of alpha as a NormalDistribution with updated mean and std.
    """
    X = data.data['X']
    y = data.data['y']
    prior_mean = data.data['mean_alpha']
    prior_var = data.data['variance_alpha']

    precision_post = 1.0 / prior_var + np.sum(X ** 2) / (sigma ** 2)
    post_var = 1.0 / precision_post
    post_mean = post_var * (prior_mean / prior_var + np.sum(X * y) / (sigma ** 2))
    post_std = np.sqrt(post_var)

    return NormalDistribution(post_mean, post_std)


class MixtureDistribution[T](Distribution[T]):
    def __init__(self, components: List[Distribution[T]], weights: List[float] = None):
        self.components = components
        if weights is None:
            self.weights = [1 / len(components)] * len(components)
        else:
            self.weights = weights

    def sample(self, n_samples: int) -> List[Information[T]]:
        samples = []
        for _ in range(n_samples):
            comp_idx = np.random.choice(len(self.components), p=self.weights)
            sample_component = self.components[comp_idx].sample(1)[0]
            samples.append(sample_component)
        return samples

    def log_prob(self, data: Information[T]) -> float:
        component_logprobs = []
        for weight, comp in zip(self.weights, self.components):
            component_logprobs.append(np.log(weight) + comp.log_prob(data))
        return logsumexp(component_logprobs)

    def expectation(self, func: Callable[[Information[T]], float]) -> 'Distribution':
        raise NotImplementedError("expectation not implemented for MixtureDistribution")
    
    def summarize(self):
        """
        Compute an approximate summarizing distribution (Normal) of the mixture,
        using the weighted mean and variance of component means and variances.
        Assumes components have attributes `.mean` and `.std` (standard deviation).
        """

        import numpy as np

        means = np.array([comp.mean for comp in self.components])
        stds = np.array([comp.std for comp in self.components])  # or std_dev attribute
        weights = np.array(self.weights)

        mean_mix = np.sum(weights * means)
        var_mix = np.sum(weights * (stds**2 + means**2)) - mean_mix**2
        std_mix = np.sqrt(var_mix)

        # Return a NormalDistribution summarizing the mixture
        return NormalDistribution(mean_mix, std_mix)

    def __repr__(self):
        return f"MixtureDistribution(components={self.components}, weights={self.weights})"


@task
def robust_regression(data: Information[T], sigma: float, dof: float) -> Distribution:
    """
    Implements robust regression using a Student's t-distributed noise model:
    
        y_i = alpha * X_i + epsilon_i,
        epsilon_i ~ StudentT(nu=dof, mu=0, sigma=sigma)
    
    where:
    - 'alpha' is the regression coefficient with a prior Normal(mean_alpha, variance_alpha)
    - noise (epsilon) follows a Student's t-distribution with 'dof' degrees of freedom,
      allowing heavier tails than Gaussian noise.
    
    Parameters:
    - data: object containing fields 'X', 'y', 'mean_alpha' (prior mean), 'variance_alpha' (prior variance)
    - sigma: scale parameter for the Student's t noise
    - dof: degrees of freedom for the Student's t noise
    
    Returns:
    - Posterior distribution of alpha as a NormalDistribution (approximate posterior mean and std from MCMC)
    """

    X = data.data['X']
    y = data.data['y']
    prior_mean = data.data['mean_alpha']
    prior_var = data.data['variance_alpha']

    with pm.Model() as model:
        # Prior on regression coefficient alpha
        alpha = pm.Normal('alpha', mu=prior_mean, sigma=np.sqrt(prior_var))
        
        # Likelihood with Student's t noise
        y_obs = pm.StudentT('y_obs', nu=dof, mu=alpha * X, sigma=sigma, observed=y)
        
        # Run inference: use MAP initialization to speed up sampling or just sample
        trace = pm.sample(2000, tune=1000, cores=1, return_inferencedata=False, progressbar=False)

    # Extract posterior mean and std of alpha
    post_mean = np.mean(trace['alpha'])
    post_std = np.std(trace['alpha'])

    return NormalDistribution(post_mean, post_std)


@flow
def bayesbag_linreg(
    data: Information[T], 
    sigma: float, 
    n_bootstrap: int = 100
    ) -> Distribution:
    
    bd = bootstrap_distribution(data)
    bootstrap_samples = bd.sample(n_bootstrap)
    dists = [simple_linreg(sample, sigma) for sample in bootstrap_samples]
    joint_distribution = MixtureDistribution(components=dists)
    return joint_distribution



def bayesbag_regression(
    data: Information,
    model_func: Callable[[Information[T]], Distribution],
    n_bootstrap: int = 100,
    sample_size: int | None = None,
    model_kwargs: dict = None,
) -> Distribution:
    """
    Abstract BayesBag regression procedure that:
    1) Creates a bootstrap distribution from the data,
    2) Draws bootstrap samples,
    3) Applies a generic model function to each bootstrap sample to get posterior distributions,
    4) Combines these bootstrap posteriors into a mixture distribution approximating the bagged posterior.

    Parameters:
    ----------
    data : Information
        Original dataset.
    model_func : Callable[[Information, ...], Distribution]
        Generic regression model function returning posterior distribution given data.
    n_bootstrap : int
        Number of bootstrap samples to draw.
    sample_size : int | None
        Size of each bootstrap sample (default: size of original data).
    model_kwargs : dict | None
        Optional extra params to pass to model_func.

    Returns:
    -------
    Distribution
        A MixtureDistribution representing the bagged posterior across bootstrap samples.
    """

    if model_kwargs is None:
        model_kwargs = {}

    # Create bootstrap distribution with specified sample size
    bd = bootstrap_distribution(data, sample_size=sample_size)
    
    # Draw bootstrap samples
    bootstrap_samples = bd.sample(n_bootstrap)
    
    # Fit model and get posterior per bootstrap sample
    dists = [model_func(sample, **model_kwargs) for sample in bootstrap_samples]
    
    # Combine posterior distributions into a mixture distribution
    joint_distribution = MixtureDistribution(components=dists)

    return joint_distribution



# EXAMPLE USAGE:
if __name__ == "__main__":
    # Generate synthetic data
    # np.random.seed(0)
    n = 200
    X = np.linspace(0, 10, n)
    true_alpha = 2.5
    sigma = 1.0
    y = true_alpha * X + np.random.normal(0, sigma, size=n)

    # Prior parameters
    mean_alpha = 2.5
    variance_alpha = 5.0

    data = Information[Mapping[str, np.ndarray | float]]({
        'X': X,
        'y': y,
        'mean_alpha': mean_alpha,
        'variance_alpha': variance_alpha
    })

    # Run BayesBag regression with simple_linreg
    post_bayesbag = bayesbag_regression(
        data,
        model_func=simple_linreg,
        n_bootstrap=100,
        sample_size=None,
        model_kwargs={'sigma': sigma}
    )

    print("Posterior bagged posterior (mixture) with simple_linreg:")
    print(post_bayesbag.summarize())

    post_bayesbag_robust = bayesbag_regression(
        data,
        model_func=robust_regression,
        n_bootstrap=20,       # fewer samples since MCMC is slower
        sample_size=None,
        model_kwargs={'sigma': sigma, 'dof': 4.0}
    )

    print("Posterior bagged posterior (mixture) with robust_regression:")
    print(post_bayesbag_robust.summarize())


