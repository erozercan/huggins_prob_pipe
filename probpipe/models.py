from typing import Callable, List, TypeVar, Mapping, Optional, Dict, Any
from abc import ABC, abstractmethod
import numpy as np
import pymc as pm
import scipy.stats as sp
from scipy.special import logsumexp
from prefect import flow, task, unmapped
from numpy.typing import NDArray
from .distributions import Distribution, NormalDistribution, BootstrapDistribution
import pytensor.tensor as pt

T=TypeVar("T")


# class Input:
#     def __init__(self, name, dtype, ndim, constraints):
#         self.name = name
#         self.dtype = dtype
#         self.ndim = ndim
#         self.constraints = constraints
#         # Choose scalar/vector/matrix based on ndim for real use
#         if dtype is float and (ndim is None or ndim == 0):
#             self.var = pt.scalar(name)
#         elif dtype is float and ndim == 1:
#             self.var = pt.vector(name)
#         elif dtype is float and ndim == 2:
#             self.var = pt.matrix(name)
#         else:
#             self.var = name + "_var"   # fallback or error

#     def __str__(self):
#         return f"<Input: {self.name} (dtype={self.dtype}, ndim={self.ndim})>"
    
#     def __call__(self, *args, **kwargs):
#         return self.var

# class Workflow:
#     def __init__(self):
#         self.inputs: Dict[str, Input] = {}
#         self._defaults: Dict[str, Any] = {}
#         self._run_func: Optional[Callable] = None

#     def Input(self, name: Optional[str], dtype: type, ndim: Optional[int] = None, constraints: Optional[tuple] = None):
#         if name in self.inputs:
#             raise ValueError(f"Input '{name}' already defined")
#         inp = Input(name, dtype, ndim, constraints)
#         if name is not None:
#             self.inputs[name] = inp
#         return inp.var  # Return symbolic tensor variable

#     def instantiate(self, **input_values):
#         self._defaults.update(input_values)

#     def run(self, **runtime_inputs):
#         print("Workflow.run called!")
#         # Compose inputs: runtime inputs override defaults
#         inputs = {**self._defaults, **runtime_inputs}

#         # Check all required inputs are provided
#         missing = [k for k, inp in self.inputs.items() if k not in inputs]
#         if missing:
#             raise ValueError(f"Missing inputs for run: {missing}")

#         if self._run_func is None:
#             raise RuntimeError("No run function defined for this workflow")

#         # Call the run function, pass inputs matching declared names only
#         # Support only inputs declared with names (ignore unnamed inputs)
#         call_args = {k: inputs[k] for k in self.inputs}
#         return self._run_func(**call_args)

#     def __enter__(self):
#         return self

#     def __exit__(self, exc_type, exc_value, traceback):
#         pass

#     def run_decorator(self, func: Callable):
#         print("Registering user-defined run function!")
#         if self._run_func is not None:
#             raise RuntimeError("Run function already set")
#         self._run_func = func
#         return func
    
#     def show_inputs(self):
#         for name, inp in self.inputs.items():
#             print(inp)

class InputSpec:
    def __init__(self, name: str, dtype: type, ndim: Optional[int], constraints: Optional[tuple]):
        self.name = name
        self.dtype = dtype
        self.ndim = ndim
        self.constraints = constraints
        
        # Instead of symbolic variable, just placeholder for future data
        self.var = f"{name}_placeholder"

    def __str__(self):
        return f"<InputSpec: {self.name} (dtype={self.dtype.__name__}, ndim={self.ndim})>"
    
    def __call__(self, *args, **kwargs):
        # Just returns placeholder or can be extended to hold data
        return self.var


class Workflow:
    def __init__(self):
        self.inputs: Dict[str, InputSpec] = {}
        self._defaults: Dict[str, Any] = {}
        self._run_func: Optional[Callable] = None

    def Input(self, name: Optional[str], dtype: type, ndim: Optional[int] = None, constraints: Optional[tuple] = None):
        if name in self.inputs:
            raise ValueError(f"Input '{name}' already defined")
        inp = InputSpec(name, dtype, ndim, constraints)
        if name is not None:
            self.inputs[name] = inp
        return inp.var  # just the placeholder string here

    def instantiate(self, **input_values):
        self._defaults.update(input_values)

    def run(self, **runtime_inputs):
        # print("Workflow.run called!")
        inputs = {**self._defaults, **runtime_inputs}
        missing = [k for k, inp in self.inputs.items() if k not in inputs]
        if missing:
            raise ValueError(f"Missing inputs for run: {missing}")
        if self._run_func is None:
            raise RuntimeError("No run function defined for this workflow")
        call_args = {k: inputs[k] for k in self.inputs}
        return self._run_func(**call_args)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def run_decorator(self, func: Callable):
        # print("Registering user-defined run function!")
        if self._run_func is not None:
            raise RuntimeError("Run function already set")
        self._run_func = func
        return func

    def show_inputs(self):
        for name, inp in self.inputs.items():
            print(inp)




#regular function
def simple_linreg(data: NDArray, sigma: float) -> NormalDistribution:
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


def simple_linreg


# @task
# def robust_regression(data: NDArray, sigma: float, dof: float) -> NormalDistribution:
#     """
#     Implements robust regression using a Student's t-distributed noise model:
    
#         y_i = alpha * X_i + epsilon_i,
#         epsilon_i ~ StudentT(nu=dof, mu=0, sigma=sigma)
    
#     where:
#     - 'alpha' is the regression coefficient with a prior Normal(mean_alpha, variance_alpha)
#     - noise (epsilon) follows a Student's t-distribution with 'dof' degrees of freedom,
#       allowing heavier tails than Gaussian noise.
    
#     Parameters:
#     - data: object containing fields 'X', 'y', 'mean_alpha' (prior mean), 'variance_alpha' (prior variance)
#     - sigma: scale parameter for the Student's t noise
#     - dof: degrees of freedom for the Student's t noise
    
#     Returns:
#     - Posterior distribution of alpha as a NormalDistribution (approximate posterior mean and std from MCMC)
#     """

#     X = data.data['X']
#     y = data.data['y']
#     prior_mean = data.data['mean_alpha']
#     prior_var = data.data['variance_alpha']

#     with pm.Model() as model:
#         # Prior on regression coefficient alpha
#         alpha = pm.Normal('alpha', mu=prior_mean, sigma=np.sqrt(prior_var))
        
#         # Likelihood with Student's t noise
#         y_obs = pm.StudentT('y_obs', nu=dof, mu=alpha * X, sigma=sigma, observed=y)
        
#         # Run inference: use MAP initialization to speed up sampling or just sample
#         trace = pm.sample(2000, tune=1000, cores=1, return_inferencedata=False, progressbar=False)

#     # Extract posterior mean and std of alpha
#     post_mean = np.mean(trace['alpha'])
#     post_std = np.std(trace['alpha'])

#     return NormalDistribution(post_mean, post_std)

# #@task
# #def bootstrap_distribution(
# #    data: NDArray,
# #    sample_size: float | int | None,
# #    axis: int = 0
# #) -> BootstrapDistribution:
# #    return BootstrapDistribution(data, sample_size, axis)


# @flow
# def bayesbag_linreg(
#     data: NDArray, 
#     sigma: float, 
#     n_bootstrap: int = 100
#     ) -> MixtureDistribution:
    
#     bd = BootstrapDistribution(data)
#     bootstrap_samples = bd.sample(n_bootstrap)
#     dists = [simple_linreg(sample, sigma) for sample in bootstrap_samples]
#     joint_distribution = MixtureDistribution(components=dists)
#     return joint_distribution

# @task
# def bayesbag_regression(
#     data: NDArray,
#     model_func: Callable[[NDArray], Distribution],
#     n_bootstrap: int = 100,
#     sample_size: int | None = None,
#     model_kwargs: dict = None,
# ) -> MixtureDistribution:
#     """
#     Abstract BayesBag regression procedure that:
#     1) Creates a bootstrap distribution from the data,
#     2) Draws bootstrap samples,
#     3) Applies a generic model function to each bootstrap sample to get posterior distributions,
#     4) Combines these bootstrap posteriors into a mixture distribution approximating the bagged posterior.

#     Parameters:
#     ----------
#     data : Information
#         Original dataset.
#     model_func : Callable[[Information, ...], Distribution]
#         Generic regression model function returning posterior distribution given data.
#     n_bootstrap : int
#         Number of bootstrap samples to draw.
#     sample_size : int | None
#         Size of each bootstrap sample (default: size of original data).
#     model_kwargs : dict | None
#         Optional extra params to pass to model_func.

#     Returns:
#     -------
#     Distribution
#         A MixtureDistribution representing the bagged posterior across bootstrap samples.
#     """

#     if model_kwargs is None:
#         model_kwargs = {}

#     # Create bootstrap distribution with specified sample size
#     bd = BootstrapDistribution(data, sample_size=sample_size)
    
#     # Draw bootstrap samples
#     bootstrap_samples = bd.sample(n_bootstrap)
    
#     # Fit model and get posterior per bootstrap sample
#     dists = [model_func(sample, **model_kwargs) for sample in bootstrap_samples]
    
#     # Combine posterior distributions into a mixture distribution
#     joint_distribution = MixtureDistribution(components=dists)

#     return joint_distribution























