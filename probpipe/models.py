from typing import Callable, List, TypeVar, Mapping, Optional, Dict, Any
from abc import ABC, abstractmethod
import numpy as np
import pymc as pm
from prefect import flow, task, unmapped
from numpy.typing import NDArray
from .distributions import NormalDistribution, EmpiricalDistribution, JointEmpiricalDistribution, MultiVarNormalDistribution
import pytensor.tensor as pt
import matplotlib.pyplot as plt
from functools import wraps

T=TypeVar("T")


class Input:
    def __init__(self, name, dtype, ndim, constraints):
        self.name = name
        self.dtype = dtype
        self.ndim = ndim
        self.constraints = constraints
        # Choose scalar/vector/matrix based on ndim for real use
        if dtype is float and (ndim is None or ndim == 0):
            self.var = pt.scalar(name)
        elif dtype is float and ndim == 1:
            self.var = pt.vector(name)
        elif dtype is float and ndim == 2:
            self.var = pt.matrix(name)
        else:
            self.var = name + "_var"   # fallback or error

class Workflow:
    def __init__(self):
        self._run_func = None
        self._current_prior = None          # density-only
        self._last_posterior = None         # empirical
        self._posteriors = []
        self.inputs = {}
        self._defaults = {}

    def Input(self, name: Optional[str], dtype: type, ndim: Optional[int] = None, constraints: Optional[tuple] = None):
        if name in self.inputs:
            raise ValueError(f"Input '{name}' already defined")
        inp = Input(name, dtype, ndim, constraints)
        if name is not None:
            self.inputs[name] = inp
        return inp.var  # Return symbolic tensor variable

    def instantiate(self, **input_values):
        self._defaults.update(input_values)

    def set_prior(self, prior_density):
        """Set the current prior (must be a density-backed object)."""
        self._current_prior = prior_density

    def get_prior(self):
        return self._current_prior

    def last_posterior(self):
        return self._last_posterior

    def all_posteriors(self):
        return list(self._posteriors)

    def run(self, **kwargs):
        if self._run_func is None:
            raise RuntimeError("No run function registered")
        return self._run_func(**kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def run_decorator(self, func: Callable):
        if self._run_func is not None:
            raise RuntimeError("Run function already set")
        
        @wraps(func)
        def wrapper(*args, data, prior_dist=None, obs_dist=None, **kwargs):
            #happening last
            
            #mu_alpha = posterior.samples["alpha"].mean()
            #sigma_alpha = posterior.samples["alpha"].std()

            #alpha=NormalDistribution(mean=mu_alpha, std_dev=sigma_alpha)
            
            #mu_beta = posterior.samples["beta"].mean(axis=0)
            #sigma_beta = posterior.samples["beta"].std(axis=0)

            #beta=NormalDistribution(mean=mu_beta, std_dev=sigma_beta)
            
            #sigma_sigma = posterior.samples["sigma"].std()  # scale for HalfNormal
            #sigma=HalfNorm(std_dev=sigma_sigma)

            #prior_dists={#"alpha":alpha,
                         #"beta": beta
                         #"sigma":sigma
                         #}

            #INPUT CONVERSION (type conversion for inputs)
            prior_in = prior_dist if prior_dist is not None else self._current_prior
            if prior_in is None:
                raise ValueError("No prior provided. Call pp.set_prior(...) first or pass prior_dist=...")
            
            # If caller accidentally passed an empirical prior, converts it to density
            prior_density = ensure_density_prior(prior_in)

            #CALL FUNC with a density prior
            posterior=func(data= data, prior_dist=prior_density, obs_dist=obs_dist, **kwargs)

            #OUTPUT: NEXT PRIOR (/w density), keeping it internal 
            next_prior_density = posterior_to_prior_density(posterior) 
            self._current_prior = next_prior_density

            #Storing posterior for later inspection
            self._last_posterior = posterior
            self._posteriors.append(posterior)
            return posterior

        self._run_func = wrapper
        return wrapper

#@task
def ensure_density_prior(prior):
    """
    If `prior` is already a density-backed object (e.g., MultiVNormalDistribution),
    return it. If it’s empirical (e.g., JointEmpiricalDistribution with 'beta'),
    fit a density (e.g., MVN) and return that.
    """
    # Density type (example)
    if hasattr(prior, "mean") and hasattr(prior, "cov"):
        return prior
    
    # Dict form: {"beta": density_or_empirical}
    if isinstance(prior, dict):
        b = prior["beta"]
        if hasattr(b, "mean") and hasattr(b, "cov"):
            return prior  # already density
        else:
            # empirical → density
            mu = b.samples.mean(axis=0)
            cov = np.cov(b.samples, rowvar=False)
            cov = 0.5 * (cov + cov.T) + 1e-6*np.eye(cov.shape[0])
            return {"beta": MultiVarNormalDistribution(mean=mu, cov=cov)}
        
    # Empirical beta alone
    if hasattr(prior, "samples"):
        mu = prior.samples.mean(axis=0)
        cov = np.cov(prior.samples, rowvar=False)
        cov = 0.5 * (cov + cov.T) + 1e-6*np.eye(cov.shape[0])
        return MultiVarNormalDistribution(mean=mu, cov=cov)

    raise TypeError("Unsupported prior type for density conversion")

#@task
def posterior_to_prior_density(posterior):
    """
    Build the next iteration prior (a density) from the empirical posterior.
    Keep it simple: only carry over beta as MVN; keep sigma weak/constant if desired.
    """
    beta_samps = posterior.samples["beta"]               # (n, d)
    mu = beta_samps.mean(axis=0)
    cov = np.cov(beta_samps, rowvar=False)
    cov = 0.5 * (cov + cov.T) + 1e-6*np.eye(cov.shape[0])
    return {"beta": MultiVarNormalDistribution(mean=mu, cov=cov)}



#@task
def simple_linreg(data: NDArray, prior_dist, obs_dist,num_samples=1000,step_size=0.01, sample_NUTS: bool=True) -> EmpiricalDistribution:
    """
    y = alpha * X + epsilon, where epsilon ~ Normal(0, sigma^2) 

    alpha ~ Normal(mean_alpha, variance_alpha)
   
    Parameters:
    - data: Information object containing 'X', 'y', 'mean_alpha' (prior mean), and 'variance_alpha' (prior variance).
    - sigma: known standard deviation of Gaussian noise in the linear model.

    Returns:
    - Posterior distribution of alpha as a NormalDistribution with updated mean and std.
    """

    X=data["X"] 
    Y=data["Y"] 

    #d=X.shape[1]

    #Model is a basic bayesian linear regression: Y= alpha + beta_1 * X1 + beta_2 * X2
    basic_model = pm.Model()

    with basic_model:
        # Priors for unknown model parameters
        #alpha = pm.Normal("alpha", mu=prior_dist["alpha"].mean, sigma=prior_dist["alpha"].std_dev)
        #beta = pm.Normal("beta", mu=prior_dist["beta"].mean, sigma=prior_dist["beta"].std_dev, shape=d) 
        beta = pm.MvNormal("beta", mu=prior_dist.mean, cov=prior_dist.cov)
        #sigma = pm.HalfNormal("sigma", sigma=prior_dist["sigma"].std_dev)
        sigma = pm.HalfNormal("sigma", sigma=5.0)

        Xd = pm.Data("X", X)
        Yd = pm.Data("Y", Y)

        # Expected value of outcome
        #mu = alpha + beta[0] * X[0] + beta[1] * X[1]
        #mu = alpha + X @ beta
        #mu = X @ beta
        mu = pm.math.dot(Xd, beta)
        #mu=mu = alpha + pm.math.dot(X, beta) if X = np.column_stack([X1, X2])
        #mu = alpha + pm.math.dot(X.T, beta) if X = np.array([X1, X2])

        # Likelihood (sampling distribution) of observations

        if isinstance(obs_dist, NormalDistribution):
            Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=Yd)
        else:
            raise ValueError("No obs_dist was set!")

        if sample_NUTS:
        #NUTS SAMPLER
            with basic_model:
                # draw 1000 posterior samples
                #target_accept=0.95 helps reduce divergent transitions.
                idata = pm.sample(num_samples, tune=num_samples, chains=2, step_size=step_size, return_inferencedata=True)
        else:
            #SLICE SAMPLER
            with basic_model:
                # instantiate sampler
                step = pm.Slice()
                # draw 5000 posterior samples
                idata = pm.sample(num_samples, tune=num_samples, chains=2, step=step_size)
                #trace = pm.sample(draws=1000, tune=500, chains=2, progressbar=False, cores=1)

    #print(idata.posterior["beta"].dims)

    samples = {
    #"alpha": idata.posterior["alpha"].stack(sample=("chain", "draw")).values,
    #"beta": idata.posterior["beta"].stack(sample=("chain", "draw")).values,
    "beta": idata.posterior["beta"].stack(sample=("chain", "draw")).transpose("sample", "beta_dim_0").values
    #"sigma": idata.posterior["sigma"].stack(sample=("chain", "draw")).values
    }

    #print(f"Beta values shape is {samples["beta"].shape}")


    return JointEmpiricalDistribution(samples)



def plot_posterior_evolution(all_posteriors, param_name: str):
    """
    Plot posterior mean and 90% credible interval for each iteration.
    
    Supports both scalar and vector-valued parameters.
    """
    means = []
    lowers = []
    uppers = []

    for posterior in all_posteriors:
        samples = posterior.samples[param_name]  # shape: (n_samples,) or (n_samples, d)

        # Handle scalar or vector
        if samples.ndim == 1:
            mean = samples.mean()
            lower = np.percentile(samples, 5)
            upper = np.percentile(samples, 95)
        else:  # vector-valued
            mean = samples.mean(axis=0)
            lower = np.percentile(samples, 5, axis=0)
            upper = np.percentile(samples, 95, axis=0)

        means.append(mean)
        lowers.append(lower)
        uppers.append(upper)

    means = np.array(means)
    lowers = np.array(lowers)
    uppers = np.array(uppers)

    iterations = np.arange(1, len(all_posteriors) + 1)

    plt.figure(figsize=(8, 5))

    if means.ndim == 1:  # scalar case
        plt.title(param_name)
        plt.plot(iterations, means, label=f"{param_name} mean")
        plt.fill_between(iterations, lowers, uppers, alpha=0.3, label="90% CI")
        plt.show()
    else:  # vector case
        for i in range(means.shape[1]):
            plt.title(param_name)
            plt.plot(iterations, means[:, i], label=f"{param_name}[{i}] mean")
            plt.fill_between(iterations, lowers[:, i], uppers[:, i], alpha=0.3, label=f"{param_name}[{i}] 90% CI")
            plt.show()




# import arviz as az
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import pymc as pm


# def model(X: np.ndarray, Y: np.ndarray)->pm.Model:
#     #for now X is dataset with two attributes: X1 and X2
#     #Y is the true values

#     #Model is a basic bayesian linear regression: Y= alpha + beta_1 * X1 + beta_2 * X2
#     basic_model = pm.Model()

#     with basic_model:
#         # Priors for unknown model parameters
#         alpha = pm.Normal("alpha", mu=0, sigma=10)
#         beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
#         sigma = pm.HalfNormal("sigma", sigma=1)

#         # Expected value of outcome
#         mu = alpha + beta[0] * X[0] + beta[1] * X[1]
#         #mu=mu = alpha + pm.math.dot(X, beta) if X = np.column_stack([X1, X2])
#         #mu = alpha + pm.math.dot(X.T, beta) if X = np.array([X1, X2])

#         # Likelihood (sampling distribution) of observations
#         Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=Y)

#     return basic_model, (alpha, beta, sigma)

# def run(model: pm.Model, sample_NUTS: bool=True)->az.InferenceData:

#     if sample_NUTS:
#         #NUTS SAMPLER
#         with model:
#             # draw 1000 posterior samples
#             #target_accept=0.95 helps reduce divergent transitions.
#             idata = pm.sample(1000, tune=1000, return_inferencedata=True)
#     else:
#         #SLICE SAMPLER
#         with model:
#             # instantiate sampler
#             step = pm.Slice()
#             # draw 5000 posterior samples
#             idata = pm.sample(5000, step=step)

#     return idata

# #IMPLEMENT LATER
# def prior_predictive_sampling(model: pm.Model, sample_num=100):
#     with model:
#         prior_samples = pm.sample_prior_predictive(sample_num)

#     return prior_samples


# def posterior_analysis(idata: az.InferenceData)->None:

#     az.plot_trace(idata, combined=True)
#     plt.suptitle("Posterior Trace Plot", fontsize=16)  # Use suptitle for multi-subplot figures
#     #plt.tight_layout()
#     plt.show()
    
#     #The next easy model-checking step is to see if the NUTS sampler performed as expected.
#     # An energy plot is a way of checking if the NUTS algorithm was able to adequately explore the
#     # posterior distribution. If it was not, one runs the risk of biased posterior estimates when
#     # parts of the posterior are not visited with adequate frequency. The plot shows two density
#     # estimates: one is the marginal energy distribution of the sampling run and the other is
#     # the distribution of the energy transitions between steps. This is all a little abstract,
#     # but all we are looking for is for the distributions to be similar to one another.
#     az.plot_energy(idata)
#     plt.title("Energy Plot for NUTS Sampler")
#     #plt.tight_layout()
#     plt.show()

#     summary_df = az.summary(idata, round_to=2)
#     print(summary_df)

#     return summary_df



# if __name__ == "__main__":

#     RANDOM_SEED = 8927
#     rng = np.random.default_rng(RANDOM_SEED)
#     az.style.use("arviz-darkgrid")

#     # True parameter values
#     alpha, sigma = 1, 1
#     beta = [1, 2.5]

#     # Size of dataset
#     size = 100

#     # Predictor variable
#     X1 = np.random.randn(size)
#     X2 = np.random.randn(size) * 0.2

#     # Simulate outcome variable

#     Y = alpha + beta[0] * X1 + beta[1] * X2 + rng.normal(size=size) * sigma

#     fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 4))
#     axes[0].scatter(X1, Y, alpha=0.6)
#     axes[1].scatter(X2, Y, alpha=0.6)
#     axes[0].set_ylabel("Y")
#     axes[0].set_xlabel("X1")
#     axes[1].set_xlabel("X2")

#     plt.show()

#     X=np.array([X1, X2])
#     #X = np.column_stack([X1, X2]) this X==np.array([X1, X2]).T

#     bayes_linreg_model, _=model(X, Y)

#     idata=run(bayes_linreg_model)

#     posterior_analysis(idata)




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
