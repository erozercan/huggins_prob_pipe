from probpipe.distributions import NormalDistribution, Distribution_Density, MultiVarNormalDistribution, Distribution
from probpipe.models import Input, Workflow, simple_linreg, plot_posterior_evolution, robust_linreg
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
from prefect import task, flow
from numpy.typing import NDArray
from typing import Callable


if __name__ == "__main__":
    np.random.seed(42)
    d = 2
    batch_size = 100
    num_iterations = 3
    #true_alpha = 1.0
    true_beta = np.array([2.0, 3.0])
    obs_noise = 1.0

    current_prior=...

    #prior_dists = {
        #"alpha": NormalDistribution(mean=0.0, std_dev=10.0),
    #    "beta": NormalDistribution(mean=np.zeros(d), std_dev=np.ones(d) * 10.0),
    #    "sigma": HalfNorm(std_dev=5.0)
    #}


    obs_dist = NormalDistribution(mean=0.0, std_dev=obs_noise)
    all_posteriors = []
    #current_prior = {"beta": MultiVarNormalDistribution(mean=np.zeros(d), cov=np.eye(d) * 1.0)}

    current_prior=MultiVarNormalDistribution(mean=np.zeros(d), cov=np.eye(d) * 1.0)

    with Workflow() as pp:
        ## WHERE TO USE THESE VARIABLES???
        #prior_dist = pp.Input('prior_dist', object)
        #data      = pp.Input('data', float)
        #sigma     = pp.Input('sigma', float)

        #@task
        @pp.run_decorator
        def _run(data: NDArray,
                prior_dist: Distribution_Density,
                obs_dist: Distribution, 
                func: Callable[[NDArray, Distribution, Distribution], Distribution],
                conversion_type="Multivariate Gaussian KDE" #or "Multivariate Normal"
                ):
            return func(data, prior_dist, obs_dist)
        

    for i in range(num_iterations):
        print(f"\n--- Iteration {i+1} ---")

        # Next batch
        X_batch = np.random.randn(batch_size, d)
        #y_batch = true_alpha + X_batch @ true_beta + np.random.normal(0, obs_noise, size=batch_size)
        y_batch = X_batch @ true_beta + np.random.normal(0, obs_noise, size=batch_size)
        data_batch = {"X": X_batch, "Y": y_batch}

        posterior = pp.run( 
            data=data_batch,
            prior_dist=current_prior,
            obs_dist=obs_dist,
            func=simple_linreg, #or robust_linreg
            conversion_type="Multivariate Gaussian KDE") 

        current_prior=posterior

        #posterior.summary()
        all_posteriors.append(posterior)


    #plot_posterior_evolution(all_posteriors, "alpha")
    plot_posterior_evolution(all_posteriors, "beta")
    #plot_posterior_evolution(pp._posteriors, "sigma")

