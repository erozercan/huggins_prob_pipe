from probpipe import NormalDistribution, EmpiricalDistribution, metropolis_hastings
from probpipe.models import Input, Workflow
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import scipy.stats as sp



# Generate some observed data assuming true mean is 3
observed_data = np.random.normal(3, 1, size=100)

def log_prior(x):
    return sp.norm(0, 5).logpdf(x)

def log_likelihood(x, data):
    return np.sum(sp.norm(loc=x, scale=1).logpdf(data))

def log_posterior(x):
    return log_prior(x) + log_likelihood(x, observed_data)


mcmc = metropolis_hastings(
    log_prob_fn=log_posterior,
    proposal_std=0.5,
    initial_state=0.0,
    n_samples=5000
)

