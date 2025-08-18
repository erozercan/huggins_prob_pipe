from probpipe import NormalDistribution, EmpiricalDistribution, metropolis_hastings
from probpipe.models import Workflow
# import aesara.tensor as at
# from aesara.compile.ops import as_op

import pymc as pm
# import theano.tensor as at
# from theano.compile.ops import as_op


import numpy as np


with Workflow() as pppymc:
    prior_dist = pppymc.Input('prior_dist', object)
    data      = pppymc.Input('data', float)
    sigma     = pppymc.Input('sigma', float)
    
    @pppymc.run_decorator
    def run(prior_dist, data, sigma):
        with pm.Model() as model:

            
            alpha = pm.Normal("alpha", mu=prior_dist.mean, sigma=prior_dist.std_dev) ####
            

            pm.Normal("obs", mu=alpha * data, sigma=sigma, observed=data)
            
            trace = pm.sample(draws=1000, tune=500, chains=2, progressbar=False, cores=1)
        
        samples = trace.posterior["alpha"].stack(sample=("chain", "draw")).values.flatten()

        # Convert to NormalDistribution 
        emp_dist = EmpiricalDistribution(samples)
        normal_dist = NormalDistribution.from_distribution(emp_dist)


        return normal_dist




np.random.seed(42)  

prior0 = NormalDistribution(mean=0, std_dev=1)  




data1 = np.random.normal(loc=2.0, scale=0.5, size=120)
sigma1 = 0.1


data2 = np.random.normal(loc=2.5, scale=0.6, size=120)
sigma2 = 0.1

data3 = np.random.normal(loc=3.0, scale=0.7, size=120)
sigma3 = 0.1


# Step 1: run workflow with initial prior and data1 
posterior1 = pppymc.run(prior_dist=prior0, data=data1, sigma=sigma1)
# # Step 2: use posterior1 as prior for new data2
posterior2 = pppymc.run(prior_dist=posterior1, data=data2, sigma=sigma2)
# # Step 3: chain again with posterior2 
posterior3 = pppymc.run(prior_dist=posterior2, data=data3, sigma=sigma3)




