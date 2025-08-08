from probpipe import NormalDistribution, EmpiricalDistribution, metropolis_hastings
from probpipe.models import Workflow
import numpy as np


np.random.seed(42)  

with Workflow() as pp:
    prior_dist = pp.Input('prior_dist', object)
    data      = pp.Input('data', float)
    sigma     = pp.Input('sigma', float)
    
    @pp.run_decorator
    def inference_step(prior_dist, data, sigma):
        # Define log posterior = log prior + log likelihood

        def log_posterior(mu):

            log_prior = prior_dist.log_prob(mu)
            normal = NormalDistribution(mean=mu, std_dev=sigma)
            log_likelihood = np.sum(normal.log_prob(data))

            return log_prior + log_likelihood

        # Run Metropolis-Hastings
        n_samples = 5000
        proposal_std = 0.5
        initial_state = 0.0  
        
        posterior_samples = metropolis_hastings(log_posterior, initial_state, n_samples, proposal_std, burn_in = 1000)
        
        emp_dist = EmpiricalDistribution(posterior_samples)
        normal_dist = NormalDistribution.from_distribution(emp_dist)

        # Return posterior as EmpiricalDistribution
        return normal_dist

    

## This should be simplified
## Todos
## Make a simple linear regression function and work with workflow
## Prefect on this workflow 




# Create initial prior
prior0 = NormalDistribution(mean=0, std_dev=1)  

data1 = np.random.normal(loc=2.0, scale=0.5, size=120)
sigma1 = 0.1

data2 = np.random.normal(loc=2.5, scale=0.6, size=120)
sigma2 = 0.1

data3 = np.random.normal(loc=3.0, scale=0.7, size=120)
sigma3 = 0.1





# Step 1: run workflow with initial prior and data1 
pp.instantiate(prior_dist=prior0, data=data1, sigma=sigma1)
posterior1 = pp.run()

print(f"Posterior 1 mean estimate: {posterior1.mean}")
print(f"Posterior 1 std estimate: {posterior1.std_dev}")


# Step 2: use posterior1 (approximate) as prior for data2
pp.instantiate(prior_dist=posterior1, data=data2, sigma=sigma2)
posterior2 = pp.run()

print(f"Posterior 2 mean estimate: {posterior2.mean}")
print(f"Posterior 2 std estimate: {posterior2.std_dev}")


# Step 3: use posterior2 as prior for data3
posterior3 = pp.run(prior_dist=posterior2, data=data3, sigma=sigma3)

print(f"Posterior 3 mean estimate: {posterior3.mean}")
print(f"Posterior 3 std estimate: {posterior3.std_dev}")



