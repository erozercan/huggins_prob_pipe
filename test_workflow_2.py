from probpipe import NormalDistribution, Workflow, EmpiricalDistribution, KDELogPDF
from probpipe.models import Input
import numpy as np
import pymc as pm

import matplotlib.pyplot as plt
import numpy as np

with Workflow() as pp:
    prior_dist = pp.Input('prior_dist', object)
    data      = pp.Input('data', float)
    sigma     = pp.Input('sigma', float)
    
    @pp.run_decorator
    def run(prior_dist, data, sigma):
        with pm.Model() as model:

            if isinstance(prior_dist, EmpiricalDistribution):
                alpha = pm.Flat("alpha")
                prior_logp = prior_dist.log_prob(alpha)
                pm.Potential("prior_factor", prior_logp)
            else:
                alpha = pm.Normal("alpha", mu=prior_dist.mean, sigma=prior_dist.std_dev)
            

            pm.Normal("obs", mu=alpha * data, sigma=sigma, observed=data)
            
            trace = pm.sample(draws=1000, tune=500, chains=2, progressbar=False, cores=1)
        
        samples = trace.posterior["alpha"].stack(sample=("chain", "draw")).values.flatten()
        return EmpiricalDistribution(samples)


np.random.seed(42)  

prior0 = NormalDistribution(mean=0, std_dev=1)  




data1 = np.random.normal(loc=2.0, scale=0.5, size=120)
sigma1 = 0.1


data2 = np.random.normal(loc=2.5, scale=0.6, size=120)
sigma2 = 0.1

data3 = np.random.normal(loc=3.0, scale=0.7, size=120)
sigma3 = 0.1



# Step 1: run workflow with initial prior and data1 
posterior1 = pp.run(prior_dist=prior0, data=data1, sigma=sigma1)
# Step 2: use posterior1 as prior for new data2
posterior2 = pp.run(prior_dist=posterior1, data=data2, sigma=sigma2)
# Step 3: chain again with posterior2 
posterior3 = pp.run(prior_dist=posterior2, data=data3, sigma=sigma3)


plt.figure(figsize=(10,6))

# Plot histogram (density=True normalizes the area to 1)
plt.hist(posterior1.samples, bins=30, alpha=0.4, density=True, label='Posterior 1')
plt.hist(posterior2.samples, bins=30, alpha=0.4, density=True, label='Posterior 2')
plt.hist(posterior3.samples, bins=30, alpha=0.4, density=True, label='Posterior 3')

plt.xlabel('alpha')
plt.ylabel('Density')
plt.title('Chained Posterior Distributions for alpha')
plt.legend()
plt.tight_layout()
plt.show()


from scipy.stats import gaussian_kde

xgrid = np.linspace(
    min(posterior1.samples.min(), posterior2.samples.min(), posterior3.samples.min()),
    max(posterior1.samples.max(), posterior2.samples.max(), posterior3.samples.max()),
    250
)
plt.figure(figsize=(10,6))
for i, posterior in enumerate([posterior1, posterior2, posterior3], 1):
    kde = gaussian_kde(posterior.samples)
    plt.plot(xgrid, kde(xgrid), lw=2, label=f'Posterior {i}')
plt.xlabel('alpha')
plt.ylabel('Density')
plt.title('Chained Posterior KDEs for alpha')
plt.legend()
plt.tight_layout()
plt.show()