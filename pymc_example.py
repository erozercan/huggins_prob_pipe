import pymc as pm
import numpy as np

x = np.linspace(0, 1, 10)
y = 1.5 * x + 0.1 * np.random.randn(10)

with pm.Model() as model:


    # alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=1)

    mu =  beta * x

    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)




trace = pm.sample(model=model)



print(pm.summary(trace))
print(type(model))
print(model)