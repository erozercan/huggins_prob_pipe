data {
  int<lower=0> N;         // number of observations
  vector[N] x;            // predictor
  vector[N] y;            // outcome
}

parameters {
  real alpha;             // intercept
  real beta;              // slope
  real<lower=0> sigma;    // noise scale
}

model {
  alpha ~ normal(0, 10);
  beta ~ normal(0, 10);
  sigma ~ normal(0, 5);

  y ~ normal(alpha + beta * x, sigma);
}


