import jax.numpy as jnp
import blackjax
from jax import random, lax


# Single NUTS step function
def nuts_sample(rng_key, logdensity, current_position, step_size=0.01):
    dim = current_position.shape[0]
    inverse_mass_matrix = jnp.ones(dim)
    nuts = blackjax.nuts(logdensity, step_size=step_size, inverse_mass_matrix=inverse_mass_matrix)

    state = nuts.init(current_position)
    state, info = nuts.step(rng_key, state)
    return state.position, info.acceptance_rate


# Vectorized chain sampling with lax.scan
def run_chain_nuts(key, initial_position, logdensity_fn, num_samples, step_size=0.01):
    def one_step(carry, rng_key):
        position = carry
        new_position, accept_rate = nuts_sample(rng_key, logdensity_fn, position, step_size)
        return new_position, (new_position, accept_rate)

    keys = random.split(key, num_samples)
    _, (samples, accept_rates) = lax.scan(one_step, initial_position, keys)
    return samples, accept_rates