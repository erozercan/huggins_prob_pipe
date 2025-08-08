import numpy as np
from typing import Callable, Iterator, Optional

def metropolis_hastings_iter(
    log_prob_fn: Callable[[float], float],
    initial_state: float,
    proposal_std: float,
    random_seed: Optional[int] = None
) -> Iterator[float]:
    """
    Metropolis-Hastings MCMC sampler as an iterator/generator.

    Parameters
    ----------
    log_prob_fn: Callable[[float], float]
        Function returning the log-probability of a state.
    initial_state: float
        Starting point of the Markov chain.
    proposal_std: float
        Standard deviation of the Gaussian proposal distribution.
    random_seed: Optional[int]
        Seed for reproducibility.

    Yields
    ------
    float
        Next MCMC sample.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    current = initial_state
    current_log_prob = log_prob_fn(current)

    while True:
        # Propose new candidate from N(current, proposal_std^2)
        candidate = np.random.normal(current, proposal_std)
        candidate_log_prob = log_prob_fn(candidate)

        # Acceptance log-ratio
        log_accept_ratio = candidate_log_prob - current_log_prob

        # Accept or reject
        if np.log(np.random.rand()) < log_accept_ratio:
            current = candidate
            current_log_prob = candidate_log_prob

        yield current


def metropolis_hastings(
    log_prob_fn: Callable[[float], float],
    initial_state: float,
    n_samples: int,
    proposal_std: float,
    burn_in: int,
    random_seed: Optional[int] = None) -> np.ndarray:
    """
    Runs the metropolis_hastings_iter generator to produce a fixed number of samples.

    Returns all samples in a numpy array.
    """
    sampler = metropolis_hastings_iter(log_prob_fn, initial_state, proposal_std, random_seed)
    samples = np.empty(n_samples)
    for i in range(n_samples):
        samples[i] = next(sampler)
    samples = samples[burn_in:]  # Discard burn-in samples
    return samples  