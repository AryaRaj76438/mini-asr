import numpy as np

from mini_asr.hmm import HMM
from mini_asr.probability.numerical import logsumexp

def forward_log(
        model: HMM,
        observations: np.ndarray,
)->tuple[np.ndarray, float]:
    """
    Forward algorithm in log-space
    Parameters:
        model: HMM model
        observations: observation IDs with shape (T,)
    Return:
        log_alpha: Forward probabilities in log-space, shape: (T, n_states)
        log_likelihood: log probability of the complete observation sequence
    """

    observations = np.asarray(observations, dtype=np.int64)

    if observations.ndim !=1:
        raise ValueError("observations must be 1-D")

    if len(observations)==0:
        raise ValueError("observations cannot be empty")

    log_initial = model.log_initial_prob()
    log_transition = model.log_transition_prob()
    log_emission = model.emission_log_probability(observations)

    n_frames = len(observations)
    n_states =  model.n_states

    log_alpha = np.empty((n_frames, n_states), dtype=np.float64)

    log_alpha[0] = log_initial + log_emission[0]

    for t in range(1, n_frames):
        for state in range(n_states):
            previous = log_alpha[t - 1] + log_transition[:, state]
            log_alpha[t, state] = log_emission[t, state] + logsumexp(previous)

    log_likelihood = float(logsumexp(log_alpha[-1]))

    return log_alpha, log_likelihood


def forward(
        model: HMM,
        observations: np.ndarray,
)-> float:
    """
    Return the probability of observation sequence
    """
    _, log_likelihood = forward_log(model, observations)

    return float(np.exp(log_likelihood))