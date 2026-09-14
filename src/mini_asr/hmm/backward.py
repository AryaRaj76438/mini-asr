import numpy as np

from mini_asr.hmm.hmm import HMM
from mini_asr.probability.numerical import logsumexp

def backward_log(
        model: HMM,
        observations: np.ndarray,
)->tuple[np.ndarray, float]:
    """
    model: HMM model
    observations: observation IDs with shape(T,)

    :return
        log_beta: backward probabilities in log-space, shape = (T, n_states)
        log_likelihood: log probability of the observation  sequence
    """
    observations = np.asarray(observations,  dtype=np.float64)

    if observations.ndim!=1:
        raise ValueError("observations  must  be 1-D")
    if len(observations)==0:
        raise ValueError("Observations cannot be empty")

    n_frames = len(observations)
    n_states = model.n_states

    log_transition = model.log_transition_prob()
    log_emission = model.emission_log_probability(observations)

    log_beta = np.empty((n_frames, n_states), dtype=np.float64)

    #Initialization, beta_T[i]=1, in log space it'll be 0
    log_beta[-1] = 0.0

    for t in range(n_frames-2, -1, -1):
        for state in range(n_states):
            next_states = log_transition[state] + log_emission[t+1] + log_beta[t+1]
            log_beta[t, state] = logsumexp(next_states)

    first_frame = model.log_initial_prob() + log_emission[0] + log_beta[0]

    log_likelihood = float(logsumexp(first_frame))

    return log_beta, log_likelihood

def backward(
        model: HMM,
        observations: np.ndarray,
)->float:
    """
    Return the probability of an  observation sequence using the backward  algorithm
    """

    _, log_likelihood = backward_log(model, observations)
    return float(np.exp(log_likelihood))
