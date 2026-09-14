import numpy as np
from scipy.special.cython_special import log_wright_bessel

from mini_asr.hmm.hmm import HMM
from mini_asr.hmm.algorithms import forward_backward_log

def state_posteriors(
        model: HMM,
        observations: np.ndarray,
)->np.ndarray:
    """
    Compute gamma: gamma[t,i] = P(s_t=i|O))
    Parameters:
        model: HMM model
        observations: observation sequence of shape (T,).
    return
        gamma: state posterior probabilities of shape (T, n_states)
    """
    observations = np.asarray(observations,dtype=np.float64)

    log_alpha, log_beta, log_likelihood = forward_backward_log(model, observations)

    log_gamma = log_alpha + log_beta - log_likelihood
    gamma = np.exp(log_gamma)
    gamma /= gamma.sum(axis=1, keepdims=True)

    return gamma

def transition_posteriors(
        model: HMM,
        observations: np.ndarray,
)->np.ndarray:
    """
    Compute xi: xi[t,i,j] = P(s_t=i, s_{t+1}=j|O)
    return:
        xi: shape (T-1, n_states, n_states)
    """

    observations = np.asarray(observations, dtype=np.int64)

    if observations.ndim!=1:
        raise ValueError("observations must be 1-D")
    if len(observations)==0:
        raise ValueError("observations cannot be empty.")

    log_transition = model.log_transition_prob()
    log_emission = model.emission_log_probability(observations)

    n_frames = len(observations)
    n_states = model.n_states

    if n_frames==1:
        return np.empty((0, n_states, n_states), dtype=np.float64)

    log_alpha, log_beta, log_likelihood = forward_backward_log(model, observations)

    xi = np.empty((n_frames-1, n_states, n_states), dtype=np.float64)

    for t in range(n_frames-1):
        log_xi = (
                log_alpha[t, :, None]
                + log_transition
                + log_emission[t+1, None, :]
                + log_beta[t+1,None, :]
                -log_likelihood
        )
        xi[t] = np.exp(log_xi)
        total = xi[t].sum()
        if total >0.0:
            xi[t]/=total
    return xi
