import numpy as np

from mini_asr.hmm import GMMHMM
from mini_asr.hmm.hmm import HMM

def viterbi_log(
        model: HMM,
        observations: np.ndarray,
)->tuple[np.ndarray, float]:
    """
    Find the most likely hidden state sequence using  the viterbi algorithm in log-space.
    Parameter:
        model: HMM-model
        observations: Observation IDs with shape(T,)
    return:
        states: most likely state sequence, shape(T,)
        log_probability: log probability of the best state path
    """
    observations = np.asarray(observations, dtype=np.float64)
    # if observations.ndim!=1:
    #     raise ValueError("observations must be 1-D")
    if len(observations)==0:
        raise ValueError("Observations cannot be empty")

    log_initial = model.log_initial_prob()
    log_transition = model.log_transition_prob()
    log_emission = model.emission_log_probability(observations)

    n_frames = len(observations)
    n_states = model.n_states

    if log_emission.shape!=(n_frames, n_states):
        raise ValueError("emission_log_probability returned an unexpected shape")

    log_delta = np.empty((n_frames, n_states), dtype=np.float64)
    backpointer = np.empty((n_frames, n_states), dtype=np.int64)

    log_delta[0] = log_initial+log_emission[0]
    backpointer[0] = -1

    for t in range(1, n_frames):
        for state in range(n_states):
            scores = log_delta[t-1]+log_transition[:, state]
            best_previous_state = np.argmax(scores)

            log_delta[t, state] = scores[best_previous_state]+log_emission[t, state]
            backpointer[t, state] = best_previous_state

    final_state = int(np.argmax(log_delta[-1]))
    log_probability = float(log_delta[-1, final_state])

    states = np.empty(n_frames, dtype=np.int64)
    states[-1] = final_state

    for t in  range(n_frames-1, 0, -1):
        states[t-1] = backpointer[t, states[t]]

    return states, log_probability

def viterbi(model: HMM, observations: np.ndarray)->tuple[np.ndarray, float]:
    """
    Convenience wrapper around viterbi_log()
    return:
        states: most likely state sequence
        probability: probability of the best state path
    """
    states, log_probability = viterbi_log(model, observations)
    return states, float(np.exp(log_probability))