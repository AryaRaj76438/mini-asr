from __future__ import annotations

import numpy as np

from mini_asr.hmm.hmm import HMM
from mini_asr.hmm.forward import forward_log
from mini_asr.hmm.backward import backward_log

def _validate_observations(observations: np.ndarray)->np.ndarray:
    observations = np.asarray(observations, dtype=np.float64)

    if observations.ndim != 1:
        raise ValueError("Observations must be 1-D")
    if len(observations)==0:
        raise ValueError("Observations cannot be  empty")
    return observations

def forward_backward_log(
        model: HMM,
        observations: np.ndarray,
)->tuple[np.ndarray, np.ndarray, float]:
    """
    Return:
        log_alpha: forward probabilities in log-space
        log_beta: backward probabilities in log-space
        log_likelihood: log probability of the observation sequence
    """
    log_alpha, forward_likelihood = forward_log(model, observations)
    log_beta, backward_likelihood = backward_log(model, observations)

    if not np.isclose(forward_likelihood, backward_likelihood, rtol=1e-10, atol=1e-10):
        raise RuntimeError("Forward and Backward log-likehoods donot match")
    return log_alpha, log_beta, forward_likelihood