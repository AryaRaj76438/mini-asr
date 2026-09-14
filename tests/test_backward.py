from statistics import mode

import numpy as np

from mini_asr.hmm import HMM
from mini_asr.hmm.backward import (backward_log, backward)
from mini_asr.hmm.forward import forward


def create_hmm()->HMM:
    return HMM(
        initial_prob=np.array([0.6,  0.4]),
        transition_prob=np.array([[0.7, 0.3], [0.4, 0.6]]),
        emission_prob=np.array([[0.9,  0.1], [0.2, 0.8]])
    )

def test_backward_matches_forward():
    model = create_hmm()
    observations = np.array([0, 1, 0])
    forward_probability = forward(model, observations)
    backward_probability = backward(model, observations)

    assert np.isclose(forward_probability,backward_probability)

def test_backward_log_matches_probability():
    model = create_hmm()
    observations = np.array([0,1,0])
    probability = backward(model, observations)

    _, log_probability = backward_log(model, observations)

    assert np.isclose(np.exp(log_probability), probability)

def test_backward_shape():
    model = create_hmm()

    observations = np.array([0,1,0,1])
    log_beta, _ = backward_log(model, observations)
    assert log_beta.shape == (4,2)

def test_backward_final_values():
    model = create_hmm()
    observations = np.array([0,1,0])
    log_beta, _ = backward_log(model, observations)
    assert np.allclose(log_beta[-1], 0.0)
