import numpy as np

from mini_asr import probability
from mini_asr.hmm import HMM
from mini_asr.hmm.viterbi import (viterbi_log, viterbi)

def create_hmm()->HMM:
    return HMM(
        initial_prob=np.array([0.6,  0.4]),
        transition_prob=np.array([[0.7, 0.3], [0.4, 0.6]]),
        emission_prob=np.array([[0.9,  0.1], [0.2, 0.8]])
    )

def test_viterbi_shape():
    model = create_hmm()
    observations = np.array([0,1,0,1])

    states, probability = viterbi(model, observations)
    assert states.shape==(4, )
    assert np.isfinite(probability)

    # state validity
    assert np.all(states>=0)
    assert np.all(states<model.n_states)
