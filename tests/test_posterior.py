import numpy as np

from mini_asr.hmm.hmm import HMM
from mini_asr.hmm.posterior import state_posteriors, transition_posteriors

def create_model():
    return HMM(
        initial_prob=np.array([0.6,0.4]),
        transition_prob=np.array([[0.7, 0.3], [0.4, 0.6]]),
        emission_prob=np.array([[0.9, 0.1], [0.2, 0.8]])
    )

def test_state_posteriors():
    gamma = state_posteriors(create_model(), np.array([0,1,0]))

    assert gamma.shape==(3,2)
    assert np.allclose(gamma.sum(axis=1), 1)
    assert np.all((0<=gamma) & (gamma<=1))

def test_transition_posteriors():
    xi = transition_posteriors(create_model(), np.array([0,1,0]))
    assert xi.shape==(2,2,2)
    assert np.allclose(xi.sum(axis=(1,2)), 1)
    assert np.all((0<=xi )& (xi<1))

def test_gamma_xi_consistency():
    model = create_model()
    observations = np.array([0,1,0])

    gamma = state_posteriors(model, observations)
    xi = transition_posteriors(model, observations)

    assert np.allclose(xi.sum(axis=2), gamma[:-1])
    assert np.allclose(xi.sum(axis=1), gamma[1:])
