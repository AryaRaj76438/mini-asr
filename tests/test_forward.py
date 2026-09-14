import numpy as np

from mini_asr.hmm import HMM
from mini_asr.hmm.forward import forward, forward_log


def create_hmm() -> HMM:
    return HMM(
        initial_prob=np.array([0.6, 0.4]),
        transition_prob=np.array([
            [0.7, 0.3],
            [0.4, 0.6],
        ]),
        emission_prob=np.array([
            [0.9, 0.1],
            [0.2, 0.8],
        ]),
    )


def test_forward_matches_direct_probability():
    model = create_hmm()
    observations = np.array([0, 1])

    expected = model.probability_of_observation_sequence(observations)
    result = forward(model, observations)

    assert np.isclose(result, expected)


def test_forward_log_matches_probability():
    model = create_hmm()
    observations = np.array([0, 1, 0])

    probability = forward(model, observations)
    _, log_probability = forward_log(model, observations)

    assert np.isclose(np.exp(log_probability), probability)


def test_forward_shape():
    model = create_hmm()
    observations = np.array([0, 1, 0, 1])

    log_alpha, _ = forward_log(model, observations)

    assert log_alpha.shape == (4, 2)


def test_forward_values_are_finite():
    model = create_hmm()
    observations = np.array([0, 1, 0, 1])

    log_alpha, log_likelihood = forward_log(model, observations)

    assert np.isfinite(log_alpha).all()
    assert np.isfinite(log_likelihood)