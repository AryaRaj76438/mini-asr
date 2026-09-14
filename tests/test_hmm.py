
import numpy as np
import pytest

from mini_asr.hmm import HMM

def create_hmm()->HMM:
    initial_prob = np.array([0.6,0.4])

    transition_prob = np.array([
        [0.7, 0.3],
        [0.4, 0.6]
    ])
    emission_prob = np.array([
        [0.9,0.1],
        [0.2,0.8]
    ])
    return HMM(
        initial_prob=initial_prob,
        transition_prob=transition_prob,
        emission_prob=emission_prob
    )

def test_hmm_dimension():
    model = create_hmm()

    assert model.n_states == 2
    assert model.n_observations==2

def test_transition_rows_sum_to_one():
    model = create_hmm()

    assert np.allclose(model.transition_prob.sum(axis=1), 1.0)

def test_emission_rows_sum_to_one():
    model = create_hmm()

    assert np.allclose(model.emission_prob.sum(axis=1),1.0)

def test_emission_log_prob():
    model = create_hmm()

    observations = np.array([0,1,0])
    result = model.emission_log_probability(observations)
    assert result.shape==(3,2)

    expected = np.log(np.array([
        [0.9, 0.2],
        [0.1, 0.8],
        [0.9, 0.2]
    ]))

    assert np.allclose(result, expected)

def test_probability_of_sequence():
    model = create_hmm()
    observations = np.array([0,1])

    probability = model.probability_of_observation_sequence(observations)
    expected = (
            0.6 * 0.9 * 0.7 * 0.1
            + 0.6 * 0.9 * 0.3 * 0.8
            + 0.4 * 0.2 * 0.4 * 0.1
            + 0.4 * 0.2 * 0.6 * 0.8
    )

    assert np.isclose(probability, expected)

def test_invalid_transition_matrix():
    initial_prob = np.array([0.5,0.5,])

    transition_prob = np.array([
        [0.8, 0.8],
        [0.4, 0.6],
    ])

    emission_prob = np.array([
        [0.9, 0.1],
        [0.2, 0.8],
    ])

    with pytest.raises(ValueError):
        HMM(
            initial_prob,
            transition_prob,
            emission_prob,
        )