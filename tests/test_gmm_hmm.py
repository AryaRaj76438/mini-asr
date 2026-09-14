import numpy as np

from mini_asr.hmm import GMMHMM
from mini_asr.hmm.viterbi import viterbi

def create_model()->GMMHMM:
    initial_prob = np.array([0.6,0.4])

    transition_prob = np.array([
        [0.7, 0.3],
        [0.4, 0.6]
    ])

    return GMMHMM(
        initial_prob=initial_prob,
        transition_prob=transition_prob,
        n_components=2,
        max_gmm_iterations=20,
        random_state=42
    )

def test_gmm_hmm_viterbi():
    rng = np.random.default_rng(42)
    X = np.vstack([
        rng.normal(loc=3.0, scale=0.5, size=(50, 2)),
        rng.normal(loc=3.0, scale=0.5, size=(50, 2))
    ])

    model = create_model()
    model.fit(X)
    states, log_probability = viterbi(model, X)
    assert states.shape==(100,)
    assert np.all(states>=0)
    assert np.all(states<model.n_states)
    assert np.isfinite(log_probability)

def test_gmm_hmm_fit():
    rng = np.random.default_rng(42)
    X = np.vstack([
        rng.normal(loc=-3.0, scale=0.5, size=(50,2)),
        rng.normal(loc=3.0, scale=0.5, size=(50,2))
    ])

    model = create_model()
    model.fit(X)
    assert len(model.gmms)==2

    for gmm in  model.gmms:
        assert gmm.means_ is  not None
        assert gmm.variances_ is not None
        assert gmm.weights_ is not None

def test_emission_probability_shape():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50,2))
    model = create_model()
    model.fit(X)

    result = model.emission_log_probability(X)
    assert result.shape == (50,2)
    assert np.isfinite(result).all()

def test_log_parameters():
    model = create_model()
    assert np.allclose(np.exp(model.log_initial_prob()), model.initial_prob)

