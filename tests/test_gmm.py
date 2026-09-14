import numpy as np

from mini_asr.probability.gmm import GMM


def test_gmm_fit():
    rng = np.random.default_rng(42)

    cluster_1 = rng.normal(loc=-3.0, scale=0.5, size=(100, 2))
    cluster_2 = rng.normal(loc=3.0, scale=0.5, size=(100, 2))
    X = np.vstack([cluster_1, cluster_2])

    model = GMM(n_components=2, random_state=42)
    model.fit(X)

    assert model.weights_ is not None
    assert model.means_ is not None
    assert model.variances_ is not None

    assert model.weights_.shape == (2,)
    assert model.means_.shape == (2, 2)
    assert model.variances_.shape == (2, 2)


def test_gmm_weights_sum_to_one():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 2))

    model = GMM(n_components=3, random_state=42)
    model.fit(X)

    assert np.allclose(np.sum(model.weights_), 1.0)


def test_responsibilities_sum_to_one():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 2))

    model = GMM(n_components=3, random_state=42)
    model.fit(X)

    responsibilities = model.predict_prob(X)

    assert responsibilities.shape == (100, 3)
    assert np.allclose(responsibilities.sum(axis=1), 1.0)


def test_score_samples():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 2))

    model = GMM(n_components=2, random_state=42)
    model.fit(X)

    scores = model.score_samples(X)

    assert scores.shape == (50,)
    assert np.isfinite(scores).all()