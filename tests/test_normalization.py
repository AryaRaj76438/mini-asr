import numpy as np
from mini_asr.features.normalization import mean_variance_normalize

def test_normalization():
    features = np.random.default_rng(42).normal(size=(1000,13))
    normalized = mean_variance_normalize(features)
    means = np.mean(normalized, axis=0)
    std = np.std(normalized, axis=0)

    assert np.allclose(means, 0.0, atol=1e-6)
    assert np.allclose(std, 1.0, atol=1e-6)


def test_constant_features_are_finite():
    features = np.ones((20, 13))
    normalized = mean_variance_normalize(features)
    assert np.all(np.isfinite(normalized))