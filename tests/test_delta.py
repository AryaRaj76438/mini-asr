import numpy as np

from mini_asr.features.delta import (delta_features, delta_delta_features)

def test_delta_shape():
    features = np.random.default_rng(42).normal(size=(100,13))
    delta = delta_features(features)
    assert delta.shape==features.shape

def test_delta_delta_shape():
    features = np.random.default_rng(42).normal(size=(100,13))
    delta_delta = delta_delta_features(features)
    assert delta_delta.shape==features.shape

def test_constant_signal_has_zero_delta():
    features = np.ones((20, 13))
    delta = delta_features(features)
    assert np.allclose(delta, 0.0)