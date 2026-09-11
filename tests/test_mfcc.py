from pyexpat import features

import numpy as np

from mini_asr.features.mfcc import compute_mfcc, compute_mfcc_features


def test_mfcc_shape():
    sample_rate = 16000

    waveform = np.random.default_rng(42).normal(
        0,
        1,
        sample_rate,
    ).astype(np.float32)

    features = compute_mfcc(
        waveform,
        sample_rate,
        n_fft=400,
        n_mels=40,
        n_mfcc=13,
    )

    assert features.ndim == 2
    assert features.shape[1] == 13


def test_mfcc_is_finite():
    sample_rate = 16000
    waveform = np.zeros(sample_rate,dtype=np.float32,)
    features = compute_mfcc(waveform,sample_rate,)
    assert np.all(np.isfinite(features))

def test_mfcc_with_deltas():
    sample_rate = 16000
    waveform = np.random.default_rng(42).normal(0,1, sample_rate,).astype(np.float32)
    features = compute_mfcc_features(waveform, sample_rate, n_mfcc=13, include_delta=True, include_delta_delta=True)
    assert features.shape[1]==39

