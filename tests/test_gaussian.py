import numpy as np

from mini_asr.probability.gaussian import (
    gaussian_pdf,
    log_gaussian_pdf,
    log_multivariate_gaussian_pdf,
)

def test_gaussian_peak():
    probability = gaussian_pdf(
        x=0.0,
        mean=0.0,
        variance=1.0,
    )

    assert probability > 0

def test_log_gaussian_matches_pdf():
    x = 0.5
    pdf = gaussian_pdf(x=x,mean=0.0, variance=1.0)
    log_pdf = log_gaussian_pdf(x, mean=0.0, variance=1.0)
    assert np.allclose(log_pdf,  np.log(pdf))

def test_multivariate_gaussian():
    x = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
    ])

    mean = np.array([0.0, 0.0])
    variance = np.array([1.0, 1.0])
    result = log_multivariate_gaussian_pdf(x,mean,variance,)

    assert result.shape == (2,)
    assert np.all(np.isfinite(result))