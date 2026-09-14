from cmath import sqrt

import numpy as np
from prompt_toolkit.layout import dimension


def gaussian_pdf(
    x: np.ndarray | float,
    mean: float,
    variance: float,
) -> np.ndarray:
    """
    Compute a one-dimensional Gaussian PDF.
    """
    if variance <= 0:
        raise ValueError("variance must be positive")

    x = np.asarray(x, dtype=np.float64)

    coefficient = 1.0 / np.sqrt(2.0 * np.pi * variance)
    exponent = -((x - mean) ** 2) / (2.0 * variance)

    return coefficient * np.exp(exponent)

def log_gaussian_pdf(
        x:np.ndarray,
        mean:float,
        variance:float
)-> np.ndarray:
    """
    compute log probability of a 1-D Gaussian
    """
    if variance<=0:
        raise ValueError("Variance must be  positive")
    x = np.asarray(x, dtype=np.float64)

    return (
        -(0.5*np.log(2.0*np.pi*variance))
        -((x-mean)**2)/(2.0*variance)
    )

def log_multivariate_gaussian_pdf(
        x:np.ndarray,
        mean:np.ndarray,
        variance:np.ndarray
)->np.ndarray:
    """
    Log probability for diagonal-covariance multivariate Gaussian.
    """
    x = np.asarray(x, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)

    if mean.ndim != 1:
        raise ValueError("mean must be 1-D")

    if variance.ndim != 1:
        raise ValueError("variance must be 1-D")

    if x.shape[-1] != mean.shape[0]:
        raise ValueError("Dimension mismatch")

    if variance.shape != mean.shape:
        raise ValueError("variance and mean must have same shape")

    if np.any(variance <= 0):
        raise ValueError("all variances must be positive")

    dimension = mean.shape[0]
    diff = x-mean
    quadratic = np.sum((diff**2)/variance, axis=-1)
    log_determinant = np.sum(np.log(variance))

    return -0.5 * (
            dimension * np.log(2.0 * np.pi)
            + log_determinant
            + quadratic
    )