import numpy as np

def mean_variance_normalize(
        features: np.ndarray,
        eps:int=1e-8,
)->np.ndarray:
    """
    Normalize each feature dimension to approximately
    zero mean and unit variance.
    """
    if features.ndim!=2:
        raise ValueError("features must be 2-D")

    mean = np.mean(features, axis=0, keepdims=True)
    std = np.std(features, axis=0, keepdims=True)

    return (features-mean)/np.maximum(std, eps)