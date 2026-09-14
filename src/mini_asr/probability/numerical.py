from unittest import result

import numpy as np

def safe_log(
        x: np.ndarray|float,
        eps: float=1e-300
)->np.ndarray:
    """
    Compute log(x) while avoiding log(0)
    """
    if eps<=0:
        raise ValueError("eps must be positive")

    return np.log(np.maximum(x, eps))

def logsumexp(x: np.ndarray, axis=None) -> np.ndarray:
    """
    Numerically stable computation of log(sum(exp(x))).
    """
    x = np.asarray(x, dtype=np.float64)

    if x.size == 0:
        raise ValueError("x cannot be empty")

    max_x = np.max(x, axis=axis, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        result = max_x + np.log(np.sum(np.exp(x - max_x), axis=axis, keepdims=True))

    result = np.where(np.isneginf(max_x), -np.inf, result)

    if axis is not None:
        result = np.squeeze(result, axis=axis)
    else:
        result = result.squeeze()

    return result
