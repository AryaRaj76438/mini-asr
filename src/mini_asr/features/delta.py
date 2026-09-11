import numpy as np

def delta_features(
        features:np.ndarray,
        width:int=2,
)->np.ndarray:
    """
    Compute first-order temporal derivatives.
    Parameters
    ----------
    features:Shape: (num_frames, num_features)
    width: Number of frames used on each side.

    Returns
    -------
    delta: Same shape as features.
    """

    if features.ndim!=2:
        raise ValueError("features must be 2-D")

    if width<=0:
        raise ValueError("width must be positive")

    num_frames = features.shape[0]
    padded = np.pad(features, ((width,width), (0,0)),mode="edge")
    denominator = 2*sum(n*n for n in range(1, width+1))
    delta = np.zeros_like(features)

    for t in range(num_frames):
        numerator = np.zeros(features.shape[1],dtype=features.dtype)

        for n in range(1, width+1):
            numerator += n* (padded[t+width+n]-padded[t+width-n])

        delta[t] = numerator/denominator

    return delta

def delta_delta_features(
        features: np.ndarray,
        width: int=2
)->np.ndarray:
    """
        Compute second-order temporal derivatives.
    """
    delta = delta_features(features, width)
    return delta_features(delta, width)