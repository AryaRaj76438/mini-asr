import numpy as np

def stft(frames: np.ndarray,
         n_fft: int | None = None,
)->np.ndarray:
    """
    Compute the real FFT of each frame.
    Parameters
    ----------
    frames: Shape: (num_frames, frame_length)

    n_fft: FFT size. Defaults to frame_length.

    Returns
    -------
    spectrum:
        Complex spectrum with shape
        (num_frames, n_fft // 2 + 1).
    """
    if frames.ndim!=2:
        raise ValueError("frames are not 2-D")

    if n_fft is None:
        n_fft = frames.shape[1]

    if n_fft<=0:
        raise ValueError("n_fft must be positive")

    return np.fft.rfft(frames, n=n_fft, axis=1)

def power_spectrum(spectrum: np.ndarray) -> np.ndarray:
    """
    Convert complex FFT output to power spectrum.
    """
    return (np.abs(spectrum) ** 2)