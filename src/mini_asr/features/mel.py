import numpy as np


def hz_to_mel(frequency_hz: float | np.ndarray) -> float | np.ndarray:
    """Convert frequency from Hertz to Mel."""
    return 2595.0 * np.log10(1.0 + np.asarray(frequency_hz) / 700.0)


def mel_to_hz(mel: float | np.ndarray) -> float | np.ndarray:
    """Convert frequency from Mel to Hertz."""
    return 700.0 * (
        10.0 ** (np.asarray(mel) / 2595.0) - 1.0
    )


def create_mel_filterbank(
        sample_rate:int,
        n_fft:int,
        n_mels:int,
        f_min: float=0.0,
        f_max: float|None =None,
)->np.ndarray:
    """
    Create triangular Mel filter banks.
    Returns
    -------
    filters:
        Shape: (n_mels, n_fft // 2 + 1)
    """
    if f_max is None:
        f_max = sample_rate/2.0

    if f_min<0:
        raise ValueError("f_min must be positive")

    if f_max<=f_min:
        raise ValueError("f_max must be greater than f_min")

    if f_max>sample_rate/2:
        raise ValueError("f_max can't exceeds Nyquist frequency")

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)

    mel_points = np.linspace(mel_min,  mel_max, n_mels+2)

    hz_points = mel_to_hz(mel_points)

    bin_points = np.floor(
        (n_fft+1)*hz_points/sample_rate
    ).astype(int)

    n_freq_bins = n_fft//2+1

    filters = np.zeros((n_mels, n_freq_bins), dtype=np.float64)

    for m in range(1, n_mels+1):
        left = bin_points[m-1]
        center = bin_points[m]
        right = bin_points[m+1]

        center = max(center, left+1)
        right = max(center+1, right)

        right = min(right, n_freq_bins-1)

        for k in range(left, center):
            filters[m-1, k] = (k-left)/(center-left)

        for k in range(center, right):
            filters[m-1, k] = (right-k)/(right-center)

    return filters


def mel_spectrogram(
        power_spectrum: np.ndarray,
        filterbank: np.ndarray
)->np.ndarray:
    """
        Apply Mel filter banks to a power spectrogram.
        Parameters
        ----------
        power_spectrum:
            Shape: (num_frames, frequency_bins)
        filterbank:
            Shape: (n_mels, frequency_bins)
        Returns
        -------
        mel_energy:
            Shape: (num_frames, n_mels)
    """
    if power_spectrum.ndim != 2:
        raise ValueError("power_spectrum must be 2-D")

    if filterbank.ndim != 2:
        raise ValueError("filterbank must be 2-D")

    if power_spectrum.shape[1] != filterbank.shape[1]:
        raise ValueError("Frequency dimensions do not match")

    return power_spectrum @ filterbank.T

def log_mel_spectrogram(
        mel_energy: np.ndarray,
        eps:  float=1e-10
)->np.ndarray:
    """Convert Mel energies to log-Mel energies."""
    if eps<=0:
        raise ValueError("eps should be positive")

    return np.log(np.maximum(mel_energy, eps))