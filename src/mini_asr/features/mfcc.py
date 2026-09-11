import numpy as np

from mini_asr.audio.framing import (apply_hann_window, frame_signal)
from mini_asr.audio.stft import (power_spectrum, stft)
from mini_asr.features.mel import (create_mel_filterbank,log_mel_spectrogram, mel_spectrogram)
from mini_asr.features.delta import (delta_features, delta_delta_features)

import numpy as np

from mini_asr.features.normalization import mean_variance_normalize


def dct_type_2(
    x: np.ndarray,
    n_coefficients: int = 13,
) -> np.ndarray:
    """
    Compute the DCT-II along the last dimension.

    Parameters
    ----------
    x:
        Input array with shape (..., n).

    n_coefficients:
        Number of DCT coefficients to retain.
        Must satisfy:
            1 <= n_coefficients <= n

    Returns
    -------
    np.ndarray
        DCT-II coefficients with shape (..., n_coefficients).

    Notes
    -----
    This implements the unnormalized DCT-II:

        X[k] = sum_{n=0}^{N-1}
               x[n] * cos(pi / N * (n + 0.5) * k)

    where k = 0, 1, ..., n_coefficients - 1.
    """

    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")

    if n_coefficients < 1:
        raise ValueError(
            "n_coefficients must be at least 1"
        )

    n = x.shape[-1]

    if n < 1:
        raise ValueError(
            "input must have a non-empty last dimension"
        )

    if n_coefficients > n:
        raise ValueError(
            "n_coefficients cannot be greater than input dimension"
        )

    # DCT coefficient indices:
    # [0, 1, 2, ..., n_coefficients - 1]
    indices = np.arange(n_coefficients)[:, None]

    # Input sample indices:
    # [0, 1, 2, ..., n - 1]
    samples = np.arange(n)[None, :]

    # DCT-II basis matrix
    #
    # Shape:
    # (n_coefficients, n)
    basis = np.cos(
        np.pi / n
        * (samples + 0.5)
        * indices
    )

    # x shape:
    # (..., n)
    #
    # basis.T shape:
    # (n, n_coefficients)
    #
    # result shape:
    # (..., n_coefficients)
    result = x @ basis.T

    return result

def dct(
        mel_energy: np.ndarray,
        num_ceps: int=13,
)->np.ndarray:
    """
    Parameters:
        mel_energy: Shape:(n_mels, )
        num_ceps: number of coefficients to return

    :return
    mfcc: DCT Coefficients: Shape(num_ceps,)
    """

    print(f"shape of mel_energy: {mel_energy.shape}")
    if mel_energy.ndim != 1:
        raise ValueError("mel_energy must be a 1D array")

    n = mel_energy.shape[0]

    if num_ceps>n:
        raise ValueError("num_ceps cannot be greater than number of mel-filters")
    samples = np.arange(n)[None, :]
    indices = np.arange(num_ceps)[:, None]

    basis = np.cos(np.pi/n*(samples*0.5)*indices)
    coefficients = basis@mel_energy
    return coefficients

def mfcc_from_log_mel(
    log_mel: np.ndarray,
    n_mfcc: int = 13,
) -> np.ndarray:
    """
    Compute MFCC coefficients from log-Mel energies.
    """
    if log_mel.ndim != 2:
        raise ValueError("log_mel must be 2-D")

    return dct_type_2(log_mel, n_coefficients=n_mfcc)

def compute_mfcc(
    waveform: np.ndarray,
    sample_rate: int,
    n_fft: int = 400,
    n_mels: int = 40,
    n_mfcc: int = 13,
    frame_duration_ms: float = 25.0,
    hop_duration_ms: float = 10.0,
) -> np.ndarray:
    """
    Compute MFCC features from a waveform.
    """
    frames = frame_signal(waveform,sample_rate,frame_duration_ms,hop_duration_ms)
    frames = apply_hann_window(frames)
    spectrum = stft(frames, n_fft)
    power = power_spectrum(spectrum)
    filterbank = create_mel_filterbank(sample_rate, n_fft, n_mels)
    mel_energy = mel_spectrogram(power, filterbank)
    log_mel = log_mel_spectrogram(mel_energy)

    return mfcc_from_log_mel(log_mel, n_mfcc=n_mfcc)


def compute_mfcc_features(
        waveform:np.ndarray,
        sample_rate:int=16000,
        n_fft:int=400,
        n_mels:int=40,
        n_mfcc:int=13,
        frame_duration_ms: float = 25.0,
        hop_duration_ms: float = 10.0,
        include_delta: bool = True,
        include_delta_delta: bool = True,
        normalize: bool = False,
)->np.ndarray:
    """Compute MFCC, delta, delta-delta features"""
    static = compute_mfcc(waveform, sample_rate, n_fft, n_mels, n_mfcc,
                          frame_duration_ms, hop_duration_ms)
    feature_list = [static]

    if include_delta:
        feature_list.append(delta_features(features=static))

    if include_delta_delta:
        feature_list.append(delta_delta_features(static))

    features = np.concatenate(feature_list, axis=1)
    if normalize:
        features = mean_variance_normalize(features)
    return features