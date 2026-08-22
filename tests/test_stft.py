import numpy as np

from mini_asr.audio.stft import power_spectrum, stft


def test_stft_shape():
    frames = np.zeros((10, 400))
    spectrum = stft(frames, n_fft=400)
    assert spectrum.shape == (10, 201)


def test_power_spectrum_non_negative():
    spectrum = np.array(
        [[1 + 2j, 3 + 4j]],
        dtype=np.complex128,
    )

    power = power_spectrum(spectrum)

    assert np.all(power >= 0)