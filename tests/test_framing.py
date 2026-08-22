import numpy as np

from mini_asr.audio.framing import frame_signal


def test_frame_signal_shape():
    waveform = np.zeros(16000)

    frames = frame_signal(
        waveform,
        sample_rate=16000,
        frame_duration_ms=25,
        hop_duration_ms=10,
    )

    assert frames.shape[1] == 400


def test_frame_signal_overlap():
    waveform = np.arange(1000, dtype=np.float32)

    frames = frame_signal(
        waveform,
        sample_rate=1000,
        frame_duration_ms=100,
        hop_duration_ms=50,
    )

    assert frames[0, -1] == 99
    assert frames[1, 0] == 50