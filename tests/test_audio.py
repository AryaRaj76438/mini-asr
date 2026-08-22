import numpy as np
from mini_asr.audio.io import audio_duration

def test_audio_duration():
    waveform = np.zeros(16000)
    sample_rate = 16000

    duration = audio_duration(waveform, sample_rate)

    assert duration == 1.0