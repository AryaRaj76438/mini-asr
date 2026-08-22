from pathlib import Path

import numpy as np
import soundfile as sf

def load_audio(path: str|Path)->tuple[np.ndarray, int]:
    """
    Load an audio file.

    Returns
    -------
    waveform:Audio samples as a 1-D float32 NumPy array.
    sample_rate: Number of samples per second.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    waveform, sample_rate = sf.read(path, dtype="float32")

    if waveform.ndim>1:
        waveform = np.mean(waveform, axis=1)

    return waveform, sample_rate

def audio_duration(waveform: np.ndarray, sample_rate: int)->float:
    """Return audio duration in seconds."""
    if sample_rate<=0:
        raise ValueError("Sample rate should be positive")
    return len(waveform)/sample_rate