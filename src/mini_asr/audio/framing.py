import numpy as np

def frame_signal(
    waveform: np.ndarray,
    sample_rate: int,
    frame_duration_ms: float = 25.0,
    hop_duration_ms: float = 10.0,
) -> np.ndarray:
    """
    Split a waveform into overlapping frames.
    Returns
    -------
    frames: Array with shape (num_frames, frame_length).
    """
    if waveform.ndim != 1:
        raise ValueError("waveform must be 1-D")

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    if frame_duration_ms <= 0 or hop_duration_ms <= 0:
        raise ValueError("frame and hop durations must be positive")

    frame_length = int(round(sample_rate*frame_duration_ms/1000))
    hop_length = int(round(sample_rate * hop_duration_ms / 1000))

    if len(waveform)<frame_length:
        raise ValueError("waveform is shorter than  one frame")

    num_frames = 1 + (len(waveform) - frame_length) // hop_length

    frames = np.empty(
        (num_frames, frame_length),
        dtype=waveform.dtype
    )

    for i in range(num_frames):
        start = i*hop_length
        end = start + frame_length
        frames[i] = waveform[start: end]

    return frames

def apply_hann_window(frames: np.ndarray)->np.ndarray:
    """
    Apply a Hann window to every frame.
    """
    if frames.ndim != 2:
        raise ValueError("frame should be 2-D")

    window = np.hanning(frames.shape[1])

    return frames*window

def pre_emphasis(
        waveform:np.ndarray,
        coefficient:float=0.97
)->np.ndarray:
    """
    Apply first order pre-emphasis filter
    """
    if waveform.ndim!=1:
        raise ValueError("waveform must be 1-D")

    if not 0.0<=coefficient<=1.0:
        raise ValueError("coefficient must be [0,1]")

    emphasized = np.empty_like(waveform)
    emphasized[0] = waveform[0]
    emphasized[1:] = waveform[1:]-coefficient*waveform[:-1]
    return emphasized
