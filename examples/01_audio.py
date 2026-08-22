import matplotlib.pyplot as plt

from mini_asr.audio.framing import (
    apply_hann_window,
    frame_signal,
)
from mini_asr.audio.io import (
    audio_duration,
    load_audio,
)
from mini_asr.audio.stft import (
    power_spectrum,
    stft,
)


audio_path = "/Users/arya/PycharmProjects/mini-asr/data/raw/sample.wav"

waveform, sample_rate = load_audio(audio_path)

print(f"Sample rate: {sample_rate} Hz")
print(f"Samples: {len(waveform)}")
print(f"Duration: {audio_duration(waveform, sample_rate):.2f} seconds")

frames = frame_signal(
    waveform,
    sample_rate,
    frame_duration_ms=25,
    hop_duration_ms=10,
)

print(f"Frames: {frames.shape}")

windowed_frames = apply_hann_window(frames)

spectrum = stft(windowed_frames)

power = power_spectrum(spectrum)

print(f"Spectrum shape: {spectrum.shape}")
print(f"Power spectrum shape: {power.shape}")


plt.figure(figsize=(10, 4))
plt.plot(waveform)
plt.title("Waveform")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 4))
plt.imshow(
    power.T,
    origin="lower",
    aspect="auto",
)
plt.title("Power Spectrogram")
plt.xlabel("Frame")
plt.ylabel("Frequency Bin")
plt.colorbar(label="Power")
plt.tight_layout()
plt.show()