import matplotlib.pyplot as plt

from mini_asr.audio.io import load_audio
from mini_asr.features.delta import (
    delta_delta_features,
    delta_features,
)
from mini_asr.features.mfcc import compute_mfcc


waveform, sample_rate = load_audio("/Users/arya/PycharmProjects/mini-asr/data/raw/sample.wav")

mfcc = compute_mfcc(
    waveform,
    sample_rate,
)

delta = delta_features(mfcc)

delta_delta = delta_delta_features(mfcc)


plt.figure(figsize=(12, 5))
plt.imshow(
    mfcc.T,
    origin="lower",
    aspect="auto",
)
plt.title("MFCC")
plt.xlabel("Frame")
plt.ylabel("Coefficient")
plt.colorbar()
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 5))
plt.imshow(
    delta.T,
    origin="lower",
    aspect="auto",
)
plt.title("Delta")
plt.xlabel("Frame")
plt.ylabel("Coefficient")
plt.colorbar()
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 5))
plt.imshow(
    delta_delta.T,
    origin="lower",
    aspect="auto",
)
plt.title("Delta-Delta")
plt.xlabel("Frame")
plt.ylabel("Coefficient")
plt.colorbar()
plt.tight_layout()
plt.show()