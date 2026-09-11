import matplotlib.pyplot as plt
import numpy as np
from mini_asr.audio.io import load_audio
from mini_asr.features.mfcc import compute_mfcc_features
from tests.test import sample_rate

audio_path = "/Users/arya/PycharmProjects/mini-asr/data/raw/sample.wav"
waveform, sample_rate = load_audio(audio_path)
features = compute_mfcc_features(waveform, sample_rate, n_mfcc=13, include_delta=False, include_delta_delta=False)

plt.figure(figsize=(12,6))
plt.imshow(features.T,origin="lower",aspect="auto")

plt.xlabel("Frame")
plt.ylabel("MFCC coefficient")
plt.title("MFCC Features")
plt.colorbar(label="Value")

plt.tight_layout()
plt.show()