from mini_asr.audio.io import load_audio
from mini_asr.features.mfcc import (compute_mfcc,compute_mfcc_features)

waveform, sample_rate = load_audio("/Users/arya/PycharmProjects/mini-asr/data/raw/sample.wav")

features = compute_mfcc(waveform, sample_rate, n_mfcc=13)
print(features)
print(f"shape: {features.shape}")

features = compute_mfcc_features(waveform, sample_rate)
print(features)
print(f"shape: {features.shape}")