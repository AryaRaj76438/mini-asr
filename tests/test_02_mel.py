import matplotlib.pyplot as plt
import numpy as np

from mini_asr.features.mel import (create_mel_filterbank, mel_to_hz)

sample_rate = 16000
n_fft  = 400
n_mels = 40

filters = create_mel_filterbank(sample_rate, n_fft, n_mels)
frequencies = np.linspace(0, sample_rate/2, n_fft//2 +1)

plt.figure(figsize=(12,6))

for filter_bank in filters:
    plt.plot(frequencies, filter_bank)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Filter Weight")
plt.title("Mel Filter Bank")
plt.tight_layout()
plt.show()