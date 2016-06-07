import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft, ifft
import numpy as np

data = wavfile.read('data/1.wav')[1][:44000]
window = 1000
step = 500
current = 0
fourier = []

# XXX fix last window
while current < 44000:
    fourier.append(fft(data[current:current+window]))
    current += step

print len(fourier)
plt.plot(np.real(fourier[10]))
plt.show()

# XXX matrix will be 15 (freq cutoff) * 88 (nr windows) * 2 (real + complex)

