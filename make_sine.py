import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

amplitude = 30000
samplerate = 44100
time = 20. # seconds
note = 0.5 # seconds
A = 440.
hz = []

foo = int(time / note - 0.5 * (time / note))

for n in range(-foo, foo):
    freq = A * (2**(1./12))**n
    hz.append(freq)

waves = []

for freq in hz:
    waves.append(np.sin(2 * np.pi * freq * np.linspace(0, note, note*samplerate)))

wave = np.concatenate(waves)

wave = amplitude * wave #/ len(hz)
wavfile.write('sines.wav', samplerate, wave.astype(np.int16))



