import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

amplitude = 30000
time = 5
samplerate = 44100
hz = [440]
wave = np.zeros(time * samplerate)

for freq in hz:
    wave += np.sin(2 * np.pi * freq * np.linspace(0, time, time  * samplerate))

wave = amplitude * wave / len(hz)
wavfile.write('sines.wav', samplerate, wave.astype(np.int16))



