from keras.layers import Convolution1D, UpSampling1D, AveragePooling1D
from keras.models import Sequential
from keras.optimizers import SGD
from scipy.io import wavfile
import numpy as np
from scipy.fftpack import fft
from sklearn.metrics import mean_squared_error
import os
import matplotlib.pyplot as plt


train = False
downsample_factor = 10
np.random.seed(41125)

one = wavfile.read("data/train/1.wav")[1]
max = -np.min(one)
one = one[::downsample_factor] / max
one = one[:880]
sample_rate = len(one)

xf = np.linspace(0.0, 1.0/(2.0 * 1.0/880), 880/2)
one = fft(one[:880])
fig, ax = plt.subplots()
ax.plot(xf, 2.0/880 * np.abs(one[:880/2]))
plt.show()


x = []
waves = os.listdir("data/train")
int_waves = [int(i.split(".")[0]) for i in waves]
int_waves.sort()
for name in int_waves:
    wav = wavfile.read("data/train/%d.wav" % name)[1]
    wav = wav[::downsample_factor] / max
    wav = wav[:len(one)]
    wav = np.resize(wav, (1, sample_rate, 1)).astype(np.float32)
    x.append(wav)

x = np.vstack(x)
indices = np.arange(len(x))
np.random.shuffle(indices)
y = x[indices[:100]]
x = x[indices[100:]]

model = Sequential()
model.add(Convolution1D(32, 3, border_mode='same', activation="tanh", input_shape=(sample_rate, 1)))
model.add(AveragePooling1D(pool_length=2, stride=None, border_mode="valid"))
model.add(Convolution1D(64, 3, border_mode='same', activation="tanh"))
model.add(AveragePooling1D(pool_length=2, stride=None, border_mode="valid"))
model.add(Convolution1D(128, 3, border_mode='same', activation="tanh"))
model.add(UpSampling1D(length=2))
model.add(Convolution1D(64, 3, border_mode='same', activation="tanh"))
model.add(UpSampling1D(length=2))
model.add(Convolution1D(32, 3, border_mode='same', activation="tanh"))
model.add(Convolution1D(1, 3, border_mode='same', activation="tanh"))

model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
if train:
    print("NOW FITTING")
    model.fit(x, x, nb_epoch=100, batch_size=64)
    model.save_weights("weights_fft.dat", True)

model.load_weights("weights_fft.dat")


predictions = model.predict_on_batch(x)
error = mean_squared_error(np.resize(x, (len(x), sample_rate)), np.resize(predictions, (len(predictions), sample_rate)))
print("Train Error: %.4f" % error)
for i in range(len(predictions)):
    prediction = np.resize(predictions[i], (sample_rate,))
    wavfile.write("train_predictions/prediction_%d.wav" % (indices[i+100]+1), sample_rate, prediction * max)

predictions = model.predict_on_batch(y)
error = mean_squared_error(np.resize(y, (len(y), sample_rate)), np.resize(predictions, (len(predictions), sample_rate)))
print("Test Error: %.4f" % error)

for i in range(len(predictions)):
    prediction = np.resize(predictions[i], (sample_rate,))
    wavfile.write("predictions/prediction_%d.wav" % (indices[i]+1), sample_rate, prediction * max)
# one = wavfile.read("data/train/381.wav")[1]
# plt.plot(np.linspace(1,100))
# plt.plot(y[0][:500] * max)
# plt.plot(one[:5000])
# plt.show()
