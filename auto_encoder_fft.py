from keras.layers import Convolution1D, UpSampling1D, AveragePooling1D, MaxPooling1D, LeakyReLU
from keras.models import Sequential
from keras.optimizers import SGD
from scipy.io import wavfile
import numpy as np
from scipy.fftpack import fft, ifft
from sklearn.metrics import mean_squared_error
import os

train = True
downsample_factor = 43.0
sample_rate = 44100
normalizing_constant = sample_rate * 20000.0
# N = 880
np.random.seed(41125)

one = wavfile.read("data/train/1.wav")[1]
# max = -np.min(one)
# one = 1.0*one/max
one = np.imag(fft(one))
# one_new = np.abs(ifft(fourier))


one /= normalizing_constant
one = one[100:484]

test = wavfile.read("data/train/1.wav")[1]
fourier = np.imag(fft(test))
test_new = -np.imag(ifft(fourier))

# k = np.arange(N)
# T = 1.0 * N / 44100 * downsample_factor
# xf = np.linspace(0.0, N / T, N/2.0)
# fig, ax = plt.subplots()
# ax.plot(xf, 2.0 / 880 * np.abs(one[:880 / 2]))
# plt.show()
x = []
wav512 = None
waves = os.listdir("data/train")
int_waves = [int(i.split(".")[0]) for i in waves]
int_waves.sort()
for name in int_waves:
    wav = wavfile.read("data/train/%d.wav" % name)[1]
    wav = np.imag(fft(wav))
    wav /= normalizing_constant
    wav = wav[100:484]
    wav = np.resize(wav, (1, len(wav), 1)).astype(np.float32)
    if name == 512:
        wav512 = wav
    x.append(wav)

x = np.vstack(x)
indices = np.arange(len(x))
np.random.shuffle(indices)
y = x[indices[:100]]
x = x[indices[100:]]

model = Sequential()
model.add(Convolution1D(32, 3, border_mode='same', activation="tanh", input_shape=(len(one), 1)))
# model.add(MaxPooling1D(pool_length=2, stride=None, border_mode="valid"))
# model.add(Convolution1D(32, 3, border_mode='same', activation="tanh"))
# model.add(Convolution1D(32, 3, border_mode='same', activation="tanh"))
# model.add(Convolution1D(32, 3, border_mode='same', activation="tanh"))
# model.add(Convolution1D(32, 3, border_mode='same', activation="tanh"))
# model.add(Convolution1D(32, 3, border_mode='same', activation="tanh"))
# model.add(Convolution1D(32, 3, border_mode='same', activation="tanh"))
# # model.add(MaxPooling1D(pool_length=2, stride=None, border_mode="valid"))
# # model.add(Convolution1D(1, 3, border_mode='same', activation="relu"))
# #
# model.add(UpSampling1D(length=2))
# model.add(Convolution1D(32, 10, border_mode='same', activation="tanh",))
# model.add(UpSampling1D(length=2))
# model.add(Convolution1D(1, 3, border_mode='same', activation="relu"))
# model.add(Convolution1D(1, 3, border_mode='same', activation="relu"))
# model.add(Convolution1D(1, 3, border_mode='same', activation="relu"))
# model.add(Convolution1D(1, 3, border_mode='same', activation="relu"))
model.add(Convolution1D(1, 3, border_mode='same', activation="tanh",))

model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
if train:
    print("NOW FITTING")
    model.fit(x, x, nb_epoch=1000, batch_size=64)
    model.save_weights("weights_fft.dat", True)

model.load_weights("weights_fft.dat")

predictions = model.predict_on_batch(x)
error = mean_squared_error(np.resize(x, (len(x), sample_rate)), np.resize(predictions, (len(predictions), sample_rate)))
print("Train Error: %.4f" % error)
for i in range(len(predictions)):
    prediction = np.resize(predictions[i], (384,))
    prediction = np.concatenate([np.zeros(100), prediction, np.zeros(44100 - 484)])
    # prediction = np.concatenate([prediction, prediction[::-1]])
    prediction *= normalizing_constant
    wav = -np.imag(ifft(prediction)) * 2.0
    wav = wav.astype(np.int16)
    wavfile.write("train_predictions/prediction_%d.wav" % (indices[i + 100] + 1), sample_rate, wav)

predictions = model.predict_on_batch(y)
error = mean_squared_error(np.resize(y, (len(y), sample_rate)), np.resize(predictions, (len(predictions), sample_rate)))
print("Test Error: %.4f" % error)

for i in range(len(predictions)):
    prediction = np.resize(predictions[i], (384,))
    prediction = np.concatenate([np.zeros(100), prediction, np.zeros(44100 - 484)])
    # prediction = np.concatenate([prediction, prediction[::-1]])
    prediction *= normalizing_constant
    wav = -np.imag(ifft(prediction)) * 2.0
    wav = wav.astype(np.int16)
    wavfile.write("predictions/prediction_%d.wav" % (indices[i] + 1), sample_rate, wav)
# one = wavfile.read("data/train/381.wav")[1]
# plt.plot(np.linspace(1,100))
# plt.plot(y[0][:500] * max)
# plt.plot(one[:5000])
# plt.show()
