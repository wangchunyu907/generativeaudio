from keras.layers import Convolution1D
from keras.models import Sequential
from keras.optimizers import SGD
from scipy.io import wavfile
import numpy as np
from sklearn.metrics import mean_squared_error
import os


train = True
downsample_factor = 40
np.random.seed(41125)

one = wavfile.read("data/train/1.wav")[1]
max = -np.min(one)
one = one[::downsample_factor] / max
sample_rate = len(one)

x = []
waves = os.listdir("data/train")
int_waves = [int(i.split(".")[0]) for i in waves]
int_waves.sort()
for name in int_waves:
    wav = wavfile.read("data/train/%d.wav" % name)[1]
    wav = wav[::downsample_factor] / max
    wav = np.resize(wav, (1, 1, sample_rate)).astype(np.float32)
    x.append(wav)

x = np.vstack(x)
indices = np.arange(len(x))
np.random.shuffle(indices)
y = x[indices[:100]]
x = x[indices[100:]]

model = Sequential()
model.add(Convolution1D(1024, 5, border_mode='same', activation="tanh", input_shape=(1, sample_rate)))
model.add(Convolution1D(512, 3, border_mode='same', activation="tanh"))
model.add(Convolution1D(512, 3, border_mode='same', activation="tanh"))
model.add(Convolution1D(1024, 3, border_mode='same', activation="tanh"))
model.add(Convolution1D(sample_rate, 5, border_mode='same', activation="tanh"))
model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
if train:
    print("NOW FITTING")
    model.fit(x, x, nb_epoch=100, batch_size=64)
    model.save_weights("weights.dat", True)

model.load_weights("weights.dat")


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
