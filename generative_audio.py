from keras.layers import Convolution1D
from keras.models import Sequential
from keras.optimizers import SGD
from scipy.io import wavfile
import numpy as np

sample_rate = 1024
activation = "tanh"
init = "glorot_uniform"
filter_size = 3

downsample_factor = 43
np.random.seed(41125)

one = wavfile.read("sines.wav")[1]
max_value = -np.min(one)
one = one[::downsample_factor] / max_value
one = one[:1024]
one = np.resize(one, (1, sample_rate, 1))

model = Sequential()
model.add(Convolution1D(48, filter_size, border_mode='same', activation=activation, init=init, input_shape=(sample_rate, 1)))
model.add(Convolution1D(48, filter_size, border_mode='same', activation=activation, init=init))
model.add(Convolution1D(80, 2, border_mode='valid', activation=activation, init=init, subsample_length=2))
model.add(Convolution1D(80, filter_size, border_mode='same', activation=activation, init=init))
model.add(Convolution1D(112, 2, border_mode='valid', activation=activation, init=init, subsample_length=2))
model.add(Convolution1D(112, filter_size, border_mode='same', activation=activation, init=init))
model.add(Convolution1D(112, filter_size, border_mode='same', activation=activation, init=init))
model.add(Convolution1D(176, 2, border_mode='valid', activation=activation, init=init, subsample_length=2))
model.add(Convolution1D(176, filter_size, border_mode='same', activation=activation, init=init))
model.add(Convolution1D(176, filter_size, border_mode='same', activation=activation, init=init))

model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))

predictions = model.predict(one)[0]
predictions = np.swapaxes(predictions, 0, 1)
for i in range(len(predictions)):
    prediction = np.resize(predictions[i], (sample_rate / 8,))
    unnormal = prediction * max_value
    unnormal = unnormal.astype(np.int16)
    wavfile.write("generative_predictions/prediction_%d.wav" % i, sample_rate, unnormal)
