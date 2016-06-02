from keras.layers import Convolution1D
from keras.models import Sequential
from keras.optimizers import SGD
from scipy.io import wavfile
import numpy as np
from scipy.signal import decimate

one = wavfile.read("data/train/1.wav")[1]
two = wavfile.read("data/train/2.wav")[1]
three = wavfile.read("data/train/3.wav")[1]

one = one[::100]
two = two[::100]
three = three[::100]

sample_rate = len(one)
wavfile.write("one.wav", sample_rate, one)


one = np.resize(one, (1, 1, sample_rate))
two = np.resize(two, (1, 1, sample_rate))
three = np.resize(three, (1, 1, sample_rate))

model = Sequential()
model.add(Convolution1D(256, 3, border_mode='same', activation="relu", input_shape=(1, sample_rate)))
model.add(Convolution1D(sample_rate, 3, border_mode='same', activation="relu"))
model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.001, momentum=0.9, nesterov=True))
print("NOW FITTING")
model.fit(one, two, nb_epoch=10000, batch_size=1, verbose=True)
prediction = model.predict(one)

prediction = np.resize(prediction, (sample_rate,))

wavfile.write("prediction.wav", sample_rate, prediction)
