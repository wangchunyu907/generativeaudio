from keras.layers import Convolution1D, UpSampling1D, AveragePooling1D, MaxPooling1D
from keras.models import Sequential
from keras.optimizers import SGD
from scipy.io import wavfile
import numpy as np
from sklearn.metrics import mean_squared_error
import os
import matplotlib.pyplot as plt

downsample_factor = 40
np.random.seed(41125)

# Load style sound and sample and normalize it.
rate, style = wavfile.read('data/sines.wav')
amplitude = np.max(style)
style = style[::downsample_factor] / amplitude
sample_rate = rate / downsample_factor

# Noise input.
noise = np.random.rand(len(style))

# The sound array that we want to optimize.
x = np.random.rand(len(style))

# Create content network.
# TODO Fix these layers. Add stride layers.
model = Sequential()
model.add(Convolution1D(32, 32, border_mode='same', activation="tanh", input_shape=(sample_rate, 1)))
model.add(Convolution1D(32, 32, border_mode='same', activation="tanh"))
model.add(Convolution1D(32, 16, border_mode='same', activation="tanh"))
model.add(Convolution1D(1, 8, border_mode='same', activation="tanh"))

model.compile()

def evaluate(Xn):
    # Predict content
    original = model.predict(noise)
    prediction = model.predict(Xn)
    error = mean_squared_error(original, prediction)

    # TODO Predict style
    # I think in the paper they use their 'semantic' layers for patches.
    # Looks like 
    # Something nearest neighbour to select best patch for mean squared error

    # It seems that they use a convolution layer that looks like this:
    # - model.add(Convolution1D(1, 8, border_mode='same', activation="tanh"))
    # to get the nearest neighbour for patches. They use a convolution layer with
    # one filter to get the most activated patch or something.

    # TODO the scipy bfgs optimization needs loss + gradient
    return error #, gradients


x, v, info = scipy.optimize.fmin_l_bfgs_b(
                evaluate,
                x.astype(np.float64).flatten(),
                maxfun = 10)

x *= amplitude
wavfile.write('output.wav', sample_rate, x.astype(np.int16))


