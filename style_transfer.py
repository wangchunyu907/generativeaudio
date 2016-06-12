import scipy
from keras.engine import Input
from keras.layers import Convolution1D, UpSampling1D, AveragePooling1D, MaxPooling1D
from keras.models import Model
from keras.optimizers import SGD
# from  keras import backend as K
from scipy.io import wavfile
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.metrics import mean_squared_error
import theano.tensor as T
import theano

print("starting")
activation = "tanh"
init = "glorot_uniform"
filter_size = 256

sample_rate = 1024
downsample_factor = 43
np.random.seed(41125)
content_factor = 1.0
style_factor = 1.0

# Load style sound and sample and normalize it.
rate, style = wavfile.read('data/sines.wav')
amplitude = -np.min(style)
style = style[::downsample_factor] / amplitude
style = style[:sample_rate]
# sample_rate = rate / downsample_factor

# Noise input.
noise = np.random.rand(len(style)).astype(np.float32)
# noise = style.astype(np.float32)
unnormal_noise = noise * amplitude
wavfile.write('noise.wav', sample_rate, unnormal_noise.astype(np.int16))
noise = np.reshape(noise, (1, sample_rate, 1))
# The sound array that we want to optimize.
x = np.random.rand(len(style)).astype(np.float32)
x = np.reshape(x, (1, sample_rate, 1))

style = style.astype(np.float32)
style = np.reshape(style, (1, sample_rate, 1))

print("building model")
# Create content network.
inputs = Input(shape=(sample_rate, 1))
layers = Convolution1D(48, filter_size, border_mode='same', activation=activation, init=init)(inputs)
layers = Convolution1D(48, filter_size, border_mode='same', activation=activation, init=init)(layers)
layers = Convolution1D(80, 2, border_mode='valid', activation=activation, init=init, subsample_length=2)(layers)
layers = Convolution1D(80, filter_size, border_mode='same', activation=activation, init=init)(layers)
layers = Convolution1D(80, filter_size, border_mode='same', activation=activation, init=init)(layers)
layers = Convolution1D(112, 2, border_mode='valid', activation=activation, init=init, subsample_length=2)(layers)
layers = Convolution1D(112, filter_size, border_mode='same', activation=activation, init=init)(layers)
layers = Convolution1D(112, filter_size, border_mode='same', activation=activation, init=init)(layers)
layers = Convolution1D(176, 2, border_mode='valid', activation=activation, init=init, subsample_length=2)(layers)
layers = Convolution1D(176, filter_size, border_mode='same', activation=activation, init=init)(layers)
layers = Convolution1D(176, filter_size, border_mode='same', activation=activation, init=init)(layers)

model = Model(input=inputs, output=layers)
model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))

X = Input(shape=(sample_rate, 1))
# predict = theano.function([model.layers[0].input], [model.layers[-1].output], allow_input_downcast=True)
# xc = predict(noise)
# xs = predict(style)
xc = model.predict(noise)
xs = model.predict(style)
xg = model(X)
loss = content_factor * T.mean((xg[0] - xc[0]) ** 2) + style_factor * T.mean(
    (T.dot(xg[0], xg[0].T) - T.dot(xs[0], xs[0].T)) ** 2)
gradient_function = theano.function([X], T.flatten(T.grad(loss, X)), allow_input_downcast=True)
loss_function = theano.function([X], loss, allow_input_downcast=True)


def evaluate(Xn):
    # Predict content
    # original = model.predict(noise)
    # prediction = model.predict(current_x)
    # content_loss = mean_squared_error(original[0], prediction[0])
    #
    # # TODO Predict style
    # # I think in the paper they use their 'semantic' layers for patches.
    # # Looks like
    # # Something nearest neighbour to select best patch for mean squared error
    #
    # # It seems that they use a convolution layer that looks like this:
    # # - model.add(Convolution1D(1, 8, border_mode='same', activation="tanh"))
    # # to get the nearest neighbour for patches. They use a convolution layer with
    # # one filter to get the most activated patch or something.
    #
    # style_loss = mean_squared_error(np.dot(original[0], original[0].T), np.dot(prediction[0], prediction[0].T))
    #
    # total_loss = content_loss + style_loss

    current_x = np.reshape(Xn, (1, sample_rate, 1)).astype(np.float32)
    gradients = gradient_function(current_x)
    total_loss = loss_function(current_x)
    return total_loss.astype(np.float64), gradients.astype(np.float64)


bounds = [[-1.0, 1.0]]
bounds = np.repeat(bounds, sample_rate, axis=0)

# print(scipy.optimize.check_grad(loss_function, gradient_function, x))
print("optimizing")
# result = scipy.optimize.minimize(evaluate, x.flatten(), method="BFGS", jac=True,
#                                  options={"disp": True, "gtol": 1e-5})
# print(result)
# y = result.x
y, Vn, info = scipy.optimize.fmin_l_bfgs_b(
    evaluate,
    x.astype(np.float64).flatten(),
    bounds=bounds,
    factr=0.0, pgtol=0.0,  # Disable automatic termination, set low threshold.
    m=5,  # Maximum correlations kept in memory by algorithm.
    maxfun=100,  # Limit number of calls to evaluate().
    iprint=1)
print(y)
y *= amplitude
wavfile.write('output.wav', sample_rate, y.astype(np.int16))
print("done.")
