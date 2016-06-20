import scipy
from keras.engine import Input
from keras.layers import Convolution1D
from keras.models import Model
from scipy.io import wavfile
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import theano.tensor as T
import theano
from scipy.signal import resample
import matplotlib.pyplot as plt

print("starting")
activation = "relu"
init = "glorot_uniform"

sample_rate = 4096
filter_size = sample_rate / 5
downsample_factor = 4
np.random.seed(41126)
content_factor = 0.1
style_factor = 1.0
smoothness_factor = 1e-5
# total_samples = sample_rate * 5
total_samples = sample_rate * 20
# rate, noise = wavfile.read('drum/2/output1120.wav')
rate, noise = wavfile.read('data/mario.wav')
# noise = noise[44100*46:44100*66]
print('loaded content')
# total_samples = int(len(noise) / rate) * sample_rate
noise = resample(noise, total_samples)
# time = np.arange(len(noise))
# time /= sample_rate
db = 20 * np.log10(noise)
amplitude = np.max(np.abs(noise))
# noise = noise / amplitude * 15000
# plt.plot(noise)
# plt.show()
wavfile.write('output/content.wav', sample_rate, noise.astype(np.int16))
noise /= amplitude


# Load style sound and sample and normalize it.
rate, style = wavfile.read('data/drum.wav')
# style = style[]
print('loaded style')
total_samples2 = int(len(style) / rate) * sample_rate
style = resample(style, total_samples2)
wavfile.write('output/input.wav', sample_rate, style.astype(np.int16))
amplitude = np.max(np.abs(style))
# style = style[::downsample_factor] / amplitude
# style = style[:total_samples]
# sample_rate = rate / downsample_factor
# style = np.hstack((style, style))

# noise = np.random.uniform(-1, 1, total_samples)
# noise /= 256.0
# noise = noise + style[:total_samples]
# # noise = style[:total_samples]
# # amplitude = np.max(np.abs(noise))
# unnormal_noise = noise * amplitude
# wavfile.write('output/content.wav', sample_rate, unnormal_noise.astype(np.int16))

# noise = noise[:len(style)]
# total_samples = total_samples2
# style = style[:len(noise)]

style /= amplitude

# Noise input.
# noise = np.random.rand(total_samples).astype(np.float32)
# noise = noise * 2 - 1
## noise /= amplitude
## noise = style.astype(np.float32)
# unnormal_noise = noise * amplitude
# wavfile.write('noise.wav', sample_rate, unnormal_noise.astype(np.int16))
# noise = np.reshape(noise, (1, total_samples, 1))

style = style.astype(np.float32)
style_samples = len(style)
style = np.reshape(style, (1, style_samples, 1))
noise = noise.astype(np.float32)
noise = np.reshape(noise, (1, total_samples, 1))
models = []
print("building model")
# Create content network.
# layers = Convolution1D(48, filter_size, border_mode='same', activation=activation, init=init)(inputs)
# layers = Convolution1D(48, filter_size, border_mode='same', activation=activation, init=init)(layers)
# layers = Convolution1D(80, 2, border_mode='valid', activation=activation, init=init, subsample_length=2)(layers)
# layers = Convolution1D(80, filter_size, border_mode='same', activation=activation, init=init)(layers)
# layers = Convolution1D(80, filter_size, border_mode='same', activation=activation, init=init)(layers)
# layers = Convolution1D(112, 2, border_mode='valid', activation=activation, init=init, subsample_length=2)(layers)
# layers = Convolution1D(112, filter_size, border_mode='same', activation=activation, init=init)(layers)
# layers = Convolution1D(112, filter_size, border_mode='same', activation=activation, init=init)(layers)
# layers = Convolution1D(176, 2, border_mode='valid', activation=activation, init=init, subsample_length=2)(layers)
# layers = Convolution1D(176, filter_size, border_mode='same', activation=activation, init=init)(layers)
# layers = Convolution1D(176, filter_size, border_mode='same', activation=activation, init=init)(layers)
# layers = Convolution1D(16, filter_size, border_mode='same', activation=activation, init=init)(inputs)

# filters = np.array(
#     [sample_rate / 4])
# for f in filters:
#     inputs = Input(shape=(total_samples, 1))
#     layers = Convolution1D(16, f/8, border_mode='same', activation=activation, init=init)(inputs)
#     layers = Convolution1D(16, f/8, border_mode='same', activation=activation, init=init)(layers)
#     layers = Convolution1D(32, 2, border_mode='valid', activation=activation, init=init, subsample_length=2)(layers)
#     layers = Convolution1D(32, f/4, border_mode='same', activation=activation, init=init)(layers)
#     layers = Convolution1D(32, f/4, border_mode='same', activation=activation, init=init)(layers)
#     layers = Convolution1D(64, 2, border_mode='valid', activation=activation, init=init, subsample_length=2)(layers)
#     layers = Convolution1D(64, f/2, border_mode='same', activation=activation, init=init)(layers)
#     layers = Convolution1D(64, f/2, border_mode='same', activation=activation, init=init)(layers)
#     layers = Convolution1D(128, 2, border_mode='valid', activation=activation, init=init, subsample_length=2)(layers)
#     layers = Convolution1D(128, f, border_mode='same', activation=activation, init=init)(layers)
#     layers = Convolution1D(128, f, border_mode='same', activation=activation, init=init)(layers)
#
#     model = Model(input=inputs, output=layers)
#     model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
#     models.append(model)

filters = np.array(
    # [sample_rate, sample_rate / 2, sample_rate / 3, sample_rate / 4, sample_rate / 6, sample_rate / 8, sample_rate / 16,
    [sample_rate/4]).astype(np.int)
for f in filters:
    inputs = Input(shape=(None, 1))
    layers = Convolution1D(512, f, border_mode='same', activation=activation, init=init, bias=False)(inputs)
    # layers = ELU()(layers)
    # input = Input(shape=(None, 1))
    # encoded = Convolution1D(512, 256, border_mode='same', activation="relu")(input)
    # encoded = Convolution1D(32, 256, border_mode='same', subsample_length=2, activation="relu")(encoded)
    # encoded = Convolution1D(64, 128, border_mode='same', activation="relu")(encoded)
    #
    # layers = Convolution1D(64, 128, border_mode='same', activation="tanh")(encoded)
    # layers = UpSampling1D(length=2)(layers)
    # layers = Convolution1D(32, 256, border_mode='same', activation="tanh")(layers)
    # layers = Convolution1D(1, 256, border_mode='same', activation="tanh")(encoded)

    # model = Model(input=input, output=layers)
    # encoder_model = Model(input=input, output=encoded)
    # model.load_weights("weights_pre.dat")

    model = Model(input=inputs, output=layers)
    # model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
    # encoder_model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
    models.append(model)

X = Input(shape=(total_samples, 1))
# X = T.tensor3('input')
# predict = theano.function([model.layers[0].input], [model.layers[-1].output], allow_input_downcast=True)
# xc = predict(noise)
# xs = predict(style)
xc = []
xs = []
xg = []
minimum = 10e7
loss = 0
for model in models:
    current_xc = model.predict(noise)
    current_xs = model.predict(style)
    current_xg = model(X)
    # xc.append(current_xc[0])
    # xs.append(current_xs[0])
    # xg.append(current_xg[0])
    # # * 10e7
    # xc = T.concatenate(xc, axis=1)
    # xs = T.concatenate(xs, axis=1)
    # xg = T.concatenate(xg, axis=1)

    xc = current_xc[0]
    xs = current_xs[0]
    xg = current_xg[0]

    xs_gram = 1.0 * T.dot(xs.T, xs) / style_samples
    xg_gram = 1.0 * T.dot(xg.T, xg) / total_samples
    # loss = content_factor * T.mean((xg[0] - xc[0]) ** 2) + style_factor * T.mean(
    #     (T.dot(xg[0], T.transpose(xg[0])) - np.dot(xs[0], xs[0].T)) ** 2)
    # loss = style_factor * T.sum(T.square(xs_gram - xg_gram)) / T.sum(T.square(xs_gram))
    # loss = content_factor * T.mean(T.square(xg)) + style_factor * T.sum(T.square(xs_gram - xg_gram)) / T.sum(
    loss += content_factor * T.mean(T.square(xg - xc))
    loss += style_factor * T.sum(T.square(xs_gram - xg_gram)) / T.sum(
        T.square(xs_gram))
    minimum = T.minimum(minimum, (style_factor * T.sum(T.square(xs_gram - xg_gram)) / T.sum(
        T.square(xs_gram))))
# loss += smoothness_factor * T.sum((T.square(xg[:-1, :] - xg[1:, :])))
# loss += 9 * minimum
loss *= 10e7
gradient_function = theano.function([X], T.flatten(T.grad(loss, X)), allow_input_downcast=True)
loss_function = theano.function([X], loss, allow_input_downcast=True)
iteration_count = 0


def optimization_callback(xk):
    global iteration_count
    if iteration_count % 10 == 0:
        current_x = np.copy(xk)
        current_x *= amplitude
        wavfile.write('output/output%d.wav' % iteration_count, sample_rate, current_x.astype(np.int16))
    # print(xk)
    # print(loss_function(np.reshape(xk, (1, total_samples, 1))))
    # print(gradient_function(np.reshape(xk, (1, total_samples, 1))))
    iteration_count += 1


def evaluate(Xn):
    current_x = np.reshape(Xn, (1, total_samples, 1)).astype(np.float32)
    gradients = gradient_function(current_x)
    total_loss = loss_function(current_x)
    return total_loss.astype(np.float64), gradients.astype(np.float64)


bounds = [[-0.9, 0.9]]
bounds = np.repeat(bounds, total_samples, axis=0)

# print(scipy.optimize.check_grad(loss_function, gradient_function, x))
print("optimizing")
# result = scipy.optimize.minimize(evaluate, x.flatten(), method="BFGS", jac=True,
#                                  options={"disp": True, "gtol": 1e-5})
# print(result)
# y = result.x
y, Vn, info = scipy.optimize.fmin_l_bfgs_b(
    evaluate,
    noise.astype(np.float64).flatten(),
    bounds=bounds,
    factr=0.0, pgtol=0.0,
    maxfun=30000,  # Limit number of calls to evaluate().
    iprint=1,
    approx_grad=False,
    callback=optimization_callback)

print(y)

y *= amplitude
wavfile.write('output/output.wav', sample_rate, y.astype(np.int16))
print("done.")
