import scipy
from keras.engine import Input
from keras.layers import Convolution1D, ELU, LeakyReLU
from keras.models import Model
from scipy.io import wavfile
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import theano.tensor as T
import theano
from scipy.signal import resample
import matplotlib.pyplot as plt

print("starting")
activation = "tanh"
init = "glorot_uniform"

sample_rate = 4096
filter_size = sample_rate / 5
downsample_factor = 4
np.random.seed(41126)
content_factor = 0.1
style_factor = 1.0
smoothness_factor = 1e-5

total_samples = sample_rate * 20
rate, noise = wavfile.read('data/queen.wav')
# noise = noise[rate * 0: rate * 35]
print('loaded content')
noise = resample(noise, total_samples)
db = 20 * np.log10(noise)
amplitude = np.max(np.abs(noise))
wavfile.write('output/content.wav', sample_rate, noise.astype(np.int16))
noise /= amplitude
noise *= 0.9
noise = noise.astype(np.float32)
noise = np.reshape(noise, (1, total_samples, 1))


x = np.random.uniform(-1, 1, total_samples)
x /= 256.0
# x = noise
# noise = style[:total_samples]
# amplitude = np.max(np.abs(noise))
unnormal_noise = x.flatten() * amplitude
wavfile.write('output/x.wav', sample_rate, unnormal_noise.astype(np.int16))

# Load style sound and sample and normalize it.

styles = []
for i in range(1, 2):
    rate, style = wavfile.read('data/mario.wav')
    # style = style[rate * 0: rate * 35]
    total_samples2 = int(len(style) / rate) * sample_rate
    style = resample(style, total_samples2)
    wavfile.write('output/input%d.wav' % i, sample_rate, style.astype(np.int16))
    amplitude = np.max(np.abs(style))
    style /= amplitude
    style *= 0.9
    style = style.astype(np.float32)
    style_samples = len(style)
    style = np.reshape(style, (1, style_samples, 1))
    styles.append(style)

models = []
content_model = None
print("building model")
filters = np.array(
    # [sample_rate, sample_rate / 2, sample_rate / 3, sample_rate / 4, sample_rate / 6, sample_rate / 8, sample_rate / 16,
    [sample_rate]).astype(np.int)
for f in filters:
    inputs = Input(shape=(None, 1))
    # layers1 = Convolution1D(64, 1024, border_mode='same', activation=activation, init=init)(inputs)
    # # layers1 = ELU()(layers1)
    # layers1 = Convolution1D(128, 512, border_mode='same', activation=activation, init=init, subsample_length=2)(layers1)
    # # layers1 = ELU()(layers1)
    # layers2 = Convolution1D(128, 512, border_mode='same', activation=activation, init=init)(layers1)
    # # layers2 = ELU()(layers2)
    # layers2 = Convolution1D(256, 256, border_mode='same', activation=activation, init=init, subsample_length=2)(layers2)
    # # layers2 = ELU()(layers2)
    # layers3 = Convolution1D(256, 256, border_mode='same', activation=activation, init=init)(layers2)
    # # layers3 = ELU()(layers3)
    # layers3 = Convolution1D(512, 128, border_mode='same', activation=activation, init=init, subsample_length=2)(layers3)
    # layers3 = ELU()(layers3)
    # layers4 = Convolution1D(512, 128, border_mode='same', activation=activation, init=init)(layers3)
    # layers4 = LeakyReLU()(layers4)
    # layers4 = Convolution1D(512, 128, border_mode='same', activation=activation, init=init, subsample_length=2)(layers4)
    # layers4 = LeakyReLU()(layers4)

    layers1 = Convolution1D(16, f / 16, border_mode='same', activation=activation, init=init)(inputs)
    layers2 = Convolution1D(16, f / 16, border_mode='same', activation=activation, subsample_length=2, init=init)(
        layers1)
    layers3 = Convolution1D(2048, f / 8, border_mode='same', activation="relu", subsample_length=2, init=init)(
        layers2)
    # models.append(Model(input=inputs, output=layers2))
    content_model = Model(input=inputs, output=layers3)
    # models.append(Model(input=inputs, output=layers2))
    models.append(Model(input=inputs, output=layers3))

X = Input(shape=(total_samples, 1))

xc = []
xs = []
xg = []
minimum = 10e7
loss = 0
model_factors = [1.0]
for model, model_factor in zip(models, model_factors):
    # current_xc = model.predict(noise)
    current_xg = model(X)
    # xc.append(current_xc[0])
    # xs.append(current_xs[0])
    # xg.append(current_xg[0])
    # # * 10e7
    # xc = T.concatenate(xc, axis=1)
    # xs = T.concatenate(xs, axis=1)
    # xg = T.concatenate(xg, axis=1)

    # xc = current_xc[0]
    xg = current_xg[0]
    xs_gram = 0
    for style in styles:
        current_xs = model.predict(style)
        xs = current_xs[0]
        xs_gram += 1.0 * np.dot(xs.T, xs) / style.shape[1]
    xs_gram /= len(styles)
    xg_gram = 1.0 * T.dot(xg.T, xg) / total_samples

    loss += model_factor * style_factor * T.sum(T.square(xs_gram - xg_gram)) / T.sum(
        T.square(xs_gram))

loss /= len(models)
current_xc = content_model.predict(noise)
current_xg = content_model(X)
xc = current_xc[0]
xg = current_xg[0]
loss += content_factor * T.mean(T.square(X - noise) * np.exp(-20*(abs(noise))))
# loss += content_factor * T.mean(T.square(xg - xc))
# loss += content_factor * T.mean(T.square(xg - xc) * (1 - T.abs_(xc)))
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
        current_x = current_x[np.newaxis, :, np.newaxis]
        # xc = []
        # xs = []
        # xg = []
        # minimum = 10e7
        # style_loss = 0
        # model_factors = [1.0]
        # for model, model_factor in zip(models, model_factors):
        #     # current_xc = model.predict(noise)
        #     current_xg = model.predict(current_x)
        #     # xc.append(current_xc[0])
        #     # xs.append(current_xs[0])
        #     # xg.append(current_xg[0])
        #     # # * 10e7
        #     # xc = T.concatenate(xc, axis=1)
        #     # xs = T.concatenate(xs, axis=1)
        #     # xg = T.concatenate(xg, axis=1)
        #
        #     # xc = current_xc[0]
        #     xg = current_xg[0]
        #     xs_gram = 0
        #     for style in styles:
        #         current_xs = model.predict(style)
        #         xs = current_xs[0]
        #         xs_gram += 1.0 * np.dot(xs.T, xs) / style.shape[1]
        #     xs_gram /= len(styles)
        #     xg_gram = 1.0 * np.dot(xg.T, xg) / total_samples
        #
        #     style_loss += model_factor * style_factor * np.sum(np.square(xs_gram - xg_gram)) / np.sum(
        #         np.square(xs_gram))
        #
        # style_loss /= len(models)
        # current_xc = content_model.predict(noise)
        # current_xg = content_model.predict(current_x)
        # xc = current_xc[0]
        # xg = current_xg[0]
        # content_loss = content_factor * np.mean(np.square(xg - xc))
        # content_loss *= 10e7
        # style_loss *= 10e7
        # print("Content Loss: %.4f, Style Loss: %.4f" % (np.log(content_loss), np.log(style_loss)))
    iteration_count += 1




def evaluate(Xn):
    current_x = np.reshape(Xn, (1, total_samples, 1)).astype(np.float32)
    gradients = gradient_function(current_x)
    total_loss = loss_function(current_x)
    return total_loss.astype(np.float64), gradients.astype(np.float64)


bounds = [[-0.9, 0.9]]
bounds = np.repeat(bounds, total_samples, axis=0)

print("optimizing")

y, Vn, info = scipy.optimize.fmin_l_bfgs_b(
    evaluate,
    x.astype(np.float64).flatten(),
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
