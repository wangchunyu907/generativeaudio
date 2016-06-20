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

total_samples = sample_rate * 20
rate, noise = wavfile.read('data/mariomain.wav')
noise = noise[rate*5 : rate*10]
print('loaded content')
noise = resample(noise, total_samples)
db = 20 * np.log10(noise)
amplitude = np.max(np.abs(noise))
wavfile.write('output/content.wav', sample_rate, noise.astype(np.int16))
noise /= amplitude
noise = noise.astype(np.float32)
noise = np.reshape(noise, (1, total_samples, 1))

# noise = np.random.uniform(-1, 1, total_samples)
# noise /= 256.0
# # noise = style[:total_samples]
# # amplitude = np.max(np.abs(noise))
# unnormal_noise = noise * amplitude
# wavfile.write('output/content.wav', sample_rate, unnormal_noise.astype(np.int16))


# Load style sound and sample and normalize it.

styles = []
for i in range(1, 17):
    rate, style = wavfile.read('data/generated_style/%d.wav' % i)
    total_samples2 = int(len(style) / rate) * sample_rate
    style = resample(style, total_samples2)
    wavfile.write('output/input%d.wav' % i, sample_rate, style.astype(np.int16))
    amplitude = np.max(np.abs(style))
    style /= amplitude
    style = style.astype(np.float32)
    style_samples = len(style)
    style = np.reshape(style, (1, style_samples, 1))
    styles.append(style)

models = []
print("building model")

filters = np.array(
    # [sample_rate, sample_rate / 2, sample_rate / 3, sample_rate / 4, sample_rate / 6, sample_rate / 8, sample_rate / 16,
    [sample_rate / 8]).astype(np.int)
for f in filters:
    inputs = Input(shape=(None, 1))
    layers = Convolution1D(512, 256, border_mode='same', activation=activation, init=init)(inputs)
    decode = Convolution1D(1, 256, border_mode='same', activation="tanh", init=init)(layers)

    model = Model(input=inputs, output=decode)
    encoder_model = Model(input=inputs, output=layers)
    model.load_weights("weights_pre.dat")
    models.append(encoder_model)

X = Input(shape=(total_samples, 1))

xc = []
xs = []
xg = []
minimum = 10e7
loss = 0
for model in models:
    current_xc = model.predict(noise)
    current_xg = model(X)
    # xc.append(current_xc[0])
    # xs.append(current_xs[0])
    # xg.append(current_xg[0])
    # # * 10e7
    # xc = T.concatenate(xc, axis=1)
    # xs = T.concatenate(xs, axis=1)
    # xg = T.concatenate(xg, axis=1)

    xc = current_xc[0]
    xg = current_xg[0]
    xs_gram = 1
    for style in styles:
        current_xs = model.predict(style)
        xs = current_xs[0]
        xs_gram *= 1.0 * np.dot(xs.T, xs) / style.shape[1]
    xs_gram **= 1.0 / len(styles)
    xg_gram = 1.0 * T.dot(xg.T, xg) / total_samples

    # loss += content_factor * T.mean(T.square(xg - xc))
    loss += style_factor * T.sum(T.square(xs_gram - xg_gram)) / T.sum(
        T.square(xs_gram))

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
