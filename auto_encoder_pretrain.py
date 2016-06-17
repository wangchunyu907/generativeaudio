from keras.engine import Input
from keras.engine import Model
from keras.layers import Convolution1D, UpSampling1D, AveragePooling1D, MaxPooling1D
from keras.models import Sequential
from keras.optimizers import SGD
from scipy.io import wavfile
import numpy as np
from scipy.signal import resample
from sklearn.metrics import mean_squared_error
import os
import matplotlib.pyplot as plt

train = True
sample_rate = 4096
np.random.seed(41126)
seconds = 5
total_samples = sample_rate * seconds
amplitude = 0

x = []
waves = os.listdir("data/pretrain")
int_waves = [int(i.split(".")[0]) for i in waves]
int_waves.sort()
for name in int_waves:
    rate, wav = wavfile.read("data/pretrain/%d.wav" % name)
    wav = wav[:seconds * rate]
    wav = resample(wav, total_samples)
    if amplitude == 0:
        amplitude = np.max(np.abs(wav))
    wav /= amplitude
    wav = np.resize(wav, (1, total_samples, 1)).astype(np.float32)
    x.append(wav)

print('loaded content')
x = np.vstack(x)
indices = np.arange(len(x))
np.random.shuffle(indices)
# y = x[indices[:100]]
# x = x[indices[100:]]

input = Input(shape=(total_samples, 1))
layers = Convolution1D(512, 256, border_mode='same', activation="relu")(input)
# layers = Convolution1D(32, 256, border_mode='same', subsample_length=2, activation="relu")(layers)
# layers = Convolution1D(512, 128, border_mode='same', activation="relu")(layers)
# # layers = Convolution1D(64, 128, border_mode='same', subsample_length=2, activation="relu")(layers)
# # model.add(Convolution1D(64, 128, border_mode='same', subsample_length=2, activation="tanh"))
# # model.add(Convolution1D(128, 64, border_mode='same', subsample_length=2, activation="tanh"))
# # # model.add(Convolution1D(32, 32, border_mode='same', subsample_length=2, activation="relu"))
# # # model.add(MaxPooling1D(pool_length=2, stride=None, border_mode="valid"))
# # #
# # # model.add(Convolution1D(4, 3, border_mode='same', activation="tanh"))
# # # model.add(UpSampling1D(length=2))
# # # model.add(Convolution1D(2, 3, border_mode='same', activation="tanh"))
# # # model.add(UpSampling1D(length=2))
# # # model.add(Convolution1D(32, 32, border_mode='same', activation="relu"))
# # model.add(UpSampling1D(length=2))
# # model.add(Convolution1D(128, 64, border_mode='same', activation="tanh"))
# # model.add(UpSampling1D(length=2))
# # model.add(Convolution1D(64, 128, border_mode='same', activation="tanh"))
# # layers = UpSampling1D(length=2)(layers)
# layers = Convolution1D(128, 128, border_mode='same', activation="tanh")(layers)
# layers = UpSampling1D(length=2)(layers)
# layers = Convolution1D(32, 256, border_mode='same', activation="tanh")(layers)
layers = Convolution1D(1, 256, border_mode='same', activation="tanh")(layers)

model = Model(input=input, output=layers)
model.load_weights("weights_pre.dat")
model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.001, momentum=0.9, nesterov=True))
if train:
    print("NOW FITTING")
    model.fit(x, x, nb_epoch=100, batch_size=5)
    model.save_weights("weights_pre.dat", True)

model.load_weights("weights_pre.dat")

predictions = model.predict_on_batch(x)
error = mean_squared_error(np.resize(x, (len(x), total_samples)),
                           np.resize(predictions, (len(predictions), total_samples)))
print("Train Error: %.4f" % error)
for i in range(len(predictions)):
    prediction = np.resize(predictions[i], (total_samples,))
    unnormal = prediction * amplitude
    unnormal = unnormal.astype(np.int16)
    wavfile.write("pretrain_predictions/prediction_%d.wav" % i, sample_rate, unnormal)
    # to_plot = wavfile.read("train_predictions/prediction_%d.wav" % (indices[i+100]+1))[1]
    # plt.plot(to_plot[:100])
    # plt.show()
    # print("hi")

# test_predictions = model.predict_on_batch(y)
# error = mean_squared_error(np.resize(y, (len(y), sample_rate)),
#                            np.resize(predictions, (len(test_predictions), sample_rate)))
# print("Test Error: %.4f" % error)
#
# for i in range(len(test_predictions)):
#     prediction = np.resize(test_predictions[i], (sample_rate,))
#     unnormal = prediction * max
#     unnormal = unnormal.astype(np.int16)
#     wavfile.write("predictions/prediction_%d.wav" % (indices[i] + 1), sample_rate, unnormal)

# one = wavfile.read("data/train/381.wav")[1][::43]
# plt.plot(test_predictions[0][:20] * max)
# plt.plot(one[:20])
# plt.show()
