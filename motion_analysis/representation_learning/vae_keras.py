import numpy as np
import matplotlib.pyplot as plt 
import sys

from nn.AnimationPlotLines import animation_plot

from keras.layers import *
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.callbacks import EarlyStopping
from keras.datasets import mnist
from keras.optimizers import Nadam
from tools.utils import load_locomotion

rng = np.random.RandomState(23455)

datasets, std, mean = load_locomotion(rng)

x_train = datasets[0][0][:320]
x_train = x_train.swapaxes(1, 2)

print x_train.shape

I = np.arange(len(x_train))
rng.shuffle(I)
x_train = x_train[I]

batch_size = 10
original_dim = 66*240
latent_dim = 100
intermediate_dim = 500
epsilon_std = 0.1
nb_epoch = 50

input_motion = Input(batch_shape=(batch_size, x_train.shape[1], x_train.shape[2]))

encoder1 = Dropout(0.15)
# shape = (240, 64)
encoder2 = Convolution1D(64, 25, border_mode='same')
encoder3 = Activation('relu')
# shape = (120, 64)
encoder4 = MaxPooling1D(pool_length=2, stride=None)

encoder5 = Dropout(0.15)
# shape = (60, 256)
encoder_mean1 = Convolution1D(128, 25, border_mode='same')
encoder_mean2 = Activation('relu')
# shape = (30, 256)
encoder_mean3 = MaxPooling1D(pool_length=2, stride=None)

# shape = (60, 256)
encoder_std1 = Convolution1D(128, 25, border_mode='same')
encoder_std2 =  Activation('relu')
# shape = (30, 256)
encoder_std3 = MaxPooling1D(pool_length=2, stride=None)

def sampling(args):
    z_mean, z_log_std = args
    epsilon = K.random_normal(shape=(batch_size, 60, 128),
                              mean=0., std=epsilon_std)
    return z_mean + K.exp(z_log_std) * epsilon

decoder1 = UpSampling1D(length=2)
decoder2 = Dropout(0.15)
decoder3 = Convolution1D(128, 25, border_mode='same')
decoder4 = Activation('relu')

decoder5 = UpSampling1D(length=2)
decoder6 = Dropout(0.25)
decoder7 = Convolution1D(66, 25, border_mode='same')
decoder8 = Activation('relu')

##########################################################################

x = encoder1 (input_motion)
x = encoder2 (x)
x = encoder3 (x)
x = encoder4 (x)
x = encoder5 (x)

h_m = encoder_mean1 (x)
h_m = encoder_mean2 (h_m)
z_mean = encoder_mean3 (h_m)

h_s = encoder_std1 (x)
h_s = encoder_std2 (h_s)
z_log_std = encoder_std3 (h_s)

z = Lambda(sampling, output_shape=(60, 128))([z_mean, z_log_std])

x = decoder1 (z)
x = decoder2 (x)
x = decoder3 (x)
x = decoder4 (x)
x = decoder5 (x)
x = decoder6 (x)
x = decoder7 (x)
x = decoder8 (x)

def vae_loss(input_motion, x):
    mse_loss = objectives.mse(input_motion, x)
    kl_loss = - 0.5 * K.mean(1 + z_log_std - K.square(z_mean) - K.exp(z_log_std))
    return mse_loss + 0.5 * kl_loss

vae = Model(input_motion, x)

nadam = Nadam(lr=0.00005, beta_1=0.7, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

vae.compile(optimizer=nadam, loss=vae_loss)

vae.fit(x_train, x_train,
        shuffle=True,
        nb_epoch=nb_epoch,
        batch_size=batch_size)

dec_input = Input(shape=(30,256))
g = decoder1 (dec_input)
g = decoder2 (g)
g = decoder3 (g)
g = decoder4 (g)
g = decoder5 (g)
g = decoder6 (g)
g = decoder7 (g)
g = decoder8 (g)

generator = Model(dec_input, g)
generator.save_weights('locomotion_vae_keras_6.h5', overwrite=True)

generator_input = np.random.uniform(size=(x_train.shape[0], 60, 128))
x_predicted = generator.predict(generator_input, batch_size=batch_size)
x_predicted = x_predicted.swapaxes(1, 2)

result = x_predicted * (std + 1e-10) + mean

new1 = result[25:26]
new2 = result[26:27]
new3 = result[0:1]
