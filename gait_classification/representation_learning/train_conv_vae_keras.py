import numpy as np
import matplotlib.pyplot as plt

from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.models import Model
from tools.utils import load_styletransfer

rng = np.random.RandomState(23455)

latent_dim = 2
intermediate_dim = 10
batch_size = 1
epsilon_std = 0.01
nb_epoch = 10

datasets = load_styletransfer(rng)

# Shape = (MB, 240, 66)
x_train = datasets[0][0].swapaxes(1, 2)

input_motion = Input(batch_shape=(batch_size, x_train.shape[1], x_train.shape[2]))

# shape = (240, 64)
x = Convolution1D(64, 25, activation='relu', border_mode='same')(input_motion)
# shape = (120, 64)
x = MaxPooling1D(pool_length=2, stride=None)(x)
#x = Convolution1D(128, 25, border_mode='same')(x)
#x = MaxPooling1D(pool_length=2, stride=None)(x)
#x = Convolution1D(256, 25, border_mode='same')(x)
#x = MaxPooling1D(pool_length=2, stride=None)(x)

# shape = (120, 64)
x = Flatten()(x)

# shape = (128)
h = Dense(intermediate_dim, activation='relu')(x)

## Estimate mean and std dev of a Gaussian

# shape = (10)
z_mean = Dense(latent_dim)(h)
z_log_std = Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_std = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., std=epsilon_std)
    return z_mean + K.exp(z_log_std) * epsilon

# shape = (10)
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_std])
# shape = (128)
x = Dense(intermediate_dim, activation='relu')(z)

# shape = (120*64)
x = Dense(120*64)(x)
# shape = (120, 64)
x = Reshape((120, 64))(x)

#x = UpSampling1D(length=2)(x)
#x = Convolution1D(256, 25, border_mode='same')(x)
#x = UpSampling1D(length=2)(x)
<<<<<<< HEAD
decoded = Convolution1D(66, 25, activation='linear', border_mode='same')(x)

autoencoder = Model(input_motion	, decoded)
autoencoder.compile(optimizer='adadelta', loss='mse')

autoencoder.fit(x_train, x_train,
                nb_epoch=5,
                batch_size=100,
                validation_data=(x_valid, x_valid),
                shuffle=True)

#batch_size = 16
#original_dim = 784
#latent_dim = 2
#intermediate_dim = 128
#epsilon_std = 0.01
#nb_epoch = 5
#
#x = Input(batch_shape=(batch_size, original_dim))

#h = Dense(intermediate_dim, activation='relu')(x)
#z_mean = Dense(latent_dim)(h)
#z_log_std = Dense(latent_dim)(h)
#
#def sampling(args):
#    z_mean, z_log_std = args
#    epsilon = K.random_normal(shape=(batch_size, latent_dim),
#                              mean=0., std=epsilon_std)
#    return z_mean + K.exp(z_log_std) * epsilon
#
## note that "output_shape" isn't necessary with the TensorFlow backend
## so you could write `Lambda(sampling)([z_mean, z_log_std])`
#z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_std])
#
## we instantiate these layers separately so as to reuse them later
#decoder_h = Dense(intermediate_dim, activation='relu')
#decoder_mean = Dense(original_dim, activation='sigmoid')
#h_decoded = decoder_h(z)
#x_decoded_mean = decoder_mean(h_decoded)
#
#def vae_loss(x, x_decoded_mean):
#    xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
=======
#x = Convolution1D(128, 25, border_mode='same')(x)
#x = UpSampling1D(length=2)(x)
#x = Convolution1D(64, 25, activation='relu', border_mode='same')(x)

# shape = (240, 64)
x = UpSampling1D(length=2)(x)
# shape = (240, 66)
x = Convolution1D(66, 25, activation='linear', border_mode='same')(x)

def vae_loss(input_motion, x):
    xent_loss = objectives.mse(input_motion, x)
>>>>>>> db7b92dfeb18c72a87ec3ffbdbdf9a377bb88326
#    kl_loss = - 0.5 * K.mean(1 + z_log_std - K.square(z_mean) - K.exp(z_log_std), axis=-1)
#    return xent_loss + kl_loss
    return xent_loss

vae = Model(input_motion, x)
vae.compile(optimizer='rmsprop', loss=vae_loss)

vae.fit(x_train, x_train,
        shuffle=True,
        nb_epoch=nb_epoch,
        batch_size=batch_size)
