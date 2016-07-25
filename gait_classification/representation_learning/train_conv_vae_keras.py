import numpy as np
import matplotlib.pyplot as plt

np.random.seed(23455)

from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.models import Model
from keras.optimizers import Nadam
from tools.utils import load_hdm05_easy, load_cmu

rng = np.random.RandomState(23455)

datasets = load_cmu(rng)

# Shape = (MB, 240, 66)
x_train = datasets[0][0][:17000]
x_train = x_train.swapaxes(1, 2)

batch_size = 1
latent_dim = 100
intermediate_dim = 500
epsilon_std = 0.01
nb_epoch = 100


input_motion = Input(batch_shape=(batch_size, x_train.shape[1], x_train.shape[2]))

# shape = (240, 64)
x = Convolution1D(64, 25, activation='relu', border_mode='same')(input_motion)
# shape = (120, 64)
x = MaxPooling1D(pool_length=2, stride=None)(x)
x = Convolution1D(128, 25, border_mode='same')(x)
x = MaxPooling1D(pool_length=2, stride=None)(x)
x = Convolution1D(256, 25, border_mode='same')(x)
x = MaxPooling1D(pool_length=2, stride=None)(x)

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

# shape = (50)
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_std])
# shape = (128)
x = Dense(intermediate_dim, activation='relu')(z)

# shape = (120*64)
#x = Dense(120*64)(x)
#x = Reshape((120, 64))(x)
# shape = (120, 64)
x = Dense(30*256)(x)
x = Reshape((30, 256))(x)

x = UpSampling1D(length=2)(x)
x = Convolution1D(256, 25, border_mode='same')(x)
x = UpSampling1D(length=2)(x)
x = Convolution1D(128, 25, border_mode='same')(x)

# shape = (240, 64)
x = UpSampling1D(length=2)(x)
# shape = (240, 66)
x = Convolution1D(66, 25, activation='linear', border_mode='same')(x)

def vae_loss(input_motion, x):
    mse_loss = objectives.mse(input_motion, x)
#    kl_loss = - 0.5 * K.mean(1 + z_log_std - K.square(z_mean) - K.exp(z_log_std), axis=-1)
    kl_loss = - 0.5 * K.mean(1 + z_log_std - K.square(z_mean) - K.exp(z_log_std))
    return mse_loss + kl_loss

vae = Model(input_motion, x)

nadam = Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

vae.compile(optimizer=nadam, loss=vae_loss)

vae.fit(x_train, x_train,
        shuffle=True,
        nb_epoch=nb_epoch,
        batch_size=batch_size)

vae.save_weights('vae.h5')

hdm05_datasets = load_hdm05_easy(rng)

# Shape = (MB, 240, 66)
x_train = hdm05_datasets[0][0][:1900]
x_train = x_train.swapaxes(1, 2)

x_valid = hdm05_datasets[1][0][:600]
x_valid = x_valid.swapaxes(1, 2)

y_train = hdm05_datasets[0][1][:1900]
y_valid = hdm05_datasets[1][1][:600]

# build a model to project inputs on the latent space
encoder = Model(input_motion, z_mean)

# (900, 50)
x_ft_train = encoder.predict(x_train, batch_size=batch_size)
x_ft_valid = encoder.predict(x_valid, batch_size=batch_size)

x_ft_input = Input(shape=(latent_dim,))
x_ft = Dense(50, activation='relu')(x_ft_input)
x_ft = Dense(25, activation='softmax')(x_ft)

model_ft = Model(x_ft_input, x_ft)
model_ft.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                 metrics=['accuracy'])

model_ft.fit(x_ft_train, y_train,
             shuffle=True,
             nb_epoch=nb_epoch,
             batch_size=batch_size,
             validation_data=(x_ft_valid, y_valid))
