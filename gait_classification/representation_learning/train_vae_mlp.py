'''This script demonstrates how to build a variational autoencoder with Keras.
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np
import matplotlib.pyplot as plt 

np.random.seed(23455)

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras.optimizers import Nadam
from tools.utils import load_cmu

rng = np.random.RandomState(23455)

datasets = load_cmu(rng)

x_train = datasets[0][0][:17000]
x_train = x_train.reshape(x_train.shape[0], np.prod(x_train.shape[1:]))

#x_valid = datasets[1][0][:600]
#x_valid = x_valid.reshape(x_valid.shape[0], np.prod(x_valid.shape[1:]))
#
#y_train = datasets[0][1][:1900]
#y_valid = datasets[1][1][:600]

batch_size = 10
original_dim = x_train.shape[1]
latent_dim = 50
intermediate_dim = 128
epsilon_std = 0.01
nb_epoch = 100

x = Input(batch_shape=(batch_size, original_dim))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_std = Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_std = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., std=epsilon_std)
    return z_mean + K.exp(z_log_std) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
# so you could write `Lambda(sampling)([z_mean, z_log_std])`
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_std])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='linear')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

def vae_loss(x, x_decoded_mean):
    mse_loss = objectives.mse(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_std - K.square(z_mean) - K.exp(z_log_std), axis=-1)
    return mse_loss + kl_loss
#    return mse_loss

vae = Model(x, x_decoded_mean)

nadam = Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

vae.compile(optimizer=nadam, loss=vae_loss)

vae.fit(x_train, x_train,
        shuffle=True,
        nb_epoch=nb_epoch,
        batch_size=batch_size)#,
#        validation_data=(x_valid, x_valid))

vae.save_weights('cmu_mlp_vae.h5')

# build a model to project inputs on the latent space
#encoder = Model(x, z_mean)
#
## (900, 50)
#x_ft_train = encoder.predict(x_train, batch_size=batch_size)
#x_ft_valid = encoder.predict(x_valid, batch_size=batch_size)
#
##print x_ft_train.shape
##print x_ft_valid.shape
#
#x_ft_input = Input(shape=(latent_dim,))
#x_ft = Dense(30, activation='relu')(x_ft_input)
#x_ft = Dense(25, activation='softmax')(x_ft)
#
#model_ft = Model(x_ft_input, x_ft)
#model_ft.compile(optimizer='rmsprop', loss='categorical_crossentropy',
#                 metrics=['accuracy'])
#
#model_ft.fit(x_ft_train, y_train,
#             shuffle=True,
#             nb_epoch=nb_epoch,
#             batch_size=batch_size,
#             validation_data=(x_ft_valid, y_valid))

## display a 2D plot of the digit classes in the latent space
##plt.figure(figsize=(6, 6))
##plt.scatter(x_train_encoded[:, 0], x_train_encoded[:, 1], c=y_train)
##plt.colorbar()
##plt.show()
#
#### build a digit generator that can sample from the learned distribution
###decoder_input = Input(shape=(latent_dim,))
###_h_decoded = decoder_h(decoder_input)
###_x_decoded_mean = decoder_mean(_h_decoded)
###generator = Model(decoder_input, _x_decoded_mean)
###
#### display a 2D manifold of the digits
###n = 15  # figure with 15x15 digits
###digit_size = 28
###figure = np.zeros((digit_size * n, digit_size * n))
#### we will sample n points within [-15, 15] standard deviations
###grid_x = np.linspace(-15, 15, n)
###grid_y = np.linspace(-15, 15, n)
###
###for i, yi in enumerate(grid_x):
###    for j, xi in enumerate(grid_y):
###        z_sample = np.array([[xi, yi]]) * epsilon_std
###        x_decoded = generator.predict(z_sample)
###        digit = x_decoded[0].reshape(digit_size, digit_size)
###        figure[i * digit_size: (i + 1) * digit_size,
###               j * digit_size: (j + 1) * digit_size] = digit
###
###plt.figure(figsize=(10, 10))
###plt.imshow(figure)
###plt.show()
