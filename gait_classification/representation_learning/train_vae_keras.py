import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Lambda, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.models import Model
from utils import load_styletransfer

rng = np.random.RandomState(23455)

datasets = load_styletransfer(rng, split=(0.8, 0.2))

x_train = datasets[0][0]
x_train = x_train.reshape([x_train.shape[0], 1] + list(x_train.shape[1:]))
x_output = x_train[:,:,:,:]

# Shape = (MB, 66, 240)
input_motion = Input(shape=x_train.shape[1:])

x = Convolution2D(32, 66, 25, activation='relu', border_mode='same')(input_motion)
# (64, 66, 240)
x = MaxPooling2D((1, 2))(x)
# (64, 66, 120)
#x = Convolution2D(64, 66, 25, activation='relu', border_mode='same')(x)
# (128, 66, 120)
#x = MaxPooling2D((1, 2))(x)
# (128, 66, 60)
#x = Convolution2D(128, 66, 25, activation='relu', border_mode='same')(x)
# (256, 66, 60)
#x = MaxPooling2D((1, 2))(x)
# (256, 66, 30)

#x = Convolution2D(128, 66, 25, activation='relu', border_mode='same')(x)
# (256, 66, 60)
#x = UpSampling2D((1, 2))(x)
# (256, 66, 30)
#x = Convolution2D(64, 66, 25, activation='relu', border_mode='same')(x)
# (128, 66, 60)
#x = UpSampling2D((1, 2))(x)
# (128, 66, 120)
x = Convolution2D(32, 66, 25, activation='relu', border_mode='same')(x)
# (64, 66, 120)
x = UpSampling2D((1, 2))(x)
# (64, 66, 240)
x = Convolution2D(1, 66, 25, activation='relu', border_mode='same')(x)
# (1, 66, 240)

#x = UpSampling2D((1, 2))(x)
# (64, 66, 120)
#x = Convolution2D(128, 66, 25, activation='relu', border_mode='same')(x)
# (128, 66, 120)

#x = Convolution2D(1, 66, 25, activation='relu', border_mode='same')(x)
# (64, 66, 120)






#x = Convolution2D(1, 66, 25, activation='relu', border_mode='same')(input_motion)
# (64, 66, 240)

#x = Convolution2D(128, 66, 25, activation='relu', border_mode='same')(x)
# (128, 66, 240)
# (128, 66, 60)

#x = Convolution2D(256, 66, 25, activation='relu', border_mode='same')(x)
#encoded = MaxPooling2D((66, 2), border_mode='same')(x)

# at this point the representation is (256, 66, 60) i.e. 128-dimensional

#x = Convolution2D(256, 66, 25, activation='relu', border_mode='same')(encoded)
#x = UpSampling2D((66, 2))(x)

#x = Convolution2D(128, 66, 25, activation='relu', border_mode='same')(x)
#x = UpSampling2D((1, 2))(x)

#x = Convolution2D(64, 66, 25, activation='relu')(x)
#x = UpSampling2D((1, 2))(x)
#decoded = Convolution2D(1, 66, 25, activation='linear', border_mode='same')(x)

#x = Convolution1D(64, 25, activation='relu', border_mode='same')(input_motion)
#x = MaxPooling1D(pool_length=2, stride=None, border_mode='same')(input_motion)
#x = Convolution1D(128, 25, border_mode='same')(x)
#x = MaxPooling1D(pool_length=2, stride=None, border_mode='same')(x)
#encoded = Convolution1D(256, 25, border_mode='same')(x)
#encoded = MaxPooling1D(pool_length=2, stride=None, border_mode='same')(x)

# at this point the representation is (256, 64)

#x = Convolution1D(256, 25, border_mode='same')(encoded)
#x = UpSampling1D(length=2)(x)
#x = Convolution1D(128, 25, border_mode='same')(x)
#x = UpSampling1D(length=2)(x)
#x = Convolution1D(64, 25, activation='relu', border_mode='same')(x)
#x = UpSampling1D(pool_length=2)(x)
#decoded = Convolution1D(66, 25, activation='linear', border_mode='same')(x)

model = Model(input_motion, x)
model.compile(optimizer='adadelta', loss='mse')

model.fit(x_train, x_output,
          nb_epoch=50,
          batch_size=100,
#                validation_data=(x_valid, x_valid),
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
#    kl_loss = - 0.5 * K.mean(1 + z_log_std - K.square(z_mean) - K.exp(z_log_std), axis=-1)
#    return xent_loss + kl_loss
#
#vae = Model(x, x_decoded_mean)
#vae.compile(optimizer='rmsprop', loss=vae_loss)
#
## train the VAE on MNIST digits
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#
#x_train = x_train.astype('float32') / 255.
#x_test = x_test.astype('float32') / 255.
#x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
#x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
#
#vae.fit(x_train, x_train,
#        shuffle=True,
#        nb_epoch=nb_epoch,
#        batch_size=batch_size,
#        validation_data=(x_test, x_test))
#
## build a model to project inputs on the latent space
#encoder = Model(x, z_mean)
#
## display a 2D plot of the digit classes in the latent space
#x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
#plt.figure(figsize=(6, 6))
#plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
#plt.colorbar()
#plt.show()
#
## build a digit generator that can sample from the learned distribution
#decoder_input = Input(shape=(latent_dim,))
#_h_decoded = decoder_h(decoder_input)
#_x_decoded_mean = decoder_mean(_h_decoded)
#generator = Model(decoder_input, _x_decoded_mean)
#
## display a 2D manifold of the digits
#n = 15  # figure with 15x15 digits
#digit_size = 28
#figure = np.zeros((digit_size * n, digit_size * n))
## we will sample n points within [-15, 15] standard deviations
#grid_x = np.linspace(-15, 15, n)
#grid_y = np.linspace(-15, 15, n)
#
#for i, yi in enumerate(grid_x):
#    for j, xi in enumerate(grid_y):
#        z_sample = np.array([[xi, yi]]) * epsilon_std
#        x_decoded = generator.predict(z_sample)
#        digit = x_decoded[0].reshape(digit_size, digit_size)
#        figure[i * digit_size: (i + 1) * digit_size,
#               j * digit_size: (j + 1) * digit_size] = digit
#
#plt.figure(figsize=(10, 10))
#plt.imshow(figure)
#plt.show()


