from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

x_output = x_train[:,:,:7]
print x_output.shape

x_train = x_train.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 1, 28, 28))
x_output = np.reshape(x_output, (len(x_train), 1, 28, 7))

print x_train.shape
print x_output.shape

input_img = Input(shape=(1, 28, 28))

#x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_img)
#x = MaxPooling2D((2, 2), border_mode='same')(x)
#x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
#x = MaxPooling2D((2, 2), border_mode='same')(x)
#x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(input_img)
# Returns (8, 28, 28)
x = MaxPooling2D((1, 2))(input_img)
# Returns (8, 28, 14)
#x = Convolution2D(1, 3, 3, activation='relu', border_mode='same')(x)
# Returns (1, 28, 14)
x = MaxPooling2D((1, 2))(x)

encoded = x
# Returns (3, 28, 14)
#encoded = Convolution2D(1, 3, 3, activation='relu', border_mode='same')(input_img)
# Returns (1, 28, 14)

# at this point the representation is (8, 4, 4) i.e. 128-dimensional

#x = Convolution2D(8, 1, 2, activation='relu', border_mode='same')(encoded)
#x = UpSampling2D((3, 3))(x)
#x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
#x = UpSampling2D((2, 2))(x)
#x = Convolution2D(16, 3, 3, activation='relu')(x)
#x = UpSampling2D((2, 2))(x)
#decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)
decoded = encoded

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train, x_output,
                nb_epoch=50,
                batch_size=128,
                shuffle=True)
