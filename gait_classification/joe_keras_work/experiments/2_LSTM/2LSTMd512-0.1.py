from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, TimeDistributed
from keras.layers import LSTM
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Nadam
import keras
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import theano
sys.path.append('../../utils/')
from visualise_before import visualise
data = np.load('../../../data/Joe/sequential_final_frame.npz')
train_x = data['train_x']
train_y = data['train_y']
test_x = data['test_x']
test_y = data['test_y']
# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(TimeDistributed(Dense(256), input_shape=(29, 256)))
model.add(Activation(keras.layers.advanced_activations.ELU(alpha=1.0)))
model.add(LSTM(256, return_sequences=True, consume_less='gpu', \
               init='glorot_normal'))
model.add(LSTM(512, return_sequences=True, consume_less='gpu', \
               init='glorot_normal'))
model.add(Dropout(0.1))
model.add(TimeDistributed(Dense(256)))
model.add(Activation(keras.layers.advanced_activations.ELU(alpha=1.0)))
# TimedistributedDense on top - Can then set output vectors to be next sequence!
model.compile(loss='mean_squared_error', optimizer='nadam')
print('Training model...')
model.fit(train_x, train_y, batch_size=10, nb_epoch=100, validation_data=(test_x,test_y))
model.save_weights('../../weights/2LSTM-256-512d-0.1.hd5', overwrite=True)

