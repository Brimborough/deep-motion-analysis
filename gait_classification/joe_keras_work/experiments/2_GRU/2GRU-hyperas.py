from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, TimeDistributed
from keras.layers import LSTM, GRU
from keras.optimizers import Nadam
import keras
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import theano

'''
Data providing function:

This function is separated from model() so that hyperopt
won't reload data for each evaluation run.
'''
data = np.load('../../../data/Joe/sequential_final_frame.npz')
train_x = data['train_x']
train_y = data['train_y']
test_x = data['test_x']
test_y = data['test_y']

model = Sequential()
model.add(TimeDistributed(Dense(256), input_shape=(29,256)))
model.add(Activation(keras.layers.advanced_activations.ELU(alpha=1.0)))
model.add(GRU(512, return_sequences=True,consume_less='gpu', \
                init='glorot_normal'))
model.add(Dropout(0.02))
model.add(GRU(512, return_sequences=True, consume_less='gpu', \
               init='glorot_normal'))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(256)))
model.add(Activation(keras.layers.advanced_activations.ELU(alpha=1.0)))
model.compile(loss='mean_squared_error', optimizer='nadam')

model.fit(train_x, train_y, batch_size=10, nb_epoch=50, validation_data=(test_x,test_y), callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')])

score = model.evaluate(test_x,test_y)
print(score)
model.save_weights('../../weights/2GRU-hyperas.hd5')



