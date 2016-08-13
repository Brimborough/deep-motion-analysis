'''
    Train on t+1 vector for every t, only trying to predict the final frame, given 29 as seed.
'''

from __future__ import print_function

from models import Sequential
from keras.layers import Dense, Activation, Dropout, TimeDistributed
from keras.layers import LSTM, GRU, Input
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Nadam
import keras
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import theano
from theano import tensor as T
sys.path.append('../../../representation_learning/')

from nn.Network import InverseNetwork, AutoEncodingNetwork
from nn.AnimationPlot import animation_plot

control = np.load('../../../data/Joe/edin_shuffled_control.npz')['control']
train_control = control[:310]
test_control = control[310:]
data = np.load('../../../data/Joe/edin_shuffled.npz')['clips']
data = data[:,:,:-4]

data_std = data.std()
data_mean = data.mean(axis=2).mean(axis=0)[np.newaxis, :, np.newaxis]

data = (data - data_mean) / data_std

train_x = data[:310, :-1]
train_x = np.concatenate((train_x, train_control[:,:239]), axis=2)
train_y = data[:310, 1:]
train_y = np.concatenate((train_y, train_control[:,1:]), axis=2)
test_x = data[310:,:-1]
test_x = np.concatenate((test_x, test_control[:,:239]), axis=2)
test_y = data[310:, 1:]
test_y = np.concatenate((test_y, test_control[:,1:]), axis=2)


# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
#Potentially put LSTM here also, going over entire sequence controls....
model.add(TimeDistributed(Dense(500),input_shape=(239,66)))
model.add(Activation('relu'))
model.add(TimeDistributed(Dense(500)))
model.add(LSTM(1000, return_sequences=True, consume_less='gpu', \
               init='glorot_normal'))
model.add(LSTM(1000, return_sequences=True, consume_less='gpu', \
               init='glorot_normal'))
model.add(TimeDistributed(Dense(500)))
model.add(Activation('relu'))
model.add(TimeDistributed(Dense(500)))
model.add(Activation('relu'))
model.add(TimeDistributed(Dense(66)))

def euclid_loss(y_t, y):
	scaling = 1
	if y.ndim > 2:
		scaling = (y.shape[0])*y.shape[2]
	y_new = y.flatten()
	y_t_new = y_t.flatten()

	return scaling * T.mean(T.sqr(y_new-y_t_new))

nadam = Nadam(clipnorm=25)
model.compile(loss=euclid_loss, optimizer=nadam)

print('Training model...')
model.fit(train_x, train_y, batch_size=25, nb_epoch=9000)

score = model.evaluate(test_x,test_y)
print(score)
model.save_weights('../../weights/ERD-ns-con.hd5', overwrite=True)

