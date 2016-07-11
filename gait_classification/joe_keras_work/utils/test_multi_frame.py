from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, TimeDistributed
from keras.layers import LSTM, GRU
from keras.optimizers import Nadam
import keras
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import theano

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def test_mf(weight_file):

	model = Sequential()
	model.add(TimeDistributed(Dense(256), input_shape=(29,256)))
	model.add(Activation(keras.layers.advanced_activations.ELU(alpha=1.0)))
	model.add(GRU(256, return_sequences=True, consume_less='gpu', \
	                init='glorot_normal'))
	model.add(Dropout(0.117))
	model.add(GRU(512, return_sequences=True, consume_less='gpu', \
	               init='glorot_normal'))
	model.add(Dropout(0.0266))
	model.add(TimeDistributed(Dense(256)))
	model.add(Activation(keras.layers.advanced_activations.ELU(alpha=1.0)))
	# TimedistributedDense on top - Can then set output vectors to be next sequence!

	model.compile(loss='mean_squared_error', optimizer='nadam')

	model.load_weights(str(weight_file))
	data = np.load('../../data/Joe/sequential_final_frame.npz')
	train_x = data['train_x']
	train_y = data['train_y']
    
	pred = model.predict(train_x[0:1])
	print(pred.shape)
	print(rmse(pred, train_y[0:1][-1:]))

	train_x[0,28,:] = 0
	pred = model.predict(train_x[0:1])

	print(rmse(pred, train_y[0:1][-2:-1]))


test_mf('../weights/sequential-0.6.hd5')