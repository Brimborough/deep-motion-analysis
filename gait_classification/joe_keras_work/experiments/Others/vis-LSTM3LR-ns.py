'''
    Train on t+1 vector for every t, only trying to predict the final frame, given 29 as seed.
'''

from __future__ import print_function
from keras.models import Sequential
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
sys.path.append('../../utils/')
from visualise_mocap import visualise

# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
#Potentially put LSTM here also, going over entire sequence controls....
model.add(TimeDistributed(Dense(66),input_shape=(239,66)))
model.add(LSTM(1000, return_sequences=True, consume_less='gpu', \
               init='glorot_normal'))
model.add(LSTM(1000, return_sequences=True, consume_less='gpu', \
               init='glorot_normal'))
model.add(LSTM(1000, return_sequences=True, consume_less='gpu', \
               init='glorot_normal'))
model.add(TimeDistributed(Dense(66)))

def euclid_loss(y_t, y):
	scaling = 1
	if y.ndim > 2:
		scaling = (y.shape[0])*y.shape[2]
	y_new = y.flatten()
	y_t_new = y_t.flatten()

	return scaling * T.mean(T.sqr(y_new-y_t_new))

model.compile(loss=euclid_loss, optimizer='nadam')

num_frame_pred = 16
visualise(model, 'LSTM3LR-noskip.hd5',orig_file="Joe/edin_shuffled.npz", num_frame_pred=num_frame_pred, num_pred_iter=1,\
	 anim_frame_start=216, test_start=310, control=False)