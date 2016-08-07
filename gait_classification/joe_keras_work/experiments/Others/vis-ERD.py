'''
    Train on t+1 vector for every t, only trying to predict the final frame, given 29 as seed.
'''

from __future__ import print_function
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, TimeDistributed
from keras.layers import LSTM, GRU, Input, merge
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
inp = Input(shape=(239,66))
i = TimeDistributed(Dense(500))(inp)
i = Activation('relu')(i)
i = TimeDistributed(Dense(500))(i)

l1 = LSTM(1000, return_sequences=True, consume_less='gpu', \
               init='glorot_normal')(i)

input_lstm2 = merge([i, l1], mode='concat', concat_axis=2)
l2 = LSTM(1000, return_sequences=True, consume_less='gpu', \
               init='glorot_normal')(input_lstm2)

input_fcout = merge([l2, l1], mode='concat', concat_axis=2)

i = TimeDistributed(Dense(500))(input_fcout)
i = Activation('relu')(i)
i = TimeDistributed(Dense(500))(i)
i = Activation('relu')(i)
out = TimeDistributed(Dense(66))(i)

model = Model(input=inp,output=out)

def euclid_loss(y_t, y):
	scaling = 1
	if y.ndim > 2:
		scaling = (y.shape[0])*y.shape[2]
	y_new = y.flatten()
	y_t_new = y_t.flatten()

	return scaling * T.mean(T.sqr(y_new-y_t_new))

nadam = Nadam(clipnorm=25)
model.compile(loss=euclid_loss, optimizer=nadam)

num_frame_pred = 160
visualise(model, 'ERD.hd5',orig_file="Joe/edin_shuffled.npz", num_frame_pred=num_frame_pred, num_pred_iter=1,\
	 anim_frame_start=150, test_start=310, control=False)