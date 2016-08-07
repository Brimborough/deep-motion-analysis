from __future__ import print_function
import os    
os.environ['THEANO_FLAGS'] = "device=cpu"
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, TimeDistributed
from keras.layers import LSTM, merge, Input
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

print('Build model...')
inp = Input(shape=(29,259))
i = TimeDistributed(Dense(256))(inp)
i = Activation(keras.layers.advanced_activations.ELU(alpha=1.0))(i)

l1 = LSTM(256, return_sequences=True, consume_less='gpu', \
               init='glorot_normal')(i)

input_lstm2 = merge([i, l1], mode='concat')
l2 = LSTM(256, return_sequences=True, consume_less='gpu', \
               init='glorot_normal')(input_lstm2)

input_lstm3 = merge([i, l2], mode='concat')
l3 = LSTM(256, return_sequences=True, consume_less='gpu', \
               init='glorot_normal')(input_lstm3)

input_fcout = merge([l2, l1, l3], mode='concat')

i = Dropout(0.2)(input_fcout)
i = TimeDistributed(Dense(512))(i)
i = Activation(keras.layers.advanced_activations.ELU(alpha=1.0))(i)
i = TimeDistributed(Dense(256))(i)
out = Activation(keras.layers.advanced_activations.ELU(alpha=1.0))(i)

model = Model(input=inp,output=out)

model.compile(loss='mean_squared_error', optimizer='nadam')

num_frame_pred = 28
for frame in [1,2,5,8,10]:
	visualise(model, '256x3-2FC-SNN.hd5',orig_file="Joe/edin_shuffled.npz", frame=frame, num_frame_pred=num_frame_pred, num_pred_iter=0,\
	 anim_frame_start=((30-num_frame_pred)*8), anim_frame_end=232, test_start=310, control=True)