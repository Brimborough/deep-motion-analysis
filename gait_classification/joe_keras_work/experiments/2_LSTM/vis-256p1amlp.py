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
from visualise_after import visualise

# build the model: 2 stacked LSTM
print('Build model...')
#FUNCTIONAL MODEL
inp = Input(shape=(29,256))
i = TimeDistributed(Dense(256))(inp)
i = Activation(keras.layers.advanced_activations.ELU(alpha=1.0))(i)

i = LSTM(256, return_sequences=True, consume_less='gpu', \
               init='glorot_normal')(i)
inp2 = Input(shape=(29,3))
x = TimeDistributed(Dense(6))(inp2)
x = Activation(keras.layers.advanced_activations.ELU(alpha=1.0))(x)
i = merge([i, x], mode='concat', concat_axis=2)
i = LSTM(256, return_sequences=True, consume_less='gpu', \
               init='glorot_normal')(i)
i = TimeDistributed(Dense(256))(i)
i = Activation(keras.layers.advanced_activations.ELU(alpha=1.0))(i)
model = Model(input=[inp,inp2],output=i)

model.compile(loss='mean_squared_error', optimizer='nadam')


visualise(model, '2LSTM-256-p1amlp.hd5',orig_file="Joe/edin_shuffled.npz", frame=6, num_frame_pred=25, num_pred_iter=0, anim_frame_start=40, test_start=310, control=True)