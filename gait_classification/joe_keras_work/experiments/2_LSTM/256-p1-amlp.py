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
sys.path.append('../../../representation_learning/')

from nn.Network import InverseNetwork, AutoEncodingNetwork
from nn.AnimationPlot import animation_plot


data = np.load('../../../data/Joe/sequential_final_frame.npz')
train_x = data['train_x']
train_y = data['train_y']
test_x = data['test_x']
test_y = data['test_y']

control = np.load('../../../data/Joe/edin_shuffled_control.npz')['control']
control = control.swapaxes(1,2)
train_control = control[:310,8::8]
#train_n = np.concatenate((train_x, train_control[:,:29]), axis=2)

test_control = control[310:,8::8]
#test_n = np.concatenate((test_x, test_control[:,:29]), axis=2)

#print(train_n.shape)
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

print('Training model...')
model.fit([train_x, train_control], train_y, batch_size=10, nb_epoch=200, validation_data=([test_x, test_control],test_y))

model.save_weights('../../weights/2LSTM-256-p1amlp.hd5', overwrite=True)

