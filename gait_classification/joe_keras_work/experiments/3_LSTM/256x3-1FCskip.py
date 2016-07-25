'''
    Train on t+1 vector for every t, only trying to predict the final frame, given 29 as seed.
'''

from __future__ import print_function

from models import Model
from keras.layers import Dense, Activation, Dropout, TimeDistributed
from keras.layers import LSTM, GRU, Input, merge
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Nadam
#from optimizers import SGD
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

data = np.load('../../../data/Joe/sequential_final_frame.npz')
train_x = data['train_x']
train_y = data['train_y']
test_x = data['test_x']
test_y = data['test_y']

control = np.load('../../../data/Joe/edin_shuffled_control.npz')['control']
control = control.swapaxes(1,2)
train_control = control[:310,8::8]
train_x = np.concatenate((train_x, train_control[:,:29]), axis=2)

test_control = control[310:,8::8]
test_x = np.concatenate((test_x, test_control[:,:29]), axis=2)


# build the model: 2 stacked LSTM
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

i = TimeDistributed(Dense(256))(input_fcout)
out = Activation(keras.layers.advanced_activations.ELU(alpha=1.0))(i)

model = Model(input=inp,output=out)

nadam = Nadam(clipnorm=25)
model.compile(loss='mean_squared_error', optimizer=nadam)

print('Training model...')
model.fit(train_x, train_y, batch_size=10, nb_epoch=1200)

score = model.evaluate(test_x,test_y)
print(score)
model.save_weights('../../weights/256x3-1FC.hd5', overwrite=True)

