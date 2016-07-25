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


data = np.load('../../../data/Joe/edin_shuffled.npz')['clips']
data = data[:,:,:-4]

data_std = data.std()
data_mean = data.mean(axis=2).mean(axis=0)[np.newaxis, :, np.newaxis]

data = (data - data_mean) / data_std

train_x = data[:310, :-1]
train_y = data[:310, 1:]
test_x = data[310:,:-1]
test_y = data[310:, 1:]


# build the model: 2 stacked LSTM
print('Build model...')
inp = Input(shape=(29,256))
i = TimeDistributed(Dense(256))(inp)
i = Activation(keras.layers.advanced_activations.ELU(alpha=1.0))(i)

l1 = LSTM(256, return_sequences=True, consume_less='gpu', \
               init='glorot_normal')(i)

input_lstm2 = merge([i, l1], mode='concat')
l2 = LSTM(256, return_sequences=True, consume_less='gpu', \
               init='glorot_normal')(input_lstm2)

input_lstm3 = merge([i, l2], mode='concat')
l2 = LSTM(256, return_sequences=True, consume_less='gpu', \
               init='glorot_normal')(input_lstm3)

input_fcout = merge([l2, l1, l3], mode='concat')

i = TimeDistributed(Dense(256))(input_fcout)
i = Activation(keras.layers.advanced_activations.ELU(alpha=1.0))(i)
i = TimeDistributed(Dense(256))(i)
out = Activation(keras.layers.advanced_activations.ELU(alpha=1.0))(i)

model = Model(input=inp,output=out)

nadam = Nadam(clipnorm=25)
model.compile(loss='mean_squared_error', optimizer=nadam)

print('Training model...')
model.fit(train_x, train_y, batch_size=25, nb_epoch=1600)

score = model.evaluate(test_x,test_y)
print(score)
model.save_weights('../../weights/256x3.hd5', overwrite=True)

