'''
    Train on t+1 vector for every t, only trying to predict the final frame, given 29 as seed.
'''

from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, TimeDistributed
from keras.layers import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Nadam
import keras
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import theano
sys.path.append('../../representation_learning/')

from nn.Network import InverseNetwork, AutoEncodingNetwork
from nn.AnimationPlot import animation_plot


data = np.load('../../data/Joe/sequential_final_frame.npz')
train_x = np.concatenate((data['train_x'],data['test_x']))
print(data['train_y'].shape)
print(data['test_y'].shape)

train_y = np.concatenate((data['train_y'],data['test_y']))

print(train_x.shape)
train_z = np.zeros((321,29,512))
for i in range(0,28):
	train_z[:,i] = np.concatenate((train_x[:,i], train_x[:,i+1]), axis=1)

train_n = train_z.copy()#.reshape(321,29*512)

print(train_y.shape)

# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(TimeDistributed(Dense(256),input_shape=(29,512)))
model.add(Activation(keras.layers.advanced_activations.ELU(alpha=1.0)))
model.add(TimeDistributed(Dense(256)))
model.add(Activation(keras.layers.advanced_activations.ELU(alpha=1.0)))
model.add(TimeDistributed(Dense(256)))
model.add(Activation(keras.layers.advanced_activations.ELU(alpha=1.0)))
# TimedistributedDense on top - Can then set output vectors to be next sequence!

model.compile(loss='mean_squared_error', optimizer='nadam')

print('Training model...')
model.fit(train_n, train_y, batch_size=20, nb_epoch=200)

model.save_weights('../weights/base_line.hd5', overwrite=True)
