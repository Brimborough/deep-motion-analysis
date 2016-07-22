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

train_control = np.zeros((310,29,12))
for num,i in enumerate([8,10,12,14]):
	train_control[:,:,num*3:(num+1)*3] = np.expand_dims(control[:310,i::8],0)

train_n = np.concatenate((train_x, train_control[:,:29]), axis=2)

test_control = np.zeros((11,29,12))
for num,i in enumerate([8,10,12,14]):
	test_control[:,:,num*3:(num+1)*3] = np.expand_dims(control[310:,i::8],0)

test_n = np.concatenate((test_x, test_control[:,:29]), axis=2)

# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
#Potentially put LSTM here also, going over entire sequence controls....
model.add(TimeDistributed(Dense(256), input_shape=(29, 268)))
model.add(Activation(keras.layers.advanced_activations.ELU(alpha=1.0)))
model.add(LSTM(256, return_sequences=True, consume_less='gpu', \
               init='glorot_normal'))
model.add(LSTM(256, return_sequences=True, consume_less='gpu', \
               init='glorot_normal'))
model.add(Dropout(0.1))
model.add(TimeDistributed(Dense(256)))
model.add(Activation(keras.layers.advanced_activations.ELU(alpha=1.0)))
# TimedistributedDense on top - Can then set output vectors to be next sequence!

model.compile(loss='mean_squared_error', optimizer='nadam')

print('Training model...')
model.fit(train_n, train_y, batch_size=10, nb_epoch=200)

score = model.evaluate(test_n,test_y)
print(score)
model.save_weights('../../weights/256-256-alt.hd5', overwrite=True)

