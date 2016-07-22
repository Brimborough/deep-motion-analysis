from __future__ import print_function
import os    
os.environ['THEANO_FLAGS'] = "device=cpu"
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, TimeDistributed
from keras.layers import LSTM
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

# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(TimeDistributed(Dense(256), input_shape=(29, 259)))
model.add(Activation(keras.layers.advanced_activations.ELU(alpha=1.0)))
model.add(LSTM(512, return_sequences=True, consume_less='gpu', \
               init='glorot_normal'))
model.add(LSTM(512, return_sequences=True, consume_less='gpu', \
               init='glorot_normal'))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(256)))
model.add(Activation(keras.layers.advanced_activations.ELU(alpha=1.0)))
# TimedistributedDense on top - Can then set output vectors to be next sequence!
model.compile(loss='mean_squared_error', optimizer='nadam')

num_frame_pred = 28
for frame in [1,2,5,8,10]:
	visualise(model, '2LSTMC-512-512.2.hd5',orig_file="Joe/edin_shuffled.npz", frame=frame, num_frame_pred=num_frame_pred, num_pred_iter=0,\
	 anim_frame_start=((30-num_frame_pred)*8), anim_frame_end=232, test_start=310, control=True)