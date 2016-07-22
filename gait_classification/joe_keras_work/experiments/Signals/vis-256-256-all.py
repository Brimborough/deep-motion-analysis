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
from visualise_after import visualise

# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
#Potentially put LSTM here also, going over entire sequence controls....
model.add(TimeDistributed(Dense(256), input_shape=(29, 280)))
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

num_frame_pred = 2
running_norm = 0
running_rmse = 0
for frame in [1,2,5,8,10]:
	n, r = visualise(model, '256-256-all.hd5',orig_file="Joe/edin_shuffled.npz", frame=frame, num_frame_pred=num_frame_pred, num_pred_iter=0,\
	 anim_frame_start=((30-num_frame_pred)*8), anim_frame_end=232, test_start=310, control=True, control_type='All')

	running_rmse = running_rmse + r
	print(running_rmse)
	running_norm = running_norm + n
	print(running_norm)

print("RMSE between orig: " + str(running_rmse/5))
print("Norm between orig: " + str(running_norm/5))
