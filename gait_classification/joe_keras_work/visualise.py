from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, TimeDistributed
from keras.layers import LSTM
from keras.optimizers import Nadam
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import theano
sys.path.append('../representation_learning/')

from nn.Network import InverseNetwork, AutoEncodingNetwork
from nn.AnimationPlot import animation_plot

def data_util(preds,x):
    print(preds.shape)
    print(x.shape)
    d1 = preds[0] # - take the first
    d2 = np.concatenate((x[0], d1)) #First X
    return d2

#Load the preprocessed version, saving on computation
X = np.load('../data/Joe/data_edin_locomotion.npz')['clips']
X = np.swapaxes(X, 1, 2).astype(theano.config.floatX)
X = X[:,:-4]
preprocess = np.load('../data/Joe/preprocess.npz')
X = (X - preprocess['Xmean']) / preprocess['Xstd']


data = np.load('../data/Joe/sequential_final_frame.npz')
train_x = data['train_x']
train_y = data['train_y']
test_x = data['test_x']
test_y = data['test_y']

model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=(29, 256), consume_less='gpu', \
                init='glorot_normal'))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=True, consume_less='gpu', \
               init='glorot_normal'))
model.add(Dropout(0.2))
model.add(LSTM(1024, return_sequences=True, consume_less='gpu', \
               init='glorot_normal'))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(256)))
model.add(Activation(keras.layers.advanced_activations.ELU(alpha=1.0)))
# TimedistributedDense on top - Can then set output vectors to be next sequence!
model.compile(loss='mean_squared_error', optimizer='nadam')
model.load_weights('my_model_weights.h5')

for i in range(1):
    preds = model.predict(train_x)[:,-1:] # SHAPE - [321,29,256], want final prediction, use -1 for time distributed.
    train_x = np.expand_dims(data_util(preds,train_x),0) #Place together all, then only use the final one

d2 = train_x.swapaxes(2, 1) #Swap back
dat = d2 #For time distributed

from network import network
network.load([
    None,
    '../models/conv_ae/layer_0.npz', None, None,
    '../models/conv_ae/layer_1.npz', None, None,
    '../models/conv_ae/layer_2.npz', None, None,
])


# Run find_frame.py to find which original motion frame is being used.
Xorig = X[134:135]

# Transform dat back to original latent space
shared = theano.shared(dat).astype(theano.config.floatX)

Xrecn = InverseNetwork(network)(shared).eval()
Xrecn = np.array(Xrecn) # Just Decoding
#Xrecn = np.array(AutoEncodingNetwork(network)(Xrecn).eval()) # Will the AE help solve noise.

# Last 3 - Velocities so similar root
Xrecn[:, -3:] = Xorig[:, -3:]

#Back to original data space
Xorig = (Xorig * preprocess['Xstd']) + preprocess['Xmean']
Xrecn = (Xrecn * preprocess['Xstd']) + preprocess['Xmean']

animation_plot([Xorig, Xrecn], interval=15.15)