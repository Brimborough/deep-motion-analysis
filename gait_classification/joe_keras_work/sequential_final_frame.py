'''
    Train on t+1 vector for every t, only trying to predict the final frame, given 29 as seed.
'''

from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, TimeDistributed
from keras.layers import LSTM
from keras.optimizers import Nadam
import keras
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import theano
sys.path.append('../representation_learning/')

from nn.Network import InverseNetwork, AutoEncodingNetwork
from nn.AnimationPlot import animation_plot


def data_util(preds,x):
    preds = np.expand_dims(preds, 1)
    d1 = preds[0] # - take the first
    d2 = np.concatenate((x[0], d1)) #First X
    return d2

data = np.load('../data/Joe/sequential_final_frame.npz')
train_x = data['train_x']
train_y = data['train_y']
test_x = data['test_x']
test_y = data['test_y']

#Load the preprocessed version, saving on computation
X = np.load('../data/Joe/data_edin_locomotion.npz')['clips']
X = np.swapaxes(X, 1, 2).astype(theano.config.floatX)
X = X[:,:-4]
preprocess = np.load('../data/Joe/preprocess.npz')
X = (X - preprocess['Xmean']) / preprocess['Xstd']


# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=(29, 256), consume_less='gpu', \
                init='glorot_normal'))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=True, consume_less='gpu', \
               init='glorot_normal'))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(256)))
model.add(Activation(keras.layers.advanced_activations.ELU(alpha=1.0)))
# TimedistributedDense on top - Can then set output vectors to be next sequence!

model.compile(loss='mean_squared_error', optimizer='nadam')

print('Training model...')
hist = model.fit(train_x, train_y, batch_size=5, nb_epoch=1, validation_data=(test_x,test_y))
print(hist.history)
score = model.evaluate(test_x,test_y)
print(score)
# No eval since generative model, needs all data it can get on already miniture dataset.

#TODO - replace the final 5 frames and see how it does
#TODO: - Make 2 files, time distributed and just final outputs, as well as 2 input files. Shapes are slightly different.


for i in range(1):
    preds = model.predict(train_x)[:,-1] # SHAPE - [321,29,256], want final prediction, use -1 for time distributed.
    train_x = np.expand_dims(data_util(preds,train_x),0) #Place together all, then only use the final one

d2 = train_x.swapaxes(0, 1) #Swap back, need to concat again?!
dat = d2 #For time distributed
dat = np.expand_dims(d2, 0) #For dense, since it cuts off a dim otherwise

from network import network
network.load([
    None,
    '../models/conv_ae/layer_0.npz', None, None,
    '../models/conv_ae/layer_1.npz', None, None,
    '../models/conv_ae/layer_2.npz', None, None,
])


# Run find_frame.py to find which original motion frame is being used.
#Xorig = X[134:135]

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

