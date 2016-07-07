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
from theano import function, config, shared, sandbox
import theano.tensor as T
import time


def model(params):
    
    '''
    Data providing function:

    This function is separated from model() so that hyperopt
    won't reload data for each evaluation run.
    '''
    data = np.load('../../data/Joe/sequential_final_frame.npz')
    train_x = data['train_x']
    train_y = data['train_y']
    test_x = data['test_x']
    test_y = data['test_y']

    drop1 = params['drop1'][0]
    drop2 = params['drop2'][0]
    drop3 = params['drop3'][0]
    '''
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    '''
    model = Sequential()
    model.add(LSTM(256, return_sequences=True, input_shape=(29, 256), consume_less='gpu', \
                    init='glorot_normal'))
    model.add(Dropout(drop1))
    model.add(LSTM(512, return_sequences=True, consume_less='gpu', \
                   init='glorot_normal'))
    model.add(Dropout(drop2))
    model.add(LSTM(1024, return_sequences=True, consume_less='gpu', \
                   init='glorot_normal'))
    model.add(Dropout(drop3))
    model.add(TimeDistributed(Dense(256)))
    model.add(Activation(keras.layers.advanced_activations.ELU(alpha=1.0)))
    model.compile(loss='mean_squared_error', optimizer='nadam') 
    

    model.fit(train_x, train_y, batch_size=20, nb_epoch=100, validation_data=(test_x,test_y))
    
    loss = model.evaluate(test_x, test_y, verbose=0)

    print(loss)
    

    return loss



# Write a function like this called 'main'
def main(job_id, params):
    print('Anything printed here will end up in the output directory for job #%d' % job_id)
    print(params)

    return model(params)
