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
sys.path.append('../../../representation_learning/')
from nn.Network import InverseNetwork, AutoEncodingNetwork
from nn.AnimationPlot import animation_plot

def data_util(preds,x, sample_num):
    d1 = preds[sample_num] # - take the first
    d2 = np.concatenate((x[sample_num,:-28], d1)) #First X
    return d2

def visualise(model, weight_file, frame):
    #Load the preprocessed version, saving on computation
    X = np.load('../../../data/Joe/edin_shuffled.npz')['clips']
    X = np.swapaxes(X, 1, 2).astype(theano.config.floatX)
    X = X[:,:-4]
    preprocess = np.load('../../../data/Joe/preprocess.npz')
    X = (X - preprocess['Xmean']) / preprocess['Xstd']


    data = np.load('../../../data/Joe/sequential_final_frame.npz')
    train_x = data['train_x']
    train_y = data['train_y']
    test_x = data['test_x']
    test_y = data['test_y']

    model.load_weights('../../weights/'+str(weight_file))

    pre_lat = np.load('../../../data/Joe/pre_proc_lstm.npz')
    orig = np.concatenate([train_x[1:2],train_y[1:2][:,-1:]], axis=1)
    orig = (orig*pre_lat['std']) + pre_lat['mean']
    orig = orig.swapaxes(2,1)
    print(orig.shape)

    for i in range(1):
        preds = model.predict(train_x)[:,-29:] # SHAPE - [321,29,256], want final prediction, use -1 for time distributed.
        train_x = np.expand_dims(data_util(preds,train_x, 1), frame) #Place together all, then only use the final one

    
    train_x = (train_x*pre_lat['std']) + pre_lat['mean']

    d2 = train_x.swapaxes(2, 1) #Swap back
    dat = d2 #For time distributed

    from network import network
    network.load([
        None,
        '../../../models/conv_ae/layer_0.npz', None, None,
        '../../../models/conv_ae/layer_1.npz', None, None,
        '../../../models/conv_ae/layer_2.npz', None, None,
    ])


    # Run find_frame.py to find which original motion frame is being used.
    Xorig = X[frame:frame+1]

    # Transform dat back to original latent space
    shared = theano.shared(orig).astype(theano.config.floatX)

    Xrecn = InverseNetwork(network)(shared).eval()
    Xrecn = np.array(Xrecn)

    # Transform dat back to original latent space
    shared2 = theano.shared(dat).astype(theano.config.floatX)

    Xpred = InverseNetwork(network)(shared2).eval()
    Xpred = np.array(Xpred) # Just Decoding
    #Xrecn = np.array(AutoEncodingNetwork(network)(Xrecn).eval()) # Will the AE help solve noise.

    # Last 3 - Velocities so similar root
    Xrecn[:, -3:] = Xorig[:, -3:]
    Xpred[:, -3:] = Xorig[:, -3:]

    #Back to original data space
    Xorig = (Xorig * preprocess['Xstd']) + preprocess['Xmean']
    Xrecn = (Xrecn * preprocess['Xstd']) + preprocess['Xmean']
    Xpred = (Xpred * preprocess['Xstd']) + preprocess['Xmean']

    animation_plot([Xorig, Xrecn, Xpred], interval=15.15, labels=['Root','Reconstruction', 'Predicted'])

    #### DOING SO WELL BECUASE IT's OVERFIT!!!!, TRAIN AGAIN!