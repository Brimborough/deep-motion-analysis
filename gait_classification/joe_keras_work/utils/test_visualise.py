from __future__ import print_function
import os    
os.environ['THEANO_FLAGS'] = "device=cpu"
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, TimeDistributed
from keras.layers import LSTM
from keras.optimizers import nadam
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import theano

sys.path.append('../../representation_learning/')
from nn.Network import InverseNetwork, AutoEncodingNetwork
from nn.AnimationPlot import animation_plot


def test_vis(frame, test_frame):

    X = np.load('../../data/Joe/edin_shuffled.npz')['clips']
    X = np.swapaxes(X, 1, 2).astype(theano.config.floatX)
    X = X[:,:-4]
    preprocess = np.load('../../data/Joe/preprocess.npz')
    X = (X - preprocess['Xmean']) / preprocess['Xstd']

    print(preprocess['Xmean'].shape)
    data = np.load('../../data/Joe/sequential_final_frame.npz')
    train_x = data['train_x']
    train_y = data['train_y']
    test_x = data['test_x']
    test_y = data['test_y']

    pre_lat = np.load('../../data/Joe/pre_proc_lstm.npz')
    orig = np.concatenate([train_x[frame:frame+1],train_y[frame:frame+1][:,-1:]], axis=1)
    orig = (orig*pre_lat['std']) + pre_lat['mean']
    orig = orig.swapaxes(2,1)

    frame = test_frame
    test = np.concatenate([test_x[frame:frame+1],test_y[frame:frame+1][:,-1:]], axis=1)
    test = (test*pre_lat['std']) + pre_lat['mean']
    test = test.swapaxes(2,1)

    from network import network
    network.load([
        None,
        '../../models/conv_ae/layer_0.npz', None, None,
        '../../models/conv_ae/layer_1.npz', None, None,
        '../../models/conv_ae/layer_2.npz', None, None,
    ])


    # Run find_frame.py to find which original motion frame is being used.
    Xorig = X[frame:frame+1]

    # Transform dat back to original latent space
    shared = theano.shared(orig).astype(theano.config.floatX)

    Xrecn = InverseNetwork(network)(shared).eval()
    Xrecn = np.array(Xrecn)

    shared = theano.shared(test).astype(theano.config.floatX)
    Xtest = InverseNetwork(network)(shared).eval()
    Xtest = np.array(Xtest)

    # Last 3 - Velocities so similar root
    #Xrecn[:, -3:] = Xorig[:, -3:]
    #Xtest[:, -3:] = Xorig[:, -3:]

    #Back to original data space
    Xorig = (Xorig * preprocess['Xstd']) + preprocess['Xmean']
    Xrecn = ((Xrecn * preprocess['Xstd']) + preprocess['Xmean'])[:,:,120:]
    Xtest = ((Xtest * preprocess['Xstd']) + preprocess['Xmean'])[:,:,:121]

    animation_plot([Xorig, Xrecn, Xtest], interval=15.15, labels=['Root', 'Reconstruction', 'Test'])

test_vis(137,0)
print('1')
test_vis(233,1)
print('2')
test_vis(22,2)
print('3')
test_vis(275,3)
print('5')
test_vis(65,5)
print('7')
test_vis(78,7)
print('8')
test_vis(104,8)
print('9')
test_vis(83,9)
print('10')
test_vis(170,10)