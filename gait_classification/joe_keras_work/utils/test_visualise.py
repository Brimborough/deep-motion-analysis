from __future__ import print_function
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


def test_vis(frame):

    X = np.load('../../data/Joe/edin_shuffled.npz')['clips']
    X = np.swapaxes(X, 1, 2).astype(theano.config.floatX)
    X = X[:,:-4]
    preprocess = np.load('../../data/Joe/preprocess.npz')
    X = (X - preprocess['Xmean']) / preprocess['Xstd']

    print(preprocess['Xmean'].shape)
    data = np.load('../../data/Joe/sequential_final_frame.npz')
    train_x = data['train_x']
    train_y = data['train_y']

    pre_lat = np.load('../../data/Joe/pre_proc_lstm.npz')
    orig = np.concatenate([train_x[frame:frame+1],train_y[frame:frame+1][:,-1:]], axis=1)
    orig = (orig*pre_lat['std']) + pre_lat['mean']
    orig = orig.swapaxes(2,1)

    not_shuffled = np.load('../../data/Joe/HiddenActivations.npz')['Orig'][frame:frame+1]
    not_shuffled = not_shuffled.swapaxes(2,1)
    not_shuffled = (not_shuffled - pre_lat['mean']) / pre_lat['std']
    not_shuffled = (not_shuffled * pre_lat['std']) + pre_lat['mean']
    not_shuffled = not_shuffled.swapaxes(1, 2)

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

    # Transform dat back to original latent space
    not_shuffled = theano.shared(not_shuffled).astype(theano.config.floatX)

    Xrecn = InverseNetwork(network)(shared).eval()
    Xrecn = np.array(Xrecn)

    not_shuffled = InverseNetwork(network)(not_shuffled).eval()
    not_shuffled = np.array(not_shuffled)

    # Last 3 - Velocities so similar root
    Xrecn[:, -3:] = Xorig[:, -3:]
    not_shuffled[:, -3:] = Xorig[:, -3:]

    #Back to original data space
    Xorig = (Xorig * preprocess['Xstd']) + preprocess['Xmean']
    Xrecn = (Xrecn * preprocess['Xstd']) + preprocess['Xmean']
    not_shuffled = (not_shuffled * preprocess['Xstd']) + preprocess['Xmean']

    animation_plot([Xorig, not_shuffled, Xrecn], interval=15.15, labels=['Root', 'Not Shuffled', 'Reconstruction'])


test_vis(0)