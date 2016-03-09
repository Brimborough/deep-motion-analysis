import numpy as np
import os, sys, inspect
sys.path.append('../representation_learning/')

import theano
import theano.tensor as T

from nn.Network import AutoEncodingNetwork
from nn.NoiseLayer import NoiseLayer
from nn.Pool1DLayer import Pool1DLayer

from nn.AnimationPlot import animation_plot

rng = np.random.RandomState(45425)

# 31-38: Angry kicking
# 110-123: Childlike walking
# 451-468: Sexy walk
X = np.load('./data_styletransfer.npz')['clips']
X = np.swapaxes(X, 1, 2).astype(theano.config.floatX)
# TODO: Why are we discarding joints?
X = X[:,:-4]

preprocess = np.load('styletransfer_preprocessed.npz')
X = (X - preprocess['Xmean']) / preprocess['Xstd']

from network import network
network.load([
    None,
    '../models/layer_0.npz', None, None,
    '../models/layer_1.npz', None, None,
    '../models/layer_2.npz', None, None,
])

for layer in network.layers:
    if isinstance(layer, NoiseLayer): layer.amount = 0.0
    if isinstance(layer, Pool1DLayer):  layer.depooler = lambda x, **kw: x/2

#index = rng.randint(len(X)-1)

# Angry kick
angry_index = 31
# Childlike walk
child_index = 110
# Sexy walk
sexy_index = 453
    
Xkick = X[angry_index:angry_index+1]
Xchild = X[child_index:child_index+1]
Xsexy = X[sexy_index:sexy_index+1]

Xkick = (Xkick * preprocess['Xstd']) + preprocess['Xmean']
Xchild = (Xchild * preprocess['Xstd']) + preprocess['Xmean']
Xsexy = (Xsexy * preprocess['Xstd']) + preprocess['Xmean']

animation_plot([Xkick, Xchild, Xsexy], interval=15.15, labels=['Angry kick', 'Childlike walk', 'Sexy walk'])
    
