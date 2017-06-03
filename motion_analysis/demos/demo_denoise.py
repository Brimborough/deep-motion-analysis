import numpy as np
import theano
import theano.tensor as T

import sys
sys.path.append('../representation_learning/')

from theano.tensor.shared_randomstreams import RandomStreams

from nn.Network import AutoEncodingNetwork
from nn.NoiseLayer import NoiseLayer
from nn.Pool1DLayer import Pool1DLayer

from nn.AnimationPlot import animation_plot

rng = np.random.RandomState(23455)

X = np.load('../data/data_cmu.npz')['clips']
X = np.swapaxes(X, 1, 2).astype(theano.config.floatX)
X = X[:,:-4]

preprocess = np.load('../data/Joe/preprocess.npz')
X = (X - preprocess['Xmean']) / preprocess['Xstd']

from network import network
network.load([
    None,
    '../models/conv_ae/layer_0.npz', None, None,
    '../models/conv_ae/layer_1.npz', None, None,
    '../models/conv_ae/layer_2.npz', None, None,
])

for layer in network.layers:
    if isinstance(layer, NoiseLayer): layer.amount = 0.0
    if isinstance(layer, Pool1DLayer):  layer.depooler = lambda x, **kw: x/2

while True:
    
    index = rng.randint(len(X)-1)
    amount = 0.5
    
    Xorig = X[index:index+1]
    Xnois = (Xorig * rng.binomial(size=Xorig.shape, n=1, p=(1-amount)).astype(theano.config.floatX))
    # Replace with just Network and save the output.
    #   - will work as is just demo of calling the network not training it!
    Xrecn = np.array(AutoEncodingNetwork(network)(Xnois).eval())
    Xrecn[:,-3:] = Xorig[:,-3:]
    
    Xorig = (Xorig * preprocess['Xstd']) + preprocess['Xmean']
    Xnois = (Xnois * preprocess['Xstd']) + preprocess['Xmean']
    Xrecn = (Xrecn * preprocess['Xstd']) + preprocess['Xmean']
    
    animation_plot([Xorig, Xnois, Xrecn], interval=15.15)
    
