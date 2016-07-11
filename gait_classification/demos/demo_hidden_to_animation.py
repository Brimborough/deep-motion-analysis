'''
 Batch 1 by 1 so we can get correct animations!
'''
import numpy as np
import theano
import sys
sys.path.append('../representation_learning/')

from nn.Network import InverseNetwork
from nn.NoiseLayer import NoiseLayer
from nn.Pool1DLayer import Pool1DLayer

from nn.AnimationPlot import animation_plot

rng = np.random.RandomState(23455)

#Load the preprocessed version, saving on computation
X = np.load('../data/Joe/data_edin_locomotion.npz')['clips']
X = np.swapaxes(X, 1, 2).astype(theano.config.floatX)
X = X[:,:-4]
preprocess = np.load('../data/Joe/preprocess.npz')
X = (X - preprocess['Xmean']) / preprocess['Xstd']

H = np.load('../data/Joe/HiddenActivations.npz')['Orig']

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


# Go through inputs 1by1
for input in range(len(X)):
    input = 50
    Xorig = X[input:input+1]

    #Theano shared object to pass to network
    shared = theano.shared(H[input:input+1])

    # Recreate
    Xrecno = np.array(InverseNetwork(network)(shared).eval()).astype(theano.config.floatX)

    #Last 3 - Velocities so similar root
    Xrecno[:,-3:] = Xorig[:,-3:]

    Xorig = (Xorig * preprocess['Xstd']) + preprocess['Xmean']
    Xrecno = (Xrecno * preprocess['Xstd']) + preprocess['Xmean']

    animation_plot([Xorig, Xrecno], interval=15.15)

