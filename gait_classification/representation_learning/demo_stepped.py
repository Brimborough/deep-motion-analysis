import numpy as np
import theano
import theano.tensor as T

from nn.Network import AutoEncodingNetwork
from nn.NoiseLayer import NoiseLayer
from nn.Pool1DLayer import Pool1DLayer

from nn.AnimationPlot import animation_plot

rng = np.random.RandomState(45425)

X = np.load('./data_cmu.npz')['clips']
X = np.swapaxes(X, 1, 2).astype(theano.config.floatX)
X = X[:,:-4]

preprocess = np.load('preprocess.npz')

X = (X - preprocess['Xmean']) / preprocess['Xstd']

from network import network
network.load([
    None,
    'layer_0.npz', None, None,
    'layer_1.npz', None, None,
    'layer_2.npz', None, None,
])

for layer in network.layers:
    if isinstance(layer, NoiseLayer): layer.amount = 0.0
    if isinstance(layer, Pool1DLayer):  layer.depooler = lambda x, **kw: x/2

while True:
    
    index = rng.randint(len(X)-1)
        
    Xorig = X[index:index+1]
    Xstep = Xorig[:,:,::15].repeat(15, axis=2)
        
    Xrecn = np.array(AutoEncodingNetwork(network)(Xstep).eval())
    Xrecn[:,-3:] = Xorig[:,-3:]
    
    Xorig = (Xorig * preprocess['Xstd']) + preprocess['Xmean']
    Xstep = (Xstep * preprocess['Xstd']) + preprocess['Xmean']
    Xrecn = (Xrecn * preprocess['Xstd']) + preprocess['Xmean']
    
    animation_plot([Xorig, Xstep, Xrecn], interval=15.15)
    
