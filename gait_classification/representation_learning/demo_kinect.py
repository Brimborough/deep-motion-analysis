import numpy as np
import theano
import theano.tensor as T

from nn.Network import AutoEncodingNetwork
from nn.NoiseLayer import NoiseLayer
from nn.Pool1DLayer import Pool1DLayer

from nn.AnimationPlot import animation_plot

rng = np.random.RandomState(1245)

Xkinct = np.load('./data_edin_kinect.npz')['clips']
Xxsens = np.load('./data_edin_xsens.npz')['clips']

Xkinct = np.swapaxes(Xkinct, 1, 2).astype(theano.config.floatX)
Xxsens = np.swapaxes(Xxsens, 1, 2).astype(theano.config.floatX)

Xkinct = Xkinct[:,:-4]
Xxsens = Xxsens[:,:-4]

preprocess = np.load('preprocess.npz')
Xkinct = (Xkinct - preprocess['Xmean']) / preprocess['Xstd']
Xxsens = (Xxsens - preprocess['Xmean']) / preprocess['Xstd']

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

index = (-5000)//240

while True:
    
    Xorig = Xxsens[index:index+1]
    Xnois = Xkinct[index:index+1]
    Xrecn = np.array(AutoEncodingNetwork(network)(Xnois).eval())
    Xrecn[:,-3:] = Xorig[:,-3:]
    
    Xorig = (Xorig * preprocess['Xstd']) + preprocess['Xmean']
    Xnois = (Xnois * preprocess['Xstd']) + preprocess['Xmean']
    Xrecn = (Xrecn * preprocess['Xstd']) + preprocess['Xmean']

    animation_plot([Xorig, Xnois, Xrecn], interval=15.15)
    
    index+=2