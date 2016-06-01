import numpy as np
import theano
import theano.tensor as T
from theano import function,printing

from nn.AnimationPlot import animation_plot
from nn.Network import Network, InverseNetwork
from nn.NoiseLayer import NoiseLayer
from nn.Pool1DLayer import Pool1DLayer

rng = np.random.RandomState(23455)
# Set the batch size - remember to also perform in the network.py
BATCH_SIZE = 100

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

# Values received from  data
Xnout = np.empty([17924,256,30], dtype=theano.config.floatX)
Xoout = np.empty([17924,256,30], dtype=theano.config.floatX)

#Ignore the final 24, leads to errors.
for input in range(0,len(X)-24,BATCH_SIZE):

    amount = 0.5

    # Pass data through the network 1 by 1
    Xorig = X[input:input+BATCH_SIZE]
    # Add Noise to data
    Xnois = (Xorig * rng.binomial(size=Xorig.shape, n=1, p=(1-amount)).astype(theano.config.floatX))
    # Build the noisy outputs
    Xnout[input:input + BATCH_SIZE] = np.array(Network(network)(Xnois).eval()).astype(theano.config.floatX)
    # Build the non-noisy outputs
    Xoout[input:input+BATCH_SIZE] = np.array(Network(network)(Xorig).eval()).astype(theano.config.floatX)
    i = theano.shared(input, 'i')
    printing.Print('i')(i)

#Save the noisy activations
np.savez_compressed('NoisyHiddenActivations', *[Xnout[x] for x in range(len(Xnout))])
#Save the original activations
np.savez_compressed('OriginalHiddenActivations', *[Xoout[x] for x in range(len(Xoout))])