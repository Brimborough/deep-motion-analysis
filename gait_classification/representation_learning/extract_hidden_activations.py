import numpy as np
import theano
from theano import printing

from nn.Network import Network
from nn.NoiseLayer import NoiseLayer
from nn.Pool1DLayer import Pool1DLayer

rng = np.random.RandomState(23455)
# Set the batch size - remember to also perform in the network.py
BATCH_SIZE = 4481

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

# Go through inputs in factors of 4481
for input in range(0,len(X),BATCH_SIZE):

    amount = 0.5

    # Pass data through the network 1 by 1
    Xorig = X[input:input+BATCH_SIZE]
    # Add Noise to data
    Xnois = (Xorig * rng.binomial(size=Xorig.shape, n=1, p=(1-amount)).astype(theano.config.floatX))
    # Build the noisy outputs
    Xnout[input:input + BATCH_SIZE] = np.array(Network(network)(Xnois).eval()).astype(theano.config.floatX)
    # Build the non-noisy outputs
    Xoout[input:input+BATCH_SIZE] = np.array(Network(network)(Xorig).eval()).astype(theano.config.floatX)


#Save the noisy activations
np.savez_compressed('HiddenActivations', Noisy=Xnout, Orig=Xoout)