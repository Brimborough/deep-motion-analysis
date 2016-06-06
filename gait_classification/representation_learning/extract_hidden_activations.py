import numpy as np
import theano

from nn.Network import Network
from nn.NoiseLayer import NoiseLayer
from nn.Pool1DLayer import Pool1DLayer

rng = np.random.RandomState(23455)
# Set the batch size - remember to also perform in the network.py
BATCH_SIZE = 4481

#Load the preprocessed to save some time
#X = np.load('../data/Joe/preProcX.npz')['clips']
X = np.load('../data/data_cmu.npz')['clips']
preprocess = np.load('../data/Joe/preprocess.npz')

X = np.swapaxes(X, 1, 2).astype(theano.config.floatX)
X = X[:,:-4] # - Remove foot contact
X = (X - preprocess['Xmean']) / preprocess['Xstd']

preprocess = np.load('../data/Joe/preprocess.npz')


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
    Xnout[input:input + BATCH_SIZE] = np.array(Network(network)(Xnois).eval()).astype(theano.config.floatX)[:]
    # Build the non-noisy outputs
    Xoout[input:input+BATCH_SIZE] = np.array(Network(network)(Xorig).eval()).astype(theano.config.floatX)[:]

#Save the noisy activations
np.savez_compressed('../data/Joe/HiddenActivations', Noisy=Xnout, Orig=Xoout)