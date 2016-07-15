import numpy as np
import theano
import theano.tensor as T

from nn.AdamTrainer import AdamTrainer
from nn.Network import AutoEncodingNetwork

from network import network

rng = np.random.RandomState(23455)

X = np.load('../data/data_cmu.npz')['clips']
X = np.swapaxes(X, 1, 2).astype(theano.config.floatX)
X = X[:,:-4]

Xmean = X.mean(axis=2).mean(axis=0)[np.newaxis,:,np.newaxis]
Xmean[:,-3:] = 0.0

Xstd = np.array([[[X.std()]]]).repeat(X.shape[1], axis=1)
Xstd[:,-3:-1] = X[:,-3:-1].std()
Xstd[:,-1:  ] = X[:,-1:  ].std()

np.savez_compressed('../data/Joe/preprocess.npz', Xmean=Xmean, Xstd=Xstd)

X = (X - Xmean) / Xstd

#I = np.arange(len(X))
#rng.shuffle(I); X = X[I]

E = theano.shared(X, borrow=True)

# network.load([
    # None,
    # 'layer_0.npz', None, None,
    # 'layer_1.npz', None, None,
    # 'layer_2.npz', None, None,
# ])


def cost(network, X, Y):
    return T.mean((network(X) - Y)**2)


trainer = AdamTrainer(rng, batchsize=1, epochs=25, alpha=0.00001)
trainer.test(AutoEncodingNetwork(network), E, E, [
    None,
    '../models/conv_ae/layer_0.npz', None, None,
    '../models/conv_ae/layer_1.npz', None, None,
    '../models/conv_ae/layer_2.npz', None, None,
])

